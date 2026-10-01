from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.nhl.offline_sources import OfflineFixtureBundle, load_offline_fixture
from src.nhl.repository import (
    NHLRepositoryError,
    archive_offline_fixture,
    audit_nhl_storage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "nhl" / "fixtures" / "source_registry"
ARCHIVED_AT = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


class NHLRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "nhl"
        self.database = self.root / "test_nhl.db"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def fixture(self, name: str) -> OfflineFixtureBundle:
        return load_offline_fixture(FIXTURE_ROOT / name)

    def archive(self, bundle: OfflineFixtureBundle):
        return archive_offline_fixture(
            bundle,
            database_path=self.database,
            allowed_root=self.root,
            archived_at_utc=ARCHIVED_AT,
        )

    def test_archive_is_atomic_audited_and_idempotent(self) -> None:
        bundle = self.fixture("pregame_goalies_valid.json")
        first = self.archive(bundle)
        second = self.archive(bundle)
        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(first.bundle_sha256, second.bundle_sha256)
        self.assertEqual(first.observation_count, len(bundle.observations))

        audit = audit_nhl_storage(self.database, allowed_root=self.root)
        self.assertTrue(audit.schema_valid)
        self.assertEqual(audit.run_count, 1)
        self.assertEqual(audit.observation_count, len(bundle.observations))
        self.assertEqual(audit.unknown_count, 1)
        self.assertEqual(audit.bundle_sha256s, (first.bundle_sha256,))

    def test_unknown_value_is_stored_as_explicit_null(self) -> None:
        self.archive(self.fixture("pregame_goalies_valid.json"))
        with closing(sqlite3.connect(self.database)) as connection:
            row = connection.execute(
                """
                SELECT value_state, canonical_value_json
                FROM nhl_normalized_observations
                WHERE value_state = 'UNKNOWN'
                """
            ).fetchone()
        self.assertEqual(row, ("UNKNOWN", None))

    def test_same_fixture_id_with_different_content_is_rejected(self) -> None:
        bundle = self.fixture("pregame_goalies_valid.json")
        self.archive(bundle)
        original = bundle.observations[0]
        changed_value = dict(original.value)
        changed_value["status"] = "CONFIRMED"
        changed = replace(original, value=changed_value)
        altered = replace(bundle, observations=(changed, *bundle.observations[1:]))
        with self.assertRaisesRegex(NHLRepositoryError, "autre contenu"):
            self.archive(altered)

    def test_archive_time_cannot_precede_source_evidence(self) -> None:
        bundle = self.fixture("public_schedule_valid.json")
        with self.assertRaisesRegex(NHLRepositoryError, "précéder"):
            archive_offline_fixture(
                bundle,
                database_path=self.database,
                allowed_root=self.root,
                archived_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
            )
        self.assertFalse(self.database.exists())

    def test_cross_bundle_observation_conflict_rolls_back_whole_run(self) -> None:
        first = self.fixture("public_schedule_valid.json")
        second = self.fixture("public_game_state_valid.json")
        self.archive(first)
        first_id = first.observations[0].observation_id
        observation = second.observations[0]
        evidence = replace(observation.evidence, observation_id=first_id)
        conflicting = replace(
            observation,
            observation_id=first_id,
            evidence=evidence,
        )
        altered = replace(second, observations=(conflicting,))
        with self.assertRaises(NHLRepositoryError):
            self.archive(altered)
        audit = audit_nhl_storage(self.database, allowed_root=self.root)
        self.assertEqual(audit.run_count, 1)
        self.assertEqual(audit.observation_count, len(first.observations))

    def test_provenance_mismatch_is_rejected_before_database_creation(self) -> None:
        bundle = self.fixture("public_schedule_valid.json")
        observation = bundle.observations[0]
        evidence = replace(observation.evidence, provider="wrong_provider")
        altered = replace(
            bundle,
            observations=(replace(observation, evidence=evidence),),
        )
        with self.assertRaisesRegex(NHLRepositoryError, "diverge"):
            self.archive(altered)
        self.assertFalse(self.database.exists())

    def test_audit_is_read_only_and_recalculates_every_hash(self) -> None:
        self.archive(self.fixture("stats_team_history_valid.json"))
        before = hashlib.sha256(self.database.read_bytes()).hexdigest()
        audit_nhl_storage(self.database, allowed_root=self.root)
        after = hashlib.sha256(self.database.read_bytes()).hexdigest()
        self.assertEqual(after, before)

    def test_audit_detects_tampering_after_trigger_removal(self) -> None:
        self.archive(self.fixture("public_schedule_valid.json"))
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute(
                "DROP TRIGGER nhl_normalized_observations_no_update"
            )
            connection.execute(
                """
                UPDATE nhl_normalized_observations
                SET entity_id = 'tampered'
                """
            )
        with self.assertRaises(NHLRepositoryError):
            audit_nhl_storage(self.database, allowed_root=self.root)

    def test_append_only_data_cannot_be_updated_or_deleted(self) -> None:
        self.archive(self.fixture("public_schedule_valid.json"))
        with closing(sqlite3.connect(self.database)) as connection, connection:
            for command in (
                "UPDATE nhl_ingestion_runs SET provider = provider",
                "DELETE FROM nhl_ingestion_runs",
                "UPDATE nhl_normalized_observations SET entity_id = entity_id",
                "DELETE FROM nhl_normalized_observations",
            ):
                with self.subTest(command=command):
                    with self.assertRaises(sqlite3.IntegrityError):
                        connection.execute(command)


if __name__ == "__main__":
    unittest.main()
