"""Synthetic NHL-09 tests: postgame labels never enter pregame features."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from src.nhl.asof_dataset import build_asof_game_view
from src.nhl.feature_bundle import build_asof_feature_bundle
from src.nhl.offline_sources import canonical_json_bytes, load_offline_fixture
from src.nhl.outcome_labels import (
    NHLOutcomeLabelError,
    assemble_labeled_game,
    build_audited_labeled_game,
    load_audited_final_label,
)
from src.nhl.repository import archive_offline_fixture


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "source_registry"
TARGET = 2026020001
UTC = timezone.utc


class NHLOutcomeLabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "nhl"
        self.database = self.root / "test_nhl.db"
        self.archive(FIXTURES / "public_schedule_valid.json")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def archive(self, path: Path) -> None:
        archive_offline_fixture(
            load_offline_fixture(path), database_path=self.database,
            allowed_root=self.root,
            archived_at_utc=datetime(2026, 10, 3, 12, tzinfo=UTC),
        )

    def cutoff(self) -> datetime:
        return datetime(2026, 10, 1, 21, tzinfo=UTC)

    def checkpoint(self, hour: int = 5) -> datetime:
        return datetime(2026, 10, 2, hour, tzinfo=UTC)

    def view(self):
        return build_asof_game_view(
            self.database, allowed_root=self.root, target_game_id=TARGET,
            information_cutoff_utc=self.cutoff(),
        )

    def result_fixture(
        self, *, fixture_id: str = "label.final.one", observation_id: str = "obs.label.one",
        away_score: int = 2, home_score: int = 4, winner_team_id: int = 20,
        final_at: str = "2026-10-02T03:55:00Z",
        observed_at: str = "2026-10-02T04:00:00Z",
    ) -> Path:
        document = json.loads((FIXTURES / "public_final_result_valid.json").read_text(
            encoding="utf-8"))
        document["fixture_id"] = fixture_id
        document["observed_at_utc"] = observed_at
        document["source_updated_at_utc"] = observed_at
        record = document["payload"]["records"][0]
        record["observation_id"] = observation_id
        record["entity_id"] = f"game:{TARGET}"
        record["source_game_id"] = TARGET
        record["target_game_id"] = 2026020002
        record["value"].update(
            away_score=away_score, home_score=home_score,
            winner_team_id=winner_team_id, final_observed_at_utc=final_at,
        )
        document["request"]["parameters"] = [["game_id", str(TARGET)]]
        document["response_sha256"] = sha256(canonical_json_bytes(
            document["payload"])).hexdigest()
        path = self.root / f"{fixture_id}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def labeled(self, *, checkpoint: datetime | None = None):
        return build_audited_labeled_game(
            self.database, allowed_root=self.root, target_game_id=TARGET,
            information_cutoff_utc=self.cutoff(),
            label_checkpoint_utc=checkpoint or self.checkpoint(),
        )

    def test_final_label_is_separate_deterministic_and_read_only(self) -> None:
        pregame = build_asof_feature_bundle(
            self.database, allowed_root=self.root, target_game_id=TARGET,
            information_cutoff_utc=self.cutoff(),
        )
        self.archive(self.result_fixture())
        before = sha256(self.database.read_bytes()).hexdigest()
        paired = self.labeled()
        self.assertIsNotNone(paired)
        self.assertEqual(paired, self.labeled())
        self.assertEqual(before, sha256(self.database.read_bytes()).hexdigest())
        self.assertEqual(paired.features.feature_sha256, pregame.feature_sha256)
        self.assertEqual(paired.label.winner_team_id, 20)
        self.assertEqual((paired.label.away_score, paired.label.home_score), (2, 4))
        self.assertEqual(len(paired.pair_sha256), 64)
        self.assertNotEqual(paired.pair_sha256, paired.features.feature_sha256)
        self.assertFalse(hasattr(paired, "probability"))
        self.assertFalse(hasattr(paired, "recommended_bet"))

    def test_missing_or_not_yet_available_final_remains_pending(self) -> None:
        self.assertIsNone(self.labeled())
        self.archive(self.result_fixture())
        self.assertIsNone(self.labeled(checkpoint=self.checkpoint(3)))
        self.assertIsNotNone(self.labeled(checkpoint=self.checkpoint(5)))
        with self.assertRaises(NHLOutcomeLabelError):
            self.labeled(checkpoint=datetime(2026, 10, 1, 22, tzinfo=UTC))

    def test_invalid_score_winner_or_final_timestamp_fails_closed(self) -> None:
        cases = (
            {"away_score": 4, "home_score": 4},
            {"winner_team_id": 10},
            {"final_at": "2026-10-01T20:00:00Z"},
        )
        for index, parameters in enumerate(cases):
            with self.subTest(parameters=parameters):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary) / "nhl"
                    db = root / "labels.db"
                    archive_offline_fixture(
                        load_offline_fixture(FIXTURES / "public_schedule_valid.json"),
                        database_path=db, allowed_root=root,
                        archived_at_utc=datetime(2026, 10, 3, 12, tzinfo=UTC),
                    )
                    path = self.result_fixture(
                        fixture_id=f"label.invalid.{index}",
                        observation_id=f"obs.label.invalid.{index}", **parameters)
                    archive_offline_fixture(
                        load_offline_fixture(path), database_path=db,
                        allowed_root=root,
                        archived_at_utc=datetime(2026, 10, 3, 12, tzinfo=UTC),
                    )
                    with self.assertRaises(NHLOutcomeLabelError):
                        build_audited_labeled_game(
                            db, allowed_root=root, target_game_id=TARGET,
                            information_cutoff_utc=self.cutoff(),
                            label_checkpoint_utc=self.checkpoint(),
                        )

    def test_conflicting_simultaneous_finals_fail_closed(self) -> None:
        self.archive(self.result_fixture())
        self.archive(self.result_fixture(
            fixture_id="label.final.conflict", observation_id="obs.label.conflict",
            away_score=5, home_score=4, winner_team_id=10,
        ))
        with self.assertRaises(NHLOutcomeLabelError):
            self.labeled()

    def test_mismatched_snapshot_and_game_are_rejected(self) -> None:
        self.archive(self.result_fixture())
        view = self.view()
        features = build_asof_feature_bundle(
            self.database, allowed_root=self.root, target_game_id=TARGET,
            information_cutoff_utc=self.cutoff(),
        )
        label = load_audited_final_label(
            self.database, allowed_root=self.root, view=view,
            label_checkpoint_utc=self.checkpoint(),
        )
        self.assertIsNotNone(label)
        with self.assertRaises(NHLOutcomeLabelError):
            assemble_labeled_game(
                view, replace(features, source_snapshot_sha256="0" * 64), label)
        with self.assertRaises(NHLOutcomeLabelError):
            assemble_labeled_game(view, features, replace(label, target_game_id=99))


if __name__ == "__main__":
    unittest.main()
