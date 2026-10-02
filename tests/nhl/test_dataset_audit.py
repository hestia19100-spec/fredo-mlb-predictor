"""NHL-10 synthetic checks for leakage, coverage and deterministic proof."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from src.nhl.asof_dataset import AsOfObservation, build_asof_game_view
from src.nhl.dataset_audit import (
    NHLDatasetAuditError, audit_labeled_dataset, audit_offline_labeled_dataset,
)
from src.nhl.feature_bundle import summarize_asof_feature_bundle
from src.nhl.offline_sources import canonical_json_bytes, load_offline_fixture
from src.nhl.outcome_labels import assemble_labeled_game, load_audited_final_label
from src.nhl.repository import archive_offline_fixture

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "source_registry"
UTC = timezone.utc
FIRST = 2026020001
SECOND = 2026020002


class NHLDatasetAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "nhl"
        self.database = self.root / "audit.db"
        self.archive(FIXTURES / "public_schedule_valid.json")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def archive(self, path: Path) -> None:
        archive_offline_fixture(
            load_offline_fixture(path), database_path=self.database,
            allowed_root=self.root,
            archived_at_utc=datetime(2026, 10, 3, 12, tzinfo=UTC),
        )

    def targets(self):
        return (
            (FIRST, datetime(2026, 10, 1, 21, tzinfo=UTC)),
            (SECOND, datetime(2026, 10, 2, 20, tzinfo=UTC)),
        )

    def checkpoint(self):
        return datetime(2026, 10, 3, 5, tzinfo=UTC)

    def result_fixture(self) -> Path:
        document = json.loads((FIXTURES / "public_final_result_valid.json").read_text(
            encoding="utf-8"))
        document["fixture_id"] = "audit.final.first"
        document["observed_at_utc"] = "2026-10-02T04:00:00Z"
        document["source_updated_at_utc"] = "2026-10-02T04:00:00Z"
        record = document["payload"]["records"][0]
        record["observation_id"] = "obs.audit.final.first"
        record["entity_id"] = f"game:{FIRST}"
        record["source_game_id"] = FIRST
        record["target_game_id"] = SECOND
        record["value"].update(
            away_score=2, home_score=4, winner_team_id=20,
            final_observed_at_utc="2026-10-02T03:55:00Z",
        )
        document["request"]["parameters"] = [["game_id", str(FIRST)]]
        document["response_sha256"] = sha256(canonical_json_bytes(
            document["payload"])).hexdigest()
        path = self.root / "audit.final.first.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def first_view_and_pair(self):
        self.archive(self.result_fixture())
        cutoff = self.targets()[0][1]
        view = build_asof_game_view(
            self.database, allowed_root=self.root, target_game_id=FIRST,
            information_cutoff_utc=cutoff,
        )
        features = summarize_asof_feature_bundle(view)
        label = load_audited_final_label(
            self.database, allowed_root=self.root, view=view,
            label_checkpoint_utc=self.checkpoint(),
        )
        self.assertIsNotNone(label)
        return view, assemble_labeled_game(view, features, label)

    def test_missing_results_stay_pending_and_report_is_deterministic(self) -> None:
        before = sha256(self.database.read_bytes()).hexdigest()
        missing = audit_offline_labeled_dataset(
            self.database, allowed_root=self.root,
            targets=self.targets(), label_checkpoint_utc=self.checkpoint(),
        )
        self.assertEqual((missing.candidate_count, missing.labeled_count,
                          missing.pending_game_ids), (2, 0, (FIRST, SECOND)))
        self.archive(self.result_fixture())
        sealed = sha256(self.database.read_bytes()).hexdigest()
        report = audit_offline_labeled_dataset(
            self.database, allowed_root=self.root,
            targets=self.targets(), label_checkpoint_utc=self.checkpoint(),
        )
        reverse = audit_offline_labeled_dataset(
            self.database, allowed_root=self.root,
            targets=reversed(self.targets()), label_checkpoint_utc=self.checkpoint(),
        )
        self.assertEqual(report, reverse)
        self.assertEqual(sha256(self.database.read_bytes()).hexdigest(), sealed)
        self.assertNotEqual(before, sealed)
        self.assertEqual((report.candidate_count, report.labeled_count,
                          report.pending_count, report.coverage_percent), (2, 1, 1, 50.0))
        self.assertEqual(report.pending_game_ids, (SECOND,))
        self.assertEqual(report.labeled_game_ids, (FIRST,))
        self.assertEqual(len(report.audit_sha256), 64)
        self.assertFalse(report.training_permitted)

    def test_tampered_pair_and_premature_result_fail_closed(self) -> None:
        view, pair = self.first_view_and_pair()
        self.assertEqual(audit_labeled_dataset([view], [pair]).coverage_percent, 100.0)
        altered = (
            replace(pair, pair_sha256="0" * 64),
            replace(pair, features=replace(pair.features, feature_sha256="0" * 64)),
            replace(pair, label=replace(
                pair.label, evidence_available_at_utc=view.information_cutoff_utc)),
            replace(pair, label=replace(pair.label, winner_team_id=view.away_team_id)),
        )
        for candidate in altered:
            with self.subTest(candidate=candidate):
                with self.assertRaises(NHLDatasetAuditError):
                    audit_labeled_dataset([view], [candidate])

    def test_future_or_target_result_cannot_enter_pregame_view(self) -> None:
        view, pair = self.first_view_and_pair()
        future = AsOfObservation(
            observation_id="future", kind="TEAM_STATISTICS", entity_id="team:10",
            source_game_id=SECOND, value_state="KNOWN", value={},
            effective_available_at_utc=view.information_cutoff_utc + timedelta(seconds=1),
            observation_sha256="a" * 64,
        )
        for item in (future, replace(
                future, kind="FINAL_RESULT",
                effective_available_at_utc=view.information_cutoff_utc),
                replace(future, source_game_id=FIRST,
                        effective_available_at_utc=view.information_cutoff_utc)):
            with self.subTest(item=item):
                with self.assertRaises(NHLDatasetAuditError):
                    audit_labeled_dataset([replace(view, observations=(item,))], [pair])

    def test_duplicates_and_unexpected_labels_fail_closed(self) -> None:
        view, pair = self.first_view_and_pair()
        for views, pairs in (([view, view], [pair]),
                             ([view], [pair, pair]),
                             ([replace(view, target_game_id=SECOND)], [pair])):
            with self.assertRaises(NHLDatasetAuditError):
                audit_labeled_dataset(views, pairs)
        with self.assertRaises(NHLDatasetAuditError):
            audit_labeled_dataset([], [])


if __name__ == "__main__":
    unittest.main()
