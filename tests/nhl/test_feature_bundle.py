"""Synthetic NHL-08 tests: one as-of view, no model or provider."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.nhl.asof_dataset import build_asof_game_view
from src.nhl.feature_bundle import (
    NHLFeatureBundleError,
    build_asof_feature_bundle,
    summarize_asof_feature_bundle,
)
from src.nhl.offline_sources import load_offline_fixture
from src.nhl.pregame_context_features import summarize_asof_pregame_context
from src.nhl.repository import archive_offline_fixture


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "source_registry"
TARGET = 2026020001
UTC = timezone.utc


class NHLFeatureBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "nhl"
        self.database = self.root / "test_nhl.db"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def archive(self, *names: str) -> None:
        for name in names:
            archive_offline_fixture(
                load_offline_fixture(FIXTURES / name),
                database_path=self.database, allowed_root=self.root,
                archived_at_utc=datetime(2026, 10, 3, 12, tzinfo=UTC),
            )

    def fixture_set(self) -> None:
        self.archive(
            "public_schedule_valid.json", "public_final_result_valid.json",
            "stats_team_history_valid.json", "pregame_goalies_valid.json",
            "player_availability_valid.json", "public_game_state_valid.json",
        )

    def cutoff(self, hour: int = 21, minute: int = 0) -> datetime:
        return datetime(2026, 10, 1, hour, minute, tzinfo=UTC)

    def view(self, hour: int = 21, minute: int = 0):
        return build_asof_game_view(
            self.database, allowed_root=self.root, target_game_id=TARGET,
            information_cutoff_utc=self.cutoff(hour, minute),
        )

    def bundle(self, hour: int = 21, minute: int = 0):
        return build_asof_feature_bundle(
            self.database, allowed_root=self.root, target_game_id=TARGET,
            information_cutoff_utc=self.cutoff(hour, minute),
        )

    def test_single_read_only_view_yields_deterministic_combined_proof(self) -> None:
        self.fixture_set()
        before = hashlib.sha256(self.database.read_bytes()).hexdigest()
        with patch("src.nhl.feature_bundle.build_asof_game_view",
                   wraps=build_asof_game_view) as read_view:
            first = self.bundle()
        self.assertEqual(read_view.call_count, 1)
        self.assertEqual(first, self.bundle())
        self.assertEqual(before, hashlib.sha256(self.database.read_bytes()).hexdigest())
        self.assertEqual(first.target_game_id, TARGET)
        self.assertEqual(first.source_snapshot_sha256, self.view().snapshot_sha256)
        self.assertEqual(first.history.source_snapshot_sha256, first.source_snapshot_sha256)
        self.assertEqual(first.pregame.source_snapshot_sha256, first.source_snapshot_sha256)
        self.assertEqual(first.history.home.complete_game_count, 1)
        self.assertEqual(first.history.away.complete_game_count, 0)
        self.assertIsNone(first.history.away.average_goals_for)
        self.assertFalse(first.history.minimum_sample_reached)
        self.assertEqual(first.pregame.away.goalie.evidence_state, "KNOWN")
        self.assertFalse(first.pregame.away.roster_completeness_proven)
        self.assertEqual(len(first.feature_sha256), 64)
        self.assertNotEqual(first.feature_sha256, first.source_snapshot_sha256)

    def test_new_pregame_evidence_changes_only_later_bundle(self) -> None:
        self.fixture_set()
        early = self.bundle()
        later = self.bundle(22, 50)
        self.assertEqual(early.history.home.complete_game_count,
                         later.history.home.complete_game_count)
        self.assertNotEqual(early.pregame.feature_sha256,
                            later.pregame.feature_sha256)
        self.assertNotEqual(early.feature_sha256, later.feature_sha256)
        self.assertEqual(early.pregame.home.goalie.goalie_id, 2001)
        self.assertEqual(later.pregame.home.goalie.goalie_id, 2002)

    def test_outcomes_and_target_state_cannot_enter_candidate_bundle(self) -> None:
        self.fixture_set()
        view = self.view()
        self.assertTrue(all(item.kind not in ("FINAL_RESULT", "GAME_STATE")
                            for item in view.observations))
        source = view.observations[0]
        for kind in ("FINAL_RESULT", "GAME_STATE", "UNRECOGNIZED"):
            with self.subTest(kind=kind), self.assertRaises(NHLFeatureBundleError):
                summarize_asof_feature_bundle(replace(
                    view, observations=(replace(source, kind=kind),)
                ))

    def test_after_cutoff_duplicate_and_mismatched_inputs_fail_closed(self) -> None:
        self.fixture_set()
        view = self.view()
        source = view.observations[0]
        invalid_views = (
            replace(view, observations=(source, source)),
            replace(view, observations=(replace(
                source, effective_available_at_utc=self.cutoff(22)
            ),)),
            replace(view, away_team_id=view.home_team_id),
            replace(view, information_cutoff_utc=view.scheduled_start_utc),
        )
        for invalid in invalid_views:
            with self.subTest(invalid=invalid), self.assertRaises(NHLFeatureBundleError):
                summarize_asof_feature_bundle(invalid)
        pregame = summarize_asof_pregame_context(view)
        with patch("src.nhl.feature_bundle.summarize_asof_pregame_context",
                   return_value=replace(pregame, source_snapshot_sha256="0" * 64)):
            with self.assertRaises(NHLFeatureBundleError):
                summarize_asof_feature_bundle(view)

    def test_sample_size_is_descriptive_not_a_prediction_gate(self) -> None:
        self.fixture_set()
        view = self.view()
        result = summarize_asof_feature_bundle(view, min_games_per_team=1)
        self.assertFalse(result.history.minimum_sample_reached)
        self.assertFalse(hasattr(result, "probability"))
        self.assertFalse(hasattr(result, "recommended_bet"))


if __name__ == "__main__":
    unittest.main()
