"""Offline tests for NHL-06 historical, pre-match team summaries."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
from pathlib import Path
import tempfile
import unittest

from src.nhl.asof_dataset import build_asof_game_view
from src.nhl.offline_sources import load_offline_fixture
from src.nhl.repository import archive_offline_fixture
from src.nhl.team_history_features import (
    TeamHistoryFeatureError,
    build_team_history_features,
    summarize_asof_team_history,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "source_registry"
CUTOFF = datetime(2026, 10, 1, 21, tzinfo=timezone.utc)
TARGET = 2026020001


class TeamHistoryFeaturesTests(unittest.TestCase):
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
                database_path=self.database,
                allowed_root=self.root,
                archived_at_utc=datetime(2026, 10, 3, 12, tzinfo=timezone.utc),
            )

    def view(self):
        return build_asof_game_view(
            self.database, allowed_root=self.root,
            target_game_id=TARGET, information_cutoff_utc=CUTOFF,
        )

    def test_complete_historical_home_game_is_read_only_and_not_imputed(self) -> None:
        self.archive(
            "public_schedule_valid.json", "public_final_result_valid.json",
            "stats_team_history_valid.json",
        )
        before = hashlib.sha256(self.database.read_bytes()).hexdigest()
        first = build_team_history_features(
            self.database, allowed_root=self.root,
            target_game_id=TARGET, information_cutoff_utc=CUTOFF,
        )
        second = build_team_history_features(
            self.database, allowed_root=self.root,
            target_game_id=TARGET, information_cutoff_utc=CUTOFF,
        )
        self.assertEqual(before, hashlib.sha256(self.database.read_bytes()).hexdigest())
        self.assertEqual(first, second)
        self.assertEqual(first.home.source_game_ids, (2025021312,))
        self.assertEqual(first.home.complete_game_count, 1)
        self.assertEqual(first.home.average_goals_for, Decimal("4.000000"))
        self.assertEqual(first.home.average_shots_for, Decimal("34.000000"))
        self.assertEqual(first.home.power_play_conversion_rate, Decimal("0.250000"))
        self.assertEqual(first.home.penalty_kill_rate, Decimal("1.000000"))
        self.assertEqual(first.away.complete_game_count, 0)
        self.assertIsNone(first.away.average_goals_for)
        self.assertFalse(first.minimum_sample_reached)
        self.assertEqual(first.source_snapshot_sha256, self.view().snapshot_sha256)
        self.assertEqual(len(first.feature_sha256), 64)

    def test_source_game_without_known_final_is_excluded(self) -> None:
        self.archive("public_schedule_valid.json", "stats_team_history_valid.json")
        result = build_team_history_features(
            self.database, allowed_root=self.root,
            target_game_id=TARGET, information_cutoff_utc=CUTOFF,
        )
        self.assertEqual(result.home.complete_game_count, 0)
        self.assertEqual(result.home.source_game_ids, ())
        self.assertFalse(result.minimum_sample_reached)

    def test_unknown_and_incomplete_values_are_explicit(self) -> None:
        self.archive(
            "public_schedule_valid.json", "public_final_result_valid.json",
            "stats_team_history_valid.json",
        )
        view = self.view()
        item = view.observations[0]
        unknown = replace(item, value_state="UNKNOWN", value=None)
        result = summarize_asof_team_history(replace(view, observations=(unknown,)))
        self.assertEqual(result.home.unknown_observation_ids, (item.observation_id,))
        self.assertEqual(result.home.complete_game_count, 0)
        partial = replace(item, value={"goals_for": 4})
        result = summarize_asof_team_history(replace(view, observations=(partial,)))
        self.assertEqual(result.home.incomplete_observation_ids, (item.observation_id,))
        self.assertIsNone(result.home.average_goals_for)

    def test_zero_special_team_denominators_remain_missing(self) -> None:
        self.archive(
            "public_schedule_valid.json", "public_final_result_valid.json",
            "stats_team_history_valid.json",
        )
        view = self.view()
        item = view.observations[0]
        value = dict(item.value)
        value.update(power_play_goals=0, power_play_opportunities=0,
                     penalty_kills=0, penalty_kill_opportunities=0)
        result = summarize_asof_team_history(
            replace(view, observations=(replace(item, value=value),))
        )
        self.assertIsNone(result.home.power_play_conversion_rate)
        self.assertIsNone(result.home.penalty_kill_rate)

    def test_invalid_and_duplicate_rows_fail_closed(self) -> None:
        self.archive(
            "public_schedule_valid.json", "public_final_result_valid.json",
            "stats_team_history_valid.json",
        )
        view = self.view()
        item = view.observations[0]
        for value in (-1, True, "4"):
            with self.subTest(value=value), self.assertRaises(TeamHistoryFeatureError):
                summarize_asof_team_history(replace(
                    view, observations=(replace(item, value={**item.value, "goals_for": value}),)
                ))
        with self.assertRaises(TeamHistoryFeatureError):
            summarize_asof_team_history(replace(view, observations=(item, replace(item, observation_id="duplicate"))))
        with self.assertRaises(TeamHistoryFeatureError):
            summarize_asof_team_history(replace(view, observations=(replace(item, effective_available_at_utc=datetime(2026, 10, 1, 22, tzinfo=timezone.utc)),)))
        with self.assertRaises(TeamHistoryFeatureError):
            summarize_asof_team_history(replace(view, observations=(replace(item, source_game_id=TARGET),)))

    def test_sample_threshold_is_descriptive_only(self) -> None:
        self.archive(
            "public_schedule_valid.json", "public_final_result_valid.json",
            "stats_team_history_valid.json",
        )
        view = self.view()
        home = view.observations[0]
        away = replace(home, entity_id="team:10", observation_id="obs.team.10.synthetic")
        both = replace(view, observations=(home, away))
        result = summarize_asof_team_history(both, min_games_per_team=1)
        self.assertTrue(result.minimum_sample_reached)
        self.assertEqual(result.away.complete_game_count, 1)
        self.assertFalse(summarize_asof_team_history(both, min_games_per_team=2).minimum_sample_reached)
        for threshold in (0, True, 1.5):
            with self.subTest(threshold=threshold), self.assertRaises(TeamHistoryFeatureError):
                summarize_asof_team_history(both, min_games_per_team=threshold)


if __name__ == "__main__":
    unittest.main()
