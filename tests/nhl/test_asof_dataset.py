"""Tests de la sélection NHL à un instant pré-match prouvé."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tempfile
import unittest

from src.nhl.asof_dataset import NHLAsOfError, build_asof_game_view
from src.nhl.offline_sources import load_offline_fixture
from src.nhl.repository import archive_offline_fixture
from src.nhl.temporal_policy import NHLTemporalLeakageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = PROJECT_ROOT / "tests" / "nhl" / "fixtures" / "source_registry"
TARGET = 2026020001
UTC = timezone.utc


class AsOfGameViewTests(unittest.TestCase):
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
                archived_at_utc=datetime(2026, 10, 3, 12, tzinfo=UTC),
            )

    def view(self, hour: int, minute: int = 0, *, day: int = 1, target: int = TARGET):
        return build_asof_game_view(
            self.database,
            allowed_root=self.root,
            target_game_id=target,
            information_cutoff_utc=datetime(2026, 10, day, hour, minute, tzinfo=UTC),
        )

    def test_only_pre_match_evidence_and_finalized_history_are_selected(self) -> None:
        self.archive(
            "public_schedule_valid.json",
            "public_final_result_valid.json",
            "stats_team_history_valid.json",
            "pregame_goalies_valid.json",
            "player_availability_valid.json",
            "public_game_state_valid.json",
        )
        before = hashlib.sha256(self.database.read_bytes()).hexdigest()
        first = self.view(21)
        second = self.view(21)
        after = hashlib.sha256(self.database.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertEqual(first, second)
        self.assertEqual(first.away_team_id, 10)
        self.assertEqual(first.home_team_id, 20)
        self.assertEqual(first.schedule_observation_id, "obs.game.2026020001")
        self.assertEqual(
            {item.observation_id for item in first.observations},
            {
                "obs.goalie.away.probable",
                "obs.goalie.home.confirmed.early",
                "obs.injury.1010.early",
                "obs.team.20.history.2025021312",
            },
        )
        self.assertEqual(
            first.source_final_observation_ids,
            ("obs.result.2025021312.final",),
        )
        self.assertIn("obs.goalie.home.confirmed.late", first.after_cutoff_observation_ids)
        self.assertIn("obs.injury.2020.late", first.after_cutoff_observation_ids)
        self.assertIn("obs.state.2026020001.live", first.after_cutoff_observation_ids)
        self.assertEqual(len(first.snapshot_sha256), 64)

    def test_later_cutoff_changes_goalie_version_without_using_target_state(self) -> None:
        self.archive(
            "public_schedule_valid.json",
            "pregame_goalies_valid.json",
            "public_game_state_valid.json",
        )
        early = self.view(21)
        late = self.view(22, 50)
        self.assertNotEqual(early.snapshot_sha256, late.snapshot_sha256)
        ids = {item.observation_id for item in late.observations}
        self.assertIn("obs.goalie.home.confirmed.late", ids)
        self.assertNotIn("obs.goalie.home.confirmed.early", ids)
        self.assertNotIn("obs.state.2026020001.live", ids)
        self.assertIn("obs.state.2026020001.live", late.after_cutoff_observation_ids)

    def test_historical_statistic_without_final_result_is_excluded(self) -> None:
        self.archive("public_schedule_valid.json", "stats_team_history_valid.json")
        view = self.view(21)
        self.assertEqual(view.observations, ())
        self.assertEqual(
            view.unfinalized_source_observation_ids,
            ("obs.team.20.history.2025021312",),
        )

    def test_unknown_remains_explicit_and_is_not_imputed(self) -> None:
        self.archive("public_schedule_valid.json", "pregame_goalies_valid.json")
        view = self.view(21, day=2, target=2026020002)
        self.assertEqual(len(view.observations), 1)
        self.assertEqual(view.observations[0].value_state, "UNKNOWN")
        self.assertIsNone(view.observations[0].value)

    def test_schedule_unavailable_at_cutoff_fails_closed(self) -> None:
        self.archive("public_schedule_valid.json")
        with self.assertRaisesRegex(NHLAsOfError, "calendrier"):
            build_asof_game_view(
                self.database,
                allowed_root=self.root,
                target_game_id=TARGET,
                information_cutoff_utc=datetime(2026, 9, 30, 11, tzinfo=UTC),
            )

    def test_cutoff_after_game_start_is_forbidden(self) -> None:
        self.archive("public_schedule_valid.json")
        with self.assertRaises(NHLTemporalLeakageError):
            self.view(23, 1)

    def test_boolean_game_id_is_rejected(self) -> None:
        self.archive("public_schedule_valid.json")
        with self.assertRaises(NHLAsOfError):
            build_asof_game_view(
                self.database,
                allowed_root=self.root,
                target_game_id=True,
                information_cutoff_utc=datetime(2026, 10, 1, 21, tzinfo=UTC),
            )


if __name__ == "__main__":
    unittest.main()
