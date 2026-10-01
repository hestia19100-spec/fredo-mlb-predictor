"""Synthetic, read-only tests for NHL-07 pregame context."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tempfile
import unittest

from src.nhl.asof_dataset import build_asof_game_view
from src.nhl.contracts import GoalieStatus, PlayerAvailabilityStatus
from src.nhl.offline_sources import load_offline_fixture
from src.nhl.repository import archive_offline_fixture
from src.nhl.pregame_context_features import (
    PregameContextFeatureError,
    build_pregame_context_features,
    summarize_asof_pregame_context,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "source_registry"
TARGET = 2026020001
UTC = timezone.utc


class PregameContextFeaturesTests(unittest.TestCase):
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

    def cutoff(self, hour: int, minute: int = 0, *, day: int = 1) -> datetime:
        return datetime(2026, 10, day, hour, minute, tzinfo=UTC)

    def view(self, hour: int = 21, minute: int = 0, *, day: int = 1,
             target: int = TARGET):
        return build_asof_game_view(
            self.database, allowed_root=self.root,
            target_game_id=target,
            information_cutoff_utc=self.cutoff(hour, minute, day=day),
        )

    def features(self, hour: int = 21, minute: int = 0):
        return build_pregame_context_features(
            self.database, allowed_root=self.root,
            target_game_id=TARGET,
            information_cutoff_utc=self.cutoff(hour, minute),
        )

    def test_early_cutoff_preserves_known_statuses_without_mutating_database(self) -> None:
        self.archive("public_schedule_valid.json", "pregame_goalies_valid.json",
                     "player_availability_valid.json", "public_game_state_valid.json")
        before = hashlib.sha256(self.database.read_bytes()).hexdigest()
        first = self.features()
        self.assertEqual(first, self.features())
        self.assertEqual(before, hashlib.sha256(self.database.read_bytes()).hexdigest())
        self.assertEqual(first.away.goalie.status, GoalieStatus.PROBABLE)
        self.assertEqual(first.away.goalie.goalie_id, 1001)
        self.assertEqual(first.home.goalie.status, GoalieStatus.CONFIRMED)
        self.assertEqual(first.home.goalie.goalie_id, 2001)
        self.assertEqual(first.away.observed_status_counts, (("OUT", 1),))
        self.assertEqual(first.away.observed_players[0].player_id, 1010)
        self.assertEqual(first.home.observed_players, ())
        self.assertFalse(first.away.roster_completeness_proven)
        self.assertFalse(first.home.roster_completeness_proven)
        self.assertEqual(first.source_snapshot_sha256, self.view().snapshot_sha256)
        self.assertEqual(len(first.feature_sha256), 64)
        self.assertEqual(first.unattributed_unknown_player_observation_ids, ())

    def test_later_cutoff_changes_only_observations_then_available(self) -> None:
        self.archive("public_schedule_valid.json", "pregame_goalies_valid.json",
                     "player_availability_valid.json", "public_game_state_valid.json")
        early = self.features()
        late = self.features(22, 50)
        self.assertEqual(late.home.goalie.goalie_id, 2002)
        self.assertEqual(late.home.observed_players[0].player_id, 2020)
        self.assertEqual(late.home.observed_players[0].status,
                         PlayerAvailabilityStatus.QUESTIONABLE)
        self.assertNotEqual(early.feature_sha256, late.feature_sha256)
        self.assertNotEqual(early.source_snapshot_sha256, late.source_snapshot_sha256)

    def test_unknown_source_and_missing_observation_are_distinct(self) -> None:
        self.archive("public_schedule_valid.json", "pregame_goalies_valid.json")
        view = self.view(day=2, target=2026020002)
        result = summarize_asof_pregame_context(view)
        self.assertEqual(result.away.goalie.status, GoalieStatus.UNKNOWN)
        self.assertEqual(result.away.goalie.evidence_state, "SOURCE_UNKNOWN")
        self.assertEqual(result.home.goalie.status, GoalieStatus.UNKNOWN)
        self.assertEqual(result.home.goalie.evidence_state, "NOT_OBSERVED")
        self.assertIsNone(result.home.goalie.observation_id)

    def test_unknown_player_cannot_be_silently_assigned_to_a_team(self) -> None:
        self.archive("public_schedule_valid.json", "player_availability_valid.json")
        view = self.view()
        item = view.observations[0]
        unknown = replace(item, observation_id="obs.player.unknown",
                          entity_id="player:5050", value_state="UNKNOWN", value=None)
        result = summarize_asof_pregame_context(replace(view, observations=(unknown,)))
        self.assertEqual(result.unattributed_unknown_player_observation_ids,
                         ("obs.player.unknown",))
        self.assertEqual(result.away.observed_players, ())
        self.assertEqual(result.home.observed_players, ())

    def test_inconsistent_or_duplicate_pregame_evidence_fails_closed(self) -> None:
        self.archive("public_schedule_valid.json", "pregame_goalies_valid.json",
                     "player_availability_valid.json")
        view = self.view()
        goalie = next(item for item in view.observations if item.kind == "PREGAME_GOALIE")
        player = next(item for item in view.observations if item.kind == "PLAYER_AVAILABILITY")
        broken = (
            replace(goalie, value={**goalie.value, "team_id": 999}),
            replace(goalie, value={**goalie.value, "status": "FINAL"}),
            replace(goalie, source_game_id=123),
            replace(goalie, effective_available_at_utc=self.cutoff(22)),
            replace(player, value={**player.value, "player_id": 999}),
            replace(player, value={**player.value, "team_id": 999}),
        )
        for observation in broken:
            with self.subTest(observation=observation), self.assertRaises(PregameContextFeatureError):
                summarize_asof_pregame_context(replace(view, observations=(observation,)))
        with self.assertRaises(PregameContextFeatureError):
            summarize_asof_pregame_context(replace(
                view, observations=(goalie, replace(goalie, observation_id="obs.duplicate"))
            ))
        with self.assertRaises(PregameContextFeatureError):
            summarize_asof_pregame_context(replace(view, observations=(player, player)))


if __name__ == "__main__":
    unittest.main()
