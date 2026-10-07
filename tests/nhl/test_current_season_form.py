"""NHL-29: current-season sidecar joins only as-of descriptive rows."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.nhl.current_season_form import (
    NHLCurrentSeasonFormError, POLICY_PATH, summarize_current_season_pregame_form,
)
from src.nhl.current_season_import import CurrentSeasonImport
from src.nhl.moneypuck_team_import import TeamGameRow
from src.nhl.public_schedule_candidates import ScheduledGame
from src.nhl.team_history_store import PregameTeamHistory

UTC = timezone.utc


def row(team: str, season: int, day: int, situation: str, goals: int) -> TeamGameRow:
    return TeamGameRow(
        game_id=season * 1_000_000 + 20_000 + day,
        season=season, game_date=date(season, 10, day), team=team,
        opponent="BOS" if team == "NYR" else "NYR",
        home_or_away="AWAY" if team == "NYR" else "HOME",
        situation=situation, x_goals_for=Decimal(str(goals)),
        x_goals_against=Decimal("1"), goals_for=goals, goals_against=1,
        shots_on_goal_for=Decimal("30"), shots_on_goal_against=Decimal("25"),
    )


def pair(team: str, season: int, day: int, goals: int) -> tuple[TeamGameRow, ...]:
    return tuple(row(team, season, day, situation, goals) for situation in ("all", "5on5"))


class CurrentSeasonFormTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.game = ScheduledGame(2026020099, 20262027,
                                  datetime(2026, 10, 6, 21, tzinfo=UTC), "NYR", "BOS")
        self.history = PregameTeamHistory(
            target_game_id=self.game.game_id,
            information_cutoff_utc=datetime(2026, 10, 6, 19, tzinfo=UTC),
            capture_id="historical", response_sha256="a" * 64,
            effective_available_at_utc=datetime(2026, 10, 4, 10, tzinfo=UTC),
            away_rows=pair("NYR", 2025, 1, 1) + pair("NYR", 2025, 2, 2),
            home_rows=pair("BOS", 2025, 1, 1) + pair("BOS", 2025, 2, 2),
            regular_coverage_complete=False,
        )
        self.imported = CurrentSeasonImport(
            self.root / "import", 2026, "current", "b" * 64,
            datetime(2026, 10, 4, 7, tzinfo=UTC),
            datetime(2026, 10, 5, 12, tzinfo=UTC),
            datetime(2026, 10, 5, 12, tzinfo=UTC),
            pair("NYR", 2026, 3, 4) + pair("BOS", 2026, 3, 6), 1,
        )

    def joined(self, **kwargs):
        return summarize_current_season_pregame_form(
            kwargs.get("history", self.history), kwargs.get("imported", self.imported),
            kwargs.get("game", self.game), window_games=kwargs.get("window_games", 2),
            min_games_per_team=kwargs.get("minimum", 2),
        )

    def test_recent_rows_enter_window_without_mutating_sources(self) -> None:
        result = self.joined()
        self.assertEqual(result.form.away.source_game_ids, (2026020003, 2025020002))
        self.assertEqual(result.form.home.source_game_ids, (2026020003, 2025020002))
        self.assertEqual(result.form.away.goals_for_mean, Decimal("3.0000"))
        self.assertEqual(result.form.home.goals_for_mean, Decimal("4.0000"))
        self.assertEqual((result.away_current_season_games, result.home_current_season_games), (1, 1))
        self.assertTrue(result.form.minimum_sample_reached)
        self.assertFalse(result.historical_backtest_asof_proven)
        self.assertFalse(result.training_permitted)
        self.assertFalse(result.prediction_publication_permitted)
        self.assertNotEqual(result.feature_sha256, result.form.feature_sha256)
        self.assertEqual(result.feature_sha256, self.joined().feature_sha256)
        self.assertEqual(len(result.feature_sha256), 64)
        self.assertFalse(self.root.joinpath("import").exists())

    def test_import_provenance_changes_digest(self) -> None:
        changed = replace(self.imported, source_response_sha256="c" * 64)
        self.assertNotEqual(self.joined().feature_sha256,
                            self.joined(imported=changed).feature_sha256)

    def test_late_source_or_import_is_rejected(self) -> None:
        for changed in (
            replace(self.imported, source_observed_at_utc=datetime(2026, 10, 6, 20, tzinfo=UTC)),
            replace(self.imported, imported_at_utc=datetime(2026, 10, 6, 20, tzinfo=UTC)),
            replace(self.imported, effective_available_at_utc=datetime(2026, 10, 6, 20, tzinfo=UTC)),
            replace(self.imported, training_permitted=True),
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(NHLCurrentSeasonFormError):
                    self.joined(imported=changed)

    def test_target_future_or_conflicting_row_is_rejected(self) -> None:
        for changed_row in (
            replace(self.imported.regular_rows[0], game_id=self.game.game_id),
            replace(self.imported.regular_rows[0], game_date=date(2026, 10, 6)),
        ):
            with self.subTest(changed_row=changed_row):
                changed = replace(self.imported,
                                  regular_rows=(changed_row,) + self.imported.regular_rows[1:])
                with self.assertRaises(NHLCurrentSeasonFormError):
                    self.joined(imported=changed)
        conflict = replace(self.history,
                           away_rows=self.history.away_rows + (
                               replace(self.imported.regular_rows[0], goals_for=99),))
        with self.assertRaises(NHLCurrentSeasonFormError):
            self.joined(history=conflict)

    def test_missing_current_team_is_not_imputed(self) -> None:
        only_away = replace(self.imported, regular_rows=self.imported.regular_rows[:2])
        result = self.joined(imported=only_away)
        self.assertEqual((result.away_current_season_games, result.home_current_season_games), (1, 0))
        self.assertEqual(result.form.home.source_game_ids, (2025020002, 2025020001))

    def test_protocol_cannot_authorize_prediction(self) -> None:
        import json
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        self.assertFalse(policy["training_permitted"])
        self.assertFalse(policy["prediction_publication_permitted"])
        policy["prediction_publication_permitted"] = True
        unsafe = self.root / "unsafe.json"
        unsafe.write_text(json.dumps(policy), encoding="utf-8")
        with patch("src.nhl.current_season_form.POLICY_PATH", unsafe):
            with self.assertRaises(NHLCurrentSeasonFormError):
                self.joined()


if __name__ == "__main__":
    unittest.main()
