"""NHL-25 descriptive form: no outcome, probability, or training path."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.nhl.moneypuck_team_import import TeamGameRow
from src.nhl.pregame_team_form import (
    NHLTeamFormError, POLICY_PATH, load_verified_schedule_team_form,
    summarize_pregame_team_form,
)
from src.nhl.team_history_store import PregameTeamHistory

UTC = timezone.utc


def row(team: str, game_id: int, day: int, situation: str,
        *, goals: int = 2, x_goals: str = "2.5") -> TeamGameRow:
    return TeamGameRow(
        game_id=game_id, season=2025, game_date=date(2025, 10, day),
        team=team, opponent="BOS" if team == "NYR" else "NYR",
        home_or_away="AWAY" if team == "NYR" else "HOME",
        situation=situation, x_goals_for=Decimal(x_goals),
        x_goals_against=Decimal("1.5"), goals_for=goals,
        goals_against=1, shots_on_goal_for=Decimal("29"),
        shots_on_goal_against=Decimal("26"),
    )


def history() -> PregameTeamHistory:
    def team_rows(team: str) -> tuple[TeamGameRow, ...]:
        return tuple(row(team, 2025020000 + day, day, situation,
                         goals=day, x_goals=str(day))
                     for day in (8, 9, 10) for situation in ("all", "5on5"))
    return PregameTeamHistory(
        target_game_id=2026020001,
        information_cutoff_utc=datetime(2026, 10, 6, 19, tzinfo=UTC),
        capture_id="capture-1", response_sha256="a" * 64,
        effective_available_at_utc=datetime(2026, 10, 4, 12, tzinfo=UTC),
        away_rows=team_rows("NYR"), home_rows=team_rows("BOS"),
        regular_coverage_complete=False,
    )


def form(value: PregameTeamHistory | None = None, **kwargs):
    return summarize_pregame_team_form(
        value or history(), away_abbr="NYR", home_abbr="BOS",
        target_date=date(2026, 10, 6), **kwargs,
    )


class PregameTeamFormTests(unittest.TestCase):
    def test_recent_window_is_deterministic_and_descriptive_only(self) -> None:
        result = form(window_games=2, min_games_per_team=2)
        self.assertEqual(result.away.source_game_ids, (2025020010, 2025020009))
        self.assertEqual(result.away.goals_for_mean, Decimal("9.5000"))
        self.assertEqual(result.home.five_on_five_x_goals_for_mean, Decimal("9.5000"))
        self.assertTrue(result.minimum_sample_reached)
        self.assertFalse(result.regular_coverage_complete)
        self.assertFalse(result.historical_backtest_asof_proven)
        self.assertFalse(result.training_permitted)
        self.assertFalse(result.prediction_publication_permitted)
        self.assertEqual(len(result.feature_sha256), 64)
        self.assertEqual(result.feature_sha256, form(window_games=2, min_games_per_team=2).feature_sha256)
        self.assertNotEqual(result.feature_sha256, form(window_games=3, min_games_per_team=2).feature_sha256)

    def test_short_sample_is_exposed_not_imputed(self) -> None:
        result = form(min_games_per_team=5)
        self.assertFalse(result.minimum_sample_reached)
        self.assertEqual(result.away.complete_games, 3)
        self.assertEqual(result.home.complete_games, 3)

    def test_incomplete_situation_is_not_counted(self) -> None:
        previous = history()
        partial = replace(previous, away_rows=previous.away_rows[:-1])
        result = form(partial, min_games_per_team=2)
        self.assertEqual(result.away.complete_games, 2)
        self.assertEqual(result.away.incomplete_game_ids, (2025020010,))
        self.assertEqual(result.home.complete_games, 3)

    def test_duplicate_or_conflicting_rows_are_rejected(self) -> None:
        previous = history()
        for extra in (previous.away_rows[0], replace(previous.away_rows[1], opponent="CHI")):
            with self.subTest(extra=extra):
                changed = replace(previous, away_rows=previous.away_rows + (extra,))
                with self.assertRaises(NHLTeamFormError):
                    form(changed)

    def test_target_and_future_rows_are_rejected(self) -> None:
        previous = history()
        for changed_row in (
            replace(previous.away_rows[0], game_id=previous.target_game_id),
            replace(previous.away_rows[0], game_date=date(2026, 10, 6)),
        ):
            with self.subTest(changed_row=changed_row):
                changed = replace(previous, away_rows=(changed_row,) + previous.away_rows[1:])
                with self.assertRaises(NHLTeamFormError):
                    form(changed)

    def test_after_cutoff_and_enabled_training_are_rejected(self) -> None:
        previous = history()
        for changed in (
            replace(previous, effective_available_at_utc=datetime(2026, 10, 6, 20, tzinfo=UTC)),
            replace(previous, training_permitted=True),
            replace(previous, prediction_publication_permitted=True),
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(NHLTeamFormError):
                    form(changed)

    def test_invalid_window_or_minimum_is_rejected(self) -> None:
        for window, minimum in ((0, 1), (True, 1), (2, 3), (10, 0)):
            with self.subTest(window=window, minimum=minimum):
                with self.assertRaises(NHLTeamFormError):
                    form(window_games=window, min_games_per_team=minimum)

    def test_public_entry_requires_verified_schedule_and_history_bridge(self) -> None:
        schedule = {"games": [{
            "game_id": 2026020001, "scheduled_start_utc": "2026-10-06T21:00:00Z",
            "away_abbr": "NYR", "home_abbr": "BOS",
        }]}
        with (patch("src.nhl.pregame_team_form.audit_schedule_capture", return_value=schedule),
              patch("src.nhl.pregame_team_form.load_verified_schedule_team_history", return_value=history()) as bridge):
            result = load_verified_schedule_team_form(
                Path("verified-slot"), target_game_id=2026020001, lead_minutes=120,
                min_games_per_team=2,
            )
            self.assertTrue(result.minimum_sample_reached)
            bridge.assert_called_once()
        with patch("src.nhl.pregame_team_form.audit_schedule_capture", return_value=schedule):
            with self.assertRaises(NHLTeamFormError):
                load_verified_schedule_team_form(Path("verified-slot"), target_game_id=1, lead_minutes=120)
        with patch("src.nhl.pregame_team_form.audit_schedule_capture", side_effect=ValueError("tampered")):
            with self.assertRaisesRegex(ValueError, "tampered"):
                load_verified_schedule_team_form(Path("verified-slot"), target_game_id=2026020001, lead_minutes=120)
    def test_protocol_cannot_enable_training(self) -> None:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        self.assertFalse(policy["training_permitted"])
        self.assertFalse(policy["prediction_publication_permitted"])
        policy["training_permitted"] = True
        with TemporaryDirectory() as directory:
            unsafe = Path(directory) / "unsafe_protocol.json"
            unsafe.write_text(json.dumps(policy), encoding="utf-8")
            with patch("src.nhl.pregame_team_form.POLICY_PATH", unsafe):
                with self.assertRaisesRegex(NHLTeamFormError, "Protocole"):
                    form()



if __name__ == "__main__":
    unittest.main()
