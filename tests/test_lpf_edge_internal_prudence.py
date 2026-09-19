"""Tests du choix privé à probabilité LPF maximale."""

from __future__ import annotations

import inspect
import unittest
from datetime import date, datetime, timezone
from decimal import Decimal

from src.lpf_edge_dashboard import CertifiedPrediction, CertifiedPredictionDay
from src.lpf_edge_internal_prudence import (
    LPFEdgeInternalPrudenceError,
    STRATEGY_ID,
    VISIBILITY,
    select_internal_prudence,
)


def prediction(
    game_id: int,
    *,
    home_probability: str,
    home_team_id: int,
    away_team_id: int,
) -> CertifiedPrediction:
    p_home = Decimal(home_probability)
    return CertifiedPrediction(
        prediction_id=f"prediction-{game_id}",
        batch_id="batch-2026-09-19",
        game_id=game_id,
        official_date=date(2026, 9, 19),
        away_team_id=away_team_id,
        home_team_id=home_team_id,
        scheduled_start_utc=datetime(
            2026, 9, 19, 18, game_id % 60, tzinfo=timezone.utc
        ),
        issued_at_utc=datetime(2026, 9, 19, 12, tzinfo=timezone.utc),
        p_home_win=p_home,
        p_away_win=Decimal("1") - p_home,
    )


def certified_day(
    predictions: tuple[CertifiedPrediction, ...],
) -> CertifiedPredictionDay:
    return CertifiedPredictionDay(
        target_date=date(2026, 9, 19),
        batch_id="batch-2026-09-19",
        results_commit="a" * 40,
        certified_at_utc=datetime(2026, 9, 19, 13, tzinfo=timezone.utc),
        remote_lead_minutes=Decimal("120"),
        predictions_sha256="b" * 64,
        receipt_sha256="c" * 64,
        predictions=predictions,
    )


class LPFEdgeInternalPrudenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.names = {
            1: "Home One",
            2: "Away One",
            3: "Home Two",
            4: "Away Two",
            5: "Home Three",
            6: "Away Three",
        }

    def test_highest_certified_probability_is_selected(self) -> None:
        day = certified_day(
            (
                prediction(
                    30,
                    home_probability="0.58",
                    home_team_id=1,
                    away_team_id=2,
                ),
                prediction(
                    20,
                    home_probability="0.34",
                    home_team_id=3,
                    away_team_id=4,
                ),
            )
        )

        choice = select_internal_prudence(day, team_names=self.names)

        self.assertEqual(choice.game_id, 20)
        self.assertEqual(choice.predicted_side, "AWAY")
        self.assertEqual(choice.predicted_team_name, "Away Two")
        self.assertEqual(choice.model_probability, Decimal("0.66"))
        self.assertEqual(choice.home_team_name, "Home Two")
        self.assertEqual(choice.away_team_name, "Away Two")
        self.assertEqual(choice.strategy_id, STRATEGY_ID)
        self.assertEqual(choice.visibility, VISIBILITY)
        self.assertFalse(choice.public_exposure_allowed)
        self.assertEqual(choice.rank_by_probability, 1)
        self.assertFalse(choice.tie_break_rule_applied)

    def test_tie_is_resolved_by_smallest_game_id(self) -> None:
        first = prediction(
            40,
            home_probability="0.63",
            home_team_id=1,
            away_team_id=2,
        )
        second = prediction(
            10,
            home_probability="0.37",
            home_team_id=3,
            away_team_id=4,
        )

        normal = select_internal_prudence(
            certified_day((first, second)), team_names=self.names
        )
        reversed_choice = select_internal_prudence(
            certified_day((second, first)), team_names=self.names
        )

        self.assertEqual(normal.game_id, 10)
        self.assertEqual(reversed_choice.game_id, 10)
        self.assertTrue(normal.tie_break_rule_applied)
        self.assertTrue(reversed_choice.tie_break_rule_applied)

    def test_missing_team_name_is_rejected(self) -> None:
        day = certified_day(
            (
                prediction(
                    1,
                    home_probability="0.60",
                    home_team_id=1,
                    away_team_id=99,
                ),
            )
        )

        with self.assertRaises(LPFEdgeInternalPrudenceError):
            select_internal_prudence(day, team_names=self.names)

    def test_empty_day_is_rejected(self) -> None:
        with self.assertRaises(LPFEdgeInternalPrudenceError):
            select_internal_prudence(certified_day(()), team_names=self.names)

    def test_selector_has_no_result_or_market_input(self) -> None:
        signature = inspect.signature(select_internal_prudence)
        self.assertEqual(
            tuple(signature.parameters),
            ("prediction_day", "team_names"),
        )
        source = inspect.getsource(select_internal_prudence)
        self.assertNotIn("DailyScoreSummary", source)
        self.assertNotIn("load_latest_score", source)
        self.assertNotIn("odds", source.lower())


if __name__ == "__main__":
    unittest.main()
