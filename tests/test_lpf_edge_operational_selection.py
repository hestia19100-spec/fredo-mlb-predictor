"""Le suivi complémentaire ne transforme pas un report en victoire."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import unittest

from src.lpf_edge_daily_selection import DailySelectionPick, SealedDailySelection
from src.lpf_edge_operational_selection import (
    OperationalSelectionDay, build_operational_selection,
)
from src.lpf_edge_postponement_reconciliation import RescheduledGame
from tests.test_lpf_edge_internal_prudence_reporting import make_day, make_score


def selection_for(day, odds: str):
    selected = day.predictions[0]
    pick = DailySelectionPick(
        rank=1, role="PRINCIPAL", game_id=selected.game_id,
        scheduled_start_utc=selected.scheduled_start_utc.isoformat(),
        home_team_name="Home", away_team_name="Away",
        predicted_side=selected.predicted_side, predicted_team_name="Home",
        model_probability=selected.predicted_probability,
        french_market_probability=Decimal("0.5"),
        gap_percentage_points=Decimal("10"),
        best_decimal_odds=Decimal(odds), best_bookmakers=("Book",),
        expected_value_percent=Decimal("10"),
        home_probable_pitcher_name="H", away_probable_pitcher_name="A",
    )
    return SealedDailySelection(
        target_date=day.target_date, selection_sha256="a" * 64,
        market_snapshot_sha256="b" * 64, status="PUBLISHED",
        eligible_count=1, picks=(pick,), rejection_reasons=(),
    )


class OperationalSelectionTests(unittest.TestCase):
    def test_a_report_is_void_and_excluded_from_roi_denominator(self):
        earlier = make_day(11, selected_probability="0.61")
        postponed = make_day(12, selected_probability="0.62")
        original = make_score(
            postponed, classification_correct=None, status="PENDING_POSTPONED",
        )
        pending = replace(original, pending_count=1)
        resolution = RescheduledGame(
            prediction_id=postponed.predictions[0].prediction_id,
            game_id=postponed.predictions[0].game_id,
            original_official_date=postponed.target_date,
            final_official_date=postponed.target_date + timedelta(days=1),
            away_score=2, home_score=4, receipt_sha256="e" * 64,
            observed_at_utc=datetime(2026, 9, 27, 8, tzinfo=timezone.utc),
        )
        report = build_operational_selection((
            OperationalSelectionDay(
                earlier.target_date, selection_for(earlier, "1.80"),
                make_score(earlier, classification_correct=True), {},
            ),
            OperationalSelectionDay(
                postponed.target_date, selection_for(postponed, "1.50"),
                pending, {resolution.game_id: resolution},
            ),
        ))
        self.assertEqual((report.evaluated_count, report.won_count,
                          report.lost_count, report.void_count,
                          report.supplemented_void_count, report.pending_count),
                         (1, 1, 0, 1, 1, 0))
        self.assertEqual(report.net_units, Decimal("0.80"))
        self.assertEqual(report.roi_percent, Decimal("80.00"))
        self.assertEqual(report.rows[-1].status, "VOID_RESCHEDULED")
        self.assertEqual(report.rows[-1].net_units, Decimal("0"))
        self.assertEqual("PENDING_POSTPONED", pending.results[0].outcome_status)


if __name__ == "__main__":
    unittest.main()
