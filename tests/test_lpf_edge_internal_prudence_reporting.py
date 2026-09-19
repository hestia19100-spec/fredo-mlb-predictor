"""Tests du bilan quotidien recalculé du choix privé."""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from src.lpf_edge_dashboard import (
    SCORED_FINAL_STATUS,
    CertifiedPrediction,
    CertifiedPredictionDay,
    DailyScoreResult,
    DailyScoreSummary,
)
from src.lpf_edge_internal_prudence_reporting import (
    InternalPrudenceEvidence,
    LPFEdgeInternalPrudenceReportingError,
    PrudenceResultStatus,
    build_internal_prudence_report,
)


NAMES = {1: "Home", 2: "Away", 3: "Other Home", 4: "Other Away"}


def make_day(
    day_offset: int,
    *,
    selected_probability: str,
) -> CertifiedPredictionDay:
    target = date(2026, 9, 10) + timedelta(days=day_offset)
    probability = Decimal(selected_probability)
    predictions = (
        CertifiedPrediction(
            prediction_id=f"choice-{day_offset}",
            batch_id=f"batch-{day_offset}",
            game_id=100 + day_offset,
            official_date=target,
            away_team_id=2,
            home_team_id=1,
            scheduled_start_utc=datetime(
                2026, 9, 10, 18, tzinfo=timezone.utc
            )
            + timedelta(days=day_offset),
            issued_at_utc=datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
            + timedelta(days=day_offset),
            p_home_win=probability,
            p_away_win=Decimal("1") - probability,
        ),
        CertifiedPrediction(
            prediction_id=f"lower-{day_offset}",
            batch_id=f"batch-{day_offset}",
            game_id=200 + day_offset,
            official_date=target,
            away_team_id=4,
            home_team_id=3,
            scheduled_start_utc=datetime(
                2026, 9, 10, 20, tzinfo=timezone.utc
            )
            + timedelta(days=day_offset),
            issued_at_utc=datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
            + timedelta(days=day_offset),
            p_home_win=Decimal("0.55"),
            p_away_win=Decimal("0.45"),
        ),
    )
    return CertifiedPredictionDay(
        target_date=target,
        batch_id=f"batch-{day_offset}",
        results_commit="a" * 40,
        certified_at_utc=datetime(2026, 9, 10, 13, tzinfo=timezone.utc)
        + timedelta(days=day_offset),
        remote_lead_minutes=Decimal("120"),
        predictions_sha256=f"{day_offset + 1:064x}",
        receipt_sha256=f"{day_offset + 2:064x}",
        predictions=predictions,
    )


def make_score(
    day: CertifiedPredictionDay,
    *,
    classification_correct: bool | None,
    status: str = SCORED_FINAL_STATUS,
) -> DailyScoreSummary:
    selected = day.predictions[0]
    return DailyScoreSummary(
        checkpoint_date=day.target_date + timedelta(days=1),
        scored_count=1 if status == SCORED_FINAL_STATUS else 0,
        void_count=1 if status == "VOID_CANCELLED" else 0,
        pending_count=1 if status == "PENDING_FINAL" else 0,
        correct_count=1 if classification_correct is True else 0,
        incorrect_count=1 if classification_correct is False else 0,
        accuracy=None,
        mean_log_loss=None,
        mean_brier_score=None,
        results=(
            DailyScoreResult(
                prediction_id=selected.prediction_id,
                game_id=selected.game_id,
                away_team_id=selected.away_team_id,
                home_team_id=selected.home_team_id,
                predicted_side=selected.predicted_side,
                predicted_probability=selected.predicted_probability,
                outcome_status=status,
                away_score=2 if status == SCORED_FINAL_STATUS else None,
                home_score=5 if status == SCORED_FINAL_STATUS else None,
                actual_winner="HOME" if status == SCORED_FINAL_STATUS else None,
                classification_correct=classification_correct,
            ),
        ),
    )


class LPFEdgeInternalPrudenceReportingTests(unittest.TestCase):
    def test_report_covers_wins_losses_pending_void_and_streaks(self) -> None:
        days = [
            make_day(0, selected_probability="0.61"),
            make_day(1, selected_probability="0.62"),
            make_day(2, selected_probability="0.63"),
            make_day(3, selected_probability="0.64"),
            make_day(4, selected_probability="0.65"),
        ]
        report = build_internal_prudence_report(
            (
                InternalPrudenceEvidence(days[0], NAMES, make_score(days[0], classification_correct=True)),
                InternalPrudenceEvidence(days[1], NAMES, make_score(days[1], classification_correct=True)),
                InternalPrudenceEvidence(days[2], NAMES, make_score(days[2], classification_correct=False)),
                InternalPrudenceEvidence(days[3], NAMES, None),
                InternalPrudenceEvidence(
                    days[4],
                    NAMES,
                    make_score(
                        days[4],
                        classification_correct=None,
                        status="VOID_CANCELLED",
                    ),
                ),
            )
        )

        self.assertEqual(report.total_count, 5)
        self.assertEqual(report.won_count, 2)
        self.assertEqual(report.lost_count, 1)
        self.assertEqual(report.pending_count, 1)
        self.assertEqual(report.void_count, 1)
        self.assertEqual(
            report.hit_rate_percent,
            Decimal(2) * Decimal(100) / Decimal(3),
        )
        self.assertEqual(report.mean_model_probability, Decimal("0.63"))
        self.assertEqual(report.minimum_model_probability, Decimal("0.61"))
        self.assertEqual(report.maximum_model_probability, Decimal("0.65"))
        self.assertEqual(report.longest_winning_streak, 2)
        self.assertEqual(report.longest_losing_streak, 1)
        self.assertEqual(report.current_streak_status, PrudenceResultStatus.LOST)
        self.assertEqual(report.current_streak_count, 1)
        self.assertEqual(report.days[0].score_text, "5 - 2")

    def test_rebuild_after_morning_score_updates_pending_day(self) -> None:
        day = make_day(0, selected_probability="0.61")
        before = build_internal_prudence_report(
            (InternalPrudenceEvidence(day, NAMES, None),)
        )
        after = build_internal_prudence_report(
            (
                InternalPrudenceEvidence(
                    day,
                    NAMES,
                    make_score(day, classification_correct=True),
                ),
            )
        )

        self.assertEqual(before.pending_count, 1)
        self.assertIsNone(before.hit_rate_percent)
        self.assertEqual(after.pending_count, 0)
        self.assertEqual(after.won_count, 1)
        self.assertEqual(after.hit_rate_percent, Decimal("100"))

    def test_result_of_lower_probability_match_cannot_change_choice(self) -> None:
        day = make_day(0, selected_probability="0.61")
        lower = day.predictions[1]
        wrong_score = DailyScoreSummary(
            checkpoint_date=day.target_date + timedelta(days=1),
            scored_count=1,
            void_count=0,
            pending_count=0,
            correct_count=1,
            incorrect_count=0,
            accuracy=1.0,
            mean_log_loss=None,
            mean_brier_score=None,
            results=(
                DailyScoreResult(
                    prediction_id=lower.prediction_id,
                    game_id=lower.game_id,
                    away_team_id=lower.away_team_id,
                    home_team_id=lower.home_team_id,
                    predicted_side=lower.predicted_side,
                    predicted_probability=lower.predicted_probability,
                    outcome_status=SCORED_FINAL_STATUS,
                    away_score=1,
                    home_score=2,
                    actual_winner="HOME",
                    classification_correct=True,
                ),
            ),
        )

        with self.assertRaises(LPFEdgeInternalPrudenceReportingError):
            build_internal_prudence_report(
                (InternalPrudenceEvidence(day, NAMES, wrong_score),)
            )

    def test_duplicate_day_is_rejected(self) -> None:
        day = make_day(0, selected_probability="0.61")
        item = InternalPrudenceEvidence(day, NAMES, None)
        with self.assertRaises(LPFEdgeInternalPrudenceReportingError):
            build_internal_prudence_report((item, item))

    def test_empty_history_is_rejected(self) -> None:
        with self.assertRaises(LPFEdgeInternalPrudenceReportingError):
            build_internal_prudence_report(())


if __name__ == "__main__":
    unittest.main()
