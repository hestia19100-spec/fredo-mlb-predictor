"""Tests de l’évaluation historique LPF Edge face au marché français."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
import unittest

from src.lpf_edge_dashboard import DailyScoreResult, DailyScoreSummary
from src.lpf_edge_market_comparison import (
    MarketComparison,
    MarketComparisonRow,
)
from src.lpf_edge_market_evaluation import (
    LPFEdgeMarketEvaluationError,
    build_market_evaluation_history,
)


UTC = timezone.utc
TARGET = date(2026, 9, 13)


def comparison_row(
    game_id: int,
    *,
    gap: Decimal | None,
    odds: Decimal | None,
    side: str = "HOME",
    probability: Decimal = Decimal("0.60"),
) -> MarketComparisonRow:
    return MarketComparisonRow(
        game_id=game_id,
        scheduled_start_utc=datetime(2026, 9, 13, 18, 0, tzinfo=UTC),
        home_team_name="Domicile",
        away_team_name="Extérieur",
        predicted_side=side,
        predicted_team_name="Domicile" if side == "HOME" else "Extérieur",
        model_probability=probability,
        french_market_probability=(
            None
            if gap is None
            else probability - gap / Decimal("100")
        ),
        gap_percentage_points=gap,
        best_decimal_odds=odds,
        best_bookmakers=("PMU",) if odds is not None else (),
        bookmaker_count=1 if odds is not None else 0,
    )


def comparison(
    *rows: MarketComparisonRow,
    target: date = TARGET,
) -> MarketComparison:
    return MarketComparison(
        target_date=target,
        odds_run_id=7,
        odds_completed_at_utc=datetime(2026, 9, 13, 11, 0, tzinfo=UTC),
        rows=tuple(rows),
    )


def score_result(
    game_id: int,
    *,
    correct: bool | None,
    side: str = "HOME",
    probability: Decimal = Decimal("0.60"),
) -> DailyScoreResult:
    return DailyScoreResult(
        prediction_id=f"prediction-{game_id}",
        game_id=game_id,
        away_team_id=10,
        home_team_id=20,
        predicted_side=side,
        predicted_probability=probability,
        outcome_status="SCORED_FINAL" if correct is not None else "PENDING",
        away_score=2 if correct is not None else None,
        home_score=3 if correct is not None else None,
        actual_winner=(
            "HOME" if correct is True else "AWAY" if correct is False else None
        ),
        classification_correct=correct,
    )


def score(*results: DailyScoreResult) -> DailyScoreSummary:
    correct = sum(result.classification_correct is True for result in results)
    incorrect = sum(result.classification_correct is False for result in results)
    scored = correct + incorrect
    return DailyScoreSummary(
        checkpoint_date=date(2026, 9, 14),
        scored_count=scored,
        void_count=0,
        pending_count=len(results) - scored,
        correct_count=correct,
        incorrect_count=incorrect,
        accuracy=None if scored == 0 else correct / scored,
        mean_log_loss=None,
        mean_brier_score=None,
        results=tuple(results),
    )


class LPFEdgeMarketEvaluationTests(unittest.TestCase):
    def test_win_returns_decimal_odds_minus_one_unit(self) -> None:
        history = build_market_evaluation_history(
            (
                (
                    comparison(
                        comparison_row(
                            100, gap=Decimal("3.0"), odds=Decimal("1.80")
                        )
                    ),
                    score(score_result(100, correct=True)),
                ),
            )
        )

        self.assertEqual(history.evaluated_count, 1)
        self.assertEqual(history.correct_count, 1)
        self.assertEqual(history.theoretical_net_units, Decimal("0.80"))
        self.assertEqual(history.theoretical_roi_percent, Decimal("80.00"))

    def test_loss_returns_minus_one_unit(self) -> None:
        history = build_market_evaluation_history(
            (
                (
                    comparison(
                        comparison_row(
                            100, gap=Decimal("1.0"), odds=Decimal("2.25")
                        )
                    ),
                    score(score_result(100, correct=False)),
                ),
            )
        )

        self.assertEqual(history.correct_count, 0)
        self.assertEqual(history.accuracy_percent, Decimal("0"))
        self.assertEqual(history.theoretical_net_units, Decimal("-1"))
        self.assertEqual(history.theoretical_roi_percent, Decimal("-100"))

    def test_missing_odds_and_unsettled_results_are_excluded_separately(
        self,
    ) -> None:
        history = build_market_evaluation_history(
            (
                (
                    comparison(
                        comparison_row(100, gap=None, odds=None),
                        comparison_row(
                            101, gap=Decimal("4"), odds=Decimal("1.90")
                        ),
                    ),
                    score(
                        score_result(100, correct=True),
                        score_result(101, correct=None),
                    ),
                ),
            )
        )

        self.assertEqual(history.evaluated_count, 0)
        self.assertEqual(history.missing_odds_count, 1)
        self.assertEqual(history.unsettled_count, 1)
        self.assertIsNone(history.accuracy_percent)
        self.assertIsNone(history.theoretical_roi_percent)

    def test_gap_band_boundaries_are_exact(self) -> None:
        rows = (
            comparison_row(100, gap=Decimal("-0.1"), odds=Decimal("2.0")),
            comparison_row(101, gap=Decimal("0"), odds=Decimal("2.0")),
            comparison_row(102, gap=Decimal("2"), odds=Decimal("2.0")),
            comparison_row(103, gap=Decimal("5"), odds=Decimal("2.0")),
        )
        results = tuple(
            score_result(game_id, correct=True)
            for game_id in (100, 101, 102, 103)
        )

        history = build_market_evaluation_history(
            ((comparison(*rows), score(*results)),)
        )

        self.assertEqual(
            tuple(band.evaluated_count for band in history.gap_bands),
            (1, 1, 1, 1),
        )

    def test_days_are_aggregated_in_chronological_order(self) -> None:
        earlier = date(2026, 9, 12)
        history = build_market_evaluation_history(
            (
                (
                    comparison(
                        comparison_row(200, gap=Decimal("1"), odds=Decimal("2")),
                        target=TARGET,
                    ),
                    score(score_result(200, correct=True)),
                ),
                (
                    comparison(
                        comparison_row(100, gap=Decimal("1"), odds=Decimal("2")),
                        target=earlier,
                    ),
                    score(score_result(100, correct=False)),
                ),
            )
        )

        self.assertEqual(history.completed_day_count, 2)
        self.assertEqual(history.first_date, earlier)
        self.assertEqual(history.last_date, TARGET)
        self.assertEqual(
            tuple(row.game_id for row in history.evaluated_rows),
            (100, 200),
        )

    def test_duplicate_day_is_rejected(self) -> None:
        entry = (
            comparison(
                comparison_row(100, gap=Decimal("1"), odds=Decimal("2"))
            ),
            score(score_result(100, correct=True)),
        )
        with self.assertRaisesRegex(
            LPFEdgeMarketEvaluationError,
            "journée est répétée",
        ):
            build_market_evaluation_history((entry, entry))

    def test_different_game_sets_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            LPFEdgeMarketEvaluationError,
            "mêmes matchs",
        ):
            build_market_evaluation_history(
                (
                    (
                        comparison(
                            comparison_row(
                                100, gap=Decimal("1"), odds=Decimal("2")
                            )
                        ),
                        score(score_result(101, correct=True)),
                    ),
                )
            )

    def test_prediction_disagreement_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            LPFEdgeMarketEvaluationError,
            "pronostic comparé diverge",
        ):
            build_market_evaluation_history(
                (
                    (
                        comparison(
                            comparison_row(
                                100, gap=Decimal("1"), odds=Decimal("2")
                            )
                        ),
                        score(score_result(100, correct=True, side="AWAY")),
                    ),
                )
            )

    def test_evaluator_has_no_network_database_model_or_write_dependency(
        self,
    ) -> None:
        source = Path(__file__).parents[1].joinpath(
            "src", "lpf_edge_market_evaluation.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("requests", source)
        self.assertNotIn("sqlite3", source)
        self.assertNotIn("predict_proba", source)
        self.assertNotIn("INSERT ", source)
        self.assertNotIn("UPDATE ", source)


if __name__ == "__main__":
    unittest.main()
