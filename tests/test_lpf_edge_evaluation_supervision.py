"""Tests du tableau de supervision de l'évaluation LPF Edge MLB."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
import unittest

from src.lpf_edge_dashboard import (
    CertifiedPrediction,
    CertifiedPredictionDay,
    DailyScoreResult,
    DailyScoreSummary,
)
from src.lpf_edge_daily_selection import DailySelectionPick, SealedDailySelection
from src.lpf_edge_evaluation_supervision import (
    EvaluationDayEvidence,
    LPFEdgeEvaluationSupervisionError,
    build_evaluation_supervision,
)
from src.lpf_edge_market_settlement import SealedMarketSettlement


UTC = timezone.utc


def prediction(target: date, game_id: int) -> CertifiedPrediction:
    return CertifiedPrediction(
        prediction_id=f"prediction-{game_id}",
        batch_id="a" * 64,
        game_id=game_id,
        official_date=target,
        away_team_id=game_id + 10,
        home_team_id=game_id + 20,
        scheduled_start_utc=datetime(2026, 9, 14, 18, tzinfo=UTC),
        issued_at_utc=datetime(2026, 9, 14, 12, tzinfo=UTC),
        p_home_win=Decimal("0.60"),
        p_away_win=Decimal("0.40"),
    )


def prediction_day(target: date, *game_ids: int) -> CertifiedPredictionDay:
    return CertifiedPredictionDay(
        target_date=target,
        batch_id="a" * 64,
        results_commit="b" * 40,
        certified_at_utc=datetime(2026, 9, 14, 12, 1, tzinfo=UTC),
        remote_lead_minutes=Decimal("120"),
        predictions_sha256="c" * 64,
        receipt_sha256="d" * 64,
        predictions=tuple(prediction(target, game_id) for game_id in game_ids),
    )


def score(target: date, outcomes: dict[int, bool | None], *, metric: str) -> DailyScoreSummary:
    results = tuple(
        DailyScoreResult(
            prediction_id=f"prediction-{game_id}",
            game_id=game_id,
            away_team_id=game_id + 10,
            home_team_id=game_id + 20,
            predicted_side="HOME",
            predicted_probability=Decimal("0.60"),
            outcome_status=("SCORED_FINAL" if outcome is not None else "VOID_CANCELLED"),
            away_score=2 if outcome is not None else None,
            home_score=(3 if outcome else 1) if outcome is not None else None,
            actual_winner=(
                "HOME" if outcome is True else "AWAY" if outcome is False else None
            ),
            classification_correct=outcome,
        )
        for game_id, outcome in outcomes.items()
    )
    evaluated = sum(value is not None for value in outcomes.values())
    correct = sum(value is True for value in outcomes.values())
    void = sum(value is None for value in outcomes.values())
    return DailyScoreSummary(
        checkpoint_date=target,
        scored_count=evaluated,
        void_count=void,
        pending_count=0,
        correct_count=correct,
        incorrect_count=evaluated - correct,
        accuracy=correct / evaluated if evaluated else None,
        mean_log_loss=float(metric),
        mean_brier_score=float(Decimal(metric) / Decimal("2")),
        results=results,
    )


def pick(target: date, game_id: int, rank: int, odds: str) -> DailySelectionPick:
    return DailySelectionPick(
        rank=rank,
        role="PRINCIPAL" if rank == 1 else "SECONDAIRE",
        game_id=game_id,
        scheduled_start_utc=f"{target.isoformat()}T18:00:00.000000Z",
        home_team_name=f"Domicile {game_id}",
        away_team_name=f"Extérieur {game_id}",
        predicted_side="HOME",
        predicted_team_name=f"Domicile {game_id}",
        model_probability=Decimal("0.60"),
        french_market_probability=Decimal("0.55"),
        gap_percentage_points=Decimal("5.0"),
        best_decimal_odds=Decimal(odds),
        best_bookmakers=("PMU", "Betclic"),
        expected_value_percent=Decimal("8.0"),
        home_probable_pitcher_name="Lanceur domicile",
        away_probable_pitcher_name="Lanceur extérieur",
    )


def selection(target: date, *picks: DailySelectionPick) -> SealedDailySelection:
    return SealedDailySelection(
        target_date=target,
        selection_sha256=(target.isoformat().replace("-", "") + "e" * 56)[:64],
        market_snapshot_sha256="f" * 64,
        status="PICKS_AVAILABLE" if picks else "NO_PICK",
        eligible_count=len(picks),
        picks=tuple(picks),
        rejection_reasons=(),
    )


def settlement(
    target: date,
    selected: SealedDailySelection,
    outcomes: dict[int, bool | None],
) -> SealedMarketSettlement:
    selected_results = [outcomes[item.game_id] for item in selected.picks]
    evaluated_picks = [
        item
        for item in selected.picks
        if outcomes[item.game_id] is not None
    ]
    net = sum(
        (
            item.best_decimal_odds - Decimal("1")
            if outcomes[item.game_id]
            else Decimal("-1")
        )
        for item in evaluated_picks
    )
    evaluated = len(evaluated_picks)
    correct = sum(outcomes[item.game_id] is True for item in evaluated_picks)
    return SealedMarketSettlement(
        target_date=target,
        checkpoint_date=target,
        observation_id="1" * 64,
        settlement_sha256="2" * 64,
        snapshot_sha256="f" * 64,
        evaluated_count=len(outcomes),
        correct_count=sum(value is True for value in outcomes.values()),
        missing_market_count=0,
        void_count=sum(value is None for value in outcomes.values()),
        accuracy_percent=None,
        theoretical_net_units=Decimal("0"),
        theoretical_roi_percent=None,
        daily_selection_sha256=selected.selection_sha256,
        selected_count=len(selected.picks),
        selected_evaluated_count=evaluated,
        selected_correct_count=correct,
        selected_void_count=sum(value is None for value in selected_results),
        selected_theoretical_net_units=net,
        selected_theoretical_roi_percent=(
            None if evaluated == 0 else net / Decimal(evaluated) * Decimal("100")
        ),
    )


class LPFEdgeEvaluationSupervisionTests(unittest.TestCase):
    def test_shadow_and_selector_are_aggregated_separately(self) -> None:
        first = date(2026, 9, 14)
        waiting = date(2026, 9, 15)
        third = date(2026, 9, 16)
        first_selection = selection(
            first,
            pick(first, 101, 1, "1.80"),
            pick(first, 102, 2, "1.90"),
        )
        third_selection = selection(
            third,
            pick(third, 301, 1, "1.70"),
            pick(third, 302, 2, "2.10"),
        )
        first_outcomes = {101: True, 102: False, 103: True, 104: True}
        third_outcomes = {301: False, 302: False, 303: True, 304: False}
        supervision = build_evaluation_supervision(
            (
                EvaluationDayEvidence(
                    prediction_day(first, *first_outcomes),
                    score(first, first_outcomes, metric="0.5"),
                    first_selection,
                    settlement(first, first_selection, first_outcomes),
                ),
                EvaluationDayEvidence(
                    prediction_day(waiting, 201, 202),
                    None,
                    selection(waiting),
                    None,
                ),
                EvaluationDayEvidence(
                    prediction_day(third, *third_outcomes),
                    score(third, third_outcomes, metric="0.7"),
                    third_selection,
                    settlement(third, third_selection, third_outcomes),
                ),
            )
        )

        self.assertEqual(supervision.certified_day_count, 3)
        self.assertEqual(supervision.completed_day_count, 2)
        self.assertEqual(supervision.prediction_count, 10)
        self.assertEqual(supervision.shadow_evaluated_count, 8)
        self.assertEqual(supervision.shadow_correct_count, 4)
        self.assertEqual(supervision.shadow_accuracy_percent, Decimal("50.0"))
        self.assertEqual(supervision.shadow_pending_count, 2)
        self.assertEqual(supervision.weighted_log_loss, Decimal("0.6"))
        self.assertEqual(supervision.weighted_brier_score, Decimal("0.3"))
        self.assertEqual(supervision.shadow_progress_percent, Decimal("8.00"))

        self.assertEqual(supervision.selection_publication_day_count, 3)
        self.assertEqual(supervision.selection_day_count, 2)
        self.assertEqual(supervision.no_pick_day_count, 1)
        self.assertEqual(supervision.selection.published_count, 4)
        self.assertEqual(supervision.selection.evaluated_count, 4)
        self.assertEqual(supervision.selection.correct_count, 1)
        self.assertEqual(supervision.selection.accuracy_percent, Decimal("25.00"))
        self.assertEqual(supervision.selection.theoretical_net_units, Decimal("-2.20"))
        self.assertEqual(supervision.selection.theoretical_roi_percent, Decimal("-55.00"))
        self.assertEqual(supervision.selection.current_losing_streak, 3)
        self.assertEqual(supervision.selection.maximum_losing_streak, 3)
        self.assertEqual(supervision.selection.maximum_drawdown_units, Decimal("3.00"))
        self.assertEqual(supervision.principal.correct_count, 1)
        self.assertEqual(supervision.secondary.correct_count, 0)
        self.assertEqual(supervision.secondary.maximum_losing_streak, 2)
        self.assertEqual([row.target_date for row in supervision.days], [first, waiting, third])

    def test_unsettled_selection_is_counted_as_pending(self) -> None:
        target = date(2026, 9, 14)
        selected = selection(target, pick(target, 101, 1, "1.80"))
        supervision = build_evaluation_supervision(
            (EvaluationDayEvidence(prediction_day(target, 101), None, selected, None),)
        )
        self.assertEqual(supervision.selection.published_count, 1)
        self.assertEqual(supervision.selection.pending_count, 1)
        self.assertEqual(supervision.selection.evaluated_count, 0)
        self.assertEqual(supervision.days[0].selection_status, "EN_ATTENTE")

    def test_duplicate_day_is_rejected(self) -> None:
        target = date(2026, 9, 14)
        evidence = EvaluationDayEvidence(prediction_day(target, 101), None, None, None)
        with self.assertRaisesRegex(LPFEdgeEvaluationSupervisionError, "répétée"):
            build_evaluation_supervision((evidence, evidence))

    def test_forged_selection_settlement_link_is_rejected(self) -> None:
        target = date(2026, 9, 14)
        outcomes = {101: True}
        selected = selection(target, pick(target, 101, 1, "1.80"))
        forged = settlement(target, selected, outcomes)
        forged = SealedMarketSettlement(
            **{
                field: getattr(forged, field)
                for field in forged.__dataclass_fields__
                if field != "daily_selection_sha256"
            },
            daily_selection_sha256="0" * 64,
        )
        with self.assertRaisesRegex(LPFEdgeEvaluationSupervisionError, "ne concordent"):
            build_evaluation_supervision(
                (
                    EvaluationDayEvidence(
                        prediction_day(target, 101),
                        score(target, outcomes, metric="0.5"),
                        selected,
                        forged,
                    ),
                )
            )

    def test_empty_history_is_valid(self) -> None:
        supervision = build_evaluation_supervision(())
        self.assertEqual(supervision.certified_day_count, 0)
        self.assertEqual(supervision.selection.evaluated_count, 0)
        self.assertIsNone(supervision.first_date)
        self.assertIsNone(supervision.last_date)

    def test_source_has_no_network_database_model_or_write_dependency(self) -> None:
        source = Path(__file__).parents[1].joinpath(
            "src", "lpf_edge_evaluation_supervision.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("requests", source)
        self.assertNotIn("sqlite3", source)
        self.assertNotIn("predict_proba", source)
        self.assertNotIn("write_bytes", source)
        self.assertNotIn("write_text", source)


if __name__ == "__main__":
    unittest.main()
