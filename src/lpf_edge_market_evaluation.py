"""Évaluation descriptive des écarts LPF Edge face au marché français."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Iterable

from src.lpf_edge_dashboard import DailyScoreSummary
from src.lpf_edge_market_comparison import MarketComparison


HUNDRED = Decimal("100")
ONE = Decimal("1")
LIMITED_SAMPLE_THRESHOLD = 100


class LPFEdgeMarketEvaluationError(RuntimeError):
    """Les comparaisons et résultats ne peuvent pas être évalués sûrement."""


@dataclass(frozen=True, slots=True)
class MarketEvaluationRow:
    """Résultat théorique d’une mise fixe sur un pronostic comparable."""

    target_date: date
    game_id: int
    gap_percentage_points: Decimal
    decimal_odds: Decimal
    classification_correct: bool
    theoretical_net_units: Decimal


@dataclass(frozen=True, slots=True)
class GapBandSummary:
    """Bilan d’une tranche d’écart LPF Edge – marché français."""

    label: str
    evaluated_count: int
    correct_count: int
    accuracy_percent: Decimal | None
    theoretical_net_units: Decimal
    theoretical_roi_percent: Decimal | None


@dataclass(frozen=True, slots=True)
class MarketEvaluationHistory:
    """Historique agrégé sans pari réel ni recommandation automatique."""

    evaluated_rows: tuple[MarketEvaluationRow, ...]
    completed_day_count: int
    missing_odds_count: int
    unsettled_count: int

    @property
    def evaluated_count(self) -> int:
        return len(self.evaluated_rows)

    @property
    def correct_count(self) -> int:
        return sum(row.classification_correct for row in self.evaluated_rows)

    @property
    def accuracy_percent(self) -> Decimal | None:
        if not self.evaluated_rows:
            return None
        return (
            Decimal(self.correct_count)
            / Decimal(len(self.evaluated_rows))
            * HUNDRED
        )

    @property
    def theoretical_net_units(self) -> Decimal:
        return sum(
            (row.theoretical_net_units for row in self.evaluated_rows),
            Decimal("0"),
        )

    @property
    def theoretical_roi_percent(self) -> Decimal | None:
        if not self.evaluated_rows:
            return None
        return (
            self.theoretical_net_units
            / Decimal(len(self.evaluated_rows))
            * HUNDRED
        )

    @property
    def sample_is_limited(self) -> bool:
        return self.evaluated_count < LIMITED_SAMPLE_THRESHOLD

    @property
    def first_date(self) -> date | None:
        if not self.evaluated_rows:
            return None
        return min(row.target_date for row in self.evaluated_rows)

    @property
    def last_date(self) -> date | None:
        if not self.evaluated_rows:
            return None
        return max(row.target_date for row in self.evaluated_rows)

    @property
    def gap_bands(self) -> tuple[GapBandSummary, ...]:
        definitions = (
            ("Écart négatif", None, Decimal("0")),
            ("De 0 à moins de 2 pts", Decimal("0"), Decimal("2")),
            ("De 2 à moins de 5 pts", Decimal("2"), Decimal("5")),
            ("5 pts ou plus", Decimal("5"), None),
        )
        summaries: list[GapBandSummary] = []
        for label, lower, upper in definitions:
            rows = tuple(
                row
                for row in self.evaluated_rows
                if (lower is None or row.gap_percentage_points >= lower)
                and (upper is None or row.gap_percentage_points < upper)
            )
            correct = sum(row.classification_correct for row in rows)
            net = sum(
                (row.theoretical_net_units for row in rows),
                Decimal("0"),
            )
            count = len(rows)
            summaries.append(
                GapBandSummary(
                    label=label,
                    evaluated_count=count,
                    correct_count=correct,
                    accuracy_percent=(
                        None
                        if count == 0
                        else Decimal(correct) / Decimal(count) * HUNDRED
                    ),
                    theoretical_net_units=net,
                    theoretical_roi_percent=(
                        None
                        if count == 0
                        else net / Decimal(count) * HUNDRED
                    ),
                )
            )
        return tuple(summaries)


def build_market_evaluation_history(
    completed_days: Iterable[tuple[MarketComparison, DailyScoreSummary]],
) -> MarketEvaluationHistory:
    """Évalue une unité théorique par pronostic terminé et comparable."""
    evaluated_rows: list[MarketEvaluationRow] = []
    missing_odds_count = 0
    unsettled_count = 0
    completed_dates: set[date] = set()

    for comparison, score in completed_days:
        if comparison.target_date in completed_dates:
            raise LPFEdgeMarketEvaluationError(
                "Une journée est répétée dans l’historique."
            )
        completed_dates.add(comparison.target_date)

        comparison_by_game = {row.game_id: row for row in comparison.rows}
        if len(comparison_by_game) != len(comparison.rows):
            raise LPFEdgeMarketEvaluationError(
                "Un match est répété dans la comparaison de marché."
            )
        results_by_game = {result.game_id: result for result in score.results}
        if len(results_by_game) != len(score.results):
            raise LPFEdgeMarketEvaluationError(
                "Un match est répété dans les résultats vérifiés."
            )
        if set(comparison_by_game) != set(results_by_game):
            raise LPFEdgeMarketEvaluationError(
                "La comparaison et les résultats ne couvrent pas les mêmes matchs."
            )

        for game_id, comparison_row in comparison_by_game.items():
            result = results_by_game[game_id]
            if (
                result.predicted_side != comparison_row.predicted_side
                or result.predicted_probability
                != comparison_row.model_probability
            ):
                raise LPFEdgeMarketEvaluationError(
                    "Le pronostic comparé diverge du résultat vérifié."
                )
            if (
                result.classification_correct is not None
                and type(result.classification_correct) is not bool
            ):
                raise LPFEdgeMarketEvaluationError(
                    "La classification du résultat est invalide."
                )
            if result.classification_correct is None:
                unsettled_count += 1
                continue
            if not comparison_row.has_market_comparison:
                missing_odds_count += 1
                continue
            if (
                comparison_row.gap_percentage_points is None
                or comparison_row.best_decimal_odds is None
                or not comparison_row.best_decimal_odds.is_finite()
                or comparison_row.best_decimal_odds <= ONE
            ):
                raise LPFEdgeMarketEvaluationError(
                    "La cote figée du pronostic est invalide."
                )

            net_units = (
                comparison_row.best_decimal_odds - ONE
                if result.classification_correct
                else -ONE
            )
            evaluated_rows.append(
                MarketEvaluationRow(
                    target_date=comparison.target_date,
                    game_id=game_id,
                    gap_percentage_points=(
                        comparison_row.gap_percentage_points
                    ),
                    decimal_odds=comparison_row.best_decimal_odds,
                    classification_correct=result.classification_correct,
                    theoretical_net_units=net_units,
                )
            )

    return MarketEvaluationHistory(
        evaluated_rows=tuple(
            sorted(evaluated_rows, key=lambda row: (row.target_date, row.game_id))
        ),
        completed_day_count=len(completed_dates),
        missing_odds_count=missing_odds_count,
        unsettled_count=unsettled_count,
    )


__all__ = [
    "GapBandSummary",
    "LIMITED_SAMPLE_THRESHOLD",
    "LPFEdgeMarketEvaluationError",
    "MarketEvaluationHistory",
    "MarketEvaluationRow",
    "build_market_evaluation_history",
]
