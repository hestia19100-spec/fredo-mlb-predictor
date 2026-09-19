"""Statistiques privées recalculées depuis prédictions et scores vérifiés."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Mapping, Sequence

from src.lpf_edge_dashboard import (
    DailyScoreSummary,
    PENDING_STATUSES,
    SCORED_FINAL_STATUS,
    VOID_STATUSES,
    CertifiedPredictionDay,
)
from src.lpf_edge_internal_prudence import (
    InternalPrudenceChoice,
    LPFEdgeInternalPrudenceError,
    select_internal_prudence,
)
from src.lpf_edge_market_comparison import MarketComparison


ONE = Decimal("1")
HUNDRED = Decimal("100")


class PrudenceResultStatus(str, Enum):
    WON = "WON"
    LOST = "LOST"
    PENDING = "PENDING"
    VOID = "VOID"


class LPFEdgeInternalPrudenceReportingError(ValueError):
    """Le bilan privé ne concorde pas avec les preuves vérifiées."""


@dataclass(frozen=True, slots=True)
class InternalPrudenceEvidence:
    prediction_day: CertifiedPredictionDay
    team_names: Mapping[int, str]
    score: DailyScoreSummary | None
    market_comparison: MarketComparison | None = None


@dataclass(frozen=True, slots=True)
class InternalPrudenceDayResult:
    choice: InternalPrudenceChoice
    result_status: PrudenceResultStatus
    home_score: int | None
    away_score: int | None
    best_decimal_odds: Decimal | None
    best_bookmakers: tuple[str, ...]
    net_result_euros: Decimal | None

    @property
    def score_text(self) -> str:
        if self.home_score is None or self.away_score is None:
            return "—"
        return f"{self.home_score} - {self.away_score}"


@dataclass(frozen=True, slots=True)
class InternalPrudenceReport:
    days: tuple[InternalPrudenceDayResult, ...]
    total_count: int
    won_count: int
    lost_count: int
    pending_count: int
    void_count: int
    hit_rate_percent: Decimal | None
    mean_model_probability: Decimal
    minimum_model_probability: Decimal
    maximum_model_probability: Decimal
    longest_winning_streak: int
    longest_losing_streak: int
    current_streak_status: PrudenceResultStatus | None
    current_streak_count: int
    odds_evaluated_count: int
    missing_odds_count: int
    theoretical_net_euros: Decimal
    theoretical_roi_percent: Decimal | None


def _market_odds(
    choice: InternalPrudenceChoice,
    comparison: MarketComparison | None,
) -> tuple[Decimal | None, tuple[str, ...]]:
    if comparison is None:
        return None, ()
    if comparison.target_date != choice.target_date:
        raise LPFEdgeInternalPrudenceReportingError(
            "La comparaison de marché vise une autre journée."
        )
    rows = [row for row in comparison.rows if row.game_id == choice.game_id]
    if len(rows) != 1:
        raise LPFEdgeInternalPrudenceReportingError(
            "La cote du choix interne est absente ou répétée."
        )
    row = rows[0]
    if (
        row.scheduled_start_utc != choice.scheduled_start_utc
        or row.home_team_name != choice.home_team_name
        or row.away_team_name != choice.away_team_name
        or row.predicted_side != choice.predicted_side
        or row.predicted_team_name != choice.predicted_team_name
        or row.model_probability != choice.model_probability
    ):
        raise LPFEdgeInternalPrudenceReportingError(
            "La cote ne correspond pas exactement au choix interne."
        )
    odds = row.best_decimal_odds
    bookmakers = row.best_bookmakers
    if odds is None:
        if bookmakers:
            raise LPFEdgeInternalPrudenceReportingError(
                "Des bookmakers existent sans cote française."
            )
        return None, ()
    if not odds.is_finite() or odds <= ONE or not bookmakers:
        raise LPFEdgeInternalPrudenceReportingError(
            "La meilleure cote française est invalide."
        )
    return odds, bookmakers


def _evaluate_choice(
    choice: InternalPrudenceChoice,
    score: DailyScoreSummary | None,
    market_comparison: MarketComparison | None,
) -> InternalPrudenceDayResult:
    best_odds, best_bookmakers = _market_odds(choice, market_comparison)
    if score is None:
        return InternalPrudenceDayResult(
            choice=choice,
            result_status=PrudenceResultStatus.PENDING,
            home_score=None,
            away_score=None,
            best_decimal_odds=best_odds,
            best_bookmakers=best_bookmakers,
            net_result_euros=None,
        )
    matches = [
        result
        for result in score.results
        if result.prediction_id == choice.prediction_id
        and result.game_id == choice.game_id
    ]
    if len(matches) != 1:
        raise LPFEdgeInternalPrudenceReportingError(
            "Le résultat du choix interne est absent ou répété."
        )
    result = matches[0]
    if result.outcome_status in VOID_STATUSES:
        status = PrudenceResultStatus.VOID
    elif result.outcome_status in PENDING_STATUSES:
        status = PrudenceResultStatus.PENDING
    elif (
        result.outcome_status == SCORED_FINAL_STATUS
        and result.classification_correct is True
    ):
        status = PrudenceResultStatus.WON
    elif (
        result.outcome_status == SCORED_FINAL_STATUS
        and result.classification_correct is False
    ):
        status = PrudenceResultStatus.LOST
    else:
        raise LPFEdgeInternalPrudenceReportingError(
            "Le verdict du choix interne est incohérent."
        )
    if status is PrudenceResultStatus.VOID:
        net_result = Decimal("0")
    elif status is PrudenceResultStatus.PENDING or best_odds is None:
        net_result = None
    elif status is PrudenceResultStatus.WON:
        net_result = best_odds - ONE
    else:
        net_result = -ONE
    return InternalPrudenceDayResult(
        choice=choice,
        result_status=status,
        home_score=result.home_score,
        away_score=result.away_score,
        best_decimal_odds=best_odds,
        best_bookmakers=best_bookmakers,
        net_result_euros=net_result,
    )


def _longest_streak(
    statuses: Sequence[PrudenceResultStatus],
    target: PrudenceResultStatus,
) -> int:
    longest = 0
    current = 0
    for status in statuses:
        if status is target:
            current += 1
            longest = max(longest, current)
        elif status in {PrudenceResultStatus.WON, PrudenceResultStatus.LOST}:
            current = 0
    return longest


def _current_streak(
    statuses: Sequence[PrudenceResultStatus],
) -> tuple[PrudenceResultStatus | None, int]:
    resolved = [
        status
        for status in statuses
        if status in {PrudenceResultStatus.WON, PrudenceResultStatus.LOST}
    ]
    if not resolved:
        return None, 0
    target = resolved[-1]
    count = 0
    for status in reversed(resolved):
        if status is not target:
            break
        count += 1
    return target, count


def build_internal_prudence_report(
    evidence: Sequence[InternalPrudenceEvidence],
) -> InternalPrudenceReport:
    """Recalcule le bilan ; une nouvelle routine de scoring suffit à l'actualiser."""

    if not isinstance(evidence, Sequence) or not evidence:
        raise LPFEdgeInternalPrudenceReportingError(
            "Au moins une journée certifiée est nécessaire."
        )
    ordered = sorted(evidence, key=lambda item: item.prediction_day.target_date)
    dates = [item.prediction_day.target_date for item in ordered]
    if len(dates) != len(set(dates)):
        raise LPFEdgeInternalPrudenceReportingError(
            "Une journée certifiée est répétée."
        )
    try:
        days = tuple(
            _evaluate_choice(
                select_internal_prudence(
                    item.prediction_day,
                    team_names=item.team_names,
                ),
                item.score,
                item.market_comparison,
            )
            for item in ordered
        )
    except LPFEdgeInternalPrudenceError as error:
        raise LPFEdgeInternalPrudenceReportingError(
            "Une journée ne permet pas le calcul du choix interne."
        ) from error

    statuses = [day.result_status for day in days]
    won = statuses.count(PrudenceResultStatus.WON)
    lost = statuses.count(PrudenceResultStatus.LOST)
    pending = statuses.count(PrudenceResultStatus.PENDING)
    void = statuses.count(PrudenceResultStatus.VOID)
    evaluated = won + lost
    probabilities = [day.choice.model_probability for day in days]
    odds_evaluated = sum(
        day.result_status
        in {PrudenceResultStatus.WON, PrudenceResultStatus.LOST}
        and day.net_result_euros is not None
        for day in days
    )
    missing_odds = sum(
        day.result_status
        in {PrudenceResultStatus.WON, PrudenceResultStatus.LOST}
        and day.net_result_euros is None
        for day in days
    )
    theoretical_net = sum(
        (
            day.net_result_euros
            for day in days
            if day.net_result_euros is not None
        ),
        Decimal("0"),
    )
    current_status, current_count = _current_streak(statuses)
    return InternalPrudenceReport(
        days=days,
        total_count=len(days),
        won_count=won,
        lost_count=lost,
        pending_count=pending,
        void_count=void,
        hit_rate_percent=(
            None
            if evaluated == 0
            else Decimal(won) * HUNDRED / Decimal(evaluated)
        ),
        mean_model_probability=sum(probabilities, Decimal("0"))
        / Decimal(len(probabilities)),
        minimum_model_probability=min(probabilities),
        maximum_model_probability=max(probabilities),
        longest_winning_streak=_longest_streak(
            statuses,
            PrudenceResultStatus.WON,
        ),
        longest_losing_streak=_longest_streak(
            statuses,
            PrudenceResultStatus.LOST,
        ),
        current_streak_status=current_status,
        current_streak_count=current_count,
        odds_evaluated_count=odds_evaluated,
        missing_odds_count=missing_odds,
        theoretical_net_euros=theoretical_net,
        theoretical_roi_percent=(
            None
            if odds_evaluated == 0
            else theoretical_net / Decimal(odds_evaluated) * HUNDRED
        ),
    )


__all__ = [
    "InternalPrudenceDayResult",
    "InternalPrudenceEvidence",
    "InternalPrudenceReport",
    "LPFEdgeInternalPrudenceReportingError",
    "PrudenceResultStatus",
    "build_internal_prudence_report",
]
