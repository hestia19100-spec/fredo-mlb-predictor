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


@dataclass(frozen=True, slots=True)
class InternalPrudenceDayResult:
    choice: InternalPrudenceChoice
    result_status: PrudenceResultStatus
    home_score: int | None
    away_score: int | None

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


def _evaluate_choice(
    choice: InternalPrudenceChoice,
    score: DailyScoreSummary | None,
) -> InternalPrudenceDayResult:
    if score is None:
        return InternalPrudenceDayResult(
            choice=choice,
            result_status=PrudenceResultStatus.PENDING,
            home_score=None,
            away_score=None,
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
    return InternalPrudenceDayResult(
        choice=choice,
        result_status=status,
        home_score=result.home_score,
        away_score=result.away_score,
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
    )


__all__ = [
    "InternalPrudenceDayResult",
    "InternalPrudenceEvidence",
    "InternalPrudenceReport",
    "LPFEdgeInternalPrudenceReportingError",
    "PrudenceResultStatus",
    "build_internal_prudence_report",
]
