"""Supervision agrégée et en lecture seule de l'évaluation LPF Edge MLB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Iterable

from src.lpf_edge_dashboard import CertifiedPredictionDay, DailyScoreSummary
from src.lpf_edge_daily_selection import SealedDailySelection
from src.lpf_edge_market_settlement import SealedMarketSettlement


HUNDRED = Decimal("100")
ZERO = Decimal("0")
ONE = Decimal("1")
MINIMUM_SHADOW_OBSERVATIONS = 100
MINIMUM_SELECTION_OBSERVATIONS = 100


class LPFEdgeEvaluationSupervisionError(RuntimeError):
    """Les preuves ne permettent pas de construire une supervision fiable."""


@dataclass(frozen=True, slots=True)
class EvaluationDayEvidence:
    """Preuves déjà vérifiées disponibles pour une journée certifiée."""

    prediction_day: CertifiedPredictionDay
    score: DailyScoreSummary | None
    selection: SealedDailySelection | None
    settlement: SealedMarketSettlement | None


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Résultat chronologique d'un choix prospectif."""

    target_date: date
    rank: int
    role: str
    classification_correct: bool
    theoretical_net_units: Decimal


@dataclass(frozen=True, slots=True)
class SelectionPerformance:
    """Bilan d'un ensemble de choix, à mise théorique fixe d'une unité."""

    published_count: int
    evaluated_results: tuple[SelectionResult, ...]
    void_count: int
    pending_count: int

    @property
    def evaluated_count(self) -> int:
        return len(self.evaluated_results)

    @property
    def correct_count(self) -> int:
        return sum(row.classification_correct for row in self.evaluated_results)

    @property
    def incorrect_count(self) -> int:
        return self.evaluated_count - self.correct_count

    @property
    def accuracy_percent(self) -> Decimal | None:
        if not self.evaluated_results:
            return None
        return Decimal(self.correct_count) / Decimal(self.evaluated_count) * HUNDRED

    @property
    def theoretical_net_units(self) -> Decimal:
        return sum(
            (row.theoretical_net_units for row in self.evaluated_results),
            ZERO,
        )

    @property
    def theoretical_roi_percent(self) -> Decimal | None:
        if not self.evaluated_results:
            return None
        return self.theoretical_net_units / Decimal(self.evaluated_count) * HUNDRED

    @property
    def current_losing_streak(self) -> int:
        streak = 0
        for row in reversed(self.evaluated_results):
            if row.classification_correct:
                break
            streak += 1
        return streak

    @property
    def maximum_losing_streak(self) -> int:
        maximum = 0
        current = 0
        for row in self.evaluated_results:
            if row.classification_correct:
                current = 0
            else:
                current += 1
                maximum = max(maximum, current)
        return maximum

    @property
    def maximum_drawdown_units(self) -> Decimal:
        balance = ZERO
        peak = ZERO
        drawdown = ZERO
        for row in self.evaluated_results:
            balance += row.theoretical_net_units
            peak = max(peak, balance)
            drawdown = max(drawdown, peak - balance)
        return drawdown


@dataclass(frozen=True, slots=True)
class EvaluationDayStatus:
    """Ligne synthétique d'une journée dans le tableau de supervision."""

    target_date: date
    prediction_count: int
    scored_count: int
    prediction_status: str
    selection_status: str
    selection_count: int
    selected_evaluated_count: int
    selected_correct_count: int
    selected_theoretical_net_units: Decimal


@dataclass(frozen=True, slots=True)
class EvaluationSupervision:
    """Vue consolidée sans recalcul de modèle ni mutation des preuves."""

    days: tuple[EvaluationDayStatus, ...]
    certified_day_count: int
    completed_day_count: int
    prediction_count: int
    shadow_evaluated_count: int
    shadow_correct_count: int
    shadow_void_count: int
    shadow_pending_count: int
    weighted_log_loss: Decimal | None
    weighted_brier_score: Decimal | None
    selection_publication_day_count: int
    selection_day_count: int
    no_pick_day_count: int
    selection: SelectionPerformance
    principal: SelectionPerformance
    secondary: SelectionPerformance

    @property
    def shadow_accuracy_percent(self) -> Decimal | None:
        if self.shadow_evaluated_count == 0:
            return None
        return (
            Decimal(self.shadow_correct_count)
            / Decimal(self.shadow_evaluated_count)
            * HUNDRED
        )

    @property
    def shadow_progress_percent(self) -> Decimal:
        return min(
            HUNDRED,
            Decimal(self.shadow_evaluated_count)
            / Decimal(MINIMUM_SHADOW_OBSERVATIONS)
            * HUNDRED,
        )

    @property
    def selection_progress_percent(self) -> Decimal:
        return min(
            HUNDRED,
            Decimal(self.selection.evaluated_count)
            / Decimal(MINIMUM_SELECTION_OBSERVATIONS)
            * HUNDRED,
        )

    @property
    def first_date(self) -> date | None:
        return self.days[0].target_date if self.days else None

    @property
    def last_date(self) -> date | None:
        return self.days[-1].target_date if self.days else None


def _decimal_from_float(value: float, description: str) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise LPFEdgeEvaluationSupervisionError(f"{description} est invalide.")
    return result


def _performance(
    *,
    published_count: int,
    results: Iterable[SelectionResult],
    void_count: int,
    pending_count: int,
) -> SelectionPerformance:
    ordered = tuple(sorted(results, key=lambda row: (row.target_date, row.rank)))
    if published_count != len(ordered) + void_count + pending_count:
        raise LPFEdgeEvaluationSupervisionError(
            "Le bilan des sélections ne couvre pas tous les choix publiés."
        )
    return SelectionPerformance(
        published_count=published_count,
        evaluated_results=ordered,
        void_count=void_count,
        pending_count=pending_count,
    )


def build_evaluation_supervision(
    evidence_days: Iterable[EvaluationDayEvidence],
) -> EvaluationSupervision:
    """Agrège uniquement des prédictions, sélections et verdicts déjà vérifiés."""
    evidence = tuple(
        sorted(evidence_days, key=lambda item: item.prediction_day.target_date)
    )
    dates = [item.prediction_day.target_date for item in evidence]
    if len(dates) != len(set(dates)):
        raise LPFEdgeEvaluationSupervisionError(
            "Une journée certifiée est répétée dans la supervision."
        )

    completed_day_count = 0
    prediction_count = 0
    shadow_evaluated = 0
    shadow_correct = 0
    shadow_void = 0
    shadow_pending = 0
    log_loss_total = ZERO
    log_loss_weight = 0
    brier_total = ZERO
    brier_weight = 0
    selection_publication_days = 0
    selection_days = 0
    no_pick_days = 0
    published_by_role = {"PRINCIPAL": 0, "SECONDAIRE": 0}
    void_by_role = {"PRINCIPAL": 0, "SECONDAIRE": 0}
    pending_by_role = {"PRINCIPAL": 0, "SECONDAIRE": 0}
    results_by_role: dict[str, list[SelectionResult]] = {
        "PRINCIPAL": [],
        "SECONDAIRE": [],
    }
    day_statuses: list[EvaluationDayStatus] = []

    for item in evidence:
        day = item.prediction_day
        target_date = day.target_date
        predictions = day.predictions
        if not predictions or any(row.official_date != target_date for row in predictions):
            raise LPFEdgeEvaluationSupervisionError(
                "Une journée certifiée est vide ou vise une autre date."
            )
        prediction_count += len(predictions)
        score = item.score
        scored_count = 0
        prediction_status = "EN_ATTENTE"
        if score is not None:
            if (
                len(score.results) != len(predictions)
                or score.scored_count + score.void_count + score.pending_count
                != len(predictions)
                or score.correct_count + score.incorrect_count
                != score.scored_count
            ):
                raise LPFEdgeEvaluationSupervisionError(
                    "Les résultats ne couvrent pas toutes les prédictions certifiées."
                )
            scored_count = score.scored_count
            shadow_evaluated += score.scored_count
            shadow_correct += score.correct_count
            shadow_void += score.void_count
            shadow_pending += score.pending_count
            if score.pending_count == 0:
                completed_day_count += 1
                prediction_status = "TERMINEE"
            else:
                prediction_status = "PARTIELLE"
            if score.mean_log_loss is not None and score.scored_count:
                log_loss_total += (
                    _decimal_from_float(score.mean_log_loss, "Le log loss")
                    * Decimal(score.scored_count)
                )
                log_loss_weight += score.scored_count
            if score.mean_brier_score is not None and score.scored_count:
                brier_total += (
                    _decimal_from_float(score.mean_brier_score, "Le score de Brier")
                    * Decimal(score.scored_count)
                )
                brier_weight += score.scored_count
        else:
            shadow_pending += len(predictions)

        selection = item.selection
        settlement = item.settlement
        selected_evaluated = 0
        selected_correct = 0
        selected_net = ZERO
        selection_status = "NON_DISPONIBLE"
        if selection is None:
            if settlement is not None and settlement.daily_selection_sha256 is not None:
                raise LPFEdgeEvaluationSupervisionError(
                    "Un verdict référence une sélection absente."
                )
        else:
            if selection.target_date != target_date:
                raise LPFEdgeEvaluationSupervisionError(
                    "Une sélection vise une autre journée."
                )
            selection_publication_days += 1
            if selection.selection_count == 0:
                no_pick_days += 1
                selection_status = "AUCUN_PRONO"
            else:
                selection_days += 1
                selection_status = "EN_ATTENTE"
            for pick in selection.picks:
                if pick.role not in published_by_role:
                    raise LPFEdgeEvaluationSupervisionError(
                        "Un rôle de sélection est inconnu."
                    )
                published_by_role[pick.role] += 1

            if settlement is None:
                for pick in selection.picks:
                    pending_by_role[pick.role] += 1
            else:
                if (
                    settlement.target_date != target_date
                    or settlement.daily_selection_sha256 != selection.selection_sha256
                    or settlement.selected_count != selection.selection_count
                    or score is None
                ):
                    raise LPFEdgeEvaluationSupervisionError(
                        "La sélection et son verdict ne concordent pas."
                    )
                results_by_game = {row.game_id: row for row in score.results}
                if len(results_by_game) != len(score.results):
                    raise LPFEdgeEvaluationSupervisionError(
                        "Un résultat est répété dans une journée."
                    )
                for pick in selection.picks:
                    result = results_by_game.get(pick.game_id)
                    if result is None:
                        raise LPFEdgeEvaluationSupervisionError(
                            "Un match sélectionné est absent des résultats."
                        )
                    if result.classification_correct is None:
                        void_by_role[pick.role] += 1
                        continue
                    net = pick.best_decimal_odds - ONE if result.classification_correct else -ONE
                    selection_result = SelectionResult(
                        target_date=target_date,
                        rank=pick.rank,
                        role=pick.role,
                        classification_correct=result.classification_correct,
                        theoretical_net_units=net,
                    )
                    results_by_role[pick.role].append(selection_result)
                    selected_evaluated += 1
                    selected_correct += result.classification_correct
                    selected_net += net
                if (
                    settlement.selected_evaluated_count != selected_evaluated
                    or settlement.selected_correct_count != selected_correct
                    or settlement.selected_void_count
                    != sum(
                        result.classification_correct is None
                        for pick in selection.picks
                        for result in (results_by_game[pick.game_id],)
                    )
                    or settlement.selected_theoretical_net_units != selected_net
                    or settlement.selected_theoretical_roi_percent
                    != (
                        None
                        if selected_evaluated == 0
                        else selected_net / Decimal(selected_evaluated) * HUNDRED
                    )
                ):
                    raise LPFEdgeEvaluationSupervisionError(
                        "Le bilan détaillé diverge du verdict scellé."
                    )
                selection_status = "TERMINEE" if selection.picks else "AUCUN_PRONO"

        day_statuses.append(
            EvaluationDayStatus(
                target_date=target_date,
                prediction_count=len(predictions),
                scored_count=scored_count,
                prediction_status=prediction_status,
                selection_status=selection_status,
                selection_count=0 if selection is None else selection.selection_count,
                selected_evaluated_count=selected_evaluated,
                selected_correct_count=selected_correct,
                selected_theoretical_net_units=selected_net,
            )
        )

    principal = _performance(
        published_count=published_by_role["PRINCIPAL"],
        results=results_by_role["PRINCIPAL"],
        void_count=void_by_role["PRINCIPAL"],
        pending_count=pending_by_role["PRINCIPAL"],
    )
    secondary = _performance(
        published_count=published_by_role["SECONDAIRE"],
        results=results_by_role["SECONDAIRE"],
        void_count=void_by_role["SECONDAIRE"],
        pending_count=pending_by_role["SECONDAIRE"],
    )
    combined_results = principal.evaluated_results + secondary.evaluated_results
    selection_performance = _performance(
        published_count=principal.published_count + secondary.published_count,
        results=combined_results,
        void_count=principal.void_count + secondary.void_count,
        pending_count=principal.pending_count + secondary.pending_count,
    )
    return EvaluationSupervision(
        days=tuple(day_statuses),
        certified_day_count=len(evidence),
        completed_day_count=completed_day_count,
        prediction_count=prediction_count,
        shadow_evaluated_count=shadow_evaluated,
        shadow_correct_count=shadow_correct,
        shadow_void_count=shadow_void,
        shadow_pending_count=shadow_pending,
        weighted_log_loss=(
            None if log_loss_weight == 0 else log_loss_total / Decimal(log_loss_weight)
        ),
        weighted_brier_score=(
            None if brier_weight == 0 else brier_total / Decimal(brier_weight)
        ),
        selection_publication_day_count=selection_publication_days,
        selection_day_count=selection_days,
        no_pick_day_count=no_pick_days,
        selection=selection_performance,
        principal=principal,
        secondary=secondary,
    )


__all__ = [
    "EvaluationDayEvidence",
    "EvaluationDayStatus",
    "EvaluationSupervision",
    "LPFEdgeEvaluationSupervisionError",
    "MINIMUM_SELECTION_OBSERVATIONS",
    "MINIMUM_SHADOW_OBSERVATIONS",
    "SelectionPerformance",
    "SelectionResult",
    "build_evaluation_supervision",
]
