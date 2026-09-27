"""Bilan descriptif des choix publiés, avec compléments de report vérifiés."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Mapping, Sequence

from src.lpf_edge_dashboard import DailyScoreSummary, PENDING_STATUSES, VOID_STATUSES
from src.lpf_edge_daily_selection import SealedDailySelection
from src.lpf_edge_postponement_reconciliation import RescheduledGame


class OperationalSelectionError(RuntimeError):
    """Les choix et les résultats ne peuvent pas être rapprochés."""


@dataclass(frozen=True, slots=True)
class OperationalSelectionDay:
    target_date: date
    selection: SealedDailySelection | None
    score: DailyScoreSummary | None
    resolutions: Mapping[int, RescheduledGame]


@dataclass(frozen=True, slots=True)
class OperationalSelectionRow:
    target_date: date
    role: str
    game_id: int
    predicted_team_name: str
    status: str
    net_units: Decimal | None
    resolution_sha256: str | None


@dataclass(frozen=True, slots=True)
class OperationalSelectionReport:
    rows: tuple[OperationalSelectionRow, ...]
    evaluated_count: int
    won_count: int
    lost_count: int
    void_count: int
    supplemented_void_count: int
    pending_count: int
    net_units: Decimal
    roi_percent: Decimal | None


def build_operational_selection(
    evidence: Sequence[OperationalSelectionDay],
) -> OperationalSelectionReport:
    rows: list[OperationalSelectionRow] = []
    seen_dates: set[date] = set()
    for item in sorted(evidence, key=lambda x: x.target_date):
        if item.target_date in seen_dates:
            raise OperationalSelectionError("La journée est répétée.")
        seen_dates.add(item.target_date)
        selection = item.selection
        if selection is None:
            continue
        if selection.target_date != item.target_date:
            raise OperationalSelectionError("La sélection vise une autre date.")
        results = {} if item.score is None else {x.game_id: x for x in item.score.results}
        if item.score is not None and len(results) != len(item.score.results):
            raise OperationalSelectionError("Un résultat est répété.")
        for pick in selection.picks:
            result = results.get(pick.game_id)
            supplement = item.resolutions.get(pick.game_id)
            status = "PENDING"
            net: Decimal | None = None
            proof: str | None = None
            if result is not None:
                if result.predicted_side != pick.predicted_side:
                    raise OperationalSelectionError("Le côté pronostiqué diverge.")
                if result.outcome_status in VOID_STATUSES:
                    status, net = "VOID", Decimal("0")
                elif result.outcome_status in PENDING_STATUSES:
                    if supplement is not None:
                        if (result.outcome_status != "PENDING_POSTPONED"
                                or supplement.original_official_date != item.target_date
                                or supplement.prediction_id != result.prediction_id):
                            raise OperationalSelectionError("Le complément ne correspond pas au report.")
                        status, net = "VOID_RESCHEDULED", Decimal("0")
                        proof = supplement.receipt_sha256
                elif result.outcome_status == "SCORED_FINAL":
                    if result.classification_correct is True:
                        status, net = "WON", pick.best_decimal_odds - Decimal("1")
                    elif result.classification_correct is False:
                        status, net = "LOST", Decimal("-1")
                    else:
                        raise OperationalSelectionError("Une finale n'a pas de verdict.")
                else:
                    raise OperationalSelectionError("Le statut du score est inconnu.")
            if supplement is not None and status != "VOID_RESCHEDULED" and result is not None and result.outcome_status in PENDING_STATUSES:
                raise OperationalSelectionError("Le complément n'est pas appliqué.")
            rows.append(OperationalSelectionRow(
                item.target_date, pick.role, pick.game_id,
                pick.predicted_team_name, status, net, proof,
            ))
    won = sum(x.status == "WON" for x in rows)
    lost = sum(x.status == "LOST" for x in rows)
    void = sum(x.status in {"VOID", "VOID_RESCHEDULED"} for x in rows)
    supplementary = sum(x.status == "VOID_RESCHEDULED" for x in rows)
    pending = sum(x.status == "PENDING" for x in rows)
    evaluated = won + lost
    net = sum((x.net_units for x in rows if x.net_units is not None), Decimal("0"))
    if evaluated + void + pending != len(rows):
        raise OperationalSelectionError("Les compteurs du bilan divergent.")
    return OperationalSelectionReport(
        tuple(rows), evaluated, won, lost, void, supplementary, pending,
        net, None if evaluated == 0 else net / Decimal(evaluated) * Decimal("100"),
    )
