"""Choix privé de la probabilité LPF maximale d'une journée certifiée.

Le module est volontairement pur : il ne lit ni score, ni réseau, ni modèle,
ni cote. Il reçoit uniquement un lot de prédictions déjà certifié.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Mapping

from src.lpf_edge_dashboard import CertifiedPredictionDay


STRATEGY_ID = "lpf_edge_mlb_internal_prudence_v1"
VISIBILITY = "INTERNAL_ONLY"


class LPFEdgeInternalPrudenceError(ValueError):
    """Le choix interne ne peut pas être déterminé sans ambiguïté."""


@dataclass(frozen=True, slots=True)
class InternalPrudenceChoice:
    strategy_id: str
    visibility: str
    public_exposure_allowed: bool
    target_date: date
    source_batch_id: str
    source_predictions_sha256: str
    prediction_id: str
    game_id: int
    scheduled_start_utc: datetime
    home_team_name: str
    away_team_name: str
    predicted_side: str
    predicted_team_name: str
    model_probability: Decimal
    home_probability: Decimal
    away_probability: Decimal
    rank_by_probability: int
    tie_break_rule_applied: bool


def select_internal_prudence(
    prediction_day: CertifiedPredictionDay,
    *,
    team_names: Mapping[int, str],
) -> InternalPrudenceChoice:
    """Sélectionne le maximum LPF sans jamais consulter les résultats."""

    if not isinstance(prediction_day, CertifiedPredictionDay):
        raise LPFEdgeInternalPrudenceError(
            "prediction_day doit être un lot certifié."
        )
    if not prediction_day.predictions:
        raise LPFEdgeInternalPrudenceError(
            "Le lot certifié ne contient aucune prédiction."
        )
    if not isinstance(team_names, Mapping):
        raise LPFEdgeInternalPrudenceError("Les noms d'équipes sont invalides.")

    decorated = []
    for prediction in prediction_day.predictions:
        home_name = team_names.get(prediction.home_team_id)
        away_name = team_names.get(prediction.away_team_id)
        if (
            type(home_name) is not str
            or not home_name.strip()
            or type(away_name) is not str
            or not away_name.strip()
        ):
            raise LPFEdgeInternalPrudenceError(
                "Une équipe du lot certifié n'a pas de nom vérifiable."
            )
        predicted_name = (
            home_name if prediction.predicted_side == "HOME" else away_name
        )
        decorated.append((prediction, home_name, away_name, predicted_name))

    maximum = max(item[0].predicted_probability for item in decorated)
    tied_count = sum(
        item[0].predicted_probability == maximum for item in decorated
    )
    decorated.sort(
        key=lambda item: (
            -item[0].predicted_probability,
            item[0].game_id,
            item[3].casefold(),
            item[3],
        )
    )
    selected, home_name, away_name, predicted_name = decorated[0]
    return InternalPrudenceChoice(
        strategy_id=STRATEGY_ID,
        visibility=VISIBILITY,
        public_exposure_allowed=False,
        target_date=prediction_day.target_date,
        source_batch_id=prediction_day.batch_id,
        source_predictions_sha256=prediction_day.predictions_sha256,
        prediction_id=selected.prediction_id,
        game_id=selected.game_id,
        scheduled_start_utc=selected.scheduled_start_utc,
        home_team_name=home_name,
        away_team_name=away_name,
        predicted_side=selected.predicted_side,
        predicted_team_name=predicted_name,
        model_probability=selected.predicted_probability,
        home_probability=selected.p_home_win,
        away_probability=selected.p_away_win,
        rank_by_probability=1,
        tie_break_rule_applied=tied_count > 1,
    )


__all__ = [
    "InternalPrudenceChoice",
    "LPFEdgeInternalPrudenceError",
    "STRATEGY_ID",
    "VISIBILITY",
    "select_internal_prudence",
]
