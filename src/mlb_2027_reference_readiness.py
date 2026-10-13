"""Prépare les candidats MLB 2027 pour le modèle de référence, sans prédire.

Les entrées doivent provenir d'un snapshot MLB vérifié par l'orchestrateur.
Cette fonction ne certifie ni la provenance, ni une prédiction, ni une cote.
Le modèle Shadow V2 2026 et ses preuves restent inchangés.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import Literal

from src.mlb_2027_timing import LeadWindow, build_lead_window

REFERENCE_MODEL_VERSION = "logistic_team_form_v1_platt"
REFERENCE_ARTIFACT_SHA256 = (
    "e0d4d2421ba076072c7ef8b3bc97dd9a"
    "341e26c62828a0ad9ba43f30da15ff55"
)
MINIMUM_SAME_SEASON_FINALS_PER_TEAM = 10
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MLB2027ReadinessError(ValueError):
    """Les données de préparation ne permettent pas une décision sûre."""


@dataclass(frozen=True, slots=True)
class UpcomingGame:
    game_id: int
    official_date: date
    game_type: Literal["R"]
    away_team_id: int
    home_team_id: int
    scheduled_start_utc: datetime
    status: Literal["SCHEDULED", "PRE_GAME"]


@dataclass(frozen=True, slots=True)
class PriorFinalGame:
    game_id: int
    official_date: date
    game_type: Literal["R"]
    away_team_id: int
    home_team_id: int
    away_score: int
    home_score: int
    status: Literal["FINAL"]


@dataclass(frozen=True, slots=True)
class GameReadiness:
    game_id: int
    status: Literal["REFERENCE_ELIGIBLE", "ABSTAIN"]
    reason: str
    away_same_season_finals: int
    home_same_season_finals: int
    model_version: str | None
    artifact_sha256: str | None


@dataclass(frozen=True, slots=True)
class DayReadiness:
    target_date: date
    source_receipt_sha256: str
    lead_window: LeadWindow
    games: tuple[GameReadiness, ...]

    @property
    def eligible_count(self) -> int:
        return sum(game.status == "REFERENCE_ELIGIBLE" for game in self.games)


def _positive_id(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise MLB2027ReadinessError(f"{name} must be a positive integer.")
    return value


def _teams(away: object, home: object) -> tuple[int, int]:
    away_id = _positive_id(away, "away_team_id")
    home_id = _positive_id(home, "home_team_id")
    if away_id == home_id:
        raise MLB2027ReadinessError("A team cannot play itself.")
    return away_id, home_id


def plan_reference_readiness(
    target_date: date,
    upcoming_games: tuple[UpcomingGame, ...],
    prior_final_games: tuple[PriorFinalGame, ...],
    *,
    source_receipt_sha256: str,
) -> DayReadiness:
    """Utilise V2 si chaque équipe a dix finales 2027 J-1 ; sinon s'abstient.

    Une future promotion de challenger exige un autre protocole enregistré.
    Aucun changement de moteur n'est demandé à l'utilisateur.
    """
    if type(target_date) is not date or target_date.year != 2027:
        raise MLB2027ReadinessError("Only official-date 2027 is in scope.")
    if not isinstance(source_receipt_sha256, str) or not _SHA256.fullmatch(
        source_receipt_sha256
    ):
        raise MLB2027ReadinessError("A source receipt SHA-256 is required.")
    if type(upcoming_games) is not tuple or not upcoming_games:
        raise MLB2027ReadinessError("A non-empty tuple of upcoming games is required.")
    if type(prior_final_games) is not tuple:
        raise MLB2027ReadinessError("Prior finals must be a tuple.")

    counts: Counter[int] = Counter()
    seen_ids: set[int] = set()
    for game in prior_final_games:
        if type(game) is not PriorFinalGame:
            raise MLB2027ReadinessError("A prior game is not a PriorFinalGame.")
        game_id = _positive_id(game.game_id, "prior game_id")
        away, home = _teams(game.away_team_id, game.home_team_id)
        if (
            type(game.official_date) is not date
            or game.official_date.year != 2027
            or game.official_date >= target_date
            or game.game_type != "R"
            or game.status != "FINAL"
            or type(game.away_score) is not int
            or type(game.home_score) is not int
            or min(game.away_score, game.home_score) < 0
            or game.away_score == game.home_score
            or game_id in seen_ids
        ):
            raise MLB2027ReadinessError("Prior finals must be unique 2027 J-1 decisions.")
        seen_ids.add(game_id)
        counts[away] += 1
        counts[home] += 1

    for game in upcoming_games:
        if type(game) is not UpcomingGame:
            raise MLB2027ReadinessError("A target game is not an UpcomingGame.")
        game_id = _positive_id(game.game_id, "target game_id")
        _teams(game.away_team_id, game.home_team_id)
        if (
            type(game.official_date) is not date
            or game.official_date != target_date
            or game.game_type != "R"
            or game.status not in {"SCHEDULED", "PRE_GAME"}
            or game_id in seen_ids
        ):
            raise MLB2027ReadinessError("Target games must be unique 2027 regular-season pregames.")
        seen_ids.add(game_id)

    window = build_lead_window(
        target_date, (game.scheduled_start_utc for game in upcoming_games)
    )
    ordered = sorted(upcoming_games, key=lambda game: (
        game.scheduled_start_utc, game.game_id
    ))
    decisions: list[GameReadiness] = []
    for game in ordered:
        away_count = counts[game.away_team_id]
        home_count = counts[game.home_team_id]
        away_short = away_count < MINIMUM_SAME_SEASON_FINALS_PER_TEAM
        home_short = home_count < MINIMUM_SAME_SEASON_FINALS_PER_TEAM
        if away_short and home_short:
            reason = "INSUFFICIENT_BOTH_HISTORY"
        elif away_short:
            reason = "INSUFFICIENT_AWAY_HISTORY"
        elif home_short:
            reason = "INSUFFICIENT_HOME_HISTORY"
        else:
            reason = "SAME_SEASON_HISTORY_AVAILABLE"
        eligible = not (away_short or home_short)
        decisions.append(GameReadiness(
            game_id=game.game_id,
            status="REFERENCE_ELIGIBLE" if eligible else "ABSTAIN",
            reason=reason,
            away_same_season_finals=away_count,
            home_same_season_finals=home_count,
            model_version=REFERENCE_MODEL_VERSION if eligible else None,
            artifact_sha256=REFERENCE_ARTIFACT_SHA256 if eligible else None,
        ))
    return DayReadiness(
        target_date=target_date,
        source_receipt_sha256=source_receipt_sha256,
        lead_window=window,
        games=tuple(decisions),
    )
