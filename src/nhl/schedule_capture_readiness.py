"""Audit hors ligne de l'identité des matchs d'une capture NHL scellée.

Cette projection ne charge pas la base, ne collecte rien et ne produit aucune
caractéristique ni prédiction. Les cotes présentes dans la réponse sont ignorées.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .public_schedule_capture import (
    PublicScheduleCaptureError, USER_COPY_SCHEMA_VERSION, verify_public_schedule_capture,
)


class ScheduleCaptureReadinessError(ValueError):
    """Une identité de match ne peut pas être prouvée par la capture."""


def _team_id(game: dict[str, object], side: str, abbreviation: str) -> int:
    team = game.get(side)
    if not isinstance(team, dict) or team.get("abbrev") != abbreviation:
        raise ScheduleCaptureReadinessError("Équipe ou abréviation NHL incohérente.")
    value = team.get("id")
    if type(value) is not int or value <= 0:
        raise ScheduleCaptureReadinessError("Identifiant d'équipe NHL manquant.")
    return value


def audit_schedule_capture(slot: Path) -> dict[str, object]:
    """Vérifie les IDs des matchs futurs, sans ouvrir le moindre droit modèle."""
    slot = Path(slot)
    try:
        receipt = verify_public_schedule_capture(slot)
        raw = (slot / "response.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt["response_sha256"]:
            raise ScheduleCaptureReadinessError("Réponse modifiée pendant l'audit.")
        payload = json.loads(raw)
        days = [
            day for day in payload["gameWeek"]
            if isinstance(day, dict) and day.get("date") == receipt["target_date"]
        ]
        if len(days) != 1 or not isinstance(days[0].get("games"), list):
            raise ScheduleCaptureReadinessError("Journée NHL ambiguë.")
        by_id: dict[int, dict[str, object]] = {}
        for game in days[0]["games"]:
            if not isinstance(game, dict):
                raise ScheduleCaptureReadinessError("Match NHL invalide.")
            game_id = game.get("id")
            if type(game_id) is int:
                if game_id in by_id:
                    raise ScheduleCaptureReadinessError("Match NHL dupliqué.")
                by_id[game_id] = game

        games: list[dict[str, object]] = []
        for candidate in receipt["future_regular_games"]:
            game = by_id.get(candidate["game_id"])
            if game is None:
                raise ScheduleCaptureReadinessError("Match retenu absent de la réponse.")
            away_id = _team_id(game, "awayTeam", candidate["away_abbr"])
            home_id = _team_id(game, "homeTeam", candidate["home_abbr"])
            if away_id == home_id:
                raise ScheduleCaptureReadinessError("Équipes NHL identiques.")
            games.append({
                "game_id": candidate["game_id"],
                "season": candidate["season"],
                "scheduled_start_utc": candidate["start_utc"],
                "away_team_id": away_id,
                "away_abbr": candidate["away_abbr"],
                "home_team_id": home_id,
                "home_abbr": candidate["home_abbr"],
            })
        return {
            "schema_version": "nhl21_schedule_capture_readiness_v1",
            "status": "SCHEDULE_IDENTITY_ONLY",
            "target_date": receipt["target_date"],
            "acquisition_mode": ("user_supplied_browser_copy"
                                 if receipt["schema_version"] == USER_COPY_SCHEMA_VERSION
                                 else "direct_https"),
            "observed_at_utc": receipt["observed_at_utc"],
            "response_sha256": receipt["response_sha256"],
            "game_count": len(games),
            "games": games,
            "historical_team_history_verified": False,
            "training_permitted": False,
            "prediction_publication_permitted": False,
        }
    except (PublicScheduleCaptureError, OSError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ScheduleCaptureReadinessError):
            raise
        raise ScheduleCaptureReadinessError("Capture NHL impropre à l'audit d'identité.") from error
