"""Projection hors ligne d'un calendrier NHL déjà capturé.

Ce module ne contacte aucun service. L'appelant conserve la réponse brute et sa
preuve d'heure séparément ; la projection ne prouve pas l'origine de la réponse.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json

MAX_SCHEDULE_BYTES = 8 * 1024 * 1024


class PublicScheduleError(ValueError):
    """Le calendrier ne peut pas être utilisé sans risque de fuite temporelle."""


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    game_id: int
    season: int
    start_utc: datetime
    away_abbr: str
    home_abbr: str


def _start_utc(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PublicScheduleError("Heure de début UTC absente ou invalide.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PublicScheduleError("Heure de début UTC invalide.") from error
    if parsed.utcoffset() != timedelta(0):
        raise PublicScheduleError("Heure de début non UTC.")
    return parsed.astimezone(timezone.utc)


def _abbr(value: object) -> str:
    if not isinstance(value, dict):
        raise PublicScheduleError("Équipe absente.")
    abbr = value.get("abbrev")
    if not isinstance(abbr, str) or not (2 <= len(abbr) <= 4) or not abbr.isalpha() or not abbr.isupper():
        raise PublicScheduleError("Abréviation d'équipe invalide.")
    return abbr


def parse_public_schedule(
    raw_response: bytes, target_date: date, observed_at_utc: datetime
) -> tuple[ScheduledGame, ...]:
    """Retourne uniquement les matchs réguliers futurs, sans score, cote ni effectif.

    L'heure observée doit venir du reçu de capture, pas du contenu NHL.
    Une journée absente ou une ligne admissible malformée bloque le résultat.
    """
    if not isinstance(target_date, date) or isinstance(target_date, datetime):
        raise PublicScheduleError("Date cible invalide.")
    if (not isinstance(observed_at_utc, datetime)
            or observed_at_utc.tzinfo is None
            or observed_at_utc.utcoffset() != timedelta(0)):
        raise PublicScheduleError("Observation UTC requise.")
    if not isinstance(raw_response, bytes) or not 0 < len(raw_response) <= MAX_SCHEDULE_BYTES:
        raise PublicScheduleError("Réponse absente ou trop volumineuse.")
    try:
        payload = json.loads(raw_response)
    except (ValueError, UnicodeDecodeError) as error:
        raise PublicScheduleError("Réponse JSON invalide.") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("gameWeek"), list):
        raise PublicScheduleError("Calendrier NHL invalide.")
    days = [item for item in payload["gameWeek"]
            if isinstance(item, dict) and item.get("date") == target_date.isoformat()]
    if len(days) != 1 or not isinstance(days[0].get("games"), list):
        raise PublicScheduleError("Journée cible absente ou dupliquée.")
    seen: set[int] = set()
    games: list[ScheduledGame] = []
    for item in days[0]["games"]:
        if not isinstance(item, dict):
            raise PublicScheduleError("Ligne de match invalide.")
        if item.get("gameType") != 2 or item.get("gameState") != "FUT":
            continue
        if item.get("gameScheduleState") != "OK":
            continue
        game_id, season = item.get("id"), item.get("season")
        if (type(game_id) is not int or game_id <= 0
                or type(season) is not int or not 20_000_000 <= season <= 20_999_999):
            raise PublicScheduleError("Identifiant ou saison invalide.")
        if game_id in seen:
            raise PublicScheduleError("Match dupliqué.")
        seen.add(game_id)
        start = _start_utc(item.get("startTimeUTC"))
        if start <= observed_at_utc:
            continue
        away, home = _abbr(item.get("awayTeam")), _abbr(item.get("homeTeam"))
        if away == home:
            raise PublicScheduleError("Équipes identiques.")
        games.append(ScheduledGame(game_id, season, start, away, home))
    return tuple(sorted(games, key=lambda game: (game.start_utc, game.game_id)))
