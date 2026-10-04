"""Import hors ligne des téléchargements d'équipes MoneyPuck publiés.

Un fichier téléchargé aujourd'hui ne prouve jamais qu'une ligne était disponible
avant un match passé. Ce module ne fournit donc aucune variable ni label au modèle.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from io import StringIO
from pathlib import Path
import re

from .contracts import require_utc

SOURCE_PAGE = "https://moneypuck.com/data.htm"
TEAM_GAME_DOWNLOAD = (
    "https://moneypuck.com/moneypuck/playerData/careers/"
    "gameByGame/all_teams.csv"
)
ATTRIBUTION = "Données : MoneyPuck.com"
MAX_FILE_BYTES = 256 * 1024 * 1024
SITUATIONS = frozenset({"all", "5on5"})
REQUIRED_COLUMNS = frozenset({
    "team", "season", "name", "gameId", "playerTeam", "opposingTeam",
    "home_or_away", "gameDate", "position", "situation", "xGoalsFor",
    "xGoalsAgainst", "goalsFor", "goalsAgainst", "shotsOnGoalFor",
    "shotsOnGoalAgainst",
})
TEAM_CODE = re.compile(r"^(?:[A-Z]{3}|[A-Z]\.[A-Z])$")


class NHLMoneyPuckImportError(ValueError):
    """Fichier incompatible, ambigu ou altéré."""


def _integer(value: str | None, field: str) -> int:
    try:
        result = int(value or "")
    except ValueError as error:
        raise NHLMoneyPuckImportError(f"{field} doit être entier.") from error
    if result < 0:
        raise NHLMoneyPuckImportError(f"{field} doit être positif ou nul.")
    return result


def _decimal(value: str | None, field: str) -> Decimal:
    try:
        result = Decimal(value or "")
    except InvalidOperation as error:
        raise NHLMoneyPuckImportError(f"{field} doit être numérique.") from error
    if not result.is_finite() or result < 0:
        raise NHLMoneyPuckImportError(f"{field} doit être fini et positif ou nul.")
    return result


def _goals(value: str | None, field: str) -> int:
    number = _decimal(value, field)
    if number != number.to_integral_value():
        raise NHLMoneyPuckImportError(f"{field} doit être un nombre de buts entier.")
    return int(number)


@dataclass(frozen=True, slots=True)
class TeamGameRow:
    game_id: int
    season: int
    game_date: date
    team: str
    opponent: str
    home_or_away: str
    situation: str
    x_goals_for: Decimal
    x_goals_against: Decimal
    goals_for: int
    goals_against: int
    shots_on_goal_for: Decimal
    shots_on_goal_against: Decimal


@dataclass(frozen=True, slots=True)
class MoneyPuckTeamSnapshot:
    observed_at_utc: datetime
    file_sha256: str
    source_page: str
    download_url: str
    attribution: str
    rows: tuple[TeamGameRow, ...]
    training_permitted: bool = False

    def available_by(self, information_cutoff_utc: datetime) -> bool:
        """Vérifie uniquement la réception, pas la finalité des matchs sources."""
        require_utc(information_cutoff_utc, field_name="information_cutoff_utc")
        return self.observed_at_utc <= information_cutoff_utc


def _row(record: dict[str, str | None], line: int) -> TeamGameRow:
    for field in ("team", "playerTeam", "name", "opposingTeam"):
        if not TEAM_CODE.fullmatch(record.get(field) or ""):
            raise NHLMoneyPuckImportError(f"Ligne {line}: code {field} invalide.")
    team = record["team"]
    if record["playerTeam"] != team or record["name"] != team:
        raise NHLMoneyPuckImportError(f"Ligne {line}: identité d'équipe incohérente.")
    if record["opposingTeam"] == team:
        raise NHLMoneyPuckImportError(f"Ligne {line}: adversaire identique.")
    side = record.get("home_or_away")
    if side not in ("HOME", "AWAY"):
        raise NHLMoneyPuckImportError(f"Ligne {line}: lieu invalide.")
    try:
        game_date = datetime.strptime(record.get("gameDate") or "", "%Y%m%d").date()
    except ValueError as error:
        raise NHLMoneyPuckImportError(f"Ligne {line}: date invalide.") from error
    game_id = _integer(record.get("gameId"), "gameId")
    season = _integer(record.get("season"), "season")
    if game_id == 0 or season < 2007 or game_id // 1_000_000 != season:
        raise NHLMoneyPuckImportError(f"Ligne {line}: match/saison incohérents.")
    return TeamGameRow(
        game_id=game_id, season=season, game_date=game_date,
        team=team or "", opponent=record["opposingTeam"] or "",
        home_or_away=side, situation=record["situation"] or "",
        x_goals_for=_decimal(record.get("xGoalsFor"), "xGoalsFor"),
        x_goals_against=_decimal(record.get("xGoalsAgainst"), "xGoalsAgainst"),
        goals_for=_goals(record.get("goalsFor"), "goalsFor"),
        goals_against=_goals(record.get("goalsAgainst"), "goalsAgainst"),
        shots_on_goal_for=_decimal(record.get("shotsOnGoalFor"), "shotsOnGoalFor"),
        shots_on_goal_against=_decimal(record.get("shotsOnGoalAgainst"), "shotsOnGoalAgainst"),
    )


def import_team_games(path: Path, *, season: int | None = None,
                      seasons: tuple[int, ...] | None = None) -> MoneyPuckTeamSnapshot:
    """Lit un fichier local; aucune requête réseau, écriture ni rétrodatation."""
    if season is not None and (type(season) is not int or season < 2007):
        raise NHLMoneyPuckImportError("La saison filtrée est invalide.")
    if seasons is not None and (
        season is not None or type(seasons) is not tuple or not seasons
        or any(type(value) is not int or value < 2007 for value in seasons)
        or len(seasons) != len(set(seasons))
    ):
        raise NHLMoneyPuckImportError("Les saisons filtrées sont invalides.")
    allowed_seasons = {str(value) for value in seasons} if seasons is not None else None
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_FILE_BYTES:
            raise NHLMoneyPuckImportError("Taille de fichier interdite.")
        raw = path.read_bytes()
        if len(raw) != size or path.stat().st_size != size:
            raise NHLMoneyPuckImportError("Le fichier a changé pendant la lecture.")
        decoded = raw.decode("utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise NHLMoneyPuckImportError("Fichier CSV local illisible.") from error
    reader = csv.DictReader(StringIO(decoded, newline=""), strict=True)
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise NHLMoneyPuckImportError("En-tête CSV absent ou dupliqué.")
    if not REQUIRED_COLUMNS <= set(reader.fieldnames):
        raise NHLMoneyPuckImportError("Colonnes MoneyPuck requises absentes.")
    seen: set[tuple[int, str, str]] = set()
    rows: list[TeamGameRow] = []
    try:
        for line, record in enumerate(reader, start=2):
            if None in record or record.get("position") != "Team Level":
                raise NHLMoneyPuckImportError(f"Ligne {line}: structure inattendue.")
            if record.get("situation") not in SITUATIONS:
                continue
            if season is not None and record.get("season") != str(season):
                continue
            if allowed_seasons is not None and record.get("season") not in allowed_seasons:
                continue
            parsed = _row(record, line)
            key = (parsed.game_id, parsed.team, parsed.situation)
            if key in seen:
                raise NHLMoneyPuckImportError(f"Ligne {line}: match d'équipe dupliqué.")
            seen.add(key)
            rows.append(parsed)
    except csv.Error as error:
        raise NHLMoneyPuckImportError("CSV MoneyPuck mal formé.") from error
    if not rows:
        raise NHLMoneyPuckImportError("Aucune ligne d'équipe pour la saison.")
    rows.sort(key=lambda item: (item.game_id, item.team, item.situation))
    observed = datetime.now(timezone.utc)
    return MoneyPuckTeamSnapshot(
        observed_at_utc=observed, file_sha256=sha256(raw).hexdigest(),
        source_page=SOURCE_PAGE, download_url=TEAM_GAME_DOWNLOAD,
        attribution=ATTRIBUTION, rows=tuple(rows),
    )
