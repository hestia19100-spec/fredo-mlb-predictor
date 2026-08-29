"""Client contrôlé pour récupérer le calendrier depuis l’API MLB."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
import json
from typing import Any

import requests


MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_SPORT_ID = 1
REQUEST_TIMEOUT_SECONDS = 30
MAX_SCHEDULE_RANGE_DAYS = 31
DISPLAY_LIMIT = 20

FINAL_STATUS_CODES = frozenset({"F"})
FINAL_STATUS_DETAILS = frozenset(
    {
        "FINAL",
        "GAME OVER",
        "COMPLETED EARLY",
    }
)
POSTPONED_STATUS_DETAILS = frozenset({"POSTPONED"})


class MLBAPIError(RuntimeError):
    """Erreur compréhensible liée à l’API MLB."""


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    """Représentation contrôlée d’un match reçu depuis MLB."""

    game_id: int
    season: int
    official_date: str
    game_datetime_utc: str
    game_type: str
    status_code: str
    status_detail: str
    away_team_id: int
    away_team_name: str
    home_team_id: int
    home_team_name: str
    away_score: int | None
    home_score: int | None
    venue_id: int | None
    venue_name: str | None
    doubleheader: str | None
    game_number: int | None
    away_probable_pitcher_id: int | None
    away_probable_pitcher_name: str | None
    home_probable_pitcher_id: int | None
    home_probable_pitcher_name: str | None


@dataclass(frozen=True, slots=True)
class ScheduleFetchResult:
    """Résultat auditable d’une récupération par période."""

    start_date: date
    end_date: date
    game_types: tuple[str, ...]
    request_parameters: dict[str, object]
    raw_content: bytes
    games: tuple[ScheduledGame, ...]


def _optional_int(value: Any) -> int | None:
    """Convertit une valeur facultative en nombre entier."""
    if value is None:
        return None

    return int(value)


def _parse_probable_pitcher(
    team_data: dict[str, Any],
) -> tuple[int | None, str | None]:
    """Extrait le lanceur probable lorsqu’il est annoncé."""
    probable_pitcher = team_data.get("probablePitcher")

    if not isinstance(probable_pitcher, dict):
        return None, None

    pitcher_id = _optional_int(probable_pitcher.get("id"))
    pitcher_name = (
        str(probable_pitcher["fullName"])
        if probable_pitcher.get("fullName") is not None
        else None
    )

    return pitcher_id, pitcher_name


def _parse_game(raw_game: dict[str, Any]) -> ScheduledGame:
    """Transforme un match MLB brut en structure contrôlée."""
    try:
        away_data = raw_game["teams"]["away"]
        home_data = raw_game["teams"]["home"]
        away_team = away_data["team"]
        home_team = home_data["team"]
        status = raw_game["status"]
        venue = raw_game.get("venue") or {}

        (
            away_probable_pitcher_id,
            away_probable_pitcher_name,
        ) = _parse_probable_pitcher(away_data)

        (
            home_probable_pitcher_id,
            home_probable_pitcher_name,
        ) = _parse_probable_pitcher(home_data)

        return ScheduledGame(
            game_id=int(raw_game["gamePk"]),
            season=int(raw_game["season"]),
            official_date=str(raw_game["officialDate"]),
            game_datetime_utc=str(raw_game["gameDate"]),
            game_type=str(raw_game["gameType"]),
            status_code=str(
                status.get("statusCode")
                or status.get("codedGameState")
                or "UNKNOWN"
            ),
            status_detail=str(status["detailedState"]),
            away_team_id=int(away_team["id"]),
            away_team_name=str(away_team["name"]),
            home_team_id=int(home_team["id"]),
            home_team_name=str(home_team["name"]),
            away_score=_optional_int(away_data.get("score")),
            home_score=_optional_int(home_data.get("score")),
            venue_id=_optional_int(venue.get("id")),
            venue_name=(
                str(venue["name"])
                if venue.get("name") is not None
                else None
            ),
            doubleheader=(
                str(raw_game["doubleHeader"])
                if raw_game.get("doubleHeader") is not None
                else None
            ),
            game_number=_optional_int(raw_game.get("gameNumber")),
            away_probable_pitcher_id=away_probable_pitcher_id,
            away_probable_pitcher_name=away_probable_pitcher_name,
            home_probable_pitcher_id=home_probable_pitcher_id,
            home_probable_pitcher_name=home_probable_pitcher_name,
        )
    except (KeyError, TypeError, ValueError) as error:
        game_reference = raw_game.get("gamePk", "inconnu")
        raise MLBAPIError(
            f"Réponse MLB incomplète pour le match {game_reference}."
        ) from error


def _game_identity(
    game: ScheduledGame,
) -> tuple[int, str, int, int]:
    """Retourne les éléments qui ne doivent jamais changer."""
    return (
        game.season,
        game.game_type,
        game.away_team_id,
        game.home_team_id,
    )


def _is_final_game(game: ScheduledGame) -> bool:
    """Indique si une occurrence décrit un match terminé."""
    status_code = game.status_code.strip().upper()
    status_detail = game.status_detail.strip().upper()

    return (
        status_code in FINAL_STATUS_CODES
        or status_detail in FINAL_STATUS_DETAILS
    )


def _is_postponed_game(game: ScheduledGame) -> bool:
    """Indique si une occurrence décrit un match reporté."""
    status_detail = game.status_detail.strip().upper()
    return status_detail in POSTPONED_STATUS_DETAILS


def _select_canonical_game(
    existing_game: ScheduledGame,
    candidate_game: ScheduledGame,
) -> ScheduledGame:
    """Fusionne deux occurrences cohérentes d’un même match."""
    if _game_identity(existing_game) != _game_identity(candidate_game):
        raise MLBAPIError(
            "Occurrences contradictoires pour le match MLB "
            f"{candidate_game.game_id}."
        )

    if existing_game == candidate_game:
        return existing_game

    existing_is_final = _is_final_game(existing_game)
    candidate_is_final = _is_final_game(candidate_game)

    if existing_is_final != candidate_is_final:
        if candidate_is_final:
            return candidate_game

        return existing_game

    existing_is_postponed = _is_postponed_game(existing_game)
    candidate_is_postponed = _is_postponed_game(candidate_game)

    if existing_is_postponed != candidate_is_postponed:
        if candidate_is_postponed:
            return existing_game

        return candidate_game

    raise MLBAPIError(
        "Occurrences contradictoires pour le match MLB "
        f"{candidate_game.game_id}."
    )


def _parse_schedule_payload(
    payload: dict[str, Any],
) -> list[ScheduledGame]:
    """Transforme et contrôle tous les matchs d’une réponse."""
    date_blocks = payload.get("dates", [])
    if not isinstance(date_blocks, list):
        raise MLBAPIError("Liste des dates absente de la réponse MLB.")

    game_occurrences: list[ScheduledGame] = []

    for date_block in date_blocks:
        if not isinstance(date_block, dict):
            raise MLBAPIError("Bloc de date MLB invalide.")

        raw_games = date_block.get("games", [])
        if not isinstance(raw_games, list):
            raise MLBAPIError("Liste des matchs MLB invalide.")

        for raw_game in raw_games:
            if not isinstance(raw_game, dict):
                raise MLBAPIError("Match MLB invalide.")

            game_occurrences.append(_parse_game(raw_game))

    reported_total = payload.get("totalGames")
    if reported_total is not None:
        try:
            expected_total = int(reported_total)
        except (TypeError, ValueError) as error:
            raise MLBAPIError(
                "Le total de matchs MLB est invalide."
            ) from error

        if expected_total != len(game_occurrences):
            raise MLBAPIError(
                "La réponse MLB semble partielle : "
                f"{len(game_occurrences)} matchs lus sur "
                f"{expected_total} annoncés."
            )

    canonical_games: dict[int, ScheduledGame] = {}

    for game in game_occurrences:
        existing_game = canonical_games.get(game.game_id)

        if existing_game is None:
            canonical_games[game.game_id] = game
            continue

        canonical_games[game.game_id] = _select_canonical_game(
            existing_game,
            game,
        )

    return list(canonical_games.values())


def _request_schedule(
    parameters: dict[str, object],
) -> tuple[bytes, dict[str, Any]]:
    """Télécharge puis décode une réponse MLB unique."""
    headers = {
        "User-Agent": "fredo-mlb-predictor/0.1",
    }

    try:
        response = requests.get(
            MLB_SCHEDULE_URL,
            params=parameters,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise MLBAPIError(
            f"Impossible de joindre l’API MLB : {error}"
        ) from error

    raw_content = response.content

    if not isinstance(raw_content, bytes) or not raw_content:
        raise MLBAPIError("L’API MLB a renvoyé une réponse vide.")

    try:
        payload = json.loads(raw_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MLBAPIError(
            "L’API MLB n’a pas renvoyé un document JSON valide."
        ) from error

    if not isinstance(payload, dict):
        raise MLBAPIError("Format général inattendu dans la réponse MLB.")

    return raw_content, payload


def _normalize_game_types(
    game_types: Iterable[str],
) -> tuple[str, ...]:
    """Normalise les types de matchs demandés."""
    normalized_types = tuple(
        sorted(
            {
                str(game_type).strip().upper()
                for game_type in game_types
                if str(game_type).strip()
            }
        )
    )

    if not normalized_types:
        raise ValueError("Au moins un type de match est obligatoire.")

    return normalized_types


def fetch_schedule(target_date: date) -> list[ScheduledGame]:
    """Récupère les matchs MLB d’une journée pour l’interface."""
    parameters: dict[str, object] = {
        "sportId": MLB_SPORT_ID,
        "date": target_date.isoformat(),
        "hydrate": "probablePitcher",
    }

    _, payload = _request_schedule(parameters)
    return _parse_schedule_payload(payload)


def fetch_schedule_range(
    start_date: date,
    end_date: date,
    game_types: Iterable[str] = ("R",),
) -> ScheduleFetchResult:
    """Récupère une période auditable limitée à 31 jours."""
    if end_date < start_date:
        raise ValueError(
            "La date de fin ne peut pas précéder la date de début."
        )

    inclusive_days = (end_date - start_date).days + 1
    if inclusive_days > MAX_SCHEDULE_RANGE_DAYS:
        raise ValueError(
            "Une récupération ne peut pas dépasser 31 jours."
        )

    normalized_game_types = _normalize_game_types(game_types)

    parameters: dict[str, object] = {
        "sportId": MLB_SPORT_ID,
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "gameTypes": ",".join(normalized_game_types),
        "hydrate": "probablePitcher",
    }

    raw_content, payload = _request_schedule(parameters)
    games = _parse_schedule_payload(payload)

    return ScheduleFetchResult(
        start_date=start_date,
        end_date=end_date,
        game_types=normalized_game_types,
        request_parameters=dict(parameters),
        raw_content=raw_content,
        games=tuple(games),
    )


def _parse_date(value: str) -> date:
    """Valide une date écrite sous la forme AAAA-MM-JJ."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "La date doit respecter le format AAAA-MM-JJ."
        ) from error


def _pitcher_label(pitcher_name: str | None) -> str:
    """Prépare un nom de lanceur lisible."""
    return pitcher_name or "non annoncé"


def _display_games(
    games: Sequence[ScheduledGame],
) -> None:
    """Affiche un échantillon lisible des matchs."""
    for game in games[:DISPLAY_LIMIT]:
        away_pitcher = _pitcher_label(
            game.away_probable_pitcher_name
        )
        home_pitcher = _pitcher_label(
            game.home_probable_pitcher_name
        )

        print(
            f"- {game.away_team_name} ({away_pitcher})"
            f" @ {game.home_team_name} ({home_pitcher})"
            f" | {game.status_detail}"
        )

    hidden_games = len(games) - DISPLAY_LIMIT
    if hidden_games > 0:
        print(f"... {hidden_games} autres matchs non affichés")


def main() -> None:
    """Teste une journée ou une période depuis le terminal."""
    parser = argparse.ArgumentParser(
        description="Récupère le calendrier MLB."
    )
    parser.add_argument(
        "--date",
        dest="target_date",
        type=_parse_date,
        help="Journée au format AAAA-MM-JJ.",
    )
    parser.add_argument(
        "--start-date",
        dest="start_date",
        type=_parse_date,
        help="Début de période au format AAAA-MM-JJ.",
    )
    parser.add_argument(
        "--end-date",
        dest="end_date",
        type=_parse_date,
        help="Fin de période au format AAAA-MM-JJ.",
    )
    arguments = parser.parse_args()

    has_start_date = arguments.start_date is not None
    has_end_date = arguments.end_date is not None

    if has_start_date != has_end_date:
        parser.error(
            "--start-date et --end-date doivent être utilisés ensemble."
        )

    if arguments.target_date is not None and has_start_date:
        parser.error(
            "--date ne peut pas être combiné avec une période."
        )

    try:
        if has_start_date and has_end_date:
            result = fetch_schedule_range(
                arguments.start_date,
                arguments.end_date,
            )
            games: Sequence[ScheduledGame] = result.games

            print(
                "Période demandée : "
                f"{result.start_date.isoformat()} au "
                f"{result.end_date.isoformat()}"
            )
            print(f"Octets bruts reçus : {len(result.raw_content)}")
        else:
            target_date = arguments.target_date or date.today()
            games = fetch_schedule(target_date)
            print(f"Date demandée : {target_date.isoformat()}")
    except (MLBAPIError, ValueError) as error:
        raise SystemExit(f"Erreur MLB : {error}") from error

    print(f"Matchs MLB récupérés : {len(games)}")

    if not games:
        print("Aucun match MLB trouvé.")
        return

    _display_games(games)


if __name__ == "__main__":
    main()