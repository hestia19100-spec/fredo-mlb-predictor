"""Client minimal pour récupérer le calendrier depuis l’API MLB."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from typing import Any

import requests


MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_SPORT_ID = 1
REQUEST_TIMEOUT_SECONDS = 20


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


def fetch_schedule(target_date: date) -> list[ScheduledGame]:
    """Récupère les matchs MLB d’une date précise."""
    parameters = {
        "sportId": MLB_SPORT_ID,
        "date": target_date.isoformat(),
        "hydrate": "probablePitcher",
    }
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
        payload = response.json()
    except requests.RequestException as error:
        raise MLBAPIError(
            f"Impossible de joindre l’API MLB : {error}"
        ) from error
    except ValueError as error:
        raise MLBAPIError(
            "L’API MLB n’a pas renvoyé un document JSON valide."
        ) from error

    if not isinstance(payload, dict):
        raise MLBAPIError("Format général inattendu dans la réponse MLB.")

    date_blocks = payload.get("dates", [])
    if not isinstance(date_blocks, list):
        raise MLBAPIError("Liste des dates absente de la réponse MLB.")

    games: list[ScheduledGame] = []

    for date_block in date_blocks:
        if not isinstance(date_block, dict):
            raise MLBAPIError("Bloc de date MLB invalide.")

        raw_games = date_block.get("games", [])
        if not isinstance(raw_games, list):
            raise MLBAPIError("Liste des matchs MLB invalide.")

        for raw_game in raw_games:
            if not isinstance(raw_game, dict):
                raise MLBAPIError("Match MLB invalide.")

            games.append(_parse_game(raw_game))

    return games


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


def main() -> None:
    """Teste la récupération du calendrier depuis le terminal."""
    parser = argparse.ArgumentParser(
        description="Récupère le calendrier MLB d’une date."
    )
    parser.add_argument(
        "--date",
        dest="target_date",
        type=_parse_date,
        default=date.today(),
        help="Date au format AAAA-MM-JJ.",
    )
    arguments = parser.parse_args()

    try:
        games = fetch_schedule(arguments.target_date)
    except MLBAPIError as error:
        raise SystemExit(f"Erreur MLB : {error}") from error

    print(f"Date demandée : {arguments.target_date.isoformat()}")
    print(f"Matchs MLB récupérés : {len(games)}")

    if not games:
        print("Aucun match MLB trouvé pour cette date.")
        return

    for game in games:
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


if __name__ == "__main__":
    main()