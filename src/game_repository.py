"""Enregistrement du calendrier MLB dans SQLite."""

from __future__ import annotations

import argparse
from datetime import date
import sqlite3
from typing import Iterable

from src.database import get_connection, initialize_database
from src.mlb_api import MLBAPIError, ScheduledGame, fetch_schedule


def _upsert_team(
    connection: sqlite3.Connection,
    team_id: int,
    team_name: str,
) -> None:
    """Ajoute une équipe ou actualise son nom."""
    connection.execute(
        """
        INSERT INTO teams (team_id, name)
        VALUES (?, ?)
        ON CONFLICT(team_id) DO UPDATE SET
            name = excluded.name,
            updated_at = CURRENT_TIMESTAMP
        """,
        (team_id, team_name),
    )


def _upsert_game(
    connection: sqlite3.Connection,
    game: ScheduledGame,
) -> None:
    """Ajoute un match ou actualise ses informations."""
    connection.execute(
        """
        INSERT INTO games (
            game_id,
            season,
            official_date,
            game_datetime_utc,
            game_type,
            status_code,
            status_detail,
            away_team_id,
            home_team_id,
            away_score,
            home_score,
            venue_id,
            venue_name,
            doubleheader,
            game_number
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_id) DO UPDATE SET
            season = excluded.season,
            official_date = excluded.official_date,
            game_datetime_utc = excluded.game_datetime_utc,
            game_type = excluded.game_type,
            status_code = excluded.status_code,
            status_detail = excluded.status_detail,
            away_team_id = excluded.away_team_id,
            home_team_id = excluded.home_team_id,
            away_score = excluded.away_score,
            home_score = excluded.home_score,
            venue_id = excluded.venue_id,
            venue_name = excluded.venue_name,
            doubleheader = excluded.doubleheader,
            game_number = excluded.game_number,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            game.game_id,
            game.season,
            game.official_date,
            game.game_datetime_utc,
            game.game_type,
            game.status_code,
            game.status_detail,
            game.away_team_id,
            game.home_team_id,
            game.away_score,
            game.home_score,
            game.venue_id,
            game.venue_name,
            game.doubleheader,
            game.game_number,
        ),
    )


def save_schedule(games: Iterable[ScheduledGame]) -> int:
    """Enregistre une liste de matchs sans créer de doublons."""
    game_list = list(games)
    initialize_database()

    with get_connection() as connection:
        for game in game_list:
            _upsert_team(
                connection,
                game.away_team_id,
                game.away_team_name,
            )
            _upsert_team(
                connection,
                game.home_team_id,
                game.home_team_name,
            )
            _upsert_game(connection, game)

    return len(game_list)


def count_games_for_date(target_date: date) -> int:
    """Compte les matchs enregistrés pour une date."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM games
            WHERE official_date = ?
            """,
            (target_date.isoformat(),),
        ).fetchone()

    return int(row["total"])


def count_teams() -> int:
    """Compte les équipes enregistrées."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM teams
            """
        ).fetchone()

    return int(row["total"])


def _parse_date(value: str) -> date:
    """Valide une date écrite sous la forme AAAA-MM-JJ."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "La date doit respecter le format AAAA-MM-JJ."
        ) from error


def main() -> None:
    """Récupère puis enregistre une journée MLB."""
    parser = argparse.ArgumentParser(
        description="Enregistre le calendrier MLB dans SQLite."
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
        saved_games = save_schedule(games)
        database_games = count_games_for_date(arguments.target_date)
        database_teams = count_teams()
    except (MLBAPIError, sqlite3.Error) as error:
        raise SystemExit(f"Erreur pendant l’enregistrement : {error}") from error

    print(f"Date traitée : {arguments.target_date.isoformat()}")
    print(f"Matchs enregistrés ou actualisés : {saved_games}")
    print(f"Matchs présents en base pour cette date : {database_games}")
    print(f"Équipes présentes en base : {database_teams}")


if __name__ == "__main__":
    main()