"""Enregistrement du calendrier MLB dans SQLite."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sqlite3
from typing import Iterable

from src.database import (
    DATABASE_PATH,
    get_connection,
    initialize_database,
)
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


def _upsert_pitcher(
    connection: sqlite3.Connection,
    pitcher_id: int | None,
    pitcher_name: str | None,
) -> int | None:
    """Ajoute un lanceur annoncé ou retourne None s’il manque."""
    if pitcher_id is None or pitcher_name is None:
        return None

    clean_name = pitcher_name.strip()
    if not clean_name:
        return None

    connection.execute(
        """
        INSERT INTO pitchers (pitcher_id, full_name)
        VALUES (?, ?)
        ON CONFLICT(pitcher_id) DO UPDATE SET
            full_name = excluded.full_name,
            updated_at = CURRENT_TIMESTAMP
        """,
        (pitcher_id, clean_name),
    )

    return pitcher_id


def _upsert_game(
    connection: sqlite3.Connection,
    game: ScheduledGame,
    away_probable_pitcher_id: int | None,
    home_probable_pitcher_id: int | None,
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
            game_number,
            away_probable_pitcher_id,
            home_probable_pitcher_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            away_probable_pitcher_id =
                excluded.away_probable_pitcher_id,
            home_probable_pitcher_id =
                excluded.home_probable_pitcher_id,
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
            away_probable_pitcher_id,
            home_probable_pitcher_id,
        ),
    )


def save_schedule(
    games: Iterable[ScheduledGame],
    database_path: Path = DATABASE_PATH,
) -> int:
    """Enregistre des matchs dans la base demandée."""
    game_list = list(games)
    initialize_database(database_path)

    with get_connection(database_path) as connection:
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

            away_pitcher_id = _upsert_pitcher(
                connection,
                game.away_probable_pitcher_id,
                game.away_probable_pitcher_name,
            )
            home_pitcher_id = _upsert_pitcher(
                connection,
                game.home_probable_pitcher_id,
                game.home_probable_pitcher_name,
            )

            _upsert_game(
                connection,
                game,
                away_pitcher_id,
                home_pitcher_id,
            )

    return len(game_list)


def count_games_for_date(
    target_date: date,
    database_path: Path = DATABASE_PATH,
) -> int:
    """Compte les matchs enregistrés pour une date."""
    initialize_database(database_path)

    with get_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM games
            WHERE official_date = ?
            """,
            (target_date.isoformat(),),
        ).fetchone()

    return int(row["total"])


def count_teams(
    database_path: Path = DATABASE_PATH,
) -> int:
    """Compte les équipes enregistrées."""
    initialize_database(database_path)

    with get_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM teams
            """
        ).fetchone()

    return int(row["total"])


def count_pitchers(
    database_path: Path = DATABASE_PATH,
) -> int:
    """Compte les lanceurs enregistrés."""
    initialize_database(database_path)

    with get_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM pitchers
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
    """Récupère puis enregistre une journée dans la base principale."""
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
        database_pitchers = count_pitchers()
    except (MLBAPIError, sqlite3.Error) as error:
        raise SystemExit(
            f"Erreur pendant l’enregistrement : {error}"
        ) from error

    print(f"Date traitée : {arguments.target_date.isoformat()}")
    print(f"Matchs enregistrés ou actualisés : {saved_games}")
    print(f"Matchs présents en base pour cette date : {database_games}")
    print(f"Équipes présentes en base : {database_teams}")
    print(f"Lanceurs présents en base : {database_pitchers}")


if __name__ == "__main__":
    main()