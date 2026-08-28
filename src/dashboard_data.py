"""Lecture des données SQLite destinées à l’interface."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date

from src.database import get_connection, initialize_database


@dataclass(frozen=True, slots=True)
class StoredGame:
    """Match MLB relu depuis SQLite."""

    game_id: int
    official_date: str
    game_datetime_utc: str
    away_team_name: str
    home_team_name: str
    away_score: int | None
    home_score: int | None
    status_detail: str
    venue_name: str | None


def load_games_for_date(target_date: date) -> list[StoredGame]:
    """Charge les matchs enregistrés pour une date précise."""
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                games.game_id,
                games.official_date,
                games.game_datetime_utc,
                away_team.name AS away_team_name,
                home_team.name AS home_team_name,
                games.away_score,
                games.home_score,
                games.status_detail,
                games.venue_name
            FROM games
            INNER JOIN teams AS away_team
                ON away_team.team_id = games.away_team_id
            INNER JOIN teams AS home_team
                ON home_team.team_id = games.home_team_id
            WHERE games.official_date = ?
            ORDER BY games.game_datetime_utc, games.game_id
            """,
            (target_date.isoformat(),),
        ).fetchall()

    return [
        StoredGame(
            game_id=int(row["game_id"]),
            official_date=str(row["official_date"]),
            game_datetime_utc=str(row["game_datetime_utc"]),
            away_team_name=str(row["away_team_name"]),
            home_team_name=str(row["home_team_name"]),
            away_score=(
                int(row["away_score"])
                if row["away_score"] is not None
                else None
            ),
            home_score=(
                int(row["home_score"])
                if row["home_score"] is not None
                else None
            ),
            status_detail=str(row["status_detail"]),
            venue_name=(
                str(row["venue_name"])
                if row["venue_name"] is not None
                else None
            ),
        )
        for row in rows
    ]


def _parse_date(value: str) -> date:
    """Valide une date écrite sous la forme AAAA-MM-JJ."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "La date doit respecter le format AAAA-MM-JJ."
        ) from error


def _format_score(game: StoredGame) -> str:
    """Prépare un score lisible."""
    if game.away_score is None or game.home_score is None:
        return "Score indisponible"

    return f"{game.away_score} - {game.home_score}"


def main() -> None:
    """Teste la lecture des matchs depuis SQLite."""
    parser = argparse.ArgumentParser(
        description="Lit les matchs présents dans SQLite."
    )
    parser.add_argument(
        "--date",
        dest="target_date",
        type=_parse_date,
        default=date.today(),
        help="Date au format AAAA-MM-JJ.",
    )
    arguments = parser.parse_args()

    games = load_games_for_date(arguments.target_date)

    print(f"Date lue : {arguments.target_date.isoformat()}")
    print(f"Matchs lus depuis SQLite : {len(games)}")

    for game in games:
        print(
            f"- {game.away_team_name} @ {game.home_team_name}"
            f" | {game.status_detail}"
            f" | {_format_score(game)}"
        )


if __name__ == "__main__":
    main()