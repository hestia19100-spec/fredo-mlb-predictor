"""Connexion et initialisation de la base de données SQLite."""

from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "fredo_mlb.db"
SCHEMA_VERSION = "3"


def get_connection() -> sqlite3.Connection:
    """Ouvre une connexion SQLite configurée pour l’application."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def _add_pitcher_columns_if_needed(
    connection: sqlite3.Connection,
) -> None:
    """Met à niveau une ancienne table games sans effacer ses données."""
    rows = connection.execute(
        "PRAGMA table_info(games)"
    ).fetchall()
    column_names = {str(row["name"]) for row in rows}

    if "away_probable_pitcher_id" not in column_names:
        connection.execute(
            """
            ALTER TABLE games
            ADD COLUMN away_probable_pitcher_id INTEGER
                REFERENCES pitchers (pitcher_id)
            """
        )

    if "home_probable_pitcher_id" not in column_names:
        connection.execute(
            """
            ALTER TABLE games
            ADD COLUMN home_probable_pitcher_id INTEGER
                REFERENCES pitchers (pitcher_id)
            """
        )


def initialize_database() -> Path:
    """Crée ou met à niveau les premières tables du projet."""
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS app_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                abbreviation TEXT,
                league_name TEXT,
                division_name TEXT,
                active INTEGER NOT NULL DEFAULT 1
                    CHECK (active IN (0, 1)),
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS pitchers (
                pitcher_id INTEGER PRIMARY KEY,
                full_name TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS games (
                game_id INTEGER PRIMARY KEY,
                season INTEGER NOT NULL,
                official_date TEXT NOT NULL,
                game_datetime_utc TEXT,
                game_type TEXT NOT NULL,
                status_code TEXT NOT NULL,
                status_detail TEXT NOT NULL,
                away_team_id INTEGER NOT NULL,
                home_team_id INTEGER NOT NULL,
                away_score INTEGER,
                home_score INTEGER,
                venue_id INTEGER,
                venue_name TEXT,
                doubleheader TEXT,
                game_number INTEGER,
                away_probable_pitcher_id INTEGER,
                home_probable_pitcher_id INTEGER,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (away_team_id)
                    REFERENCES teams (team_id),

                FOREIGN KEY (home_team_id)
                    REFERENCES teams (team_id),

                FOREIGN KEY (away_probable_pitcher_id)
                    REFERENCES pitchers (pitcher_id),

                FOREIGN KEY (home_probable_pitcher_id)
                    REFERENCES pitchers (pitcher_id),

                CHECK (away_team_id <> home_team_id),
                CHECK (away_score IS NULL OR away_score >= 0),
                CHECK (home_score IS NULL OR home_score >= 0)
            );

            CREATE INDEX IF NOT EXISTS idx_games_official_date
                ON games (official_date);

            CREATE INDEX IF NOT EXISTS idx_games_teams
                ON games (away_team_id, home_team_id);
            """
        )

        _add_pitcher_columns_if_needed(connection)

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_games_probable_pitchers
            ON games (
                away_probable_pitcher_id,
                home_probable_pitcher_id
            )
            """
        )

        connection.execute(
            """
            INSERT INTO app_metadata (key, value)
            VALUES ('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (SCHEMA_VERSION,),
        )

    return DATABASE_PATH


def list_tables() -> list[str]:
    """Retourne la liste des tables présentes dans la base."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()

    return [row["name"] for row in rows]


def main() -> None:
    """Lance un contrôle simple de SQLite."""
    database_path = initialize_database()
    tables = ", ".join(list_tables())

    print(f"Base SQLite prête : {database_path}")
    print(f"Tables présentes : {tables}")


if __name__ == "__main__":
    main()