"""Connexion, initialisation et suivi de la base SQLite."""

from hashlib import sha256
from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "fredo_mlb.db"

SCHEMA_VERSION = 3
BASELINE_MIGRATION_NAME = "baseline_pitchers"
BASELINE_SCHEMA_SIGNATURE = (
    "v3|app_metadata|teams|pitchers|games|"
    "games.away_probable_pitcher_id|"
    "games.home_probable_pitcher_id"
)
BASELINE_MIGRATION_CHECKSUM = sha256(
    BASELINE_SCHEMA_SIGNATURE.encode("utf-8")
).hexdigest()


class DatabaseMigrationError(RuntimeError):
    """Signale une incohérence dans l’historique des migrations."""


def get_connection(
    database_path: Path = DATABASE_PATH,
) -> sqlite3.Connection:
    """Ouvre une connexion vers la base SQLite demandée."""
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
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


def _register_baseline_migration(
    connection: sqlite3.Connection,
) -> None:
    """Enregistre ou contrôle la structure de référence actuelle."""
    existing_row = connection.execute(
        """
        SELECT name, checksum
        FROM schema_migrations
        WHERE version = ?
        """,
        (SCHEMA_VERSION,),
    ).fetchone()

    if existing_row is None:
        connection.execute(
            """
            INSERT INTO schema_migrations (
                version,
                name,
                checksum
            )
            VALUES (?, ?, ?)
            """,
            (
                SCHEMA_VERSION,
                BASELINE_MIGRATION_NAME,
                BASELINE_MIGRATION_CHECKSUM,
            ),
        )
        return

    existing_name = str(existing_row["name"])
    existing_checksum = str(existing_row["checksum"])

    if (
        existing_name != BASELINE_MIGRATION_NAME
        or existing_checksum != BASELINE_MIGRATION_CHECKSUM
    ):
        raise DatabaseMigrationError(
            "La migration de référence SQLite a été modifiée."
        )


def initialize_database(
    database_path: Path = DATABASE_PATH,
) -> Path:
    """Crée ou met à niveau la base SQLite demandée."""
    with get_connection(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY
                    CHECK (version > 0),
                name TEXT NOT NULL UNIQUE,
                checksum TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

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

        _register_baseline_migration(connection)

        connection.execute(
            """
            INSERT INTO app_metadata (key, value)
            VALUES ('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (str(SCHEMA_VERSION),),
        )

    return database_path


def list_tables(
    database_path: Path = DATABASE_PATH,
) -> list[str]:
    """Retourne les tables présentes dans la base demandée."""
    with get_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()

    return [str(row["name"]) for row in rows]


def list_applied_migrations(
    database_path: Path = DATABASE_PATH,
) -> list[int]:
    """Retourne les numéros des migrations enregistrées."""
    with get_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT version
            FROM schema_migrations
            ORDER BY version
            """
        ).fetchall()

    return [int(row["version"]) for row in rows]


def main() -> None:
    """Lance un contrôle simple de la base principale."""
    database_path = initialize_database()
    tables = ", ".join(list_tables(database_path))
    migrations = ", ".join(
        str(version)
        for version in list_applied_migrations(database_path)
    )

    print(f"Base SQLite prête : {database_path}")
    print(f"Tables présentes : {tables}")
    print(f"Migrations enregistrées : {migrations}")


if __name__ == "__main__":
    main()