"""Tests automatiques de la base SQLite."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.database import (
    get_connection,
    initialize_database,
    list_tables,
)


class DatabaseTests(unittest.TestCase):
    """Vérifie la structure dans une base temporaire isolée."""

    def setUp(self) -> None:
        """Crée une base vide propre à chaque test."""
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)

        self.database_path = (
            Path(self.temporary_directory.name) / "test_database.db"
        )

    def test_required_tables_exist(self) -> None:
        """Les tables indispensables doivent toujours être présentes."""
        initialize_database(self.database_path)

        actual_tables = set(list_tables(self.database_path))
        required_tables = {
            "app_metadata",
            "games",
            "pitchers",
            "teams",
        }

        self.assertTrue(
            required_tables.issubset(actual_tables),
            msg=(
                "Tables manquantes : "
                f"{sorted(required_tables - actual_tables)}"
            ),
        )

    def test_games_contains_probable_pitcher_columns(self) -> None:
        """Les matchs doivent pouvoir référencer les deux lanceurs."""
        initialize_database(self.database_path)

        with get_connection(self.database_path) as connection:
            rows = connection.execute(
                "PRAGMA table_info(games)"
            ).fetchall()

        actual_columns = {str(row["name"]) for row in rows}
        required_columns = {
            "away_probable_pitcher_id",
            "home_probable_pitcher_id",
        }

        self.assertTrue(
            required_columns.issubset(actual_columns),
            msg=(
                "Colonnes manquantes : "
                f"{sorted(required_columns - actual_columns)}"
            ),
        )


if __name__ == "__main__":
    unittest.main()