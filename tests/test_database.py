"""Tests automatiques de la base SQLite."""

import unittest

from src.database import (
    get_connection,
    initialize_database,
    list_tables,
)


class DatabaseTests(unittest.TestCase):
    """Vérifie la structure minimale de la base."""

    def test_required_tables_exist(self) -> None:
        """Les tables indispensables doivent toujours être présentes."""
        initialize_database()

        actual_tables = set(list_tables())
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
        initialize_database()

        with get_connection() as connection:
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