"""Tests automatiques de la base SQLite."""

import unittest

from src.database import initialize_database, list_tables


class DatabaseTests(unittest.TestCase):
    """Vérifie la structure minimale de la base."""

    def test_required_tables_exist(self) -> None:
        """Les trois premières tables doivent toujours être présentes."""
        initialize_database()

        actual_tables = set(list_tables())
        required_tables = {
            "app_metadata",
            "games",
            "teams",
        }

        self.assertTrue(
            required_tables.issubset(actual_tables),
            msg=(
                "Tables manquantes : "
                f"{sorted(required_tables - actual_tables)}"
            ),
        )


if __name__ == "__main__":
    unittest.main()