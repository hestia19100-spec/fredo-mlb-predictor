"""Tests du tableau local de préparation avant les prédictions."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from src import lpf_edge_daily_operations as operations


TARGET = date(2026, 9, 13)
NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)


class LPFEdgePredictionPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "games.db"

    def _create_database(self) -> None:
        connection = sqlite3.connect(self.database)
        try:
            connection.executescript(
                """
                CREATE TABLE teams (
                    team_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                );
                CREATE TABLE pitchers (
                    pitcher_id INTEGER PRIMARY KEY,
                    full_name TEXT NOT NULL
                );
                CREATE TABLE games (
                    game_id INTEGER PRIMARY KEY,
                    official_date TEXT NOT NULL,
                    game_datetime_utc TEXT,
                    away_team_id INTEGER NOT NULL,
                    home_team_id INTEGER NOT NULL,
                    away_probable_pitcher_id INTEGER,
                    home_probable_pitcher_id INTEGER
                );
                """
            )
            connection.executemany(
                "INSERT INTO teams VALUES (?, ?)",
                [
                    (10, "Visiteurs A"),
                    (20, "Domicile A"),
                    (30, "Visiteurs B"),
                    (40, "Domicile B"),
                ],
            )
            connection.executemany(
                "INSERT INTO pitchers VALUES (?, ?)",
                [
                    (100, "Lanceur A"),
                    (200, "Lanceur B"),
                    (300, "Lanceur C"),
                ],
            )
            connection.executemany(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        2,
                        TARGET.isoformat(),
                        "2026-09-13T19:00:00Z",
                        30,
                        40,
                        200,
                        300,
                    ),
                    (
                        1,
                        TARGET.isoformat(),
                        "2026-09-13T18:00:00Z",
                        10,
                        20,
                        100,
                        None,
                    ),
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def test_missing_database_returns_empty_without_creation(self) -> None:
        preparation = operations.load_prediction_preparation(
            TARGET,
            now_utc=NOW,
            database_path=self.database,
        )
        self.assertEqual(preparation.games, ())
        self.assertEqual(preparation.game_count, 0)
        self.assertEqual(preparation.expected_pitcher_count, 0)
        self.assertIsNone(preparation.prediction_deadline_utc)
        self.assertFalse(self.database.exists())

    def test_summary_and_rows_use_only_saved_information(self) -> None:
        self._create_database()
        before = self.database.read_bytes()

        preparation = operations.load_prediction_preparation(
            TARGET,
            now_utc=NOW,
            database_path=self.database,
        )

        self.assertEqual([game.game_id for game in preparation.games], [1, 2])
        self.assertEqual(preparation.game_count, 2)
        self.assertEqual(preparation.expected_pitcher_count, 4)
        self.assertEqual(preparation.announced_pitcher_count, 3)
        self.assertEqual(preparation.missing_pitcher_count, 1)
        self.assertEqual(preparation.complete_game_count, 1)
        self.assertEqual(
            preparation.first_start_utc,
            datetime(2026, 9, 13, 18, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            preparation.prediction_deadline_utc,
            datetime(2026, 9, 13, 16, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(preparation.remaining_minutes, 240)
        self.assertEqual(
            preparation.games[0].away_probable_pitcher_name,
            "Lanceur A",
        )
        self.assertIsNone(
            preparation.games[0].home_probable_pitcher_name
        )
        self.assertEqual(self.database.read_bytes(), before)

    def test_reader_never_calls_network_or_prediction_engine(self) -> None:
        self._create_database()
        forbidden = AssertionError("effet externe interdit")
        with (
            mock.patch(
                "requests.get",
                side_effect=forbidden,
                create=True,
            ) as network,
            mock.patch.object(
                operations,
                "execute_daily_prediction_publication",
                side_effect=forbidden,
            ) as prediction,
        ):
            preparation = operations.load_prediction_preparation(
                TARGET,
                now_utc=NOW,
                database_path=self.database,
            )
        self.assertEqual(preparation.game_count, 2)
        network.assert_not_called()
        prediction.assert_not_called()

    def test_missing_team_fails_closed(self) -> None:
        self._create_database()
        connection = sqlite3.connect(self.database)
        try:
            connection.execute("DELETE FROM teams WHERE team_id = 10")
            connection.commit()
        finally:
            connection.close()

        with self.assertRaisesRegex(
            operations.DailyOperationsError,
            "équipe locale est absente",
        ):
            operations.load_prediction_preparation(
                TARGET,
                now_utc=NOW,
                database_path=self.database,
            )

    def test_naive_clock_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            operations.DailyOperationsError,
            "horloge du tableau",
        ):
            operations.load_prediction_preparation(
                TARGET,
                now_utc=datetime(2026, 9, 13, 12, 0),
                database_path=self.database,
            )


if __name__ == "__main__":
    unittest.main()
