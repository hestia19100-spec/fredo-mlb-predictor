"""Tests de la consultation locale des cotes Moneyline dans LPF Edge."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.lpf_edge_odds_display import (
    LPFEdgeOddsDisplayError,
    load_latest_moneyline_odds_display,
)


class LPFEdgeOddsDisplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temporary.name)
        self.database = self.root / "mlb.sqlite3"
        self.target = date(2026, 9, 13)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create_database(self) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.executescript(
                """
                CREATE TABLE teams (
                    team_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                );
                CREATE TABLE games (
                    game_id INTEGER PRIMARY KEY,
                    official_date TEXT NOT NULL,
                    game_datetime_utc TEXT NOT NULL,
                    away_team_id INTEGER NOT NULL,
                    home_team_id INTEGER NOT NULL
                );
                CREATE TABLE odds_ingestion_runs (
                    run_id INTEGER PRIMARY KEY,
                    target_official_date TEXT NOT NULL,
                    completed_at_utc TEXT,
                    status TEXT NOT NULL
                );
                CREATE TABLE odds_events (
                    odds_event_id INTEGER PRIMARY KEY,
                    run_id INTEGER NOT NULL,
                    matched_game_id INTEGER
                );
                CREATE TABLE moneyline_odds (
                    odds_quote_id INTEGER PRIMARY KEY,
                    odds_event_id INTEGER NOT NULL,
                    run_id INTEGER NOT NULL,
                    game_id INTEGER NOT NULL,
                    bookmaker_key TEXT NOT NULL,
                    bookmaker_title TEXT NOT NULL,
                    bookmaker_last_update_utc TEXT NOT NULL,
                    observed_at_utc TEXT NOT NULL,
                    away_decimal_odds TEXT NOT NULL,
                    home_decimal_odds TEXT NOT NULL
                );
                """
            )
            connection.executemany(
                "INSERT INTO teams (team_id, name) VALUES (?, ?)",
                (
                    (10, "Équipe extérieure A"),
                    (20, "Équipe domicile A"),
                    (30, "Équipe extérieure B"),
                    (40, "Équipe domicile B"),
                ),
            )
            connection.executemany(
                """
                INSERT INTO games (
                    game_id,
                    official_date,
                    game_datetime_utc,
                    away_team_id,
                    home_team_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (101, "2026-09-13", "2026-09-13T17:10:00Z", 10, 20),
                    (102, "2026-09-13", "2026-09-13T20:10:00Z", 30, 40),
                ),
            )

    def _insert_runs_and_quotes(self) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.executemany(
                """
                INSERT INTO odds_ingestion_runs (
                    run_id,
                    target_official_date,
                    completed_at_utc,
                    status
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (1, "2026-09-13", "2026-09-13T08:00:00Z", "success"),
                    (2, "2026-09-13", "2026-09-13T08:30:00Z", "error"),
                    (3, "2026-09-13", "2026-09-13T09:00:00Z", "success"),
                ),
            )
            connection.execute(
                """
                INSERT INTO odds_events (
                    odds_event_id, run_id, matched_game_id
                ) VALUES (301, 3, 101)
                """
            )
            connection.executemany(
                """
                INSERT INTO moneyline_odds (
                    odds_quote_id,
                    odds_event_id,
                    run_id,
                    game_id,
                    bookmaker_key,
                    bookmaker_title,
                    bookmaker_last_update_utc,
                    observed_at_utc,
                    away_decimal_odds,
                    home_decimal_odds
                ) VALUES (?, 301, 3, 101, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        401,
                        "book_a",
                        "Book A",
                        "2026-09-13T08:58:00Z",
                        "2026-09-13T09:00:00Z",
                        "2.15",
                        "1.74",
                    ),
                    (
                        402,
                        "book_b",
                        "Book B",
                        "2026-09-13T08:59:00Z",
                        "2026-09-13T09:00:00Z",
                        "2.20",
                        "1.74",
                    ),
                ),
            )

    def test_missing_database_returns_empty_without_creation(self) -> None:
        display = load_latest_moneyline_odds_display(
            self.target,
            database_path=self.database,
        )

        self.assertIsNone(display.run_id)
        self.assertEqual(display.games, ())
        self.assertFalse(self.database.exists())

    def test_games_remain_visible_before_any_odds_table_exists(self) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.executescript(
                """
                CREATE TABLE teams (team_id INTEGER PRIMARY KEY, name TEXT);
                CREATE TABLE games (
                    game_id INTEGER PRIMARY KEY,
                    official_date TEXT,
                    game_datetime_utc TEXT,
                    away_team_id INTEGER,
                    home_team_id INTEGER
                );
                INSERT INTO teams VALUES (10, 'Extérieur'), (20, 'Domicile');
                INSERT INTO games VALUES (
                    101, '2026-09-13', '2026-09-13T17:10:00Z', 10, 20
                );
                """
            )

        display = load_latest_moneyline_odds_display(
            self.target,
            database_path=self.database,
        )

        self.assertIsNone(display.run_id)
        self.assertEqual(display.game_count, 1)
        self.assertEqual(display.quoted_game_count, 0)
        self.assertEqual(display.games[0].home_team_name, "Domicile")
        self.assertEqual(display.games[0].away_team_name, "Extérieur")

    def test_latest_success_aggregates_best_prices_home_first(self) -> None:
        self._create_database()
        self._insert_runs_and_quotes()

        display = load_latest_moneyline_odds_display(
            self.target,
            database_path=self.database,
        )

        self.assertEqual(display.run_id, 3)
        self.assertEqual(display.quote_count, 2)
        self.assertEqual(display.game_count, 2)
        self.assertEqual(display.quoted_game_count, 1)
        self.assertEqual(display.missing_game_count, 1)
        self.assertEqual(display.bookmaker_titles, ("Book A", "Book B"))

        first = display.games[0]
        self.assertEqual(first.home_team_name, "Équipe domicile A")
        self.assertEqual(first.away_team_name, "Équipe extérieure A")
        self.assertEqual(first.bookmaker_count, 2)
        self.assertEqual(first.home_best_decimal_odds, Decimal("1.74"))
        self.assertEqual(first.home_best_bookmakers, ("Book A", "Book B"))
        self.assertEqual(first.away_best_decimal_odds, Decimal("2.20"))
        self.assertEqual(first.away_best_bookmakers, ("Book B",))
        self.assertEqual(
            first.latest_bookmaker_update_utc.isoformat(),
            "2026-09-13T08:59:00+00:00",
        )

        second = display.games[1]
        self.assertFalse(second.has_odds)
        self.assertIsNone(second.home_best_decimal_odds)
        self.assertIsNone(second.away_best_decimal_odds)

    def test_latest_error_does_not_hide_previous_success(self) -> None:
        self._create_database()
        self._insert_runs_and_quotes()
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                INSERT INTO odds_ingestion_runs VALUES (
                    4, '2026-09-13', '2026-09-13T09:30:00Z', 'error'
                )
                """
            )

        display = load_latest_moneyline_odds_display(
            self.target,
            database_path=self.database,
        )

        self.assertEqual(display.run_id, 3)
        self.assertEqual(display.quote_count, 2)

    def test_invalid_saved_decimal_fails_closed(self) -> None:
        self._create_database()
        self._insert_runs_and_quotes()
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                UPDATE moneyline_odds
                SET home_decimal_odds = '1.0'
                WHERE odds_quote_id = 401
                """
            )

        with self.assertRaisesRegex(
            LPFEdgeOddsDisplayError,
            "doit dépasser 1",
        ):
            load_latest_moneyline_odds_display(
                self.target,
                database_path=self.database,
            )

    def test_reader_has_no_network_model_or_write_dependency(self) -> None:
        source = Path(__file__).parents[1].joinpath(
            "src", "lpf_edge_odds_display.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("requests", source)
        self.assertNotIn("predict_proba", source)
        self.assertNotIn("initialize_", source)
        self.assertNotIn("INSERT ", source)
        self.assertNotIn("UPDATE ", source)
        self.assertIn("mode=ro", source)


if __name__ == "__main__":
    unittest.main()
