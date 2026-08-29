"""Tests de l’enregistrement des matchs MLB."""

from dataclasses import replace
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.database import get_connection
from src.game_repository import (
    count_games_for_date,
    count_pitchers,
    count_teams,
    save_schedule,
)
from src.mlb_api import ScheduledGame


class GameRepositoryTests(unittest.TestCase):
    """Teste l’enregistrement dans une base temporaire."""

    def setUp(self) -> None:
        """Crée une base isolée pour chaque test."""
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)

        self.database_path = (
            Path(self.temporary_directory.name) / "test_games.db"
        )

    def test_schedule_upsert_does_not_create_duplicates(self) -> None:
        """Une actualisation doit modifier le match existant."""
        scheduled_game = ScheduledGame(
            game_id=123456,
            season=2026,
            official_date="2026-08-28",
            game_datetime_utc="2026-08-28T23:10:00Z",
            game_type="R",
            status_code="S",
            status_detail="Scheduled",
            away_team_id=100,
            away_team_name="Away Team",
            home_team_id=200,
            home_team_name="Home Team",
            away_score=None,
            home_score=None,
            venue_id=300,
            venue_name="Test Ballpark",
            doubleheader="N",
            game_number=1,
            away_probable_pitcher_id=400,
            away_probable_pitcher_name="Away Pitcher",
            home_probable_pitcher_id=500,
            home_probable_pitcher_name="Home Pitcher",
        )

        first_saved_count = save_schedule(
            [scheduled_game],
            self.database_path,
        )

        final_game = replace(
            scheduled_game,
            status_code="F",
            status_detail="Final",
            away_score=4,
            home_score=2,
        )

        second_saved_count = save_schedule(
            [final_game],
            self.database_path,
        )

        target_date = date.fromisoformat(
            final_game.official_date
        )

        self.assertEqual(first_saved_count, 1)
        self.assertEqual(second_saved_count, 1)
        self.assertEqual(
            count_games_for_date(
                target_date,
                self.database_path,
            ),
            1,
        )
        self.assertEqual(
            target_date.isoformat(),
            "2026-08-28",
        )
        self.assertEqual(count_teams(self.database_path), 2)
        self.assertEqual(count_pitchers(self.database_path), 2)

        with get_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT
                    status_detail,
                    away_score,
                    home_score,
                    away_probable_pitcher_id,
                    home_probable_pitcher_id
                FROM games
                WHERE game_id = ?
                """,
                (scheduled_game.game_id,),
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row["status_detail"], "Final")
        self.assertEqual(row["away_score"], 4)
        self.assertEqual(row["home_score"], 2)
        self.assertEqual(
            row["away_probable_pitcher_id"],
            400,
        )
        self.assertEqual(
            row["home_probable_pitcher_id"],
            500,
        )


if __name__ == "__main__":
    unittest.main()