"""Tests du service de collecte MLB auditable."""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.game_repository import count_games_for_date
from src.ingestion_repository import get_ingestion_run
from src.ingestion_service import run_schedule_ingestion
from src.mlb_api import (
    ScheduledGame,
    ScheduleFetchResult,
)
from src.raw_archive import load_raw_archive


class IngestionServiceTests(unittest.TestCase):
    """Teste le service sans appeler Internet."""

    def setUp(self) -> None:
        """Crée une base et un dossier de données temporaires."""
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)

        temporary_root = Path(self.temporary_directory.name)
        self.database_path = temporary_root / "test_service.db"
        self.data_directory = temporary_root / "data"
        self.start_date = date(2026, 8, 28)
        self.end_date = date(2026, 8, 28)

    def build_fetch_result(self) -> ScheduleFetchResult:
        """Construit une collecte MLB simulée."""
        game = ScheduledGame(
            game_id=123456,
            season=2026,
            official_date="2026-08-28",
            game_datetime_utc="2026-08-28T23:10:00Z",
            game_type="R",
            status_code="F",
            status_detail="Final",
            away_team_id=100,
            away_team_name="Away Team",
            home_team_id=200,
            home_team_name="Home Team",
            away_score=4,
            home_score=2,
            venue_id=300,
            venue_name="Test Ballpark",
            doubleheader="N",
            game_number=1,
            away_probable_pitcher_id=400,
            away_probable_pitcher_name="Away Pitcher",
            home_probable_pitcher_id=500,
            home_probable_pitcher_name="Home Pitcher",
        )

        parameters = {
            "sportId": 1,
            "startDate": "2026-08-28",
            "endDate": "2026-08-28",
            "gameTypes": "R",
            "hydrate": "probablePitcher",
        }

        return ScheduleFetchResult(
            start_date=self.start_date,
            end_date=self.end_date,
            game_types=("R",),
            request_parameters=parameters,
            raw_content=b'{"totalGames":1,"dates":[]}',
            games=(game,),
        )

    @patch("src.ingestion_service.fetch_schedule_range")
    def test_successful_service_run_is_fully_audited(
        self,
        mocked_fetch: Mock,
    ) -> None:
        """Une réussite doit archiver, enregistrer et journaliser."""
        fetch_result = self.build_fetch_result()
        mocked_fetch.return_value = fetch_result

        result = run_schedule_ingestion(
            start_date=self.start_date,
            end_date=self.end_date,
            database_path=self.database_path,
            data_directory=self.data_directory,
            code_version="test-version",
        )

        self.assertEqual(result.run_id, 1)
        self.assertEqual(result.games_received, 1)
        self.assertEqual(result.games_saved, 1)
        self.assertEqual(result.code_version, "test-version")
        self.assertEqual(
            count_games_for_date(
                self.start_date,
                self.database_path,
            ),
            1,
        )

        journal = get_ingestion_run(
            result.run_id,
            self.database_path,
        )

        self.assertEqual(journal["status"], "success")
        self.assertEqual(journal["records_received"], 1)
        self.assertEqual(journal["records_saved"], 1)
        self.assertEqual(
            journal["response_sha256"],
            result.response_sha256,
        )

        archive_path = (
            self.data_directory.parent
            / result.archive_relative_path
        )

        self.assertTrue(archive_path.exists())
        self.assertEqual(
            load_raw_archive(archive_path),
            fetch_result.raw_content,
        )

        mocked_fetch.assert_called_once_with(
            self.start_date,
            self.end_date,
            game_types=("R",),
        )

    @patch("src.ingestion_service.fetch_schedule_range")
    def test_failed_service_run_is_journaled(
        self,
        mocked_fetch: Mock,
    ) -> None:
        """Une erreur doit clôturer le journal en statut error."""
        mocked_fetch.side_effect = RuntimeError(
            "API simulée indisponible"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "API simulée indisponible",
        ):
            run_schedule_ingestion(
                start_date=self.start_date,
                end_date=self.end_date,
                database_path=self.database_path,
                data_directory=self.data_directory,
                code_version="test-version",
            )

        journal = get_ingestion_run(
            1,
            self.database_path,
        )

        self.assertEqual(journal["status"], "error")
        self.assertIn(
            "API simulée indisponible",
            str(journal["error_message"]),
        )
        self.assertEqual(
            list(self.data_directory.rglob("*.json.gz")),
            [],
        )


if __name__ == "__main__":
    unittest.main()