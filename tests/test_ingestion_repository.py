"""Tests du journal des collectes MLB."""

from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.ingestion_repository import (
    get_ingestion_run,
    mark_ingestion_error,
    mark_ingestion_success,
    start_ingestion_run,
)


class IngestionRepositoryTests(unittest.TestCase):
    """Teste les collectes dans une base temporaire."""

    def setUp(self) -> None:
        """Crée une base isolée pour chaque test."""
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)

        self.database_path = (
            Path(self.temporary_directory.name) / "test_ingestion.db"
        )

    def test_successful_ingestion_is_recorded(self) -> None:
        """Une collecte réussie doit conserver ses informations."""
        request_parameters = {
            "sportId": 1,
            "startDate": "2026-08-28",
        }

        run_id = start_ingestion_run(
            source="mlb_stats_api_schedule",
            start_date=date(2026, 8, 28),
            end_date=date(2026, 8, 28),
            game_types=("R",),
            request_parameters=request_parameters,
            code_version="test-commit",
            database_path=self.database_path,
        )

        started_run = get_ingestion_run(
            run_id,
            self.database_path,
        )

        self.assertEqual(started_run["status"], "started")
        self.assertIsNone(started_run["completed_at_utc"])
        self.assertEqual(started_run["game_types"], "R")
        self.assertEqual(
            json.loads(str(started_run["request_parameters_json"])),
            request_parameters,
        )

        mark_ingestion_success(
            run_id=run_id,
            records_received=15,
            records_saved=15,
            raw_response_path="raw/mlb/2026-08-28.json.gz",
            response_sha256="a" * 64,
            database_path=self.database_path,
        )

        completed_run = get_ingestion_run(
            run_id,
            self.database_path,
        )

        self.assertEqual(completed_run["status"], "success")
        self.assertEqual(completed_run["records_received"], 15)
        self.assertEqual(completed_run["records_saved"], 15)
        self.assertEqual(
            completed_run["raw_response_path"],
            "raw/mlb/2026-08-28.json.gz",
        )
        self.assertEqual(
            completed_run["response_sha256"],
            "a" * 64,
        )
        self.assertIsNotNone(completed_run["completed_at_utc"])

    def test_failed_ingestion_is_recorded(self) -> None:
        """Une erreur doit être conservée de façon explicite."""
        run_id = start_ingestion_run(
            source="mlb_stats_api_schedule",
            start_date=date(2026, 8, 28),
            end_date=date(2026, 8, 28),
            game_types=("R",),
            request_parameters={"sportId": 1},
            database_path=self.database_path,
        )

        mark_ingestion_error(
            run_id=run_id,
            error_message="Erreur réseau simulée",
            database_path=self.database_path,
        )

        failed_run = get_ingestion_run(
            run_id,
            self.database_path,
        )

        self.assertEqual(failed_run["status"], "error")
        self.assertEqual(
            failed_run["error_message"],
            "Erreur réseau simulée",
        )
        self.assertIsNotNone(failed_run["completed_at_utc"])


if __name__ == "__main__":
    unittest.main()