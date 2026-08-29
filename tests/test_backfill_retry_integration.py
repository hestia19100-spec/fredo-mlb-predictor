from datetime import date
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.backfill_service import execute_backfill
from src.ingestion_service import ScheduleIngestionResult
from src.mlb_api import MLBAPIError
from src.retry_policy import RetryPolicy


class BackfillRetryIntegrationTests(unittest.TestCase):
    """Vérifie les tentatives réseau du backfill."""

    def setUp(self) -> None:
        """Crée un environnement SQLite isolé."""
        self.temporary_directory = TemporaryDirectory()
        temporary_path = Path(self.temporary_directory.name)
        self.data_directory = temporary_path / "data"
        self.database_path = (
            self.data_directory / "retry_integration_test.db"
        )

    def tearDown(self) -> None:
        """Supprime l’environnement temporaire."""
        self.temporary_directory.cleanup()

    def build_result(
        self,
        **arguments: object,
    ) -> ScheduleIngestionResult:
        """Construit un résultat simulé sans appel réseau."""
        start_date = arguments["start_date"]
        end_date = arguments["end_date"]

        if not isinstance(start_date, date):
            raise AssertionError("Date de début simulée invalide.")

        if not isinstance(end_date, date):
            raise AssertionError("Date de fin simulée invalide.")

        return ScheduleIngestionResult(
            run_id=200,
            start_date=start_date,
            end_date=end_date,
            games_received=5,
            games_saved=5,
            archive_relative_path=(
                "data/raw/mlb_schedule/"
                f"{start_date.isoformat()}_retry_test.json.gz"
            ),
            response_sha256="c" * 64,
            code_version="retry-integration-test",
        )

    def test_mlb_errors_are_retried_until_success(self) -> None:
        """Deux erreurs MLB doivent être suivies d’une réussite."""
        calls = 0
        delays: list[float] = []

        def flaky_ingestion(
            **arguments: object,
        ) -> ScheduleIngestionResult:
            nonlocal calls
            calls += 1

            if calls < 3:
                raise MLBAPIError(
                    f"Erreur MLB temporaire n° {calls}"
                )

            return self.build_result(**arguments)

        with patch(
            "src.backfill_service.run_schedule_ingestion",
            side_effect=flaky_ingestion,
        ):
            execution = execute_backfill(
                start_date=date(2026, 8, 26),
                end_date=date(2026, 8, 26),
                database_path=self.database_path,
                data_directory=self.data_directory,
                retry_policy=RetryPolicy(
                    max_attempts=3,
                    initial_delay_seconds=1.0,
                    backoff_multiplier=2.0,
                ),
                chunk_delay_seconds=0,
                sleep_function=delays.append,
            )

        self.assertEqual(calls, 3)
        self.assertEqual(delays, [1.0, 2.0])
        self.assertEqual(execution.executed_chunks, 1)
        self.assertEqual(execution.attempt_counts, (3,))
        self.assertEqual(execution.total_attempts, 3)
        self.assertEqual(execution.retried_chunks, 1)

    def test_exhausted_mlb_errors_are_raised(self) -> None:
        """La troisième erreur MLB doit arrêter l’exécution."""
        calls = 0
        delays: list[float] = []

        def failing_ingestion(
            **arguments: object,
        ) -> ScheduleIngestionResult:
            nonlocal calls
            calls += 1
            raise MLBAPIError(
                f"Erreur MLB persistante n° {calls}"
            )

        with patch(
            "src.backfill_service.run_schedule_ingestion",
            side_effect=failing_ingestion,
        ):
            with self.assertRaises(MLBAPIError):
                execute_backfill(
                    start_date=date(2026, 8, 26),
                    end_date=date(2026, 8, 26),
                    database_path=self.database_path,
                    data_directory=self.data_directory,
                    retry_policy=RetryPolicy(
                        max_attempts=3,
                        initial_delay_seconds=1.0,
                        backoff_multiplier=2.0,
                    ),
                    chunk_delay_seconds=0,
                    sleep_function=delays.append,
                )

        self.assertEqual(calls, 3)
        self.assertEqual(delays, [1.0, 2.0])

    def test_sqlite_error_is_not_retried(self) -> None:
        """Une erreur SQLite doit remonter dès le premier essai."""
        calls = 0
        delays: list[float] = []

        def failing_ingestion(
            **arguments: object,
        ) -> ScheduleIngestionResult:
            nonlocal calls
            calls += 1
            raise sqlite3.OperationalError(
                "Erreur SQLite simulée"
            )

        with patch(
            "src.backfill_service.run_schedule_ingestion",
            side_effect=failing_ingestion,
        ):
            with self.assertRaises(sqlite3.OperationalError):
                execute_backfill(
                    start_date=date(2026, 8, 26),
                    end_date=date(2026, 8, 26),
                    database_path=self.database_path,
                    data_directory=self.data_directory,
                    retry_policy=RetryPolicy(max_attempts=3),
                    chunk_delay_seconds=0,
                    sleep_function=delays.append,
                )

        self.assertEqual(calls, 1)
        self.assertEqual(delays, [])

    def test_two_chunks_have_one_intermediate_pause(self) -> None:
        """Deux lots réussis doivent être séparés par une pause."""
        delays: list[float] = []

        with patch(
            "src.backfill_service.run_schedule_ingestion",
            side_effect=self.build_result,
        ) as mocked_ingestion:
            execution = execute_backfill(
                start_date=date(2026, 1, 1),
                end_date=date(2026, 2, 1),
                max_chunks=2,
                database_path=self.database_path,
                data_directory=self.data_directory,
                retry_policy=RetryPolicy(max_attempts=1),
                chunk_delay_seconds=1.5,
                sleep_function=delays.append,
            )

        self.assertEqual(mocked_ingestion.call_count, 2)
        self.assertEqual(delays, [1.5])
        self.assertEqual(execution.attempt_counts, (1, 1))
        self.assertEqual(execution.total_attempts, 2)
        self.assertEqual(execution.retried_chunks, 0)

    def test_invalid_chunk_delays_are_rejected(self) -> None:
        """Une pause invalide doit être refusée avant tout appel."""
        invalid_delays = (
            -1,
            True,
            float("nan"),
            float("inf"),
        )

        for invalid_delay in invalid_delays:
            with self.subTest(invalid_delay=invalid_delay):
                with patch(
                    "src.backfill_service.run_schedule_ingestion"
                ) as mocked_ingestion:
                    with self.assertRaises(ValueError):
                        execute_backfill(
                            start_date=date(2026, 8, 26),
                            end_date=date(2026, 8, 26),
                            database_path=self.database_path,
                            data_directory=self.data_directory,
                            chunk_delay_seconds=invalid_delay,
                        )

                mocked_ingestion.assert_not_called()


if __name__ == "__main__":
    unittest.main()