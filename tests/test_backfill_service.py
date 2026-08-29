from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.backfill_service import (
    ACTION_COLLECT,
    ACTION_SKIP,
    build_backfill_preview,
)
from src.ingestion_repository import (
    mark_ingestion_error,
    mark_ingestion_success,
    start_ingestion_run,
)
from src.ingestion_service import (
    INGESTION_SOURCE,
    build_schedule_request_parameters,
)


class BackfillServiceTests(unittest.TestCase):
    """Vérifie l’aperçu et la reprise des collectes historiques."""

    def setUp(self) -> None:
        """Crée une base SQLite isolée pour chaque test."""
        self.temporary_directory = TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name)
            / "backfill_service_test.db"
        )

    def tearDown(self) -> None:
        """Supprime la base temporaire."""
        self.temporary_directory.cleanup()

    def record_run(
        self,
        *,
        start_date: date,
        end_date: date,
        status: str,
    ) -> int:
        """Enregistre une collecte contrôlée dans la base temporaire."""
        game_types = ("R",)
        request_parameters = build_schedule_request_parameters(
            start_date=start_date,
            end_date=end_date,
            game_types=game_types,
        )

        run_id = start_ingestion_run(
            source=INGESTION_SOURCE,
            start_date=start_date,
            end_date=end_date,
            game_types=game_types,
            request_parameters=request_parameters,
            code_version="test-version",
            database_path=self.database_path,
        )

        if status == "success":
            mark_ingestion_success(
                run_id=run_id,
                records_received=10,
                records_saved=10,
                raw_response_path=(
                    f"data/raw/mlb_schedule/test-{run_id}.json.gz"
                ),
                response_sha256="a" * 64,
                database_path=self.database_path,
            )
        elif status == "error":
            mark_ingestion_error(
                run_id=run_id,
                error_message="Erreur simulée",
                database_path=self.database_path,
            )

        return run_id

    def test_pending_period_is_split_without_network_call(self) -> None:
        """Un aperçu doit planifier sans jamais contacter MLB."""
        with patch(
            "src.ingestion_service.fetch_schedule_range"
        ) as mocked_fetch:
            preview = build_backfill_preview(
                start_date=date(2026, 1, 1),
                end_date=date(2026, 2, 1),
                database_path=self.database_path,
            )

        mocked_fetch.assert_not_called()
        self.assertEqual(preview.total_chunks, 2)
        self.assertEqual(preview.pending_chunks, 2)
        self.assertEqual(preview.skipped_chunks, 0)
        self.assertEqual(
            [plan.action for plan in preview.chunk_plans],
            [ACTION_COLLECT, ACTION_COLLECT],
        )

    def test_exact_successful_chunk_is_skipped(self) -> None:
        """Un lot strictement identique et réussi doit être ignoré."""
        selected_date = date(2026, 8, 28)
        run_id = self.record_run(
            start_date=selected_date,
            end_date=selected_date,
            status="success",
        )

        preview = build_backfill_preview(
            start_date=selected_date,
            end_date=selected_date,
            database_path=self.database_path,
        )

        self.assertEqual(preview.total_chunks, 1)
        self.assertEqual(preview.skipped_chunks, 1)
        self.assertEqual(preview.pending_chunks, 0)
        self.assertEqual(
            preview.chunk_plans[0].action,
            ACTION_SKIP,
        )
        self.assertEqual(
            preview.chunk_plans[0].successful_run_id,
            run_id,
        )
        self.assertFalse(
            preview.chunk_plans[0].should_collect
        )

    def test_failed_chunk_remains_pending(self) -> None:
        """Un lot en erreur doit rester à collecter."""
        selected_date = date(2026, 8, 27)
        self.record_run(
            start_date=selected_date,
            end_date=selected_date,
            status="error",
        )

        preview = build_backfill_preview(
            start_date=selected_date,
            end_date=selected_date,
            database_path=self.database_path,
        )

        self.assertEqual(preview.skipped_chunks, 0)
        self.assertEqual(preview.pending_chunks, 1)
        self.assertEqual(
            preview.chunk_plans[0].action,
            ACTION_COLLECT,
        )
        self.assertTrue(
            preview.chunk_plans[0].should_collect
        )

    def test_successful_subperiod_does_not_skip_larger_chunk(
        self,
    ) -> None:
        """Une réussite partielle ne doit pas masquer un lot complet."""
        self.record_run(
            start_date=date(2026, 8, 28),
            end_date=date(2026, 8, 28),
            status="success",
        )

        preview = build_backfill_preview(
            start_date=date(2026, 8, 1),
            end_date=date(2026, 8, 31),
            database_path=self.database_path,
        )

        self.assertEqual(preview.total_chunks, 1)
        self.assertEqual(preview.skipped_chunks, 0)
        self.assertEqual(preview.pending_chunks, 1)
        self.assertEqual(
            preview.chunk_plans[0].action,
            ACTION_COLLECT,
        )


if __name__ == "__main__":
    unittest.main()