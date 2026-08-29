from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.backfill_service import (
    ACTION_COLLECT,
    ACTION_SKIP,
    build_backfill_preview,
    execute_backfill,
)
from src.ingestion_repository import (
    mark_ingestion_error,
    mark_ingestion_success,
    start_ingestion_run,
)
from src.ingestion_service import (
    INGESTION_SOURCE,
    ScheduleIngestionResult,
    build_schedule_request_parameters,
)


class BackfillServiceTests(unittest.TestCase):
    """Vérifie l’aperçu et la reprise des collectes historiques."""

    def setUp(self) -> None:
        """Crée un environnement isolé pour chaque test."""
        self.temporary_directory = TemporaryDirectory()
        temporary_path = Path(self.temporary_directory.name)
        self.database_path = (
            temporary_path / "backfill_service_test.db"
        )
        self.data_directory = temporary_path / "data"

    def tearDown(self) -> None:
        """Supprime l’environnement temporaire."""
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

    def build_fake_ingestion_result(
        self,
        **arguments: object,
    ) -> ScheduleIngestionResult:
        """Simule une collecte sans effectuer d’appel réseau."""
        start_date = arguments["start_date"]
        end_date = arguments["end_date"]
        code_version = arguments.get("code_version")

        if not isinstance(start_date, date):
            raise AssertionError("Date de début invalide dans le test.")

        if not isinstance(end_date, date):
            raise AssertionError("Date de fin invalide dans le test.")

        normalized_code_version = (
            str(code_version)
            if code_version is not None
            else None
        )

        return ScheduleIngestionResult(
            run_id=100,
            start_date=start_date,
            end_date=end_date,
            games_received=5,
            games_saved=5,
            archive_relative_path=(
                "data/raw/mlb_schedule/"
                f"{start_date.isoformat()}_test.json.gz"
            ),
            response_sha256="b" * 64,
            code_version=normalized_code_version,
        )

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

    def test_default_execution_collects_only_one_chunk(
        self,
    ) -> None:
        """La limite par défaut doit empêcher un second appel."""
        with patch(
            "src.backfill_service.run_schedule_ingestion",
            side_effect=self.build_fake_ingestion_result,
        ) as mocked_ingestion:
            execution = execute_backfill(
                start_date=date(2026, 1, 1),
                end_date=date(2026, 2, 1),
                database_path=self.database_path,
                data_directory=self.data_directory,
                code_version="test-backfill",
            )

        self.assertEqual(execution.preview.pending_chunks, 2)
        self.assertEqual(execution.executed_chunks, 1)
        self.assertEqual(execution.remaining_pending_chunks, 1)
        self.assertEqual(execution.games_received, 5)
        self.assertEqual(execution.games_saved, 5)
        self.assertEqual(mocked_ingestion.call_count, 1)

        call_arguments = mocked_ingestion.call_args.kwargs
        self.assertEqual(
            call_arguments["start_date"],
            date(2026, 1, 1),
        )
        self.assertEqual(
            call_arguments["end_date"],
            date(2026, 1, 31),
        )

    def test_explicit_limit_can_collect_two_chunks(self) -> None:
        """Une limite explicite doit autoriser exactement deux lots."""
        with patch(
            "src.backfill_service.run_schedule_ingestion",
            side_effect=self.build_fake_ingestion_result,
        ) as mocked_ingestion:
            execution = execute_backfill(
                start_date=date(2026, 1, 1),
                end_date=date(2026, 2, 1),
                max_chunks=2,
                database_path=self.database_path,
                data_directory=self.data_directory,
                code_version="test-backfill",
            )

        self.assertEqual(execution.executed_chunks, 2)
        self.assertEqual(execution.remaining_pending_chunks, 0)
        self.assertEqual(execution.games_received, 10)
        self.assertEqual(execution.games_saved, 10)
        self.assertEqual(mocked_ingestion.call_count, 2)

        collected_starts = [
            mocked_call.kwargs["start_date"]
            for mocked_call in mocked_ingestion.call_args_list
        ]
        self.assertEqual(
            collected_starts,
            [
                date(2026, 1, 1),
                date(2026, 2, 1),
            ],
        )

    def test_execution_skips_success_and_collects_next_chunk(
        self,
    ) -> None:
        """La limite doit compter les appels, pas les lots ignorés."""
        first_start = date(2026, 1, 1)
        first_end = date(2026, 1, 31)
        self.record_run(
            start_date=first_start,
            end_date=first_end,
            status="success",
        )

        with patch(
            "src.backfill_service.run_schedule_ingestion",
            side_effect=self.build_fake_ingestion_result,
        ) as mocked_ingestion:
            execution = execute_backfill(
                start_date=first_start,
                end_date=date(2026, 2, 1),
                database_path=self.database_path,
                data_directory=self.data_directory,
            )

        self.assertEqual(execution.preview.skipped_chunks, 1)
        self.assertEqual(execution.preview.pending_chunks, 1)
        self.assertEqual(execution.executed_chunks, 1)
        self.assertEqual(execution.remaining_pending_chunks, 0)
        self.assertEqual(mocked_ingestion.call_count, 1)
        self.assertEqual(
            mocked_ingestion.call_args.kwargs["start_date"],
            date(2026, 2, 1),
        )

    def test_execution_does_nothing_when_all_chunks_succeeded(
        self,
    ) -> None:
        """Aucun appel ne doit partir lorsque tout est déjà réussi."""
        selected_date = date(2026, 8, 28)
        self.record_run(
            start_date=selected_date,
            end_date=selected_date,
            status="success",
        )

        with patch(
            "src.backfill_service.run_schedule_ingestion"
        ) as mocked_ingestion:
            execution = execute_backfill(
                start_date=selected_date,
                end_date=selected_date,
                database_path=self.database_path,
                data_directory=self.data_directory,
            )

        mocked_ingestion.assert_not_called()
        self.assertEqual(execution.executed_chunks, 0)
        self.assertEqual(execution.remaining_pending_chunks, 0)
        self.assertEqual(execution.games_received, 0)
        self.assertEqual(execution.games_saved, 0)

    def test_invalid_execution_limits_are_rejected(self) -> None:
        """Une limite invalide doit être refusée avant tout appel."""
        invalid_limits = (0, -1, True, 1.5)

        for invalid_limit in invalid_limits:
            with self.subTest(invalid_limit=invalid_limit):
                with patch(
                    "src.backfill_service.run_schedule_ingestion"
                ) as mocked_ingestion:
                    with self.assertRaises(ValueError):
                        execute_backfill(
                            start_date=date(2026, 8, 1),
                            end_date=date(2026, 8, 2),
                            max_chunks=invalid_limit,
                            database_path=self.database_path,
                            data_directory=self.data_directory,
                        )

                mocked_ingestion.assert_not_called()


if __name__ == "__main__":
    unittest.main()