from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.backfill_service import (
    ACTION_COLLECT,
    ACTION_SKIP,
    build_backfill_preview,
)
from src.ingestion_repository import (
    mark_ingestion_success,
    start_ingestion_run,
)
from src.ingestion_service import (
    ARCHIVE_SOURCE,
    INGESTION_SOURCE,
    build_schedule_request_parameters,
)
from src.raw_archive import archive_raw_response


class BackfillArchiveResumeTests(unittest.TestCase):
    """Vérifie que la reprise exige une archive réellement intacte."""

    START_DATE = date(2026, 8, 27)
    END_DATE = date(2026, 8, 27)
    GAME_TYPES = ("R",)

    def setUp(self) -> None:
        """Crée une base et un dossier data temporaires."""
        self.temporary_directory = TemporaryDirectory()
        temporary_path = Path(self.temporary_directory.name)
        self.data_directory = temporary_path / "data"
        self.database_path = (
            self.data_directory / "archive_resume_test.db"
        )
        self.raw_content = (
            b'{"dates":[],"totalGames":7,"test":"resume"}'
        )

    def tearDown(self) -> None:
        """Supprime l’environnement temporaire."""
        self.temporary_directory.cleanup()

    def start_run(self) -> int:
        """Crée le journal initial du lot contrôlé."""
        request_parameters = build_schedule_request_parameters(
            start_date=self.START_DATE,
            end_date=self.END_DATE,
            game_types=self.GAME_TYPES,
        )

        return start_ingestion_run(
            source=INGESTION_SOURCE,
            start_date=self.START_DATE,
            end_date=self.END_DATE,
            game_types=self.GAME_TYPES,
            request_parameters=request_parameters,
            code_version="archive-resume-test",
            database_path=self.database_path,
        )

    def create_archive(self):
        """Crée une archive brute valide."""
        return archive_raw_response(
            raw_content=self.raw_content,
            source_name=ARCHIVE_SOURCE,
            start_date=self.START_DATE,
            end_date=self.END_DATE,
            data_directory=self.data_directory,
        )

    def finish_run(
        self,
        *,
        run_id: int,
        raw_response_path: str,
        response_sha256: str,
    ) -> None:
        """Clôture le journal avec les références fournies."""
        mark_ingestion_success(
            run_id=run_id,
            records_received=7,
            records_saved=7,
            raw_response_path=raw_response_path,
            response_sha256=response_sha256,
            database_path=self.database_path,
        )

    def build_preview(self):
        """Construit l’aperçu isolé du lot."""
        return build_backfill_preview(
            start_date=self.START_DATE,
            end_date=self.END_DATE,
            database_path=self.database_path,
            data_directory=self.data_directory,
        )

    def test_intact_archive_allows_skip(self) -> None:
        """Une réussite avec archive intacte doit être ignorée."""
        run_id = self.start_run()
        archive = self.create_archive()
        self.finish_run(
            run_id=run_id,
            raw_response_path=archive.relative_path,
            response_sha256=archive.sha256,
        )

        preview = self.build_preview()
        plan = preview.chunk_plans[0]

        self.assertEqual(preview.skipped_chunks, 1)
        self.assertEqual(preview.pending_chunks, 0)
        self.assertEqual(plan.action, ACTION_SKIP)
        self.assertEqual(plan.successful_run_id, run_id)
        self.assertIn("archive vérifiée", plan.reason)

    def test_missing_archive_forces_collection(self) -> None:
        """Une archive absente doit remettre le lot en attente."""
        run_id = self.start_run()
        self.finish_run(
            run_id=run_id,
            raw_response_path=(
                "data/raw/mlb_schedule/2026/missing.json.gz"
            ),
            response_sha256="a" * 64,
        )

        preview = self.build_preview()
        plan = preview.chunk_plans[0]

        self.assertEqual(preview.skipped_chunks, 0)
        self.assertEqual(preview.pending_chunks, 1)
        self.assertEqual(plan.action, ACTION_COLLECT)
        self.assertEqual(plan.successful_run_id, run_id)
        self.assertIn("Archive absente", plan.reason)

    def test_corrupted_archive_forces_collection(self) -> None:
        """Une archive illisible doit remettre le lot en attente."""
        run_id = self.start_run()
        archive = self.create_archive()
        self.finish_run(
            run_id=run_id,
            raw_response_path=archive.relative_path,
            response_sha256=archive.sha256,
        )
        archive.absolute_path.write_bytes(
            b"archive volontairement corrompue"
        )

        preview = self.build_preview()
        plan = preview.chunk_plans[0]

        self.assertEqual(preview.skipped_chunks, 0)
        self.assertEqual(preview.pending_chunks, 1)
        self.assertEqual(plan.action, ACTION_COLLECT)
        self.assertEqual(plan.successful_run_id, run_id)
        self.assertIn("Archive illisible", plan.reason)

    def test_wrong_journal_hash_forces_collection(self) -> None:
        """Un SHA-256 incohérent doit remettre le lot en attente."""
        run_id = self.start_run()
        archive = self.create_archive()
        self.finish_run(
            run_id=run_id,
            raw_response_path=archive.relative_path,
            response_sha256="0" * 64,
        )

        preview = self.build_preview()
        plan = preview.chunk_plans[0]

        self.assertEqual(preview.skipped_chunks, 0)
        self.assertEqual(preview.pending_chunks, 1)
        self.assertEqual(plan.action, ACTION_COLLECT)
        self.assertEqual(plan.successful_run_id, run_id)
        self.assertIn(
            "ne correspond pas au journal",
            plan.reason,
        )


if __name__ == "__main__":
    unittest.main()