"""Tests de l'historique local et en lecture seule des sauvegardes."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import backup_service
from src import lpf_edge_daily_operations as operations


ARCHIVE_BYTES = b"ancienne sauvegarde verifiee"
ARCHIVE_SHA256 = hashlib.sha256(ARCHIVE_BYTES).hexdigest()
CODE_VERSION = "a" * 40


class LPFEdgeBackupHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.backups = self.project / "backups"

    def _archive(self, timestamp: str = "20260913T080000Z") -> Path:
        self.backups.mkdir(exist_ok=True)
        archive = self.backups / f"fredo-mlb-backup-{timestamp}.tar.gz"
        archive.write_bytes(ARCHIVE_BYTES)
        return archive

    def _verification(
        self,
        archive: Path,
        **changes: object,
    ) -> backup_service.BackupVerification:
        values = {
            "absolute_path": archive.resolve(),
            "archive_sha256": ARCHIVE_SHA256,
            "archive_size_bytes": len(ARCHIVE_BYTES),
            "file_count": 20,
            "raw_archive_count": 18,
            "code_version": CODE_VERSION,
            "created_at_utc": "2026-09-13T08:00:00+00:00",
        }
        values.update(changes)
        return backup_service.BackupVerification(**values)

    def test_missing_directory_returns_empty_without_creating_it(self) -> None:
        self.assertEqual(
            operations.list_local_backup_names(project_directory=self.project),
            (),
        )
        self.assertFalse(self.backups.exists())

    def test_canonical_archives_are_listed_newest_first_without_verification(self) -> None:
        older = self._archive("20260911T060000Z")
        newer = self._archive("20260913T080000Z")
        (self.backups / "notes.txt").write_text("ignore", encoding="utf-8")
        (self.backups / "fredo-mlb-backup-invalid.tar.gz").write_bytes(b"ignore")
        with mock.patch.object(backup_service, "verify_backup_bundle") as verify:
            names = operations.list_local_backup_names(
                project_directory=self.project,
            )
        self.assertEqual(names, (newer.name, older.name))
        verify.assert_not_called()

    def test_selected_archive_is_verified_and_loaded_exactly(self) -> None:
        archive = self._archive()
        verification = self._verification(archive)
        with mock.patch.object(
            backup_service,
            "verify_backup_bundle",
            return_value=verification,
        ) as verify:
            result = operations.load_verified_local_backup(
                archive.name,
                project_directory=self.project,
            )
        verify.assert_called_once_with(archive)
        self.assertEqual(result.filename, archive.name)
        self.assertEqual(result.relative_path, f"backups/{archive.name}")
        self.assertEqual(result.archive_bytes, ARCHIVE_BYTES)
        self.assertEqual(result.archive_sha256, ARCHIVE_SHA256)
        self.assertEqual(result.created_at_utc, verification.created_at_utc)

    def test_noncanonical_or_traversal_name_is_rejected_before_read(self) -> None:
        self._archive()
        for filename in (
            "../fredo-mlb-backup-20260913T080000Z.tar.gz",
            "fredo-mlb-backup-invalid.tar.gz",
            "notes.txt",
        ):
            with self.subTest(filename=filename):
                with mock.patch.object(
                    backup_service,
                    "verify_backup_bundle",
                ) as verify:
                    with self.assertRaisesRegex(
                        operations.DailyBackupAutomationError,
                        "nom de sauvegarde",
                    ):
                        operations.load_verified_local_backup(
                            filename,
                            project_directory=self.project,
                        )
                verify.assert_not_called()

    def test_missing_selected_archive_is_rejected(self) -> None:
        self.backups.mkdir()
        filename = "fredo-mlb-backup-20260913T080000Z.tar.gz"
        with self.assertRaisesRegex(
            operations.DailyBackupAutomationError,
            "n'existe plus",
        ):
            operations.load_verified_local_backup(
                filename,
                project_directory=self.project,
            )

    def test_verification_error_is_reported_without_altering_archive(self) -> None:
        archive = self._archive()
        with mock.patch.object(
            backup_service,
            "verify_backup_bundle",
            side_effect=backup_service.BackupError("corrompue"),
        ):
            with self.assertRaisesRegex(
                operations.DailyBackupAutomationError,
                "ne peut pas etre verifiee",
            ):
                operations.load_verified_local_backup(
                    archive.name,
                    project_directory=self.project,
                )
        self.assertEqual(archive.read_bytes(), ARCHIVE_BYTES)

    def test_tampered_metadata_or_bytes_are_rejected(self) -> None:
        archive = self._archive()
        for verification in (
            self._verification(archive, archive_sha256="b" * 64),
            self._verification(archive, archive_size_bytes=1),
            self._verification(archive, code_version="unknown"),
            self._verification(archive, created_at_utc="date-invalide"),
        ):
            with self.subTest(verification=verification):
                with mock.patch.object(
                    backup_service,
                    "verify_backup_bundle",
                    return_value=verification,
                ):
                    with self.assertRaisesRegex(
                        operations.DailyBackupAutomationError,
                        "incoherente",
                    ):
                        operations.load_verified_local_backup(
                            archive.name,
                            project_directory=self.project,
                        )

    def test_public_signatures_expose_no_directory_or_verification_override(self) -> None:
        listing = inspect.signature(operations.list_local_backup_names)
        loading = inspect.signature(operations.load_verified_local_backup)
        self.assertEqual(tuple(listing.parameters), ("project_directory",))
        self.assertEqual(
            tuple(loading.parameters),
            ("filename", "project_directory"),
        )


if __name__ == "__main__":
    unittest.main()
