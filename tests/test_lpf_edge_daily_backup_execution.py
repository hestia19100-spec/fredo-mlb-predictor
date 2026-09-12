"""Tests fermes du bouton de sauvegarde locale LPF Edge."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import backup_service
from src import lpf_edge_daily_operations as operations


HEAD = "a" * 40
ARCHIVE_BYTES = b"archive locale verifiee"
ARCHIVE_SHA256 = hashlib.sha256(ARCHIVE_BYTES).hexdigest()


class LPFEdgeDailyBackupExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.backups = self.project / "backups"
        self.backups.mkdir()
        self.archive = self.backups / "fredo-mlb-backup-20260913T080000Z.tar.gz"
        self.archive.write_bytes(ARCHIVE_BYTES)

    def _git(self, *, ready: bool = True) -> operations.GitWorkspaceState:
        return operations.GitWorkspaceState(
            available=True,
            branch="main",
            head_commit=HEAD,
            origin_main_commit=HEAD,
            clean=ready,
            synchronized=True,
        )

    def _result(self, **changes: object) -> backup_service.BackupResult:
        values = {
            "absolute_path": self.archive.resolve(),
            "relative_path": f"backups/{self.archive.name}",
            "archive_sha256": ARCHIVE_SHA256,
            "archive_size_bytes": len(ARCHIVE_BYTES),
            "file_count": 12,
            "raw_archive_count": 10,
            "code_version": HEAD,
        }
        values.update(changes)
        return backup_service.BackupResult(**values)

    def _verification(self, **changes: object) -> backup_service.BackupVerification:
        values = {
            "absolute_path": self.archive.resolve(),
            "archive_sha256": ARCHIVE_SHA256,
            "archive_size_bytes": len(ARCHIVE_BYTES),
            "file_count": 12,
            "raw_archive_count": 10,
            "code_version": HEAD,
            "created_at_utc": "2026-09-13T08:00:00+00:00",
        }
        values.update(changes)
        return backup_service.BackupVerification(**values)

    def _run(self) -> operations.DailyBackupPublication:
        return operations.execute_verified_local_backup(
            project_directory=self.project,
        )

    def test_success_creates_reverifies_and_loads_exact_archive(self) -> None:
        with (
            mock.patch.object(
                operations,
                "inspect_git_workspace",
                return_value=self._git(),
            ) as inspect_git,
            mock.patch.object(
                backup_service,
                "create_backup_bundle",
                return_value=self._result(),
            ) as create,
            mock.patch.object(
                backup_service,
                "verify_backup_bundle",
                return_value=self._verification(),
            ) as verify,
        ):
            result = self._run()

        inspect_git.assert_called_once_with(project_directory=self.project.resolve())
        create.assert_called_once_with(
            database_path=self.project / "data" / "fredo_mlb.db",
            data_directory=self.project / "data",
            backup_directory=self.project / "backups",
            code_version=HEAD,
        )
        verify.assert_called_once_with(self.archive.resolve())
        self.assertEqual(result.filename, self.archive.name)
        self.assertEqual(result.archive_bytes, ARCHIVE_BYTES)
        self.assertEqual(result.archive_sha256, ARCHIVE_SHA256)
        self.assertEqual(result.file_count, 12)

    def test_dirty_repository_stops_before_backup_service(self) -> None:
        with (
            mock.patch.object(
                operations,
                "inspect_git_workspace",
                return_value=self._git(ready=False),
            ),
            mock.patch.object(backup_service, "create_backup_bundle") as create,
        ):
            with self.assertRaisesRegex(
                operations.DailyBackupAutomationError,
                "depot doit etre propre",
            ) as caught:
                self._run()
        self.assertEqual(caught.exception.stage, operations.DailyBackupStage.PREFLIGHT)
        create.assert_not_called()

    def test_controlled_creation_failure_is_reported(self) -> None:
        with (
            mock.patch.object(
                operations,
                "inspect_git_workspace",
                return_value=self._git(),
            ),
            mock.patch.object(
                backup_service,
                "create_backup_bundle",
                side_effect=backup_service.BackupError("collecte active"),
            ),
        ):
            with self.assertRaisesRegex(
                operations.DailyBackupAutomationError,
                "collecte active",
            ) as caught:
                self._run()
        self.assertEqual(caught.exception.stage, operations.DailyBackupStage.CREATION)

    def test_foreign_creation_result_is_rejected_before_verification(self) -> None:
        with (
            mock.patch.object(
                operations,
                "inspect_git_workspace",
                return_value=self._git(),
            ),
            mock.patch.object(
                backup_service,
                "create_backup_bundle",
                return_value=mock.Mock(),
            ),
            mock.patch.object(backup_service, "verify_backup_bundle") as verify,
        ):
            with self.assertRaisesRegex(
                operations.DailyBackupAutomationError,
                "recu de sauvegarde inattendu",
            ):
                self._run()
        verify.assert_not_called()

    def test_second_verification_failure_is_reported(self) -> None:
        with (
            mock.patch.object(
                operations,
                "inspect_git_workspace",
                return_value=self._git(),
            ),
            mock.patch.object(
                backup_service,
                "create_backup_bundle",
                return_value=self._result(),
            ),
            mock.patch.object(
                backup_service,
                "verify_backup_bundle",
                side_effect=backup_service.BackupError("archive corrompue"),
            ),
        ):
            with self.assertRaises(operations.DailyBackupAutomationError) as caught:
                self._run()
        self.assertEqual(caught.exception.stage, operations.DailyBackupStage.VERIFICATION)

    def test_tampered_receipt_or_verification_is_rejected(self) -> None:
        for result, verification in (
            (self._result(archive_sha256="b" * 64), self._verification()),
            (self._result(), self._verification(file_count=11)),
            (self._result(relative_path="backups/intrus.tar.gz"), self._verification()),
        ):
            with self.subTest(result=result, verification=verification):
                with (
                    mock.patch.object(
                        operations,
                        "inspect_git_workspace",
                        return_value=self._git(),
                    ),
                    mock.patch.object(
                        backup_service,
                        "create_backup_bundle",
                        return_value=result,
                    ),
                    mock.patch.object(
                        backup_service,
                        "verify_backup_bundle",
                        return_value=verification,
                    ),
                ):
                    with self.assertRaisesRegex(
                        operations.DailyBackupAutomationError,
                        "divergent",
                    ):
                        self._run()

    def test_public_signature_exposes_only_project_directory(self) -> None:
        signature = inspect.signature(operations.execute_verified_local_backup)
        self.assertEqual(tuple(signature.parameters), ("project_directory",))


if __name__ == "__main__":
    unittest.main()
