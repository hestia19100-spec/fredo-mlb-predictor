"""Tests du service de sauvegarde locale."""

from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import sqlite3
import tarfile
from tempfile import TemporaryDirectory
import unittest

from src.backup_service import (
    BackupError,
    create_backup_bundle,
    verify_backup_bundle,
)


class BackupServiceTests(unittest.TestCase):
    """Contrôle la création et la vérification des sauvegardes."""

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.project_directory = Path(
            self.temporary_directory.name
        )

        self.data_directory = (
            self.project_directory
            / "data"
        )

        self.raw_directory = (
            self.data_directory
            / "raw"
            / "mlb_schedule"
            / "2021"
        )

        self.backup_directory = (
            self.project_directory
            / "backups"
        )

        self.database_path = (
            self.data_directory
            / "fredo_mlb.db"
        )

        self.raw_directory.mkdir(
            parents=True
        )

        self.connection = sqlite3.connect(
            self.database_path
        )

        self.connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        self.connection.execute(
            "PRAGMA wal_autocheckpoint = 0"
        )

        self.connection.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY
            )
            """
        )

        self.connection.execute(
            """
            CREATE TABLE ingestion_runs (
                run_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL
            )
            """
        )

        self.connection.execute(
            """
            CREATE TABLE games (
                game_id INTEGER PRIMARY KEY,
                name TEXT
            )
            """
        )

        self.connection.execute(
            """
            INSERT INTO schema_migrations
            VALUES (3)
            """
        )

        self.connection.execute(
            """
            INSERT INTO schema_migrations
            VALUES (4)
            """
        )

        self.connection.execute(
            """
            INSERT INTO games
            VALUES (1, 'Test Game')
            """
        )

        self.connection.commit()

        for index in (1, 2):
            archive_path = (
                self.raw_directory
                / f"sample_{index}.json.gz"
            )

            with gzip.open(
                archive_path,
                "wb",
            ) as stream:
                stream.write(
                    json.dumps(
                        {"index": index}
                    ).encode("utf-8")
                )

        self.created_at = datetime(
            2026,
            8,
            29,
            15,
            30,
            tzinfo=timezone.utc,
        )

    def tearDown(self) -> None:
        self.connection.close()
        self.temporary_directory.cleanup()

    def create_backup(self):
        """Crée une sauvegarde dans le dossier temporaire."""
        return create_backup_bundle(
            database_path=self.database_path,
            data_directory=self.data_directory,
            backup_directory=self.backup_directory,
            code_version="abc123",
            created_at_utc=self.created_at,
        )

    def test_backup_round_trip_includes_database_and_raw_files(
        self,
    ) -> None:
        """La sauvegarde doit contenir SQLite et les archives raw."""
        result = self.create_backup()

        verification = verify_backup_bundle(
            result.absolute_path
        )

        self.assertTrue(
            result.absolute_path.is_file()
        )

        self.assertEqual(
            verification.file_count,
            3,
        )

        self.assertEqual(
            verification.raw_archive_count,
            2,
        )

        self.assertEqual(
            verification.code_version,
            "abc123",
        )

        self.assertEqual(
            verification.archive_sha256,
            result.archive_sha256,
        )

        with tarfile.open(
            result.absolute_path,
            "r:gz",
        ) as archive:
            names = set(
                archive.getnames()
            )

        self.assertEqual(
            names,
            {
                "manifest.json",
                "data/fredo_mlb.db",
                (
                    "data/raw/mlb_schedule/2021/"
                    "sample_1.json.gz"
                ),
                (
                    "data/raw/mlb_schedule/2021/"
                    "sample_2.json.gz"
                ),
            },
        )

    def test_corrupted_backup_is_rejected(
        self,
    ) -> None:
        """Une archive altérée ne doit jamais être déclarée valide."""
        result = self.create_backup()

        content = bytearray(
            result.absolute_path.read_bytes()
        )

        content[len(content) // 2] ^= 0xFF

        result.absolute_path.write_bytes(
            content
        )

        with self.assertRaises(
            BackupError
        ):
            verify_backup_bundle(
                result.absolute_path
            )

    def test_active_ingestion_is_rejected(
        self,
    ) -> None:
        """Une collecte en cours doit bloquer la sauvegarde."""
        self.connection.execute(
            """
            INSERT INTO ingestion_runs
            VALUES (1, 'started')
            """
        )

        self.connection.commit()

        with self.assertRaisesRegex(
            BackupError,
            "encore en cours",
        ):
            self.create_backup()

    def test_invalid_raw_gzip_is_rejected(
        self,
    ) -> None:
        """Une fausse archive raw ne doit pas être sauvegardée."""
        invalid_path = (
            self.raw_directory
            / "invalid.json.gz"
        )

        invalid_path.write_bytes(
            b"not gzip"
        )

        with self.assertRaisesRegex(
            BackupError,
            "gzip invalide",
        ):
            self.create_backup()

    def test_existing_backup_is_never_overwritten(
        self,
    ) -> None:
        """Une sauvegarde existante ne doit jamais être remplacée."""
        first_result = self.create_backup()

        first_content = (
            first_result
            .absolute_path
            .read_bytes()
        )

        with self.assertRaisesRegex(
            BackupError,
            "existe déjà",
        ):
            self.create_backup()

        self.assertEqual(
            first_result.absolute_path.read_bytes(),
            first_content,
        )

    def test_unsafe_tar_member_is_rejected(
        self,
    ) -> None:
        """Un chemin tentant de sortir de l’archive doit être refusé."""
        unsafe_path = (
            self.backup_directory
            / "unsafe.tar.gz"
        )

        self.backup_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        with tarfile.open(
            unsafe_path,
            "w:gz",
        ) as archive:
            content = b"unsafe"

            member = tarfile.TarInfo(
                "../outside.txt"
            )

            member.size = len(content)

            archive.addfile(
                member,
                io.BytesIO(content),
            )

        with self.assertRaisesRegex(
            BackupError,
            "Chemin interne",
        ):
            verify_backup_bundle(
                unsafe_path
            )


if __name__ == "__main__":
    unittest.main()