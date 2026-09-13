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
from unittest import mock
import zlib

from src import backup_service
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

    def test_active_odds_ingestion_is_rejected(self) -> None:
        """Une collecte de cotes en cours bloque aussi la sauvegarde."""
        self.connection.execute(
            """
            CREATE TABLE odds_ingestion_runs (
                run_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            INSERT INTO odds_ingestion_runs
            VALUES (1, 'started')
            """
        )
        self.connection.commit()

        with self.assertRaisesRegex(BackupError, "cotes"):
            self.create_backup()

    def test_zlib_error_during_member_listing_is_wrapped(self) -> None:
        """L'erreur observee dans getmembers devient toujours BackupError."""
        result = self.create_backup()
        content = result.absolute_path.read_bytes()
        error = zlib.error("invalid distance too far back")
        with (
            mock.patch.object(tarfile.TarFile, "getmembers", side_effect=error) as members,
            mock.patch.object(backup_service, "_inspect_database") as inspect_database,
        ):
            with self.assertRaisesRegex(BackupError, "Vérification de l'archive") as caught:
                verify_backup_bundle(result.absolute_path)
            members.assert_called_once()
            inspect_database.assert_not_called()
        self.assertIs(caught.exception.__cause__, error)
        self.assertEqual(result.absolute_path.read_bytes(), content)

    def test_zlib_error_during_manifest_read_is_wrapped(self) -> None:
        """La meme erreur tardive dans le manifeste reste controlee."""
        result = self.create_backup()
        error = zlib.error("invalid block type")
        with mock.patch.object(backup_service, "_load_manifest", side_effect=error) as read:
            with self.assertRaises(BackupError) as caught:
                verify_backup_bundle(result.absolute_path)
            read.assert_called_once()
        self.assertIs(caught.exception.__cause__, error)

    def test_zlib_error_during_member_stream_read_is_wrapped(self) -> None:
        """La lecture des donnees ne laisse pas echapper une erreur zlib."""
        result = self.create_backup()
        content = result.absolute_path.read_bytes()
        original_hash_stream = backup_service._hash_stream
        error = zlib.error("invalid distance code")
        member_streams = []

        def hash_stream(stream, destination=None):
            if isinstance(stream, tarfile.ExFileObject):
                member_streams.append(stream)
                raise error
            return original_hash_stream(stream, destination)

        with mock.patch.object(backup_service, "_hash_stream", side_effect=hash_stream):
            with self.assertRaises(BackupError) as caught:
                verify_backup_bundle(result.absolute_path)
        self.assertIs(caught.exception.__cause__, error)
        self.assertEqual(len(member_streams), 1)
        self.assertTrue(member_streams[0].closed)
        self.assertEqual(result.absolute_path.read_bytes(), content)

    def test_real_invalid_deflate_backup_is_rejected(self) -> None:
        """Un bloc DEFLATE de type reserve fournit une corruption deterministe."""
        # En-tete gzip valide, bloc BFINAL=1/BTYPE=3 interdit, trailer factice.
        corrupt = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\x07" + b"\x00" * 8
        with self.assertRaises(zlib.error):
            gzip.decompress(corrupt)
        self.backup_directory.mkdir(parents=True)
        path = self.backup_directory / "invalid-deflate.tar.gz"
        path.write_bytes(corrupt)
        with self.assertRaises(BackupError):
            verify_backup_bundle(path)
        self.assertEqual(path.read_bytes(), corrupt)

    def test_real_invalid_deflate_raw_is_rejected_without_publication(self) -> None:
        """Une archive raw avec en-tete valide mais flux casse reste refusee."""
        corrupt = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\x07" + b"\x00" * 8
        path = self.raw_directory / "invalid-deflate.json.gz"
        path.write_bytes(corrupt)
        with self.assertRaisesRegex(BackupError, "gzip invalide") as caught:
            self.create_backup()
        self.assertIsInstance(caught.exception.__cause__, zlib.error)
        self.assertEqual(path.read_bytes(), corrupt)
        self.assertEqual(list(self.backup_directory.glob("*.tar.gz")), [])

    def test_existing_backup_error_is_not_rewrapped(self) -> None:
        """La correction conserve les erreurs metier existantes."""
        result = self.create_backup()
        error = BackupError("synthetic validation error")
        with mock.patch.object(tarfile.TarFile, "getmembers", side_effect=error):
            with self.assertRaises(BackupError) as caught:
                verify_backup_bundle(result.absolute_path)
        self.assertIs(caught.exception, error)

    def test_unrelated_programming_errors_are_not_hidden(self) -> None:
        """L'interception reste ciblee : pas de except Exception general."""
        result = self.create_backup()
        error = RuntimeError("synthetic programming error")
        with mock.patch.object(tarfile.TarFile, "getmembers", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                verify_backup_bundle(result.absolute_path)
        self.assertIs(caught.exception, error)

    def test_interruptions_are_not_converted_to_backup_errors(self) -> None:
        """Une interruption utilisateur n'est ni masquee ni retentee."""
        result = self.create_backup()
        for error in (KeyboardInterrupt(), SystemExit(1)):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(tarfile.TarFile, "getmembers", side_effect=error) as members:
                    with self.assertRaises(type(error)) as caught:
                        verify_backup_bundle(result.absolute_path)
                    members.assert_called_once()
                self.assertIs(caught.exception, error)

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
