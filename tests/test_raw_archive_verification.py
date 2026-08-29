from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.raw_archive import (
    RawArchiveError,
    archive_raw_response,
    verify_raw_archive,
)


class RawArchiveVerificationTests(unittest.TestCase):
    """Vérifie l’intégrité des archives utilisées pour la reprise."""

    def setUp(self) -> None:
        """Crée un dossier de données isolé."""
        self.temporary_directory = TemporaryDirectory()
        self.data_directory = (
            Path(self.temporary_directory.name) / "data"
        )
        self.raw_content = (
            b'{"dates":[],"totalGames":1,"test":"archive"}'
        )

    def tearDown(self) -> None:
        """Supprime les fichiers temporaires."""
        self.temporary_directory.cleanup()

    def create_archive(self):
        """Crée une archive valide pour les tests."""
        return archive_raw_response(
            raw_content=self.raw_content,
            source_name="mlb_schedule",
            start_date=date(2026, 8, 27),
            end_date=date(2026, 8, 27),
            data_directory=self.data_directory,
        )

    def test_valid_archive_is_verified(self) -> None:
        """Une archive intacte doit être reconnue."""
        archive = self.create_archive()

        verified_archive = verify_raw_archive(
            relative_path=archive.relative_path,
            expected_sha256=archive.sha256,
            data_directory=self.data_directory,
        )

        self.assertEqual(
            verified_archive.absolute_path,
            archive.absolute_path,
        )
        self.assertEqual(
            verified_archive.relative_path,
            archive.relative_path,
        )
        self.assertEqual(
            verified_archive.sha256,
            archive.sha256,
        )
        self.assertEqual(
            verified_archive.uncompressed_size_bytes,
            len(self.raw_content),
        )
        self.assertGreater(
            verified_archive.compressed_size_bytes,
            0,
        )

    def test_missing_archive_is_rejected(self) -> None:
        """Un fichier absent ne doit jamais être considéré réussi."""
        with self.assertRaises(RawArchiveError):
            verify_raw_archive(
                relative_path=(
                    "data/raw/mlb_schedule/2026/missing.json.gz"
                ),
                expected_sha256="a" * 64,
                data_directory=self.data_directory,
            )

    def test_wrong_sha256_is_rejected(self) -> None:
        """Une empreinte différente doit invalider l’archive."""
        archive = self.create_archive()

        with self.assertRaises(RawArchiveError):
            verify_raw_archive(
                relative_path=archive.relative_path,
                expected_sha256="0" * 64,
                data_directory=self.data_directory,
            )

    def test_corrupted_gzip_is_rejected(self) -> None:
        """Un fichier gzip illisible doit être refusé."""
        archive = self.create_archive()
        archive.absolute_path.write_bytes(
            b"contenu volontairement corrompu"
        )

        with self.assertRaises(RawArchiveError):
            verify_raw_archive(
                relative_path=archive.relative_path,
                expected_sha256=archive.sha256,
                data_directory=self.data_directory,
            )

    def test_unsafe_paths_are_rejected(self) -> None:
        """Un chemin ne doit jamais pouvoir sortir de data/raw."""
        unsafe_paths = (
            "../outside.json.gz",
            "data/raw/../../outside.json.gz",
            "/tmp/outside.json.gz",
            r"data\raw\outside.json.gz",
        )

        for unsafe_path in unsafe_paths:
            with self.subTest(unsafe_path=unsafe_path):
                with self.assertRaises(RawArchiveError):
                    verify_raw_archive(
                        relative_path=unsafe_path,
                        expected_sha256="a" * 64,
                        data_directory=self.data_directory,
                    )

    def test_malformed_sha256_is_rejected(self) -> None:
        """Une empreinte mal formée doit être refusée."""
        archive = self.create_archive()

        malformed_hashes = (
            "",
            "abc",
            "z" * 64,
        )

        for malformed_hash in malformed_hashes:
            with self.subTest(
                malformed_hash=malformed_hash
            ):
                with self.assertRaises(RawArchiveError):
                    verify_raw_archive(
                        relative_path=archive.relative_path,
                        expected_sha256=malformed_hash,
                        data_directory=self.data_directory,
                    )


if __name__ == "__main__":
    unittest.main()