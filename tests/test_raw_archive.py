"""Tests de l’archivage des réponses brutes MLB."""

from datetime import date
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.raw_archive import (
    archive_raw_response,
    load_raw_archive,
)


class RawArchiveTests(unittest.TestCase):
    """Vérifie les archives dans un dossier temporaire."""

    def setUp(self) -> None:
        """Crée un dossier de données isolé."""
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)

        self.data_directory = (
            Path(self.temporary_directory.name) / "data"
        )

    def test_raw_response_is_archived_and_reloaded(self) -> None:
        """Le contenu archivé doit rester strictement identique."""
        raw_content = (
            b'{"dates":[{"date":"2026-08-28","games":[]}]}'
        )
        expected_sha256 = sha256(raw_content).hexdigest()

        first_archive = archive_raw_response(
            raw_content=raw_content,
            source_name="mlb_schedule",
            start_date=date(2026, 8, 28),
            end_date=date(2026, 8, 28),
            data_directory=self.data_directory,
        )

        second_archive = archive_raw_response(
            raw_content=raw_content,
            source_name="mlb_schedule",
            start_date=date(2026, 8, 28),
            end_date=date(2026, 8, 28),
            data_directory=self.data_directory,
        )

        self.assertEqual(
            first_archive.absolute_path,
            second_archive.absolute_path,
        )
        self.assertEqual(first_archive.sha256, expected_sha256)
        self.assertTrue(first_archive.absolute_path.exists())
        self.assertGreater(
            first_archive.compressed_size_bytes,
            0,
        )
        self.assertEqual(
            load_raw_archive(first_archive.absolute_path),
            raw_content,
        )

        archive_files = list(
            self.data_directory.rglob("*.json.gz")
        )
        self.assertEqual(len(archive_files), 1)

    def test_empty_response_is_rejected(self) -> None:
        """Une réponse vide ne doit jamais être archivée."""
        with self.assertRaises(ValueError):
            archive_raw_response(
                raw_content=b"",
                source_name="mlb_schedule",
                start_date=date(2026, 8, 28),
                end_date=date(2026, 8, 28),
                data_directory=self.data_directory,
            )


if __name__ == "__main__":
    unittest.main()