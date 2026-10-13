"""Tests de l'archive locale MLB 2027, sans publication distante."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.mlb_2027_candidate_archive import (
    MLB2027ArchiveError,
    archive_reference_candidate,
    verify_candidate_archive,
)
from src.mlb_2027_observed_source import verify_2027_observed_source
from src.mlb_2027_reference_predictor import calculate_reference_probabilities
from tests.test_mlb_2027_observed_source import TARGET, _observed


def _candidate():
    source = verify_2027_observed_source(TARGET, (_observed(),))
    calculation = calculate_reference_probabilities(
        TARGET, source.upcoming_games, source.prior_final_games,
        source_receipt_sha256=source.receipt_sha256,
    )
    return source, calculation


ISSUED = datetime(2027, 1, 20, 20, tzinfo=timezone.utc)


class CandidateArchiveTests(unittest.TestCase):
    def test_archives_raw_response_receipt_and_abstention(self) -> None:
        source, calculation = _candidate()
        with TemporaryDirectory() as folder:
            archive = archive_reference_candidate(
                Path(folder), source, calculation, issued_at_utc=ISSUED,
            )
            manifest = verify_candidate_archive(
                archive.path, expected_manifest_sha256=archive.manifest_sha256,
            )
            self.assertEqual(manifest["status"], "LOCAL_ONLY_UNPUBLISHED")
            self.assertEqual(
                (archive.path / "mlb-response-001.json").read_bytes(),
                source.observations[0].raw_content,
            )
            self.assertEqual(
                (archive.path / "source-receipt.json").read_bytes(),
                source.receipt_bytes,
            )
            self.assertIn(b'"abstained_count":1', (archive.path / "candidate.json").read_bytes())
            self.assertFalse((archive.path / "PUBLISHED").exists())

    def test_rejects_reuse_without_overwriting(self) -> None:
        source, calculation = _candidate()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            first = archive_reference_candidate(
                root, source, calculation, issued_at_utc=ISSUED,
            )
            original = (first.path / "candidate.json").read_bytes()
            with self.assertRaises(MLB2027ArchiveError):
                archive_reference_candidate(
                    root, source, calculation, issued_at_utc=ISSUED,
                )
            self.assertEqual((first.path / "candidate.json").read_bytes(), original)

    def test_rejects_tamper_and_missing_completion(self) -> None:
        source, calculation = _candidate()
        with TemporaryDirectory() as folder:
            archive = archive_reference_candidate(
                Path(folder), source, calculation, issued_at_utc=ISSUED,
            )
            (archive.path / "mlb-response-001.json").write_bytes(b"{}")
            with self.assertRaises(MLB2027ArchiveError):
                verify_candidate_archive(
                    archive.path, expected_manifest_sha256=archive.manifest_sha256,
                )
        with TemporaryDirectory() as folder:
            archive = archive_reference_candidate(
                Path(folder), source, calculation, issued_at_utc=ISSUED,
            )
            (archive.path / "COMPLETED").unlink()
            with self.assertRaises(MLB2027ArchiveError):
                verify_candidate_archive(
                    archive.path, expected_manifest_sha256=archive.manifest_sha256,
                )

    def test_rejects_late_or_source_before_capture(self) -> None:
        source, calculation = _candidate()
        with TemporaryDirectory() as folder:
            with self.assertRaises(MLB2027ArchiveError):
                archive_reference_candidate(
                    Path(folder), source, calculation,
                    issued_at_utc=datetime(2027, 1, 20, 22, 1, tzinfo=timezone.utc),
                )
            with self.assertRaises(MLB2027ArchiveError):
                archive_reference_candidate(
                    Path(folder), source, calculation,
                    issued_at_utc=datetime(2027, 1, 20, 18, tzinfo=timezone.utc),
                )
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_rejects_changed_calculation(self) -> None:
        source, calculation = _candidate()
        with TemporaryDirectory() as folder:
            forged = replace(calculation, probabilities=())
            # Le calcul de ce jeu d'essai s'abstient déjà. Modifier l'état de préparation.
            forged = replace(
                forged, readiness=replace(
                    forged.readiness, source_receipt_sha256="0" * 64,
                ),
            )
            with self.assertRaises(MLB2027ArchiveError):
                archive_reference_candidate(
                    Path(folder), source, forged, issued_at_utc=ISSUED,
                )


if __name__ == "__main__":
    unittest.main()
