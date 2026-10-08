"""NHL-34: cumulative evidence coverage never implies prediction accuracy."""
from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from src.nhl.prospective_coverage import (
    NHLProspectiveCoverageError, ProspectiveEvidenceDay,
    audit_prospective_coverage,
)

MODULE = "src.nhl.prospective_coverage"


def _checkpoint(date: str, digest: str, game_ids: tuple[int, ...]) -> dict:
    return {
        "target_date": date, "proof_sha256": digest,
        "games": [{"game_id": game_id} for game_id in game_ids],
    }


def _manifest(date: str, checkpoint_sha: str, final: int,
              game_ids: tuple[int, ...]) -> dict:
    return {
        "target_date": date, "checkpoint_sha256": checkpoint_sha,
        "report_sha256": "f" * 64, "game_count": len(game_ids),
        "verified_final_count": final, "pending_count": len(game_ids) - final,
        "games": [
            {"game_id": game_id, "status": (
                "FINAL_VERIFIED" if index < final else "PENDING_OFFICIAL_FINAL")}
            for index, game_id in enumerate(game_ids)
        ],
    }


class ProspectiveCoverageTests(TestCase):
    def test_complete_and_unreconciled_days_are_counted_without_accuracy(self) -> None:
        first = _checkpoint("2026-10-07", "a" * 64, (53, 54, 55))
        second = _checkpoint("2026-10-08", "b" * 64, (56, 57))
        manifest = _manifest("2026-10-07", first["proof_sha256"], 3, (53, 54, 55))
        proofs = {"first.json": first, "second.json": second}
        with patch(MODULE + ".verify_checkpoint", side_effect=lambda path: proofs[path.name]), \
             patch(MODULE + ".verify_postgame_reconciliation", return_value=manifest) as verify:
            days = (
                ProspectiveEvidenceDay(Path("second.json")),
                ProspectiveEvidenceDay(Path("first.json"), Path("final.json"),
                                       (Path("capture-1"),)),
            )
            report = audit_prospective_coverage(days)
            repeated = audit_prospective_coverage(reversed(days))
        self.assertEqual((report.game_count, report.verified_final_count,
                          report.pending_count), (5, 3, 2))
        self.assertEqual(report.days[0].target_date, "2026-10-07")
        self.assertEqual(report.days[1].pending_game_ids, (56, 57))
        self.assertEqual(report.audit_sha256, repeated.audit_sha256)
        self.assertEqual(verify.call_count, 2)
        self.assertFalse(report.training_permitted)
        self.assertFalse(report.prediction_publication_permitted)
        self.assertFalse(hasattr(report, "success_rate"))

    def test_duplicate_game_across_dates_is_rejected(self) -> None:
        proofs = {
            "first.json": _checkpoint("2026-10-07", "a" * 64, (53,)),
            "second.json": _checkpoint("2026-10-08", "b" * 64, (53,)),
        }
        with patch(MODULE + ".verify_checkpoint", side_effect=lambda path: proofs[path.name]):
            with self.assertRaises(NHLProspectiveCoverageError):
                audit_prospective_coverage((
                    ProspectiveEvidenceDay(Path("first.json")),
                    ProspectiveEvidenceDay(Path("second.json")),
                ))

    def test_capture_without_verified_manifest_is_rejected(self) -> None:
        with patch(MODULE + ".verify_checkpoint", return_value=_checkpoint(
                "2026-10-07", "a" * 64, (53,))):
            with self.assertRaises(NHLProspectiveCoverageError):
                audit_prospective_coverage((ProspectiveEvidenceDay(
                    Path("first.json"), capture_slots=(Path("capture"),)),))

    def test_conflicting_manifest_is_rejected(self) -> None:
        checkpoint = _checkpoint("2026-10-07", "a" * 64, (53,))
        manifest = _manifest("2026-10-07", "b" * 64, 1, (53,))
        with patch(MODULE + ".verify_checkpoint", return_value=checkpoint), \
             patch(MODULE + ".verify_postgame_reconciliation", return_value=manifest):
            with self.assertRaises(NHLProspectiveCoverageError):
                audit_prospective_coverage((ProspectiveEvidenceDay(
                    Path("first.json"), Path("final.json")),))

    def test_noncanonical_date_is_rejected(self) -> None:
        checkpoint = _checkpoint("20261007", "a" * 64, (53,))
        with patch(MODULE + ".verify_checkpoint", return_value=checkpoint):
            with self.assertRaises(NHLProspectiveCoverageError):
                audit_prospective_coverage((
                    ProspectiveEvidenceDay(Path("first.json")),))

    def test_empty_audit_is_rejected(self) -> None:
        with self.assertRaises(NHLProspectiveCoverageError):
            audit_prospective_coverage(())
