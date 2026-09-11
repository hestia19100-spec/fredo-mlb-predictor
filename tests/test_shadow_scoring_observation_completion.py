"""Tests de fermeture atomique des observations MLB."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import shadow_prediction as shadow
from src import shadow_scoring as scoring
from tests import test_shadow_scoring_observation_receipt as receipt_tests


COMPLETED_AT = datetime(2026, 9, 11, 6, 3, 0, tzinfo=timezone.utc)


class ShadowScoringObservationCompletionTests(unittest.TestCase):
    """COMPLETED est le sixieme fichier et ferme definitivement le creneau."""

    def _fixture(self, project: Path) -> dict[str, object]:
        helper = receipt_tests.ShadowScoringObservationReceiptTests(
            "test_exact_schema_hashes_counts_and_attestations"
        )
        fixture = helper._fixture(project)
        receipt_publication = scoring.publish_scoring_observation_receipt(
            fixture["authority"],
            fixture["reservation"],
            fixture["evidence_publication"],
            fixture["documents_publication"],
            started_at_utc=receipt_tests.STARTED,
            receipt_finalized_at_utc=receipt_tests.FINALIZED,
            project_directory=project,
        )
        fixture["receipt_publication"] = receipt_publication
        return fixture

    def _build(self, fixture: dict[str, object]):
        return scoring.build_scoring_observation_completion(
            fixture["authority"],
            fixture["receipt_publication"],
            completed_at_utc=COMPLETED_AT,
        )

    def _publish(self, fixture: dict[str, object]):
        return scoring.publish_scoring_observation_completion(
            fixture["authority"],
            fixture["receipt_publication"],
            completed_at_utc=COMPLETED_AT,
            project_directory=fixture["project"],
        )

    def test_exact_terminal_schema_and_receipt_link(self) -> None:
        """Le marqueur contient seulement l'identite, le recu et l'heure."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            completion = self._build(fixture)
            marker = completion.completed_marker
            self.assertEqual(frozenset(marker), scoring._COMPLETED_MARKER_KEYS)
            self.assertEqual(marker["marker_schema_version"], 1)
            self.assertEqual(
                marker["observation_id"], fixture["reservation"].observation_id
            )
            self.assertEqual(
                marker["observation_receipt_sha256"],
                fixture["receipt_publication"].receipt_sha256,
            )
            self.assertEqual(
                marker["observation_receipt_path"],
                (
                    "shadow_scores/"
                    "logistic_team_form_v1_platt_shadow_v2_2026_v1/"
                    "2026-09-10/observations/2026-09-11/"
                    "observation_receipt.json"
                ),
            )
            self.assertEqual(marker["completed_at_utc"], "2026-09-11T06:03:00Z")

    def test_bytes_are_canonical_deterministic_and_hash_exact(self) -> None:
        """La meme fermeture produit toujours les memes octets."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            first = self._build(fixture)
            second = self._build(fixture)
            self.assertEqual(first.canonical_json_bytes, second.canonical_json_bytes)
            self.assertEqual(
                first.canonical_json_bytes,
                shadow._canonical_json_file_bytes(first.completed_marker),
            )
            self.assertEqual(
                first.completed_sha256,
                hashlib.sha256(first.canonical_json_bytes).hexdigest(),
            )

    def test_receipt_contains_no_completed_back_reference(self) -> None:
        """Le sens des empreintes reste sans cycle."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            receipt = fixture["receipt_publication"].receipt.receipt
            self.assertNotIn("COMPLETED", json.dumps(receipt, sort_keys=True))
            completion = self._build(fixture)
            self.assertIn(
                fixture["receipt_publication"].receipt_sha256,
                completion.canonical_json_bytes.decode("utf-8"),
            )

    def test_completion_cannot_precede_receipt_finalization(self) -> None:
        """Un marqueur terminal ne peut pas antidater le recu."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            with self.assertRaisesRegex(
                scoring.ScoringAdjudicationError, "ne peut pas fermer"
            ):
                scoring.build_scoring_observation_completion(
                    fixture["authority"],
                    fixture["receipt_publication"],
                    completed_at_utc=receipt_tests.STARTED,
                )

    def test_forged_receipt_hash_or_size_is_rejected(self) -> None:
        """Une description mensongere du recu ne ferme rien."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            for replacement in (
                {"receipt_sha256": "0" * 64},
                {"receipt_size_bytes": 1},
            ):
                with self.subTest(replacement=replacement):
                    fixture["receipt_publication"] = replace(
                        fixture["receipt_publication"], **replacement
                    )
                    with self.assertRaisesRegex(
                        scoring.ScoringAdjudicationError,
                        "ne peut pas fermer",
                    ):
                        self._build(fixture)
                    fixture = self._fixture(Path(temporary) / replacement.popitem()[0])

    def test_exact_sixth_file_is_published(self) -> None:
        """COMPLETED termine seul la chaine de succes."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            publication = self._publish(fixture)
            self.assertEqual(publication.completed_path.name, "COMPLETED")
            self.assertEqual(
                publication.completed_path.read_bytes(),
                publication.completion.canonical_json_bytes,
            )
            self.assertEqual(
                sorted(path.name for path in fixture["slot"].iterdir()),
                [
                    scoring.COMPLETED_FILENAME,
                    "RESERVED",
                    scoring.ADJUDICATIONS_FILENAME,
                    scoring.DAILY_REPORT_FILENAME,
                    scoring.OBSERVATION_RECEIPT_FILENAME,
                    scoring.OUTCOME_EVIDENCE_FILENAME,
                ],
            )

    def test_publication_preserves_all_five_predecessors(self) -> None:
        """Fermer une observation ne modifie aucun octet anterieur."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            before = {
                path.name: path.read_bytes() for path in fixture["slot"].iterdir()
            }
            self._publish(fixture)
            after = {
                name: (fixture["slot"] / name).read_bytes() for name in before
            }
            self.assertEqual(after, before)

    def test_second_publication_never_overwrites_or_repairs(self) -> None:
        """Une observation terminee ne peut jamais etre rouverte."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            publication = self._publish(fixture)
            before = publication.completed_path.read_bytes()
            with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                self._publish(fixture)
            self.assertEqual(publication.completed_path.read_bytes(), before)

    def test_concurrent_publishers_have_exactly_one_winner(self) -> None:
        """Deux processus ne ferment jamais deux fois la meme observation."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))

            def attempt(_: int) -> str:
                try:
                    self._publish(fixture)
                except shadow.ShadowPredictionError:
                    return "LOST"
                return "WON"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(attempt, range(2)))
            self.assertEqual(outcomes.count("WON"), 1)
            self.assertEqual(outcomes.count("LOST"), 1)
            self.assertTrue(
                (fixture["slot"] / scoring.COMPLETED_FILENAME).is_file()
            )

    def test_foreign_or_tampered_predecessor_blocks_completion(self) -> None:
        """Aucun intrus ni recu modifie n'est contourne."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            foreign = fixture["slot"] / "foreign.txt"
            foreign.write_bytes(b"keep")
            with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                self._publish(fixture)
            self.assertEqual(foreign.read_bytes(), b"keep")
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            fixture["receipt_publication"].receipt_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(shadow.ShadowPredictionError, "diverge"):
                self._publish(fixture)

    def test_failure_before_link_preserves_five_predecessors(self) -> None:
        """Une panne avant le lien ne supprime aucune preuve."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            before = {
                path.name: path.read_bytes() for path in fixture["slot"].iterdir()
            }
            with (
                mock.patch.object(
                    scoring.shadow,
                    "_publish_exclusive_verified",
                    side_effect=shadow.ShadowPredictionError("panne avant lien"),
                ),
                self.assertRaisesRegex(shadow.ShadowPredictionError, "avant lien"),
            ):
                self._publish(fixture)
            self.assertFalse(
                (fixture["slot"] / scoring.COMPLETED_FILENAME).exists()
            )
            self.assertEqual(
                {path.name: path.read_bytes() for path in fixture["slot"].iterdir()},
                before,
            )

    def test_failure_after_link_preserves_completed(self) -> None:
        """Une panne de relecture laisse le marqueur definitif en place."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            original = scoring._read_exact_local_file

            def fail_completed(path: Path, *, description: str) -> bytes:
                if path.name == scoring.COMPLETED_FILENAME:
                    raise shadow.ShadowPredictionError("panne apres lien")
                return original(path, description=description)

            with (
                mock.patch.object(
                    scoring, "_read_exact_local_file", side_effect=fail_completed
                ),
                self.assertRaisesRegex(shadow.ShadowPredictionError, "apres lien"),
            ):
                self._publish(fixture)
            completed = fixture["slot"] / scoring.COMPLETED_FILENAME
            self.assertTrue(completed.is_file())
            before = completed.read_bytes()
            with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                self._publish(fixture)
            self.assertEqual(completed.read_bytes(), before)

    def test_completion_has_no_network_model_prediction_or_adjudication_call(self) -> None:
        """La fermeture ne fait que verifier et publier des octets locaux."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            forbidden = AssertionError("appel interdit")
            with (
                mock.patch.object(scoring.requests, "get", side_effect=forbidden),
                mock.patch.object(
                    scoring,
                    "load_immutable_scoring_prediction_source",
                    side_effect=forbidden,
                ),
                mock.patch.object(
                    scoring, "adjudicate_scoring_predictions", side_effect=forbidden
                ),
                mock.patch.object(
                    scoring, "fetch_scoring_outcome_evidence", side_effect=forbidden
                ),
            ):
                publication = self._publish(fixture)
            self.assertTrue(publication.completed_path.is_file())


if __name__ == "__main__":
    unittest.main()
