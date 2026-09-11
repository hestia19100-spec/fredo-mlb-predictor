"""Tests du recu immuable de chaque observation MLB."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import shadow_prediction as shadow
from src import shadow_scoring as scoring
from src import shadow_scoring_registration as registration


TARGET = "2026-09-10"
CHECKPOINT = "2026-09-11"
STARTED = datetime(2026, 9, 11, 6, 0, 0, tzinfo=timezone.utc)
FINALIZED = datetime(2026, 9, 11, 6, 2, 0, tzinfo=timezone.utc)


class ShadowScoringObservationReceiptTests(unittest.TestCase):
    """Le cinquieme fichier relie sans cycle toutes les preuves anterieures."""

    def _fixture(self, project: Path) -> dict[str, object]:
        protocol_path = (
            Path(__file__).resolve().parents[1]
            / scoring_registration_path()
        )
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        runtime_commit = "5" * 40
        authority = scoring.ScoringExecutionAuthority(
            protocol=protocol,
            registration={},
            scoring_protocol_sha256=(
                registration.EXPECTED_SCORING_PROTOCOL_SHA256
            ),
            protocol_introduction_commit="1" * 40,
            registration_commit="2" * 40,
            registration_sha256="3" * 64,
            registration_remote_evidence_sha256="4" * 64,
            runtime_code_commit=runtime_commit,
            scoring_engine_sha256="6" * 64,
        )
        slot = (
            project
            / scoring.SCORING_OUTPUT_ROOT_RELATIVE_PATH
            / TARGET
            / "observations"
            / CHECKPOINT
        )
        slot.mkdir(parents=True)
        observation_id = scoring.build_observation_id(
            scoring_protocol_sha256=(
                registration.EXPECTED_SCORING_PROTOCOL_SHA256
            ),
            target_official_date=TARGET,
            checkpoint_utc_date=CHECKPOINT,
        )
        marker = {
            "marker_schema_version": 1,
            "protocol_id": registration.EXPECTED_PROTOCOL_ID,
            "scoring_protocol_sha256": (
                registration.EXPECTED_SCORING_PROTOCOL_SHA256
            ),
            "target_official_date": TARGET,
            "checkpoint_utc_date": CHECKPOINT,
            "observation_id": observation_id,
            "reserved_at_utc": "2026-09-11T06:00:01Z",
            "runtime_code_commit": runtime_commit,
        }
        marker_bytes = shadow._canonical_json_file_bytes(marker)
        (slot / "RESERVED").write_bytes(marker_bytes)
        reservation = scoring.ScoringObservationReservation(
            slot_path=slot,
            target_official_date=TARGET,
            checkpoint_utc_date=CHECKPOINT,
            observation_id=observation_id,
            reserved_marker=marker,
            reserved_marker_sha256=hashlib.sha256(marker_bytes).hexdigest(),
        )
        body = b'{"dates":[],"totalGames":0}'
        expected_url = (
            "https://statsapi.mlb.com/api/v1/schedule?"
            "sportId=1&startDate=2026-09-10&endDate=2026-09-10&"
            "gameTypes=R&hydrate=probablePitcher"
        )
        raw_evidence = {
            "evidence_schema_version": 1,
            "target_official_date": TARGET,
            "checkpoint_utc_date": CHECKPOINT,
            "request_url": expected_url,
            "request_method": "GET",
            "application_request_headers": dict(scoring.OUTCOME_REQUEST_HEADERS),
            "effective_url": expected_url,
            "response_status_code": 200,
            "response_redirect_count": 0,
            "selected_response_headers": {
                "date": "Fri, 11 Sep 2026 06:01:00 GMT",
                "content-type": "application/json",
            },
            "response_received_at_utc": "2026-09-11T06:01:01Z",
            "response_body_base64": base64.b64encode(body).decode("ascii"),
            "response_body_sha256": hashlib.sha256(body).hexdigest(),
        }
        evidence_json = shadow._canonical_json_file_bytes(raw_evidence)
        evidence_bytes = shadow._canonical_gzip_bytes(evidence_json)
        evidence = scoring.ScoringOutcomeObservationEvidence(
            target_official_date=TARGET,
            checkpoint_utc_date=CHECKPOINT,
            outcome_http_date_utc="2026-09-11T06:01:00Z",
            response_received_at_utc="2026-09-11T06:01:01Z",
            response_body_sha256=hashlib.sha256(body).hexdigest(),
            flattened_occurrence_count=0,
            raw_evidence=raw_evidence,
            canonical_json_bytes=evidence_json,
            canonical_gzip_bytes=evidence_bytes,
            canonical_gzip_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
        )
        evidence_path = slot / scoring.OUTCOME_EVIDENCE_FILENAME
        evidence_path.write_bytes(evidence_bytes)
        evidence_publication = scoring.ScoringOutcomeEvidencePublication(
            reservation=reservation,
            evidence_path=evidence_path,
            evidence_sha256=evidence.canonical_gzip_sha256,
            evidence_size_bytes=len(evidence_bytes),
            outcome_http_date_utc=evidence.outcome_http_date_utc,
            response_received_at_utc=evidence.response_received_at_utc,
            response_body_sha256=evidence.response_body_sha256,
            flattened_occurrence_count=0,
            evidence=evidence,
        )
        adjudications = b"prediction_id,outcome_status\na,SCORED_FINAL\n"
        report = {
            "report_schema_version": 1,
            "status": "PROVISIONAL_DAILY_NO_VERDICT",
            "protocol_id": registration.EXPECTED_PROTOCOL_ID,
            "target_official_date": TARGET,
            "checkpoint_utc_date": CHECKPOINT,
            "observation_id": observation_id,
            "certified_prediction_count": 1,
            "scored_count": 1,
            "void_count": 0,
            "pending_count": 0,
            "correct_count": 1,
            "incorrect_count": 0,
            "accuracy": 1.0,
            "mean_log_loss": 0.2,
            "mean_brier_score": 0.04,
            "disclaimer": (
                "PROVISIONAL_DAILY_NO_VERDICT_SMALL_SAMPLE_"
                "DO_NOT_CHANGE_MODEL_OR_PROTOCOL"
            ),
        }
        report_bytes = shadow._canonical_json_file_bytes(report)
        documents = scoring.ScoringObservationDocuments(
            target_official_date=TARGET,
            checkpoint_utc_date=CHECKPOINT,
            observation_id=observation_id,
            adjudications_csv_bytes=adjudications,
            adjudications_sha256=hashlib.sha256(adjudications).hexdigest(),
            daily_report=report,
            daily_report_bytes=report_bytes,
            daily_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        )
        adjudications_path = slot / scoring.ADJUDICATIONS_FILENAME
        report_path = slot / scoring.DAILY_REPORT_FILENAME
        adjudications_path.write_bytes(adjudications)
        report_path.write_bytes(report_bytes)
        documents_publication = scoring.ScoringObservationDocumentsPublication(
            reservation=reservation,
            evidence_publication=evidence_publication,
            documents=documents,
            adjudications_path=adjudications_path,
            adjudications_sha256=documents.adjudications_sha256,
            adjudications_size_bytes=len(adjudications),
            daily_report_path=report_path,
            daily_report_sha256=documents.daily_report_sha256,
            daily_report_size_bytes=len(report_bytes),
        )
        return {
            "project": project,
            "slot": slot,
            "authority": authority,
            "reservation": reservation,
            "evidence_publication": evidence_publication,
            "documents_publication": documents_publication,
        }

    def _build(self, fixture: dict[str, object]):
        return scoring.build_scoring_observation_receipt(
            fixture["authority"],
            fixture["reservation"],
            fixture["evidence_publication"],
            fixture["documents_publication"],
            started_at_utc=STARTED,
            receipt_finalized_at_utc=FINALIZED,
        )

    def _publish(self, fixture: dict[str, object]):
        return scoring.publish_scoring_observation_receipt(
            fixture["authority"],
            fixture["reservation"],
            fixture["evidence_publication"],
            fixture["documents_publication"],
            started_at_utc=STARTED,
            receipt_finalized_at_utc=FINALIZED,
            project_directory=fixture["project"],
        )

    def test_exact_schema_hashes_counts_and_attestations(self) -> None:
        """Le recu lie exactement les trois sorties et leurs compteurs."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            built = self._build(fixture)
            receipt = built.receipt
            self.assertEqual(frozenset(receipt), scoring._OBSERVATION_RECEIPT_KEYS)
            self.assertEqual(
                frozenset(receipt["counts"]), scoring._OBSERVATION_COUNTS_KEYS
            )
            self.assertEqual(
                receipt["counts"],
                {
                    "certified_prediction_count": 1,
                    "scored_count": 1,
                    "void_count": 0,
                    "pending_count": 0,
                    "correct_count": 1,
                    "incorrect_count": 0,
                },
            )
            self.assertEqual(
                frozenset(receipt["negative_attestations"]),
                scoring._OBSERVATION_NEGATIVE_ATTESTATIONS_KEYS,
            )
            self.assertTrue(all(receipt["negative_attestations"].values()))
            self.assertEqual(
                receipt["outcome_evidence_sha256"],
                fixture["evidence_publication"].evidence_sha256,
            )
            self.assertEqual(
                receipt["adjudications_sha256"],
                fixture["documents_publication"].adjudications_sha256,
            )
            self.assertEqual(
                receipt["daily_report_sha256"],
                fixture["documents_publication"].daily_report_sha256,
            )

    def test_paths_and_timestamps_are_canonical(self) -> None:
        """Tous les liens sont relatifs, dates et propres au creneau."""
        with tempfile.TemporaryDirectory() as temporary:
            receipt = self._build(self._fixture(Path(temporary))).receipt
            prefix = (
                "shadow_scores/logistic_team_form_v1_platt_shadow_v2_2026_v1/"
                "2026-09-10/observations/2026-09-11/"
            )
            self.assertEqual(
                receipt["outcome_evidence_path"],
                prefix + scoring.OUTCOME_EVIDENCE_FILENAME,
            )
            self.assertEqual(
                receipt["adjudications_path"],
                prefix + scoring.ADJUDICATIONS_FILENAME,
            )
            self.assertEqual(
                receipt["daily_report_path"],
                prefix + scoring.DAILY_REPORT_FILENAME,
            )
            self.assertEqual(receipt["started_at_utc"], "2026-09-11T06:00:00Z")
            self.assertEqual(
                receipt["receipt_finalized_at_utc"], "2026-09-11T06:02:00Z"
            )

    def test_bytes_are_canonical_and_deterministic(self) -> None:
        """Deux constructions identiques donnent les memes octets."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            first = self._build(fixture)
            second = self._build(fixture)
            self.assertEqual(first.canonical_json_bytes, second.canonical_json_bytes)
            self.assertEqual(
                first.canonical_json_bytes,
                shadow._canonical_json_file_bytes(first.receipt),
            )
            self.assertEqual(
                first.receipt_sha256,
                hashlib.sha256(first.canonical_json_bytes).hexdigest(),
            )

    def test_receipt_has_no_self_hash_or_completed_back_reference(self) -> None:
        """Le sens des empreintes reste acyclique."""
        with tempfile.TemporaryDirectory() as temporary:
            receipt = self._build(self._fixture(Path(temporary))).receipt
            encoded = json.dumps(receipt, sort_keys=True)
            self.assertNotIn("observation_receipt_sha256", encoded)
            self.assertNotIn("COMPLETED", encoded)

    def test_invalid_chronology_is_rejected(self) -> None:
        """Le recu ne peut etre finalise avant la reponse MLB."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            with self.assertRaisesRegex(
                scoring.ScoringAdjudicationError, "chronologie"
            ):
                scoring.build_scoring_observation_receipt(
                    fixture["authority"],
                    fixture["reservation"],
                    fixture["evidence_publication"],
                    fixture["documents_publication"],
                    started_at_utc=STARTED,
                    receipt_finalized_at_utc=STARTED,
                )

    def test_forged_document_hash_is_rejected(self) -> None:
        """Un rapport declare sous une autre empreinte ne traverse pas."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            fixture["documents_publication"] = replace(
                fixture["documents_publication"], daily_report_sha256="0" * 64
            )
            with self.assertRaisesRegex(
                scoring.ScoringAdjudicationError, "sources du recu"
            ):
                self._build(fixture)

    def test_exact_fifth_file_is_published(self) -> None:
        """Le recu est le seul cinquieme fichier du chemin de succes."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            publication = self._publish(fixture)
            self.assertEqual(publication.receipt_path.name, "observation_receipt.json")
            self.assertEqual(
                publication.receipt_path.read_bytes(),
                publication.receipt.canonical_json_bytes,
            )
            self.assertEqual(
                publication.receipt_sha256,
                publication.receipt.receipt_sha256,
            )

    def test_publication_preserves_all_four_predecessors(self) -> None:
        """Aucune preuve deja publiee n'est modifiee par le recu."""
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
        """Un recu existant consomme definitivement cette position."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            publication = self._publish(fixture)
            before = publication.receipt_path.read_bytes()
            with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                self._publish(fixture)
            self.assertEqual(publication.receipt_path.read_bytes(), before)

    def test_concurrent_publishers_have_exactly_one_winner(self) -> None:
        """Deux processus ne peuvent jamais publier le meme recu."""
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
            self.assertEqual(
                sorted(path.name for path in fixture["slot"].iterdir()),
                [
                    "RESERVED",
                    scoring.ADJUDICATIONS_FILENAME,
                    scoring.DAILY_REPORT_FILENAME,
                    scoring.OBSERVATION_RECEIPT_FILENAME,
                    scoring.OUTCOME_EVIDENCE_FILENAME,
                ],
            )

    def test_foreign_or_tampered_predecessor_is_terminal(self) -> None:
        """Le recu ne repare ni un intrus ni un rapport modifie."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            foreign = fixture["slot"] / "foreign.txt"
            foreign.write_bytes(b"keep")
            with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                self._publish(fixture)
            self.assertEqual(foreign.read_bytes(), b"keep")
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            fixture["documents_publication"].daily_report_path.write_bytes(
                b"tampered"
            )
            with self.assertRaisesRegex(shadow.ShadowPredictionError, "diverge"):
                self._publish(fixture)

    def test_failure_after_link_preserves_the_receipt(self) -> None:
        """Une panne de relecture ne rend jamais le chemin reparable."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            original = scoring._read_exact_local_file

            def fail_receipt(path: Path, *, description: str) -> bytes:
                if path.name == scoring.OBSERVATION_RECEIPT_FILENAME:
                    raise shadow.ShadowPredictionError("panne apres lien")
                return original(path, description=description)

            with (
                mock.patch.object(
                    scoring, "_read_exact_local_file", side_effect=fail_receipt
                ),
                self.assertRaisesRegex(shadow.ShadowPredictionError, "apres lien"),
            ):
                self._publish(fixture)
            receipt_path = fixture["slot"] / scoring.OBSERVATION_RECEIPT_FILENAME
            self.assertTrue(receipt_path.is_file())
            before = receipt_path.read_bytes()
            with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                self._publish(fixture)
            self.assertEqual(receipt_path.read_bytes(), before)

    def test_builder_and_publication_do_not_call_network_model_or_predictions(self) -> None:
        """Le recu ne fait que relier des preuves deja disponibles."""
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
            ):
                publication = self._publish(fixture)
            self.assertTrue(publication.receipt_path.is_file())


def scoring_registration_path() -> Path:
    return registration.SCORING_PROTOCOL_RELATIVE_PATH


if __name__ == "__main__":
    unittest.main()
