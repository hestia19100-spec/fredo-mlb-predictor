"""Tests de publication append-only des documents quotidiens MLB."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import shadow_prediction as shadow
from src import shadow_scoring as scoring
from src import shadow_scoring_registration as registration


TARGET = "2026-09-10"
CHECKPOINT = "2026-09-11"


class ShadowScoringDailyPublicationTests(unittest.TestCase):
    """Le CSV et le rapport occupent exactement les positions trois et quatre."""

    def _fixture(self, project: Path) -> dict[str, object]:
        target = date.fromisoformat(TARGET)
        checkpoint = date.fromisoformat(CHECKPOINT)
        slot = (
            project
            / scoring.SCORING_OUTPUT_ROOT_RELATIVE_PATH
            / TARGET
            / "observations"
            / CHECKPOINT
        )
        slot.mkdir(parents=True)
        observation_id = scoring.build_observation_id(
            scoring_protocol_sha256=registration.EXPECTED_SCORING_PROTOCOL_SHA256,
            target_official_date=target,
            checkpoint_utc_date=checkpoint,
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
            "runtime_code_commit": "5" * 40,
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
        raw_body = b'{"dates":[],"totalGames":0}'
        canonical_gzip = shadow._canonical_gzip_bytes(b"outcome-evidence\n")
        evidence = scoring.ScoringOutcomeObservationEvidence(
            target_official_date=TARGET,
            checkpoint_utc_date=CHECKPOINT,
            outcome_http_date_utc="2026-09-11T06:01:00Z",
            response_received_at_utc="2026-09-11T06:01:01Z",
            response_body_sha256=hashlib.sha256(raw_body).hexdigest(),
            flattened_occurrence_count=0,
            raw_evidence={},
            canonical_json_bytes=b"outcome-evidence\n",
            canonical_gzip_bytes=canonical_gzip,
            canonical_gzip_sha256=hashlib.sha256(canonical_gzip).hexdigest(),
        )
        evidence_path = slot / scoring.OUTCOME_EVIDENCE_FILENAME
        evidence_path.write_bytes(canonical_gzip)
        evidence_publication = scoring.ScoringOutcomeEvidencePublication(
            reservation=reservation,
            evidence_path=evidence_path,
            evidence_sha256=evidence.canonical_gzip_sha256,
            evidence_size_bytes=len(canonical_gzip),
            outcome_http_date_utc=evidence.outcome_http_date_utc,
            response_received_at_utc=evidence.response_received_at_utc,
            response_body_sha256=evidence.response_body_sha256,
            flattened_occurrence_count=0,
            evidence=evidence,
        )
        adjudications = b"prediction_id,outcome_status\n" + b"a,SCORED_FINAL\n"
        report = {
            "report_schema_version": 1,
            "status": "PROVISIONAL_DAILY_NO_VERDICT",
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
        batch = scoring.ScoringAdjudicationBatch(
            target_official_date=TARGET,
            checkpoint_utc_date=CHECKPOINT,
            outcome_evidence_sha256=evidence.canonical_gzip_sha256,
            adjudications=(),
            certified_prediction_count=0,
            scored_count=0,
            void_count=0,
            pending_count=0,
            correct_count=0,
            incorrect_count=0,
        )
        return {
            "project": project,
            "slot": slot,
            "authority": mock.sentinel.authority,
            "reservation": reservation,
            "evidence_publication": evidence_publication,
            "prediction_source": mock.sentinel.prediction_source,
            "batch": batch,
            "documents": documents,
            "canonical_gzip": canonical_gzip,
            "marker_bytes": marker_bytes,
        }

    def _publish(self, fixture: dict[str, object]):
        with (
            mock.patch.object(
                scoring,
                "build_scoring_observation_documents",
                return_value=fixture["documents"],
            ) as builder,
            mock.patch.object(
                scoring,
                "_validate_outcome_evidence",
                return_value=({}, b"{}", fixture["canonical_gzip"]),
            ),
        ):
            publication = scoring.publish_scoring_observation_documents(
                fixture["authority"],
                fixture["reservation"],
                fixture["evidence_publication"],
                fixture["prediction_source"],
                fixture["batch"],
                project_directory=fixture["project"],
            )
        builder.assert_called_once_with(
            fixture["authority"],
            fixture["reservation"],
            fixture["prediction_source"],
            fixture["batch"],
        )
        return publication

    def test_exact_third_and_fourth_files_are_published(self) -> None:
        """Les fichiers suivent strictement RESERVED puis la preuve MLB."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            publication = self._publish(fixture)
            slot = fixture["slot"]
            self.assertEqual(
                sorted(path.name for path in slot.iterdir()),
                [
                    "RESERVED",
                    scoring.ADJUDICATIONS_FILENAME,
                    scoring.DAILY_REPORT_FILENAME,
                    scoring.OUTCOME_EVIDENCE_FILENAME,
                ],
            )
            documents = fixture["documents"]
            self.assertEqual(
                publication.adjudications_path.read_bytes(),
                documents.adjudications_csv_bytes,
            )
            self.assertEqual(
                publication.daily_report_path.read_bytes(),
                documents.daily_report_bytes,
            )
            self.assertEqual(
                publication.adjudications_sha256,
                documents.adjudications_sha256,
            )
            self.assertEqual(
                publication.daily_report_sha256,
                documents.daily_report_sha256,
            )

    def test_publication_order_is_adjudications_then_report(self) -> None:
        """Le rapport ne peut exister avant le CSV qu'il resume."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            order: list[str] = []
            original = shadow._publish_exclusive_verified

            def observed(destination: Path, content: bytes) -> str:
                order.append(destination.name)
                return original(destination, content)

            with mock.patch.object(
                scoring.shadow,
                "_publish_exclusive_verified",
                side_effect=observed,
            ):
                self._publish(fixture)
            self.assertEqual(
                order,
                [scoring.ADJUDICATIONS_FILENAME, scoring.DAILY_REPORT_FILENAME],
            )

    def test_second_publication_never_overwrites_or_repairs(self) -> None:
        """Un creneau publie est definitivement consomme."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            self._publish(fixture)
            before = {
                path.name: path.read_bytes() for path in fixture["slot"].iterdir()
            }
            with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                self._publish(fixture)
            after = {
                path.name: path.read_bytes() for path in fixture["slot"].iterdir()
            }
            self.assertEqual(after, before)

    def test_concurrent_publishers_have_exactly_one_winner(self) -> None:
        """Deux processus logiques ne publient jamais le meme document."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))

            def attempt() -> str:
                try:
                    scoring.publish_scoring_observation_documents(
                        fixture["authority"],
                        fixture["reservation"],
                        fixture["evidence_publication"],
                        fixture["prediction_source"],
                        fixture["batch"],
                        project_directory=fixture["project"],
                    )
                except shadow.ShadowPredictionError:
                    return "LOST"
                return "WON"

            with (
                mock.patch.object(
                    scoring,
                    "build_scoring_observation_documents",
                    return_value=fixture["documents"],
                ),
                mock.patch.object(
                    scoring,
                    "_validate_outcome_evidence",
                    return_value=({}, b"{}", fixture["canonical_gzip"]),
                ),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                results = list(executor.map(lambda _: attempt(), range(2)))
            self.assertEqual(results.count("WON"), 1)
            self.assertEqual(results.count("LOST"), 1)
            self.assertEqual(len(list(fixture["slot"].iterdir())), 4)

    def test_preexisting_third_path_is_terminal(self) -> None:
        """Aucun fichier inattendu ne peut etre efface ou contourne."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            foreign = fixture["slot"] / "foreign.txt"
            foreign.write_bytes(b"keep")
            with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                self._publish(fixture)
            self.assertEqual(foreign.read_bytes(), b"keep")
            self.assertFalse(
                (fixture["slot"] / scoring.ADJUDICATIONS_FILENAME).exists()
            )

    def test_failure_after_csv_preserves_partial_slot(self) -> None:
        """Une panne au quatrieme fichier ne permet aucune reparation."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            original = shadow._publish_exclusive_verified

            def fail_report(destination: Path, content: bytes) -> str:
                if destination.name == scoring.DAILY_REPORT_FILENAME:
                    raise shadow.ShadowPredictionError("panne simulee")
                return original(destination, content)

            with (
                mock.patch.object(
                    scoring.shadow,
                    "_publish_exclusive_verified",
                    side_effect=fail_report,
                ),
                self.assertRaisesRegex(shadow.ShadowPredictionError, "panne simulee"),
            ):
                self._publish(fixture)
            self.assertTrue(
                (fixture["slot"] / scoring.ADJUDICATIONS_FILENAME).is_file()
            )
            self.assertFalse(
                (fixture["slot"] / scoring.DAILY_REPORT_FILENAME).exists()
            )
            with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                self._publish(fixture)

    def test_persisted_outcome_evidence_must_be_exact(self) -> None:
        """Une preuve MLB modifiee bloque les documents derives."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            fixture["evidence_publication"].evidence_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "preuve MLB persiste diverge",
            ):
                self._publish(fixture)

    def test_forged_evidence_publication_is_rejected(self) -> None:
        """Le recu en memoire doit correspondre aux octets archives."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            fixture["evidence_publication"] = replace(
                fixture["evidence_publication"],
                evidence_sha256="0" * 64,
            )
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "preuve MLB publiee diverge",
            ):
                self._publish(fixture)

    def test_reserved_marker_must_remain_byte_exact(self) -> None:
        """La publication ne remplace jamais l'intention initiale."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            (fixture["slot"] / "RESERVED").write_bytes(b"{}\n")
            with self.assertRaises(shadow.ShadowPredictionError):
                self._publish(fixture)
            self.assertFalse(
                (fixture["slot"] / scoring.ADJUDICATIONS_FILENAME).exists()
            )

    def test_existing_adjudications_are_never_reused(self) -> None:
        """Meme des octets identiques ne rendent pas un chemin reparable."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            documents = fixture["documents"]
            (fixture["slot"] / scoring.ADJUDICATIONS_FILENAME).write_bytes(
                documents.adjudications_csv_bytes
            )
            with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                self._publish(fixture)

    def test_publication_preserves_reserved_and_outcome_bytes(self) -> None:
        """Les deux preuves precedentes restent strictement inchangees."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            reserved_before = (fixture["slot"] / "RESERVED").read_bytes()
            evidence_before = fixture["evidence_publication"].evidence_path.read_bytes()
            self._publish(fixture)
            self.assertEqual(
                (fixture["slot"] / "RESERVED").read_bytes(), reserved_before
            )
            self.assertEqual(
                fixture["evidence_publication"].evidence_path.read_bytes(),
                evidence_before,
            )

    def test_publication_performs_no_network_sqlite_model_or_prediction_call(self) -> None:
        """Seuls les deux fichiers canoniques deja derives sont publies."""
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
                    scoring,
                    "adjudicate_scoring_predictions",
                    side_effect=forbidden,
                ),
            ):
                publication = self._publish(fixture)
            self.assertTrue(publication.adjudications_path.is_file())


if __name__ == "__main__":
    unittest.main()
