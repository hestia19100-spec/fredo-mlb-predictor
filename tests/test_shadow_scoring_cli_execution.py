"""Tests de la commande publique du scoring quotidien MLB 2026."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import shadow_prediction as shadow
from src import shadow_scoring as scoring
from src import shadow_scoring_registration as registration


TARGET = "2026-09-10"
CHECKPOINT = "2026-09-11"
OBSERVATION_ID = "a" * 64


class ShadowScoringCLIExecutionTests(unittest.TestCase):
    """Le CLI ne contourne jamais l'orchestrateur public deja valide."""

    def _completed_result(
        self,
        directory: Path,
    ) -> tuple[scoring.ScoringObservationCompletionPublication, bytes]:
        receipt_document = {
            "checkpoint_utc_date": CHECKPOINT,
            "observation_id": OBSERVATION_ID,
            "receipt_schema_version": 1,
            "status": "PROVISIONAL_DAILY_NO_VERDICT",
            "target_official_date": TARGET,
        }
        receipt_bytes = shadow._canonical_json_file_bytes(receipt_document)
        receipt_path = directory / scoring.OBSERVATION_RECEIPT_FILENAME
        receipt_path.write_bytes(receipt_bytes)
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        receipt = scoring.ScoringObservationReceipt(
            target_official_date=TARGET,
            checkpoint_utc_date=CHECKPOINT,
            observation_id=OBSERVATION_ID,
            receipt=receipt_document,
            canonical_json_bytes=receipt_bytes,
            receipt_sha256=receipt_sha256,
            receipt_finalized_at_utc="2026-09-11T06:01:03Z",
        )
        receipt_publication = scoring.ScoringObservationReceiptPublication(
            reservation=mock.sentinel.reservation,
            evidence_publication=mock.sentinel.evidence,
            documents_publication=mock.sentinel.documents,
            receipt=receipt,
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
            receipt_size_bytes=len(receipt_bytes),
        )
        completed_marker = {
            "completed_at_utc": "2026-09-11T06:01:04Z",
            "marker_schema_version": 1,
            "observation_id": OBSERVATION_ID,
            "observation_receipt_path": (
                "shadow_scores/logistic_team_form_v1_platt_shadow_v2_"
                "2026_v1/2026-09-10/observations/2026-09-11/"
                "observation_receipt.json"
            ),
            "observation_receipt_sha256": receipt_sha256,
        }
        completed_bytes = shadow._canonical_json_file_bytes(completed_marker)
        completed_path = directory / scoring.COMPLETED_FILENAME
        completed_path.write_bytes(completed_bytes)
        completion = scoring.ScoringObservationCompletion(
            target_official_date=TARGET,
            checkpoint_utc_date=CHECKPOINT,
            observation_id=OBSERVATION_ID,
            completed_marker=completed_marker,
            canonical_json_bytes=completed_bytes,
            completed_sha256=hashlib.sha256(completed_bytes).hexdigest(),
            completed_at_utc="2026-09-11T06:01:04Z",
        )
        return (
            scoring.ScoringObservationCompletionPublication(
                receipt_publication=receipt_publication,
                completion=completion,
                completed_path=completed_path,
                completed_sha256=completion.completed_sha256,
                completed_size_bytes=len(completed_bytes),
            ),
            receipt_bytes,
        )

    def _failed_result(
        self,
        directory: Path,
    ) -> tuple[scoring.ScoringObservationFailure, bytes]:
        marker = {
            "checkpoint_utc_date": CHECKPOINT,
            "error_message": "MLB indisponible",
            "error_type": "ScoringOutcomeRequestError",
            "failed_at_utc": "2026-09-11T06:01:02Z",
            "marker_schema_version": 1,
            "observation_id": OBSERVATION_ID,
            "protocol_id": registration.EXPECTED_PROTOCOL_ID,
            "stage": "OUTCOME_REQUEST",
            "target_official_date": TARGET,
        }
        marker_bytes = shadow._canonical_json_file_bytes(marker)
        path = directory / scoring.FAILED_FILENAME
        path.write_bytes(marker_bytes)
        return (
            scoring.ScoringObservationFailure(
                reservation=mock.sentinel.reservation,
                failed_path=path,
                failed_marker=marker,
                failed_marker_sha256=hashlib.sha256(marker_bytes).hexdigest(),
                stage="OUTCOME_REQUEST",
                error_type="ScoringOutcomeRequestError",
            ),
            marker_bytes,
        )

    def _arguments(self) -> list[str]:
        return [
            "--target-official-date",
            TARGET,
            "--checkpoint-utc-date",
            CHECKPOINT,
            "--execute-scoring",
        ]

    def test_success_calls_only_public_orchestrator_and_prints_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, receipt_bytes = self._completed_result(Path(temporary))
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    scoring,
                    "execute_scoring_observation",
                    return_value=result,
                ) as execute,
                redirect_stdout(stdout),
            ):
                code = scoring.main(self._arguments())

        self.assertEqual(code, 0)
        execute.assert_called_once_with(TARGET, CHECKPOINT)
        self.assertEqual(stdout.getvalue().encode("utf-8"), receipt_bytes)

    def test_controlled_failure_is_printed_and_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, marker_bytes = self._failed_result(Path(temporary))
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    scoring,
                    "execute_scoring_observation",
                    return_value=result,
                ) as execute,
                redirect_stdout(stdout),
            ):
                code = scoring.main(self._arguments())

        self.assertEqual(code, 1)
        execute.assert_called_once_with(TARGET, CHECKPOINT)
        self.assertEqual(stdout.getvalue().encode("utf-8"), marker_bytes)

    def test_explicit_execution_flag_is_mandatory(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(
                scoring,
                "execute_scoring_observation",
                side_effect=AssertionError("execution interdite"),
            ) as execute,
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            scoring.main(self._arguments()[:-1])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--execute-scoring est obligatoire", stderr.getvalue())
        execute.assert_not_called()

    def test_both_dates_are_mandatory(self) -> None:
        cases = (
            ["--checkpoint-utc-date", CHECKPOINT, "--execute-scoring"],
            ["--target-official-date", TARGET, "--execute-scoring"],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit):
                    scoring.main(arguments)
                self.assertIn("required", stderr.getvalue())

    def test_domain_error_is_reported_without_traceback_or_stdout(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                scoring,
                "execute_scoring_observation",
                side_effect=shadow.ShadowPredictionError("refus exact"),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            scoring.main(self._arguments())

        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("error: refus exact", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_unexpected_programming_error_is_not_hidden(self) -> None:
        with mock.patch.object(
            scoring,
            "execute_scoring_observation",
            side_effect=TypeError("bug"),
        ):
            with self.assertRaisesRegex(TypeError, "bug"):
                scoring.main(self._arguments())

    def test_completion_output_rejects_tampered_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, _receipt_bytes = self._completed_result(Path(temporary))
            result.receipt_publication.receipt_path.write_bytes(b"falsifie\n")
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "diverge",
            ):
                scoring._canonical_scoring_execution_output_bytes(result)

    def test_failure_output_rejects_tampered_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, _marker_bytes = self._failed_result(Path(temporary))
            result.failed_path.write_bytes(b"falsifie\n")
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "diverge",
            ):
                scoring._canonical_scoring_execution_output_bytes(result)

    def test_foreign_result_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            shadow.ShadowPredictionError,
            "preuve terminale exacte",
        ):
            scoring._canonical_scoring_execution_output_bytes(mock.Mock())

    def test_cli_exposes_no_runtime_path_clock_or_network_override(self) -> None:
        forbidden = (
            "--project-directory",
            "--runtime-code-commit",
            "--database-path",
            "--model",
            "--clock",
            "--request",
            "--retry",
        )
        for flag in forbidden:
            with self.subTest(flag=flag):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit):
                    scoring.main([*self._arguments(), flag, "value"])
                self.assertIn("unrecognized arguments", stderr.getvalue())

    def test_cli_does_not_expose_model_or_sqlite_operations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, _receipt_bytes = self._completed_result(Path(temporary))
            protected_names = (
                "_load_frozen_shadow_model",
                "_predict_frozen_shadow_model_once",
                "execute_shadow_prediction",
            )
            with (
                mock.patch.object(
                    scoring,
                    "execute_scoring_observation",
                    return_value=result,
                ),
                mock.patch.object(
                    shadow,
                    protected_names[0],
                    side_effect=AssertionError(protected_names[0]),
                ) as model,
                mock.patch.object(
                    shadow,
                    protected_names[1],
                    side_effect=AssertionError(protected_names[1]),
                ) as predict,
                mock.patch.object(
                    shadow,
                    protected_names[2],
                    side_effect=AssertionError(protected_names[2]),
                ) as shadow_execution,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(scoring.main(self._arguments()), 0)
            model.assert_not_called()
            predict.assert_not_called()
            shadow_execution.assert_not_called()


if __name__ == "__main__":
    unittest.main()
