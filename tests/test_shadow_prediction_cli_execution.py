"""Tests du drapeau CLI explicite d'execution fantome MLB v2."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import shadow_prediction as shadow


TARGET = "2026-09-05"
RECEIPT = {
    "batch": {
        "batch_id": "b" * 64,
        "status": "COMPLETED_WITH_PREDICTIONS",
    },
    "receipt_schema_version": 1,
}


class ShadowPredictionCLIExecutionTests(unittest.TestCase):
    """Le CLI ne doit offrir aucun raccourci autour de l'orchestrateur."""

    def test_default_mode_remains_inert_preview(self) -> None:
        preview = mock.Mock()
        preview.to_canonical_json.return_value = '{"mode":"PREVIEW"}'
        stdout = io.StringIO()
        with (
            mock.patch.object(
                shadow,
                "preview_shadow_prediction",
                return_value=preview,
            ) as preview_mock,
            mock.patch.object(
                shadow,
                "execute_shadow_prediction",
                side_effect=AssertionError("execution interdite"),
            ),
            redirect_stdout(stdout),
        ):
            code = shadow.main(["--target-official-date", TARGET])

        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), '{"mode":"PREVIEW"}\n')
        preview_mock.assert_called_once_with(TARGET)

    def test_execute_flag_calls_only_public_orchestrator(self) -> None:
        stdout = io.StringIO()
        with (
            mock.patch.object(
                shadow,
                "preview_shadow_prediction",
                side_effect=AssertionError("apercu interdit"),
            ),
            mock.patch.object(
                shadow,
                "execute_shadow_prediction",
                return_value=RECEIPT,
            ) as execute_mock,
            redirect_stdout(stdout),
        ):
            code = shadow.main(
                ["--target-official-date", TARGET, "--execute-shadow"]
            )

        self.assertEqual(code, 0)
        execute_mock.assert_called_once_with(TARGET)
        self.assertEqual(json.loads(stdout.getvalue()), RECEIPT)
        self.assertEqual(
            stdout.getvalue().encode("utf-8"),
            shadow._canonical_json_file_bytes(RECEIPT),
        )

    def test_new_completion_and_exact_duplicate_have_same_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            slot = Path(temporary_directory) / TARGET
            slot.mkdir()
            receipt_bytes = shadow._canonical_json_file_bytes(RECEIPT)
            (slot / shadow.RECEIPT_FILENAME).write_bytes(receipt_bytes)
            completion = shadow.ShadowCompletionPublication(
                slot_path=slot,
                completed_path=slot / shadow.COMPLETED_FILENAME,
                completed_relative_path=(
                    f"shadow_results/logistic_team_form_v1_platt_shadow_v2/"
                    f"{TARGET}/COMPLETED"
                ),
                completed_sha256="c" * 64,
                completed_size_bytes=1,
                completed_at_utc="2026-09-05T10:00:00Z",
                batch_id="b" * 64,
                receipt_relative_path=(
                    f"shadow_results/logistic_team_form_v1_platt_shadow_v2/"
                    f"{TARGET}/receipt.json"
                ),
                receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
                batch_status="COMPLETED_WITH_PREDICTIONS",
                earliest_predicted_scheduled_start_utc=(
                    "2026-09-05T20:00:00Z"
                ),
                local_completion_lead_seconds=36000,
                official_prediction_created=True,
            )
            outputs: list[str] = []
            for result in (completion, RECEIPT):
                stdout = io.StringIO()
                with (
                    mock.patch.object(
                        shadow,
                        "execute_shadow_prediction",
                        return_value=result,
                    ),
                    redirect_stdout(stdout),
                ):
                    self.assertEqual(
                        shadow.main(
                            [
                                "--target-official-date",
                                TARGET,
                                "--execute-shadow",
                            ]
                        ),
                        0,
                    )
                outputs.append(stdout.getvalue())

        self.assertEqual(outputs, [receipt_bytes.decode("utf-8")] * 2)

    def test_execution_error_is_reported_without_stdout(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                shadow,
                "execute_shadow_prediction",
                side_effect=shadow.ShadowPredictionError("refus exact"),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            shadow.main(
                ["--target-official-date", TARGET, "--execute-shadow"]
            )

        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("error: refus exact", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(
            stderr.getvalue().startswith(
                "usage: python -m src.shadow_prediction"
            )
        )

    def test_completion_receipt_must_still_match_terminal_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            slot = Path(temporary_directory)
            receipt_bytes = shadow._canonical_json_file_bytes(RECEIPT)
            (slot / shadow.RECEIPT_FILENAME).write_bytes(receipt_bytes)
            completion = shadow.ShadowCompletionPublication(
                slot_path=slot,
                completed_path=slot / shadow.COMPLETED_FILENAME,
                completed_relative_path="x/COMPLETED",
                completed_sha256="c" * 64,
                completed_size_bytes=1,
                completed_at_utc="2026-09-05T10:00:00Z",
                batch_id="b" * 64,
                receipt_relative_path="x/receipt.json",
                receipt_sha256="0" * 64,
                batch_status="COMPLETED_WITH_PREDICTIONS",
                earliest_predicted_scheduled_start_utc=None,
                local_completion_lead_seconds=None,
                official_prediction_created=True,
            )
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "diverge",
            ):
                shadow._canonical_execution_receipt_bytes(completion)

    def test_cli_exposes_no_runtime_or_path_override(self) -> None:
        forbidden = (
            "--project-directory",
            "--database-path",
            "--data-directory",
            "--runtime-code-commit",
            "--model",
            "--clock",
        )
        for flag in forbidden:
            with self.subTest(flag=flag):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit):
                    shadow.main(
                        [
                            "--target-official-date",
                            TARGET,
                            "--execute-shadow",
                            flag,
                            "value",
                        ]
                    )
                self.assertIn("unrecognized arguments", stderr.getvalue())

    def test_unexpected_programming_error_is_not_hidden(self) -> None:
        with mock.patch.object(
            shadow,
            "execute_shadow_prediction",
            side_effect=TypeError("bug"),
        ):
            with self.assertRaisesRegex(TypeError, "bug"):
                shadow.main(
                    ["--target-official-date", TARGET, "--execute-shadow"]
                )


if __name__ == "__main__":
    unittest.main()
