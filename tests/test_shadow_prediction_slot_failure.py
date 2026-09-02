"""Tests de fermeture terminale FAILED.json d'un slot fantome MLB v2."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import src.shadow_prediction as shadow_prediction
from src.shadow_prediction import (
    EXPECTED_MODEL_ARTIFACT_SHA256,
    EXPECTED_SHADOW_PROTOCOL_SHA256,
    SHADOW_RESULT_ROOT_RELATIVE_PATH,
    ShadowPredictionError,
    ShadowPredictionSlotConsumedError,
    ShadowPredictionSlotState,
    fail_shadow_prediction_slot,
    inspect_shadow_prediction_slot,
    reserve_shadow_prediction_slot,
)


PROTOCOL_SHA256 = EXPECTED_SHADOW_PROTOCOL_SHA256
EXECUTION_MANIFEST_SHA256 = "2" * 64
MODEL_ARTIFACT_SHA256 = EXPECTED_MODEL_ARTIFACT_SHA256
RUNTIME_CODE_COMMIT = "4" * 40
TARGET_DATE = "2026-09-03"
RESERVED_AT_UTC = "2026-09-02T18:00:00Z"
FAILED_AT_UTC = "2026-09-02T18:00:01Z"


def _canonical_json_file_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _process_failure_worker(
    reservation,
    project_directory: str,
    start_gate,
    result_queue,
    index: int,
) -> None:
    """Tente la meme fermeture depuis un vrai processus independant."""
    try:
        start_gate.wait(timeout=15)
        fail_shadow_prediction_slot(
            reservation,
            failed_at_utc=FAILED_AT_UTC,
            stage=f"PROCESS_STAGE_{index}",
            error_type="ShadowPredictionError",
            error_message=f"process failure {index}",
            project_directory=Path(project_directory),
        )
    except ShadowPredictionSlotConsumedError:
        result_queue.put(("consumed", index))
    except BaseException as error:
        result_queue.put(("error", type(error).__name__, str(error)))
    else:
        result_queue.put(("failed", index))


class ShadowPredictionSlotFailureTests(unittest.TestCase):
    """Une erreur post-reservation doit fermer le slot exactement une fois."""

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project = Path(self.temporary_directory.name)
        self.result_root = self.project.joinpath(
            *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts
        )
        self.result_root.mkdir(parents=True)
        self.slot = self.result_root / TARGET_DATE
        self.reservation = self._reserve()

    def _reserve(self, *, project: Path | None = None):
        target_project = project if project is not None else self.project
        return reserve_shadow_prediction_slot(
            TARGET_DATE,
            reserved_at_utc=RESERVED_AT_UTC,
            runtime_code_commit=RUNTIME_CODE_COMMIT,
            shadow_protocol_sha256=PROTOCOL_SHA256,
            execution_manifest_sha256=EXECUTION_MANIFEST_SHA256,
            model_artifact_sha256=MODEL_ARTIFACT_SHA256,
            project_directory=target_project,
        )

    def _fail(self, **overrides):
        arguments: dict[str, object] = {
            "reservation": self.reservation,
            "failed_at_utc": FAILED_AT_UTC,
            "stage": "SOURCE_SNAPSHOT",
            "error_type": "ShadowPredictionError",
            "error_message": "Échec contrôlé après réservation.",
            "project_directory": self.project,
        }
        arguments.update(overrides)
        return fail_shadow_prediction_slot(**arguments)

    def _expected_failed_marker(self, **overrides):
        marker = {
            "marker_schema_version": 1,
            "batch_id": self.reservation.batch_id,
            "slot_key": self.reservation.slot_key,
            "target_official_date": TARGET_DATE,
            "failed_at_utc": FAILED_AT_UTC,
            "stage": "SOURCE_SNAPSHOT",
            "error_type": "ShadowPredictionError",
            "error_message": "Échec contrôlé après réservation.",
            "shadow_protocol_sha256": PROTOCOL_SHA256,
            "execution_manifest_sha256": EXECUTION_MANIFEST_SHA256,
            "runtime_code_commit": RUNTIME_CODE_COMMIT,
        }
        marker.update(overrides)
        return marker

    def test_success_publishes_exact_failed_and_preserves_prior_files(
        self,
    ) -> None:
        partial = self.slot / "activation_reverification.remote.json.gz"
        partial.write_bytes(b"partial evidence\n")
        reserved_before = (self.slot / "RESERVED").read_bytes()

        failure = self._fail()

        expected_marker = self._expected_failed_marker()
        expected_bytes = _canonical_json_file_bytes(expected_marker)
        self.assertEqual(failure.slot_path, self.slot)
        self.assertEqual(failure.failed_marker, expected_marker)
        self.assertEqual(
            failure.failed_marker_sha256,
            hashlib.sha256(expected_bytes).hexdigest(),
        )
        self.assertEqual((self.slot / "FAILED.json").read_bytes(), expected_bytes)
        self.assertEqual((self.slot / "RESERVED").read_bytes(), reserved_before)
        self.assertEqual(partial.read_bytes(), b"partial evidence\n")
        self.assertEqual(
            inspect_shadow_prediction_slot(
                TARGET_DATE,
                shadow_protocol_sha256=PROTOCOL_SHA256,
                execution_manifest_sha256=EXECUTION_MANIFEST_SHA256,
                model_artifact_sha256=MODEL_ARTIFACT_SHA256,
                project_directory=self.project,
            ).state,
            ShadowPredictionSlotState.FAILED_CONSUMED,
        )

    def test_failure_reads_only_reserved_and_its_own_published_marker(
        self,
    ) -> None:
        actual_read_bytes = Path.read_bytes
        read_names: list[str] = []

        def recording_read_bytes(path: Path) -> bytes:
            read_names.append(path.name)
            return actual_read_bytes(path)

        with patch.object(Path, "read_bytes", new=recording_read_bytes):
            self._fail()

        self.assertEqual(read_names, ["RESERVED", "FAILED.json"])

    def test_invalid_runtime_values_fail_before_slot_access(self) -> None:
        invalid_cases = (
            {"reservation": None},
            {"failed_at_utc": "2026-09-02T18:00:01+00:00"},
            {"failed_at_utc": "2026-09-02T17:59:59Z"},
            {"stage": ""},
            {"stage": "   "},
            {"error_type": 1},
            {"error_message": ""},
        )

        for overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                with patch.object(
                    shadow_prediction,
                    "_require_shadow_result_root",
                    side_effect=AssertionError("acces disque interdit"),
                ):
                    with self.assertRaises(ShadowPredictionError):
                        self._fail(**overrides)

        self.assertFalse((self.slot / "FAILED.json").exists())

    def test_forged_reservation_fields_are_rejected_before_disk_access(
        self,
    ) -> None:
        altered_marker = dict(self.reservation.reserved_marker)
        altered_marker["runtime_code_commit"] = "5" * 40
        ambiguous_marker = dict(self.reservation.reserved_marker)
        ambiguous_marker["marker_schema_version"] = True
        forged_cases = (
            replace(self.reservation, slot_key="5" * 64),
            replace(self.reservation, batch_id="6" * 64),
            replace(self.reservation, reserved_marker_sha256="7" * 64),
            replace(self.reservation, reserved_marker=altered_marker),
            replace(
                self.reservation,
                reserved_marker=ambiguous_marker,
                reserved_marker_sha256=hashlib.sha256(
                    _canonical_json_file_bytes(ambiguous_marker)
                ).hexdigest(),
            ),
        )

        for forged in forged_cases:
            with self.subTest(forged=forged):
                with patch.object(
                    shadow_prediction,
                    "_require_shadow_result_root",
                    side_effect=AssertionError("acces disque interdit"),
                ):
                    with self.assertRaises(ShadowPredictionError):
                        self._fail(reservation=forged)

        self.assertFalse((self.slot / "FAILED.json").exists())

    def test_reservation_for_another_project_is_rejected(self) -> None:
        other_directory = TemporaryDirectory()
        self.addCleanup(other_directory.cleanup)
        other_project = Path(other_directory.name)
        other_project.joinpath(
            *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts
        ).mkdir(parents=True)

        with self.assertRaises(ShadowPredictionError):
            self._fail(project_directory=other_project)

        self.assertFalse((self.slot / "FAILED.json").exists())
        self.assertEqual(
            list(
                other_project.joinpath(
                    *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts
                ).iterdir()
            ),
            [],
        )

    def test_missing_or_noncanonical_reserved_cannot_be_repaired(self) -> None:
        cases = (
            None,
            b"not json\n",
            _canonical_json_file_bytes(
                {
                    **self.reservation.reserved_marker,
                    "runtime_code_commit": "5" * 40,
                }
            ),
        )

        for index, replacement in enumerate(cases):
            with self.subTest(index=index):
                if index:
                    temporary_directory = TemporaryDirectory()
                    self.addCleanup(temporary_directory.cleanup)
                    project = Path(temporary_directory.name)
                    project.joinpath(
                        *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts
                    ).mkdir(parents=True)
                    reservation = self._reserve(project=project)
                    reserved_path = reservation.slot_path / "RESERVED"
                else:
                    project = self.project
                    reservation = self.reservation
                    reserved_path = self.slot / "RESERVED"

                reserved_path.unlink()
                if replacement is not None:
                    reserved_path.write_bytes(replacement)

                with self.assertRaises(ShadowPredictionError):
                    self._fail(
                        reservation=reservation,
                        project_directory=project,
                    )

                self.assertFalse(
                    (reservation.slot_path / "FAILED.json").exists()
                )

    def test_existing_failed_is_terminal_and_never_overwritten(self) -> None:
        first = self._fail()
        failed_path = self.slot / "FAILED.json"
        before = failed_path.read_bytes()

        with self.assertRaises(ShadowPredictionSlotConsumedError):
            self._fail(error_message="second contenu interdit")

        self.assertEqual(failed_path.read_bytes(), before)
        self.assertEqual(
            hashlib.sha256(before).hexdigest(),
            first.failed_marker_sha256,
        )

    def test_any_existing_terminal_path_blocks_failure_without_reading_reserved(
        self,
    ) -> None:
        for terminal_name in ("COMPLETED", "FAILED.json"):
            with self.subTest(terminal_name=terminal_name):
                temporary_directory = TemporaryDirectory()
                self.addCleanup(temporary_directory.cleanup)
                project = Path(temporary_directory.name)
                project.joinpath(
                    *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts
                ).mkdir(parents=True)
                reservation = self._reserve(project=project)
                terminal = reservation.slot_path / terminal_name
                terminal.mkdir()

                with patch.object(
                    shadow_prediction,
                    "_read_canonical_json_object",
                    side_effect=AssertionError("RESERVED ne doit pas etre lu"),
                ):
                    with self.assertRaises(
                        ShadowPredictionSlotConsumedError
                    ):
                        self._fail(
                            reservation=reservation,
                            project_directory=project,
                        )

                self.assertTrue(terminal.is_dir())

    def test_concurrent_failures_have_exactly_one_immutable_winner(self) -> None:
        def fail(index: int) -> str:
            try:
                self._fail(
                    stage=f"STAGE_{index}",
                    error_message=f"failure {index}",
                )
                return "failed"
            except ShadowPredictionSlotConsumedError:
                return "consumed"

        with ThreadPoolExecutor(max_workers=12) as executor:
            outcomes = list(executor.map(fail, range(12)))

        self.assertEqual(outcomes.count("failed"), 1)
        self.assertEqual(outcomes.count("consumed"), 11)
        failed_bytes = (self.slot / "FAILED.json").read_bytes()
        failed = json.loads(failed_bytes)
        self.assertEqual(failed_bytes, _canonical_json_file_bytes(failed))
        winner = int(failed["stage"].removeprefix("STAGE_"))
        self.assertIn(winner, range(12))
        self.assertEqual(failed["error_message"], f"failure {winner}")

    def test_concurrent_processes_have_exactly_one_terminal_winner(self) -> None:
        context = multiprocessing.get_context("spawn")
        process_count = 6
        start_gate = context.Barrier(process_count)
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_process_failure_worker,
                args=(
                    self.reservation,
                    str(self.project),
                    start_gate,
                    result_queue,
                    index,
                ),
            )
            for index in range(process_count)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=20)
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    self.fail("Un processus concurrent ne s'est pas termine.")
                self.assertEqual(process.exitcode, 0)

            outcomes = []
            for _index in range(process_count):
                try:
                    outcomes.append(result_queue.get(timeout=5))
                except Empty:
                    self.fail("Un processus concurrent n'a rendu aucun resultat.")
        finally:
            result_queue.close()
            result_queue.join_thread()

        labels = [outcome[0] for outcome in outcomes]
        self.assertEqual(labels.count("failed"), 1)
        self.assertEqual(labels.count("consumed"), process_count - 1)
        self.assertNotIn("error", labels)
        failed = json.loads((self.slot / "FAILED.json").read_bytes())
        winner = int(failed["stage"].removeprefix("PROCESS_STAGE_"))
        self.assertIn(winner, range(process_count))
        self.assertEqual(
            failed["error_message"],
            f"process failure {winner}",
        )

    def test_publication_failure_before_link_preserves_incomplete_slot(
        self,
    ) -> None:
        partial = self.slot / "partial.bin"
        partial.write_bytes(b"keep\n")

        with patch.object(
            shadow_prediction,
            "_publish_exclusive_verified",
            side_effect=OSError("publication failure"),
        ):
            with self.assertRaises(OSError):
                self._fail()

        self.assertEqual(partial.read_bytes(), b"keep\n")
        self.assertTrue((self.slot / "RESERVED").is_file())
        self.assertFalse((self.slot / "FAILED.json").exists())

    def test_post_publication_failure_preserves_terminal_failed(self) -> None:
        actual_publication = shadow_prediction._publish_exclusive_verified

        def publish_then_fail(destination: Path, content: bytes) -> str:
            actual_publication(destination, content)
            raise OSError("failure after durable publication")

        with patch.object(
            shadow_prediction,
            "_publish_exclusive_verified",
            side_effect=publish_then_fail,
        ):
            with self.assertRaises(OSError):
                self._fail()

        before = (self.slot / "FAILED.json").read_bytes()
        with self.assertRaises(ShadowPredictionSlotConsumedError):
            self._fail()
        self.assertEqual((self.slot / "FAILED.json").read_bytes(), before)

    def test_slot_replaced_by_non_directory_is_rejected_without_output(
        self,
    ) -> None:
        reserved_path = self.slot / "RESERVED"
        reserved_path.unlink()
        self.slot.rmdir()
        self.slot.write_bytes(b"foreign slot\n")

        with self.assertRaises(ShadowPredictionError):
            self._fail()

        self.assertEqual(self.slot.read_bytes(), b"foreign slot\n")


if __name__ == "__main__":
    unittest.main()
