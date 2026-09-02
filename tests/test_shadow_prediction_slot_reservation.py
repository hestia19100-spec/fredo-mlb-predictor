"""Tests de reservation atomique d'un slot fantome MLB v2."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
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
    ShadowPredictionSlotInspection,
    ShadowPredictionSlotState,
    inspect_shadow_prediction_slot,
    reserve_shadow_prediction_slot,
)


PROTOCOL_SHA256 = EXPECTED_SHADOW_PROTOCOL_SHA256
EXECUTION_MANIFEST_SHA256 = "2" * 64
MODEL_ARTIFACT_SHA256 = EXPECTED_MODEL_ARTIFACT_SHA256
RUNTIME_CODE_COMMIT = "4" * 40
TARGET_DATE = "2026-09-03"
RESERVED_AT_UTC = "2026-09-02T18:00:00Z"


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


class ShadowPredictionSlotReservationTests(unittest.TestCase):
    """RESERVED doit consommer exactement une fois le slot journalier."""

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project = Path(self.temporary_directory.name)
        self.result_root = self.project.joinpath(
            *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts
        )
        self.slot = self.result_root / TARGET_DATE

    def _reserve(self):
        return reserve_shadow_prediction_slot(
            TARGET_DATE,
            reserved_at_utc=RESERVED_AT_UTC,
            runtime_code_commit=RUNTIME_CODE_COMMIT,
            shadow_protocol_sha256=PROTOCOL_SHA256,
            execution_manifest_sha256=EXECUTION_MANIFEST_SHA256,
            model_artifact_sha256=MODEL_ARTIFACT_SHA256,
            project_directory=self.project,
        )

    def _expected_marker(self, *, batch_id: str, slot_key: str):
        return {
            "marker_schema_version": 1,
            "batch_id": batch_id,
            "slot_key": slot_key,
            "target_official_date": TARGET_DATE,
            "reserved_at_utc": RESERVED_AT_UTC,
            "shadow_protocol_sha256": PROTOCOL_SHA256,
            "execution_manifest_sha256": EXECUTION_MANIFEST_SHA256,
            "runtime_code_commit": RUNTIME_CODE_COMMIT,
        }

    def test_success_creates_only_canonical_reserved(self) -> None:
        self.result_root.mkdir(parents=True)

        reservation = self._reserve()

        expected_marker = self._expected_marker(
            batch_id=reservation.batch_id,
            slot_key=reservation.slot_key,
        )
        expected_bytes = _canonical_json_file_bytes(expected_marker)
        self.assertEqual(reservation.slot_path, self.slot)
        self.assertEqual(reservation.reserved_marker, expected_marker)
        self.assertEqual(
            reservation.reserved_marker_sha256,
            hashlib.sha256(expected_bytes).hexdigest(),
        )
        self.assertEqual((self.slot / "RESERVED").read_bytes(), expected_bytes)
        self.assertEqual(
            sorted(path.name for path in self.slot.iterdir()),
            ["RESERVED"],
        )
        self.assertEqual(
            inspect_shadow_prediction_slot(
                TARGET_DATE,
                shadow_protocol_sha256=PROTOCOL_SHA256,
                execution_manifest_sha256=EXECUTION_MANIFEST_SHA256,
                model_artifact_sha256=MODEL_ARTIFACT_SHA256,
                project_directory=self.project,
            ).state,
            ShadowPredictionSlotState.INCOMPLETE_CONSUMED,
        )

    def test_inspection_precedes_every_result_tree_creation(self) -> None:
        self.result_root.mkdir(parents=True)
        actual_inspection = shadow_prediction.inspect_shadow_prediction_slot
        observations: list[bool] = []

        def inspecting_first(*args, **kwargs):
            observations.append(self.slot.exists())
            return actual_inspection(*args, **kwargs)

        with patch.object(
            shadow_prediction,
            "inspect_shadow_prediction_slot",
            side_effect=inspecting_first,
        ) as inspection_mock:
            self._reserve()

        self.assertEqual(observations, [False])
        inspection_mock.assert_called_once()
        self.assertTrue((self.slot / "RESERVED").is_file())

    def test_every_consumed_state_is_rejected_before_preparation(self) -> None:
        for state in (
            ShadowPredictionSlotState.COMPLETED_EXACT,
            ShadowPredictionSlotState.COMPLETED_MISMATCH,
            ShadowPredictionSlotState.FAILED_CONSUMED,
            ShadowPredictionSlotState.INCOMPLETE_CONSUMED,
        ):
            with self.subTest(state=state.value):
                inspection = ShadowPredictionSlotInspection(
                    state=state,
                    slot_path=self.slot,
                    slot_key="5" * 64,
                    batch_id="6" * 64,
                    receipt={} if state is ShadowPredictionSlotState.COMPLETED_EXACT else None,
                )
                with (
                    patch.object(
                        shadow_prediction,
                        "inspect_shadow_prediction_slot",
                        return_value=inspection,
                    ),
                    patch.object(
                        shadow_prediction,
                        "_require_shadow_result_root",
                        side_effect=AssertionError("preparation interdite"),
                    ),
                ):
                    with self.assertRaises(
                        ShadowPredictionSlotConsumedError
                    ):
                        self._reserve()

                self.assertEqual(list(self.project.iterdir()), [])

    def test_second_identical_reservation_never_reuses_or_overwrites(self) -> None:
        self.result_root.mkdir(parents=True)
        first = self._reserve()
        reserved_path = self.slot / "RESERVED"
        before = reserved_path.read_bytes()

        with self.assertRaises(ShadowPredictionSlotConsumedError):
            self._reserve()

        self.assertEqual(reserved_path.read_bytes(), before)
        self.assertEqual(
            hashlib.sha256(before).hexdigest(),
            first.reserved_marker_sha256,
        )
        self.assertEqual(
            sorted(path.name for path in self.slot.iterdir()),
            ["RESERVED"],
        )

    def test_preexisting_incomplete_slot_is_never_reused(self) -> None:
        self.slot.mkdir(parents=True)
        existing = self.slot / "private-evidence.bin"
        existing.write_bytes(b"do not touch\n")

        with self.assertRaises(ShadowPredictionSlotConsumedError):
            self._reserve()

        self.assertEqual(existing.read_bytes(), b"do not touch\n")
        self.assertFalse((self.slot / "RESERVED").exists())

    def test_all_runtime_values_are_validated_before_inspection(self) -> None:
        invalid_cases = (
            {"target_official_date": "2025-09-03"},
            {"target_official_date": "2026-02-30"},
            {"reserved_at_utc": "2026-09-02T18:00:00+00:00"},
            {"reserved_at_utc": "2026-09-04T00:00:00Z"},
            {"runtime_code_commit": "A" * 40},
            {"runtime_code_commit": "4" * 39},
            {"shadow_protocol_sha256": "A" * 64},
            {"execution_manifest_sha256": "2" * 63},
            {"model_artifact_sha256": True},
        )
        defaults: dict[str, object] = {
            "target_official_date": TARGET_DATE,
            "reserved_at_utc": RESERVED_AT_UTC,
            "runtime_code_commit": RUNTIME_CODE_COMMIT,
            "shadow_protocol_sha256": PROTOCOL_SHA256,
            "execution_manifest_sha256": EXECUTION_MANIFEST_SHA256,
            "model_artifact_sha256": MODEL_ARTIFACT_SHA256,
            "project_directory": self.project,
        }

        for overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                arguments = {**defaults, **overrides}
                with patch.object(
                    shadow_prediction,
                    "inspect_shadow_prediction_slot",
                    side_effect=AssertionError("inspection interdite"),
                ):
                    with self.assertRaises(ShadowPredictionError):
                        reserve_shadow_prediction_slot(**arguments)
                self.assertEqual(list(self.project.iterdir()), [])

    def test_missing_result_root_is_rejected_without_creating_it(self) -> None:
        with self.assertRaises(ShadowPredictionError):
            self._reserve()

        self.assertFalse(self.project.joinpath("shadow_results").exists())

    def test_final_slot_mkdir_is_exclusive(self) -> None:
        self.result_root.mkdir(parents=True)
        actual_mkdir = Path.mkdir
        calls: list[dict[str, object]] = []

        def recording_mkdir(path: Path, *args, **kwargs):
            if path == self.slot:
                calls.append(dict(kwargs))
            return actual_mkdir(path, *args, **kwargs)

        with patch.object(Path, "mkdir", new=recording_mkdir):
            self._reserve()

        self.assertEqual(calls, [{"exist_ok": False}])

    def test_mkdir_race_has_no_overwrite_or_cleanup(self) -> None:
        self.result_root.mkdir(parents=True)
        actual_inspection = shadow_prediction.inspect_shadow_prediction_slot

        def lose_race(*args, **kwargs):
            inspection = actual_inspection(*args, **kwargs)
            self.slot.mkdir(exist_ok=False)
            (self.slot / "rival.bin").write_bytes(b"winner\n")
            return inspection

        with patch.object(
            shadow_prediction,
            "inspect_shadow_prediction_slot",
            side_effect=lose_race,
        ):
            with self.assertRaises(ShadowPredictionSlotConsumedError):
                self._reserve()

        self.assertEqual((self.slot / "rival.bin").read_bytes(), b"winner\n")
        self.assertFalse((self.slot / "RESERVED").exists())

    def test_failure_after_slot_mkdir_leaves_empty_consumed_slot(self) -> None:
        self.result_root.mkdir(parents=True)

        with patch.object(
            shadow_prediction,
            "_fsync_parent_directory",
            side_effect=OSError("directory fsync failure"),
        ):
            with self.assertRaises(OSError):
                self._reserve()

        self.assertTrue(self.slot.is_dir())
        self.assertEqual(list(self.slot.iterdir()), [])
        self.assertEqual(
            inspect_shadow_prediction_slot(
                TARGET_DATE,
                shadow_protocol_sha256=PROTOCOL_SHA256,
                execution_manifest_sha256=EXECUTION_MANIFEST_SHA256,
                model_artifact_sha256=MODEL_ARTIFACT_SHA256,
                project_directory=self.project,
            ).state,
            ShadowPredictionSlotState.INCOMPLETE_CONSUMED,
        )

    def test_publication_failure_never_removes_created_slot(self) -> None:
        self.result_root.mkdir(parents=True)

        with patch.object(
            shadow_prediction,
            "_publish_exclusive_verified",
            side_effect=OSError("publication failure"),
        ):
            with self.assertRaises(OSError):
                self._reserve()

        self.assertTrue(self.slot.is_dir())
        self.assertEqual(list(self.slot.iterdir()), [])

    def test_post_publication_failure_preserves_reserved(self) -> None:
        self.result_root.mkdir(parents=True)
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
                self._reserve()

        reserved_path = self.slot / "RESERVED"
        self.assertTrue(reserved_path.is_file())
        marker = json.loads(reserved_path.read_bytes())
        self.assertEqual(set(marker), set(self._expected_marker(
            batch_id=marker["batch_id"],
            slot_key=marker["slot_key"],
        )))
        self.assertEqual(
            inspect_shadow_prediction_slot(
                TARGET_DATE,
                shadow_protocol_sha256=PROTOCOL_SHA256,
                execution_manifest_sha256=EXECUTION_MANIFEST_SHA256,
                model_artifact_sha256=MODEL_ARTIFACT_SHA256,
                project_directory=self.project,
            ).state,
            ShadowPredictionSlotState.INCOMPLETE_CONSUMED,
        )

    def test_concurrent_reservations_have_exactly_one_winner(self) -> None:
        self.result_root.mkdir(parents=True)

        def reserve(_index: int) -> str:
            try:
                self._reserve()
                return "reserved"
            except ShadowPredictionSlotConsumedError:
                return "consumed"

        with ThreadPoolExecutor(max_workers=12) as executor:
            outcomes = list(executor.map(reserve, range(12)))

        self.assertEqual(outcomes.count("reserved"), 1)
        self.assertEqual(outcomes.count("consumed"), 11)
        self.assertEqual(
            sorted(path.name for path in self.slot.iterdir()),
            ["RESERVED"],
        )
        reserved_bytes = (self.slot / "RESERVED").read_bytes()
        marker = json.loads(reserved_bytes)
        self.assertEqual(
            reserved_bytes,
            _canonical_json_file_bytes(marker),
        )


if __name__ == "__main__":
    unittest.main()
