"""Tests de l'inspection locale en lecture seule d'un slot fantome v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.shadow_prediction import (
    SHADOW_RESULT_ROOT_RELATIVE_PATH,
    SHADOW_SLOT_SUCCESS_FILENAMES,
    ShadowPredictionError,
    ShadowPredictionSlotState,
    build_batch_id,
    build_slot_key,
    inspect_shadow_prediction_slot,
)


PROTOCOL_SHA256 = "1" * 64
EXECUTION_MANIFEST_SHA256 = "2" * 64
MODEL_ARTIFACT_SHA256 = "3" * 64
TARGET_DATE = "2026-09-03"
DATA_FILENAMES = frozenset(
    {
        "activation_reverification.remote.json.gz",
        "source_snapshot.json.gz",
        "candidate_ledger.csv",
        "features.csv",
        "predictions.csv",
    }
)

RECEIPT_SECTION_KEYS = {
    "batch": {
        "batch_id",
        "slot_key",
        "target_official_date",
        "status",
        "earliest_predicted_scheduled_start_utc",
    },
    "activation": {
        "execution_manifest_introduction_commit",
        "activation_introduction_commit",
        "activation_path",
        "activation_sha256",
        "activation_verified_at_utc",
        "activation_remote_reverified_at_utc",
        "activation_remote_ref",
        "activation_remote_reverification_query_url",
        "activation_remote_reverification_effective_url",
        "activation_remote_reverification_status_code",
        "activation_remote_reverification_redirect_count",
        "activation_remote_reverification_response_received_at_utc",
        "activation_remote_reverification_response_body_sha256",
        "activation_remote_reverification_evidence_path",
        "activation_remote_reverification_evidence_sha256",
        "minimum_target_official_date",
    },
    "times": {
        "started_at_utc",
        "reserved_at_utc",
        "schedule_observed_at_utc",
        "information_cutoff_utc",
        "issued_at_utc",
        "receipt_finalized_at_utc",
        "mlb_http_date_utc",
        "mlb_http_response_received_at_utc",
        "clock_skew_seconds",
        "schedule_age_seconds",
    },
    "schedule_http_response": {
        "effective_url",
        "status_code",
        "redirect_count",
        "date_header_raw",
        "date_header_utc",
        "received_at_utc",
        "body_sha256",
    },
    "lineage": {
        "runtime_code_commit",
        "shadow_service_module_sha256",
        "shadow_protocol_sha256",
        "execution_manifest_sha256",
        "model_artifact_sha256",
        "artifact_manifest_sha256",
        "model_protocol_sha256",
        "evaluation_protocol_sha256",
        "evaluation_report_sha256",
        "evaluation_results_commit",
    },
    "source": {
        "sqlite_snapshot_sha256",
        "sqlite_snapshot_size_bytes",
        "source_snapshot_path",
        "source_snapshot_sha256",
        "schedule_ingestion_run_id",
        "schedule_source",
        "schedule_requested_start_date",
        "schedule_requested_end_date",
        "schedule_game_types",
        "schedule_request_parameters_json",
        "schedule_ingestion_completed_at_utc",
        "schedule_raw_archive_path",
        "schedule_raw_archive_sha256",
    },
    "counts": {
        "schedule_games",
        "eligible_games",
        "predicted_games",
        "excluded_games_by_reason",
    },
    "output_hashes": {
        "activation_reverification_evidence_sha256",
        "candidate_ledger_sha256",
        "features_sha256",
        "predictions_sha256",
    },
    "model_invariants": {
        "fit_calls",
        "partial_fit_calls",
        "recalibration_calls",
        "threshold_tuning_calls",
        "feature_selection_calls",
        "predict_proba_calls",
        "artifact_state_sha256_before",
        "artifact_state_sha256_after",
        "artifact_state_unchanged",
        "warning_policy_id",
        "approved_compatibility_warning_count",
        "unexpected_warning_count",
    },
    "negative_attestations": {
        "ALL_TARGET_GAMES_UNSTARTED_AT_INFORMATION_CUTOFF",
        "TARGET_OUTCOMES_NOT_AVAILABLE_AT_INFORMATION_CUTOFF",
        "TARGET_SCORE_VALUES_REDACTED_FROM_TRACKED_TARGET_ROWS",
        "TARGET_SCORES_NOT_USED_AS_FEATURES",
        "ODDS_NOT_READ",
        "BETTING_RECOMMENDATIONS_NOT_COMPUTED",
        "SEASON_2026_METRICS_NOT_COMPUTED",
    },
    "runtime_versions": {
        "python",
        "numpy",
        "pandas",
        "scipy",
        "scikit_learn",
        "joblib",
    },
}
EXCLUSION_KEYS = {
    "POSTPONED",
    "CANCELLED",
    "START_TIME_MISSING",
    "INSUFFICIENT_BOTH_HISTORY",
    "INSUFFICIENT_AWAY_HISTORY",
    "INSUFFICIENT_HOME_HISTORY",
}


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


class ShadowPredictionSlotInspectionTests(unittest.TestCase):
    """Le slot est classe sans aucune entree officielle ni ecriture."""

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project = Path(self.temporary_directory.name)
        self.slot = self.project.joinpath(
            *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts,
            TARGET_DATE,
        )
        self.slot_key = build_slot_key(
            shadow_protocol_sha256=PROTOCOL_SHA256,
            target_official_date=TARGET_DATE,
        )
        self.batch_id = build_batch_id(
            slot_key=self.slot_key,
            execution_manifest_sha256=EXECUTION_MANIFEST_SHA256,
            model_artifact_sha256=MODEL_ARTIFACT_SHA256,
        )

    def _inspect(self):
        return inspect_shadow_prediction_slot(
            TARGET_DATE,
            shadow_protocol_sha256=PROTOCOL_SHA256,
            execution_manifest_sha256=EXECUTION_MANIFEST_SHA256,
            model_artifact_sha256=MODEL_ARTIFACT_SHA256,
            project_directory=self.project,
        )

    def _reserved(self) -> dict[str, object]:
        return {
            "marker_schema_version": 1,
            "batch_id": self.batch_id,
            "slot_key": self.slot_key,
            "target_official_date": TARGET_DATE,
            "reserved_at_utc": "2026-09-02T18:00:00Z",
            "shadow_protocol_sha256": PROTOCOL_SHA256,
            "execution_manifest_sha256": EXECUTION_MANIFEST_SHA256,
            "runtime_code_commit": "4" * 40,
        }

    def _receipt(self) -> dict[str, object]:
        receipt: dict[str, object] = {
            "receipt_schema_version": 1,
            **{
                section: {key: None for key in keys}
                for section, keys in RECEIPT_SECTION_KEYS.items()
            },
        }
        batch = receipt["batch"]
        lineage = receipt["lineage"]
        times = receipt["times"]
        counts = receipt["counts"]
        source = receipt["source"]
        output_hashes = receipt["output_hashes"]
        attestations = receipt["negative_attestations"]
        assert isinstance(batch, dict)
        assert isinstance(lineage, dict)
        assert isinstance(times, dict)
        assert isinstance(counts, dict)
        assert isinstance(source, dict)
        assert isinstance(output_hashes, dict)
        assert isinstance(attestations, dict)
        batch.update(
            {
                "batch_id": self.batch_id,
                "slot_key": self.slot_key,
                "target_official_date": TARGET_DATE,
                "status": "COMPLETED_NO_ELIGIBLE_GAMES",
                "earliest_predicted_scheduled_start_utc": None,
            }
        )
        lineage.update(
            {
                "runtime_code_commit": "4" * 40,
                "shadow_protocol_sha256": PROTOCOL_SHA256,
                "execution_manifest_sha256": EXECUTION_MANIFEST_SHA256,
                "model_artifact_sha256": MODEL_ARTIFACT_SHA256,
            }
        )
        times.update(
            {
                "reserved_at_utc": "2026-09-02T18:00:00Z",
                "receipt_finalized_at_utc": "2026-09-02T19:00:00Z",
            }
        )
        counts.update(
            {
                "schedule_games": 0,
                "eligible_games": 0,
                "predicted_games": 0,
                "excluded_games_by_reason": {
                    key: 0 for key in EXCLUSION_KEYS
                },
            }
        )
        source.update(
            {
                "sqlite_snapshot_sha256": "5" * 64,
                "source_snapshot_sha256": "6" * 64,
                "schedule_raw_archive_sha256": "7" * 64,
            }
        )
        output_hashes.update(
            {
                "activation_reverification_evidence_sha256": "8" * 64,
                "candidate_ledger_sha256": "9" * 64,
                "features_sha256": "a" * 64,
                "predictions_sha256": "b" * 64,
            }
        )
        attestations.update({key: True for key in attestations})
        return receipt

    def _completed(self, receipt_bytes: bytes) -> dict[str, object]:
        return {
            "marker_schema_version": 1,
            "batch_id": self.batch_id,
            "receipt_path": (
                SHADOW_RESULT_ROOT_RELATIVE_PATH
                / TARGET_DATE
                / "receipt.json"
            ).as_posix(),
            "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "completed_at_utc": "2026-09-02T19:01:00Z",
        }

    def _write_completed_slot(self) -> None:
        self.slot.mkdir(parents=True)
        for name in DATA_FILENAMES:
            (self.slot / name).write_bytes(f"opaque:{name}\n".encode("ascii"))
        (self.slot / "RESERVED").write_bytes(
            _canonical_json_file_bytes(self._reserved())
        )
        receipt_bytes = _canonical_json_file_bytes(self._receipt())
        (self.slot / "receipt.json").write_bytes(receipt_bytes)
        (self.slot / "COMPLETED").write_bytes(
            _canonical_json_file_bytes(self._completed(receipt_bytes))
        )

    def _replace_receipt(self, receipt: dict[str, object]) -> None:
        receipt_bytes = _canonical_json_file_bytes(receipt)
        (self.slot / "receipt.json").write_bytes(receipt_bytes)
        completed = self._completed(receipt_bytes)
        (self.slot / "COMPLETED").write_bytes(
            _canonical_json_file_bytes(completed)
        )

    def _reset_slot(self) -> None:
        if self.slot.exists():
            for path in self.slot.iterdir():
                if path.is_dir() and not path.is_symlink():
                    path.rmdir()
                else:
                    path.unlink()
            self.slot.rmdir()

    def test_absent_slot_derives_identifiers_without_creating_anything(
        self,
    ) -> None:
        before = list(self.project.rglob("*"))

        inspection = self._inspect()

        self.assertEqual(inspection.state, ShadowPredictionSlotState.ABSENT)
        self.assertEqual(inspection.slot_key, self.slot_key)
        self.assertEqual(inspection.batch_id, self.batch_id)
        self.assertEqual(inspection.slot_path, self.slot)
        self.assertIsNone(inspection.receipt)
        self.assertEqual(list(self.project.rglob("*")), before)

    def test_exact_completed_slot_reads_only_three_json_documents(self) -> None:
        self._write_completed_slot()
        before = {
            path.name: path.read_bytes() for path in self.slot.iterdir()
        }
        original_read_bytes = Path.read_bytes
        read_names: list[str] = []

        def guarded_read_bytes(path: Path) -> bytes:
            if path.name in DATA_FILENAMES:
                raise AssertionError(f"donnee ouverte : {path.name}")
            read_names.append(path.name)
            return original_read_bytes(path)

        with patch.object(Path, "read_bytes", new=guarded_read_bytes):
            inspection = self._inspect()

        self.assertEqual(
            inspection.state,
            ShadowPredictionSlotState.COMPLETED_EXACT,
        )
        self.assertEqual(inspection.receipt, self._receipt())
        self.assertCountEqual(
            read_names,
            ["RESERVED", "receipt.json", "COMPLETED"],
        )
        after = {
            path.name: path.read_bytes() for path in self.slot.iterdir()
        }
        self.assertEqual(after, before)

    def test_regular_failed_marker_consumes_slot_without_json_read(self) -> None:
        self.slot.mkdir(parents=True)
        (self.slot / "FAILED.json").write_bytes(b"not read by inspection\n")
        with patch.object(
            Path,
            "read_bytes",
            side_effect=AssertionError("FAILED.json must not be read"),
        ):
            inspection = self._inspect()

        self.assertEqual(
            inspection.state,
            ShadowPredictionSlotState.FAILED_CONSUMED,
        )
        self.assertIsNone(inspection.receipt)

    def test_nonterminal_or_non_directory_slot_is_incomplete(self) -> None:
        self.slot.mkdir(parents=True)
        self.assertEqual(
            self._inspect().state,
            ShadowPredictionSlotState.INCOMPLETE_CONSUMED,
        )
        self._reset_slot()
        self.slot.parent.mkdir(parents=True, exist_ok=True)
        self.slot.write_bytes(b"not a directory")
        self.assertEqual(
            self._inspect().state,
            ShadowPredictionSlotState.INCOMPLETE_CONSUMED,
        )

    def test_completed_requires_exactly_all_eight_regular_outputs(self) -> None:
        for missing_name in set(SHADOW_SLOT_SUCCESS_FILENAMES) - {"COMPLETED"}:
            with self.subTest(missing=missing_name):
                self._write_completed_slot()
                (self.slot / missing_name).unlink()
                self.assertEqual(
                    self._inspect().state,
                    ShadowPredictionSlotState.COMPLETED_MISMATCH,
                )
                self._reset_slot()

        self._write_completed_slot()
        (self.slot / "FAILED.json").write_bytes(b"{}\n")
        self.assertEqual(
            self._inspect().state,
            ShadowPredictionSlotState.COMPLETED_MISMATCH,
        )

    def test_each_completed_identity_field_must_match(self) -> None:
        identity_fields = {
            "slot_key": ("batch", "slot_key", "f" * 64),
            "target_official_date": (
                "batch",
                "target_official_date",
                "2026-09-04",
            ),
            "shadow_protocol_sha256": (
                "lineage",
                "shadow_protocol_sha256",
                "f" * 64,
            ),
            "execution_manifest_sha256": (
                "lineage",
                "execution_manifest_sha256",
                "f" * 64,
            ),
            "model_artifact_sha256": (
                "lineage",
                "model_artifact_sha256",
                "f" * 64,
            ),
        }
        for name, (section_name, field, wrong_value) in identity_fields.items():
            with self.subTest(field=name):
                self._write_completed_slot()
                receipt = self._receipt()
                section = receipt[section_name]
                assert isinstance(section, dict)
                section[field] = wrong_value
                self._replace_receipt(receipt)

                self.assertEqual(
                    self._inspect().state,
                    ShadowPredictionSlotState.COMPLETED_MISMATCH,
                )
                self._reset_slot()

    def test_json_canonicality_and_exact_keys_are_mandatory(self) -> None:
        for filename in ("RESERVED", "receipt.json", "COMPLETED"):
            with self.subTest(noncanonical=filename):
                self._write_completed_slot()
                path = self.slot / filename
                value = json.loads(path.read_bytes())
                path.write_text(
                    json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if filename == "receipt.json":
                    receipt_bytes = path.read_bytes()
                    (self.slot / "COMPLETED").write_bytes(
                        _canonical_json_file_bytes(
                            self._completed(receipt_bytes)
                        )
                    )
                self.assertEqual(
                    self._inspect().state,
                    ShadowPredictionSlotState.COMPLETED_MISMATCH,
                )
                self._reset_slot()

        for filename in ("RESERVED", "COMPLETED"):
            with self.subTest(extra_key=filename):
                self._write_completed_slot()
                path = self.slot / filename
                value = json.loads(path.read_bytes())
                value["unexpected"] = True
                path.write_bytes(_canonical_json_file_bytes(value))
                self.assertEqual(
                    self._inspect().state,
                    ShadowPredictionSlotState.COMPLETED_MISMATCH,
                )
                self._reset_slot()

        self._write_completed_slot()
        receipt = self._receipt()
        receipt["unexpected"] = True
        self._replace_receipt(receipt)
        self.assertEqual(
            self._inspect().state,
            ShadowPredictionSlotState.COMPLETED_MISMATCH,
        )
        self._reset_slot()

        for section_name in RECEIPT_SECTION_KEYS:
            with self.subTest(receipt_extra_key=section_name):
                self._write_completed_slot()
                receipt = self._receipt()
                section = receipt[section_name]
                assert isinstance(section, dict)
                section["unexpected"] = True
                self._replace_receipt(receipt)
                self.assertEqual(
                    self._inspect().state,
                    ShadowPredictionSlotState.COMPLETED_MISMATCH,
                )
                self._reset_slot()

    def test_duplicate_key_invalid_hash_and_invalid_path_are_mismatches(
        self,
    ) -> None:
        self._write_completed_slot()
        (self.slot / "COMPLETED").write_bytes(
            b'{"marker_schema_version":1,"marker_schema_version":1}\n'
        )
        self.assertEqual(
            self._inspect().state,
            ShadowPredictionSlotState.COMPLETED_MISMATCH,
        )
        self._reset_slot()

        for field, value in (
            ("receipt_sha256", "0" * 64),
            ("receipt_path", "receipt.json"),
        ):
            with self.subTest(field=field):
                self._write_completed_slot()
                completed_path = self.slot / "COMPLETED"
                completed = json.loads(completed_path.read_bytes())
                completed[field] = value
                completed_path.write_bytes(
                    _canonical_json_file_bytes(completed)
                )
                self.assertEqual(
                    self._inspect().state,
                    ShadowPredictionSlotState.COMPLETED_MISMATCH,
                )
                self._reset_slot()

    def test_lstat_rejects_symbolic_output_without_following_it(self) -> None:
        self._write_completed_slot()
        output = self.slot / "features.csv"
        original_lstat = Path.lstat

        def symbolic_output_lstat(path: Path):
            if path == output:
                return SimpleNamespace(st_mode=stat.S_IFLNK | 0o777)
            return original_lstat(path)

        with patch.object(Path, "lstat", new=symbolic_output_lstat):
            self.assertEqual(
                self._inspect().state,
                ShadowPredictionSlotState.COMPLETED_MISMATCH,
            )

    def test_lstat_rejects_symbolic_result_ancestor(self) -> None:
        self._write_completed_slot()
        result_ancestor = self.project / "shadow_results"
        original_lstat = Path.lstat

        def symbolic_ancestor_lstat(path: Path):
            if path == result_ancestor:
                return SimpleNamespace(st_mode=stat.S_IFLNK | 0o777)
            return original_lstat(path)

        with patch.object(Path, "lstat", new=symbolic_ancestor_lstat):
            self.assertEqual(
                self._inspect().state,
                ShadowPredictionSlotState.INCOMPLETE_CONSUMED,
            )

    def test_lstat_rejects_windows_junction_reparse_point(self) -> None:
        self._write_completed_slot()
        result_ancestor = self.project / "shadow_results"
        original_lstat = Path.lstat

        def junction_ancestor_lstat(path: Path):
            if path == result_ancestor:
                return SimpleNamespace(
                    st_mode=stat.S_IFDIR | 0o777,
                    st_file_attributes=0x0400,
                )
            return original_lstat(path)

        with patch.object(Path, "lstat", new=junction_ancestor_lstat):
            self.assertEqual(
                self._inspect().state,
                ShadowPredictionSlotState.INCOMPLETE_CONSUMED,
            )

    def test_invalid_identity_input_fails_without_slot_access(self) -> None:
        with self.assertRaises(ShadowPredictionError):
            inspect_shadow_prediction_slot(
                TARGET_DATE,
                shadow_protocol_sha256="A" * 64,
                execution_manifest_sha256=EXECUTION_MANIFEST_SHA256,
                model_artifact_sha256=MODEL_ARTIFACT_SHA256,
                project_directory=self.project,
            )
        self.assertEqual(list(self.project.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
