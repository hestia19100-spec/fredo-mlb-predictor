"""Orchestration fermee d'un slot shadow v2 deja reserve et autorise."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import shadow_prediction as shadow


TARGET_DATE = "2026-09-03"
RESERVED_AT = "2026-09-02T09:00:00Z"
FAILED_AT = datetime(2026, 9, 2, 9, 1, tzinfo=timezone.utc)
RUNTIME_VERSIONS = (
    ("joblib", "1.5.3"),
    ("numpy", "2.5.2"),
    ("pandas", "3.0.5"),
    ("python", "3.12.1"),
    ("scikit_learn", "1.9.0"),
    ("scipy", "1.18.1"),
)
MANIFEST_SHA256 = "1" * 64
RUNTIME_COMMIT = "2" * 40
MANIFEST_COMMIT = "3" * 40
ACTIVATION_COMMIT = "4" * 40
ACTIVATION_SHA256 = "5" * 64
SERVICE_SHA256 = "6" * 64


class SyntheticStageError(RuntimeError):
    """Erreur deterministe injectee dans un jalon unique."""


class ShadowReservedOrchestrationTests(unittest.TestCase):
    """Le coeur reserve appelle chaque primitive une fois et dans l'ordre."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name).resolve()
        self.database = self.project / "data" / "fredo_mlb.db"
        self.data_directory = self.project / "data"
        self.slot = self.project / shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH / TARGET_DATE

        slot_key = shadow.build_slot_key(
            shadow_protocol_sha256=shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
            target_official_date=TARGET_DATE,
        )
        batch_id = shadow.build_batch_id(
            slot_key=slot_key,
            execution_manifest_sha256=MANIFEST_SHA256,
            model_artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
        )
        marker = {
            "marker_schema_version": 1,
            "batch_id": batch_id,
            "slot_key": slot_key,
            "target_official_date": TARGET_DATE,
            "reserved_at_utc": RESERVED_AT,
            "shadow_protocol_sha256": shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
            "execution_manifest_sha256": MANIFEST_SHA256,
            "runtime_code_commit": RUNTIME_COMMIT,
        }
        marker_bytes = shadow._canonical_json_file_bytes(marker)
        self.reservation = shadow.ShadowPredictionSlotReservation(
            slot_path=self.slot,
            slot_key=slot_key,
            batch_id=batch_id,
            reserved_marker=marker,
            reserved_marker_sha256=hashlib.sha256(marker_bytes).hexdigest(),
        )
        self.activation = shadow.ShadowActivationReverificationPublication(
            slot_path=self.slot,
            evidence_path=self.slot / shadow.ACTIVATION_REVERIFICATION_FILENAME,
            evidence_relative_path=(
                shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH
                / TARGET_DATE
                / shadow.ACTIVATION_REVERIFICATION_FILENAME
            ).as_posix(),
            evidence_sha256="7" * 64,
            activation_introduction_commit=ACTIVATION_COMMIT,
            activation_remote_ref=shadow.GITHUB_REMOTE_REF,
            activation_remote_reverified_at_utc="2026-09-02T08:29:59Z",
            response_received_at_utc="2026-09-02T08:30:00Z",
            response_body_sha256="8" * 64,
        )
        self.context = shadow.ShadowReceiptExecutionContext(
            started_at_utc="2026-09-02T08:00:00Z",
            execution_manifest_introduction_commit=MANIFEST_COMMIT,
            activation_sha256=ACTIVATION_SHA256,
            activation_verified_at_utc="2026-09-01T12:00:00Z",
            minimum_target_official_date="2026-09-03",
            shadow_service_module_sha256=SERVICE_SHA256,
            runtime_versions=RUNTIME_VERSIONS,
        )
        self.source = shadow.ShadowSourceSnapshotPublication(
            slot_path=self.slot,
            snapshot_path=self.slot / shadow.SOURCE_SNAPSHOT_FILENAME,
            snapshot_relative_path="source",
            snapshot_sha256="9" * 64,
            snapshot_size_bytes=1,
            sqlite_snapshot_sha256="a" * 64,
            sqlite_snapshot_size_bytes=1,
            schedule_ingestion_run_id=1,
            schedule_attempts=1,
            information_cutoff_utc="2026-09-02T09:00:10Z",
        )
        self.candidates = self._candidate_publication(eligible=1)
        self.feature_rows = [["official-feature-row"]]
        self.prerequisites = shadow.ShadowModelPrerequisites(
            artifact_path=self.project / shadow.EXPECTED_MODEL_ARTIFACT_PATH,
            artifact_relative_path=shadow.EXPECTED_MODEL_ARTIFACT_PATH,
            artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
            artifact_size_bytes=shadow.EXPECTED_MODEL_ARTIFACT_SIZE_BYTES,
            artifact_bytes=b"synthetic",
            artifact_manifest_sha256=shadow.EXPECTED_ARTIFACT_MANIFEST_SHA256,
            model_protocol_sha256=shadow.EXPECTED_MODEL_PROTOCOL_SHA256,
            shadow_protocol_sha256=shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
            runtime_versions=RUNTIME_VERSIONS,
        )
        self.loaded = shadow.ShadowLoadedModel(
            artifact=object(),
            artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
            runtime_versions=RUNTIME_VERSIONS,
            approved_deserialization_warning_count=0,
        )
        self.probabilities = object()
        self.predictions = object()
        self.receipt = object()
        self.completion = object()

    def _candidate_publication(self, *, eligible: int):
        return shadow.ShadowCandidateFeaturesPublication(
            slot_path=self.slot,
            candidate_ledger_path=self.slot / shadow.CANDIDATE_LEDGER_FILENAME,
            candidate_ledger_relative_path="candidate",
            candidate_ledger_sha256="b" * 64,
            candidate_ledger_size_bytes=1,
            candidate_row_count=eligible,
            features_path=self.slot / shadow.FEATURES_FILENAME,
            features_relative_path="features",
            features_sha256="c" * 64,
            features_size_bytes=1,
            feature_row_count=eligible,
            eligible_game_count=eligible,
            excluded_games_by_reason=(),
            earliest_eligible_scheduled_start_utc=(
                "2026-09-03T18:00:00Z" if eligible else None
            ),
        )

    @contextmanager
    def _pipeline(
        self,
        *,
        feature_rows=None,
        fail_stage: str | None = None,
        failure_marker_error: BaseException | None = None,
        prerequisites=None,
        loaded=None,
    ):
        rows = self.feature_rows if feature_rows is None else feature_rows
        candidate = self._candidate_publication(eligible=len(rows))
        prereq = self.prerequisites if prerequisites is None else prerequisites
        loaded_model = self.loaded if loaded is None else loaded
        calls: list[str] = []

        def action(name, result):
            def invoke(*_args, **_kwargs):
                calls.append(name)
                if name == fail_stage:
                    raise SyntheticStageError(f"echec {name}")
                return result

            return invoke

        with ExitStack() as stack:
            mocks = {
                "source": stack.enter_context(patch.object(
                    shadow,
                    "capture_and_publish_source_snapshot",
                    side_effect=action("source", self.source),
                )),
                "candidates": stack.enter_context(patch.object(
                    shadow,
                    "build_and_publish_candidate_ledger_and_features",
                    side_effect=action("candidates", candidate),
                )),
                "features": stack.enter_context(patch.object(
                    shadow,
                    "_read_validated_prediction_predecessors",
                    side_effect=action(
                        "features",
                        (b"s", b"c", b"f", rows, TARGET_DATE, "cutoff"),
                    ),
                )),
                "prerequisites": stack.enter_context(patch.object(
                    shadow,
                    "verify_frozen_model_prerequisites",
                    side_effect=action("prerequisites", prereq),
                )),
                "loading": stack.enter_context(patch.object(
                    shadow,
                    "_load_frozen_shadow_model",
                    side_effect=action("loading", loaded_model),
                )),
                "prediction": stack.enter_context(patch.object(
                    shadow,
                    "_predict_frozen_shadow_model_once",
                    side_effect=action("prediction", self.probabilities),
                )),
                "predictions": stack.enter_context(patch.object(
                    shadow,
                    "_build_and_publish_shadow_predictions",
                    side_effect=action("predictions", self.predictions),
                )),
                "receipt": stack.enter_context(patch.object(
                    shadow,
                    "_build_and_publish_shadow_receipt",
                    side_effect=action("receipt", self.receipt),
                )),
                "completion": stack.enter_context(patch.object(
                    shadow,
                    "_complete_shadow_prediction_slot",
                    side_effect=action("completion", self.completion),
                )),
            }

            def fail_slot(*_args, **_kwargs):
                calls.append("failed")
                if failure_marker_error is not None:
                    raise failure_marker_error
                return object()

            mocks["failed"] = stack.enter_context(patch.object(
                shadow,
                "fail_shadow_prediction_slot",
                side_effect=fail_slot,
            ))
            mocks["clock"] = stack.enter_context(patch.object(
                shadow,
                "_utc_now",
                return_value=FAILED_AT,
            ))
            yield calls, mocks, candidate, rows

    def _execute(self, *, context=None):
        return shadow._execute_reserved_shadow_prediction(
            self.reservation,
            self.activation,
            self.context if context is None else context,
            database_path=self.database,
            data_directory=self.data_directory,
            project_directory=self.project,
        )

    def test_nonempty_batch_has_one_exact_ordered_model_path(self):
        with self._pipeline() as (calls, mocks, _candidate, rows):
            result = self._execute()
        self.assertIs(result, self.completion)
        self.assertEqual(calls, [
            "source",
            "candidates",
            "features",
            "prerequisites",
            "loading",
            "prediction",
            "predictions",
            "receipt",
            "completion",
        ])
        mocks["prediction"].assert_called_once_with(
            self.loaded,
            rows,
            project_directory=self.project,
        )
        mocks["predictions"].assert_called_once_with(
            self.reservation,
            self.activation,
            self.source,
            _candidate,
            self.probabilities,
            project_directory=self.project,
        )
        mocks["failed"].assert_not_called()
        mocks["clock"].assert_not_called()

    def test_source_receives_only_the_explicit_official_paths(self):
        with self._pipeline() as (_calls, mocks, _candidate, _rows):
            self._execute()
        mocks["source"].assert_called_once_with(
            self.reservation,
            self.activation,
            database_path=self.database,
            data_directory=self.data_directory,
            project_directory=self.project,
        )

    def test_empty_batch_never_verifies_loads_or_calls_model(self):
        with self._pipeline(feature_rows=[]) as (
            calls,
            mocks,
            candidate,
            _rows,
        ):
            result = self._execute()
        self.assertIs(result, self.completion)
        self.assertEqual(calls, [
            "source",
            "candidates",
            "features",
            "predictions",
            "receipt",
            "completion",
        ])
        for name in ("prerequisites", "loading", "prediction"):
            mocks[name].assert_not_called()
        self.assertEqual(candidate.eligible_game_count, 0)
        self.assertIs(mocks["predictions"].call_args.args[4], None)
        mocks["failed"].assert_not_called()

    def test_receipt_and_completion_receive_the_same_predecessors(self):
        with self._pipeline() as (_calls, mocks, candidate, _rows):
            self._execute()
        mocks["receipt"].assert_called_once_with(
            self.reservation,
            self.activation,
            self.source,
            candidate,
            self.predictions,
            self.context,
            project_directory=self.project,
        )
        mocks["completion"].assert_called_once_with(
            self.reservation,
            self.activation,
            self.source,
            candidate,
            self.predictions,
            self.receipt,
            project_directory=self.project,
        )

    def test_candidate_count_mismatch_fails_before_any_model_or_output(self):
        candidate = self._candidate_publication(eligible=2)
        with self._pipeline() as (calls, mocks, _ignored, _rows):
            mocks["candidates"].side_effect = lambda *_a, **_k: (
                calls.append("candidates") or candidate
            )
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "nombre de lignes",
            ):
                self._execute()
        self.assertEqual(calls, ["source", "candidates", "features", "failed"])
        for name in (
            "prerequisites",
            "loading",
            "prediction",
            "predictions",
            "receipt",
            "completion",
        ):
            mocks[name].assert_not_called()
        self.assertEqual(
            mocks["failed"].call_args.kwargs["stage"],
            shadow._EXECUTION_STAGE_FEATURE_REVALIDATION,
        )

    def test_manifest_runtime_mismatch_fails_before_model_loading(self):
        mismatched = replace(
            self.prerequisites,
            runtime_versions=tuple(
                (name, "different" if name == "numpy" else version)
                for name, version in RUNTIME_VERSIONS
            ),
        )
        with self._pipeline(prerequisites=mismatched) as (calls, mocks, *_):
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "versions du modele",
            ):
                self._execute()
        self.assertEqual(calls[-1], "failed")
        mocks["loading"].assert_not_called()
        mocks["prediction"].assert_not_called()
        self.assertEqual(
            mocks["failed"].call_args.kwargs["stage"],
            shadow._EXECUTION_STAGE_MODEL_PREREQUISITES,
        )

    def test_loaded_runtime_mismatch_fails_before_predict_proba(self):
        mismatched = replace(
            self.loaded,
            runtime_versions=tuple(
                (name, "different" if name == "numpy" else version)
                for name, version in RUNTIME_VERSIONS
            ),
        )
        with self._pipeline(loaded=mismatched) as (calls, mocks, *_):
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "modele charge",
            ):
                self._execute()
        self.assertEqual(calls[-1], "failed")
        mocks["prediction"].assert_not_called()
        self.assertEqual(
            mocks["failed"].call_args.kwargs["stage"],
            shadow._EXECUTION_STAGE_MODEL_LOADING,
        )

    def test_invalid_authorization_fails_before_source_or_model(self):
        invalid = replace(
            self.context,
            minimum_target_official_date="2026-09-04",
        )
        with self._pipeline() as (calls, mocks, *_):
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "borne active",
            ):
                self._execute(context=invalid)
        self.assertEqual(calls, ["failed"])
        mocks["source"].assert_not_called()
        mocks["prerequisites"].assert_not_called()
        self.assertEqual(
            mocks["failed"].call_args.kwargs["stage"],
            shadow._EXECUTION_STAGE_AUTHORIZATION,
        )

    def test_every_stage_error_is_terminal_and_stops_later_calls(self):
        cases = (
            ("source", shadow._EXECUTION_STAGE_SOURCE),
            ("candidates", shadow._EXECUTION_STAGE_CANDIDATES),
            ("features", shadow._EXECUTION_STAGE_FEATURE_REVALIDATION),
            ("prerequisites", shadow._EXECUTION_STAGE_MODEL_PREREQUISITES),
            ("loading", shadow._EXECUTION_STAGE_MODEL_LOADING),
            ("prediction", shadow._EXECUTION_STAGE_MODEL_PREDICTION),
            ("predictions", shadow._EXECUTION_STAGE_PREDICTIONS),
            ("receipt", shadow._EXECUTION_STAGE_RECEIPT),
            ("completion", shadow._EXECUTION_STAGE_COMPLETION),
        )
        for injected, expected_stage in cases:
            with self.subTest(stage=injected):
                with self._pipeline(fail_stage=injected) as (
                    calls,
                    mocks,
                    *_rest,
                ):
                    with self.assertRaisesRegex(
                        SyntheticStageError,
                        f"echec {injected}",
                    ):
                        self._execute()
                self.assertEqual(calls.count(injected), 1)
                self.assertEqual(calls[-1], "failed")
                mocks["failed"].assert_called_once()
                failure = mocks["failed"].call_args
                self.assertIs(failure.args[0], self.reservation)
                self.assertEqual(failure.kwargs["stage"], expected_stage)
                self.assertEqual(
                    failure.kwargs["error_type"],
                    "SyntheticStageError",
                )
                self.assertEqual(
                    failure.kwargs["error_message"],
                    f"echec {injected}",
                )
                self.assertEqual(
                    failure.kwargs["failed_at_utc"],
                    "2026-09-02T09:01:00Z",
                )
                mocks["clock"].assert_called_once_with()

    def test_empty_exception_message_is_replaced_by_canonical_fallback(self):
        with self._pipeline(fail_stage="source") as (_calls, mocks, *_):
            mocks["source"].side_effect = SyntheticStageError()
            with self.assertRaises(SyntheticStageError):
                self._execute()
        self.assertEqual(
            mocks["failed"].call_args.kwargs["error_message"],
            "SyntheticStageError sans message.",
        )

    def test_failure_marker_conflict_never_masks_original_error(self):
        marker_error = shadow.ShadowPredictionSlotConsumedError("winner")
        with self._pipeline(
            fail_stage="source",
            failure_marker_error=marker_error,
        ) as (_calls, mocks, *_):
            with self.assertRaisesRegex(SyntheticStageError, "echec source"):
                self._execute()
        mocks["failed"].assert_called_once()

    def test_ambiguous_failure_marker_error_is_reported_with_original_cause(self):
        marker_error = OSError("failed marker unavailable")
        with self._pipeline(
            fail_stage="source",
            failure_marker_error=marker_error,
        ) as (_calls, mocks, *_):
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "FAILED.json n'a pas pu etre publie",
            ) as raised:
                self._execute()
        self.assertIsInstance(raised.exception.__cause__, SyntheticStageError)
        mocks["failed"].assert_called_once()

    def test_keyboard_interrupt_is_not_hidden_or_converted_to_failed(self):
        with self._pipeline() as (_calls, mocks, *_):
            mocks["source"].side_effect = KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                self._execute()
        mocks["failed"].assert_not_called()
        mocks["clock"].assert_not_called()

    def test_system_exit_is_not_hidden_or_converted_to_failed(self):
        with self._pipeline() as (_calls, mocks, *_):
            mocks["source"].side_effect = SystemExit(2)
            with self.assertRaises(SystemExit):
                self._execute()
        mocks["failed"].assert_not_called()
        mocks["clock"].assert_not_called()

    def test_orchestrator_has_no_retry_clock_model_or_stage_override(self):
        signature = inspect.signature(
            shadow._execute_reserved_shadow_prediction
        )
        self.assertEqual(
            tuple(signature.parameters),
            (
                "reservation",
                "activation_publication",
                "execution_context",
                "database_path",
                "data_directory",
                "project_directory",
            ),
        )
        forbidden = {
            "clock",
            "now",
            "retry",
            "model",
            "predictor",
            "stage",
            "failure_handler",
        }
        self.assertFalse(forbidden & set(signature.parameters))

    def test_orchestrator_does_not_call_standalone_state_primitives(self):
        guards = (
            "_snapshot_frozen_shadow_model_state",
            "_verify_frozen_shadow_model_state_unchanged",
        )
        with ExitStack() as stack:
            protected = [
                stack.enter_context(patch.object(
                    shadow,
                    name,
                    side_effect=AssertionError(name),
                ))
                for name in guards
            ]
            with self._pipeline():
                self._execute()
        for mocked in protected:
            mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
