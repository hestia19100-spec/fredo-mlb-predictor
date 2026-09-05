"""Publication canonique du septieme fichier shadow v2, sans COMPLETED."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src import shadow_prediction as shadow

try:
    from tests import test_shadow_prediction_predictions_publication as prediction_tests
except ImportError:
    import test_shadow_prediction_predictions_publication as prediction_tests


RECEIPT_FINALIZED_AT = datetime(
    2026, 9, 3, 12, 0, 1, tzinfo=timezone.utc
)
RECEIPT_FINALIZED_TEXT = "2026-09-03T12:00:01Z"
MANIFEST_INTRODUCTION_COMMIT = "6" * 40
ACTIVATION_SHA256 = "8" * 64
SERVICE_SHA256 = "9" * 64


class ShadowReceiptPublicationTests(unittest.TestCase):
    """Le recu relie les six preuves sans fermer le slot."""

    def setUp(self) -> None:
        self.fixture = prediction_tests.ShadowPredictionsPublicationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.project = self.fixture.project
        self.slot = self.fixture.slot
        self.predictions = self.fixture._publish()
        self.context = self._context()

    def _context(self, **changes):
        values = {
            "started_at_utc": "2026-09-02T08:00:01Z",
            "execution_manifest_introduction_commit": (
                MANIFEST_INTRODUCTION_COMMIT
            ),
            "activation_sha256": ACTIVATION_SHA256,
            "activation_verified_at_utc": "2026-09-02T08:00:00Z",
            "minimum_target_official_date": "2026-09-03",
            "shadow_service_module_sha256": SERVICE_SHA256,
            "runtime_versions": prediction_tests.RUNTIME_VERSIONS,
        }
        values.update(changes)
        return shadow.ShadowReceiptExecutionContext(**values)

    def _publish(self, *, context=None, instant=RECEIPT_FINALIZED_AT):
        selected = self.context if context is None else context
        with patch.object(shadow, "_utc_now", return_value=instant):
            return shadow._build_and_publish_shadow_receipt(
                self.fixture.fixture.reservation,
                self.fixture.fixture.activation_publication,
                self.fixture.source,
                self.fixture.candidates,
                self.predictions,
                selected,
                project_directory=self.project,
            )

    def _receipt_path(self) -> Path:
        return self.slot / "receipt.json"

    def _assert_no_receipt(self) -> None:
        self.assertFalse(self._receipt_path().exists())

    def _read_receipt(self):
        content = self._receipt_path().read_bytes()
        return content, json.loads(content)

    def test_smoke_exact_file_set_and_nonterminal_publication(self):
        before = {path.name: path.read_bytes() for path in self.slot.iterdir()}
        result = self._publish()
        content, receipt = self._read_receipt()
        self.assertEqual(result.receipt_path, self._receipt_path())
        self.assertEqual(result.receipt_sha256, hashlib.sha256(content).hexdigest())
        self.assertEqual(result.receipt_size_bytes, len(content))
        self.assertEqual(result.receipt_finalized_at_utc, RECEIPT_FINALIZED_TEXT)
        self.assertEqual(result.batch_status, "COMPLETED_WITH_PREDICTIONS")
        self.assertEqual(result.predicted_game_count, 2)
        self.assertTrue(result.official_prediction_created)
        self.assertTrue(result.receipt_created)
        self.assertFalse(result.slot_completed)
        self.assertEqual(receipt["receipt_schema_version"], 1)
        self.assertEqual(
            set(path.name for path in self.slot.iterdir()),
            {
                "RESERVED",
                "activation_reverification.remote.json.gz",
                "source_snapshot.json.gz",
                "candidate_ledger.csv",
                "features.csv",
                "predictions.csv",
                "receipt.json",
            },
        )
        self.assertFalse((self.slot / "COMPLETED").exists())
        for name, original in before.items():
            self.assertEqual((self.slot / name).read_bytes(), original, name)
        with self.assertRaises(FrozenInstanceError):
            result.receipt_size_bytes = 0

    def test_receipt_has_exact_canonical_schema_and_lineage(self):
        result = self._publish()
        content, receipt = self._read_receipt()
        reserved = self.fixture.fixture.reservation.reserved_marker
        activation = self.fixture.fixture.activation_publication
        source_payload = json.loads(
            gzip.decompress(self.fixture.source.snapshot_path.read_bytes())
        )
        ingestion = source_payload["schedule_ingestion"]
        sqlite_snapshot = source_payload["sqlite_snapshot"]

        self.assertEqual(set(receipt), shadow._RECEIPT_TOP_LEVEL_KEYS)
        for section, keys in shadow._RECEIPT_SECTION_KEYS.items():
            self.assertEqual(set(receipt[section]), keys, section)
        self.assertEqual(content, shadow._canonical_json_file_bytes(receipt))
        self.assertTrue(content.endswith(b"\n"))
        self.assertNotIn(b"\r", content)
        self.assertEqual(
            result.receipt_relative_path,
            "shadow_results/logistic_team_form_v1_platt_shadow_v2/"
            "2026-09-03/receipt.json",
        )
        self.assertEqual(receipt["batch"], {
            "batch_id": self.fixture.fixture.reservation.batch_id,
            "slot_key": self.fixture.fixture.reservation.slot_key,
            "target_official_date": "2026-09-03",
            "status": "COMPLETED_WITH_PREDICTIONS",
            "earliest_predicted_scheduled_start_utc": "2026-09-03T18:00:00Z",
        })
        self.assertEqual(receipt["lineage"], {
            "runtime_code_commit": reserved["runtime_code_commit"],
            "shadow_service_module_sha256": SERVICE_SHA256,
            "shadow_protocol_sha256": shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
            "execution_manifest_sha256": reserved["execution_manifest_sha256"],
            "model_artifact_sha256": shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
            "artifact_manifest_sha256": shadow.EXPECTED_ARTIFACT_MANIFEST_SHA256,
            "model_protocol_sha256": shadow.EXPECTED_MODEL_PROTOCOL_SHA256,
            "evaluation_protocol_sha256": shadow.EXPECTED_EVALUATION_PROTOCOL_SHA256,
            "evaluation_report_sha256": shadow.EXPECTED_EVALUATION_REPORT_SHA256,
            "evaluation_results_commit": shadow.EXPECTED_EVALUATION_RESULTS_COMMIT,
        })
        self.assertEqual(receipt["source"], {
            "sqlite_snapshot_sha256": sqlite_snapshot["sha256"],
            "sqlite_snapshot_size_bytes": sqlite_snapshot["size_bytes"],
            "source_snapshot_path": self.fixture.source.snapshot_relative_path,
            "source_snapshot_sha256": self.fixture.source.snapshot_sha256,
            "schedule_ingestion_run_id": ingestion["run_id"],
            "schedule_source": ingestion["source"],
            "schedule_requested_start_date": ingestion["requested_start_date"],
            "schedule_requested_end_date": ingestion["requested_end_date"],
            "schedule_game_types": ingestion["game_types"],
            "schedule_request_parameters_json": ingestion["request_parameters_json"],
            "schedule_ingestion_completed_at_utc": ingestion["completed_at_utc"],
            "schedule_raw_archive_path": ingestion["raw_archive_path"],
            "schedule_raw_archive_sha256": ingestion["raw_archive_sha256"],
        })
        self.assertEqual(receipt["output_hashes"], {
            "activation_reverification_evidence_sha256": activation.evidence_sha256,
            "candidate_ledger_sha256": self.fixture.candidates.candidate_ledger_sha256,
            "features_sha256": self.fixture.candidates.features_sha256,
            "predictions_sha256": self.predictions.predictions_sha256,
        })

    def test_times_activation_and_http_proofs_are_exact(self):
        self._publish()
        _, receipt = self._read_receipt()
        activation = self.fixture.fixture.activation_publication
        raw_activation = json.loads(
            gzip.decompress(activation.evidence_path.read_bytes())
        )
        source_payload = json.loads(
            gzip.decompress(self.fixture.source.snapshot_path.read_bytes())
        )
        ingestion = source_payload["schedule_ingestion"]
        self.assertEqual(receipt["times"], {
            "started_at_utc": "2026-09-02T08:00:01Z",
            "reserved_at_utc": "2026-09-03T09:59:00Z",
            "schedule_observed_at_utc": "2026-09-03T10:00:02Z",
            "information_cutoff_utc": "2026-09-03T10:00:03Z",
            "issued_at_utc": prediction_tests.ISSUED_TEXT,
            "receipt_finalized_at_utc": RECEIPT_FINALIZED_TEXT,
            "mlb_http_date_utc": "2026-09-03T10:00:01Z",
            "mlb_http_response_received_at_utc": "2026-09-03T10:00:02Z",
            "clock_skew_seconds": 1,
            "schedule_age_seconds": 1,
        })
        self.assertEqual(receipt["schedule_http_response"], {
            "effective_url": ingestion["response_effective_url"],
            "status_code": ingestion["response_status_code"],
            "redirect_count": ingestion["response_redirect_count"],
            "date_header_raw": ingestion["mlb_http_date_header_raw"],
            "date_header_utc": ingestion["mlb_http_date_utc"],
            "received_at_utc": ingestion["mlb_http_response_received_at_utc"],
            "body_sha256": ingestion["response_body_sha256"],
        })
        self.assertEqual(receipt["activation"], {
            "execution_manifest_introduction_commit": MANIFEST_INTRODUCTION_COMMIT,
            "activation_introduction_commit": activation.activation_introduction_commit,
            "activation_path": shadow.ACTIVATION_RELATIVE_PATH.as_posix(),
            "activation_sha256": ACTIVATION_SHA256,
            "activation_verified_at_utc": "2026-09-02T08:00:00Z",
            "activation_remote_reverified_at_utc": activation.activation_remote_reverified_at_utc,
            "activation_remote_ref": shadow.GITHUB_REMOTE_REF,
            "activation_remote_reverification_query_url": raw_activation["request_url"],
            "activation_remote_reverification_effective_url": raw_activation["effective_url"],
            "activation_remote_reverification_status_code": raw_activation["response_status_code"],
            "activation_remote_reverification_redirect_count": raw_activation["response_redirect_count"],
            "activation_remote_reverification_response_received_at_utc": raw_activation["response_received_at_utc"],
            "activation_remote_reverification_response_body_sha256": raw_activation["response_body_sha256"],
            "activation_remote_reverification_evidence_path": activation.evidence_relative_path,
            "activation_remote_reverification_evidence_sha256": activation.evidence_sha256,
            "minimum_target_official_date": "2026-09-03",
        })

    def test_counts_model_invariants_attestations_and_runtime_are_exact(self):
        self._publish()
        _, receipt = self._read_receipt()
        exclusions = dict(self.fixture.candidates.excluded_games_by_reason)
        self.assertEqual(receipt["counts"], {
            "schedule_games": self.fixture.candidates.candidate_row_count,
            "eligible_games": self.fixture.candidates.eligible_game_count,
            "predicted_games": self.predictions.prediction_row_count,
            "excluded_games_by_reason": exclusions,
        })
        self.assertEqual(
            receipt["counts"]["schedule_games"],
            receipt["counts"]["eligible_games"] + sum(exclusions.values()),
        )
        self.assertEqual(receipt["model_invariants"], {
            "fit_calls": 0,
            "partial_fit_calls": 0,
            "recalibration_calls": 0,
            "threshold_tuning_calls": 0,
            "feature_selection_calls": 0,
            "predict_proba_calls": 1,
            "artifact_state_sha256_before": prediction_tests.STATE_HASH,
            "artifact_state_sha256_after": prediction_tests.STATE_HASH,
            "artifact_state_unchanged": True,
            "warning_policy_id": "VERIFIED_JOBLIB_NUMPY_COMPATIBILITY_V1",
            "approved_compatibility_warning_count": 3,
            "unexpected_warning_count": 0,
        })
        self.assertEqual(
            receipt["negative_attestations"],
            {name: True for name in shadow._RECEIPT_SECTION_KEYS["negative_attestations"]},
        )
        self.assertEqual(
            receipt["runtime_versions"], dict(prediction_tests.RUNTIME_VERSIONS)
        )

    def test_rows_remain_bound_across_candidate_features_and_predictions(self):
        self._publish()
        with self.fixture.candidates.candidate_ledger_path.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            candidate_rows = list(csv.DictReader(handle))
        with self.fixture.candidates.features_path.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            feature_rows = list(csv.DictReader(handle))
        with self.predictions.predictions_path.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            prediction_rows = list(csv.DictReader(handle))
        eligible_ids = [
            row["game_id"]
            for row in candidate_rows
            if row["eligibility_status"] == "ELIGIBLE"
        ]
        self.assertEqual(eligible_ids, [row["game_id"] for row in feature_rows])
        self.assertEqual(eligible_ids, [row["game_id"] for row in prediction_rows])
        self.assertEqual(len(set(eligible_ids)), len(eligible_ids))
        for row in prediction_rows:
            home = float(row["p_home_win"])
            away = float(row["p_away_win"])
            self.assertEqual(away, 1.0 - home)
            self.assertEqual(home + away, 1.0)

    def test_empty_batch_has_header_only_outputs_and_no_model_trace(self):
        fixture = prediction_tests.candidate_tests.ShadowCandidateFeaturesTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        source = fixture._prepare(games=())
        candidates = fixture._build()
        with patch.object(shadow, "_utc_now", return_value=prediction_tests.ISSUED_AT):
            predictions = shadow._build_and_publish_shadow_predictions(
                fixture.reservation,
                fixture.activation_publication,
                source,
                candidates,
                None,
                project_directory=fixture.project,
            )
        context = self._context()
        with patch.object(shadow, "_utc_now", return_value=RECEIPT_FINALIZED_AT):
            result = shadow._build_and_publish_shadow_receipt(
                fixture.reservation,
                fixture.activation_publication,
                source,
                candidates,
                predictions,
                context,
                project_directory=fixture.project,
            )
        receipt = json.loads(result.receipt_path.read_bytes())
        self.assertEqual(result.batch_status, "COMPLETED_NO_ELIGIBLE_GAMES")
        self.assertEqual(result.predicted_game_count, 0)
        self.assertFalse(result.official_prediction_created)
        self.assertIsNone(result.earliest_predicted_scheduled_start_utc)
        self.assertEqual(receipt["model_invariants"]["predict_proba_calls"], 0)
        self.assertIsNone(receipt["model_invariants"]["artifact_state_sha256_before"])
        self.assertIsNone(receipt["model_invariants"]["artifact_state_sha256_after"])
        self.assertEqual(receipt["runtime_versions"], dict(prediction_tests.RUNTIME_VERSIONS))
        self.assertFalse((fixture.slot / "COMPLETED").exists())

    def test_context_type_flags_identifiers_dates_and_runtime_are_closed(self):
        invalid_contexts = [
            {},
            self._context(started_at_utc="2026-09-02T07:59:59Z"),
            self._context(started_at_utc="2026-09-03T10:00:00Z"),
            self._context(execution_manifest_introduction_commit="bad"),
            self._context(activation_sha256="bad"),
            self._context(activation_verified_at_utc="bad"),
            self._context(minimum_target_official_date="2026-08-30"),
            self._context(minimum_target_official_date="2026-09-04"),
            self._context(shadow_service_module_sha256="bad"),
            self._context(runtime_versions=prediction_tests.RUNTIME_VERSIONS[:-1]),
            self._context(runtime_versions=tuple(reversed(prediction_tests.RUNTIME_VERSIONS))),
        ]
        for context in invalid_contexts:
            with self.subTest(context=context):
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._publish(context=context)
                self._assert_no_receipt()
        for name in ("execution_manifest_verified", "activation_verified", "execution_ready"):
            with self.subTest(flag=name):
                context = deepcopy(self.context)
                object.__setattr__(context, name, False)
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._publish(context=context)
                self._assert_no_receipt()

    def test_every_prediction_publication_binding_is_revalidated(self):
        changes = {
            "predictions_sha256": "a" * 64,
            "predictions_size_bytes": self.predictions.predictions_size_bytes + 1,
            "prediction_row_count": self.predictions.prediction_row_count + 1,
            "issued_at_utc": "2026-09-03T12:00:01Z",
            "earliest_predicted_scheduled_start_utc": "2026-09-03T18:00:01Z",
            "artifact_sha256": "a" * 64,
            "protocol_sha256": "a" * 64,
            "code_commit": "a" * 40,
            "artifact_state_sha256_after": "a" * 64,
            "runtime_versions": prediction_tests.RUNTIME_VERSIONS[:-1],
            "predict_proba_calls": 2,
            "predictions_path": self.slot / "other.csv",
            "predictions_relative_path": "other.csv",
        }
        for name, value in changes.items():
            with self.subTest(name=name):
                forged = replace(self.predictions, **{name: value})
                with self.assertRaises(shadow.ShadowPredictionError):
                    with patch.object(shadow, "_utc_now", return_value=RECEIPT_FINALIZED_AT):
                        shadow._build_and_publish_shadow_receipt(
                            self.fixture.fixture.reservation,
                            self.fixture.fixture.activation_publication,
                            self.fixture.source,
                            self.fixture.candidates,
                            forged,
                            self.context,
                            project_directory=self.project,
                        )
                self._assert_no_receipt()

    def test_upstream_publication_proofs_cannot_be_forged(self):
        cases = (
            (
                "activation",
                replace(
                    self.fixture.fixture.activation_publication,
                    evidence_sha256="a" * 64,
                ),
                self.fixture.source,
                self.fixture.candidates,
            ),
            (
                "source",
                self.fixture.fixture.activation_publication,
                replace(self.fixture.source, snapshot_sha256="a" * 64),
                self.fixture.candidates,
            ),
            (
                "candidates",
                self.fixture.fixture.activation_publication,
                self.fixture.source,
                replace(
                    self.fixture.candidates,
                    candidate_ledger_sha256="a" * 64,
                ),
            ),
        )
        for label, activation, source, candidates in cases:
            with self.subTest(label=label):
                with patch.object(shadow, "_utc_now", return_value=RECEIPT_FINALIZED_AT):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        shadow._build_and_publish_shadow_receipt(
                            self.fixture.fixture.reservation,
                            activation,
                            source,
                            candidates,
                            self.predictions,
                            self.context,
                            project_directory=self.project,
                        )
                self._assert_no_receipt()

    def test_each_predecessor_mutation_is_rejected_before_receipt(self):
        paths = (
            self.fixture.fixture.activation_publication.evidence_path,
            self.fixture.source.snapshot_path,
            self.fixture.candidates.candidate_ledger_path,
            self.fixture.candidates.features_path,
            self.predictions.predictions_path,
        )
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"x")
                try:
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._publish()
                    self._assert_no_receipt()
                finally:
                    path.write_bytes(original)

    def test_preexisting_receipt_terminal_or_foreign_path_is_never_modified(self):
        for name in ("receipt.json", "COMPLETED", "FAILED.json", "foreign.txt"):
            with self.subTest(name=name):
                path = self.slot / name
                path.write_bytes(b"foreign\n")
                try:
                    with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                        self._publish()
                    self.assertEqual(path.read_bytes(), b"foreign\n")
                finally:
                    path.unlink()
                self._assert_no_receipt()

    def test_second_publication_never_reuses_or_overwrites_receipt(self):
        first = self._publish()
        preserved = first.receipt_path.read_bytes()
        with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
            self._publish()
        self.assertEqual(first.receipt_path.read_bytes(), preserved)

    def test_failure_before_link_keeps_exact_six_predecessors(self):
        with patch.object(
            shadow,
            "_publish_exclusive_verified",
            side_effect=OSError("synthetic pre-link failure"),
        ):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._publish()
        self._assert_no_receipt()
        self.assertEqual(len(tuple(self.slot.iterdir())), 6)

    def test_failure_after_link_preserves_receipt_and_forbids_repair(self):
        actual = shadow._publish_exclusive_verified

        def publish_then_fail(destination, content):
            digest = actual(destination, content)
            if destination.name == "receipt.json":
                raise OSError("synthetic post-link failure")
            return digest

        with patch.object(
            shadow, "_publish_exclusive_verified", side_effect=publish_then_fail
        ):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._publish()
        preserved = self._receipt_path().read_bytes()
        with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
            self._publish()
        self.assertEqual(self._receipt_path().read_bytes(), preserved)
        self.assertFalse((self.slot / "COMPLETED").exists())

    def test_predecessor_mutation_after_link_leaves_terminal_partial_slot(self):
        actual = shadow._publish_exclusive_verified
        predictions_path = self.predictions.predictions_path

        def publish_then_mutate(destination, content):
            digest = actual(destination, content)
            if destination.name == "receipt.json":
                predictions_path.write_bytes(predictions_path.read_bytes() + b"x")
            return digest

        with patch.object(
            shadow, "_publish_exclusive_verified", side_effect=publish_then_mutate
        ):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._publish()
        preserved = self._receipt_path().read_bytes()
        with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
            self._publish()
        self.assertEqual(self._receipt_path().read_bytes(), preserved)

    def test_concurrent_publishers_have_exactly_one_winner(self):
        def attempt(_):
            try:
                return ("ok", self._publish().receipt_sha256)
            except shadow.ShadowPredictionSlotConsumedError as error:
                return ("consumed", type(error).__name__)

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(attempt, range(2)))
        self.assertEqual([kind for kind, _ in outcomes].count("ok"), 1)
        self.assertEqual([kind for kind, _ in outcomes].count("consumed"), 1)
        self.assertEqual(len(tuple(self.slot.iterdir())), 7)
        self.assertFalse((self.slot / "COMPLETED").exists())

    def test_receipt_time_is_captured_once_after_all_data_predecessors(self):
        with patch.object(shadow, "_utc_now", return_value=RECEIPT_FINALIZED_AT) as clock:
            shadow._build_and_publish_shadow_receipt(
                self.fixture.fixture.reservation,
                self.fixture.fixture.activation_publication,
                self.fixture.source,
                self.fixture.candidates,
                self.predictions,
                self.context,
                project_directory=self.project,
            )
        clock.assert_called_once_with()

    def test_receipt_before_issued_or_after_two_hour_limit_is_rejected(self):
        invalid = (
            datetime(2026, 9, 3, 11, 59, 59, tzinfo=timezone.utc),
            datetime(2026, 9, 3, 16, 0, 1, tzinfo=timezone.utc),
        )
        for instant in invalid:
            with self.subTest(instant=instant):
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._publish(instant=instant)
                self._assert_no_receipt()

    def test_two_hour_boundary_is_inclusive_for_receipt(self):
        result = self._publish(
            instant=datetime(2026, 9, 3, 16, 0, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(result.receipt_finalized_at_utc, "2026-09-03T16:00:00Z")

    def test_receipt_stage_never_loads_calls_or_trains_model_or_reads_sources(self):
        guards = (
            "verify_frozen_model_prerequisites",
            "_load_frozen_shadow_model",
            "_snapshot_frozen_shadow_model_state",
            "_predict_frozen_shadow_model_once",
            "run_observed_schedule_ingestion",
        )
        stack = []
        for name in guards:
            mocked = patch.object(
                shadow, name, side_effect=AssertionError(f"appel interdit: {name}")
            )
            stack.append(mocked)
            mocked.start()
            self.addCleanup(mocked.stop)
        with patch.object(
            shadow.sqlite3, "connect", side_effect=AssertionError("SQLite interdit")
        ), patch.object(
            shadow.requests, "get", side_effect=AssertionError("reseau interdit")
        ):
            self._publish()

    def test_private_function_exposes_no_clock_model_network_or_policy_override(self):
        signature = inspect.signature(shadow._build_and_publish_shadow_receipt)
        self.assertEqual(
            tuple(signature.parameters),
            (
                "reservation",
                "activation_publication",
                "source_publication",
                "candidate_publication",
                "predictions_publication",
                "execution_context",
                "project_directory",
            ),
        )
        forbidden = {
            "now", "clock", "model", "artifact", "network", "session",
            "database_path", "protocol", "receipt", "overwrite", "repair",
            "completed_at_utc",
        }
        self.assertTrue(forbidden.isdisjoint(signature.parameters))
        self.assertTrue(shadow._build_and_publish_shadow_receipt.__name__.startswith("_"))
        source = inspect.getsource(shadow._build_and_publish_shadow_receipt)
        self.assertNotIn("COMPLETED_FILENAME", source)
        self.assertNotIn("joblib", source)
        self.assertNotIn("predict_proba(", source)


if __name__ == "__main__":
    unittest.main()
