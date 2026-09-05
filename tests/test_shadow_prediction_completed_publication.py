"""Publication atomique du huitieme et dernier fichier shadow v2."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import multiprocessing
from pathlib import Path
from queue import Empty
import unittest
from unittest.mock import patch

from src import shadow_prediction as shadow

try:
    from tests import test_shadow_prediction_receipt_publication as receipt_tests
except ImportError:
    import test_shadow_prediction_receipt_publication as receipt_tests


COMPLETED_AT = datetime(2026, 9, 3, 12, 0, 2, tzinfo=timezone.utc)
COMPLETED_TEXT = "2026-09-03T12:00:02Z"


def _completion_process_worker(
    reservation,
    activation,
    source,
    candidates,
    predictions,
    receipt,
    project_directory: str,
    start_gate,
    result_queue,
    index: int,
) -> None:
    """Tente la fermeture depuis un processus independant."""
    try:
        start_gate.wait(timeout=15)
        with patch.object(shadow, "_utc_now", return_value=COMPLETED_AT):
            shadow._complete_shadow_prediction_slot(
                reservation,
                activation,
                source,
                candidates,
                predictions,
                receipt,
                project_directory=Path(project_directory),
            )
    except shadow.ShadowPredictionSlotConsumedError:
        result_queue.put(("consumed", index))
    except BaseException as error:
        result_queue.put(("error", type(error).__name__, str(error)))
    else:
        result_queue.put(("completed", index))


class ShadowCompletedPublicationTests(unittest.TestCase):
    """COMPLETED ferme le slot sans pouvoir modifier ses sept preuves."""

    def setUp(self) -> None:
        self.fixture = receipt_tests.ShadowReceiptPublicationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.project = self.fixture.project
        self.slot = self.fixture.slot
        self.receipt = self.fixture._publish()
        predictions_fixture = self.fixture.fixture
        candidate_fixture = predictions_fixture.fixture
        self.reservation = candidate_fixture.reservation
        self.activation = candidate_fixture.activation_publication
        self.source = predictions_fixture.source
        self.candidates = predictions_fixture.candidates
        self.predictions = self.fixture.predictions

    def _complete(self, *, receipt=None, instant=COMPLETED_AT):
        selected_receipt = self.receipt if receipt is None else receipt
        with patch.object(shadow, "_utc_now", return_value=instant):
            return shadow._complete_shadow_prediction_slot(
                self.reservation,
                self.activation,
                self.source,
                self.candidates,
                self.predictions,
                selected_receipt,
                project_directory=self.project,
            )

    def _completed_path(self) -> Path:
        return self.slot / "COMPLETED"

    def _slot_bytes(self) -> dict[str, bytes]:
        return {path.name: path.read_bytes() for path in self.slot.iterdir()}

    def _new_fixture(self):
        fixture = receipt_tests.ShadowReceiptPublicationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        receipt = fixture._publish()
        predictions_fixture = fixture.fixture
        candidate_fixture = predictions_fixture.fixture
        return (
            fixture,
            receipt,
            candidate_fixture.reservation,
            candidate_fixture.activation_publication,
            predictions_fixture.source,
            predictions_fixture.candidates,
            fixture.predictions,
        )

    def _complete_fixture(
        self,
        fixture_values,
        *,
        instant=COMPLETED_AT,
        receipt_override=None,
    ):
        fixture, receipt, reservation, activation, source, candidates, predictions = (
            fixture_values
        )
        selected_receipt = receipt if receipt_override is None else receipt_override
        with patch.object(shadow, "_utc_now", return_value=instant):
            return shadow._complete_shadow_prediction_slot(
                reservation,
                activation,
                source,
                candidates,
                predictions,
                selected_receipt,
                project_directory=fixture.project,
            )

    def test_smoke_publishes_exact_eighth_file_without_mutating_predecessors(self):
        before = self._slot_bytes()
        result = self._complete()
        content = result.completed_path.read_bytes()
        marker = json.loads(content)
        self.assertEqual(
            set(path.name for path in self.slot.iterdir()),
            set(shadow.SHADOW_SLOT_SUCCESS_FILENAMES),
        )
        for name, original in before.items():
            self.assertEqual((self.slot / name).read_bytes(), original, name)
        self.assertEqual(result.slot_path, self.slot)
        self.assertEqual(result.completed_path, self._completed_path())
        self.assertEqual(result.completed_sha256, hashlib.sha256(content).hexdigest())
        self.assertEqual(result.completed_size_bytes, len(content))
        self.assertEqual(result.completed_at_utc, COMPLETED_TEXT)
        self.assertEqual(result.batch_id, self.reservation.batch_id)
        self.assertEqual(result.receipt_sha256, self.receipt.receipt_sha256)
        self.assertEqual(result.batch_status, "COMPLETED_WITH_PREDICTIONS")
        self.assertEqual(
            result.earliest_predicted_scheduled_start_utc,
            "2026-09-03T18:00:00Z",
        )
        self.assertEqual(result.local_completion_lead_seconds, 21_598)
        self.assertTrue(result.official_prediction_created)
        self.assertTrue(result.receipt_created)
        self.assertTrue(result.slot_completed)
        self.assertEqual(marker["completed_at_utc"], COMPLETED_TEXT)

    def test_marker_is_exact_canonical_and_bound_to_persisted_receipt(self):
        result = self._complete()
        expected = {
            "marker_schema_version": 1,
            "batch_id": self.reservation.batch_id,
            "receipt_path": self.receipt.receipt_relative_path,
            "receipt_sha256": hashlib.sha256(
                self.receipt.receipt_path.read_bytes()
            ).hexdigest(),
            "completed_at_utc": COMPLETED_TEXT,
        }
        expected_bytes = shadow._canonical_json_file_bytes(expected)
        self.assertEqual(result.completed_path.read_bytes(), expected_bytes)
        self.assertEqual(
            result.completed_relative_path,
            "shadow_results/logistic_team_form_v1_platt_shadow_v2/"
            "2026-09-03/COMPLETED",
        )
        self.assertEqual(set(expected), shadow._COMPLETED_MARKER_KEYS)
        self.assertNotIn("score", expected_bytes.decode("utf-8").lower())
        self.assertNotIn("odds", expected_bytes.decode("utf-8").lower())
        self.assertNotIn("metric", expected_bytes.decode("utf-8").lower())

    def test_completed_result_is_frozen(self):
        result = self._complete()
        with self.assertRaises(FrozenInstanceError):
            result.completed_at_utc = "2026-09-03T12:00:03Z"

    def test_completed_slot_is_recognized_as_exact_by_public_inspection(self):
        self._complete()
        inspection = shadow.inspect_shadow_prediction_slot(
            "2026-09-03",
            shadow_protocol_sha256=self.reservation.reserved_marker[
                "shadow_protocol_sha256"
            ],
            execution_manifest_sha256=self.reservation.reserved_marker[
                "execution_manifest_sha256"
            ],
            model_artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
            project_directory=self.project,
        )
        self.assertIs(
            inspection.state,
            shadow.ShadowPredictionSlotState.COMPLETED_EXACT,
        )
        self.assertEqual(
            inspection.receipt["batch"]["batch_id"],
            self.reservation.batch_id,
        )

    def test_empty_batch_completes_without_model_trace_or_lead_value(self):
        fixture_type = (
            receipt_tests.prediction_tests.candidate_tests
            .ShadowCandidateFeaturesTests
        )
        candidate_fixture = fixture_type()
        candidate_fixture.setUp()
        self.addCleanup(candidate_fixture.doCleanups)
        source = candidate_fixture._prepare(games=())
        candidates = candidate_fixture._build()
        with patch.object(
            shadow,
            "_utc_now",
            return_value=receipt_tests.prediction_tests.ISSUED_AT,
        ):
            predictions = shadow._build_and_publish_shadow_predictions(
                candidate_fixture.reservation,
                candidate_fixture.activation_publication,
                source,
                candidates,
                None,
                project_directory=candidate_fixture.project,
            )
        with patch.object(
            shadow,
            "_utc_now",
            return_value=receipt_tests.RECEIPT_FINALIZED_AT,
        ):
            receipt = shadow._build_and_publish_shadow_receipt(
                candidate_fixture.reservation,
                candidate_fixture.activation_publication,
                source,
                candidates,
                predictions,
                self.fixture._context(),
                project_directory=candidate_fixture.project,
            )
        with patch.object(shadow, "_utc_now", return_value=COMPLETED_AT):
            result = shadow._complete_shadow_prediction_slot(
                candidate_fixture.reservation,
                candidate_fixture.activation_publication,
                source,
                candidates,
                predictions,
                receipt,
                project_directory=candidate_fixture.project,
            )
        self.assertEqual(result.batch_status, "COMPLETED_NO_ELIGIBLE_GAMES")
        self.assertIsNone(result.earliest_predicted_scheduled_start_utc)
        self.assertIsNone(result.local_completion_lead_seconds)
        self.assertFalse(result.official_prediction_created)
        self.assertEqual(len(tuple(candidate_fixture.slot.iterdir())), 8)

    def test_receipt_finalization_must_precede_completion(self):
        with self.assertRaises(shadow.ShadowPredictionError):
            self._complete(
                instant=datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)
            )
        self.assertFalse(self._completed_path().exists())

    def test_two_hour_completion_boundary_is_inclusive(self):
        result = self._complete(
            instant=datetime(2026, 9, 3, 16, 0, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(result.local_completion_lead_seconds, 7_200)
        inspection = shadow.inspect_shadow_prediction_slot(
            "2026-09-03",
            shadow_protocol_sha256=self.reservation.reserved_marker[
                "shadow_protocol_sha256"
            ],
            execution_manifest_sha256=self.reservation.reserved_marker[
                "execution_manifest_sha256"
            ],
            model_artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
            project_directory=self.project,
        )
        self.assertIs(
            inspection.state,
            shadow.ShadowPredictionSlotState.COMPLETED_EXACT,
        )

    def test_public_inspection_rejects_a_forged_late_completed_marker(self):
        result = self._complete()
        marker = json.loads(result.completed_path.read_bytes())
        marker["completed_at_utc"] = "2026-09-03T16:00:01Z"
        result.completed_path.write_bytes(
            shadow._canonical_json_file_bytes(marker)
        )
        inspection = shadow.inspect_shadow_prediction_slot(
            "2026-09-03",
            shadow_protocol_sha256=self.reservation.reserved_marker[
                "shadow_protocol_sha256"
            ],
            execution_manifest_sha256=self.reservation.reserved_marker[
                "execution_manifest_sha256"
            ],
            model_artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
            project_directory=self.project,
        )
        self.assertIs(
            inspection.state,
            shadow.ShadowPredictionSlotState.COMPLETED_MISMATCH,
        )

    def test_late_completion_is_rejected_without_terminal_repair(self):
        with self.assertRaises(shadow.ShadowPredictionError):
            self._complete(
                instant=datetime(2026, 9, 3, 16, 0, 1, tzinfo=timezone.utc)
            )
        self.assertEqual(len(tuple(self.slot.iterdir())), 7)
        self.assertFalse(self._completed_path().exists())

    def test_each_receipt_publication_binding_is_revalidated(self):
        invalid = {
            "slot_path": self.project,
            "receipt_path": self.project / "receipt.json",
            "receipt_relative_path": "receipt.json",
            "receipt_sha256": "a" * 64,
            "receipt_size_bytes": self.receipt.receipt_size_bytes + 1,
            "receipt_finalized_at_utc": "2026-09-03T12:00:00Z",
            "batch_status": "COMPLETED_NO_ELIGIBLE_GAMES",
            "schedule_game_count": self.receipt.schedule_game_count + 1,
            "eligible_game_count": self.receipt.eligible_game_count + 1,
            "predicted_game_count": self.receipt.predicted_game_count + 1,
            "earliest_predicted_scheduled_start_utc": "2026-09-03T18:00:01Z",
            "execution_manifest_sha256": "b" * 64,
            "protocol_sha256": "c" * 64,
            "artifact_sha256": "d" * 64,
            "official_prediction_created": False,
        }
        for name, value in invalid.items():
            with self.subTest(name=name):
                forged = replace(self.receipt, **{name: value})
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._complete(receipt=forged)
                self.assertFalse(self._completed_path().exists())

    def test_every_receipt_section_is_revalidated_even_with_matching_new_hash(self):
        mutations = (
            ("activation", "activation_path", "activation.json"),
            ("times", "clock_skew_seconds", 999_999),
            ("schedule_http_response", "status_code", 201),
            ("lineage", "evaluation_report_sha256", "a" * 64),
            ("source", "schedule_source", "forged"),
            ("counts", "schedule_games", 99),
            ("output_hashes", "features_sha256", "b" * 64),
            ("model_invariants", "fit_calls", 1),
            (
                "negative_attestations",
                "ODDS_NOT_READ",
                False,
            ),
            ("runtime_versions", "python", "0.0"),
        )
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                fixture_values = self._new_fixture()
                fixture, receipt = fixture_values[:2]
                original = json.loads(receipt.receipt_path.read_bytes())
                payload = json.loads(json.dumps(original))
                payload[section][key] = value
                content = shadow._canonical_json_file_bytes(payload)
                receipt.receipt_path.write_bytes(content)
                forged = replace(
                    receipt,
                    receipt_sha256=hashlib.sha256(content).hexdigest(),
                    receipt_size_bytes=len(content),
                )
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._complete_fixture(
                        fixture_values,
                        receipt_override=forged,
                    )
                self.assertFalse((fixture.slot / "COMPLETED").exists())

    def test_each_of_the_seven_predecessors_is_revalidated(self):
        for index in range(7):
            with self.subTest(index=index):
                values = self._new_fixture()
                fixture = values[0]
                paths = sorted(fixture.slot.iterdir(), key=lambda path: path.name)
                path = paths[index]
                path.write_bytes(path.read_bytes() + b"x")
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._complete_fixture(values)
                self.assertFalse((fixture.slot / "COMPLETED").exists())

    def test_preexisting_completed_failed_or_foreign_path_is_never_modified(self):
        for name in ("COMPLETED", "FAILED.json", "foreign.bin"):
            with self.subTest(name=name):
                values = self._new_fixture()
                fixture = values[0]
                path = fixture.slot / name
                path.write_bytes(b"preserve")
                with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                    self._complete_fixture(values)
                self.assertEqual(path.read_bytes(), b"preserve")

    def test_second_completion_never_reuses_or_overwrites_marker(self):
        first = self._complete()
        preserved = first.completed_path.read_bytes()
        with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
            self._complete(
                instant=datetime(2026, 9, 3, 12, 0, 3, tzinfo=timezone.utc)
            )
        self.assertEqual(first.completed_path.read_bytes(), preserved)

    def test_concurrent_publishers_have_exactly_one_winner(self):
        def attempt(_):
            try:
                result = shadow._complete_shadow_prediction_slot(
                    self.reservation,
                    self.activation,
                    self.source,
                    self.candidates,
                    self.predictions,
                    self.receipt,
                    project_directory=self.project,
                )
                return ("ok", result.completed_sha256)
            except shadow.ShadowPredictionSlotConsumedError as error:
                return ("consumed", type(error).__name__)

        with patch.object(shadow, "_utc_now", return_value=COMPLETED_AT):
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(attempt, range(2)))
        self.assertEqual([kind for kind, _ in outcomes].count("ok"), 1)
        self.assertEqual([kind for kind, _ in outcomes].count("consumed"), 1)
        self.assertEqual(len(tuple(self.slot.iterdir())), 8)

    def test_concurrent_processes_have_exactly_one_terminal_winner(self):
        context = multiprocessing.get_context("spawn")
        start_gate = context.Event()
        result_queue = context.Queue()
        common = (
            self.reservation,
            self.activation,
            self.source,
            self.candidates,
            self.predictions,
            self.receipt,
            str(self.project),
            start_gate,
            result_queue,
        )
        processes = [
            context.Process(
                target=_completion_process_worker,
                args=common + (index,),
            )
            for index in range(4)
        ]
        for process in processes:
            process.start()
        start_gate.set()
        outcomes = []
        try:
            for _ in processes:
                outcomes.append(result_queue.get(timeout=30))
        except Empty as error:
            self.fail(f"Un processus de fermeture n'a pas repondu : {error}")
        finally:
            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
            result_queue.close()
            result_queue.join_thread()

        self.assertTrue(all(process.exitcode == 0 for process in processes))
        self.assertFalse(
            [outcome for outcome in outcomes if outcome[0] == "error"],
            outcomes,
        )
        self.assertEqual(
            sum(outcome[0] == "completed" for outcome in outcomes),
            1,
        )
        self.assertEqual(
            sum(outcome[0] == "consumed" for outcome in outcomes),
            3,
        )
        self.assertEqual(len(tuple(self.slot.iterdir())), 8)

    def test_failure_before_atomic_link_preserves_exact_seven_files(self):
        before = self._slot_bytes()
        with patch.object(
            shadow,
            "_publish_exclusive_verified",
            side_effect=OSError("before link"),
        ):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._complete()
        self.assertEqual(self._slot_bytes(), before)
        self.assertFalse(self._completed_path().exists())

    def test_failure_after_link_preserves_terminal_and_forbids_repair(self):
        original = shadow._publish_exclusive_verified

        def publish_then_fail(path, content):
            original(path, content)
            raise OSError("after link")

        with patch.object(
            shadow,
            "_publish_exclusive_verified",
            side_effect=publish_then_fail,
        ):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._complete()
        preserved = self._completed_path().read_bytes()
        with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
            self._complete()
        self.assertEqual(self._completed_path().read_bytes(), preserved)

    def test_predecessor_mutation_after_link_preserves_completed_without_repair(self):
        original = shadow._publish_exclusive_verified

        def publish_then_mutate(path, content):
            digest = original(path, content)
            self.receipt.receipt_path.write_bytes(b"mutated after link")
            return digest

        with patch.object(
            shadow,
            "_publish_exclusive_verified",
            side_effect=publish_then_mutate,
        ):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._complete()
        preserved = self._completed_path().read_bytes()
        self.assertTrue(preserved)
        with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
            self._complete()
        self.assertEqual(self._completed_path().read_bytes(), preserved)

    def test_completion_time_is_captured_once_between_receipt_reads(self):
        events: list[str] = []
        original = shadow._read_validated_completion_predecessors

        def read(*args, **kwargs):
            events.append("read")
            return original(*args, **kwargs)

        def clock():
            events.append("clock")
            return COMPLETED_AT

        with patch.object(
            shadow,
            "_read_validated_completion_predecessors",
            side_effect=read,
        ), patch.object(shadow, "_utc_now", side_effect=clock) as clock_mock:
            shadow._complete_shadow_prediction_slot(
                self.reservation,
                self.activation,
                self.source,
                self.candidates,
                self.predictions,
                self.receipt,
                project_directory=self.project,
            )
        clock_mock.assert_called_once_with()
        self.assertEqual(events, ["read", "clock", "read"])

    def test_stage_never_loads_calls_or_trains_model_or_reads_external_sources(self):
        guards = (
            "verify_frozen_model_prerequisites",
            "_load_frozen_shadow_model",
            "_snapshot_frozen_shadow_model_state",
            "_predict_frozen_shadow_model_once",
            "run_observed_schedule_ingestion",
            "capture_and_publish_source_snapshot",
        )
        patchers = []
        for name in guards:
            patcher = patch.object(
                shadow,
                name,
                side_effect=AssertionError(f"appel interdit: {name}"),
            )
            patchers.append(patcher)
            patcher.start()
            self.addCleanup(patcher.stop)
        with patch.object(
            shadow.sqlite3,
            "connect",
            side_effect=AssertionError("SQLite interdit"),
        ), patch.object(
            shadow.requests,
            "get",
            side_effect=AssertionError("reseau interdit"),
        ):
            self._complete()

    def test_private_signature_exposes_no_clock_model_network_or_policy_override(self):
        signature = inspect.signature(shadow._complete_shadow_prediction_slot)
        self.assertEqual(
            tuple(signature.parameters),
            (
                "reservation",
                "activation_publication",
                "source_publication",
                "candidate_publication",
                "predictions_publication",
                "receipt_publication",
                "project_directory",
            ),
        )
        forbidden = {
            "now",
            "clock",
            "model",
            "artifact",
            "network",
            "session",
            "database_path",
            "protocol",
            "overwrite",
            "repair",
            "completed_at_utc",
        }
        self.assertTrue(forbidden.isdisjoint(signature.parameters))
        source = inspect.getsource(shadow._complete_shadow_prediction_slot)
        self.assertNotIn("joblib", source)
        self.assertNotIn("predict_proba(", source)
        self.assertNotIn("requests.", source)
        self.assertNotIn("sqlite3.", source)


if __name__ == "__main__":
    unittest.main()
