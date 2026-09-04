"""Appel unique et probabilites : modeles/donnees synthetiques exclusivement."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import inspect
import io
import json
from pathlib import Path
import socket
import sys
import threading
import unittest
from unittest import mock
import warnings

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression

from src import shadow_prediction as shadow

if __package__:
    from . import test_shadow_prediction_model_loading as loading_tests
else:
    import test_shadow_prediction_model_loading as loading_tests


APPROVED = "Setting the shape on a NumPy array has been deprecated."


def _emit(message=APPROVED, category=DeprecationWarning, module="joblib.numpy_pickle"):
    warnings.warn_explicit(message, category, filename="synthetic.py", lineno=1, module=module)


def _rehash(row):
    row[20] = shadow.build_feature_row_sha256(
        feature_row_values_in_features_columns_exact_order_excluding_feature_row_sha256=row[:20])
    return row


def _row(game_id=100, start="2026-09-04T23:10:00Z"):
    target = "2026-09-04"
    values = [
        shadow.build_prediction_id(shadow_protocol_sha256=shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
            game_id=game_id, official_date_at_snapshot=target, scheduled_start_utc_at_snapshot=start),
        "b" * 64, game_id,
        shadow.build_occurrence_key(game_id=game_id, official_date_at_snapshot=target,
                                   scheduled_start_utc_at_snapshot_or_null=start),
        2026, target, 11, 22, start, "2026-09-03", "2026-09-02", "2026-09-03",
        12, "0.583333", "4.666667", "3.250000", 11, "0.545455", "3.909091", "4.181818",
        "",
    ]
    return _rehash(values)


class ShadowModelPredictionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = loading_tests.ShadowModelLoadingTests(methodName="runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.project = self.fixture.project
        self.loaded = self.fixture._load()
        self.classifier = self.loaded.artifact.calibrated_classifier
        self.rows = [_row(100), _row(101)]

    def run_prediction(self, rows=None, loaded=None, project=None):
        return shadow._predict_frozen_shadow_model_once(
            self.loaded if loaded is None else loaded,
            self.rows if rows is None else rows,
            project_directory=self.project if project is None else project)

    def predict_stub(self, **kwargs):
        return mock.patch.object(CalibratedClassifierCV, "predict_proba", **kwargs)

    def normal_output(self):
        return np.array([[0.2, 0.8], [0.75, 0.25]], dtype=np.float64)

    def test_one_vectorized_call_between_exact_two_full_artifact_dumps(self):
        events = []
        actual_dump = joblib.dump
        def dump(artifact, buffer, **kwargs):
            self.assertIs(artifact, self.loaded.artifact)
            self.assertIs(type(buffer), io.BytesIO)
            self.assertEqual(kwargs, {"compress": 3})
            self.assertTrue(shadow._MODEL_LOADING_LOCK.locked())
            events.append("dump")
            return actual_dump(artifact, buffer, **kwargs)
        def predict(matrix):
            events.append("predict")
            self.assertEqual(matrix.shape, (2, 8))
            self.assertEqual(matrix.dtype, np.float64)
            self.assertFalse(matrix.flags.writeable)
            self.assertTrue(shadow._MODEL_LOADING_LOCK.locked())
            expected = np.array([[float(value) for value in row[12:20]] for row in self.rows])
            np.testing.assert_array_equal(matrix, expected)
            self.assertEqual(matrix[0, 1], float("0.583333"))
            self.assertNotEqual(matrix[0, 1], 7 / 12)
            return self.normal_output()
        with (mock.patch.object(joblib, "dump", side_effect=dump) as dumps,
              self.predict_stub(side_effect=predict) as calls):
            result = self.run_prediction()
        self.assertEqual(events, ["dump", "predict", "dump"])
        self.assertEqual(dumps.call_count, 2)
        calls.assert_called_once()
        self.assertEqual(result.feature_rows, tuple(map(tuple, self.rows)))
        self.assertEqual(result.probabilities, ((1.0 - 0.8, 0.8), (0.75, 0.25)))
        self.assertEqual(result.artifact_state_sha256_before, result.artifact_state_sha256_after)
        self.assertEqual(result.runtime_versions, self.loaded.runtime_versions)
        self.assertEqual(result.artifact_sha256, self.loaded.artifact_sha256)
        self.assertEqual(result.predict_proba_calls, 1)
        self.assertTrue(result.artifact_state_unchanged)
        self.assertEqual(result.unexpected_warning_count, 0)
        for flag in ("execution_ready", "activation_verified", "execution_manifest_verified",
                     "official_prediction_created"):
            self.assertIs(getattr(result, flag), False)
        with self.assertRaises(FrozenInstanceError):
            result.predict_proba_calls = 2

    def test_real_synthetic_fitted_classifier_predicts_without_fitting_in_operation(self):
        # Fit de fixture uniquement, sans fichier ni statistique MLB.
        rng = np.random.default_rng(1234)
        features = rng.normal(size=(60, 8))
        labels = np.arange(60) % 2
        fitted = CalibratedClassifierCV(LogisticRegression(), method="sigmoid", cv=3)
        fitted.fit(features, labels)
        artifact = replace(self.fixture.artifact, calibrated_classifier=fitted)
        self.fixture._pin(loading_tests._dump_synthetic(artifact))
        loaded = self.fixture._load()
        rows = [_row(100), _row(101)]
        actual_predict = CalibratedClassifierCV.predict_proba
        calls = []
        def predict(classifier, matrix):
            calls.append(matrix.shape)
            return actual_predict(classifier, matrix)
        with (mock.patch.object(CalibratedClassifierCV, "predict_proba", predict),
              mock.patch.object(CalibratedClassifierCV, "fit", side_effect=AssertionError("fit interdit")),
              mock.patch.object(LogisticRegression, "fit", side_effect=AssertionError("fit interdit"))):
            result = self.run_prediction(rows, loaded)
        self.assertEqual(calls, [(2, 8)])
        self.assertEqual(len(result.probabilities), 2)
        self.assertEqual(result.artifact_state_sha256_before, result.artifact_state_sha256_after)
        self.assertTrue(all(0 < home < 1 and away == 1 - home for away, home in result.probabilities))

    def test_single_row_batch_keeps_two_dimensional_matrix(self):
        with self.predict_stub(return_value=np.array([[0.4, 0.6]])) as predict:
            result = self.run_prediction([self.rows[0]])
        self.assertEqual(predict.call_args.args[0].shape, (1, 8))
        self.assertEqual(result.probabilities, ((0.4, 0.6),))

    def test_feature_input_is_copied_not_modified_or_aliased(self):
        original = deepcopy(self.rows)
        def predict(matrix):
            self.rows[0][13] = "0.000000"
            return self.normal_output()
        with self.predict_stub(side_effect=predict):
            result = self.run_prediction()
        self.assertEqual(result.feature_rows, tuple(map(tuple, original)))
        self.assertNotEqual(result.feature_rows[0][13], self.rows[0][13])

    def test_numeric_game_id_order_and_next_utc_day_are_preserved(self):
        rows = [_row(2), _row(10), _row(1, "2026-09-05T01:10:00Z")]
        with self.predict_stub(return_value=np.array([[0.4, 0.6]] * 3)):
            result = self.run_prediction(rows)
        self.assertEqual([row[2] for row in result.feature_rows], [2, 10, 1])

    def test_empty_or_untyped_outer_inputs_fail_before_import_or_model(self):
        for rows in ([], (), None, {}, "rows", iter(self.rows), np.array(self.rows, dtype=object)):
            with self.subTest(kind=type(rows).__name__):
                with (mock.patch.object(shadow, "import_module") as imports,
                      mock.patch.object(joblib, "dump") as dump, self.predict_stub() as predict):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        shadow._predict_frozen_shadow_model_once(
                            self.loaded, rows, project_directory=self.project)
                    imports.assert_not_called()
                    dump.assert_not_called()
                    predict.assert_not_called()

    def test_no_extra_missing_reordered_or_mapping_columns(self):
        variants = [[{}], [dict(zip(shadow._FEATURES_COLUMNS, self.rows[0]))],
                    [self.rows[0][:-1]], [self.rows[0] + ["odds"]],
                    [list(reversed(self.rows[0]))], ["bad"]]
        for rows in variants:
            with self.subTest(rows=repr(rows)[:80]), self.predict_stub() as predict:
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.run_prediction(rows)
                predict.assert_not_called()

    def test_tuple_inputs_are_supported_and_success_never_aliases_output_array(self):
        rows = tuple(map(tuple, self.rows))
        original = deepcopy(self.rows)
        output = self.normal_output()
        with self.predict_stub(return_value=output):
            result = self.run_prediction(rows)
        saved = result.probabilities
        output[:] = 0
        self.assertEqual(result.probabilities, saved)
        self.assertEqual(self.rows, original)
        self.assertEqual(result.feature_rows, rows)

    def test_mixed_batch_or_target_dates_fail_even_with_individually_valid_rows(self):
        for kind in ("batch", "date"):
            rows = deepcopy(self.rows)
            row = rows[1]
            if kind == "batch":
                row[1] = "a" * 64
            else:
                row[5], row[8], row[9] = "2026-09-05", "2026-09-05T23:10:00Z", "2026-09-04"
                row[0] = shadow.build_prediction_id(shadow_protocol_sha256=shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
                    game_id=row[2], official_date_at_snapshot=row[5], scheduled_start_utc_at_snapshot=row[8])
                row[3] = shadow.build_occurrence_key(game_id=row[2], official_date_at_snapshot=row[5],
                    scheduled_start_utc_at_snapshot_or_null=row[8])
            _rehash(row)
            self.assertEqual(shadow._validate_shadow_prediction_feature_rows([row]), (tuple(row),))
            with self.subTest(kind=kind), self.predict_stub() as predict:
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.run_prediction(rows)
                predict.assert_not_called()

    def test_feature_hash_and_identifier_tampering_are_rejected(self):
        for index in (0, 1, 3, 20):
            rows = deepcopy(self.rows)
            rows[0][index] = "a" * 64
            if index in (0, 1, 3):
                _rehash(rows[0])
            with self.subTest(index=index), self.predict_stub() as predict:
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.run_prediction(rows)
                predict.assert_not_called()
        rows = deepcopy(self.rows)
        rows[0][14] = "9.000000"  # ancien hash volontairement conserve
        with self.predict_stub() as predict:
            with self.assertRaises(shadow.ShadowPredictionError):
                self.run_prediction(rows)
            predict.assert_not_called()

    def test_duplicate_and_unsorted_rows_are_not_silently_reordered(self):
        for rows in ([self.rows[0], self.rows[0]], list(reversed(self.rows))):
            with self.subTest(rows=rows), self.predict_stub() as predict:
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.run_prediction(rows)
                predict.assert_not_called()

    def test_identity_history_and_j_minus_one_semantics_are_required(self):
        cases = ((2, 0), (4, 2025), (5, "2025-09-04"), (5, "2026-08-30"),
                 (6, 0), (7, 11), (9, "2026-09-04"), (9, "2026-09-02"),
                 (10, "2026-09-04"), (11, "2025-12-31"), (12, 9), (16, -1),
                 (13, "1.000001"), (17, "2.000000"), (14, "9" * 400 + ".000000"))
        for index, value in cases:
            rows = deepcopy(self.rows)
            rows[0][index] = value
            _rehash(rows[0])
            with self.subTest(index=index, value=value), self.predict_stub() as predict:
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.run_prediction(rows)
                predict.assert_not_called()

    def test_no_bool_or_noncanonical_rate_coercion(self):
        for index, value in ((12, True), (16, 10.0), (13, 0.5), (13, "0.5"),
                             (14, "nan"), (15, "inf"), (18, "-0.000000"),
                             (19, " 1.000000"), (17, "1e-1"), (2, "100")):
            rows = deepcopy(self.rows)
            rows[0][index] = value
            with self.subTest(index=index, value=value), self.predict_stub() as predict:
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.run_prediction(rows)
                predict.assert_not_called()

    def test_nonrepresentable_integer_or_overflow_is_rejected_before_dump(self):
        for value in (2 ** 53 + 1, 10 ** 400):
            rows = deepcopy(self.rows)
            rows[0][12] = value
            _rehash(rows[0])
            with self.subTest(value=value):
                with (mock.patch.object(joblib, "dump") as dump, self.predict_stub() as predict):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.run_prediction(rows)
                    dump.assert_not_called()
                    predict.assert_not_called()

    def test_invalid_loaded_envelope_fails_before_read_import_or_prediction(self):
        for loaded in (object(), replace(self.loaded, artifact_sha256="a" * 64),
                       replace(self.loaded, approved_deserialization_warning_count=True)):
            with (self.subTest(loaded=type(loaded).__name__), mock.patch.object(Path, "read_bytes") as read,
                  mock.patch.object(shadow, "import_module") as imports, self.predict_stub() as predict):
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.run_prediction(loaded=loaded)
                read.assert_not_called()
                imports.assert_not_called()
                predict.assert_not_called()

    def test_wrong_classes_metadata_or_feature_count_fail_before_predict(self):
        for field, value in (("classes_", np.array([1, 0])), ("classes_", np.array([0., 1.])),
                             ("n_features_in_", 7), ("method", "isotonic")):
            with (self.subTest(field=field, value=value), mock.patch.object(self.classifier, field, value),
                  self.predict_stub() as predict):
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.run_prediction()
                predict.assert_not_called()

    def test_wrong_runtime_and_each_pinned_document_block_prediction(self):
        with (mock.patch.object(shadow, "_installed_model_runtime_versions", return_value={}),
              self.predict_stub() as predict):
            with self.assertRaises(shadow.ShadowPredictionError):
                self.run_prediction()
            predict.assert_not_called()
        for relative in loading_tests.METADATA_PATHS:
            path = self.project / relative
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            try:
                with self.subTest(relative=relative), self.predict_stub() as predict:
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.run_prediction()
                    predict.assert_not_called()
            finally:
                path.write_bytes(original)

    def test_invalid_project_does_not_predict(self):
        for project in (str(self.project), Path("relative"), self.project / "absent"):
            with self.subTest(project=project), self.predict_stub() as predict:
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.run_prediction(project=project)
                predict.assert_not_called()

    def test_invalid_shapes_types_and_dtypes_fail_without_second_prediction(self):
        outputs = (None, [[0.2, 0.8], [0.75, 0.25]], np.array([0.8, 0.25]),
                   np.ones((1, 2)), np.ones((2, 1)), np.ones((2, 3)),
                   np.ones((2, 2, 1)), np.array([[True, False]] * 2),
                   np.array([[0, 1]] * 2), np.array([[0.2, 0.8]] * 2, dtype=object),
                   np.array([["0.2", "0.8"]] * 2), np.array([[0.2, 0.8]] * 2, dtype=complex),
                   np.ma.array([[0.2, 0.8]] * 2, mask=False))
        for output in outputs:
            with self.subTest(output=repr(output)):
                with (self.predict_stub(return_value=output) as predict,
                      mock.patch.object(joblib, "dump", wraps=joblib.dump) as dump):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.run_prediction()
                    predict.assert_called_once()
                    self.assertEqual(dump.call_count, 2)

    def test_nan_infinite_out_of_bounds_negative_zero_and_bad_sums_are_rejected(self):
        pairs = ((np.nan, 0.5), (0.5, np.nan), (np.inf, 0.5), (0.5, -np.inf),
                 (-0.1, 1.1), (1.1, -0.1), (0.2, 0.2), (0.5 + 2e-12, 0.5),
                 (-0.0, 1.0), (1.0, -0.0))
        for pair in pairs:
            with self.subTest(pair=pair), self.predict_stub(return_value=np.array([pair, [0.2, 0.8]])) as predict:
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.run_prediction()
                predict.assert_called_once()

    def test_endpoints_are_preserved_and_away_is_derived_from_home(self):
        for output in (np.array([[1., 0.], [0., 1.]]),
                       np.array([[0.5 + 5e-13, 0.5], [0.75, 0.25]])):
            with self.subTest(output=output), self.predict_stub(return_value=output):
                result = self.run_prediction()
            self.assertEqual(result.probabilities, tuple((1 - float(r[1]), float(r[1])) for r in output))

    def test_float32_output_values_are_not_normalized_or_rounded(self):
        output = np.array([[0.5, 0.5], [0.75, 0.25]], dtype=np.float32)
        with self.predict_stub(return_value=output):
            result = self.run_prediction()
        self.assertEqual(result.probabilities, ((0.5, 0.5), (0.75, 0.25)))

    def test_every_warning_during_predict_including_approved_is_fatal(self):
        for category in (DeprecationWarning, UserWarning, RuntimeWarning, FutureWarning):
            filters = list(warnings.filters)
            with self.subTest(category=category):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with (self.predict_stub(side_effect=lambda matrix: _emit(category=category)) as predict,
                          mock.patch.object(joblib, "dump", wraps=joblib.dump) as dump):
                        with self.assertRaises(shadow.ShadowPredictionError) as caught:
                            self.run_prediction()
                        self.assertIsInstance(caught.exception.__cause__, category)
                        predict.assert_called_once()
                        self.assertEqual(dump.call_count, 1)
                self.assertEqual(warnings.filters, filters)
                self.assertFalse(shadow._MODEL_LOADING_LOCK.locked())

    def test_inconsistent_sklearn_warning_is_fatal_during_predict(self):
        def predict(matrix):
            warnings.warn(InconsistentVersionWarning(estimator_name="Synthetic",
                          current_sklearn_version="1.9.0", original_sklearn_version="0.0.0"))
        with self.predict_stub(side_effect=predict):
            with self.assertRaises(shadow.ShadowPredictionError):
                self.run_prediction()

    def test_approved_state_warnings_are_counted_separately_then_once_in_total(self):
        loaded = replace(self.loaded, approved_deserialization_warning_count=7)
        actual = joblib.dump
        def dump(*args, **kwargs):
            _emit()
            _emit()
            return actual(*args, **kwargs)
        with (mock.patch.object(joblib, "dump", side_effect=dump),
              self.predict_stub(return_value=self.normal_output())):
            result = self.run_prediction(loaded=loaded)
        self.assertEqual(result.approved_deserialization_warning_count, 7)
        self.assertEqual(result.approved_state_serialization_warning_count, 4)
        self.assertEqual(result.approved_compatibility_warning_count, 11)
        self.assertEqual(result.warning_policy_id, shadow._MODEL_WARNING_POLICY_ID)

    def test_approved_warning_is_fatal_during_matrix_or_output_validation(self):
        for phase in ("matrix", "output"):
            with self.subTest(phase=phase), ExitStack() as stack:
                predict = stack.enter_context(self.predict_stub(return_value=self.normal_output()))
                if phase == "matrix":
                    stack.enter_context(mock.patch.object(np, "asarray", side_effect=lambda *a, **k: _emit()))
                else:
                    stack.enter_context(mock.patch.object(shadow, "_format_probability_float",
                                                         side_effect=lambda *a, **k: _emit()))
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.run_prediction()
                self.assertEqual(predict.call_count, 0 if phase == "matrix" else 1)
            self.assertFalse(shadow._MODEL_LOADING_LOCK.locked())

    def test_invalid_last_row_does_not_return_partial_probabilities(self):
        output = np.array([[0.2, 0.8], [np.nan, 0.4]])
        with self.predict_stub(return_value=output) as predict:
            with self.assertRaises(shadow.ShadowPredictionError):
                self.run_prediction()
            predict.assert_called_once()

    def test_unexpected_warning_or_exception_in_each_dump_aborts_without_retry(self):
        actual = joblib.dump
        for failure_at in (1, 2):
            for kind in ("warning", "error"):
                count = 0
                def dump(*args, **kwargs):
                    nonlocal count
                    count += 1
                    if count == failure_at:
                        if kind == "warning":
                            _emit("bad", UserWarning)
                        raise OSError("synthetic")
                    return actual(*args, **kwargs)
                with (self.subTest(failure_at=failure_at, kind=kind),
                      mock.patch.object(joblib, "dump", side_effect=dump),
                      self.predict_stub(return_value=self.normal_output()) as predict):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.run_prediction()
                    self.assertEqual(count, failure_at)
                    self.assertEqual(predict.call_count, failure_at - 1)
                self.assertFalse(shadow._MODEL_LOADING_LOCK.locked())

    def test_errors_interruptions_restore_filters_alias_lock_without_retry(self):
        for error in (ValueError("bad"), OSError("bad"), KeyboardInterrupt(), SystemExit(1)):
            filters = list(warnings.filters)
            sentinel = object()
            with (self.subTest(error=type(error).__name__),
                  mock.patch.object(sys.modules["__main__"], "CalibratedModelArtifact", sentinel, create=True),
                  self.predict_stub(side_effect=error) as predict,
                  mock.patch.object(joblib, "dump", wraps=joblib.dump) as dump):
                expected = shadow.ShadowPredictionError if isinstance(error, Exception) else type(error)
                with self.assertRaises(expected):
                    self.run_prediction()
                predict.assert_called_once()
                self.assertEqual(dump.call_count, 1)
                self.assertIs(sys.modules["__main__"].CalibratedModelArtifact, sentinel)
            self.assertEqual(warnings.filters, filters)
            self.assertFalse(shadow._MODEL_LOADING_LOCK.locked())

    def test_estimator_state_mutation_is_rejected_without_repair(self):
        self.classifier.synthetic_coefficients_ = np.array([1., 2.])
        def predict(matrix):
            self.classifier.synthetic_coefficients_[0] = 9
            return self.normal_output()
        with self.predict_stub(side_effect=predict) as predict_call:
            with self.assertRaisesRegex(shadow.ShadowPredictionError, "a change"):
                self.run_prediction()
            predict_call.assert_called_once()
        self.assertEqual(self.classifier.synthetic_coefficients_[0], 9)

    def test_model_envelope_replacement_or_changed_warning_count_is_rejected(self):
        for key in ("artifact", "approved_deserialization_warning_count"):
            original = getattr(self.loaded, key)
            def predict(matrix):
                value = replace(self.loaded.artifact) if key == "artifact" else original + 1
                object.__setattr__(self.loaded, key, value)
                return self.normal_output()
            try:
                with self.subTest(key=key), self.predict_stub(side_effect=predict):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.run_prediction()
            finally:
                object.__setattr__(self.loaded, key, original)

    def test_metadata_classes_documents_and_runtime_changed_by_predict_are_rejected(self):
        for kind in ("classes", "metadata", "document", "runtime"):
            with self.subTest(kind=kind), ExitStack() as stack:
                def predict(matrix):
                    if kind == "classes":
                        stack.enter_context(mock.patch.object(self.classifier, "classes_", np.array([1, 0])))
                    elif kind == "metadata":
                        stack.enter_context(mock.patch.object(self.classifier, "method", "isotonic"))
                    elif kind == "runtime":
                        stack.enter_context(mock.patch.object(np, "__version__", "0.0.0"))
                    else:
                        path = self.project / loading_tests.SHADOW_PATH
                        content = path.read_bytes()
                        stack.callback(path.write_bytes, content)
                        path.write_bytes(content + b" ")
                    return self.normal_output()
                stack.enter_context(self.predict_stub(side_effect=predict))
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.run_prediction()

    def test_input_matrix_write_or_shape_change_is_rejected(self):
        for kind in ("write", "flag", "shape"):
            def predict(matrix):
                if kind == "shape":
                    matrix.shape = (4, 4)
                else:
                    matrix.setflags(write=True)
                    if kind == "write":
                        matrix[0, 0] = 999
                        matrix.setflags(write=False)
                return self.normal_output()
            with self.subTest(kind=kind), self.predict_stub(side_effect=predict):
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.run_prediction()

    def test_readonly_matrix_blocks_normal_inplace_write(self):
        def predict(matrix):
            matrix[0, 0] = 999
            return self.normal_output()
        with self.predict_stub(side_effect=predict):
            with self.assertRaises(shadow.ShadowPredictionError):
                self.run_prediction()

    def test_same_thread_reentry_into_loader_snapshot_or_predict_is_rejected(self):
        def predict(matrix):
            for callback in (self.fixture._load, self.run_prediction,
                             lambda: shadow._snapshot_frozen_shadow_model_state(self.loaded, project_directory=self.project)):
                with self.assertRaisesRegex(shadow.ShadowPredictionError, "deja en cours"):
                    callback()
            return self.normal_output()
        with self.predict_stub(side_effect=predict) as calls:
            self.run_prediction()
            calls.assert_called_once()

    def test_concurrent_operations_cannot_enter_during_any_transaction_phase(self):
        actual_dump = joblib.dump
        for stage in ("before", "predict", "after"):
            entered, release = threading.Event(), threading.Event()
            count = 0
            def block():
                entered.set()
                if not release.wait(5):
                    raise AssertionError("Concurrence expiree")
            def dump(*args, **kwargs):
                nonlocal count
                count += 1
                if (stage, count) in (("before", 1), ("after", 2)):
                    block()
                return actual_dump(*args, **kwargs)
            def predict(matrix):
                if stage == "predict":
                    block()
                return self.normal_output()
            with (self.subTest(stage=stage), mock.patch.object(joblib, "dump", side_effect=dump),
                  self.predict_stub(side_effect=predict) as calls, ThreadPoolExecutor(max_workers=1) as pool):
                future = pool.submit(self.run_prediction)
                try:
                    self.assertTrue(entered.wait(5))
                    for callback in (self.fixture._load, self.run_prediction,
                                     lambda: shadow._snapshot_frozen_shadow_model_state(self.loaded, project_directory=self.project)):
                        with self.assertRaisesRegex(shadow.ShadowPredictionError, "deja en cours"):
                            callback()
                finally:
                    release.set()
                future.result(timeout=5)
                calls.assert_called_once()
            self.assertFalse(shadow._MODEL_LOADING_LOCK.locked())

    def test_active_loader_blocks_prediction_before_any_new_work(self):
        entered, release = threading.Event(), threading.Event()
        actual = joblib.load
        def load(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError("Concurrence expiree")
            return actual(*args, **kwargs)
        with (mock.patch.object(joblib, "load", side_effect=load), self.predict_stub() as predict,
              ThreadPoolExecutor(max_workers=1) as pool):
            future = pool.submit(self.fixture._load)
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(shadow.ShadowPredictionError, "deja en cours"):
                    self.run_prediction()
                predict.assert_not_called()
            finally:
                release.set()
            future.result(timeout=5)

    def test_internal_snapshot_body_requires_existing_lock(self):
        with mock.patch.object(joblib, "dump") as dump:
            with self.assertRaises(shadow.ShadowPredictionError):
                shadow._snapshot_frozen_shadow_model_state_under_lock(self.loaded, project_directory=self.project)
            dump.assert_not_called()

    def test_operation_reads_only_metadata_no_model_network_database_training_or_output(self):
        allowed = {self.project / path for path in loading_tests.METADATA_PATHS}
        original_open = io.open
        before = {path: path.read_bytes() for path in allowed}
        def read(path, mode="r", *args, **kwargs):
            self.assertIn(Path(path), allowed)
            self.assertFalse(any(c in mode for c in "wax+"))
            return original_open(path, mode, *args, **kwargs)
        forbidden = AssertionError("Operation officielle interdite")
        with ExitStack() as stack:
            stack.enter_context(mock.patch("io.open", side_effect=read))
            for obj, name in ((joblib, "load"), (shadow.requests, "get"), (shadow.sqlite3, "connect"),
                              (socket, "create_connection"), (shadow, "_publish_exclusive_verified"),
                              (shadow, "reserve_shadow_prediction_slot"), (shadow, "fail_shadow_prediction_slot"),
                              (Path, "write_bytes"), (Path, "mkdir"), (Path, "unlink"),
                              (CalibratedClassifierCV, "fit"), (CalibratedClassifierCV, "predict"),
                              (LogisticRegression, "fit"), (LogisticRegression, "predict_proba")):
                stack.enter_context(mock.patch.object(obj, name, side_effect=forbidden))
            stack.enter_context(self.predict_stub(return_value=self.normal_output()))
            self.run_prediction()
        self.assertEqual({path: path.read_bytes() for path in allowed}, before)
        self.assertFalse(list(self.project.rglob("*.joblib")))


class ShadowModelPredictionPolicyTests(unittest.TestCase):
    def test_contract_is_frozen_and_api_has_no_prediction_callback_or_policy_override(self):
        protocol = json.loads((loading_tests.PROJECT / loading_tests.SHADOW_PATH).read_bytes())
        self.assertEqual(protocol["model_use"]["maximum_predict_proba_calls_per_nonempty_batch"], 1)
        self.assertEqual(protocol["model_use"]["expected_classes"], [0, 1])
        self.assertEqual(protocol["model_use"]["away_probability_formula"], "1 - p_home")
        self.assertEqual(protocol["model_use"]["warning_policy"]["warnings_during_predict_proba"], "ABORT_ENTIRE_RUN")
        self.assertEqual(protocol["feature_contract"]["feature_columns_in_exact_order"],
                         list(shadow._FEATURE_ROW_FIELD_NAMES[12:20]))
        self.assertEqual(protocol["feature_contract"]["feature_quantization"]["matrix_dtype"], "numpy.float64")
        self.assertEqual(protocol["receipt_contract"]["probability_pairs_must_sum_to_one_with_absolute_tolerance"], 1e-12)
        self.assertEqual(list(inspect.signature(shadow._predict_frozen_shadow_model_once).parameters),
                         ["loaded_model", "feature_rows", "project_directory"])

    def test_preview_never_calls_new_prediction_or_state_helpers(self):
        forbidden = AssertionError("Apercu doit rester inerte")
        with (mock.patch.object(shadow, "_predict_frozen_shadow_model_once", side_effect=forbidden),
              mock.patch.object(shadow, "_snapshot_frozen_shadow_model_state_under_lock", side_effect=forbidden),
              mock.patch.object(shadow, "import_module", side_effect=forbidden)):
            result = shadow.preview_shadow_prediction("2026-09-04", project_directory=loading_tests.PROJECT)
        self.assertFalse(result.execution_ready)
        self.assertFalse(result.model_deserialized)
        self.assertFalse(result.predictions_computed)


if __name__ == "__main__":
    unittest.main()
