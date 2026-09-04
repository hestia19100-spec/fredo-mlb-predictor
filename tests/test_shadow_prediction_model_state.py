"""Empreintes d'etat shadow : objets synthetiques, jamais le modele reel."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
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

# unittest discover charge des modules sans package, contrairement a -m unittest.
if __package__:
    from . import test_shadow_prediction_model_loading as loading_tests
else:
    import test_shadow_prediction_model_loading as loading_tests


APPROVED = "Setting the shape on a NumPy array has been deprecated."


def _emit(message=APPROVED, category=DeprecationWarning, module="joblib.numpy_pickle"):
    warnings.warn_explicit(message, category, filename="synthetic.py", lineno=1,
                           module=module)


class ShadowModelStateTests(unittest.TestCase):
    def setUp(self):
        # Reutilise uniquement la fixture synthetique, pas ses tests.
        self.fixture = loading_tests.ShadowModelLoadingTests(methodName="runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.project = self.fixture.project
        self.loaded = self.fixture._load()
        self.classifier = self.loaded.artifact.calibrated_classifier
        self.classifier.synthetic_coefficients_ = np.array([[0.1, -0.3]])

    def snapshot(self, loaded=None, project=None):
        return shadow._snapshot_frozen_shadow_model_state(
            self.loaded if loaded is None else loaded,
            project_directory=self.project if project is None else project,
        )

    def verify(self, before, loaded=None):
        return shadow._verify_frozen_shadow_model_state_unchanged(
            self.loaded if loaded is None else loaded, before,
            project_directory=self.project,
        )

    def test_exact_single_dump_of_entire_artifact_to_memory_compress_three(self):
        real_dump = joblib.dump
        captures = []
        def capture(artifact, buffer, **kwargs):
            self.assertIs(artifact, self.loaded.artifact)
            self.assertIs(type(buffer), io.BytesIO)
            self.assertEqual(kwargs, {"compress": 3})
            self.assertNotIn("CalibratedModelArtifact", vars(sys.modules["__main__"]))
            output = real_dump(artifact, buffer, **kwargs)
            captures.append((buffer, buffer.getvalue()))
            return output
        with mock.patch.object(joblib, "dump", side_effect=capture) as dump:
            result = self.snapshot()
        dump.assert_called_once()
        self.assertTrue(captures[0][0].closed)
        self.assertEqual(result.artifact_state_sha256,
                         hashlib.sha256(captures[0][1]).hexdigest())
        self.assertIs(result.loaded_model, self.loaded)
        self.assertEqual(result.runtime_versions, self.loaded.runtime_versions)
        # Etat en memoire enrichi synthetiquement != empreinte des octets charges.
        self.assertNotEqual(result.artifact_state_sha256, self.loaded.artifact_sha256)
        with self.assertRaises(FrozenInstanceError):
            result.artifact_state_sha256 = "0" * 64

    def test_success_recomputes_after_hash_and_does_not_authorize_prediction(self):
        before = self.snapshot()
        with mock.patch.object(joblib, "dump", wraps=joblib.dump) as dump:
            verified = self.verify(before)
        dump.assert_called_once()
        self.assertEqual(verified.artifact_state_sha256_before, before.artifact_state_sha256)
        self.assertEqual(verified.artifact_state_sha256_after, before.artifact_state_sha256)
        self.assertIs(verified.artifact_state_unchanged, True)
        self.assertIs(type(verified.approved_state_serialization_warning_count), int)
        for name in ("execution_ready", "execution_manifest_verified", "activation_verified"):
            self.assertIs(getattr(verified, name), False)
        # Une comparaison ne pretend pas avoir observe un appel de prediction.
        self.assertFalse(hasattr(verified, "predict_proba_calls"))
        with self.assertRaises(FrozenInstanceError):
            verified.artifact_state_unchanged = False

    def test_array_mutation_between_measurements_is_rejected_without_repair(self):
        before = self.snapshot()
        self.classifier.synthetic_coefficients_[0, 1] = 8.5
        with mock.patch.object(joblib, "dump", wraps=joblib.dump) as dump:
            with self.assertRaisesRegex(shadow.ShadowPredictionError, "etat serialise.*change"):
                self.verify(before)
        dump.assert_called_once()
        self.assertEqual(self.classifier.synthetic_coefficients_[0, 1], 8.5)

    def test_estimator_parameter_mutation_is_detected(self):
        before = self.snapshot()
        self.classifier.cv = 7
        with self.assertRaisesRegex(shadow.ShadowPredictionError, "etat serialise.*change"):
            self.verify(before)
        self.assertEqual(self.classifier.cv, 7)

    def test_different_loaded_instance_is_rejected_even_with_identical_digest(self):
        before = self.snapshot()
        for other in (replace(self.loaded), self.fixture._load()):
            with self.subTest(other=other is self.loaded):
                with mock.patch.object(joblib, "dump") as dump:
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.verify(before, other)
                    dump.assert_not_called()

    def test_invalid_before_measurement_is_rejected_before_files_or_dump(self):
        before = self.snapshot()
        cases = [None, {}, self.loaded]
        for name, value in (
            ("artifact_state_sha256", "x" * 64),
            ("artifact_state_sha256", "A" * 64),
            ("artifact_state_sha256", b"0" * 64),
            ("artifact_state_sha256", None),
            ("loaded_model", replace(self.loaded)),
            ("runtime_versions", list(self.loaded.runtime_versions)),
            ("runtime_versions", ()),
            ("warning_policy_id", "OTHER"),
            ("approved_state_serialization_warning_count", True),
            ("approved_state_serialization_warning_count", -1),
            ("approved_state_serialization_warning_count", 1.0),
        ):
            wrong = copy(before)
            object.__setattr__(wrong, name, value)
            cases.append(wrong)
        with (mock.patch.object(Path, "read_bytes") as read,
              mock.patch.object(joblib, "dump") as dump):
            for case in cases:
                with self.subTest(case=repr(case)):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.verify(case)
            read.assert_not_called()
            dump.assert_not_called()

    def test_invalid_loaded_envelopes_fail_before_files_imports_or_dump(self):
        cases = [object(), {}, self.loaded.artifact, self.fixture.proof]
        for name, value in (
            ("artifact_sha256", "0" * 64), ("artifact_sha256", True),
            ("warning_policy_id", "OTHER"), ("model_deserialized", False),
            ("predictions_computed", True), ("execution_ready", True),
            ("activation_verified", True), ("execution_manifest_verified", True),
            ("approved_deserialization_warning_count", -1),
            ("approved_deserialization_warning_count", True),
            ("runtime_versions", list(self.loaded.runtime_versions)),
            ("runtime_versions", tuple(reversed(self.loaded.runtime_versions))),
            ("runtime_versions", self.loaded.runtime_versions * 2),
            ("runtime_versions", (("python", "3.12.1"),)),
            ("runtime_versions", (("python", True),)),
            ("runtime_versions", ("python",)),
            ("runtime_versions", tuple((k, " " + v) for k, v in self.loaded.runtime_versions)),
        ):
            wrong = copy(self.loaded)
            object.__setattr__(wrong, name, value)
            cases.append(wrong)
        with (mock.patch.object(Path, "read_bytes") as read,
              mock.patch.object(shadow, "import_module") as imports,
              mock.patch.object(joblib, "dump") as dump):
            for case in cases:
                with self.subTest(case=type(case).__name__):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.snapshot(case)
            read.assert_not_called()
            imports.assert_not_called()
            dump.assert_not_called()

    def test_exact_metadata_and_classifier_checks_run_before_dump(self):
        artifacts = [object(), replace(self.loaded.artifact, dataset_sha256="0" * 64),
                     replace(self.loaded.artifact, calibrated_classifier=object())]
        for artifact in artifacts:
            with self.subTest(artifact=type(artifact).__name__):
                with mock.patch.object(joblib, "dump") as dump:
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.snapshot(replace(self.loaded, artifact=artifact))
                    dump.assert_not_called()

    def test_classifier_classes_shape_and_features_are_revalidated(self):
        for name, value in (("method", "isotonic"), ("n_features_in_", True),
                            ("n_features_in_", 7), ("classes_", np.array([1, 0]))):
            with self.subTest(name=name):
                with (mock.patch.object(self.classifier, name, value),
                      mock.patch.object(joblib, "dump") as dump):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.snapshot()
                    dump.assert_not_called()

    def test_each_pinned_document_is_required_before_runtime_import(self):
        for relative in loading_tests.METADATA_PATHS:
            path = self.project / relative
            original = path.read_bytes()
            with self.subTest(relative=relative):
                path.write_bytes(original + b" ")
                try:
                    with (mock.patch.object(shadow, "import_module") as imports,
                          mock.patch.object(joblib, "dump") as dump):
                        with self.assertRaises(shadow.ShadowPredictionError):
                            self.snapshot()
                        imports.assert_not_called()
                        dump.assert_not_called()
                finally:
                    path.write_bytes(original)

    def test_wrong_project_is_rejected_without_dump(self):
        for project in (str(self.project), Path("relative"), self.project / "missing"):
            with self.subTest(project=project), mock.patch.object(joblib, "dump") as dump:
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.snapshot(project=project)
                dump.assert_not_called()

    def test_each_installed_runtime_must_match_before_imports(self):
        for key in loading_tests.RUNTIME:
            wrong = dict(loading_tests.RUNTIME, **{key: "0.0.0"})
            with self.subTest(key=key):
                with (mock.patch.object(shadow, "_installed_model_runtime_versions", return_value=wrong),
                      mock.patch.object(shadow, "import_module") as imports,
                      mock.patch.object(joblib, "dump") as dump):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.snapshot()
                    imports.assert_not_called()
                    dump.assert_not_called()

    def test_canonical_but_wrong_loaded_runtime_is_rejected(self):
        wrong = dict(self.loaded.runtime_versions, python="0.0.0")
        with mock.patch.object(joblib, "dump") as dump:
            with self.assertRaises(shadow.ShadowPredictionError):
                self.snapshot(replace(self.loaded, runtime_versions=tuple(sorted(wrong.items()))))
            dump.assert_not_called()

    def test_each_imported_runtime_must_match_before_dump(self):
        for name, module in loading_tests.RUNTIME_MODULES.items():
            with self.subTest(name=name):
                with (mock.patch.object(module, "__version__", "0.0.0"),
                      mock.patch.object(joblib, "dump") as dump):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.snapshot()
                    dump.assert_not_called()

    def test_runtime_changed_during_serialization_is_rejected(self):
        actual_dump = joblib.dump
        for kind in ("installed", "imported", "loaded"):
            with self.subTest(kind=kind), ExitStack() as stack:
                def mutate(*args, **kwargs):
                    result = actual_dump(*args, **kwargs)
                    if kind == "installed":
                        stack.enter_context(mock.patch.object(shadow, "_installed_model_runtime_versions",
                                                              return_value={}))
                    elif kind == "imported":
                        stack.enter_context(mock.patch.object(np, "__version__", "0.0.0"))
                    else:
                        previous = self.loaded.runtime_versions
                        stack.callback(object.__setattr__, self.loaded, "runtime_versions", previous)
                        object.__setattr__(self.loaded, "runtime_versions", tuple(sorted(
                            dict(previous, python="0.0.0").items())))
                    return result
                stack.enter_context(mock.patch.object(joblib, "dump", side_effect=mutate))
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.snapshot()

    def test_documents_changed_during_serialization_are_rejected(self):
        actual_dump = joblib.dump
        for relative in loading_tests.METADATA_PATHS:
            path = self.project / relative
            original = path.read_bytes()
            with self.subTest(relative=relative):
                def mutate(*args, **kwargs):
                    output = actual_dump(*args, **kwargs)
                    path.write_bytes(original + b" ")
                    return output
                try:
                    with mock.patch.object(joblib, "dump", side_effect=mutate):
                        with self.assertRaises(shadow.ShadowPredictionError):
                            self.snapshot()
                finally:
                    path.write_bytes(original)

    def test_metadata_changed_during_serialization_is_rejected(self):
        actual_dump = joblib.dump
        def mutate(*args, **kwargs):
            output = actual_dump(*args, **kwargs)
            self.classifier.method = "isotonic"
            return output
        with mock.patch.object(joblib, "dump", side_effect=mutate):
            with self.assertRaises(shadow.ShadowPredictionError):
                self.snapshot()
        self.assertEqual(self.classifier.method, "isotonic")

    def test_whole_artifact_replaced_during_serialization_is_rejected(self):
        actual_dump = joblib.dump
        def replace_artifact(*args, **kwargs):
            output = actual_dump(*args, **kwargs)
            object.__setattr__(self.loaded, "artifact", replace(self.loaded.artifact))
            return output
        with mock.patch.object(joblib, "dump", side_effect=replace_artifact):
            with self.assertRaisesRegex(shadow.ShadowPredictionError, "remplace"):
                self.snapshot()

    def test_exact_warning_count_and_stage_specific_total(self):
        # Un compteur de chargement doit rester separe du compteur d'etat.
        loaded = replace(self.loaded, approved_deserialization_warning_count=7)
        real_dump = joblib.dump
        def warned(*args, **kwargs):
            _emit()
            _emit()
            return real_dump(*args, **kwargs)
        with mock.patch.object(joblib, "dump", side_effect=warned):
            before = self.snapshot(loaded)
            verified = self.verify(before, loaded)
        self.assertEqual(before.approved_state_serialization_warning_count, 2)
        self.assertEqual(verified.approved_state_serialization_warning_count, 4)
        self.assertEqual(loaded.approved_deserialization_warning_count, 7)

    def test_registered_warning_regex_semantics_are_preserved(self):
        real_dump = joblib.dump
        cases = ((APPROVED.upper(), "joblib.numpy_pickle"),
                 (APPROVED + " additional text", "joblib.numpy_pickle_extra"))
        for message, module in cases:
            with self.subTest(message=message, module=module):
                def warned(*args, **kwargs):
                    _emit(message=message, module=module)
                    return real_dump(*args, **kwargs)
                with mock.patch.object(joblib, "dump", side_effect=warned):
                    self.assertEqual(self.snapshot().approved_state_serialization_warning_count, 1)

    def test_other_warnings_abort_without_retry_and_restore_filters(self):
        cases = ((APPROVED, UserWarning, "joblib.numpy_pickle"),
                 (APPROVED, FutureWarning, "joblib.numpy_pickle"),
                 (APPROVED, DeprecationWarning, "Joblib.numpy_pickle"),
                 (APPROVED, DeprecationWarning, "elsewhere"),
                 ("prefix " + APPROVED, DeprecationWarning, "joblib.numpy_pickle"),
                 ("Unexpected warning", DeprecationWarning, "joblib.numpy_pickle"))
        for message, category, module in cases:
            filters = list(warnings.filters)
            with self.subTest(message=message, category=category, module=module):
                def warned(*args, **kwargs):
                    _emit(message, category, module)
                with mock.patch.object(joblib, "dump", side_effect=warned) as dump:
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.snapshot()
                    dump.assert_called_once()
                self.assertEqual(warnings.filters, filters)
                self.assertFalse(shadow._MODEL_LOADING_LOCK.locked())

    def test_global_ignore_cannot_hide_unexpected_or_approved_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with mock.patch.object(joblib, "dump", side_effect=lambda *a, **k: _emit("bad", UserWarning)):
                with self.assertRaises(shadow.ShadowPredictionError):
                    self.snapshot()
            actual = joblib.dump
            def warned(*args, **kwargs):
                _emit()
                return actual(*args, **kwargs)
            with mock.patch.object(joblib, "dump", side_effect=warned):
                self.assertEqual(self.snapshot().approved_state_serialization_warning_count, 1)

    def test_inconsistent_sklearn_version_warning_is_not_approved(self):
        def warned(*args, **kwargs):
            warnings.warn(InconsistentVersionWarning(estimator_name="Synthetic",
                          current_sklearn_version="1.9.0", original_sklearn_version="0.0.0"))
        with mock.patch.object(joblib, "dump", side_effect=warned):
            with self.assertRaises(shadow.ShadowPredictionError):
                self.snapshot()

    def test_approved_warning_outside_dump_aborts(self):
        original_read = Path.read_bytes
        for warn_at in (1, 4):
            calls = 0
            def read(path):
                nonlocal calls
                calls += 1
                if calls == warn_at:
                    _emit()
                return original_read(path)
            with self.subTest(warn_at=warn_at):
                with (mock.patch.object(Path, "read_bytes", read),
                      mock.patch.object(joblib, "dump", wraps=joblib.dump) as dump):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.snapshot()
                    self.assertEqual(dump.call_count, 0 if warn_at == 1 else 1)
        for hook in ("_import_shadow_model_runtime", "_validate_loaded_shadow_model_metadata"):
            with self.subTest(hook=hook):
                with mock.patch.object(shadow, hook, side_effect=lambda *a, **k: _emit()):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self.snapshot()

    def test_errors_and_interruptions_close_buffer_and_release_lock_without_retry(self):
        for error in (ValueError("bad"), OSError("bad"), RuntimeError("bad"),
                      KeyboardInterrupt(), SystemExit(1)):
            with self.subTest(error=type(error).__name__):
                buffers = []
                filters = list(warnings.filters)
                sentinel = object()
                def fail(artifact, buffer, **kwargs):
                    buffers.append(buffer)
                    buffer.write(b"partial")
                    raise error
                with (mock.patch.object(sys.modules["__main__"], "CalibratedModelArtifact", sentinel, create=True),
                      mock.patch.object(joblib, "dump", side_effect=fail) as dump):
                    expected = shadow.ShadowPredictionError if isinstance(error, Exception) else type(error)
                    with self.assertRaises(expected):
                        self.snapshot()
                    self.assertIs(sys.modules["__main__"].CalibratedModelArtifact, sentinel)
                    dump.assert_called_once()
                self.assertTrue(buffers[0].closed)
                self.assertEqual(warnings.filters, filters)
                self.assertFalse(shadow._MODEL_LOADING_LOCK.locked())
                self.snapshot()  # Ressources liberees ; pas une reprise officielle.

    def test_empty_serializer_output_is_rejected(self):
        with mock.patch.object(joblib, "dump", return_value=None) as dump:
            with self.assertRaisesRegex(shadow.ShadowPredictionError, "vide"):
                self.snapshot()
            dump.assert_called_once()

    def test_import_failure_never_attempts_dump(self):
        with (mock.patch.object(shadow, "import_module", side_effect=ImportError("synthetic")),
              mock.patch.object(joblib, "dump") as dump):
            with self.assertRaises(shadow.ShadowPredictionError):
                self.snapshot()
            dump.assert_not_called()
        self.assertFalse(shadow._MODEL_LOADING_LOCK.locked())

    def test_second_snapshot_error_never_returns_verified_result(self):
        before = self.snapshot()
        with mock.patch.object(joblib, "dump", side_effect=OSError("bad")) as dump:
            with self.assertRaises(shadow.ShadowPredictionError):
                self.verify(before)
            dump.assert_called_once()

    def test_no_network_model_file_sqlite_dataset_prediction_training_or_output(self):
        original_read = io.open
        allowed = {self.project / path for path in loading_tests.METADATA_PATHS}
        contents = {path: path.read_bytes() for path in allowed}
        seen = []
        def read(path, mode="r", *args, **kwargs):
            self.assertIn(Path(path), allowed)
            self.assertFalse(any(c in mode for c in "wax+"))
            seen.append(Path(path))
            return original_read(path, mode, *args, **kwargs)
        with ExitStack() as stack:
            stack.enter_context(mock.patch("io.open", side_effect=read))
            forbidden = AssertionError("Operation officielle interdite")
            for obj, name in ((joblib, "load"), (shadow.requests, "get"),
                              (shadow.sqlite3, "connect"), (socket, "create_connection"),
                              (shadow, "_publish_exclusive_verified"), (shadow, "reserve_shadow_prediction_slot"),
                              (Path, "write_bytes"), (Path, "mkdir"), (Path, "unlink"),
                              (CalibratedClassifierCV, "fit"), (CalibratedClassifierCV, "predict_proba"),
                              (CalibratedClassifierCV, "predict"), (LogisticRegression, "fit"),
                              (LogisticRegression, "predict_proba")):
                stack.enter_context(mock.patch.object(obj, name, side_effect=forbidden))
            self.verify(self.snapshot())
        self.assertEqual(set(seen), allowed)
        self.assertEqual({path: path.read_bytes() for path in allowed}, contents)
        self.assertEqual(set(self.project.rglob("*.joblib")), set())

    def test_concurrent_serializer_and_loader_cannot_touch_first_operation(self):
        entered, release = threading.Event(), threading.Event()
        actual = joblib.dump
        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError("Test de concurrence expire")
            return actual(*args, **kwargs)
        sentinel = object()
        with (mock.patch.object(sys.modules["__main__"], "CalibratedModelArtifact", sentinel, create=True),
              mock.patch.object(joblib, "dump", side_effect=blocked) as dump,
              mock.patch.object(joblib, "load") as load,
              ThreadPoolExecutor(max_workers=1) as pool):
            first = pool.submit(self.snapshot)
            try:
                self.assertTrue(entered.wait(5))
                filters = list(warnings.filters)
                with self.assertRaisesRegex(shadow.ShadowPredictionError, "deja en cours"):
                    self.snapshot()
                with self.assertRaisesRegex(shadow.ShadowPredictionError, "deja en cours"):
                    self.fixture._load()
                self.assertEqual(warnings.filters, filters)
                self.assertIs(sys.modules["__main__"].CalibratedModelArtifact, sentinel)
                load.assert_not_called()
                self.assertEqual(dump.call_count, 1)
            finally:
                release.set()
            first.result(timeout=5)
        self.assertFalse(shadow._MODEL_LOADING_LOCK.locked())

    def test_active_loader_excludes_state_serialization(self):
        entered, release = threading.Event(), threading.Event()
        actual = joblib.load
        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError("Test de concurrence expire")
            return actual(*args, **kwargs)
        with (mock.patch.object(joblib, "load", side_effect=blocked),
              mock.patch.object(joblib, "dump") as dump,
              ThreadPoolExecutor(max_workers=1) as pool):
            first = pool.submit(self.fixture._load)
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(shadow.ShadowPredictionError, "deja en cours"):
                    self.snapshot()
                dump.assert_not_called()
            finally:
                release.set()
            first.result(timeout=5)
        self.assertFalse(shadow._MODEL_LOADING_LOCK.locked())

    def test_reentrant_serialization_is_rejected_without_deadlock(self):
        actual = joblib.dump
        def reenter(*args, **kwargs):
            with self.assertRaisesRegex(shadow.ShadowPredictionError, "deja en cours"):
                self.snapshot()
            with self.assertRaisesRegex(shadow.ShadowPredictionError, "deja en cours"):
                self.fixture._load()
            return actual(*args, **kwargs)
        with mock.patch.object(joblib, "dump", side_effect=reenter) as dump:
            self.snapshot()
            dump.assert_called_once()

    def test_legacy_alias_is_not_created_or_changed_for_state_serialization(self):
        main = sys.modules["__main__"]
        sentinel = object()
        for original in (sentinel, None):
            with mock.patch.object(main, "CalibratedModelArtifact", original, create=True):
                self.verify(self.snapshot())
                self.assertIs(main.CalibratedModelArtifact, original)


class ShadowModelStatePolicyTests(unittest.TestCase):
    def test_serializer_matches_frozen_shadow_contract_not_legacy_evaluation_policy(self):
        doc = json.loads((loading_tests.PROJECT / loading_tests.SHADOW_PATH).read_bytes())
        state = doc["model_use"]["artifact_state_sha256_method"]
        self.assertEqual(state, {
            "serializer": "joblib.dump", "destination": "io.BytesIO", "compress": 3,
            "digest": "sha256(buffer.getvalue()).hexdigest()",
            "runtime_versions_must_equal_execution_manifest": True,
            "warning_handling": "model_use.warning_policy",
            "computed_immediately_before_predict_proba": True,
            "computed_immediately_after_predict_proba": True,
        })
        policy = doc["model_use"]["warning_policy"]
        self.assertEqual(shadow._MODEL_WARNING_POLICY_ID, policy["policy_id"])
        self.assertIn("ARTIFACT_STATE_SERIALIZATION",
                      policy["approved_load_or_state_serialization_warning"]["allowed_stages"])

    def test_private_apis_have_no_serializer_hash_runtime_or_prediction_override(self):
        self.assertEqual(list(inspect.signature(shadow._snapshot_frozen_shadow_model_state).parameters),
                         ["loaded_model", "project_directory"])
        self.assertEqual(list(inspect.signature(shadow._verify_frozen_shadow_model_state_unchanged).parameters),
                         ["loaded_model", "before", "project_directory"])

    def test_preview_has_no_state_measurement_or_execution_mode(self):
        forbidden = AssertionError("Apercu doit rester inerte")
        with (mock.patch.object(shadow, "_snapshot_frozen_shadow_model_state", side_effect=forbidden),
              mock.patch.object(shadow, "_verify_frozen_shadow_model_state_unchanged", side_effect=forbidden),
              mock.patch.object(shadow, "import_module", side_effect=forbidden),
              mock.patch.object(joblib, "load", side_effect=forbidden),
              mock.patch.object(joblib, "dump", side_effect=forbidden)):
            result = shadow.preview_shadow_prediction("2026-09-04", project_directory=loading_tests.PROJECT)
        self.assertFalse(result.execution_ready)
        self.assertFalse(result.model_deserialized)
        self.assertFalse(result.predictions_computed)


if __name__ == "__main__":
    unittest.main()
