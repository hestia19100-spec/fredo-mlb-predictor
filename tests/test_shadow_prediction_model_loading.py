"""Chargement shadow teste uniquement avec des artefacts synthetiques."""

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
import tempfile
import threading
import unittest
from unittest import mock
import warnings

import joblib
import numpy as np
import pandas
import scipy
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression

from src import calibrated_model
from src import shadow_prediction as shadow


PROJECT = Path(__file__).resolve().parents[1]
SHADOW_PATH = shadow.SHADOW_PROTOCOL_RELATIVE_PATH.as_posix()
MANIFEST_PATH = shadow.EXPECTED_ARTIFACT_MANIFEST_PATH
PROTOCOL_PATH = shadow.EXPECTED_MODEL_PROTOCOL_PATH
ARTIFACT_PATH = shadow.EXPECTED_MODEL_ARTIFACT_PATH
METADATA_PATHS = (SHADOW_PATH, MANIFEST_PATH, PROTOCOL_PATH)
RUNTIME = {
    "python": "3.12.1", "numpy": "2.5.2", "pandas": "3.0.5",
    "scipy": "1.18.1", "scikit_learn": "1.9.0", "joblib": "1.5.3",
}
RUNTIME_MODULES = {
    "numpy": np, "pandas": pandas, "scipy": scipy,
    "scikit_learn": sklearn, "joblib": joblib,
}


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                       sort_keys=True, indent=2) + "\n").encode("utf-8")


def _dump_synthetic(artifact):
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer, compress=3)
    return buffer.getvalue()


class _ArtifactSubclass(calibrated_model.CalibratedModelArtifact):
    pass


class ShadowModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.project = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.shadow_doc = json.loads((PROJECT / SHADOW_PATH).read_bytes())
        self.manifest = json.loads((PROJECT / MANIFEST_PATH).read_bytes())
        self.protocol_bytes = (PROJECT / PROTOCOL_PATH).read_bytes()
        self.stack.enter_context(mock.patch.object(shadow.platform, "python_version",
                                                  return_value=RUNTIME["python"]))
        self.stack.enter_context(mock.patch.object(shadow.distribution_metadata, "version",
            side_effect=lambda name: RUNTIME[name.replace("scikit-learn", "scikit_learn")]))
        for key, module in RUNTIME_MODULES.items():
            self.stack.enter_context(mock.patch.object(module, "__version__", RUNTIME[key]))
        # Construction en memoire, sans fit, sans donnees ni probabilites.
        classifier = CalibratedClassifierCV(method="sigmoid")
        classifier.classes_ = np.array([0, 1], dtype=np.int64)
        classifier.n_features_in_ = 8
        m, c = self.manifest["model"], self.manifest["chronology"]
        self.artifact = calibrated_model.CalibratedModelArtifact(
            artifact_format_version=1,
            calibrated_model_version=m["calibrated_model_version"],
            base_model_version=m["base_model_version"],
            code_version=self.manifest["artifact"]["code_version"],
            dataset_version=self.manifest["dataset"]["version"],
            dataset_sha256=self.manifest["dataset"]["sha256"],
            protocol_sha256=self.manifest["protocol"]["sha256"],
            feature_columns=tuple(m["feature_columns"]),
            base_training_seasons=tuple(c["base_training_seasons"]),
            calibration_season=c["calibration_season"],
            sealed_test_seasons=tuple(c["sealed_test_seasons"]),
            recent_seasons=tuple(c["recent_seasons"]),
            calibration_method=m["calibration_method"],
            sklearn_version=RUNTIME["scikit_learn"], numpy_version=RUNTIME["numpy"],
            calibrated_classifier=classifier,
        )
        self._pin(_dump_synthetic(self.artifact))

    def _pin(self, content):
        # Substitutions de fixtures uniquement : l'API ne propose aucun bypass.
        self.content = content
        self.manifest["artifact"]["sha256"] = _sha(content)
        self.manifest["artifact"]["size_bytes"] = len(content)
        manifest_bytes = _json_bytes(self.manifest)
        self.shadow_doc["validated_lineage"]["model_artifact"].update(
            sha256=_sha(content), size_bytes=len(content))
        self.shadow_doc["validated_lineage"]["artifact_manifest"]["sha256"] = _sha(manifest_bytes)
        shadow_bytes = _json_bytes(self.shadow_doc)
        for relative, raw in zip(METADATA_PATHS, (shadow_bytes, manifest_bytes, self.protocol_bytes)):
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        for name, value in (
            ("EXPECTED_MODEL_ARTIFACT_SHA256", _sha(content)),
            ("EXPECTED_MODEL_ARTIFACT_SIZE_BYTES", len(content)),
            ("EXPECTED_ARTIFACT_MANIFEST_SHA256", _sha(manifest_bytes)),
            ("EXPECTED_SHADOW_PROTOCOL_SHA256", _sha(shadow_bytes)),
        ):
            self.stack.enter_context(mock.patch.object(shadow, name, value))
        self.proof = shadow.ShadowModelPrerequisites(
            artifact_path=self.project / ARTIFACT_PATH,
            artifact_relative_path=ARTIFACT_PATH,
            artifact_sha256=_sha(content), artifact_size_bytes=len(content),
            artifact_bytes=content, artifact_manifest_sha256=_sha(manifest_bytes),
            model_protocol_sha256=_sha(self.protocol_bytes),
            shadow_protocol_sha256=_sha(shadow_bytes),
            runtime_versions=tuple(sorted(RUNTIME.items())),
        )

    def _load(self, proof=None):
        return shadow._load_frozen_shadow_model(
            self.proof if proof is None else proof, project_directory=self.project)

    def test_real_synthetic_joblib_round_trip_has_no_prediction_authority(self):
        with mock.patch.object(joblib, "load", wraps=joblib.load) as load:
            result = self._load()
        load.assert_called_once()
        args, kwargs = load.call_args
        self.assertIs(type(args[0]), io.BytesIO)
        self.assertEqual(args[0].getvalue(), self.content)
        self.assertEqual(kwargs, {})
        self.assertIs(type(result.artifact), calibrated_model.CalibratedModelArtifact)
        self.assertEqual(result.artifact.feature_columns, self.artifact.feature_columns)
        self.assertEqual(result.artifact_sha256, _sha(self.content))
        self.assertEqual(result.runtime_versions, tuple(sorted(RUNTIME.items())))
        self.assertIs(result.model_deserialized, True)
        self.assertIs(type(result.approved_deserialization_warning_count), int)
        self.assertGreaterEqual(result.approved_deserialization_warning_count, 0)
        for flag in ("execution_ready", "predictions_computed",
                     "execution_manifest_verified", "activation_verified"):
            self.assertIs(getattr(result, flag), False)
        with self.assertRaises(FrozenInstanceError):
            result.execution_ready = True

    def test_legacy_main_alias_is_used_only_during_real_load_then_restored(self):
        cls = calibrated_model.CalibratedModelArtifact
        main = sys.modules["__main__"]
        with (mock.patch.object(cls, "__module__", "__main__"),
              mock.patch.object(main, "CalibratedModelArtifact", cls, create=True)):
            content = _dump_synthetic(self.artifact)
        self._pin(content)
        sentinel = object()
        actual_load = joblib.load
        def guarded(buffer):
            self.assertIs(main.CalibratedModelArtifact, cls)
            return actual_load(buffer)
        with (mock.patch.object(main, "CalibratedModelArtifact", sentinel, create=True),
              mock.patch.object(joblib, "load", side_effect=guarded)):
            result = self._load()
            self.assertIs(main.CalibratedModelArtifact, sentinel)
        self.assertIs(type(result.artifact), cls)

    def test_missing_alias_is_removed_on_success(self):
        main = sys.modules["__main__"]
        sentinel = object()
        original = vars(main).pop("CalibratedModelArtifact", sentinel)
        try:
            self._load()
            self.assertNotIn("CalibratedModelArtifact", vars(main))
        finally:
            if original is not sentinel:
                vars(main)["CalibratedModelArtifact"] = original

    def test_existing_none_alias_is_preserved(self):
        with mock.patch.object(sys.modules["__main__"], "CalibratedModelArtifact", None, create=True):
            self._load()
            self.assertIsNone(sys.modules["__main__"].CalibratedModelArtifact)

    def test_alias_filters_and_lock_are_restored_on_every_loading_exception(self):
        main = sys.modules["__main__"]
        sentinel = object()
        filters = list(warnings.filters)
        for error in (ValueError("bad"), OSError("bad"), RuntimeError("bad"),
                      KeyboardInterrupt(), SystemExit(1)):
            with self.subTest(error=type(error).__name__):
                expected = shadow.ShadowPredictionError if isinstance(error, Exception) else type(error)
                with (mock.patch.object(main, "CalibratedModelArtifact", sentinel, create=True),
                      mock.patch.object(joblib, "load", side_effect=error) as load):
                    with self.assertRaises(expected):
                        self._load()
                    load.assert_called_once()
                    self.assertIs(main.CalibratedModelArtifact, sentinel)
                self.assertEqual(warnings.filters, filters)
                self.assertFalse(shadow._MODEL_LOADING_LOCK.locked())

    def test_missing_alias_remains_absent_after_loading_exception(self):
        main = sys.modules["__main__"]
        sentinel = object()
        original = vars(main).pop("CalibratedModelArtifact", sentinel)
        try:
            with mock.patch.object(joblib, "load", side_effect=ValueError("bad")):
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._load()
            self.assertNotIn("CalibratedModelArtifact", vars(main))
        finally:
            if original is not sentinel:
                vars(main)["CalibratedModelArtifact"] = original

    def test_only_metadata_is_read_no_model_file_network_database_or_writes(self):
        original_open = io.open
        seen = []
        allowed = {self.project / path for path in METADATA_PATHS}
        before = {p: p.read_bytes() for p in allowed}
        def guard(path, mode="r", *args, **kwargs):
            self.assertIn(Path(path), allowed)
            self.assertFalse(any(c in mode for c in "wax+"))
            seen.append(Path(path))
            return original_open(path, mode, *args, **kwargs)
        forbidden = AssertionError("Operation officielle interdite")
        with (mock.patch("io.open", side_effect=guard),
              mock.patch.object(shadow.requests, "get", side_effect=forbidden),
              mock.patch.object(shadow.sqlite3, "connect", side_effect=forbidden),
              mock.patch.object(socket, "create_connection", side_effect=forbidden),
              mock.patch.object(shadow, "_publish_exclusive_verified", side_effect=forbidden),
              mock.patch.object(Path, "mkdir", side_effect=forbidden),
              mock.patch.object(Path, "write_bytes", side_effect=forbidden),
              mock.patch.object(Path, "unlink", side_effect=forbidden),
              mock.patch.object(joblib, "dump", side_effect=forbidden)):
            self._load()
        self.assertEqual(set(seen), allowed)
        self.assertEqual(len(seen), 6)
        self.assertEqual({p: p.read_bytes() for p in allowed}, before)
        self.assertFalse((self.project / ARTIFACT_PATH).exists())

    def test_fit_predict_and_calibration_entry_points_are_never_called(self):
        forbidden = AssertionError("Fit ou prediction interdit")
        with ExitStack() as stack:
            for cls, names in (
                (CalibratedClassifierCV, ("fit", "predict", "predict_proba")),
                (LogisticRegression, ("fit", "predict", "predict_proba")),
                (calibrated_model, ("prepare_calibrated_model", "prepare_loaded_calibrated_model",
                                    "load_calibrated_artifact", "write_calibrated_artifact")),
            ):
                for name in names:
                    stack.enter_context(mock.patch.object(cls, name, side_effect=forbidden))
            self._load()

    def test_prerequisite_primitive_connects_to_loader_without_reopening_joblib(self):
        path = self.project / ARTIFACT_PATH
        path.parent.mkdir()
        path.write_bytes(self.content)
        proof = shadow.verify_frozen_model_prerequisites(project_directory=self.project)
        path.write_bytes(b"replaced after verification")
        with mock.patch.object(joblib, "load", wraps=joblib.load) as load:
            result = self._load(proof)
        self.assertEqual(load.call_args.args[0].getvalue(), self.content)
        self.assertEqual(result.artifact_sha256, _sha(self.content))
        self.assertEqual(path.read_bytes(), b"replaced after verification")

    def test_forged_proof_is_rejected_before_file_read_import_or_load(self):
        changes = (
            ("artifact_bytes", b"bad"), ("artifact_bytes", bytearray(self.content)),
            ("artifact_sha256", "0" * 64), ("artifact_size_bytes", True),
            ("artifact_size_bytes", len(self.content) + 1),
            ("artifact_relative_path", "models/other.joblib"),
            ("artifact_manifest_sha256", "0" * 64), ("model_protocol_sha256", "0" * 64),
            ("shadow_protocol_sha256", "0" * 64),
            ("artifact_path", self.project / "elsewhere.joblib"),
            ("artifact_path", str(self.project / ARTIFACT_PATH)),
            ("validation_scope", "ALL"), ("execution_ready", True),
            ("activation_verified", True), ("execution_manifest_verified", True),
            ("model_deserialized", True), ("predictions_computed", True),
            ("runtime_versions_source", "USER"), ("runtime_versions", list(self.proof.runtime_versions)),
        )
        for key, value in changes:
            with self.subTest(field=key, value=type(value).__name__):
                wrong = copy(self.proof)
                object.__setattr__(wrong, key, value)
                with (mock.patch.object(Path, "read_bytes") as read,
                      mock.patch.object(shadow, "import_module") as imports,
                      mock.patch.object(joblib, "load") as load):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._load(wrong)
                    read.assert_not_called()
                    imports.assert_not_called()
                    load.assert_not_called()

    def test_nonproof_and_wrong_project_are_rejected(self):
        with mock.patch.object(shadow, "import_module") as imports:
            for wrong in (object(), {}, self.artifact):
                with self.subTest(wrong=type(wrong).__name__):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._load(wrong)
            for project in (Path("relative"), str(self.project), self.project / ".." / "other"):
                with self.subTest(project=project):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        shadow._load_frozen_shadow_model(self.proof, project_directory=project)
            imports.assert_not_called()

    def test_every_metadata_hash_is_required_before_imports(self):
        for relative in METADATA_PATHS:
            with self.subTest(relative=relative):
                path = self.project / relative
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                try:
                    with mock.patch.object(shadow, "import_module") as imports:
                        with self.assertRaises(shadow.ShadowPredictionError):
                            self._load()
                        imports.assert_not_called()
                finally:
                    path.write_bytes(original)

    def test_metadata_changed_during_load_is_rejected_after_alias_restoration(self):
        path = self.project / MANIFEST_PATH
        main, sentinel = sys.modules["__main__"], object()
        def mutate(_):
            path.write_bytes(path.read_bytes() + b" ")
            return self.artifact
        with (mock.patch.object(main, "CalibratedModelArtifact", sentinel, create=True),
              mock.patch.object(joblib, "load", side_effect=mutate)):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._load()
            self.assertIs(main.CalibratedModelArtifact, sentinel)

    def test_each_installed_version_is_rechecked_before_imports(self):
        for key in RUNTIME:
            with self.subTest(key=key):
                with (mock.patch.object(shadow, "_installed_model_runtime_versions",
                                       return_value=dict(RUNTIME, **{key: "0.0.0"})),
                      mock.patch.object(shadow, "import_module") as imports):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._load()
                    imports.assert_not_called()

    def test_each_imported_module_version_must_match_before_load(self):
        for key, module in RUNTIME_MODULES.items():
            with self.subTest(key=key):
                with (mock.patch.object(module, "__version__", "0.0.0"),
                      mock.patch.object(joblib, "load") as load):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._load()
                    load.assert_not_called()

    def test_proof_runtime_must_equal_manifest_exactly(self):
        for versions in ((), tuple(reversed(self.proof.runtime_versions)),
                         self.proof.runtime_versions + (("extra", "1"),),
                         tuple(sorted(dict(RUNTIME, numpy="0").items()))):
            with self.subTest(versions=versions), mock.patch.object(shadow, "import_module") as imports:
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._load(replace(self.proof, runtime_versions=versions))
                imports.assert_not_called()

    def test_runtime_change_during_load_is_rejected(self):
        for mode in ("installed", "imported"):
            with self.subTest(mode=mode):
                name = ("_installed_model_runtime_versions" if mode == "installed"
                        else "_shadow_imported_runtime_versions")
                with mock.patch.object(shadow, name,
                                       side_effect=[RUNTIME, dict(RUNTIME, joblib="0")]):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._load()

    def test_import_failure_does_not_change_alias_or_attempt_load(self):
        sentinel, main = object(), sys.modules["__main__"]
        with (mock.patch.object(main, "CalibratedModelArtifact", sentinel, create=True),
              mock.patch.object(shadow, "import_module", side_effect=ImportError("missing")),
              mock.patch.object(joblib, "load") as load):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._load()
            load.assert_not_called()
            self.assertIs(main.CalibratedModelArtifact, sentinel)
        self.assertFalse(shadow._MODEL_LOADING_LOCK.locked())

    def _warning_load(self, message, category, module, *, repeat=1):
        def load(_):
            for _ in range(repeat):
                warnings.warn_explicit(message, category, filename="synthetic.py",
                                       lineno=1, module=module)
            return self.artifact
        return load

    def test_exact_approved_warnings_are_counted_even_when_repeated(self):
        emit = self._warning_load("Setting the shape on a NumPy array has been deprecated.",
                                  DeprecationWarning, "joblib.numpy_pickle", repeat=2)
        filters = list(warnings.filters)
        with mock.patch.object(joblib, "load", side_effect=emit) as load:
            result = self._load()
        self.assertEqual(result.approved_deserialization_warning_count, 2)
        self.assertEqual(result.warning_policy_id, "VERIFIED_JOBLIB_NUMPY_COMPATIBILITY_V1")
        self.assertEqual(warnings.filters, filters)
        load.assert_called_once()

    def test_registered_regex_semantics_are_preserved(self):
        class SubDeprecation(DeprecationWarning):
            pass
        for message, category, module in (
            ("setting the SHAPE on a NumPy array has been deprecated suffix", DeprecationWarning, "joblib.numpy_pickle"),
            ("Setting the shape on a NumPy array has been deprecated", SubDeprecation, "joblib.numpy_pickle.compat"),
        ):
            with self.subTest(message=message, module=module):
                with mock.patch.object(joblib, "load", side_effect=self._warning_load(message, category, module)):
                    self.assertEqual(self._load().approved_deserialization_warning_count, 1)

    def test_every_other_warning_aborts_without_retry_and_restores_filters(self):
        approved = "Setting the shape on a NumPy array has been deprecated."
        cases = (
            (approved, UserWarning, "joblib.numpy_pickle"),
            (approved, FutureWarning, "joblib.numpy_pickle"),
            (approved, DeprecationWarning, "Joblib.numpy_pickle"),
            (approved, DeprecationWarning, "other.numpy_pickle"),
            ("prefix " + approved, DeprecationWarning, "joblib.numpy_pickle"),
            ("Another deprecation", DeprecationWarning, "joblib.numpy_pickle"),
        )
        for message, category, module in cases:
            with self.subTest(category=category, module=module, message=message):
                filters = list(warnings.filters)
                with mock.patch.object(joblib, "load", side_effect=self._warning_load(message, category, module)) as load:
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._load()
                    load.assert_called_once()
                self.assertEqual(warnings.filters, filters)
                self.assertFalse(shadow._MODEL_LOADING_LOCK.locked())

    def test_inconsistent_sklearn_version_warning_is_never_allowed(self):
        def load(_):
            warnings.warn(InconsistentVersionWarning(
                estimator_name="Synthetic", current_sklearn_version="1.9.0",
                original_sklearn_version="0.0.0"))
            return self.artifact
        with mock.patch.object(joblib, "load", side_effect=load):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._load()

    def test_global_ignore_filter_does_not_hide_an_unexpected_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            emit = self._warning_load("unexpected", UserWarning, "elsewhere")
            with mock.patch.object(joblib, "load", side_effect=emit):
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._load()

    def test_approved_warning_is_forbidden_during_import_or_metadata_validation(self):
        emit = self._warning_load("Setting the shape on a NumPy array has been deprecated.",
                                  DeprecationWarning, "joblib.numpy_pickle")
        with mock.patch.object(shadow, "import_module", side_effect=emit):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._load()
        with mock.patch.object(shadow, "_validate_loaded_shadow_model_metadata",
                               side_effect=lambda *args: emit(None)):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._load()

    def test_all_artifact_metadata_fields_must_match_registered_manifest(self):
        for name in self.artifact.__dataclass_fields__:
            if name == "calibrated_classifier":
                continue
            original = getattr(self.artifact, name)
            wrong = True if type(original) is int else ([] if type(original) is tuple else "wrong")
            with self.subTest(field=name):
                artifact = replace(self.artifact, **{name: wrong})
                with mock.patch.object(joblib, "load", return_value=artifact):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._load()

    def test_warnings_during_preflight_and_final_file_checks_also_abort(self):
        original_read = Path.read_bytes
        for warning_at in (1, 4):
            with self.subTest(warning_at=warning_at):
                calls = 0
                def warned_read(path):
                    nonlocal calls
                    calls += 1
                    if calls == warning_at:
                        warnings.warn_explicit(
                            "Setting the shape on a NumPy array has been deprecated.",
                            DeprecationWarning, filename="synthetic.py", lineno=1,
                            module="joblib.numpy_pickle")
                    return original_read(path)
                with (mock.patch.object(Path, "read_bytes", warned_read),
                      mock.patch.object(joblib, "load", return_value=self.artifact) as load):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._load()
                    self.assertEqual(load.call_count, 0 if warning_at == 1 else 1)
                self.assertFalse(shadow._MODEL_LOADING_LOCK.locked())

    def test_artifact_subclasses_and_other_objects_are_rejected(self):
        subclass = object.__new__(_ArtifactSubclass)
        for obj in (subclass, {}, object(), self.artifact.calibrated_classifier):
            with self.subTest(obj=type(obj).__name__):
                with mock.patch.object(joblib, "load", return_value=obj):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._load()

    def test_wrong_classifier_or_calibration_method_is_rejected(self):
        for classifier in (object(), CalibratedClassifierCV(method="isotonic")):
            with self.subTest(classifier=type(classifier).__name__):
                artifact = replace(self.artifact, calibrated_classifier=classifier)
                with mock.patch.object(joblib, "load", return_value=artifact):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._load()

    def test_classes_have_exact_shape_order_and_integer_type(self):
        for classes in (None, [0, 1], np.array([1, 0]), np.array([0]), np.array([0, 1, 2]),
                        np.array([[0, 1]]), np.array([False, True]), np.array([0., 1.]),
                        np.array(["0", "1"]), np.array([0, 1], dtype=object)):
            with self.subTest(classes=repr(classes)):
                self.artifact.calibrated_classifier.classes_ = classes
                with mock.patch.object(joblib, "load", return_value=self.artifact):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._load()

    def test_classifier_requires_eight_features_without_coercion(self):
        for n_features in (None, True, np.bool_(True), 7, 9, 8.0, "8"):
            with self.subTest(n_features=n_features):
                self.artifact.calibrated_classifier.n_features_in_ = n_features
                with mock.patch.object(joblib, "load", return_value=self.artifact):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._load()
        self.artifact.calibrated_classifier.n_features_in_ = np.int64(8)
        with mock.patch.object(joblib, "load", return_value=self.artifact):
            self._load()

    def test_concurrent_loader_is_rejected_without_touching_alias_or_first_loader(self):
        entered, release = threading.Event(), threading.Event()
        sentinel, main = object(), sys.modules["__main__"]
        def blocked(_):
            entered.set()
            if not release.wait(5):
                raise AssertionError("Test de concurrence expire")
            self.assertIs(main.CalibratedModelArtifact, calibrated_model.CalibratedModelArtifact)
            return self.artifact
        with (mock.patch.object(main, "CalibratedModelArtifact", sentinel, create=True),
              mock.patch.object(joblib, "load", side_effect=blocked) as load,
              ThreadPoolExecutor(max_workers=1) as executor):
            first = executor.submit(self._load)
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(shadow.ShadowPredictionError, "deja en cours"):
                    self._load()
                self.assertIs(main.CalibratedModelArtifact, calibrated_model.CalibratedModelArtifact)
                self.assertEqual(load.call_count, 1)
            finally:
                release.set()
            self.assertIs(first.result(timeout=5).artifact, self.artifact)
            self.assertIs(main.CalibratedModelArtifact, sentinel)

    def test_reentrant_loader_is_rejected_without_deadlock(self):
        def nested(_):
            with self.assertRaisesRegex(shadow.ShadowPredictionError, "deja en cours"):
                self._load()
            return self.artifact
        with mock.patch.object(joblib, "load", side_effect=nested) as load:
            self._load()
            load.assert_called_once()

    def test_private_api_has_no_hash_runtime_or_loader_override(self):
        parameters = inspect.signature(shadow._load_frozen_shadow_model).parameters
        self.assertEqual(list(parameters), ["prerequisites", "project_directory"])
        self.assertIs(parameters["project_directory"].kind, inspect.Parameter.KEYWORD_ONLY)
        with self.assertRaises(TypeError):
            shadow._load_frozen_shadow_model(self.proof, expected_sha256="0" * 64)

    def test_preview_never_loads_or_imports_model_libraries(self):
        forbidden = AssertionError("Chargement dans apercu interdit")
        with (mock.patch.object(shadow, "_load_frozen_shadow_model", side_effect=forbidden),
              mock.patch.object(shadow, "import_module", side_effect=forbidden),
              mock.patch.object(joblib, "load", side_effect=forbidden)):
            result = shadow.preview_shadow_prediction("2026-09-04", project_directory=self.project)
        self.assertFalse(result.execution_ready)
        self.assertFalse(result.model_artifact_read)


class ShadowModelLoadingRegisteredPolicyTests(unittest.TestCase):
    def test_loader_warning_constants_match_frozen_v2_policy(self):
        document = json.loads((PROJECT / SHADOW_PATH).read_bytes())
        policy = document["model_use"]["warning_policy"]
        approved = policy["approved_load_or_state_serialization_warning"]
        self.assertEqual(shadow._MODEL_WARNING_POLICY_ID, policy["policy_id"])
        self.assertEqual(shadow._MODEL_WARNING_MESSAGE, approved["message_regex"])
        self.assertEqual(shadow._MODEL_WARNING_MODULE, approved["module_regex"])
        self.assertEqual(approved["category"], "DeprecationWarning")
        self.assertIn("ARTIFACT_DESERIALIZATION", approved["allowed_stages"])
        self.assertEqual(policy["default"], "ABORT_ENTIRE_RUN")


if __name__ == "__main__":
    unittest.main()
