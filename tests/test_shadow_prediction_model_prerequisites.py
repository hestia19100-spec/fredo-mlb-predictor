"""Prerequis du modele fige, sans aucun modele reel ni deserialisation."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import FrozenInstanceError
import builtins
import hashlib
import inspect
import io
import json
from pathlib import Path
import pickle
import socket
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

from src import shadow_prediction as shadow


PROJECT = Path(__file__).resolve().parents[1]
SHADOW_PATH = shadow.SHADOW_PROTOCOL_RELATIVE_PATH.as_posix()
MANIFEST_PATH = shadow.EXPECTED_ARTIFACT_MANIFEST_PATH
MODEL_PROTOCOL_PATH = shadow.EXPECTED_MODEL_PROTOCOL_PATH
ARTIFACT_PATH = shadow.EXPECTED_MODEL_ARTIFACT_PATH
EXPECTED_PATHS = (SHADOW_PATH, MANIFEST_PATH, MODEL_PROTOCOL_PATH, ARTIFACT_PATH)
RUNTIME = {
    "python": "3.12.1", "numpy": "2.5.2", "pandas": "3.0.5",
    "scipy": "1.18.1", "scikit_learn": "1.9.0", "joblib": "1.5.3",
}
DISTRIBUTIONS = {
    "numpy": "2.5.2", "pandas": "3.0.5", "scipy": "1.18.1",
    "scikit-learn": "1.9.0", "joblib": "1.5.3",
}


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _json_bytes(payload):
    return (json.dumps(payload, ensure_ascii=False, allow_nan=False,
                       sort_keys=True, indent=2) + "\n").encode("utf-8")


class _PickleTrap:
    def __reduce__(self):
        # Cette expression echouerait si le faux modele etait interprete.
        return eval, ("1 / 0",)


class ShadowModelPrerequisiteTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.project = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.shadow_doc = json.loads((PROJECT / SHADOW_PATH).read_bytes())
        self.manifest = json.loads((PROJECT / MANIFEST_PATH).read_bytes())
        self.model_protocol = json.loads((PROJECT / MODEL_PROTOCOL_PATH).read_bytes())
        raw = pickle.dumps(_PickleTrap(), protocol=4)
        self.artifact = raw + b"\0" * (1589 - len(raw))
        (self.project / ARTIFACT_PATH).parent.mkdir(parents=True)
        (self.project / ARTIFACT_PATH).write_bytes(self.artifact)
        self._pin_documents()
        self.stack.enter_context(mock.patch.object(
            shadow.platform, "python_version", return_value=RUNTIME["python"]))
        self.versions = self.stack.enter_context(mock.patch.object(
            shadow.distribution_metadata, "version",
            side_effect=lambda distribution: DISTRIBUTIONS[distribution]))

    def _pin_documents(self):
        # Les substitutions de hashes n'existent que dans ces fixtures.
        # Aucun argument public ne permet de changer un hash ou une version.
        artifact_hash = _digest(self.artifact)
        protocol_bytes = _json_bytes(self.model_protocol)
        protocol_hash = _digest(protocol_bytes)
        self.manifest["artifact"]["sha256"] = artifact_hash
        self.manifest["protocol"]["sha256"] = protocol_hash
        manifest_bytes = _json_bytes(self.manifest)
        manifest_hash = _digest(manifest_bytes)
        self.shadow_doc["validated_lineage"]["model_artifact"]["sha256"] = artifact_hash
        self.shadow_doc["validated_lineage"]["artifact_manifest"]["sha256"] = manifest_hash
        self.shadow_doc["validated_lineage"]["model_protocol"]["sha256"] = protocol_hash
        shadow_bytes = _json_bytes(self.shadow_doc)
        for relative, content in (
            (SHADOW_PATH, shadow_bytes), (MANIFEST_PATH, manifest_bytes),
            (MODEL_PROTOCOL_PATH, protocol_bytes),
        ):
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        for name, digest in (
            ("EXPECTED_MODEL_ARTIFACT_SHA256", artifact_hash),
            ("EXPECTED_MODEL_PROTOCOL_SHA256", protocol_hash),
            ("EXPECTED_ARTIFACT_MANIFEST_SHA256", manifest_hash),
            ("EXPECTED_SHADOW_PROTOCOL_SHA256", _digest(shadow_bytes)),
        ):
            self.stack.enter_context(mock.patch.object(shadow, name, digest))

    def _check(self):
        return shadow.verify_frozen_model_prerequisites(project_directory=self.project)

    def _files(self):
        return {p.relative_to(self.project).as_posix(): p.read_bytes()
                for p in self.project.rglob("*") if p.is_file()}

    def test_verified_bytes_proof_is_immutable_and_not_an_execution_authorization(self):
        before = self._files()
        proof = self._check()
        self.assertIs(type(proof), shadow.ShadowModelPrerequisites)
        self.assertEqual(proof.artifact_bytes, self.artifact)
        self.assertIs(type(proof.artifact_bytes), bytes)
        self.assertEqual(proof.artifact_path, self.project / ARTIFACT_PATH)
        self.assertEqual(proof.artifact_relative_path, ARTIFACT_PATH)
        self.assertEqual(proof.artifact_size_bytes, 1589)
        self.assertEqual(proof.artifact_sha256, _digest(self.artifact))
        self.assertEqual(proof.runtime_versions, tuple(sorted(RUNTIME.items())))
        self.assertEqual(proof.validation_scope, "FROZEN_MODEL_FILES_AND_INSTALLED_RUNTIME_ONLY")
        for flag in ("execution_manifest_verified", "activation_verified",
                     "model_deserialized", "predictions_computed", "execution_ready"):
            self.assertIs(getattr(proof, flag), False)
        self.assertEqual(self._files(), before)
        self.assertNotIn(repr(self.artifact), repr(proof))
        with self.assertRaises(FrozenInstanceError):
            proof.execution_ready = True
        for relative, attribute in (
            (SHADOW_PATH, "shadow_protocol_sha256"),
            (MANIFEST_PATH, "artifact_manifest_sha256"),
            (MODEL_PROTOCOL_PATH, "model_protocol_sha256"),
        ):
            self.assertEqual(getattr(proof, attribute), _digest(before[relative]))

    def test_only_four_files_are_read_and_nothing_is_created(self):
        original_open = io.open
        seen = []
        allowed = {self.project / path for path in EXPECTED_PATHS}

        def guarded_open(path, mode="r", *args, **kwargs):
            self.assertNotIn("+", mode)
            self.assertFalse(any(letter in mode for letter in "wax"))
            self.assertIn(Path(path), allowed)
            seen.append(Path(path))
            return original_open(path, mode, *args, **kwargs)

        forbidden = AssertionError("Operation hors perimetre")
        with (mock.patch("io.open", side_effect=guarded_open),
              mock.patch.object(Path, "mkdir", side_effect=forbidden),
              mock.patch.object(Path, "write_bytes", side_effect=forbidden),
              mock.patch.object(Path, "write_text", side_effect=forbidden),
              mock.patch.object(Path, "unlink", side_effect=forbidden),
              mock.patch.object(shadow.requests, "get", side_effect=forbidden),
              mock.patch.object(shadow.sqlite3, "connect", side_effect=forbidden),
              mock.patch.object(socket, "create_connection", side_effect=forbidden),
              mock.patch.object(shadow, "_publish_exclusive_verified", side_effect=forbidden)):
            self._check()
        self.assertEqual(set(seen), allowed)
        self.assertEqual(len(seen), 8)

    def test_no_ml_import_no_pickle_load_and_no_legacy_alias_change(self):
        forbidden = AssertionError("Import ou chargement du modele interdit")
        original_import = builtins.__import__
        blocked = {"numpy", "pandas", "scipy", "sklearn", "joblib", "pickle"}

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in blocked or name.startswith("src.calibrated_model"):
                raise forbidden
            return original_import(name, *args, **kwargs)

        sentinel = object()
        fake_joblib = types.ModuleType("joblib")
        fake_joblib.load = mock.Mock(side_effect=forbidden)
        fake_joblib.dump = mock.Mock(side_effect=forbidden)
        with (mock.patch.object(builtins, "__import__", side_effect=guarded_import),
              mock.patch.object(pickle, "load", side_effect=forbidden),
              mock.patch.object(pickle, "loads", side_effect=forbidden),
              mock.patch.dict(sys.modules, {"joblib": fake_joblib}),
              mock.patch.object(sys.modules["__main__"], "CalibratedModelArtifact",
                                sentinel, create=True)):
            self._check()
            self.assertIs(sys.modules["__main__"].CalibratedModelArtifact, sentinel)
        fake_joblib.load.assert_not_called()
        fake_joblib.dump.assert_not_called()

    def test_each_metadata_hash_is_checked_before_json_interpretation(self):
        for relative in EXPECTED_PATHS[:3]:
            with self.subTest(relative=relative):
                path = self.project / relative
                original = path.read_bytes()
                path.write_bytes(b"not json")
                try:
                    with mock.patch.object(shadow, "_decode_pinned_model_json") as decode:
                        with self.assertRaises(shadow.ShadowPredictionError):
                            self._check()
                        decode.assert_not_called()
                finally:
                    path.write_bytes(original)

    def test_model_hash_and_size_are_both_required(self):
        path = self.project / ARTIFACT_PATH
        for content in (b"", self.artifact[:-1], self.artifact + b"x",
                        b"x" + self.artifact[1:]):
            with self.subTest(length=len(content)):
                path.write_bytes(content)
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._check()
        path.write_bytes(self.artifact)

    def test_wrong_model_size_is_rejected_without_reading_its_bytes(self):
        path = self.project / ARTIFACT_PATH
        path.write_bytes(b"short")
        original_read = Path.read_bytes
        def guard(current):
            self.assertNotEqual(current, path)
            return original_read(current)
        with mock.patch.object(Path, "read_bytes", guard):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._check()

    def test_missing_files_or_directories_are_not_recreated(self):
        for relative in EXPECTED_PATHS:
            with self.subTest(relative=relative):
                path = self.project / relative
                original = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._check()
                    self.assertFalse(path.exists())
                finally:
                    path.write_bytes(original)

    def test_directory_instead_of_model_is_rejected(self):
        path = self.project / ARTIFACT_PATH
        path.unlink()
        path.mkdir()
        with self.assertRaises(shadow.ShadowPredictionError):
            self._check()
        self.assertTrue(path.is_dir())

    def test_symlink_components_are_rejected_without_reading_target(self):
        # Simuler uniquement lstat permet de tester aussi sur Windows sans
        # privilege de creation de liens. Le refus precede toute lecture.
        original_mode = shadow._lstat_mode
        for bad in (self.project, self.project / "models",
                    self.project / ARTIFACT_PATH):
            with self.subTest(bad=bad):
                def mode(path):
                    return stat.S_IFLNK if path == bad else original_mode(path)
                with mock.patch.object(shadow, "_lstat_mode", side_effect=mode):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._check()

    def test_real_symbolic_model_link_is_rejected(self):
        path = self.project / ARTIFACT_PATH
        other = self.project / "outside.joblib"
        other.write_bytes(self.artifact)
        path.unlink()
        try:
            path.symlink_to(other)
        except (OSError, NotImplementedError):
            self.skipTest("Liens symboliques indisponibles sur ce systeme.")
        with self.assertRaises(shadow.ShadowPredictionError):
            self._check()
        self.assertEqual(other.read_bytes(), self.artifact)

    def test_reparse_or_unreadable_path_is_closed(self):
        original = shadow._lstat_mode
        for refused in (shadow._PATH_UNREADABLE, shadow._PATH_MISSING):
            with self.subTest(refused=refused):
                with mock.patch.object(shadow, "_lstat_mode", side_effect=lambda path:
                    refused if path == self.project / ARTIFACT_PATH else original(path)):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._check()

    def test_wrong_project_path_is_rejected_without_opening_files(self):
        for path in (str(self.project), Path("relative"), self.project / ".." / "other"):
            with self.subTest(path=path), mock.patch.object(Path, "read_bytes") as read:
                with self.assertRaises(shadow.ShadowPredictionError):
                    shadow.verify_frozen_model_prerequisites(project_directory=path)
                read.assert_not_called()

    def test_every_runtime_version_must_equal_artifact_manifest_before_model_read(self):
        original_read = Path.read_bytes
        def guarded(path):
            self.assertNotEqual(path, self.project / ARTIFACT_PATH)
            return original_read(path)
        for key in RUNTIME:
            with self.subTest(key=key):
                wrong = dict(RUNTIME, **{key: "0.0.0"})
                with (mock.patch.object(shadow, "_installed_model_runtime_versions", return_value=wrong),
                      mock.patch.object(Path, "read_bytes", guarded)):
                    with self.assertRaisesRegex(shadow.ShadowPredictionError, key):
                        self._check()

    def test_runtime_collection_uses_exact_distributions_and_python(self):
        self._check()
        expected = [mock.call(name) for name in DISTRIBUTIONS] * 2
        self.assertEqual(self.versions.call_args_list, expected)

    def test_missing_distribution_stops_before_reading_model(self):
        for name in DISTRIBUTIONS:
            with self.subTest(name=name):
                def version(distribution):
                    if distribution == name:
                        raise shadow.distribution_metadata.PackageNotFoundError(name)
                    return DISTRIBUTIONS[distribution]
                with mock.patch.object(shadow.distribution_metadata, "version", side_effect=version):
                    with self.assertRaisesRegex(shadow.ShadowPredictionError, name):
                        self._check()

    def test_invalid_distribution_versions_are_rejected(self):
        for value in (None, "", " ", True, 1, "2.5.2 ", "2.5.2.post1"):
            with self.subTest(value=value):
                with mock.patch.object(shadow.distribution_metadata, "version", return_value=value):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._check()

    def test_unreadable_distribution_metadata_is_wrapped(self):
        with mock.patch.object(shadow.distribution_metadata, "version", side_effect=OSError("denied")):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._check()

    def test_runtime_changes_during_verification_are_rejected(self):
        with mock.patch.object(shadow, "_installed_model_runtime_versions",
                               side_effect=[RUNTIME, dict(RUNTIME, joblib="9.0.0")]):
            with self.assertRaisesRegex(shadow.ShadowPredictionError, "change"):
                self._check()

    def test_metadata_changes_during_model_read_are_rejected(self):
        original_read = Path.read_bytes
        for relative in EXPECTED_PATHS[:3]:
            with self.subTest(relative=relative):
                path = self.project / relative
                before = path.read_bytes()
                def mutate(current):
                    content = original_read(current)
                    if current == self.project / ARTIFACT_PATH:
                        path.write_bytes(before + b" ")
                    return content
                try:
                    with mock.patch.object(Path, "read_bytes", mutate):
                        with self.assertRaises(shadow.ShadowPredictionError):
                            self._check()
                finally:
                    path.write_bytes(before)

    def test_model_change_before_second_read_is_rejected(self):
        original = shadow._installed_model_runtime_versions
        calls = 0
        def runtime():
            nonlocal calls
            calls += 1
            if calls == 2:
                (self.project / ARTIFACT_PATH).write_bytes(b"x" + self.artifact[1:])
            return original()
        with mock.patch.object(shadow, "_installed_model_runtime_versions", side_effect=runtime):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._check()

    def test_same_bytes_file_replacement_during_read_is_rejected(self):
        original_read = Path.read_bytes
        path = self.project / ARTIFACT_PATH
        def replace_during_read(current):
            content = original_read(current)
            if current == path:
                replacement = self.project / "replacement"
                replacement.write_bytes(content)
                replacement.replace(path)
            return content
        with mock.patch.object(Path, "read_bytes", replace_during_read):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._check()

    def test_read_permission_failure_is_wrapped(self):
        with mock.patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._check()

    def test_contradictory_manifest_identity_and_policy_are_rejected(self):
        original = deepcopy(self.manifest)
        cases = (
            ("artifact", "size_bytes", True), ("artifact", "size_bytes", 1590),
            ("artifact", "path", "models/other.joblib"),
            ("artifact", "serializer", "pickle"),
            ("artifact", "code_version", "f" * 40),
            ("model", "calibration_method", "isotonic"),
            ("model", "feature_columns", list(reversed(self.manifest["model"]["feature_columns"]))),
            ("loading_policy", "verify_sha256_before_deserialization", False),
            ("loading_policy", "accept_untrusted_artifact", True),
        )
        for section, key, value in cases:
            with self.subTest(section=section, key=key, value=value):
                self.manifest = deepcopy(original)
                self.manifest[section][key] = value
                self._pin_documents()
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._check()

    def test_runtime_schema_is_exact_and_has_no_version_ranges(self):
        for wrong in (None, {}, dict(RUNTIME, extra="1"), dict(RUNTIME, python=True),
                      dict(RUNTIME, python=" 3.12.1"), dict(RUNTIME, numpy=">=2.5")):
            with self.subTest(runtime=wrong):
                self.manifest["runtime"] = wrong
                self._pin_documents()
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._check()

    def test_model_protocol_feature_order_cannot_drift(self):
        self.model_protocol["features"].reverse()
        self._pin_documents()
        with self.assertRaises(shadow.ShadowPredictionError):
            self._check()

    def test_json_decoder_rejects_duplicates_nonfinite_values_and_wrong_roots(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}',
                    b'[]', b'null', b'bad', b'\xef\xbb\xbf{}', b'\xff'):
            with self.subTest(raw=raw):
                with self.assertRaises(shadow.ShadowPredictionError):
                    shadow._decode_pinned_model_json(raw, name="synthetic")

    def test_no_model_or_runtime_override_is_exposed(self):
        signature = inspect.signature(shadow.verify_frozen_model_prerequisites)
        self.assertEqual(list(signature.parameters), ["project_directory"])
        self.assertIs(signature.parameters["project_directory"].kind,
                      inspect.Parameter.KEYWORD_ONLY)
        with self.assertRaises(TypeError):
            shadow.verify_frozen_model_prerequisites(expected_sha256="0" * 64)
        with self.assertRaises(TypeError):
            shadow.verify_frozen_model_prerequisites(runtime_versions=RUNTIME)

    def test_preview_never_invokes_the_new_verifier(self):
        with mock.patch.object(shadow, "verify_frozen_model_prerequisites",
                               side_effect=AssertionError("Controle officiel dans apercu")):
            preview = shadow.preview_shadow_prediction("2026-09-03", project_directory=self.project)
        self.assertFalse(preview.model_artifact_read)
        self.assertFalse(preview.execution_ready)


class ShadowModelRegisteredContractTests(unittest.TestCase):
    def test_real_registered_metadata_keeps_exact_hashes_and_runtime(self):
        for relative, expected in (
            (SHADOW_PATH, shadow.EXPECTED_SHADOW_PROTOCOL_SHA256),
            (MANIFEST_PATH, shadow.EXPECTED_ARTIFACT_MANIFEST_SHA256),
            (MODEL_PROTOCOL_PATH, shadow.EXPECTED_MODEL_PROTOCOL_SHA256),
        ):
            self.assertEqual(_digest((PROJECT / relative).read_bytes()), expected)
        documents = [json.loads((PROJECT / name).read_bytes()) for name in EXPECTED_PATHS[:3]]
        self.assertEqual(shadow._validate_model_prerequisite_contracts(*documents), RUNTIME)


if __name__ == "__main__":
    unittest.main()
