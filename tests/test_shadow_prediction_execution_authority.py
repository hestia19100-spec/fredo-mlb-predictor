"""Tests fermes du preflight immuable shadow MLB v2."""

from __future__ import annotations

import ast
import base64
from contextlib import ExitStack
from datetime import date
import hashlib
import inspect
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import src.shadow_prediction as shadow


COMMIT = "a" * 40
SERVICE_COMMIT = "b" * 40
MANIFEST_COMMIT = "c" * 40
ACTIVATION_COMMIT = "d" * 40
RUNTIME = {
    "joblib": "1.5.2",
    "numpy": "2.3.3",
    "pandas": "2.3.3",
    "python": "3.12.1",
    "scikit_learn": "1.7.2",
    "scipy": "1.16.2",
}


class ShadowExecutionAuthorityTests(unittest.TestCase):
    """Le preflight ne doit laisser aucune autorisation implicite."""

    def _valid_manifest(self) -> dict[str, object]:
        project = shadow.PROJECT_DIRECTORY
        closure = shadow._local_python_runtime_closure(project)
        hashes = {
            relative: hashlib.sha256(
                project.joinpath(*Path(relative).parts).read_bytes()
            ).hexdigest()
            for relative in closure
        }
        requirements = project / "requirements.txt"
        return {
            "execution_manifest_schema_version": 1,
            "status": "FROZEN_BEFORE_SHADOW_V2_ACTIVATION",
            "shadow_service_code_commit": SERVICE_COMMIT,
            "shadow_service_module_path": "src/shadow_prediction.py",
            "shadow_service_module_sha256": hashes[
                "src/shadow_prediction.py"
            ],
            "transitive_runtime_file_sha256_map": dict(sorted(hashes.items())),
            "shadow_protocol_path": (
                "shadow_protocols/"
                "logistic_team_form_v1_platt_shadow_v2.json"
            ),
            "shadow_protocol_sha256": shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
            "model_artifact_path": (
                "models/logistic_team_form_v1_platt.joblib"
            ),
            "model_artifact_sha256": shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
            "model_artifact_size_bytes": 1589,
            "requirements_path": "requirements.txt",
            "requirements_sha256": hashlib.sha256(
                requirements.read_bytes()
            ).hexdigest(),
            "test_suite_result": {
                "command": "python -m unittest discover -s tests -v",
                "status": "OK",
                "tests_run": 634,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "completed_at_utc": "2026-09-01T10:00:00Z",
                "tested_code_commit": SERVICE_COMMIT,
            },
            "runtime_versions": dict(sorted(RUNTIME.items())),
            "minimum_target_official_date": "2026-09-02",
            "created_at_utc": "2026-09-01T10:01:00Z",
        }

    def _raw_remote_evidence(
        self,
    ) -> tuple[dict[str, object], bytes, bytes, str]:
        body = shadow._canonical_json_bytes(
            {
                "base_commit": {"sha": MANIFEST_COMMIT},
                "merge_base_commit": {"sha": MANIFEST_COMMIT},
                "status": "identical",
            }
        )
        body_sha256 = hashlib.sha256(body).hexdigest()
        request_url = shadow.GITHUB_COMPARE_URL_TEMPLATE.format(
            expected_commit=MANIFEST_COMMIT
        )
        raw: dict[str, object] = {
            "evidence_schema_version": 1,
            "request_url": request_url,
            "request_method": "GET",
            "application_request_headers": dict(shadow.GITHUB_REQUEST_HEADERS),
            "effective_url": request_url,
            "response_status_code": 200,
            "response_redirect_count": 0,
            "selected_response_headers": {
                "date": "Tue, 01 Sep 2026 18:47:51 GMT",
                "content-type": "application/json",
                "etag": None,
                "x-github-request-id": "request-1",
            },
            "response_received_at_utc": "2026-09-01T18:47:52Z",
            "response_body_base64": base64.b64encode(body).decode("ascii"),
            "response_body_sha256": body_sha256,
        }
        raw_json = shadow._canonical_json_file_bytes(raw)
        raw_gzip = shadow._canonical_gzip_bytes(raw_json)
        return raw, raw_json, raw_gzip, body_sha256

    def _valid_activation(
        self,
        raw_gzip: bytes,
        body_sha256: str,
    ) -> dict[str, object]:
        request_url = shadow.GITHUB_COMPARE_URL_TEMPLATE.format(
            expected_commit=MANIFEST_COMMIT
        )
        return {
            "activation_schema_version": 1,
            "status": "ACTIVATED_BEFORE_FIRST_SHADOW_V2_BATCH",
            "shadow_protocol_path": shadow.SHADOW_PROTOCOL_RELATIVE_PATH.as_posix(),
            "shadow_protocol_sha256": shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
            "execution_manifest_path": (
                shadow.EXECUTION_MANIFEST_RELATIVE_PATH.as_posix()
            ),
            "execution_manifest_sha256": "e" * 64,
            "execution_manifest_introduction_commit": MANIFEST_COMMIT,
            "execution_manifest_remote_ref": "refs/heads/main",
            "execution_manifest_remote_query_url": request_url,
            "execution_manifest_remote_effective_url": request_url,
            "execution_manifest_remote_response_status_code": 200,
            "execution_manifest_remote_response_redirect_count": 0,
            "execution_manifest_remote_http_date_utc": (
                "2026-09-01T18:47:51Z"
            ),
            "execution_manifest_remote_response_received_at_utc": (
                "2026-09-01T18:47:52Z"
            ),
            "execution_manifest_remote_response_body_sha256": body_sha256,
            "raw_remote_evidence_path": (
                shadow.ACTIVATION_REMOTE_EVIDENCE_RELATIVE_PATH.as_posix()
            ),
            "raw_remote_evidence_sha256": hashlib.sha256(raw_gzip).hexdigest(),
            "minimum_target_official_date": "2026-09-02",
            "created_at_utc": "2026-09-01T18:47:53Z",
            "claim_level": (
                "REMOTE_SERVER_ATTESTED_NOT_CRYPTOGRAPHICALLY_TIMESTAMPED"
            ),
        }

    def _write_activation_project(
        self,
        root: Path,
        activation: dict[str, object],
        raw_gzip: bytes,
    ) -> None:
        activation_path = root.joinpath(*shadow.ACTIVATION_RELATIVE_PATH.parts)
        raw_path = root.joinpath(
            *shadow.ACTIVATION_REMOTE_EVIDENCE_RELATIVE_PATH.parts
        )
        activation_path.parent.mkdir(parents=True)
        activation_path.write_bytes(shadow._canonical_json_file_bytes(activation))
        raw_path.write_bytes(raw_gzip)

    def test_runtime_closure_is_exact_recursive_and_has_package_init(self) -> None:
        """Le manifeste futur doit couvrir chaque module local reel."""
        self.assertEqual(
            shadow._local_python_runtime_closure(shadow.PROJECT_DIRECTORY),
            (
                "src/__init__.py",
                "src/baseline_model.py",
                "src/calibrated_model.py",
                "src/database.py",
                "src/game_repository.py",
                "src/ingestion_repository.py",
                "src/ingestion_service.py",
                "src/mlb_api.py",
                "src/raw_archive.py",
                "src/retry_policy.py",
                "src/shadow_prediction.py",
                "src/training_dataset.py",
            ),
        )

    def test_canonical_json_reader_rejects_ambiguity_and_format_drift(self) -> None:
        """Aucune derniere cle gagnante ou mise en forme libre n'est admise."""
        self.assertEqual(
            shadow._read_canonical_json_bytes(
                b'{"a":1}\n', description="test"
            ),
            {"a": 1},
        )
        for invalid in (b'{"a":1,"a":2}\n', b'{"a": 1}\n', b'{"a":1}'):
            with self.subTest(invalid=invalid), self.assertRaises(
                shadow.ShadowPredictionError
            ):
                shadow._read_canonical_json_bytes(invalid, description="test")

    def test_git_runner_is_shell_free_bounded_and_has_no_stdin(self) -> None:
        """Les identifiants Git ne peuvent jamais devenir une commande shell."""
        completed = SimpleNamespace(returncode=0, stdout=b"ok\n", stderr=b"")
        with patch.object(subprocess, "run", return_value=completed) as run:
            result = shadow._run_preflight_git(Path("/project"), ("status",))
        self.assertEqual(result, (0, b"ok\n"))
        _, kwargs = run.call_args
        self.assertIs(kwargs["shell"], False)
        self.assertIs(kwargs["check"], False)
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["timeout"], 30)

    def test_git_introduction_rejects_every_later_touch(self) -> None:
        """Un manifeste ou une activation ne peut jamais etre mis a jour."""
        outputs = iter(
            (
                (0, (MANIFEST_COMMIT + "\n").encode()),
                (0, (COMMIT + "\n" + MANIFEST_COMMIT + "\n").encode()),
            )
        )
        with patch.object(shadow, "_run_preflight_git", side_effect=outputs):
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError, "modifie apres"
            ):
                shadow._git_immutable_introduction_blob(
                    Path("/project"),
                    shadow.EXECUTION_MANIFEST_RELATIVE_PATH,
                    b"manifest",
                )

    def test_git_introduction_is_resolved_from_a_real_repository(self) -> None:
        """Le premier ajout Git reel est retrouve sans branche ni shell caches."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(
                ["git", "init", "-q", root],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for key, value in (
                ("user.email", "shadow-tests@example.invalid"),
                ("user.name", "Shadow Tests"),
            ):
                subprocess.run(
                    ["git", "-C", root, "config", key, value],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            relative = shadow.PurePosixPath("proof/manifest.json")
            path = root.joinpath(*relative.parts)
            path.parent.mkdir()
            content = b'{"status":"frozen"}\n'
            path.write_bytes(content)
            subprocess.run(
                ["git", "-C", root, "add", relative.as_posix()],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "-C", root, "commit", "-q", "-m", "introduce"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            commit, blob = shadow._git_immutable_introduction_blob(
                root, relative, content
            )
            expected = subprocess.run(
                ["git", "-C", root, "rev-parse", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout.decode().strip()
            self.assertEqual(commit, expected)
            self.assertEqual(blob, content)

            path.write_bytes(b'{"status":"changed"}\n')
            subprocess.run(
                ["git", "-C", root, "add", relative.as_posix()],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "-C", root, "commit", "-q", "-m", "change"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError, "modifie apres"
            ):
                shadow._git_immutable_introduction_blob(
                    root, relative, path.read_bytes()
                )

    def test_git_introduction_rejects_local_blob_substitution(self) -> None:
        """Le bon historique ne permet jamais de substituer le fichier local."""
        outputs = iter(
            (
                (0, (MANIFEST_COMMIT + "\n").encode()),
                (0, (MANIFEST_COMMIT + "\n").encode()),
                (0, b"original"),
            )
        )
        with patch.object(shadow, "_run_preflight_git", side_effect=outputs):
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError, "differe de son blob"
            ):
                shadow._git_immutable_introduction_blob(
                    Path("/project"),
                    shadow.EXECUTION_MANIFEST_RELATIVE_PATH,
                    b"substitute",
                )

    def test_valid_manifest_binds_closure_versions_tests_and_git(self) -> None:
        """Tous les octets runtime proviennent du commit service declare."""
        manifest = self._valid_manifest()

        def blob(_project: Path, _commit: str, relative: str) -> bytes:
            return shadow.PROJECT_DIRECTORY.joinpath(
                *Path(relative).parts
            ).read_bytes()

        with (
            patch.object(shadow, "_git_blob_at_commit", side_effect=blob),
            patch.object(shadow, "_require_git_ancestor") as ancestor,
            patch.object(
                shadow,
                "_installed_model_runtime_versions",
                return_value=dict(RUNTIME),
            ),
        ):
            result = shadow._validate_execution_manifest(
                manifest,
                project_directory=shadow.PROJECT_DIRECTORY,
                protocol_sha256=shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
                manifest_introduction_commit=MANIFEST_COMMIT,
                runtime_code_commit=COMMIT,
            )
        self.assertEqual(result[0], SERVICE_COMMIT)
        self.assertEqual(result[1], manifest["shadow_service_module_sha256"])
        self.assertEqual(result[3], tuple(sorted(manifest["runtime_versions"].items())))
        self.assertEqual(result[4], "2026-09-02")
        self.assertEqual(ancestor.call_count, 3)

    def test_manifest_rejects_schema_closure_runtime_and_test_drift(self) -> None:
        """Une seule valeur non figee suffit a fermer le preflight."""
        cases: list[tuple[str, object]] = [
            ("extra", 1),
            ("status", "OTHER"),
            ("model_artifact_size_bytes", 1590),
        ]
        for field_name, value in cases:
            manifest = self._valid_manifest()
            manifest[field_name] = value
            with self.subTest(field=field_name), self.assertRaises(
                shadow.ShadowPredictionError
            ):
                shadow._validate_execution_manifest(
                    manifest,
                    project_directory=shadow.PROJECT_DIRECTORY,
                    protocol_sha256=shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
                    manifest_introduction_commit=MANIFEST_COMMIT,
                    runtime_code_commit=COMMIT,
                )

        for mutation in ("missing", "extra", "wrong_hash"):
            manifest = self._valid_manifest()
            hashes = manifest["transitive_runtime_file_sha256_map"]
            assert isinstance(hashes, dict)
            if mutation == "missing":
                hashes.pop("src/database.py")
            elif mutation == "extra":
                hashes["src/unrelated.py"] = "f" * 64
                manifest["transitive_runtime_file_sha256_map"] = dict(
                    sorted(hashes.items())
                )
            else:
                hashes["src/database.py"] = "f" * 64
            with self.subTest(mutation=mutation), self.assertRaises(
                shadow.ShadowPredictionError
            ), patch.object(
                shadow,
                "_git_blob_at_commit",
                return_value=b"unreachable-or-different",
            ), patch.object(
                shadow,
                "_installed_model_runtime_versions",
                return_value=dict(RUNTIME),
            ):
                shadow._validate_execution_manifest(
                    manifest,
                    project_directory=shadow.PROJECT_DIRECTORY,
                    protocol_sha256=shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
                    manifest_introduction_commit=MANIFEST_COMMIT,
                    runtime_code_commit=COMMIT,
                )

    def test_activation_binds_canonical_gzip_and_introduction_blobs(self) -> None:
        """Activation JSON et preuve gzip doivent raconter le meme GET GitHub."""
        _raw, _raw_json, raw_gzip, body_sha256 = self._raw_remote_evidence()
        activation = self._valid_activation(raw_gzip, body_sha256)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_activation_project(root, activation, raw_gzip)

            def immutable(
                _project: Path,
                relative: object,
                local: bytes,
                **_kwargs: object,
            ) -> tuple[str, bytes]:
                self.assertIn(
                    relative,
                    {
                        shadow.ACTIVATION_RELATIVE_PATH,
                        shadow.ACTIVATION_REMOTE_EVIDENCE_RELATIVE_PATH,
                    },
                )
                return ACTIVATION_COMMIT, local

            with (
                patch.object(
                    shadow,
                    "_git_immutable_introduction_blob",
                    side_effect=immutable,
                ),
                patch.object(shadow, "_require_git_ancestor") as ancestor,
            ):
                result = shadow._read_and_validate_activation(
                    project_directory=root,
                    protocol_sha256=shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
                    manifest_sha256="e" * 64,
                    manifest_introduction_commit=MANIFEST_COMMIT,
                    manifest_minimum_date="2026-09-02",
                    runtime_code_commit=COMMIT,
                )
        self.assertEqual(result[1:], (
            ACTIVATION_COMMIT,
            "2026-09-01T18:47:51Z",
            "2026-09-02",
        ))
        self.assertEqual(ancestor.call_count, 2)

    def test_activation_rejects_corruption_binding_and_date_drift(self) -> None:
        """Une preuve alteree ou une date trop precoce ne peut etre reparee."""
        _raw, _raw_json, raw_gzip, body_sha256 = self._raw_remote_evidence()
        mutations = (
            ("status", "OTHER"),
            ("execution_manifest_remote_response_status_code", 201),
            ("minimum_target_official_date", "2026-08-31"),
            ("execution_manifest_remote_response_body_sha256", "f" * 64),
        )
        for field_name, value in mutations:
            activation = self._valid_activation(raw_gzip, body_sha256)
            activation[field_name] = value
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._write_activation_project(root, activation, raw_gzip)
                with self.subTest(field=field_name), patch.object(
                    shadow,
                    "_git_immutable_introduction_blob",
                    return_value=(ACTIVATION_COMMIT, b"unused"),
                ), self.assertRaises(shadow.ShadowPredictionError):
                    shadow._read_and_validate_activation(
                        project_directory=root,
                        protocol_sha256=shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
                        manifest_sha256="e" * 64,
                        manifest_introduction_commit=MANIFEST_COMMIT,
                        manifest_minimum_date="2026-09-02",
                        runtime_code_commit=COMMIT,
                    )

        activation = self._valid_activation(raw_gzip, body_sha256)
        corrupted = raw_gzip[:-1] + bytes([raw_gzip[-1] ^ 1])
        activation["raw_remote_evidence_sha256"] = hashlib.sha256(
            corrupted
        ).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_activation_project(root, activation, corrupted)
            with patch.object(
                shadow,
                "_git_immutable_introduction_blob",
                return_value=(ACTIVATION_COMMIT, b"unused"),
            ), self.assertRaises(shadow.ShadowPredictionError):
                shadow._read_and_validate_activation(
                    project_directory=root,
                    protocol_sha256=shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
                    manifest_sha256="e" * 64,
                    manifest_introduction_commit=MANIFEST_COMMIT,
                    manifest_minimum_date="2026-09-02",
                    runtime_code_commit=COMMIT,
                )

    def test_roots_must_exist_be_tracked_and_not_ignored(self) -> None:
        """Resultats, activations et certifications restent versionnables."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("shadow_results/example")
            (root / relative).mkdir(parents=True)
            with patch.object(
                shadow,
                "_run_preflight_git",
                side_effect=((1, b""), (0, b"shadow_results/example/.gitkeep\n")),
            ):
                shadow._require_tracked_nonignored_root(
                    root, shadow.PurePosixPath(relative.as_posix())
                )
            with patch.object(
                shadow,
                "_run_preflight_git",
                return_value=(0, b"ignored\n"),
            ), self.assertRaisesRegex(
                shadow.ShadowPredictionError, "gitignore"
            ):
                shadow._require_tracked_nonignored_root(
                    root, shadow.PurePosixPath(relative.as_posix())
                )
            with patch.object(
                shadow,
                "_run_preflight_git",
                side_effect=((1, b""), (0, b"")),
            ), self.assertRaisesRegex(
                shadow.ShadowPredictionError, "non versionnee"
            ):
                shadow._require_tracked_nonignored_root(
                    root, shadow.PurePosixPath(relative.as_posix())
                )
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError, "Racine versionnee absente"
            ):
                shadow._require_tracked_nonignored_root(
                    root, shadow.PurePosixPath("missing/root")
                )

    def _authority_patches(
        self,
        stack: ExitStack,
        root: Path,
        *,
        status: bytes = b"",
    ) -> list[str]:
        events: list[str] = []
        manifest = self._valid_manifest()
        manifest_bytes = shadow._canonical_json_file_bytes(manifest)
        stack.enter_context(
            patch.object(
                shadow,
                "_read_frozen_protocol",
                return_value=({}, shadow.EXPECTED_SHADOW_PROTOCOL_SHA256),
            )
        )
        stack.enter_context(patch.object(shadow, "_validate_protocol_contract"))
        stack.enter_context(
            patch.object(
                shadow,
                "_read_regular_project_file",
                return_value=manifest_bytes,
            )
        )
        stack.enter_context(
            patch.object(
                shadow,
                "_read_canonical_json_bytes",
                return_value=manifest,
            )
        )

        def git(_project: Path, args: object, **_kwargs: object) -> tuple[int, bytes]:
            command = tuple(args)
            if command == ("rev-parse", "--show-toplevel"):
                return 0, (str(root.resolve()) + "\n").encode()
            if command == ("rev-parse", "HEAD"):
                return 0, (COMMIT + "\n").encode()
            if command == ("status", "--porcelain=v1", "--untracked-files=all"):
                return 0, status
            raise AssertionError(command)

        stack.enter_context(patch.object(shadow, "_run_preflight_git", side_effect=git))
        stack.enter_context(
            patch.object(
                shadow,
                "_git_immutable_introduction_blob",
                return_value=(MANIFEST_COMMIT, manifest_bytes),
            )
        )

        def validate_manifest(*_args: object, **_kwargs: object) -> tuple[object, ...]:
            events.append("manifest")
            return (
                SERVICE_COMMIT,
                "1" * 64,
                "2" * 64,
                tuple(sorted(RUNTIME.items())),
                "2026-09-02",
            )

        stack.enter_context(
            patch.object(
                shadow,
                "_validate_execution_manifest",
                side_effect=validate_manifest,
            )
        )

        def activation(**_kwargs: object) -> tuple[str, str, str, str]:
            events.append("activation")
            return (
                "3" * 64,
                ACTIVATION_COMMIT,
                "2026-09-01T18:47:51Z",
                "2026-09-02",
            )

        stack.enter_context(
            patch.object(
                shadow,
                "_read_and_validate_activation",
                side_effect=activation,
            )
        )

        def root_check(*_args: object, **_kwargs: object) -> None:
            events.append("root")

        stack.enter_context(
            patch.object(
                shadow,
                "_require_tracked_nonignored_root",
                side_effect=root_check,
            )
        )
        stack.enter_context(
            patch.object(
                shadow,
                "inspect_shadow_prediction_slot",
                side_effect=AssertionError("authority must not inspect slot"),
            )
        )
        return events

    def test_authority_returns_closed_proof_without_external_inputs(self) -> None:
        """Le preflight lie les preuves sans toucher slot, MLB, SQLite ou modele."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            events = self._authority_patches(stack, root)
            for forbidden in (
                "fetch_activation_reverification_evidence",
                "reserve_shadow_prediction_slot",
                "capture_and_publish_source_snapshot",
                "verify_frozen_model_prerequisites",
                "_load_frozen_shadow_model",
                "_predict_frozen_shadow_model_once",
            ):
                stack.enter_context(
                    patch.object(
                        shadow,
                        forbidden,
                        side_effect=AssertionError(forbidden),
                    )
                )
            authority = shadow.verify_shadow_execution_authority(
                date(2026, 9, 2), project_directory=root
            )
        self.assertEqual(events, ["manifest", "activation", "root", "root", "root"])
        self.assertEqual(authority.runtime_code_commit, COMMIT)
        self.assertEqual(authority.execution_manifest_introduction_commit, MANIFEST_COMMIT)
        self.assertEqual(authority.activation_introduction_commit, ACTIVATION_COMMIT)
        for flag in (
            "output_slot_inspected",
            "output_slot_reserved",
            "network_request_performed",
            "sqlite_read",
            "model_artifact_read",
            "model_deserialized",
            "predictions_computed",
        ):
            self.assertIs(getattr(authority, flag), False)

    def test_dirty_repository_stops_before_immutable_authority(self) -> None:
        """Une modification locale bloque avant activation ou modele."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            events = self._authority_patches(stack, root, status=b" M source.py\n")
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError, "strictement propre"
            ):
                shadow.verify_shadow_execution_authority(
                    "2026-09-02", project_directory=root
                )
        self.assertEqual(events, [])

    def test_target_bounds_are_checked_without_runtime_override(self) -> None:
        """Saison et date minimale viennent uniquement des preuves figees."""
        signature = inspect.signature(shadow.verify_shadow_execution_authority)
        self.assertEqual(
            tuple(signature.parameters),
            ("target_official_date", "project_directory"),
        )
        with patch.object(
            shadow,
            "_preflight_project_directory",
            side_effect=AssertionError("must stop before project"),
        ):
            with self.assertRaises(shadow.ShadowPredictionError):
                shadow.verify_shadow_execution_authority("2025-09-02")

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            self._authority_patches(stack, root)
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError, "date minimale"
            ):
                shadow.verify_shadow_execution_authority(
                    "2026-09-01", project_directory=root
                )

    def test_authority_ast_has_no_network_sqlite_model_or_output_call(self) -> None:
        """La primitive d'autorite reste un preflight local sans effet."""
        tree = ast.parse(inspect.getsource(shadow.verify_shadow_execution_authority))
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertTrue(
            {
                "_read_frozen_protocol",
                "_run_preflight_git",
                "_validate_execution_manifest",
                "_read_and_validate_activation",
            }.issubset(called_names)
        )
        self.assertTrue(
            called_names.isdisjoint(
                {
                    "inspect_shadow_prediction_slot",
                    "fetch_activation_reverification_evidence",
                    "reserve_shadow_prediction_slot",
                    "capture_and_publish_source_snapshot",
                    "verify_frozen_model_prerequisites",
                    "_load_frozen_shadow_model",
                    "_predict_frozen_shadow_model_once",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
