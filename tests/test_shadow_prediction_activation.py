"""Tests fermes de l'activation append-only du mode fantome MLB v2."""

from __future__ import annotations

import ast
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import shadow_prediction as shadow


MANIFEST_COMMIT = "c" * 40
RUNTIME_COMMIT = "d" * 40
MANIFEST_SHA256 = "e" * 64
MINIMUM_DATE = "2026-09-06"
CREATED_AT = "2026-09-05T10:00:02Z"


def _activation_root_worker(
    project_text: str,
    start: object,
    results: object,
) -> None:
    """Processus isole utilise pour la vraie course mkdir."""
    try:
        start.wait(10)  # type: ignore[attr-defined]
        root = shadow._create_activation_root_exclusive(Path(project_text))
        results.put(("winner", str(root)))  # type: ignore[attr-defined]
    except shadow.ShadowActivationConsumedError:
        results.put(("consumed", None))  # type: ignore[attr-defined]
    except BaseException as error:
        results.put(("unexpected", repr(error)))  # type: ignore[attr-defined]


class ShadowActivationTests(unittest.TestCase):
    """L'activation doit etre distante, unique et impossible a reparer."""

    def _evidence(self) -> shadow.ShadowActivationReverificationEvidence:
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
        raw = {
            "evidence_schema_version": 1,
            "request_url": request_url,
            "request_method": "GET",
            "application_request_headers": dict(
                shadow.GITHUB_REQUEST_HEADERS
            ),
            "effective_url": request_url,
            "response_status_code": 200,
            "response_redirect_count": 0,
            "selected_response_headers": {
                "date": "Sat, 05 Sep 2026 10:00:00 GMT",
                "content-type": "application/json",
                "etag": None,
                "x-github-request-id": "activation-request",
            },
            "response_received_at_utc": "2026-09-05T10:00:01Z",
            "response_body_base64": base64.b64encode(body).decode("ascii"),
            "response_body_sha256": body_sha256,
        }
        canonical_json = shadow._canonical_json_file_bytes(raw)
        canonical_gzip = shadow._canonical_gzip_bytes(canonical_json)
        return shadow.ShadowActivationReverificationEvidence(
            activation_introduction_commit=MANIFEST_COMMIT,
            activation_remote_ref=shadow.GITHUB_REMOTE_REF,
            activation_remote_reverified_at_utc="2026-09-05T10:00:00Z",
            response_received_at_utc="2026-09-05T10:00:01Z",
            response_body_sha256=body_sha256,
            raw_evidence=raw,
            canonical_json_bytes=canonical_json,
            canonical_gzip_bytes=canonical_gzip,
            canonical_gzip_sha256=hashlib.sha256(
                canonical_gzip
            ).hexdigest(),
        )

    def _activation(
        self,
        evidence: shadow.ShadowActivationReverificationEvidence,
        *,
        created_at: str = CREATED_AT,
    ) -> dict[str, object]:
        request_url = shadow.GITHUB_COMPARE_URL_TEMPLATE.format(
            expected_commit=MANIFEST_COMMIT
        )
        return {
            "activation_schema_version": 1,
            "status": "ACTIVATED_BEFORE_FIRST_SHADOW_V2_BATCH",
            "shadow_protocol_path": (
                shadow.SHADOW_PROTOCOL_RELATIVE_PATH.as_posix()
            ),
            "shadow_protocol_sha256": (
                shadow.EXPECTED_SHADOW_PROTOCOL_SHA256
            ),
            "execution_manifest_path": (
                shadow.EXECUTION_MANIFEST_RELATIVE_PATH.as_posix()
            ),
            "execution_manifest_sha256": MANIFEST_SHA256,
            "execution_manifest_introduction_commit": MANIFEST_COMMIT,
            "execution_manifest_remote_ref": shadow.GITHUB_REMOTE_REF,
            "execution_manifest_remote_query_url": request_url,
            "execution_manifest_remote_effective_url": request_url,
            "execution_manifest_remote_response_status_code": 200,
            "execution_manifest_remote_response_redirect_count": 0,
            "execution_manifest_remote_http_date_utc": (
                evidence.activation_remote_reverified_at_utc
            ),
            "execution_manifest_remote_response_received_at_utc": (
                evidence.response_received_at_utc
            ),
            "execution_manifest_remote_response_body_sha256": (
                evidence.response_body_sha256
            ),
            "raw_remote_evidence_path": (
                shadow.ACTIVATION_REMOTE_EVIDENCE_RELATIVE_PATH.as_posix()
            ),
            "raw_remote_evidence_sha256": (
                evidence.canonical_gzip_sha256
            ),
            "minimum_target_official_date": MINIMUM_DATE,
            "created_at_utc": created_at,
            "claim_level": (
                "REMOTE_SERVER_ATTESTED_NOT_CRYPTOGRAPHICALLY_TIMESTAMPED"
            ),
        }

    def _publish(
        self,
        root: Path,
        *,
        created_at: str = CREATED_AT,
    ) -> shadow.ShadowActivationPublication:
        evidence = self._evidence()
        activation = self._activation(evidence, created_at=created_at)
        activation_bytes = shadow._canonical_json_file_bytes(activation)
        return shadow._publish_prepared_activation_once(
            project_directory=root,
            activation=activation,
            activation_bytes=activation_bytes,
            evidence=evidence,
            manifest_sha256=MANIFEST_SHA256,
            manifest_introduction_commit=MANIFEST_COMMIT,
            minimum_target_official_date=MINIMUM_DATE,
        )

    def _activation_preflight_patches(
        self,
        stack: ExitStack,
        root: Path,
        *,
        minimum_date: str = MINIMUM_DATE,
    ) -> shadow.ShadowActivationReverificationEvidence:
        manifest = {"manifest": 1}
        manifest_bytes = shadow._canonical_json_file_bytes(manifest)
        manifest_path = root.joinpath(
            *shadow.EXECUTION_MANIFEST_RELATIVE_PATH.parts
        )
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_bytes(manifest_bytes)
        evidence = self._evidence()
        stack.enter_context(
            mock.patch.object(
                shadow,
                "_read_frozen_protocol",
                return_value=(
                    {},
                    shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(shadow, "_validate_protocol_contract")
        )
        stack.enter_context(
            mock.patch.object(
                shadow,
                "_require_exact_clean_git_root",
                return_value=RUNTIME_COMMIT,
            )
        )
        stack.enter_context(
            mock.patch.object(
                shadow,
                "_git_immutable_introduction_blob",
                return_value=(MANIFEST_COMMIT, manifest_bytes),
            )
        )
        stack.enter_context(
            mock.patch.object(
                shadow,
                "_validate_execution_manifest",
                return_value=(
                    "a" * 40,
                    "1" * 64,
                    "2" * 64,
                    (),
                    minimum_date,
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(
                shadow,
                "_require_activation_paths_not_ignored",
            )
        )
        stack.enter_context(
            mock.patch.object(
                shadow,
                "fetch_activation_reverification_evidence",
                return_value=evidence,
            )
        )
        stack.enter_context(
            mock.patch.object(
                shadow,
                "_utc_now",
                return_value=datetime(
                    2026, 9, 5, 10, 0, 2, tzinfo=timezone.utc
                ),
            )
        )
        return evidence

    def test_public_activation_publishes_exact_two_canonical_files(self) -> None:
        """Le succes lie le manifeste, la preuve gzip et activation.json."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            evidence = self._activation_preflight_patches(stack, root)
            result = shadow.activate_shadow_protocol(
                project_directory=root
            )

            self.assertEqual(
                {path.name for path in result.activation_root.iterdir()},
                {"activation.json", "execution_manifest.remote.json.gz"},
            )
            activation_bytes = result.activation_path.read_bytes()
            self.assertEqual(
                activation_bytes,
                shadow._canonical_json_file_bytes(result.activation),
            )
            self.assertEqual(
                result.evidence_path.read_bytes(),
                evidence.canonical_gzip_bytes,
            )
            self.assertEqual(
                hashlib.sha256(activation_bytes).hexdigest(),
                result.activation_sha256,
            )
            self.assertEqual(
                result.activation["execution_manifest_introduction_commit"],
                MANIFEST_COMMIT,
            )
            self.assertEqual(
                frozenset(result.activation), shadow._ACTIVATION_KEYS
            )

    def test_activation_has_no_daily_model_mlb_or_sqlite_operation(self) -> None:
        """L'activation ne peut pas devenir silencieusement un lot officiel."""
        forbidden = (
            "reserve_shadow_prediction_slot",
            "capture_and_publish_source_snapshot",
            "run_observed_schedule_ingestion",
            "verify_frozen_model_prerequisites",
            "_load_frozen_shadow_model",
            "_predict_frozen_shadow_model_once",
            "execute_shadow_prediction",
        )
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            self._activation_preflight_patches(stack, root)
            for name in forbidden:
                stack.enter_context(
                    mock.patch.object(
                        shadow,
                        name,
                        side_effect=AssertionError(name),
                    )
                )
            stack.enter_context(
                mock.patch.object(
                    shadow.sqlite3,
                    "connect",
                    side_effect=AssertionError("sqlite3.connect"),
                )
            )
            shadow.activate_shadow_protocol(project_directory=root)

    def test_existing_or_partial_root_is_terminal_before_any_read(self) -> None:
        """Une racine precedente ne declenche ni verification ni reparation."""
        for filename in (None, "execution_manifest.remote.json.gz"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                activation_root = root.joinpath(
                    *shadow.SHADOW_ACTIVATION_ROOT_RELATIVE_PATH.parts
                )
                activation_root.mkdir(parents=True)
                if filename is not None:
                    (activation_root / filename).write_bytes(b"partial")
                with mock.patch.object(
                    shadow,
                    "_read_frozen_protocol",
                    side_effect=AssertionError("lecture interdite"),
                ), mock.patch.object(
                    shadow,
                    "fetch_activation_reverification_evidence",
                    side_effect=AssertionError("reseau interdit"),
                ), self.assertRaises(shadow.ShadowActivationConsumedError):
                    shadow.activate_shadow_protocol(project_directory=root)

    def test_remote_failure_creates_no_activation_tree(self) -> None:
        """Un refus GitHub intervient avant acquisition de la racine."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            self._activation_preflight_patches(stack, root)
            stack.enter_context(
                mock.patch.object(
                    shadow,
                    "fetch_activation_reverification_evidence",
                    side_effect=shadow.ShadowPredictionError("offline"),
                )
            )
            with self.assertRaisesRegex(shadow.ShadowPredictionError, "offline"):
                shadow.activate_shadow_protocol(project_directory=root)
            self.assertFalse((root / "shadow_activations").exists())

    def test_dirty_or_ignored_project_stops_before_network(self) -> None:
        """Git propre et chemins versionnables sont des preconditions fermees."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            self._activation_preflight_patches(stack, root)
            stack.enter_context(
                mock.patch.object(
                    shadow,
                    "_require_exact_clean_git_root",
                    side_effect=shadow.ShadowPredictionError("strictement propre"),
                )
            )
            fetch = stack.enter_context(
                mock.patch.object(
                    shadow,
                    "fetch_activation_reverification_evidence",
                )
            )
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError, "strictement propre"
            ):
                shadow.activate_shadow_protocol(project_directory=root)
            fetch.assert_not_called()

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            self._activation_preflight_patches(stack, root)
            stack.enter_context(
                mock.patch.object(
                    shadow,
                    "_require_activation_paths_not_ignored",
                    side_effect=shadow.ShadowPredictionError("gitignore"),
                )
            )
            fetch = stack.enter_context(
                mock.patch.object(
                    shadow,
                    "fetch_activation_reverification_evidence",
                )
            )
            with self.assertRaisesRegex(shadow.ShadowPredictionError, "gitignore"):
                shadow.activate_shadow_protocol(project_directory=root)
            fetch.assert_not_called()

    def test_gitignore_check_covers_root_and_both_exact_files(self) -> None:
        """La verification Git porte sur chaque chemin fige de l'activation."""
        calls: list[tuple[str, ...]] = []

        def git(
            _project: Path,
            arguments: object,
            **_kwargs: object,
        ) -> tuple[int, bytes]:
            calls.append(tuple(arguments))  # type: ignore[arg-type]
            return 1, b""

        with mock.patch.object(shadow, "_run_preflight_git", side_effect=git):
            shadow._require_activation_paths_not_ignored(Path("/project"))
        self.assertEqual(
            calls,
            [
                (
                    "check-ignore",
                    "-q",
                    "--",
                    shadow.SHADOW_ACTIVATION_ROOT_RELATIVE_PATH.as_posix(),
                ),
                (
                    "check-ignore",
                    "-q",
                    "--",
                    shadow.ACTIVATION_REMOTE_EVIDENCE_RELATIVE_PATH.as_posix(),
                ),
                (
                    "check-ignore",
                    "-q",
                    "--",
                    shadow.ACTIVATION_RELATIVE_PATH.as_posix(),
                ),
            ],
        )

        with mock.patch.object(
            shadow,
            "_run_preflight_git",
            return_value=(0, b""),
        ), self.assertRaisesRegex(shadow.ShadowPredictionError, "gitignore"):
            shadow._require_activation_paths_not_ignored(Path("/project"))

    def test_temporal_mismatch_consumes_no_root(self) -> None:
        """La date minimale doit couvrir la date HTTP avant toute ecriture."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            self._activation_preflight_patches(
                stack,
                root,
                minimum_date="2026-09-04",
            )
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError, "date minimale"
            ):
                shadow.activate_shadow_protocol(project_directory=root)
            self.assertFalse((root / "shadow_activations").exists())

        evidence = self._evidence()
        evidence = shadow.ShadowActivationReverificationEvidence(
            activation_introduction_commit=(
                evidence.activation_introduction_commit
            ),
            activation_remote_ref=evidence.activation_remote_ref,
            activation_remote_reverified_at_utc=(
                evidence.activation_remote_reverified_at_utc
            ),
            response_received_at_utc="2026-09-05T10:00:03Z",
            response_body_sha256=evidence.response_body_sha256,
            raw_evidence={
                **evidence.raw_evidence,
                "response_received_at_utc": "2026-09-05T10:00:03Z",
            },
            canonical_json_bytes=b"placeholder",
            canonical_gzip_bytes=b"placeholder",
            canonical_gzip_sha256="0" * 64,
        )
        canonical_json = shadow._canonical_json_file_bytes(
            evidence.raw_evidence
        )
        canonical_gzip = shadow._canonical_gzip_bytes(canonical_json)
        evidence = shadow.ShadowActivationReverificationEvidence(
            activation_introduction_commit=(
                evidence.activation_introduction_commit
            ),
            activation_remote_ref=evidence.activation_remote_ref,
            activation_remote_reverified_at_utc=(
                evidence.activation_remote_reverified_at_utc
            ),
            response_received_at_utc=evidence.response_received_at_utc,
            response_body_sha256=evidence.response_body_sha256,
            raw_evidence=evidence.raw_evidence,
            canonical_json_bytes=canonical_json,
            canonical_gzip_bytes=canonical_gzip,
            canonical_gzip_sha256=hashlib.sha256(canonical_gzip).hexdigest(),
        )
        activation = self._activation(evidence)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError, "ordre temporel"
            ):
                shadow._publish_prepared_activation_once(
                    project_directory=root,
                    activation=activation,
                    activation_bytes=shadow._canonical_json_file_bytes(
                        activation
                    ),
                    evidence=evidence,
                    manifest_sha256=MANIFEST_SHA256,
                    manifest_introduction_commit=MANIFEST_COMMIT,
                    minimum_target_official_date=MINIMUM_DATE,
                )
            self.assertFalse((root / "shadow_activations").exists())

    def test_concurrent_publishers_have_exactly_one_immutable_winner(self) -> None:
        """Un seul thread acquiert la racine et les autres ne la modifient pas."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def attempt(index: int) -> tuple[str, str | None]:
                try:
                    result = self._publish(
                        root,
                        created_at=f"2026-09-05T10:00:{index + 2:02d}Z",
                    )
                    return "winner", result.activation_sha256
                except shadow.ShadowActivationConsumedError:
                    return "consumed", None

            with ThreadPoolExecutor(max_workers=12) as executor:
                outcomes = list(executor.map(attempt, range(12)))

            winners = [value for state, value in outcomes if state == "winner"]
            self.assertEqual(len(winners), 1)
            self.assertEqual(
                sum(state == "consumed" for state, _ in outcomes), 11
            )
            activation_path = root.joinpath(
                *shadow.ACTIVATION_RELATIVE_PATH.parts
            )
            self.assertEqual(
                hashlib.sha256(activation_path.read_bytes()).hexdigest(),
                winners[0],
            )

    def test_concurrent_processes_have_exactly_one_root_owner(self) -> None:
        """La frontiere mkdir reste atomique entre processus independants."""
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as temporary:
            start = context.Event()
            results = context.Queue()
            processes = [
                context.Process(
                    target=_activation_root_worker,
                    args=(temporary, start, results),
                )
                for _ in range(8)
            ]
            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(20)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)
            outcomes = [results.get(timeout=5) for _ in processes]

        self.assertEqual(sum(state == "winner" for state, _ in outcomes), 1)
        self.assertEqual(sum(state == "consumed" for state, _ in outcomes), 7)
        self.assertNotIn("unexpected", {state for state, _ in outcomes})

    def test_failure_after_first_file_is_permanent_and_not_repaired(self) -> None:
        """Une interruption partielle laisse la racine consommee pour toujours."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = self._evidence()
            activation = self._activation(evidence)
            activation_bytes = shadow._canonical_json_file_bytes(activation)
            real_publish = shadow._publish_exclusive_verified
            calls = 0

            def fail_second(destination: Path, content: bytes) -> str:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise shadow.ShadowPredictionError("interruption controlee")
                return real_publish(destination, content)

            with mock.patch.object(
                shadow,
                "_publish_exclusive_verified",
                side_effect=fail_second,
            ), self.assertRaisesRegex(
                shadow.ShadowPredictionError, "interruption controlee"
            ):
                shadow._publish_prepared_activation_once(
                    project_directory=root,
                    activation=activation,
                    activation_bytes=activation_bytes,
                    evidence=evidence,
                    manifest_sha256=MANIFEST_SHA256,
                    manifest_introduction_commit=MANIFEST_COMMIT,
                    minimum_target_official_date=MINIMUM_DATE,
                )

            evidence_path = root.joinpath(
                *shadow.ACTIVATION_REMOTE_EVIDENCE_RELATIVE_PATH.parts
            )
            self.assertEqual(
                evidence_path.read_bytes(), evidence.canonical_gzip_bytes
            )
            with self.assertRaises(shadow.ShadowActivationConsumedError):
                self._publish(root)
            self.assertEqual(
                evidence_path.read_bytes(), evidence.canonical_gzip_bytes
            )
            self.assertFalse(
                root.joinpath(*shadow.ACTIVATION_RELATIVE_PATH.parts).exists()
            )

    def test_public_signature_and_ast_expose_no_runtime_override(self) -> None:
        """L'appelant ne choisit ni preuves, ni horloge, ni modele, ni chemin."""
        signature = inspect.signature(shadow.activate_shadow_protocol)
        self.assertEqual(tuple(signature.parameters), ("project_directory",))
        tree = ast.parse(inspect.getsource(shadow.activate_shadow_protocol))
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("fetch_activation_reverification_evidence", called_names)
        self.assertTrue(
            called_names.isdisjoint(
                {
                    "reserve_shadow_prediction_slot",
                    "capture_and_publish_source_snapshot",
                    "run_observed_schedule_ingestion",
                    "verify_frozen_model_prerequisites",
                    "_load_frozen_shadow_model",
                    "_predict_frozen_shadow_model_once",
                }
            )
        )

    def test_cli_activation_is_explicit_date_free_and_canonical(self) -> None:
        """Le drapeau dedie appelle seulement l'activation publique."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            activation = {"status": "ACTIVATED"}
            activation_bytes = shadow._canonical_json_file_bytes(activation)
            path = root / "activation.json"
            path.write_bytes(activation_bytes)
            result = shadow.ShadowActivationPublication(
                activation_root=root,
                evidence_path=root / "evidence.gz",
                evidence_relative_path="evidence.gz",
                evidence_sha256="1" * 64,
                evidence_size_bytes=1,
                activation_path=path,
                activation_relative_path="activation.json",
                activation_sha256=hashlib.sha256(activation_bytes).hexdigest(),
                activation_size_bytes=len(activation_bytes),
                execution_manifest_sha256=MANIFEST_SHA256,
                execution_manifest_introduction_commit=MANIFEST_COMMIT,
                execution_manifest_remote_ref=shadow.GITHUB_REMOTE_REF,
                execution_manifest_remote_http_date_utc=(
                    "2026-09-05T10:00:00Z"
                ),
                execution_manifest_remote_response_received_at_utc=(
                    "2026-09-05T10:00:01Z"
                ),
                minimum_target_official_date=MINIMUM_DATE,
                created_at_utc=CREATED_AT,
                activation=activation,
            )
            stdout = io.StringIO()
            with mock.patch.object(
                shadow,
                "activate_shadow_protocol",
                return_value=result,
            ) as activate, mock.patch.object(
                shadow,
                "execute_shadow_prediction",
                side_effect=AssertionError("execution interdite"),
            ), mock.patch.object(
                shadow,
                "preview_shadow_prediction",
                side_effect=AssertionError("apercu interdit"),
            ), redirect_stdout(stdout):
                self.assertEqual(shadow.main(["--activate-shadow"]), 0)

        activate.assert_called_once_with()
        self.assertEqual(stdout.getvalue().encode("utf-8"), activation_bytes)

    def test_cli_modes_and_target_are_closed(self) -> None:
        """Activation, execution et apercu ne peuvent pas etre confondus."""
        cases = (
            [],
            ["--execute-shadow"],
            ["--activate-shadow", "--target-official-date", "2026-09-06"],
            [
                "--activate-shadow",
                "--execute-shadow",
                "--target-official-date",
                "2026-09-06",
            ],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit):
                    shadow.main(arguments)
                self.assertTrue(stderr.getvalue().startswith("usage:"))


if __name__ == "__main__":
    unittest.main()
