"""Tests fermes de la certification prospective distante shadow v2."""

from __future__ import annotations

import ast
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from datetime import datetime, timezone
from email.utils import format_datetime
import gzip
import hashlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import shadow_certification as certification
from src import shadow_prediction as shadow


TARGET_DATE = "2026-09-10"
RESULTS_COMMIT = "c" * 40
RUNTIME_COMMIT = "b" * 40
EARLIEST_START = "2026-09-10T20:00:00Z"
REMOTE_DATE = "2026-09-10T18:00:00Z"


class _CapturedStdout:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


class ShadowCertificationTests(unittest.TestCase):
    """La preuve distante doit etre Git-only, append-only et ponctuelle."""

    def _authority(self) -> shadow.ShadowExecutionAuthority:
        return shadow.ShadowExecutionAuthority(
            runtime_code_commit=RESULTS_COMMIT,
            shadow_service_code_commit="1" * 40,
            shadow_service_module_sha256="2" * 64,
            shadow_protocol_sha256=shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
            execution_manifest_sha256="3" * 64,
            execution_manifest_introduction_commit="4" * 40,
            activation_sha256="5" * 64,
            activation_introduction_commit="6" * 40,
            activation_verified_at_utc="2026-09-06T08:00:00Z",
            minimum_target_official_date=TARGET_DATE,
            runtime_versions=(),
        )

    def _evidence(
        self,
        *,
        remote_date: str = REMOTE_DATE,
        results_commit: str = RESULTS_COMMIT,
    ) -> shadow.ShadowActivationReverificationEvidence:
        body = shadow._canonical_json_bytes(
            {
                "base_commit": {"sha": results_commit},
                "merge_base_commit": {"sha": results_commit},
                "status": "identical",
            }
        )
        body_sha256 = hashlib.sha256(body).hexdigest()
        url = shadow.GITHUB_COMPARE_URL_TEMPLATE.format(
            expected_commit=results_commit
        )
        raw = {
            "evidence_schema_version": 1,
            "request_url": url,
            "request_method": "GET",
            "application_request_headers": dict(shadow.GITHUB_REQUEST_HEADERS),
            "effective_url": url,
            "response_status_code": 200,
            "response_redirect_count": 0,
            "selected_response_headers": {
                "date": format_datetime(
                    datetime.fromisoformat(
                        remote_date.replace("Z", "+00:00")
                    ),
                    usegmt=True,
                ),
                "content-type": "application/json",
                "etag": None,
                "x-github-request-id": "certification-request",
            },
            "response_received_at_utc": remote_date,
            "response_body_base64": base64.b64encode(body).decode("ascii"),
            "response_body_sha256": body_sha256,
        }
        canonical_json = shadow._canonical_json_file_bytes(raw)
        canonical_gzip = shadow._canonical_gzip_bytes(canonical_json)
        return shadow.ShadowActivationReverificationEvidence(
            activation_introduction_commit=results_commit,
            activation_remote_ref=shadow.GITHUB_REMOTE_REF,
            activation_remote_reverified_at_utc=remote_date,
            response_received_at_utc=remote_date,
            response_body_sha256=body_sha256,
            raw_evidence=raw,
            canonical_json_bytes=canonical_json,
            canonical_gzip_bytes=canonical_gzip,
            canonical_gzip_sha256=hashlib.sha256(canonical_gzip).hexdigest(),
        )

    def _batch(self) -> certification.CompletedShadowBatchCommit:
        tree = tuple(
            {
                "path": name,
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
                "size_bytes": len(name.encode()),
            }
            for name in certification.EXPECTED_RESULT_FILENAMES
        )
        return certification.CompletedShadowBatchCommit(
            target_official_date=TARGET_DATE,
            results_commit=RESULTS_COMMIT,
            batch_id="7" * 64,
            earliest_predicted_start_utc=EARLIEST_START,
            results_tree_file_hashes=tree,
            receipt={},
        )

    def _publication_paths(
        self,
        root: Path,
    ) -> tuple[Path, Path, str, str]:
        relative_raw, relative_json = certification._certification_relative_paths(
            TARGET_DATE
        )
        raw = root.joinpath(*relative_raw.parts)
        output = root.joinpath(*relative_json.parts)
        raw.parent.mkdir(parents=True, exist_ok=True)
        return raw, output, relative_raw.as_posix(), relative_json.as_posix()

    def _publish(
        self,
        root: Path,
    ) -> certification.ShadowCertificationPublication:
        raw, output, relative_raw, relative_json = self._publication_paths(root)
        return certification._publish_prepared_certification(
            batch=self._batch(),
            evidence=self._evidence(),
            raw_evidence_path=raw,
            certification_path=output,
            raw_evidence_relative_path=relative_raw,
            certification_relative_path=relative_json,
        )

    def _committed_blobs(
        self,
        authority: shadow.ShadowExecutionAuthority,
        *,
        predicted: bool = True,
    ) -> dict[str, bytes]:
        target = TARGET_DATE
        slot_key = shadow.build_slot_key(
            shadow_protocol_sha256=authority.shadow_protocol_sha256,
            target_official_date=target,
        )
        batch_id = shadow.build_batch_id(
            slot_key=slot_key,
            execution_manifest_sha256=authority.execution_manifest_sha256,
            model_artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
        )
        reserved = {
            "marker_schema_version": 1,
            "batch_id": batch_id,
            "slot_key": slot_key,
            "target_official_date": target,
            "reserved_at_utc": "2026-09-10T16:00:00Z",
            "shadow_protocol_sha256": authority.shadow_protocol_sha256,
            "execution_manifest_sha256": authority.execution_manifest_sha256,
            "runtime_code_commit": RUNTIME_COMMIT,
        }
        row = [""] * len(shadow._PREDICTIONS_COLUMNS)
        row[0] = "8" * 64
        row[1] = batch_id
        row[2] = "123456"
        row[3] = "9" * 64
        row[4] = "2026"
        row[5] = target
        row[6] = "1"
        row[7] = "2"
        row[8] = EARLIEST_START
        predictions = shadow._canonical_csv_bytes(
            shadow._PREDICTIONS_COLUMNS,
            [row] if predicted else [],
        )
        blobs: dict[str, bytes] = {
            "RESERVED": shadow._canonical_json_file_bytes(reserved),
            shadow.ACTIVATION_REVERIFICATION_FILENAME: b"activation-evidence",
            shadow.SOURCE_SNAPSHOT_FILENAME: b"source-snapshot",
            shadow.CANDIDATE_LEDGER_FILENAME: b"candidate-ledger",
            shadow.FEATURES_FILENAME: b"features",
            shadow.PREDICTIONS_FILENAME: predictions,
        }

        def section(name: str) -> dict[str, object]:
            return {key: None for key in shadow._RECEIPT_SECTION_KEYS[name]}

        batch = section("batch")
        batch.update(
            {
                "batch_id": batch_id,
                "slot_key": slot_key,
                "target_official_date": target,
                "status": (
                    "COMPLETED_WITH_PREDICTIONS"
                    if predicted
                    else "COMPLETED_NO_ELIGIBLE_GAMES"
                ),
                "earliest_predicted_scheduled_start_utc": (
                    EARLIEST_START if predicted else None
                ),
            }
        )
        times = section("times")
        times.update(
            {
                "reserved_at_utc": reserved["reserved_at_utc"],
                "receipt_finalized_at_utc": "2026-09-10T16:10:00Z",
            }
        )
        lineage = section("lineage")
        lineage.update(
            {
                "runtime_code_commit": RUNTIME_COMMIT,
                "shadow_protocol_sha256": authority.shadow_protocol_sha256,
                "execution_manifest_sha256": authority.execution_manifest_sha256,
                "model_artifact_sha256": shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
            }
        )
        source = section("source")
        source.update(
            {
                "sqlite_snapshot_sha256": "a" * 64,
                "source_snapshot_sha256": hashlib.sha256(
                    blobs[shadow.SOURCE_SNAPSHOT_FILENAME]
                ).hexdigest(),
                "schedule_raw_archive_sha256": "b" * 64,
            }
        )
        counts = section("counts")
        counts.update(
            {
                "schedule_games": 1 if predicted else 0,
                "eligible_games": 1 if predicted else 0,
                "predicted_games": 1 if predicted else 0,
                "excluded_games_by_reason": {
                    key: 0 for key in shadow._EXCLUDED_GAMES_BY_REASON_KEYS
                },
            }
        )
        output_hashes = section("output_hashes")
        for field, filename in certification._OUTPUT_HASH_TO_FILENAME.items():
            output_hashes[field] = hashlib.sha256(blobs[filename]).hexdigest()
        negative = {key: True for key in shadow._RECEIPT_SECTION_KEYS["negative_attestations"]}
        receipt = {
            "receipt_schema_version": 1,
            "batch": batch,
            "activation": section("activation"),
            "times": times,
            "schedule_http_response": section("schedule_http_response"),
            "lineage": lineage,
            "source": source,
            "counts": counts,
            "output_hashes": output_hashes,
            "model_invariants": section("model_invariants"),
            "negative_attestations": negative,
            "runtime_versions": section("runtime_versions"),
        }
        receipt_bytes = shadow._canonical_json_file_bytes(receipt)
        blobs[shadow.RECEIPT_FILENAME] = receipt_bytes
        completed = {
            "marker_schema_version": 1,
            "batch_id": batch_id,
            "receipt_path": (
                shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH
                / target
                / shadow.RECEIPT_FILENAME
            ).as_posix(),
            "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "completed_at_utc": "2026-09-10T16:20:00Z",
        }
        blobs[shadow.COMPLETED_FILENAME] = shadow._canonical_json_file_bytes(
            completed
        )
        return blobs

    def test_runtime_roots_have_neutral_tracked_placeholders(self) -> None:
        """Les deux racines requises existent avant le premier lot."""
        project = Path(__file__).resolve().parents[1]
        for root in (
            shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH,
            shadow.SHADOW_CERTIFICATION_ROOT_RELATIVE_PATH,
        ):
            marker = project.joinpath(*root.parts, ".gitkeep")
            self.assertTrue(marker.is_file())
            self.assertEqual(marker.read_bytes(), b"\n")

    def test_constants_match_the_frozen_v2_certification_contract(self) -> None:
        """Le nouveau module applique le protocole sans le reinterpreter."""
        project = Path(__file__).resolve().parents[1]
        protocol = json.loads(
            (
                project
                / "shadow_protocols"
                / "logistic_team_form_v1_platt_shadow_v2.json"
            ).read_text(encoding="utf-8")
        )
        rules = protocol["prospective_certification"]
        self.assertEqual(
            certification.CERTIFICATION_STATUS,
            rules["certification_status_exact"],
        )
        self.assertEqual(
            certification.CERTIFICATION_CLAIM_LEVEL,
            rules["claim_level"],
        )
        self.assertEqual(
            certification.EXPECTED_RESULT_FILENAMES,
            tuple(rules["results_tree_file_hashes_expected_relative_paths"]),
        )
        self.assertEqual(
            certification._CERTIFICATION_KEYS,
            frozenset(rules["certification_keys_exact_set"]),
        )
        self.assertEqual(
            certification._TREE_HASH_KEYS,
            frozenset(
                rules["results_tree_file_hashes_entry_keys_exact_set"]
            ),
        )

    def test_module_is_separate_and_has_no_data_or_model_operation(self) -> None:
        """Certifier ne doit jamais devenir une deuxieme prediction."""
        source = inspect.getsource(certification)
        tree = ast.parse(source)
        forbidden_calls = {
            "predict_proba",
            "fit",
            "partial_fit",
            "run_observed_schedule_ingestion",
            "fetch_schedule_range_observed",
            "sqlite3",
            "load",
        }
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue(forbidden_calls.isdisjoint(called))
        self.assertNotIn("from src.database", source)
        self.assertNotIn("from src.mlb_api", source)
        self.assertNotIn("import joblib", source)

        runtime_closure = shadow._local_python_runtime_closure(
            Path(__file__).resolve().parents[1]
        )
        self.assertNotIn("src/shadow_certification.py", runtime_closure)

    def test_lead_boundary_is_exact_and_fixed_six(self) -> None:
        """Soixante minutes sont acceptees, 3 599 secondes refusees."""
        self.assertEqual(
            certification._remote_publication_lead(
                "2026-09-10T19:00:00Z",
                "2026-09-10T18:00:00Z",
            ),
            (3600, "60.000000"),
        )
        with self.assertRaises(shadow.ShadowPredictionError):
            certification._remote_publication_lead(
                "2026-09-10T18:59:59Z",
                "2026-09-10T18:00:00Z",
            )

    def test_publication_is_exact_raw_then_canonical_json(self) -> None:
        """Les deux seuls fichiers portent les octets et empreintes exacts."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._publish(root)
            self.assertEqual(
                sorted(path.name for path in result.raw_evidence_path.parent.iterdir()),
                [f"{TARGET_DATE}.json", f"{TARGET_DATE}.remote.json.gz"],
            )
            self.assertEqual(
                result.raw_evidence_path.read_bytes(),
                self._evidence().canonical_gzip_bytes,
            )
            raw_json = gzip.decompress(result.raw_evidence_path.read_bytes())
            shadow._read_canonical_json_bytes(
                raw_json,
                description="preuve distante de test",
            )
            content = result.certification_path.read_bytes()
            parsed = shadow._read_canonical_json_bytes(
                content,
                description="certification de test",
            )
            self.assertEqual(parsed, result.certification)
            self.assertEqual(
                hashlib.sha256(content).hexdigest(),
                result.certification_sha256,
            )
            self.assertEqual(result.remote_publication_lead_minutes, "120.000000")

    def test_second_publication_never_reuses_or_overwrites(self) -> None:
        """Une certification reussie consomme definitivement la date."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._publish(root)
            before = {
                path.name: path.read_bytes()
                for path in first.raw_evidence_path.parent.iterdir()
            }
            with self.assertRaises(shadow.ShadowPublicationConflictError):
                self._publish(root)
            after = {
                path.name: path.read_bytes()
                for path in first.raw_evidence_path.parent.iterdir()
            }
            self.assertEqual(after, before)

    def test_concurrent_publishers_have_exactly_one_winner(self) -> None:
        """Deux threads ne peuvent jamais certifier ou reparer la meme date."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def attempt() -> str:
                try:
                    self._publish(root)
                    return "winner"
                except shadow.ShadowPublicationConflictError:
                    return "consumed"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(lambda _: attempt(), range(2)))
            self.assertEqual(outcomes.count("winner"), 1)
            self.assertEqual(outcomes.count("consumed"), 1)

    def test_failure_after_raw_preserves_orphan_and_forbids_retry(self) -> None:
        """Une panne entre les deux liens ne doit jamais effacer la preuve."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, output, relative_raw, relative_json = self._publication_paths(root)
            real_publish = shadow._publish_exclusive_verified
            calls = 0

            def fail_second(path: Path, content: bytes) -> str:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("panne simulee")
                return real_publish(path, content)

            with mock.patch.object(
                shadow,
                "_publish_exclusive_verified",
                side_effect=fail_second,
            ):
                with self.assertRaises(OSError):
                    certification._publish_prepared_certification(
                        batch=self._batch(),
                        evidence=self._evidence(),
                        raw_evidence_path=raw,
                        certification_path=output,
                        raw_evidence_relative_path=relative_raw,
                        certification_relative_path=relative_json,
                    )
            self.assertTrue(raw.is_file())
            self.assertFalse(output.exists())

    def test_existing_or_orphan_path_blocks_before_network(self) -> None:
        """Aucun second essai ne lit GitHub apres consommation locale."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, _output, _relative_raw, _relative_json = self._publication_paths(root)
            raw.write_bytes(b"orphan")
            with mock.patch.object(
                shadow,
                "_require_tracked_nonignored_root",
            ), mock.patch.object(
                shadow,
                "_run_preflight_git",
                return_value=(1, b""),
            ), mock.patch.object(
                shadow,
                "fetch_activation_reverification_evidence",
            ) as remote:
                with self.assertRaises(shadow.ShadowPublicationConflictError):
                    certification._require_absent_certification_paths(
                        root,
                        TARGET_DATE,
                    )
            remote.assert_not_called()

    def test_orphan_is_rejected_before_clean_head_check(self) -> None:
        """Un orphelin consomme la date avant toute autre interpretation."""
        with mock.patch.object(
            shadow,
            "_preflight_project_directory",
            return_value=Path("."),
        ), mock.patch.object(
            certification,
            "_require_absent_certification_paths",
            side_effect=shadow.ShadowPublicationConflictError("consomme"),
        ) as absent, mock.patch.object(
            certification,
            "_require_exact_results_commit_head",
        ) as head, mock.patch.object(
            shadow,
            "fetch_activation_reverification_evidence",
        ) as remote:
            with self.assertRaises(shadow.ShadowPublicationConflictError):
                certification.certify_shadow_prediction(
                    TARGET_DATE,
                    RESULTS_COMMIT,
                )
        absent.assert_called_once_with(Path("."), TARGET_DATE)
        head.assert_not_called()
        remote.assert_not_called()

    def test_results_commit_must_be_exact_clean_head(self) -> None:
        """L'appelant ne peut choisir un ancien commit ou un arbre sale."""
        with mock.patch.object(
            shadow,
            "_require_exact_clean_git_root",
            return_value="d" * 40,
        ):
            with self.assertRaises(shadow.ShadowPredictionError):
                certification._require_exact_results_commit_head(
                    Path("."),
                    RESULTS_COMMIT,
                )

    def test_git_blob_validation_binds_receipt_predictions_and_completed(self) -> None:
        """Les huit empreintes viennent du commit et concordent avec le recu."""
        authority = self._authority()
        blobs = self._committed_blobs(authority)
        with mock.patch.object(shadow, "_require_git_ancestor") as ancestor:
            batch = certification._validate_committed_batch_blobs(
                project_directory=Path("."),
                target_official_date=TARGET_DATE,
                results_commit=RESULTS_COMMIT,
                authority=authority,
                blobs=blobs,
            )
        self.assertEqual(batch.earliest_predicted_start_utc, EARLIEST_START)
        self.assertEqual(
            tuple(entry["path"] for entry in batch.results_tree_file_hashes),
            certification.EXPECTED_RESULT_FILENAMES,
        )
        ancestor.assert_called_once_with(
            Path("."),
            RUNTIME_COMMIT,
            RESULTS_COMMIT,
            strict=True,
            description="execution locale vers commit de resultats",
        )

    def test_completed_batch_is_read_only_from_the_exact_results_commit(self) -> None:
        """Empreintes et tailles ne proviennent jamais de l'arbre courant."""
        authority = self._authority()
        prefix = (
            shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH / TARGET_DATE
        ).as_posix()
        expected_paths = tuple(
            f"{prefix}/{name}"
            for name in certification.EXPECTED_RESULT_FILENAMES
        )
        inspection = shadow.ShadowPredictionSlotInspection(
            state=shadow.ShadowPredictionSlotState.COMPLETED_EXACT,
            slot_path=Path(prefix),
            slot_key="1" * 64,
            batch_id="2" * 64,
            receipt={},
        )
        blob_values = {
            name: f"blob:{name}".encode("ascii")
            for name in certification.EXPECTED_RESULT_FILENAMES
        }

        def git_blob(
            project: Path,
            commit: str,
            relative_path: str,
        ) -> bytes:
            self.assertEqual(project, Path("project"))
            self.assertEqual(commit, RESULTS_COMMIT)
            self.assertTrue(relative_path.startswith(f"{prefix}/"))
            return blob_values[relative_path.rsplit("/", 1)[1]]

        with mock.patch.object(
            shadow,
            "verify_shadow_execution_authority",
            return_value=authority,
        ), mock.patch.object(
            shadow,
            "inspect_shadow_prediction_slot",
            return_value=inspection,
        ), mock.patch.object(
            certification,
            "_git_lines",
            side_effect=(expected_paths, (RESULTS_COMMIT,)),
        ), mock.patch.object(
            shadow,
            "_git_blob_at_commit",
            side_effect=git_blob,
        ) as read_blob, mock.patch.object(
            certification,
            "_validate_committed_batch_blobs",
            return_value=self._batch(),
        ) as validate:
            result = certification._read_completed_batch_commit(
                TARGET_DATE,
                RESULTS_COMMIT,
                project_directory=Path("project"),
            )

        self.assertEqual(result, self._batch())
        self.assertEqual(
            read_blob.call_count,
            len(certification.EXPECTED_RESULT_FILENAMES),
        )
        self.assertEqual(validate.call_args.kwargs["blobs"], blob_values)

    def test_changed_git_blob_is_rejected(self) -> None:
        """Un seul octet divergent interdit toute attestation distante."""
        authority = self._authority()
        blobs = self._committed_blobs(authority)
        blobs[shadow.FEATURES_FILENAME] += b"altered"
        with mock.patch.object(shadow, "_require_git_ancestor"):
            with self.assertRaises(shadow.ShadowPredictionError):
                certification._validate_committed_batch_blobs(
                    project_directory=Path("."),
                    target_official_date=TARGET_DATE,
                    results_commit=RESULTS_COMMIT,
                    authority=authority,
                    blobs=blobs,
                )

    def test_empty_completed_batch_is_never_certified(self) -> None:
        """Sans prediction et premier horaire, aucune preuve ne peut naitre."""
        authority = self._authority()
        blobs = self._committed_blobs(authority, predicted=False)
        with mock.patch.object(shadow, "_require_git_ancestor"):
            with self.assertRaises(shadow.ShadowPredictionError):
                certification._validate_committed_batch_blobs(
                    project_directory=Path("."),
                    target_official_date=TARGET_DATE,
                    results_commit=RESULTS_COMMIT,
                    authority=authority,
                    blobs=blobs,
                )

    def test_remote_failure_or_late_evidence_creates_no_file(self) -> None:
        """Sans preuve ponctuelle valide, le lot reste seulement local."""
        for outcome in (
            shadow.ShadowPredictionError("reseau indisponible"),
            self._evidence(remote_date="2026-09-10T19:00:01Z"),
        ):
            with self.subTest(outcome=type(outcome).__name__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = self._publication_paths(root)
                patch_value = (
                    mock.patch.object(
                        shadow,
                        "fetch_activation_reverification_evidence",
                        side_effect=outcome,
                    )
                    if isinstance(outcome, BaseException)
                    else mock.patch.object(
                        shadow,
                        "fetch_activation_reverification_evidence",
                        return_value=outcome,
                    )
                )
                with mock.patch.object(
                    shadow,
                    "_preflight_project_directory",
                    return_value=root,
                ), mock.patch.object(
                    certification,
                    "_require_exact_results_commit_head",
                    return_value=RESULTS_COMMIT,
                ), mock.patch.object(
                    certification,
                    "_require_absent_certification_paths",
                    return_value=paths,
                ), mock.patch.object(
                    certification,
                    "_read_completed_batch_commit",
                    return_value=self._batch(),
                ), patch_value:
                    with self.assertRaises(shadow.ShadowPredictionError):
                        certification.certify_shadow_prediction(
                            TARGET_DATE,
                            RESULTS_COMMIT,
                            project_directory=root,
                        )
                self.assertFalse(paths[0].exists())
                self.assertFalse(paths[1].exists())

    def test_public_orchestration_publishes_only_after_remote_proof(self) -> None:
        """Le chemin public lie HEAD, lot Git, preuve puis deux fichiers."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._publication_paths(root)
            with mock.patch.object(
                shadow,
                "_preflight_project_directory",
                return_value=root,
            ), mock.patch.object(
                certification,
                "_require_exact_results_commit_head",
                return_value=RESULTS_COMMIT,
            ), mock.patch.object(
                certification,
                "_require_absent_certification_paths",
                return_value=paths,
            ), mock.patch.object(
                certification,
                "_read_completed_batch_commit",
                return_value=self._batch(),
            ), mock.patch.object(
                shadow,
                "fetch_activation_reverification_evidence",
                return_value=self._evidence(),
            ) as remote:
                result = certification.certify_shadow_prediction(
                    TARGET_DATE,
                    RESULTS_COMMIT,
                    project_directory=root,
                )
            remote.assert_called_once_with(RESULTS_COMMIT)
            self.assertTrue(result.raw_evidence_path.is_file())
            self.assertTrue(result.certification_path.is_file())

    def test_cli_requires_explicit_flag_and_emits_exact_json(self) -> None:
        """Aucune certification CLI implicite ou sortie reformatee n'existe."""
        capture = _CapturedStdout()
        with mock.patch.object(
            certification,
            "certify_shadow_prediction",
            return_value=mock.sentinel.publication,
        ) as certify, mock.patch.object(
            certification,
            "_canonical_certification_output_bytes",
            return_value=b'{"status":"certified"}\n',
        ), mock.patch.object(certification.sys, "stdout", capture):
            return_code = certification.main(
                [
                    "--certify-shadow",
                    "--target-official-date",
                    TARGET_DATE,
                    "--expected-results-commit",
                    RESULTS_COMMIT,
                ]
            )
        self.assertEqual(return_code, 0)
        self.assertEqual(capture.buffer.getvalue(), b'{"status":"certified"}\n')
        certify.assert_called_once_with(TARGET_DATE, RESULTS_COMMIT)

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                certification.main(
                    [
                        "--target-official-date",
                        TARGET_DATE,
                        "--expected-results-commit",
                        RESULTS_COMMIT,
                    ]
                )


if __name__ == "__main__":
    unittest.main()
