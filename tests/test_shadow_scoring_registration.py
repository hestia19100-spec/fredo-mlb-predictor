"""Tests du preenregistrement distant du scoring shadow MLB 2026."""

from __future__ import annotations

import ast
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from copy import deepcopy
from datetime import datetime
from email.utils import format_datetime
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import shadow_certification as certification
from src import shadow_prediction as shadow
from src import shadow_scoring_registration as registration


RUNTIME_COMMIT = "a" * 40
REMOTE_DATE = "2026-09-10T14:15:00Z"


class _CapturedStdout:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


class ShadowScoringRegistrationTests(unittest.TestCase):
    """Le gel doit preceder les resultats et rester exclusif."""

    def _protocol(self) -> tuple[dict[str, object], bytes]:
        path = registration.PROJECT_DIRECTORY.joinpath(
            *registration.SCORING_PROTOCOL_RELATIVE_PATH.parts
        )
        content = path.read_bytes()
        return json.loads(content.decode("utf-8")), content

    def _batch(self) -> certification.CompletedShadowBatchCommit:
        return certification.CompletedShadowBatchCommit(
            target_official_date=registration.EXPECTED_FIRST_TARGET_DATE,
            results_commit=registration.EXPECTED_FIRST_RESULTS_COMMIT,
            batch_id=registration.EXPECTED_FIRST_BATCH_ID,
            earliest_predicted_start_utc=registration.EXPECTED_FIRST_START_UTC,
            results_tree_file_hashes=(),
            receipt={},
        )

    def _authority(self) -> registration.ScoringProtocolAuthority:
        protocol, content = self._protocol()
        return registration.ScoringProtocolAuthority(
            protocol=protocol,
            protocol_bytes=content,
            protocol_sha256=registration.EXPECTED_SCORING_PROTOCOL_SHA256,
            protocol_introduction_commit=RUNTIME_COMMIT,
            runtime_code_commit=RUNTIME_COMMIT,
            registration_service_sha256="b" * 64,
            first_batch=self._batch(),
            first_certification={},
        )

    def _evidence(
        self,
        *,
        remote_date: str = REMOTE_DATE,
    ) -> shadow.ShadowActivationReverificationEvidence:
        body = shadow._canonical_json_bytes(
            {
                "base_commit": {"sha": RUNTIME_COMMIT},
                "merge_base_commit": {"sha": RUNTIME_COMMIT},
                "status": "identical",
            }
        )
        body_sha256 = hashlib.sha256(body).hexdigest()
        url = shadow.GITHUB_COMPARE_URL_TEMPLATE.format(
            expected_commit=RUNTIME_COMMIT
        )
        date_header = format_datetime(
            datetime.fromisoformat(remote_date.replace("Z", "+00:00")),
            usegmt=True,
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
                "date": date_header,
                "content-type": "application/json",
                "etag": None,
                "x-github-request-id": "scoring-registration-test",
            },
            "response_received_at_utc": remote_date,
            "response_body_base64": base64.b64encode(body).decode("ascii"),
            "response_body_sha256": body_sha256,
        }
        canonical_json = shadow._canonical_json_file_bytes(raw)
        canonical_gzip = shadow._canonical_gzip_bytes(canonical_json)
        return shadow.ShadowActivationReverificationEvidence(
            activation_introduction_commit=RUNTIME_COMMIT,
            activation_remote_ref=shadow.GITHUB_REMOTE_REF,
            activation_remote_reverified_at_utc=remote_date,
            response_received_at_utc=remote_date,
            response_body_sha256=body_sha256,
            raw_evidence=raw,
            canonical_json_bytes=canonical_json,
            canonical_gzip_bytes=canonical_gzip,
            canonical_gzip_sha256=hashlib.sha256(canonical_gzip).hexdigest(),
        )

    def _paths(self, root: Path) -> tuple[Path, Path]:
        directory = root.joinpath(
            *registration.SCORING_REGISTRATION_ROOT_RELATIVE_PATH.parts
        )
        directory.mkdir(parents=True)
        return directory / "protocol.remote.json.gz", directory / "registration.json"

    def _publish(
        self,
        root: Path,
    ) -> registration.ScoringRegistrationPublication:
        raw_path, output_path = self._paths(root)
        return registration._publish_registration(
            self._authority(),
            self._evidence(),
            raw_evidence_path=raw_path,
            registration_path=output_path,
        )

    def test_real_protocol_is_accepted_with_exact_hash(self) -> None:
        """Le module et le test statique partagent une autorite unique."""
        protocol, content, sha256 = registration._read_scoring_protocol(
            registration.PROJECT_DIRECTORY
        )
        self.assertEqual(sha256, registration.EXPECTED_SCORING_PROTOCOL_SHA256)
        self.assertEqual(hashlib.sha256(content).hexdigest(), sha256)
        self.assertEqual(protocol["protocol_id"], registration.EXPECTED_PROTOCOL_ID)

    def test_critical_protocol_mutations_are_rejected(self) -> None:
        """Dates, lot initial et interdictions ne peuvent pas deriver."""
        protocol, _ = self._protocol()
        mutations = []
        changed = deepcopy(protocol)
        changed["fixed_horizon"]["no_early_stop"] = False
        mutations.append(changed)
        changed = deepcopy(protocol)
        changed["intent_to_observe"]["dates"].pop()
        mutations.append(changed)
        changed = deepcopy(protocol)
        changed["authorities"]["first_certified_batch"]["results_commit"] = "c" * 40
        mutations.append(changed)
        changed = deepcopy(protocol)
        changed["prospective_boundary"]["outcomes_must_not_be_read_before_remote_registration"] = False
        mutations.append(changed)
        changed = deepcopy(protocol)
        changed["forbidden_actions"].remove("STOP_EARLY_USING_OBSERVED_OUTCOMES")
        mutations.append(changed)
        changed = deepcopy(protocol)
        changed["verdict"]["coverage_gate"]["minimum_operationally_valid_dates"] = 17
        mutations.append(changed)
        changed = deepcopy(protocol)
        changed["outcome_adjudication"]["result_by_reduced_family"][
            "POSTPONED_BEFORE_DEADLINE"
        ] = "VOID_POSTPONED"
        mutations.append(changed)
        changed = deepcopy(protocol)
        changed["uncertainty"]["blocks_drawn_per_replicate_exact"] = 2
        mutations.append(changed)
        changed = deepcopy(protocol)
        changed["output_publication"]["observation_success_write_order_exact"].pop()
        mutations.append(changed)
        changed = deepcopy(protocol)
        changed["daily_reporting"]["metric_value_rules"][
            "accuracy_mean_log_loss_and_mean_brier_score_when_scored_count_is_zero"
        ] = 0.0
        mutations.append(changed)
        changed = deepcopy(protocol)
        changed["prediction_source"]["discovery_algorithm_exact"].pop(9)
        mutations.append(changed)
        for mutated in mutations:
            with self.subTest(mutated=mutated), self.assertRaises(
                shadow.ShadowPredictionError
            ):
                registration._validate_scoring_protocol(mutated)

    def test_remote_lead_accepts_exact_boundary_and_rejects_one_second_late(self) -> None:
        """La regle des soixante minutes est inclusive et non negociable."""
        seconds, minutes = registration._remote_registration_lead(
            "2026-09-10T15:15:00Z",
            "2026-09-10T15:15:00Z",
        )
        self.assertEqual(seconds, 3600)
        self.assertEqual(minutes, "60.000000")
        with self.assertRaises(shadow.ShadowPredictionError):
            registration._remote_registration_lead(
                "2026-09-10T15:15:01Z",
                "2026-09-10T15:15:01Z",
            )

    def test_remote_date_cannot_be_replayed_with_a_late_local_receive(self) -> None:
        """Une reponse ancienne ne peut pas preenregistrer apres la frontiere."""
        with self.assertRaises(shadow.ShadowPredictionError):
            registration._remote_registration_lead(
                "2026-09-10T14:00:00Z",
                "2026-09-10T15:16:00Z",
            )
        with self.assertRaises(shadow.ShadowPredictionError):
            registration._remote_registration_lead(
                "2026-09-10T14:00:00Z",
                "2026-09-10T14:05:01Z",
            )

    def test_prepared_publication_is_canonical_and_exactly_bound(self) -> None:
        """Le recu lie protocole, code, lot initial et preuve GitHub."""
        with tempfile.TemporaryDirectory() as temporary:
            publication = self._publish(Path(temporary))
            registration_bytes = publication.registration_path.read_bytes()
            parsed = json.loads(registration_bytes)
            self.assertEqual(
                registration_bytes,
                shadow._canonical_json_file_bytes(parsed),
            )
            self.assertEqual(
                publication.registration_sha256,
                hashlib.sha256(registration_bytes).hexdigest(),
            )
            self.assertEqual(parsed["scoring_protocol_sha256"], registration.EXPECTED_SCORING_PROTOCOL_SHA256)
            self.assertEqual(parsed["runtime_code_commit"], RUNTIME_COMMIT)
            self.assertEqual(parsed["first_batch_id"], registration.EXPECTED_FIRST_BATCH_ID)
            self.assertEqual(parsed["first_results_commit"], registration.EXPECTED_FIRST_RESULTS_COMMIT)
            self.assertEqual(parsed["first_certification_commit"], registration.EXPECTED_FIRST_CERTIFICATION_COMMIT)
            self.assertEqual(parsed["remote_publication_lead_minutes"], "120.000000")
            self.assertTrue(all(parsed["negative_attestations"].values()))
            self.assertEqual(
                publication.raw_evidence_path.read_bytes(),
                self._evidence().canonical_gzip_bytes,
            )

    def test_concurrent_publishers_have_exactly_one_winner(self) -> None:
        """Deux processus ne peuvent jamais partager le preenregistrement."""
        with tempfile.TemporaryDirectory() as temporary:
            raw_path, output_path = self._paths(Path(temporary))

            def publish() -> str:
                try:
                    registration._publish_registration(
                        self._authority(),
                        self._evidence(),
                        raw_evidence_path=raw_path,
                        registration_path=output_path,
                    )
                except shadow.ShadowPublicationConflictError:
                    return "CONFLICT"
                return "WINNER"

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: publish(), range(2)))
            self.assertEqual(results.count("WINNER"), 1)
            self.assertEqual(results.count("CONFLICT"), 1)
            self.assertTrue(raw_path.is_file())
            self.assertTrue(output_path.is_file())

    def test_failure_after_raw_publication_preserves_permanent_orphan(self) -> None:
        """Une panne partielle ne permet jamais de reparer apres coup."""
        with tempfile.TemporaryDirectory() as temporary:
            raw_path, output_path = self._paths(Path(temporary))
            original = shadow._publish_exclusive_verified
            calls = 0

            def fail_second(path: Path, content: bytes) -> str:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise shadow.ShadowPredictionError("panne injectee")
                return original(path, content)

            with mock.patch.object(
                shadow,
                "_publish_exclusive_verified",
                side_effect=fail_second,
            ), self.assertRaisesRegex(shadow.ShadowPredictionError, "panne"):
                registration._publish_registration(
                    self._authority(),
                    self._evidence(),
                    raw_evidence_path=raw_path,
                    registration_path=output_path,
                )
            self.assertTrue(raw_path.is_file())
            self.assertFalse(output_path.exists())
            with self.assertRaises(shadow.ShadowPublicationConflictError):
                original(raw_path, b"replacement")

    def test_authority_requires_exact_clean_introduction_commit(self) -> None:
        """Le protocole ne peut etre enregistre depuis un descendant ulterieur."""
        protocol, content = self._protocol()
        batch = self._batch()
        service_bytes = b"service exact"
        with (
            mock.patch.object(registration, "_require_absent_registration_paths"),
            mock.patch.object(
                registration,
                "_read_scoring_protocol",
                return_value=(
                    protocol,
                    content,
                    registration.EXPECTED_SCORING_PROTOCOL_SHA256,
                ),
            ),
            mock.patch.object(
                shadow,
                "_require_exact_clean_git_root",
                return_value=RUNTIME_COMMIT,
            ),
            mock.patch.object(
                shadow,
                "_git_immutable_introduction_blob",
                return_value=(RUNTIME_COMMIT, content),
            ),
            mock.patch.object(
                shadow,
                "_read_regular_project_file",
                return_value=service_bytes,
            ),
            mock.patch.object(
                shadow,
                "_git_blob_at_commit",
                return_value=service_bytes,
            ),
            mock.patch.object(
                registration,
                "_validate_first_certified_batch",
                return_value=(batch, {}),
            ),
        ):
            authority = registration.verify_scoring_protocol_authority(
                project_directory=registration.PROJECT_DIRECTORY
            )
        self.assertEqual(authority.runtime_code_commit, RUNTIME_COMMIT)
        self.assertEqual(
            authority.registration_service_sha256,
            hashlib.sha256(service_bytes).hexdigest(),
        )

        with (
            mock.patch.object(registration, "_require_absent_registration_paths"),
            mock.patch.object(
                registration,
                "_read_scoring_protocol",
                return_value=(
                    protocol,
                    content,
                    registration.EXPECTED_SCORING_PROTOCOL_SHA256,
                ),
            ),
            mock.patch.object(
                shadow,
                "_require_exact_clean_git_root",
                return_value=RUNTIME_COMMIT,
            ),
            mock.patch.object(
                shadow,
                "_git_immutable_introduction_blob",
                return_value=("c" * 40, content),
            ),
            self.assertRaises(shadow.ShadowPredictionError),
        ):
            registration.verify_scoring_protocol_authority(
                project_directory=registration.PROJECT_DIRECTORY
            )

    def test_local_failure_happens_before_network(self) -> None:
        """Aucun appel GitHub ne precede l'autorite locale complete."""
        with (
            mock.patch.object(
                registration,
                "verify_scoring_protocol_authority",
                side_effect=shadow.ShadowPredictionError("local invalide"),
            ),
            mock.patch.object(
                shadow,
                "fetch_activation_reverification_evidence",
            ) as fetch,
            self.assertRaisesRegex(shadow.ShadowPredictionError, "local invalide"),
        ):
            registration.register_scoring_protocol(
                project_directory=registration.PROJECT_DIRECTORY
            )
        fetch.assert_not_called()

    def test_success_flow_has_one_github_call_then_two_publications(self) -> None:
        """L'ordre autorite, GitHub, raw puis recu est ferme."""
        authority = self._authority()
        evidence = self._evidence()
        with tempfile.TemporaryDirectory() as temporary:
            raw_path, output_path = self._paths(Path(temporary))
            with (
                mock.patch.object(
                    registration,
                    "verify_scoring_protocol_authority",
                    return_value=authority,
                ),
                mock.patch.object(
                    registration,
                    "_require_absent_registration_paths",
                    return_value=(raw_path, output_path),
                ),
                mock.patch.object(
                    shadow,
                    "fetch_activation_reverification_evidence",
                    return_value=evidence,
                ) as fetch,
            ):
                publication = registration.register_scoring_protocol(
                    project_directory=registration.PROJECT_DIRECTORY
                )
            fetch.assert_called_once_with(RUNTIME_COMMIT)
            self.assertTrue(publication.raw_evidence_path.is_file())
            self.assertTrue(publication.registration_path.is_file())

    def test_cli_requires_explicit_flag_and_emits_exact_json(self) -> None:
        """Sans autorisation explicite, aucun fichier ne peut etre publie."""
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            registration.main([])
        self.assertIn("--register-scoring", stderr.getvalue())

        with tempfile.TemporaryDirectory() as temporary:
            publication = self._publish(Path(temporary))
            captured = _CapturedStdout()
            with (
                mock.patch.object(
                    registration,
                    "register_scoring_protocol",
                    return_value=publication,
                ),
                mock.patch.object(registration.sys, "stdout", captured),
            ):
                result = registration.main(["--register-scoring"])
            self.assertEqual(result, 0)
            self.assertEqual(
                captured.buffer.getvalue(),
                publication.registration_path.read_bytes(),
            )

    def test_module_has_no_direct_mlb_sqlite_or_model_dependency(self) -> None:
        """Le preenregistrement ne contient aucun chemin vers un resultat."""
        source_path = registration.PROJECT_DIRECTORY.joinpath(
            *registration.REGISTRATION_SERVICE_RELATIVE_PATH.parts
        )
        tree = ast.parse(source_path.read_bytes(), filename=str(source_path))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertNotIn("sqlite3", imported)
        self.assertNotIn("joblib", imported)
        self.assertNotIn("src.database", imported)
        self.assertNotIn("src.mlb_api", imported)
        self.assertNotIn("src.ingestion_service", imported)


if __name__ == "__main__":
    unittest.main()
