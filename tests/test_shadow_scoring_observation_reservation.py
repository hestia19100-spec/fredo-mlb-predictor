"""Tests du preflight et de la reservation des observations de scoring."""

from __future__ import annotations

import ast
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.utils import format_datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src import shadow_prediction as shadow
from src import shadow_scoring as scoring
from src import shadow_scoring_registration as registration


RUNTIME_COMMIT = "d" * 40
TARGET_DATE = "2026-09-10"
CHECKPOINT_DATE = "2026-09-11"
RESERVED_AT = "2026-09-11T06:00:00Z"


class ShadowScoringObservationReservationTests(unittest.TestCase):
    """Le premier fichier de chaque observation doit etre atomique."""

    def _authority(self) -> scoring.ScoringExecutionAuthority:
        return scoring.ScoringExecutionAuthority(
            protocol={},
            registration={},
            scoring_protocol_sha256=(
                registration.EXPECTED_SCORING_PROTOCOL_SHA256
            ),
            protocol_introduction_commit=(
                scoring.EXPECTED_PROTOCOL_INTRODUCTION_COMMIT
            ),
            registration_commit=scoring.EXPECTED_REGISTRATION_COMMIT,
            registration_sha256=scoring.EXPECTED_REGISTRATION_SHA256,
            registration_remote_evidence_sha256=(
                scoring.EXPECTED_REGISTRATION_REMOTE_EVIDENCE_SHA256
            ),
            runtime_code_commit=RUNTIME_COMMIT,
            scoring_engine_sha256="e" * 64,
        )

    def _root(self, temporary: str) -> Path:
        root = Path(temporary)
        root.joinpath(*scoring.SCORING_OUTPUT_ROOT_RELATIVE_PATH.parts).mkdir(
            parents=True
        )
        return root

    def _reserve(self, root: Path) -> scoring.ScoringObservationReservation:
        return scoring.reserve_scoring_observation_slot(
            self._authority(),
            TARGET_DATE,
            CHECKPOINT_DATE,
            reserved_at_utc=RESERVED_AT,
            project_directory=root,
        )

    def _remote_fixture(self) -> tuple[dict[str, object], bytes]:
        remote_date = "2026-09-10T15:09:14Z"
        expected_commit = scoring.EXPECTED_PROTOCOL_INTRODUCTION_COMMIT
        body = shadow._canonical_json_bytes(
            {
                "base_commit": {"sha": expected_commit},
                "merge_base_commit": {"sha": expected_commit},
                "status": "identical",
            }
        )
        request_url = shadow.GITHUB_COMPARE_URL_TEMPLATE.format(
            expected_commit=expected_commit
        )
        raw = {
            "evidence_schema_version": 1,
            "request_url": request_url,
            "request_method": "GET",
            "application_request_headers": dict(shadow.GITHUB_REQUEST_HEADERS),
            "effective_url": request_url,
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
                "x-github-request-id": "scoring-runtime-test",
            },
            "response_received_at_utc": remote_date,
            "response_body_base64": base64.b64encode(body).decode("ascii"),
            "response_body_sha256": hashlib.sha256(body).hexdigest(),
        }
        raw_json = shadow._canonical_json_file_bytes(raw)
        raw_gzip = shadow._canonical_gzip_bytes(raw_json)
        record = {
            "registration_schema_version": 1,
            "protocol_id": registration.EXPECTED_PROTOCOL_ID,
            "status": registration.REGISTRATION_STATUS,
            "claim_level": registration.REGISTRATION_CLAIM_LEVEL,
            "scoring_protocol_path": (
                registration.SCORING_PROTOCOL_RELATIVE_PATH.as_posix()
            ),
            "scoring_protocol_sha256": (
                registration.EXPECTED_SCORING_PROTOCOL_SHA256
            ),
            "scoring_protocol_introduction_commit": expected_commit,
            "registration_service_path": (
                registration.REGISTRATION_SERVICE_RELATIVE_PATH.as_posix()
            ),
            "registration_service_sha256": (
                scoring.EXPECTED_REGISTRATION_SERVICE_SHA256
            ),
            "runtime_code_commit": expected_commit,
            "remote_ref": shadow.GITHUB_REMOTE_REF,
            "remote_query_url": request_url,
            "remote_effective_url": request_url,
            "remote_response_status_code": 200,
            "remote_response_redirect_count": 0,
            "remote_http_date_utc": remote_date,
            "remote_response_received_at_utc": remote_date,
            "remote_response_body_sha256": hashlib.sha256(body).hexdigest(),
            "raw_remote_evidence_path": (
                registration.RAW_REMOTE_EVIDENCE_RELATIVE_PATH.as_posix()
            ),
            "raw_remote_evidence_sha256": hashlib.sha256(raw_gzip).hexdigest(),
            "first_target_official_date": registration.EXPECTED_FIRST_TARGET_DATE,
            "first_batch_id": registration.EXPECTED_FIRST_BATCH_ID,
            "first_results_commit": registration.EXPECTED_FIRST_RESULTS_COMMIT,
            "first_certification_commit": (
                registration.EXPECTED_FIRST_CERTIFICATION_COMMIT
            ),
            "first_certification_sha256": (
                registration.EXPECTED_FIRST_CERTIFICATION_SHA256
            ),
            "first_predictions_sha256": (
                registration.EXPECTED_FIRST_PREDICTIONS_SHA256
            ),
            "first_receipt_sha256": registration.EXPECTED_FIRST_RECEIPT_SHA256,
            "first_earliest_predicted_start_utc": (
                registration.EXPECTED_FIRST_START_UTC
            ),
            "remote_publication_lead_minutes": "65.766667",
            "registered_at_utc": remote_date,
            "negative_attestations": {
                key: True
                for key in sorted(registration._NEGATIVE_ATTESTATION_KEYS)
            },
        }
        return record, raw_gzip

    def test_observation_id_uses_the_exact_registered_preimage(self) -> None:
        """L'identifiant est reproductible et lie les deux dates."""
        preimage = (
            registration.EXPECTED_SCORING_PROTOCOL_SHA256
            + "\n2026-09-10\n2026-09-11\n"
        ).encode("utf-8")
        self.assertEqual(
            scoring.build_observation_id(
                scoring_protocol_sha256=(
                    registration.EXPECTED_SCORING_PROTOCOL_SHA256
                ),
                target_official_date=TARGET_DATE,
                checkpoint_utc_date=CHECKPOINT_DATE,
            ),
            hashlib.sha256(preimage).hexdigest(),
        )

    def test_reservation_publishes_only_a_canonical_reserved_marker(self) -> None:
        """Avant MLB, le creneau ne contient exactement que RESERVED."""
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            reservation = self._reserve(root)
            self.assertEqual(
                sorted(path.name for path in reservation.slot_path.iterdir()),
                ["RESERVED"],
            )
            content = (reservation.slot_path / "RESERVED").read_bytes()
            parsed = json.loads(content)
            self.assertEqual(content, shadow._canonical_json_file_bytes(parsed))
            self.assertEqual(frozenset(parsed), scoring._RESERVED_MARKER_KEYS)
            self.assertEqual(parsed["observation_id"], reservation.observation_id)
            self.assertEqual(
                hashlib.sha256(content).hexdigest(),
                reservation.reserved_marker_sha256,
            )
            inspection = scoring.inspect_scoring_observation_slot(
                self._authority(),
                TARGET_DATE,
                CHECKPOINT_DATE,
                project_directory=root,
            )
            self.assertIs(
                inspection.state,
                scoring.ScoringObservationSlotState.RESERVED_EXACT,
            )

    def test_presence_first_inspection_reads_no_git_or_project_file(self) -> None:
        """Un creneau consomme est detecte avant toute autorite externe."""
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            slot = root.joinpath(
                *scoring.SCORING_OUTPUT_ROOT_RELATIVE_PATH.parts,
                TARGET_DATE,
                "observations",
                CHECKPOINT_DATE,
            )
            slot.mkdir(parents=True)
            with (
                mock.patch.object(shadow, "_run_preflight_git") as git_call,
                mock.patch.object(shadow, "_read_regular_project_file") as read,
            ):
                inspection = (
                    scoring.inspect_scoring_observation_slot_presence_first(
                        TARGET_DATE,
                        CHECKPOINT_DATE,
                        project_directory=root,
                    )
                )
            self.assertIs(
                inspection.state,
                scoring.ScoringObservationSlotState.CONSUMED,
            )
            git_call.assert_not_called()
            read.assert_not_called()

    def test_preexisting_empty_or_invalid_slot_is_never_repaired(self) -> None:
        """La simple existence du chemin gagne contre toute seconde execution."""
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            slot = root.joinpath(
                *scoring.SCORING_OUTPUT_ROOT_RELATIVE_PATH.parts,
                TARGET_DATE,
                "observations",
                CHECKPOINT_DATE,
            )
            slot.mkdir(parents=True)
            with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                self._reserve(root)
            self.assertEqual(list(slot.iterdir()), [])

    def test_failure_after_slot_creation_leaves_a_permanent_consumed_path(self) -> None:
        """Une panne avant RESERVED ne permet aucun nettoyage ni nouvel essai."""
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            with (
                mock.patch.object(
                    shadow,
                    "_publish_exclusive_verified",
                    side_effect=shadow.ShadowPredictionError("panne injectee"),
                ),
                self.assertRaisesRegex(shadow.ShadowPredictionError, "panne"),
            ):
                self._reserve(root)
            inspection = scoring.inspect_scoring_observation_slot_presence_first(
                TARGET_DATE,
                CHECKPOINT_DATE,
                project_directory=root,
            )
            self.assertIs(
                inspection.state,
                scoring.ScoringObservationSlotState.CONSUMED,
            )
            with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                self._reserve(root)

    def test_concurrent_threads_have_exactly_one_winner(self) -> None:
        """Le mkdir atomique designe un seul gagnant dans un processus."""
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)

            def reserve() -> str:
                try:
                    self._reserve(root)
                except shadow.ShadowPredictionSlotConsumedError:
                    return "CONFLICT"
                return "WINNER"

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: reserve(), range(2)))
            self.assertEqual(results.count("WINNER"), 1)
            self.assertEqual(results.count("CONFLICT"), 1)

    def test_concurrent_processes_have_exactly_one_winner(self) -> None:
        """Deux interpreteurs independants ne partagent jamais un creneau."""
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            gate = root / "GO"
            worker = r'''
import sys, time
from pathlib import Path
from src import shadow_prediction as shadow
from src import shadow_scoring as scoring
from src import shadow_scoring_registration as registration
root, gate = Path(sys.argv[1]), Path(sys.argv[2])
authority = scoring.ScoringExecutionAuthority(
    protocol={}, registration={},
    scoring_protocol_sha256=registration.EXPECTED_SCORING_PROTOCOL_SHA256,
    protocol_introduction_commit=scoring.EXPECTED_PROTOCOL_INTRODUCTION_COMMIT,
    registration_commit=scoring.EXPECTED_REGISTRATION_COMMIT,
    registration_sha256=scoring.EXPECTED_REGISTRATION_SHA256,
    registration_remote_evidence_sha256=scoring.EXPECTED_REGISTRATION_REMOTE_EVIDENCE_SHA256,
    runtime_code_commit="d" * 40, scoring_engine_sha256="e" * 64,
)
while not gate.exists():
    time.sleep(0.001)
try:
    scoring.reserve_scoring_observation_slot(
        authority, "2026-09-10", "2026-09-11",
        reserved_at_utc="2026-09-11T06:00:00Z",
        project_directory=root,
    )
except shadow.ShadowPredictionSlotConsumedError:
    print("CONFLICT")
else:
    print("WINNER")
'''
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", worker, str(root), str(gate)],
                    cwd=registration.PROJECT_DIRECTORY,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for _ in range(2)
            ]
            gate.write_text("go\n", encoding="utf-8")
            outputs = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, stderr)
                outputs.append(stdout.strip())
            self.assertEqual(outputs.count("WINNER"), 1)
            self.assertEqual(outputs.count("CONFLICT"), 1)

    def test_checkpoint_boundaries_are_closed_before_any_path_creation(self) -> None:
        """Pas d'observation precoce, tardive ou recreee a une autre date."""
        invalid = [
            ("2026-09-10", "2026-09-10T06:00:00Z"),
            ("2026-09-11", "2026-09-11T05:59:59Z"),
            ("2026-09-11", "2026-09-12T06:00:00Z"),
            ("2026-10-13", "2026-10-13T12:00:00Z"),
        ]
        for checkpoint, timestamp in invalid:
            with self.subTest(checkpoint=checkpoint, timestamp=timestamp):
                with tempfile.TemporaryDirectory() as temporary:
                    root = self._root(temporary)
                    with self.assertRaises(shadow.ShadowPredictionError):
                        scoring.reserve_scoring_observation_slot(
                            self._authority(),
                            TARGET_DATE,
                            checkpoint,
                            reserved_at_utc=timestamp,
                            project_directory=root,
                        )
                    self.assertFalse(
                        root.joinpath(
                            *scoring.SCORING_OUTPUT_ROOT_RELATIVE_PATH.parts,
                            TARGET_DATE,
                        ).exists()
                    )

    def test_final_checkpoint_is_exactly_noon_utc(self) -> None:
        """Le dernier creneau est ferme avant 12:00:00Z inclusivement."""
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            with self.assertRaises(shadow.ShadowPredictionError):
                scoring.reserve_scoring_observation_slot(
                    self._authority(),
                    "2026-09-27",
                    "2026-10-12",
                    reserved_at_utc="2026-10-12T11:59:59Z",
                    project_directory=root,
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            result = scoring.reserve_scoring_observation_slot(
                self._authority(),
                "2026-09-27",
                "2026-10-12",
                reserved_at_utc="2026-10-12T12:00:00Z",
                project_directory=root,
            )
            self.assertTrue((result.slot_path / "RESERVED").is_file())

    def test_symlinked_output_root_is_rejected_without_publication(self) -> None:
        """Une substitution du dossier de sortie ne peut pas detourner RESERVED."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            outside.mkdir()
            output = root.joinpath(*scoring.SCORING_OUTPUT_ROOT_RELATIVE_PATH.parts)
            output.parent.mkdir(parents=True)
            try:
                output.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("Les liens symboliques ne sont pas disponibles.")
            with self.assertRaises(shadow.ShadowPredictionError):
                self._reserve(root)
            self.assertEqual(list(outside.iterdir()), [])

    def test_registration_fixture_is_fully_cross_validated(self) -> None:
        """Le recu et la preuve distante doivent rester lies octet par octet."""
        record, raw_gzip = self._remote_fixture()
        raw_sha256 = hashlib.sha256(raw_gzip).hexdigest()
        with mock.patch.object(
            scoring,
            "EXPECTED_REGISTRATION_REMOTE_EVIDENCE_SHA256",
            raw_sha256,
        ):
            scoring._validate_published_registration(record, raw_gzip)
            changed = dict(record)
            changed["remote_publication_lead_minutes"] = "65.766668"
            with self.assertRaises(shadow.ShadowPredictionError):
                scoring._validate_published_registration(changed, raw_gzip)

    def test_wrong_authority_is_rejected_before_slot_access(self) -> None:
        """Une autorite forgee n'approche jamais l'arbre des observations."""
        authority = self._authority()
        forged = scoring.ScoringExecutionAuthority(
            protocol=authority.protocol,
            registration=authority.registration,
            scoring_protocol_sha256=authority.scoring_protocol_sha256,
            protocol_introduction_commit=authority.protocol_introduction_commit,
            registration_commit=authority.registration_commit,
            registration_sha256="0" * 64,
            registration_remote_evidence_sha256=(
                authority.registration_remote_evidence_sha256
            ),
            runtime_code_commit=authority.runtime_code_commit,
            scoring_engine_sha256=authority.scoring_engine_sha256,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            with self.assertRaises(shadow.ShadowPredictionError):
                scoring.reserve_scoring_observation_slot(
                    forged,
                    TARGET_DATE,
                    CHECKPOINT_DATE,
                    reserved_at_utc=RESERVED_AT,
                    project_directory=root,
                )
            self.assertFalse(
                root.joinpath(
                    *scoring.SCORING_OUTPUT_ROOT_RELATIVE_PATH.parts,
                    TARGET_DATE,
                ).exists()
            )

    def test_module_has_no_sqlite_or_model_dependency(self) -> None:
        """La capture MLB n'autorise toujours ni SQLite ni modele."""
        source = registration.PROJECT_DIRECTORY.joinpath(
            *scoring.SCORING_ENGINE_RELATIVE_PATH.parts
        )
        tree = ast.parse(source.read_bytes(), filename=str(source))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        forbidden = {
            "sqlite3",
            "joblib",
            "src.database",
            "src.mlb_api",
            "src.ingestion_service",
        }
        self.assertTrue(imported.isdisjoint(forbidden))


if __name__ == "__main__":
    unittest.main()
