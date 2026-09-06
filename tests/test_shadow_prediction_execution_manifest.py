"""Verrouille le manifeste d'execution du mode shadow MLB v2."""

from __future__ import annotations

import hashlib
import json
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src import shadow_prediction as shadow


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_DIRECTORY / (
    "shadow_protocols/logistic_team_form_v1_platt_shadow_v2_execution.json"
)
PROTOCOL_PATH = PROJECT_DIRECTORY / (
    "shadow_protocols/logistic_team_form_v1_platt_shadow_v2.json"
)
SERVICE_COMMIT = "983a4306cba7572746b48e16edff12b5874515ff"
SERVICE_SHA256 = (
    "10c1d60c841a676f443cc5a5bc35d14df3e4470a53747b891dee991b8a237aea"
)
PROTOCOL_SHA256 = (
    "4dcae9e85bb9ed5b3f4a9f961491d87256965c42e97d11e70f31f02f52cc49c9"
)
REQUIREMENTS_SHA256 = (
    "2a3db81fcb8a01b32b0d019ef88b15d1658dee53791230ed12c80555531b0fe9"
)
EXPECTED_RUNTIME_HASHES = {
    "src/__init__.py": (
        "03859f9c882e050c4f9c602af8c314e4a1933233a7b68e11f067316946bed821"
    ),
    "src/baseline_model.py": (
        "a571c0b069266120e60ff8360ba4c5763e88d7ae209c14491cf51d9822133fb9"
    ),
    "src/calibrated_model.py": (
        "00d1016fd8149d53ce752bead80b82d84b1afc7f95c3f60f8b8778af58ac2d29"
    ),
    "src/database.py": (
        "d79ae0158760117e8754502cd0d5e31f0755d04fca3cf6bbfab0056ade4f26d5"
    ),
    "src/game_repository.py": (
        "ffb8dd20fb5a84d2c79085c050122152e2bc4f25c890f9e3bbbcba5218c03abd"
    ),
    "src/ingestion_repository.py": (
        "f81c0745f7c344a4e830bd024df55ad5740abe56d9cb56ea94a5a8c07b454507"
    ),
    "src/ingestion_service.py": (
        "94b969787ea930d67b7e4c4d64c7929c8159b592cd5c84b5a47989e9f3b676eb"
    ),
    "src/mlb_api.py": (
        "04bf3e10db2b3533473b1237c1362b12a1ee58bbe3cb9a9040194780fc2a0db6"
    ),
    "src/raw_archive.py": (
        "90926c2254b130c01258ed1424a5dc5202670778104b26de9529eea44a700073"
    ),
    "src/retry_policy.py": (
        "11c2bf33744ce144975fae9301dec49b72f7034f2a1193ba7340f92a0c80eac5"
    ),
    "src/shadow_prediction.py": SERVICE_SHA256,
    "src/training_dataset.py": (
        "9fa64a5ba3f51a619329ab9ba0119c4fca2f3e908effccacd30a3887e0da33f3"
    ),
}


def _load_manifest() -> tuple[bytes, dict[str, object]]:
    content = MANIFEST_PATH.read_bytes()
    return content, json.loads(content.decode("utf-8"))


class ShadowPredictionExecutionManifestTests(unittest.TestCase):
    """Le manifeste doit etre exact avant toute activation shadow v2."""

    def test_manifest_is_canonical_and_has_exact_schema(self) -> None:
        """Les octets et les dix-sept champs sont non ambigus."""
        content, manifest = _load_manifest()
        canonical = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        self.assertEqual(content, canonical)
        self.assertEqual(set(manifest), set(shadow._EXECUTION_MANIFEST_KEYS))

    def test_manifest_binds_exact_frozen_service_and_protocol(self) -> None:
        """Le service valide et le protocole v2 restent les seules autorites."""
        _, manifest = _load_manifest()
        self.assertEqual(manifest["execution_manifest_schema_version"], 1)
        self.assertEqual(
            manifest["status"], "FROZEN_BEFORE_SHADOW_V2_ACTIVATION"
        )
        self.assertEqual(manifest["shadow_service_code_commit"], SERVICE_COMMIT)
        self.assertEqual(
            manifest["shadow_service_module_path"], "src/shadow_prediction.py"
        )
        self.assertEqual(manifest["shadow_service_module_sha256"], SERVICE_SHA256)
        self.assertEqual(
            manifest["shadow_protocol_path"],
            "shadow_protocols/logistic_team_form_v1_platt_shadow_v2.json",
        )
        self.assertEqual(manifest["shadow_protocol_sha256"], PROTOCOL_SHA256)
        self.assertEqual(
            hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
            PROTOCOL_SHA256,
        )

    def test_transitive_runtime_closure_and_hashes_are_exact(self) -> None:
        """Chaque module local chargeable est present une seule fois et intact."""
        _, manifest = _load_manifest()
        runtime_hashes = manifest["transitive_runtime_file_sha256_map"]
        self.assertEqual(runtime_hashes, EXPECTED_RUNTIME_HASHES)
        self.assertEqual(
            tuple(runtime_hashes), tuple(sorted(EXPECTED_RUNTIME_HASHES))
        )
        self.assertEqual(
            shadow._local_python_runtime_closure(PROJECT_DIRECTORY),
            tuple(EXPECTED_RUNTIME_HASHES),
        )
        for relative_path, expected_sha256 in EXPECTED_RUNTIME_HASHES.items():
            actual = hashlib.sha256(
                (PROJECT_DIRECTORY / relative_path).read_bytes()
            ).hexdigest()
            self.assertEqual(actual, expected_sha256, relative_path)

    def test_model_requirements_and_runtime_versions_are_frozen(self) -> None:
        """L'artefact ignore reste identifie et l'environnement est integral."""
        _, manifest = _load_manifest()
        self.assertEqual(
            {
                "model_artifact_path": manifest["model_artifact_path"],
                "model_artifact_sha256": manifest["model_artifact_sha256"],
                "model_artifact_size_bytes": manifest[
                    "model_artifact_size_bytes"
                ],
                "requirements_path": manifest["requirements_path"],
                "requirements_sha256": manifest["requirements_sha256"],
            },
            {
                "model_artifact_path": (
                    "models/logistic_team_form_v1_platt.joblib"
                ),
                "model_artifact_sha256": (
                    "e0d4d2421ba076072c7ef8b3bc97dd9a341e26c62828a0ad9ba43f30da15ff55"
                ),
                "model_artifact_size_bytes": 1589,
                "requirements_path": "requirements.txt",
                "requirements_sha256": REQUIREMENTS_SHA256,
            },
        )
        self.assertEqual(
            hashlib.sha256(
                (PROJECT_DIRECTORY / "requirements.txt").read_bytes()
            ).hexdigest(),
            REQUIREMENTS_SHA256,
        )
        self.assertEqual(
            manifest["runtime_versions"],
            {
                "joblib": "1.5.3",
                "numpy": "2.5.2",
                "pandas": "3.0.5",
                "python": "3.12.1",
                "scikit_learn": "1.9.0",
                "scipy": "1.18.1",
            },
        )

    def test_full_suite_result_is_exact_and_precedes_manifest(self) -> None:
        """Les 683 tests portent exactement sur le commit du service."""
        _, manifest = _load_manifest()
        test_result = manifest["test_suite_result"]
        self.assertEqual(
            test_result,
            {
                "command": "python -m unittest discover -s tests -v",
                "completed_at_utc": "2026-09-06T07:29:14Z",
                "errors": 0,
                "failures": 0,
                "skipped": 0,
                "status": "OK",
                "tested_code_commit": SERVICE_COMMIT,
                "tests_run": 683,
            },
        )
        completed_at = datetime.fromisoformat(
            test_result["completed_at_utc"].replace("Z", "+00:00")
        )
        created_at = datetime.fromisoformat(
            manifest["created_at_utc"].replace("Z", "+00:00")
        )
        self.assertEqual(completed_at.tzinfo, timezone.utc)
        self.assertEqual(created_at.tzinfo, timezone.utc)
        self.assertLessEqual(completed_at, created_at)

    def test_minimum_date_is_explicit_and_leaves_activation_window(self) -> None:
        """Le 10 septembre est la premiere date cible autorisee."""
        _, manifest = _load_manifest()
        minimum_date = date.fromisoformat(manifest["minimum_target_official_date"])
        self.assertEqual(minimum_date, date(2026, 9, 10))
        self.assertGreaterEqual(minimum_date, date(2026, 8, 31))

    def test_service_accepts_every_frozen_manifest_value(self) -> None:
        """Le validateur d'activation accepte exactement ce manifeste."""
        _, manifest = _load_manifest()

        def committed_blob(
            project_directory: Path,
            commit: str,
            relative_path: str,
        ) -> bytes:
            self.assertEqual(project_directory, PROJECT_DIRECTORY)
            self.assertEqual(commit, SERVICE_COMMIT)
            return (PROJECT_DIRECTORY / relative_path).read_bytes()

        with (
            patch.object(
                shadow,
                "_installed_model_runtime_versions",
                return_value=dict(manifest["runtime_versions"]),
            ),
            patch.object(
                shadow,
                "_git_blob_at_commit",
                side_effect=committed_blob,
            ),
            patch.object(shadow, "_require_git_ancestor") as ancestor,
        ):
            result = shadow._validate_execution_manifest(
                manifest,
                project_directory=PROJECT_DIRECTORY,
                protocol_sha256=PROTOCOL_SHA256,
                manifest_introduction_commit="a" * 40,
                runtime_code_commit=SERVICE_COMMIT,
            )

        self.assertEqual(
            result,
            (
                SERVICE_COMMIT,
                SERVICE_SHA256,
                REQUIREMENTS_SHA256,
                tuple(sorted(manifest["runtime_versions"].items())),
                "2026-09-10",
            ),
        )
        self.assertEqual(ancestor.call_count, 3)


if __name__ == "__main__":
    unittest.main()
