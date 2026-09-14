from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "nhl_protocols"
    / "data"
    / "nhl_data_foundation_v1.json"
)
EXECUTION_MANIFEST_PATH = (
    PROJECT_ROOT
    / "shadow_protocols"
    / "logistic_team_form_v1_platt_shadow_v2_execution.json"
)


class FoundationBoundaryTests(unittest.TestCase):
    def test_protocol_is_local_draft_without_active_runtime(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(protocol["protocol_schema_version"], 1)
        self.assertEqual(
            protocol["status"],
            "DRAFT_LOCAL_BEFORE_FIRST_NHL_API_CALL",
        )
        self.assertEqual(
            protocol["base_repository_commit"],
            "88a0e4312dc0f7779c1a9178497680f68d30e9be",
        )
        self.assertTrue(
            all(value is False for value in protocol["scope"].values())
        )

    def test_foundation_stage_forbids_network_and_real_data(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        policy = protocol["network_policy"]
        self.assertFalse(policy["api_calls_allowed_in_this_stage"])
        self.assertFalse(policy["http_clients_allowed_in_foundation_modules"])
        self.assertTrue(policy["synthetic_test_data_only"])

    def test_nhl_modules_do_not_import_http_or_mlb_runtime(self) -> None:
        forbidden_prefixes = (
            "requests",
            "httpx",
            "urllib",
            "src.mlb_api",
            "src.database",
            "src.game_repository",
            "src.ingestion_service",
            "src.shadow_prediction",
            "src.shadow_scoring",
        )
        for path in sorted((PROJECT_ROOT / "src" / "nhl").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            with self.subTest(path=path.name):
                self.assertFalse(
                    any(
                        name == prefix or name.startswith(prefix + ".")
                        for name in imported
                        for prefix in forbidden_prefixes
                    ),
                    imported,
                )

    def test_mlb_shadow_v2_runtime_hashes_remain_exact(self) -> None:
        manifest = json.loads(
            EXECUTION_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        for relative_path, expected in manifest[
            "transitive_runtime_file_sha256_map"
        ].items():
            actual = hashlib.sha256(
                (PROJECT_ROOT / relative_path).read_bytes()
            ).hexdigest()
            with self.subTest(relative_path=relative_path):
                self.assertEqual(actual, expected)

        requirements_path = PROJECT_ROOT / manifest["requirements_path"]
        requirements_sha256 = hashlib.sha256(
            requirements_path.read_bytes()
        ).hexdigest()
        self.assertEqual(
            requirements_sha256,
            manifest["requirements_sha256"],
        )

    def test_nhl_local_data_root_is_ignored(self) -> None:
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("data/nhl/", gitignore.splitlines())

    def test_secret_policy_never_persists_credentials_or_hashes(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        policy = protocol["secrets_policy"]
        self.assertFalse(policy["credentials_in_code_allowed"])
        self.assertFalse(policy["credentials_in_git_allowed"])
        self.assertFalse(policy["credential_logging_allowed"])
        self.assertFalse(policy["credential_query_parameter_allowed"])
        self.assertFalse(policy["credential_hash_persistence_allowed"])


if __name__ == "__main__":
    unittest.main()
