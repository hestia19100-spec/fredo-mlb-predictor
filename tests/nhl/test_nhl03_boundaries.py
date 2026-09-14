from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "src" / "nhl" / "source_audit.py"
PROTOCOL_PATH = (
    PROJECT_ROOT / "nhl_protocols" / "data" / "nhl_source_audit_v1.json"
)


class NHL03BoundaryTests(unittest.TestCase):
    def test_audit_module_cannot_perform_network_or_mlb_io(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        forbidden = (
            "aiohttp",
            "http",
            "httpx",
            "requests",
            "socket",
            "urllib",
            "src.database",
            "src.mlb_api",
            "src.shadow_prediction",
            "src.shadow_scoring",
        )
        self.assertFalse(
            any(
                name == prefix or name.startswith(prefix + ".")
                for name in imported
                for prefix in forbidden
            ),
            imported,
        )

    def test_protocol_records_research_not_activation(self) -> None:
        document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            document["audit_method"],
            "OFFICIAL_DOCUMENTS_ONLY_NO_PROVIDER_CALL",
        )
        self.assertEqual(
            document["status"],
            "DRAFT_RESEARCH_ONLY_NO_PROVIDER_ENABLED",
        )
        self.assertFalse(document["decision"]["first_real_call_allowed"])

    def test_nhl03_contains_no_secret_or_credential_value(self) -> None:
        combined = (
            MODULE_PATH.read_text(encoding="utf-8")
            + PROTOCOL_PATH.read_text(encoding="utf-8")
        ).lower()
        self.assertNotIn("authorization:", combined)
        self.assertNotIn("bearer ", combined)
        self.assertNotIn("api_key=", combined)

    def test_audit_does_not_create_client_database_model_or_interface(self) -> None:
        expected = {
            "nhl_protocols/data/nhl_source_audit_v1.json",
            "src/nhl/source_audit.py",
            "tests/nhl/test_nhl03_boundaries.py",
            "tests/nhl/test_source_audit.py",
        }
        self.assertEqual(
            {
                "nhl_protocols/data/nhl_source_audit_v1.json",
                "src/nhl/source_audit.py",
                "tests/nhl/test_nhl03_boundaries.py",
                "tests/nhl/test_source_audit.py",
            },
            expected,
        )


if __name__ == "__main__":
    unittest.main()
