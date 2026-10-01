from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_MODULE = PROJECT_ROOT / "src" / "nhl" / "database.py"
REPOSITORY_MODULE = PROJECT_ROOT / "src" / "nhl" / "repository.py"
PROTOCOL_PATH = (
    PROJECT_ROOT / "nhl_protocols" / "data" / "nhl_storage_protocol_v1.json"
)


class NHL04BoundaryTests(unittest.TestCase):
    def test_storage_modules_have_no_network_model_interface_or_mlb_import(self) -> None:
        forbidden = (
            "aiohttp",
            "http",
            "httpx",
            "requests",
            "socket",
            "streamlit",
            "urllib",
            "src.database",
            "src.mlb_api",
            "src.shadow_prediction",
            "src.shadow_scoring",
        )
        for path in (DATABASE_MODULE, REPOSITORY_MODULE):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            with self.subTest(path=path.name):
                self.assertFalse(
                    any(
                        name == prefix or name.startswith(prefix + ".")
                        for name in imports
                        for prefix in forbidden
                    ),
                    imports,
                )

    def test_import_has_no_database_creation_expression(self) -> None:
        tree = ast.parse(DATABASE_MODULE.read_text(encoding="utf-8"))
        top_level_calls = [
            node
            for node in tree.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        ]
        self.assertEqual(top_level_calls, [])

    def test_protocol_enables_storage_but_no_real_runtime(self) -> None:
        document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(document["status"], "OFFLINE_SYNTHETIC_STORAGE_ONLY")
        self.assertFalse(document["safety"]["provider_calls_allowed"])
        self.assertFalse(document["safety"]["mlb_database_access_allowed"])
        self.assertFalse(document["safety"]["model_enabled"])
        self.assertFalse(document["safety"]["interface_enabled"])
        self.assertTrue(document["safety"]["synthetic_offline_fixtures_only"])

    def test_default_nhl_and_mlb_databases_are_distinct(self) -> None:
        text = DATABASE_MODULE.read_text(encoding="utf-8")
        self.assertIn('"data" / "nhl"', text)
        self.assertIn('"fredo_nhl.db"', text)
        self.assertIn('"fredo_mlb.db"', text)
        self.assertNotIn("from src.database", text)

    def test_nhl04_exact_file_boundary(self) -> None:
        self.assertEqual(
            {
                "nhl_protocols/data/nhl_storage_protocol_v1.json",
                "src/nhl/database.py",
                "src/nhl/repository.py",
                "tests/nhl/test_database.py",
                "tests/nhl/test_nhl04_boundaries.py",
                "tests/nhl/test_repository.py",
            },
            {
                "nhl_protocols/data/nhl_storage_protocol_v1.json",
                "src/nhl/database.py",
                "src/nhl/repository.py",
                "tests/nhl/test_database.py",
                "tests/nhl/test_nhl04_boundaries.py",
                "tests/nhl/test_repository.py",
            },
        )


if __name__ == "__main__":
    unittest.main()
