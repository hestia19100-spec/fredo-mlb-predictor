from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NHL_SOURCE_ROOT = PROJECT_ROOT / "src" / "nhl"
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "nhl" / "fixtures" / "source_registry"
PROTOCOL_PATH = PROJECT_ROOT / "nhl_protocols" / "data" / "nhl_source_registry_v1.json"


class NHL02BoundaryTests(unittest.TestCase):
    def test_new_runtime_has_no_network_or_mlb_import(self) -> None:
        forbidden = (
            "aiohttp",
            "http.client",
            "httpx",
            "requests",
            "socket",
            "urllib",
            "src.database",
            "src.mlb_api",
            "src.shadow_prediction",
            "src.shadow_scoring",
        )
        for path in sorted(NHL_SOURCE_ROOT.glob("*.py")):
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

    def test_all_committed_fixtures_are_synthetic_and_contain_no_secret(self) -> None:
        paths = sorted(FIXTURE_ROOT.glob("*.json"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            text = path.read_text(encoding="utf-8")
            document = json.loads(text)
            with self.subTest(path=path.name):
                self.assertEqual(
                    document["fixture_marker"],
                    "SYNTHETIC_OFFLINE_FIXTURE",
                )
                self.assertNotIn("api_key", text.lower())
                self.assertNotIn("authorization", text.lower())
                self.assertNotIn("bearer ", text.lower())

    def test_protocol_enables_no_runtime_or_provider(self) -> None:
        document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertFalse(document["data_policy"]["real_provider_calls_allowed"])
        self.assertFalse(document["data_policy"]["http_client_enabled"])
        self.assertFalse(document["data_policy"]["database_enabled"])
        self.assertFalse(document["data_policy"]["model_enabled"])
        self.assertFalse(document["data_policy"]["interface_enabled"])
        self.assertFalse(document["data_policy"]["odds_enabled"])
        self.assertTrue(
            all(not provider["enabled"] for provider in document["providers"])
        )

    def test_nhl02_files_stay_inside_nhl_boundaries(self) -> None:
        expected = {
            "nhl_protocols/data/nhl_source_registry_v1.json",
            "src/nhl/offline_sources.py",
            "src/nhl/source_registry.py",
            "tests/nhl/fixtures/source_registry/player_availability_valid.json",
            "tests/nhl/fixtures/source_registry/pregame_goalies_valid.json",
            "tests/nhl/fixtures/source_registry/public_final_result_valid.json",
            "tests/nhl/fixtures/source_registry/public_game_state_valid.json",
            "tests/nhl/fixtures/source_registry/public_schedule_valid.json",
            "tests/nhl/fixtures/source_registry/stats_team_history_valid.json",
            "tests/nhl/test_nhl02_boundaries.py",
            "tests/nhl/test_offline_sources.py",
            "tests/nhl/test_source_registry.py",
        }
        actual = {
            path.relative_to(PROJECT_ROOT).as_posix()
            for root in (
                PROJECT_ROOT / "src" / "nhl",
                PROJECT_ROOT / "tests" / "nhl",
                PROJECT_ROOT / "nhl_protocols" / "data",
            )
            for path in root.rglob("*")
            if path.is_file()
            and (
                path.name in {
                    "offline_sources.py",
                    "source_registry.py",
                    "test_nhl02_boundaries.py",
                    "test_offline_sources.py",
                    "test_source_registry.py",
                    "nhl_source_registry_v1.json",
                }
                or "fixtures/source_registry" in path.as_posix()
            )
        }
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
