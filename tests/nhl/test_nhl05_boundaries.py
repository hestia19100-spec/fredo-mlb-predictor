"""Garanties de périmètre : NHL-05 ne lance ni réseau ni modèle MLB."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "nhl" / "asof_dataset.py"
PROTOCOL = ROOT / "nhl_protocols" / "data" / "nhl_asof_view_protocol_v1.json"


class NHL05BoundaryTests(unittest.TestCase):
    def test_no_network_mlb_model_or_interface_imports(self) -> None:
        forbidden = (
            "aiohttp", "http", "httpx", "requests", "socket", "urllib",
            "streamlit", "src.database", "src.mlb_api", "src.shadow_prediction",
            "src.shadow_scoring", "sklearn",
        )
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertFalse(
            any(name == prefix or name.startswith(prefix + ".")
                for name in imports for prefix in forbidden),
            imports,
        )

    def test_protocol_keeps_real_runtime_disabled(self) -> None:
        document = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(document["protocol_id"], "lpf_edge_nhl_asof_view_v1")
        self.assertEqual(document["status"], "OFFLINE_SYNTHETIC_ASOF_ONLY")
        safety = document["safety"]
        self.assertTrue(safety["synthetic_offline_fixtures_only"])
        for name in (
            "provider_calls_allowed", "mlb_database_access_allowed",
            "database_mutation_allowed", "model_enabled", "odds_enabled",
            "interface_enabled", "real_bet_execution_enabled",
        ):
            self.assertFalse(safety[name], name)
        rules = document["asof_rules"]
        self.assertTrue(rules["historical_statistic_requires_source_final_by_cutoff"])
        self.assertTrue(rules["storage_audit_before_read"])
        self.assertTrue(rules["storage_read_only"])
        self.assertEqual(rules["unknown_value_policy"], "EXPLICIT_NULL_NO_IMPUTATION")


if __name__ == "__main__":
    unittest.main()
