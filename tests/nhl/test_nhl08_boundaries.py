"""NHL-08 remains offline, read-only, and isolated from MLB."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "nhl" / "feature_bundle.py"
PROTOCOL = ROOT / "nhl_protocols" / "data" / "nhl_feature_bundle_protocol_v1.json"


class NHL08BoundaryTests(unittest.TestCase):
    def test_no_network_model_mlb_or_interface_imports(self) -> None:
        forbidden = (
            "aiohttp", "http", "httpx", "requests", "socket", "urllib",
            "streamlit", "src.database", "src.mlb_api", "src.shadow_prediction",
            "src.shadow_scoring", "sklearn", "xgboost",
        )
        imports: list[str] = []
        for node in ast.walk(ast.parse(MODULE.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertFalse(any(
            name == prefix or name.startswith(prefix + ".")
            for name in imports for prefix in forbidden
        ), imports)

    def test_protocol_requires_one_audited_view_and_no_activation(self) -> None:
        document = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(document["protocol_id"], "nhl_feature_bundle_v1")
        self.assertEqual(document["stage"], "NHL-08")
        self.assertEqual(document["scope"]["input"],
                         "Synthetic offline NHL fixture archive only")
        for key in (
            "real_provider_activation", "api_requests", "model_training",
            "probability_generation", "prediction_publication", "wagering",
            "mlb_database_mutation", "nhl_database_mutation",
        ):
            self.assertFalse(document["scope"][key], key)
        for key in (
            "storage_audit_before_read", "storage_read_only",
            "single_view_for_both_feature_families",
            "effective_availability_no_later_than_cutoff",
            "source_game_requires_final_before_cutoff",
            "target_game_state_and_result_excluded",
            "component_snapshot_hashes_must_match",
        ):
            self.assertTrue(document["asof_rules"][key], key)


if __name__ == "__main__":
    unittest.main()
