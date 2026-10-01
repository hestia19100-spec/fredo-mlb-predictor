"""NHL-06 must stay offline, read-only, and separate from MLB."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "nhl" / "team_history_features.py"
PROTOCOL = ROOT / "nhl_protocols" / "data" / "nhl_team_history_protocol_v1.json"


class NHL06BoundaryTests(unittest.TestCase):
    def test_no_network_model_mlb_or_interface_imports(self) -> None:
        forbidden = (
            "aiohttp", "http", "httpx", "requests", "socket", "urllib",
            "streamlit", "src.database", "src.mlb_api", "src.shadow_prediction",
            "src.shadow_scoring", "sklearn", "xgboost",
        )
        imports = []
        for node in ast.walk(ast.parse(MODULE.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertFalse(any(
            name == prefix or name.startswith(prefix + ".")
            for name in imports for prefix in forbidden
        ), imports)

    def test_protocol_requires_isolation_and_explicit_missingness(self) -> None:
        document = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(document["protocol_id"], "nhl_team_history_features_v1")
        self.assertEqual(document["stage"], "NHL-06")
        scope = document["scope"]
        self.assertEqual(scope["input"], "Synthetic offline NHL fixture archive only")
        for key in (
            "real_provider_activation", "api_requests", "model_training",
            "prediction_publication", "wagering", "mlb_database_mutation",
            "nhl_database_mutation",
        ):
            self.assertFalse(scope[key], key)
        self.assertIn("never zero-imputed", document["aggregations"]["missing"])
        self.assertEqual(document["aggregations"]["minimum_sample_per_team"], 5)


if __name__ == "__main__":
    unittest.main()
