"""NHL-10 remains an offline diagnostic, not a prediction pipeline."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "nhl_protocols" / "data" / "nhl_dataset_audit_protocol_v1.json"
SOURCE_AUDIT = ROOT / "nhl_protocols" / "data" / "nhl_source_audit_v1.json"
MODULE = ROOT / "src" / "nhl" / "dataset_audit.py"


class NHL10Boundaries(unittest.TestCase):
    def test_protocol_keeps_real_collection_and_predictions_disabled(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        rights = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(protocol["stage"], "NHL-10")
        self.assertEqual(protocol["base_commit"],
                         "60d808be17343408d0858487bb6ca99a23f615ed")
        for key in (
            "real_provider_activation", "api_requests", "model_training",
            "probability_generation", "prediction_publication", "wagering",
            "mlb_database_mutation", "nhl_database_mutation",
        ):
            self.assertIs(protocol["scope"][key], False)
        self.assertIsNone(rights["decision"]["active_provider"])
        self.assertIs(rights["decision"]["first_real_call_allowed"], False)

    def test_audit_module_imports_no_provider_client_or_model(self) -> None:
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertFalse(imported & {
            "requests", "httpx", "urllib", "socket", "subprocess",
            "sklearn", "xgboost", "lightgbm", "mlb_api",
        })
        self.assertFalse((ROOT / "data" / "nhl.db").exists())


if __name__ == "__main__":
    unittest.main()
