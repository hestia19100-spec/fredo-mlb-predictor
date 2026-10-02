"""NHL-09 is offline, read-only, and isolated from MLB and wagering."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "nhl" / "outcome_labels.py"
PROTOCOL = ROOT / "nhl_protocols" / "data" / "nhl_outcome_label_protocol_v1.json"


class NHL09BoundaryTests(unittest.TestCase):
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

    def test_protocol_requires_later_proof_without_activation(self) -> None:
        document = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(document["protocol_id"], "nhl_outcome_label_v1")
        self.assertEqual(document["stage"], "NHL-09")
        self.assertEqual(document["scope"]["input"],
                         "Audited synthetic offline NHL fixture archive only")
        for key in (
            "real_provider_activation", "api_requests", "model_training",
            "probability_generation", "prediction_publication", "wagering",
            "mlb_database_mutation", "nhl_database_mutation",
        ):
            self.assertFalse(document["scope"][key], key)
        for key in (
            "storage_audit_before_read", "storage_read_only",
            "final_proof_must_be_available_by_explicit_label_checkpoint",
            "final_observed_after_scheduled_start",
            "winner_must_match_untied_final_score_and_pregame_teams",
            "missing_final_remains_pending",
            "simultaneous_conflicting_finals_rejected",
            "target_result_excluded_from_pregame_feature_view",
            "pair_hash_links_pregame_and_postgame_proofs",
        ):
            self.assertTrue(document["label_rules"][key], key)


if __name__ == "__main__":
    unittest.main()
