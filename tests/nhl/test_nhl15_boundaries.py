"""Frontières de sécurité du rapport descriptif NHL-15."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src/nhl/real_archive_audit.py"
PROTOCOL = ROOT / "nhl_protocols/data/nhl_real_archive_audit_protocol_v1.json"


class NHL15BoundariesTests(unittest.TestCase):
    def test_protocol_has_five_seasons_and_closed_gates(self) -> None:
        policy = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(policy["stage"], "NHL-15")
        self.assertEqual(policy["reference_season_start_years"],
                         [2021, 2022, 2023, 2024, 2025])
        self.assertTrue(policy["checks"]["tied_source_scores_unresolved_without_independent_final"])
        for key in ("historical_pregame_availability_proven",
                    "independent_final_labels_proven", "training_permitted",
                    "prediction_publication_permitted", "network_access_in_audit",
                    "database_mutation"):
            self.assertIs(policy[key], False, key)

    def test_auditor_has_no_network_database_or_model_client(self) -> None:
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        imports = [node for node in ast.walk(tree)
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = {alias.name.split(".")[0] for node in imports
                 for alias in node.names}
        self.assertFalse(names & {
            "requests", "httpx", "urllib", "sqlite3", "sklearn", "joblib",
            "xgboost", "lightgbm", "subprocess",
        })


if __name__ == "__main__":
    unittest.main()
