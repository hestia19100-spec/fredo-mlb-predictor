"""Garde-fous NHL-14 : résultats rétrospectifs non prospectifs."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

from src.nhl.moneypuck_five_season import REFERENCE_SEASONS

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "nhl_protocols/data/nhl_retrospective_label_protocol_v1.json"
MODULE = ROOT / "src/nhl/retrospective_labels.py"


class NHL14BoundariesTests(unittest.TestCase):
    def test_protocol_tracks_the_five_season_join(self) -> None:
        policy = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(policy["stage"], "NHL-14")
        self.assertEqual(policy["protocol_id"],
                         "lpf_edge_nhl_retrospective_regular_labels_v1")
        self.assertEqual(policy["input_protocol"], "lpf_edge_nhl_lagged_history_v1")
        self.assertEqual(tuple(policy["reference_season_start_years"]), REFERENCE_SEASONS)
        self.assertEqual(policy["game_type"], "02")
        self.assertEqual(policy["score_situation"], "all")
        self.assertTrue(policy["playoffs_excluded"])
        self.assertTrue(policy["feature_label_separation"])
        self.assertTrue(policy["candidate_features_recomputed_from_source"])

    def test_availability_and_operational_gates_remain_closed(self) -> None:
        policy = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        for key in (
            "historical_pregame_availability_proven", "training_permitted",
            "prediction_publication_permitted", "network_access", "database_mutation",
        ):
            self.assertIs(policy[key], False, key)

    def test_join_has_no_provider_model_or_database_client(self) -> None:
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        imports = [node for node in ast.walk(tree)
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = {alias.name.split(".")[0] for node in imports for alias in node.names}
        self.assertFalse(names & {
            "requests", "httpx", "urllib", "sqlite3", "sklearn", "joblib"
        })


if __name__ == "__main__":
    unittest.main()
