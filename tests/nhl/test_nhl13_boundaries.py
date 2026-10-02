"""Garde-fous du protocole NHL-13 et des candidats historiques."""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from src.nhl.lagged_historical_features import DEFAULT_LOOKBACK_GAMES
from src.nhl.moneypuck_five_season import REFERENCE_SEASONS, REQUIRED_SITUATIONS

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "nhl_protocols" / "data" / "nhl_lagged_history_protocol_v1.json"
MODULE = ROOT / "src" / "nhl" / "lagged_historical_features.py"


class NHL13BoundariesTests(unittest.TestCase):
    def test_protocol_matches_implementation(self) -> None:
        policy = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(policy["stage"], "NHL-13")
        self.assertEqual(policy["input_protocol"], "lpf_edge_nhl_five_season_coverage_v1")
        self.assertEqual(tuple(policy["reference_season_start_years"]), REFERENCE_SEASONS)
        self.assertEqual(set(policy["required_situations"]), REQUIRED_SITUATIONS)
        self.assertEqual(policy["default_lookback_games"], DEFAULT_LOOKBACK_GAMES)
        self.assertEqual(policy["target_game_type"], "02")
        self.assertEqual(policy["history_date_rule"], "source_game_date < target_game_date")

    def test_prospective_and_mutation_gates_remain_closed(self) -> None:
        policy = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        for key in (
            "labels_included", "source_receipt_is_historical_asof_proof",
            "training_permitted", "prediction_publication_permitted",
            "network_access", "database_mutation",
        ):
            self.assertIs(policy[key], False, key)
        self.assertTrue(policy["same_day_games_excluded"])
        self.assertTrue(policy["playoffs_excluded"])

    def test_builder_has_no_network_database_or_model_client(self) -> None:
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = {alias.name.split(".")[0] for node in imports for alias in node.names}
        self.assertFalse(names & {"requests", "httpx", "urllib", "sqlite3", "sklearn", "joblib"})


if __name__ == "__main__":
    unittest.main()
