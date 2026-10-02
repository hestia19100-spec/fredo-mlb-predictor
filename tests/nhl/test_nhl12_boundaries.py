"""Garde-fous du protocole descriptif NHL-12."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.nhl.moneypuck_five_season import (
    EXPECTED_REGULAR_GAMES_PER_SEASON,
    REFERENCE_SEASONS,
    REQUIRED_SITUATIONS,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "nhl_protocols" / "data" / "nhl_five_season_coverage_protocol_v1.json"


class NHL12BoundariesTests(unittest.TestCase):
    def test_reference_window_and_coverage_agree_with_code(self) -> None:
        payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(payload["stage"], "NHL-12")
        self.assertEqual(tuple(payload["reference_season_start_years"]), REFERENCE_SEASONS)
        self.assertEqual(payload["expected_regular_games_per_season"], EXPECTED_REGULAR_GAMES_PER_SEASON)
        self.assertEqual(set(payload["required_team_situations"]), REQUIRED_SITUATIONS)
        self.assertEqual(payload["game_types"]["regular"], "02")
        self.assertEqual(payload["game_types"]["playoffs"], "03")

    def test_prospective_safety_stays_closed(self) -> None:
        payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertTrue(payload["coverage_is_not_asof_proof"])
        for key in ("training_permitted", "prediction_publication_permitted", "network_access", "database_mutation"):
            self.assertIs(payload[key], False, key)


if __name__ == "__main__":
    unittest.main()
