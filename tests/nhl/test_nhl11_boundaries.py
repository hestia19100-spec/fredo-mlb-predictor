"""NHL-11: the MoneyPuck CSV reader is deliberately non-operational."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.nhl.moneypuck_team_import import TEAM_CODE


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "nhl_protocols/data/nhl_moneypuck_import_protocol_v1.json"


class NHL11BoundariesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    def test_only_local_offline_read_is_enabled(self) -> None:
        scope = self.protocol["scope"]
        self.assertEqual(self.protocol["stage"], "NHL-11")
        self.assertTrue(scope["local_downloaded_csv_only"])
        for key in (
            "scraping", "api_calls_in_application", "nhl_database_mutation",
            "mlb_database_mutation", "model_training", "probability_generation",
            "prediction_publication", "wagering",
        ):
            with self.subTest(key=key):
                self.assertIs(scope[key], False)

    def test_no_retroactive_asof_claim(self) -> None:
        checks = self.protocol["checks"]
        self.assertTrue(checks["observed_at_utc_captured_after_successful_import"])
        self.assertFalse(checks["historical_asof_inferred_from_game_date"])
        self.assertFalse(checks["postgame_stats_can_be_used_before_import_time"])

    def test_source_and_team_codes(self) -> None:
        source = self.protocol["source"]
        self.assertEqual(source["page"], "https://moneypuck.com/data.htm")
        self.assertIn("MoneyPuck.com", source["attribution"])
        self.assertEqual(source["intended_use"], "PERSONAL_NONCOMMERCIAL_USER_CONFIRMED")
        self.assertIsNotNone(TEAM_CODE.fullmatch("T.B"))
        self.assertIsNotNone(TEAM_CODE.fullmatch("BOS"))
        self.assertIsNone(TEAM_CODE.fullmatch("..."))


if __name__ == "__main__":
    unittest.main()
