"""Offline tests for the sealed NHL three-outcome market inspection."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from src.nhl.odds_api_candidates import (
    FRENCH_BOOKMAKER_KEYS, parse_nhl_h2h_odds,
)
from src.nhl.three_outcome_market import (
    NHLThreeOutcomeInspectionError, inspect_archived_three_outcome_market,
)

EVENT_ID = "a" * 32
START = "2026-10-10T23:00:00Z"
OBSERVED = "2026-10-10T20:00:00Z"


def _market(outcomes, *, updated="2026-10-10T19:50:00Z"):
    return {"key": "h2h", "last_update": updated, "outcomes": outcomes}


def _book(key, outcomes, *, updated="2026-10-10T19:50:00Z"):
    return {"key": key, "markets": [_market(outcomes, updated=updated)]}


def _teams():
    return [{"name": "Boston Bruins", "price": 2.1},
            {"name": "New York Rangers", "price": 1.8}]


def _three():
    return [{"name": "Boston Bruins", "price": 2.3},
            {"name": "Draw", "price": 3.4},
            {"name": "New York Rangers", "price": 2.8}]


class ThreeOutcomeMarketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.slot = Path(self.temp.name) / "capture"
        self.slot.mkdir()
        self.event = {
            "id": EVENT_ID, "sport_key": "icehockey_nhl",
            "commence_time": START, "away_team": "Boston Bruins",
            "home_team": "New York Rangers",
        }
        self.books = [_book("netbet_fr", _teams()),
                      _book("betclic_fr", _three()),
                      _book("unibet_fr", _teams())]

    def archive(self, *, observed=OBSERVED):
        events = json.dumps([self.event]).encode()
        odds = json.dumps([{**self.event, "bookmakers": self.books}]).encode()
        parsed = parse_nhl_h2h_odds(odds)
        counts = {key: {"two_team_outcomes": 0, "three_way": 0, "other": 0}
                  for key in FRENCH_BOOKMAKER_KEYS}
        for _, key, shape in parsed.observed_market_shapes:
            counts[key][shape] += 1
        receipt = {
            "schema_version": "nhl_odds_api_capture_only_v2",
            "status": "CAPTURE_ONLY_NOT_MODEL_ELIGIBLE",
            "market_requested": "h2h",
            "bookmakers_requested": list(FRENCH_BOOKMAKER_KEYS),
            "training_permitted": False,
            "prediction_publication_permitted": False,
            "observed_at_utc": observed,
            "events_sha256": sha256(events).hexdigest(),
            "odds_sha256": sha256(odds).hexdigest(),
            "event_count": 1,
            "two_way_quote_count": len(parsed.two_way_quotes),
            "rejected_market_count": len(parsed.rejected_market_keys),
            "market_shape_counts": counts,
        }
        (self.slot / "events.json").write_bytes(events)
        (self.slot / "odds.json").write_bytes(odds)
        (self.slot / "receipt.json").write_text(json.dumps(receipt))
        (self.slot / "COMPLETED").touch()
        return receipt

    def test_three_outcomes_are_shape_only_not_approved_draw_bets(self):
        self.archive()
        report = inspect_archived_three_outcome_market(self.slot)
        self.assertEqual(len(report.candidates), 1)
        quote = report.candidates[0]
        self.assertEqual((quote.bookmaker_key, quote.third_outcome_label),
                         ("betclic_fr", "Draw"))
        self.assertEqual((quote.away_decimal_odds, quote.third_decimal_odds,
                          quote.home_decimal_odds),
                         (Decimal("2.3"), Decimal("3.4"), Decimal("2.8")))
        self.assertFalse(quote.regulation_draw_verified)
        self.assertFalse(report.training_permitted)
        self.assertFalse(report.prediction_publication_permitted)
        self.assertEqual(report.late_three_outcome_shapes, 0)
        self.assertEqual(report.market_shape_counts[0], ("betclic_fr", 0, 1, 0))
        self.assertEqual(report.market_shape_counts[1], ("netbet_fr", 1, 0, 0))

    def test_observation_after_one_hour_cutoff_cannot_be_candidate(self):
        self.archive(observed="2026-10-10T22:00:01Z")
        report = inspect_archived_three_outcome_market(self.slot)
        self.assertEqual(report.candidates, ())
        self.assertEqual(report.late_three_outcome_shapes, 1)

    def test_exact_one_hour_cutoff_is_admitted(self):
        self.archive(observed="2026-10-10T22:00:00Z")
        self.assertEqual(len(inspect_archived_three_outcome_market(self.slot).candidates), 1)

    def test_tampered_raw_archive_fails_closed(self):
        self.archive()
        with (self.slot / "odds.json").open("ab") as output:
            output.write(b" ")
        with self.assertRaisesRegex(NHLThreeOutcomeInspectionError, "Empreinte"):
            inspect_archived_three_outcome_market(self.slot)

    def test_missing_completion_marker_fails_closed(self):
        self.archive()
        (self.slot / "COMPLETED").unlink()
        with self.assertRaisesRegex(NHLThreeOutcomeInspectionError, "non clôturée"):
            inspect_archived_three_outcome_market(self.slot)

    def test_inconsistent_shape_manifest_fails_closed(self):
        receipt = self.archive()
        receipt["market_shape_counts"]["betclic_fr"]["three_way"] = 0
        (self.slot / "receipt.json").write_text(json.dumps(receipt))
        with self.assertRaisesRegex(NHLThreeOutcomeInspectionError, "Comptage"):
            inspect_archived_three_outcome_market(self.slot)

    def test_malformed_third_price_fails_offline_without_new_request(self):
        self.books[1]["markets"][0]["outcomes"][1]["price"] = "bad"
        self.archive()
        with self.assertRaisesRegex(NHLThreeOutcomeInspectionError, "incompatible"):
            inspect_archived_three_outcome_market(self.slot)

    def test_future_dated_market_fails_closed(self):
        self.books[1]["markets"][0]["last_update"] = "2026-10-10T20:00:01Z"
        self.archive()
        with self.assertRaisesRegex(NHLThreeOutcomeInspectionError, "futur"):
            inspect_archived_three_outcome_market(self.slot)


if __name__ == "__main__":
    unittest.main()
