"""Offline integration tests for budgeted NHL capture-only operations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.nhl_odds_api_daily import (
    SCORES_URL,
    capture_nhl_daily_pregame,
    capture_nhl_daily_scores,
)
from src.nhl_odds_api_quota import EVENTS_URL, ODDS_URL


KEY = "super-secret-value"
NOW = datetime(2026, 10, 9, 18, tzinfo=timezone.utc)
FUTURE_ID = "a" * 32
PAST_ID = "b" * 32


def event(event_id: str, start: str) -> dict:
    return {"id": event_id, "sport_key": "icehockey_nhl", "commence_time": start,
            "home_team": "Home", "away_team": "Away"}


class Response:
    def __init__(self, url: str, body: list[dict], cost: int):
        self.url = url + "?apiKey=" + KEY
        self.status_code = 200
        self.headers = {"Content-Type": "application/json", "x-requests-last": str(cost),
                        "x-requests-used": str(cost), "x-requests-remaining": str(500 - cost)}
        self.content = json.dumps(body).encode("utf-8")


class DailyCaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ledger = self.root / "quota.sqlite"
        self.calls: list[str] = []
        self.future = event(FUTURE_ID, "2026-10-10T23:00:00Z")
        self.past = event(PAST_ID, "2026-10-09T01:00:00Z")

    def transport(self, url, *, params, timeout, allow_redirects):
        self.calls.append(url)
        if url == EVENTS_URL:
            return Response(url, [self.future], 0)
        if url == ODDS_URL:
            odds = {**self.future, "bookmakers": [{"key": "netbet_fr", "markets": [{
                "key": "h2h", "last_update": "2026-10-09T17:00:00Z",
                "outcomes": [{"name": "Home", "price": 1.8}, {"name": "Away", "price": 2.1}],
            }]}]}
            return Response(url, [odds], 1)
        if url == SCORES_URL:
            final = {**self.past, "completed": True,
                     "last_update": "2026-10-09T04:00:00Z",
                     "scores": [{"name": "Home", "score": "3"},
                                {"name": "Away", "score": "2"}]}
            return Response(url, [final], 2)
        raise AssertionError("Unexpected URL")

    def pregame(self):
        with patch("src.nhl_odds_api_capture._read_api_key", return_value=KEY):
            return capture_nhl_daily_pregame(
                ledger_path=self.ledger, root=self.root / "pregame",
                transport=self.transport, now=lambda: NOW,
            )

    def scores(self):
        with patch("src.nhl_odds_api_daily._read_api_key", return_value=KEY):
            return capture_nhl_daily_scores(
                ledger_path=self.ledger, root=self.root / "scores",
                transport=self.transport, now=lambda: NOW,
            )

    def test_budgeted_daily_cycle_archives_without_model_eligibility(self):
        pregame = self.pregame()
        scores = self.scores()
        self.assertEqual(self.calls, [EVENTS_URL, ODDS_URL, SCORES_URL])
        self.assertEqual((pregame.odds_quota_cost, scores.quota_cost, scores.final_count), (1, 2, 1))
        for slot in (pregame.path, scores.path):
            self.assertTrue((slot / "COMPLETED").is_file())
            receipt_raw = (slot / "receipt.json").read_bytes()
            self.assertNotIn(KEY.encode(), receipt_raw)
            receipt = json.loads(receipt_raw)
            self.assertFalse(receipt["training_permitted"])
            self.assertFalse(receipt["prediction_publication_permitted"])
        with sqlite3.connect(self.ledger) as connection:
            self.assertEqual(connection.execute(
                "SELECT SUM(COALESCE(actual_cost, reserved_cost)) FROM requests",
            ).fetchone(), (3,))

    def test_repeat_scores_cannot_consume_more_credits(self):
        self.scores()
        with self.assertRaisesRegex(Exception, "Collecte NHL échouée"):
            self.scores()
        self.assertEqual(self.calls, [SCORES_URL])

    def test_ledger_reservation_blocks_after_uncertain_transport(self):
        def broken(url, **kwargs):
            raise RuntimeError("?apiKey=" + KEY)
        with patch("src.nhl_odds_api_daily._read_api_key", return_value=KEY):
            with self.assertRaises(Exception) as caught:
                capture_nhl_daily_scores(
                    ledger_path=self.ledger, root=self.root / "scores",
                    transport=broken, now=lambda: NOW,
                )
        self.assertNotIn(KEY, str(caught.exception))
        with sqlite3.connect(self.ledger) as connection:
            self.assertEqual(connection.execute(
                "SELECT reserved_cost, actual_cost FROM requests WHERE kind = 'scores'",
            ).fetchone(), (2, None))


if __name__ == "__main__":
    unittest.main()
