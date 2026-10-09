"""Capture-only integration of the NHL candidate events and two-way prices."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.nhl_odds_api_capture import (
    EVENTS_URL,
    ODDS_URL,
    NHLOddsCaptureError,
    capture_nhl_odds_candidates,
)


EVENT_ID = "a" * 32
START = "2026-10-10T23:00:00Z"
OBSERVED = datetime(2026, 10, 9, 18, tzinfo=timezone.utc)


def _event() -> dict:
    return {"id": EVENT_ID, "sport_key": "icehockey_nhl", "commence_time": START,
            "home_team": "Home", "away_team": "Away"}


def _odds() -> dict:
    return {**_event(), "bookmakers": [{"key": "netbet_fr", "markets": [{
        "key": "h2h", "last_update": "2026-10-09T17:00:00Z",
        "outcomes": [{"name": "Home", "price": 1.8}, {"name": "Away", "price": 2.1}],
    }]}]}


class Response:
    def __init__(self, url: str, body: list[dict], cost: int, **changes):
        self.url = changes.get("url", url + "?apiKey=super-secret-value")
        self.status_code = changes.get("status_code", 200)
        self.headers = {"Content-Type": "application/json", "x-requests-last": str(cost),
                        "x-requests-remaining": "499", "x-requests-used": "1"}
        self.headers.update(changes.get("headers", {}))
        self.content = json.dumps(body).encode("utf-8")


class OddsCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls: list[tuple[str, dict, int, bool]] = []
        self.events = [_event()]
        self.odds = [_odds()]

    def transport(self, url, *, params, timeout, allow_redirects):
        self.calls.append((url, params, timeout, allow_redirects))
        if url == EVENTS_URL:
            return Response(url, self.events, 0)
        if url == ODDS_URL:
            return Response(url, self.odds, 1)
        raise AssertionError("Unexpected URL")

    def capture(self):
        with patch("src.nhl_odds_api_capture._read_api_key", return_value="super-secret-value"):
            return capture_nhl_odds_candidates(
                root=self.root, transport=self.transport, now=lambda: OBSERVED,
            )

    def test_two_requests_and_append_only_receipt_without_secret(self):
        result = self.capture()
        self.assertEqual((result.event_count, result.two_way_quote_count, result.odds_quota_cost), (1, 1, 1))
        self.assertEqual([call[0] for call in self.calls], [EVENTS_URL, ODDS_URL])
        self.assertEqual([call[2:] for call in self.calls], [(30, False), (30, False)])
        self.assertEqual(self.calls[1][1]["bookmakers"], "netbet_fr")
        self.assertTrue((result.path / "COMPLETED").is_file())
        receipt_bytes = (result.path / "receipt.json").read_bytes()
        self.assertNotIn(b"super-secret-value", receipt_bytes)
        receipt = json.loads(receipt_bytes)
        self.assertFalse(receipt["schedule_complete"])
        self.assertFalse(receipt["nhl_game_ids_verified"])
        self.assertFalse(receipt["prediction_publication_permitted"])
        self.assertFalse(receipt["training_permitted"])
        self.assertEqual(receipt["event_count"], 1)
        self.assertEqual(receipt["two_way_quote_count"], 1)

    def test_empty_bookmaker_feed_does_not_claim_empty_schedule(self):
        self.events = []
        self.odds = []
        result = self.capture()
        self.assertEqual(result.event_count, 0)
        self.assertFalse(json.loads((result.path / "receipt.json").read_text())["schedule_complete"])

    def test_quoted_event_missing_from_events_is_rejected_without_files(self):
        self.events = []
        with self.assertRaisesRegex(NHLOddsCaptureError, "absent"):
            self.capture()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_three_way_quote_is_archived_but_not_accepted(self):
        self.odds[0]["bookmakers"][0]["markets"][0]["outcomes"].append(
            {"name": "Draw", "price": 3.2})
        result = self.capture()
        self.assertEqual(result.two_way_quote_count, 0)
        receipt = json.loads((result.path / "receipt.json").read_text())
        self.assertEqual(receipt["rejected_market_count"], 1)

    def test_future_dated_quote_is_rejected(self):
        self.odds[0]["bookmakers"][0]["markets"][0]["last_update"] = "2026-10-10T00:00:00Z"
        with self.assertRaisesRegex(NHLOddsCaptureError, "futur"):
            self.capture()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_provider_error_is_secret_safe(self):
        def broken_transport(*args, **kwargs):
            raise RuntimeError("URL?apiKey=super-secret-value")
        self.transport = broken_transport
        with self.assertRaises(NHLOddsCaptureError) as caught:
            self.capture()
        self.assertNotIn("super-secret-value", str(caught.exception))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_double_capture_never_overwrites(self):
        first = self.capture()
        with self.assertRaisesRegex(NHLOddsCaptureError, "Archivage incomplet"):
            self.capture()
        self.assertTrue((first.path / "COMPLETED").is_file())


if __name__ == "__main__":
    unittest.main()
