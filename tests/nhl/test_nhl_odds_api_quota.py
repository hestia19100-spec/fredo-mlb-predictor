"""Offline tests for the NHL Odds API daily credit gate."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.nhl_odds_api_quota import (
    EVENTS_URL,
    ODDS_URL,
    SCORES_URL,
    NHLDailyQuotaGate,
    NHLQuotaError,
)


KEY = "super-secret-value"
EVENTS = {"apiKey": KEY}
ODDS = {"apiKey": KEY, "bookmakers": "netbet_fr", "markets": "h2h", "oddsFormat": "decimal"}
SCORES = {"apiKey": KEY, "daysFrom": "3", "dateFormat": "iso"}


class Response:
    def __init__(self, cost: str):
        self.headers = {"x-requests-last": cost}


class NHLQuotaTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "quota.sqlite"
        self.calls = []
        self.now = datetime(2026, 10, 9, 10, tzinfo=timezone.utc)

    def transport(self, url, *, params, timeout, allow_redirects):
        self.calls.append(url)
        cost = {EVENTS_URL: "0", ODDS_URL: "1", SCORES_URL: "2"}[url]
        return Response(cost)

    def gate(self, limit=3, transport=None):
        return NHLDailyQuotaGate(
            self.path, limit, transport or self.transport, now=lambda: self.now,
        )

    @staticmethod
    def call(gate, url, params):
        return gate.get(url, params=params, timeout=30, allow_redirects=False)

    def test_one_events_odds_scores_cycle_uses_three_credits(self):
        gate = self.gate()
        for url, params in ((EVENTS_URL, EVENTS), (ODDS_URL, ODDS), (SCORES_URL, SCORES)):
            self.call(gate, url, params)
        self.assertEqual(self.calls, [EVENTS_URL, ODDS_URL, SCORES_URL])
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT kind, actual_cost FROM requests ORDER BY kind",
            ).fetchall()
            self.assertEqual(rows, [("events", 0), ("odds", 1), ("scores", 2)])
            self.assertNotIn(KEY, str(rows))

    def test_repeat_is_blocked_before_transport(self):
        gate = self.gate()
        self.call(gate, ODDS_URL, ODDS)
        with self.assertRaisesRegex(NHLQuotaError, "déjà tenté"):
            self.call(gate, ODDS_URL, ODDS)
        self.assertEqual(self.calls, [ODDS_URL])

    def test_limit_blocks_before_transport(self):
        gate = self.gate(limit=1)
        self.call(gate, ODDS_URL, ODDS)
        with self.assertRaisesRegex(NHLQuotaError, "Plafond"):
            self.call(gate, SCORES_URL, SCORES)
        self.assertEqual(self.calls, [ODDS_URL])

    def test_disabled_budget_blocks_paid_calls_but_not_events(self):
        gate = self.gate(limit=0)
        self.call(gate, EVENTS_URL, EVENTS)
        with self.assertRaisesRegex(NHLQuotaError, "Plafond"):
            self.call(gate, ODDS_URL, ODDS)
        self.assertEqual(self.calls, [EVENTS_URL])

    def test_network_error_retains_reservation_and_hides_key(self):
        def broken(*args, **kwargs):
            raise RuntimeError("https://example.test/?apiKey=" + KEY)
        gate = self.gate(limit=2, transport=broken)
        with self.assertRaises(NHLQuotaError) as caught:
            self.call(gate, SCORES_URL, SCORES)
        self.assertNotIn(KEY, str(caught.exception))
        with self.assertRaisesRegex(NHLQuotaError, "Plafond"):
            self.call(gate, ODDS_URL, ODDS)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute(
                "SELECT reserved_cost, actual_cost FROM requests",
            ).fetchone(), (2, None))

    def test_bad_cost_header_retains_reservation(self):
        gate = self.gate(transport=lambda *args, **kwargs: Response("unknown"))
        with self.assertRaisesRegex(NHLQuotaError, "inconnu"):
            self.call(gate, SCORES_URL, SCORES)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute(
                "SELECT actual_cost FROM requests",
            ).fetchone(), (None,))

    def test_overpriced_response_is_not_silently_settled(self):
        gate = self.gate(transport=lambda *args, **kwargs: Response("9"))
        with self.assertRaisesRegex(NHLQuotaError, "supérieur"):
            self.call(gate, ODDS_URL, ODDS)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute(
                "SELECT reserved_cost, actual_cost FROM requests",
            ).fetchone(), (1, None))

    def test_unapproved_endpoint_and_parameters_do_not_call_transport(self):
        gate = self.gate()
        for url, params in ((ODDS_URL, {**ODDS, "markets": "h2h,totals"}),
                            (SCORES_URL, {**SCORES, "daysFrom": "1"}),
                            ("https://example.test", EVENTS)):
            with self.assertRaisesRegex(NHLQuotaError, "hors du périmètre"):
                self.call(gate, url, params)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.path.exists())

    def test_paris_date_changes_at_local_midnight(self):
        gate = self.gate(limit=1)
        self.now = datetime(2026, 10, 9, 21, 30, tzinfo=timezone.utc)
        self.call(gate, ODDS_URL, ODDS)
        self.now = datetime(2026, 10, 9, 22, 30, tzinfo=timezone.utc)
        self.call(gate, ODDS_URL, ODDS)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute(
                "SELECT paris_day FROM requests ORDER BY paris_day",
            ).fetchall(), [("2026-10-09",), ("2026-10-10",)])

    def test_concurrent_requests_cannot_pass_duplicate_or_limit(self):
        gate = self.gate(limit=1)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(
                lambda _: self._attempt(gate, ODDS_URL, ODDS), range(2),
            ))
        self.assertEqual(sorted(results), ["blocked", "ok"])
        self.assertEqual(self.calls, [ODDS_URL])

    def _attempt(self, gate, url, params):
        try:
            self.call(gate, url, params)
            return "ok"
        except NHLQuotaError:
            return "blocked"


if __name__ == "__main__":
    unittest.main()
