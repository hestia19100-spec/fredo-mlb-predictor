"""Offline validation of candidate NHL fixtures and two-way French quotes."""

from __future__ import annotations

from decimal import Decimal
import json
import unittest

from src.nhl.odds_api_candidates import (
    NHLOddsCandidateError, parse_nhl_events, parse_nhl_h2h_odds,
)


EVENT_ID = "a" * 32


def event(**changes):
    row = {
        "id": EVENT_ID, "sport_key": "icehockey_nhl",
        "commence_time": "2026-10-10T00:00:00Z",
        "away_team": "Boston Bruins", "home_team": "New York Rangers",
    }
    row.update(changes)
    return row


def bookmaker(*, outcomes=None, key="netbet_fr", markets=None):
    if outcomes is None:
        outcomes = [
            {"name": "Boston Bruins", "price": 2.1},
            {"name": "New York Rangers", "price": 1.8},
        ]
    return {
        "key": key,
        "markets": markets if markets is not None else [{
            "key": "h2h", "last_update": "2026-10-09T19:00:00Z",
            "outcomes": outcomes,
        }],
    }


def encoded(value):
    return json.dumps(value).encode("utf-8")


class NHLOddsCandidateTests(unittest.TestCase):
    def test_events_have_provider_identity_not_nhl_game_id(self):
        rows = parse_nhl_events(encoded([event()]))
        self.assertEqual(rows[0].provider_event_id, EVENT_ID)
        self.assertEqual(rows[0].away_team_name, "Boston Bruins")
        self.assertEqual(rows[0].start_utc.isoformat(), "2026-10-10T00:00:00+00:00")

    def test_wrong_sport_and_duplicate_event_fail_closed(self):
        for rows in ([event(sport_key="icehockey_ahl")], [event(), event()]):
            with self.subTest(rows=rows), self.assertRaises(NHLOddsCandidateError):
                parse_nhl_events(encoded(rows))

    def test_empty_events_are_not_interpreted_as_no_games(self):
        self.assertEqual(parse_nhl_events(b"[]"), ())

    def test_valid_french_two_way_quote(self):
        rows = parse_nhl_h2h_odds(encoded([event(bookmakers=[bookmaker()])]))
        self.assertEqual(len(rows.events), 1)
        self.assertEqual(len(rows.two_way_quotes), 1)
        self.assertEqual(rows.two_way_quotes[0].away_decimal_odds, Decimal("2.1"))
        self.assertEqual(rows.two_way_quotes[0].home_decimal_odds, Decimal("1.8"))
        self.assertEqual(rows.rejected_market_keys, ())

    def test_three_way_or_draw_market_is_rejected_without_losing_event(self):
        three = [
            {"name": "Boston Bruins", "price": 2.1},
            {"name": "Draw", "price": 3.2},
            {"name": "New York Rangers", "price": 1.8},
        ]
        rows = parse_nhl_h2h_odds(encoded([event(bookmakers=[bookmaker(outcomes=three)])]))
        self.assertEqual(len(rows.events), 1)
        self.assertEqual(rows.two_way_quotes, ())
        self.assertEqual(rows.rejected_market_keys, ((EVENT_ID, "netbet_fr"),))

    def test_betclic_two_way_does_not_bypass_provider_allowlist(self):
        rows = parse_nhl_h2h_odds(encoded([event(bookmakers=[bookmaker(key="betclic_fr")])]))
        self.assertEqual(rows.two_way_quotes, ())
        self.assertEqual(rows.rejected_market_keys, ((EVENT_ID, "betclic_fr"),))

    def test_non_french_bookmaker_is_not_admitted(self):
        rows = parse_nhl_h2h_odds(encoded([event(bookmakers=[bookmaker(key="draftkings")])]))
        self.assertEqual(rows.two_way_quotes, ())
        self.assertEqual(rows.rejected_market_keys, ((EVENT_ID, "draftkings"),))

    def test_mismatched_teams_are_rejected(self):
        wrong = [
            {"name": "Boston Bruins", "price": 2.1},
            {"name": "Montreal Canadiens", "price": 1.8},
        ]
        rows = parse_nhl_h2h_odds(encoded([event(bookmakers=[bookmaker(outcomes=wrong)])]))
        self.assertEqual(rows.two_way_quotes, ())

    def test_bad_price_or_missing_market_timestamp_fails_closed(self):
        bad = [
            {"name": "Boston Bruins", "price": True},
            {"name": "New York Rangers", "price": 1.8},
        ]
        for book in (
            bookmaker(outcomes=bad),
            bookmaker(markets=[{"key": "h2h", "outcomes": bookmaker()["markets"][0]["outcomes"]}]),
        ):
            with self.subTest(book=book), self.assertRaises(NHLOddsCandidateError):
                parse_nhl_h2h_odds(encoded([event(bookmakers=[book])]))


if __name__ == "__main__":
    unittest.main()
