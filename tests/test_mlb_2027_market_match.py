"""Tests hors réseau du rapprochement MLB 2027 / marché français h2h."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from types import SimpleNamespace
import unittest

from src.mlb_2027_market_match import (
    MLB2027MarketMatchError,
    match_mlb_2027_market,
)
from src.mlb_2027_observed_source import ObservedSourceBundle
from src.mlb_2027_reference_readiness import UpcomingGame
from src.odds_api import MoneylineBookmaker, MoneylineEvent, OddsFetchResult


DAY = date(2027, 4, 7)
START = datetime(2027, 4, 7, 23, 0, tzinfo=timezone.utc)
RECEIVED = START - timedelta(minutes=90)


def source(*, doubleheader=False):
    game_ids = (101, 102) if doubleheader else (101,)
    schedule = tuple(SimpleNamespace(
        game_id=game_id, official_date=DAY.isoformat(),
        away_team_name="New York Yankees", home_team_name="Boston Red Sox",
    ) for game_id in game_ids)
    upcoming = tuple(UpcomingGame(
        game_id=game_id, official_date=DAY, game_type="R",
        away_team_id=147, home_team_id=111,
        scheduled_start_utc=START, status="SCHEDULED",
    ) for game_id in game_ids)
    return ObservedSourceBundle(
        target_date=DAY, observations=(SimpleNamespace(games=schedule),),
        upcoming_games=upcoming, prior_final_games=(), receipt_bytes=b"receipt",
        receipt_sha256="a" * 64, latest_received_at_utc=RECEIVED,
    )


def odds(*, events=None, received=RECEIVED, region="fr", raw=None):
    if events is None:
        events = (MoneylineEvent(
            provider_event_id="event-1", commence_time_utc=START,
            away_team_name="New York Yankees", home_team_name="Boston Red Sox",
            bookmakers=(
                MoneylineBookmaker("book-1", "Book 1", RECEIVED, Decimal("1.90"), Decimal("1.85")),
                MoneylineBookmaker("book-2", "Book 2", RECEIVED, Decimal("1.80"), Decimal("1.95")),
            ),
        ),)
    if raw is None:
        raw = json.dumps([
            {
                "id": event.provider_event_id, "sport_key": "baseball_mlb",
                "commence_time": event.commence_time_utc.isoformat(),
                "away_team": event.away_team_name, "home_team": event.home_team_name,
                "bookmakers": [{
                    "key": book.key, "title": book.title,
                    "last_update": book.last_update_utc.isoformat(),
                    "markets": [{"key": "h2h", "outcomes": [
                        {"name": event.away_team_name, "price": float(book.away_decimal_odds)},
                        {"name": event.home_team_name, "price": float(book.home_decimal_odds)},
                    ]}],
                } for book in event.bookmakers],
            } for event in events
        ]).encode("utf-8")
    return OddsFetchResult(
        provider="the_odds_api_v4", sport_key="baseball_mlb",
        region=region, market="h2h", odds_format="decimal",
        response_received_at_utc=received, response_body_sha256=hashlib.sha256(raw).hexdigest(),
        quota_remaining=499, quota_used=1, quota_last_cost=1,
        raw_content=raw, events=events,
    )


class MarketMatchTests(unittest.TestCase):
    def test_best_price_for_each_team_from_separate_bookmakers(self):
        day = match_mlb_2027_market(source(), odds())
        self.assertEqual(day.games[0].status, "MATCHED")
        self.assertEqual(day.games[0].home_best_odds, Decimal("1.95"))
        self.assertEqual(day.games[0].home_best_bookmakers, ("Book 2",))
        self.assertEqual(day.games[0].away_best_odds, Decimal("1.90"))
        self.assertEqual(day.games[0].away_best_bookmakers, ("Book 1",))

    def test_no_event_is_not_presented_as_a_price(self):
        row = match_mlb_2027_market(source(), odds(events=())).games[0]
        self.assertEqual(row.status, "NO_MATCHING_EVENT")
        self.assertIsNone(row.home_best_odds)

    def test_doubleheader_collision_is_not_guessed(self):
        day = match_mlb_2027_market(source(doubleheader=True), odds())
        self.assertEqual([row.status for row in day.games], ["AMBIGUOUS_EVENT"] * 2)

    def test_stale_quotes_are_excluded(self):
        event = odds().events[0]
        stale = MoneylineBookmaker("old", "Old", RECEIVED - timedelta(hours=3), Decimal("2"), Decimal("2"))
        row = match_mlb_2027_market(
            source(), odds(events=(MoneylineEvent(
                event.provider_event_id, event.commence_time_utc,
                event.away_team_name, event.home_team_name, (stale,),
            ),)),
        ).games[0]
        self.assertEqual(row.status, "NO_FRESH_TWO_WAY_QUOTES")

    def test_wrong_region_or_tampered_raw_response_fails_closed(self):
        with self.assertRaises(MLB2027MarketMatchError):
            match_mlb_2027_market(source(), odds(region="us"))
        bad = odds()
        tampered = OddsFetchResult(
            bad.provider, bad.sport_key, bad.region, bad.market, bad.odds_format,
            bad.response_received_at_utc, bad.response_body_sha256,
            bad.quota_remaining, bad.quota_used, bad.quota_last_cost,
            b"tampered", bad.events,
        )
        with self.assertRaises(MLB2027MarketMatchError):
            match_mlb_2027_market(source(), tampered)

    def test_after_first_pitch_fails_closed(self):
        with self.assertRaises(MLB2027MarketMatchError):
            match_mlb_2027_market(source(), odds(received=START))


if __name__ == "__main__":
    unittest.main()
