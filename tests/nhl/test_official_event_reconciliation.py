"""No-network identity reconciliation tests for NHL captures."""
from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from src.nhl.official_event_reconciliation import (
    NHLEventReconciliationError,
    reconcile_archived_nhl_events,
    reconcile_nhl_schedule_events,
)
from src.nhl.public_schedule_capture import SCHEMA_VERSION as SCHEDULE_SCHEMA
from src.nhl.odds_api_candidates import SPORT_KEY


TARGET = date(2026, 10, 10)
SCHEDULE_AT = datetime(2026, 10, 10, 15, tzinfo=timezone.utc)
ODDS_AT = datetime(2026, 10, 10, 16, tzinfo=timezone.utc)
EVENT_ID = "a" * 32


def schedule(*games: dict) -> bytes:
    return json.dumps({"gameWeek": [{
        "date": TARGET.isoformat(), "games": list(games),
    }]}).encode()


def game(**changes: object) -> dict:
    item = {
        "id": 2026020070, "season": 20262027, "gameType": 2,
        "gameState": "FUT", "gameScheduleState": "OK",
        "startTimeUTC": "2026-10-10T20:00:00Z",
        "awayTeam": {
            "abbrev": "MTL",
            "placeName": {"default": "Montréal"},
            "commonName": {"default": "Canadiens"},
        },
        "homeTeam": {
            "abbrev": "BOS",
            "placeName": {"default": "Boston"},
            "commonName": {"default": "Bruins"},
        },
    }
    item.update(changes)
    return item


def event(**changes: object) -> dict:
    item = {
        "id": EVENT_ID, "sport_key": SPORT_KEY,
        "commence_time": "2026-10-10T20:00:00Z",
        "away_team": "Montreal Canadiens", "home_team": "Boston Bruins",
    }
    item.update(changes)
    return item


def events(*items: dict) -> bytes:
    return json.dumps(list(items)).encode()


class OfficialEventReconciliationTests(unittest.TestCase):
    def test_exact_identity_with_accent_and_cutoff(self) -> None:
        result = reconcile_nhl_schedule_events(
            schedule(game()), TARGET, SCHEDULE_AT, events(event()), ODDS_AT)
        self.assertEqual(
            [(x.nhl_game_id, x.provider_event_id) for x in result.matched],
            [(2026020070, EVENT_ID)],
        )
        self.assertEqual(result.unmatched_nhl_game_ids, ())
        self.assertEqual(result.unmatched_provider_event_ids, ())
        self.assertFalse(result.training_permitted)
        self.assertFalse(result.prediction_publication_permitted)

    def test_provider_feed_is_not_claimed_complete(self) -> None:
        result = reconcile_nhl_schedule_events(
            schedule(game()), TARGET, SCHEDULE_AT, events(), ODDS_AT)
        self.assertEqual(result.matched, ())
        self.assertEqual(result.unmatched_nhl_game_ids, (2026020070,))

    def test_swapped_home_away_and_changed_start_do_not_match(self) -> None:
        for row in (
            event(away_team="Boston Bruins", home_team="Montreal Canadiens"),
            event(commence_time="2026-10-10T21:00:00Z"),
        ):
            with self.subTest(row=row):
                result = reconcile_nhl_schedule_events(
                    schedule(game()), TARGET, SCHEDULE_AT, events(row), ODDS_AT)
                self.assertEqual(result.matched, ())
                self.assertEqual(result.unmatched_nhl_game_ids, (2026020070,))
                self.assertEqual(result.unmatched_provider_event_ids, (EVENT_ID,))

    def test_duplicate_provider_fixture_is_ambiguous(self) -> None:
        second = event(id="b" * 32)
        result = reconcile_nhl_schedule_events(
            schedule(game()), TARGET, SCHEDULE_AT,
            events(event(), second), ODDS_AT)
        self.assertEqual(result.matched, ())
        self.assertEqual(result.ambiguous_nhl_game_ids, (2026020070,))
        self.assertEqual(len(result.unmatched_provider_event_ids), 2)

    def test_one_hour_cutoff_includes_boundary_and_rejects_late(self) -> None:
        boundary = datetime(2026, 10, 10, 19, tzinfo=timezone.utc)
        accepted = reconcile_nhl_schedule_events(
            schedule(game()), TARGET, SCHEDULE_AT, events(event()), boundary)
        self.assertEqual(len(accepted.matched), 1)
        late = datetime(2026, 10, 10, 19, 0, 1, tzinfo=timezone.utc)
        rejected = reconcile_nhl_schedule_events(
            schedule(game()), TARGET, SCHEDULE_AT, events(event()), late)
        self.assertEqual(rejected.matched, ())
        self.assertEqual(rejected.late_nhl_game_ids, (2026020070,))

    def test_duplicate_official_fixture_is_ambiguous(self) -> None:
        result = reconcile_nhl_schedule_events(
            schedule(game(), game(id=2026020071)), TARGET, SCHEDULE_AT,
            events(event()), ODDS_AT)
        self.assertEqual(result.matched, ())
        self.assertEqual(result.ambiguous_nhl_game_ids, (2026020070, 2026020071))

    def test_late_schedule_capture_is_not_prospective(self) -> None:
        late = datetime(2026, 10, 10, 19, 1, tzinfo=timezone.utc)
        result = reconcile_nhl_schedule_events(
            schedule(game()), TARGET, late, events(event()), ODDS_AT)
        self.assertEqual(result.matched, ())
        self.assertEqual(result.late_nhl_game_ids, (2026020070,))

    def test_missing_official_full_name_fails_closed(self) -> None:
        raw = schedule(game(awayTeam={"abbrev": "MTL"}))
        result = reconcile_nhl_schedule_events(
            raw, TARGET, SCHEDULE_AT, events(event()), ODDS_AT)
        self.assertEqual(result.matched, ())
        self.assertEqual(result.unmatched_nhl_game_ids, (2026020070,))

    def test_invalid_capture_clock_is_rejected(self) -> None:
        with self.assertRaises(NHLEventReconciliationError):
            reconcile_nhl_schedule_events(
                schedule(game()), TARGET, SCHEDULE_AT.replace(tzinfo=None),
                events(event()), ODDS_AT)

    def test_archives_are_hash_verified_and_read_only(self) -> None:
        schedule_raw = schedule(game())
        events_raw = events(event())
        odds_raw = events(dict(event(), bookmakers=[]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule_slot, odds_slot = root / "schedule", root / "odds"
            schedule_slot.mkdir()
            odds_slot.mkdir()
            for slot in (schedule_slot, odds_slot):
                (slot / "COMPLETED").touch()
            (schedule_slot / "response.json").write_bytes(schedule_raw)
            schedule_receipt = {
                "schema_version": SCHEDULE_SCHEMA,
                "status": "CAPTURE_ONLY_NOT_MODEL_ELIGIBLE",
                "provider_id": "nhl_public_web_api",
                "source_url": "https://api-web.nhle.com/v1/schedule/2026-10-10",
                "target_date": TARGET.isoformat(),
                "request_started_at_utc": "2026-10-10T14:59:59Z",
                "observed_at_utc": "2026-10-10T15:00:00Z",
                "response_sha256": sha256(schedule_raw).hexdigest(),
                "historical_as_of_availability_proven": False,
                "training_permitted": False,
                "prediction_publication_permitted": False,
                "future_regular_games": [{
                    "game_id": 2026020070, "season": 20262027,
                    "start_utc": "2026-10-10T20:00:00.000000Z",
                    "away_abbr": "MTL", "home_abbr": "BOS",
                }],
            }
            (schedule_slot / "receipt.json").write_text(
                json.dumps(schedule_receipt), encoding="utf-8")
            (odds_slot / "events.json").write_bytes(events_raw)
            (odds_slot / "odds.json").write_bytes(odds_raw)
            odds_receipt = {
                "schema_version": "nhl_odds_api_capture_only_v2",
                "status": "CAPTURE_ONLY_NOT_MODEL_ELIGIBLE",
                "provider": "the_odds_api",
                "observed_at_utc": "2026-10-10T16:00:00Z",
                "events_sha256": sha256(events_raw).hexdigest(),
                "odds_sha256": sha256(odds_raw).hexdigest(),
                "event_count": 1, "two_way_quote_count": 0,
                "rejected_market_count": 0,
                "training_permitted": False,
                "prediction_publication_permitted": False,
            }
            (odds_slot / "receipt.json").write_text(
                json.dumps(odds_receipt), encoding="utf-8")
            self.assertEqual(
                len(reconcile_archived_nhl_events(schedule_slot, odds_slot).matched), 1)
            (odds_slot / "events.json").write_bytes(events(event(id="b" * 32)))
            with self.assertRaises(NHLEventReconciliationError):
                reconcile_archived_nhl_events(schedule_slot, odds_slot)


if __name__ == "__main__":
    unittest.main()
