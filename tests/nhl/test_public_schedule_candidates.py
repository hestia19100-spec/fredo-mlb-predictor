"""Contrats hors ligne du calendrier public NHL (sans appel réseau)."""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
import unittest

from src.nhl.public_schedule_candidates import (
    PublicScheduleError,
    parse_public_schedule,
)


TARGET = date(2026, 10, 3)
OBSERVED = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)


def game(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": 2026020022,
        "season": 20262027,
        "gameType": 2,
        "gameState": "FUT",
        "gameScheduleState": "OK",
        "startTimeUTC": "2026-10-03T23:00:00Z",
        "awayTeam": {"abbrev": "CHI", "score": 9, "odds": [1.5]},
        "homeTeam": {"abbrev": "BUF", "score": 0},
        "periodDescriptor": {"number": 3},
    }
    value.update(changes)
    return value


def response(*games: dict[str, object], day: str = "2026-10-03") -> bytes:
    return json.dumps({"gameWeek": [{"date": day, "games": list(games)}]}).encode()


class PublicScheduleCandidatesTests(unittest.TestCase):
    def test_only_future_regular_games_and_approved_fields(self) -> None:
        raw = response(
            game(),
            game(id=2026020023, gameState="LIVE"),
            game(id=2026010024, gameType=1),
            game(id=2026020025, startTimeUTC="2026-10-03T11:00:00Z"),
        )
        candidates = parse_public_schedule(raw, TARGET, OBSERVED)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].game_id, 2026020022)
        self.assertEqual(candidates[0].start_utc.hour, 23)
        self.assertEqual((candidates[0].away_abbr, candidates[0].home_abbr), ("CHI", "BUF"))
        self.assertEqual(
            set(candidates[0].__dataclass_fields__),
            {"game_id", "season", "start_utc", "away_abbr", "home_abbr"},
        )

    def test_no_game_is_not_an_error(self) -> None:
        self.assertEqual(parse_public_schedule(response(), TARGET, OBSERVED), ())

    def test_missing_or_duplicate_target_day_fails_closed(self) -> None:
        with self.assertRaises(PublicScheduleError):
            parse_public_schedule(response(game(), day="2026-10-04"), TARGET, OBSERVED)
        duplicate = json.dumps({"gameWeek": [
            {"date": TARGET.isoformat(), "games": []},
            {"date": TARGET.isoformat(), "games": []},
        ]}).encode()
        with self.assertRaises(PublicScheduleError):
            parse_public_schedule(duplicate, TARGET, OBSERVED)

    def test_duplicate_or_malformed_future_game_fails_closed(self) -> None:
        for raw in (
            response(game(), game()),
            response(game(startTimeUTC="2026-10-03T23:00:00")),
            response(game(awayTeam={"abbrev": "BUF"})),
            response(game(id=True)),
        ):
            with self.subTest(raw=raw[:90]), self.assertRaises(PublicScheduleError):
                parse_public_schedule(raw, TARGET, OBSERVED)

    def test_observation_must_be_utc_and_response_bounded(self) -> None:
        with self.assertRaises(PublicScheduleError):
            parse_public_schedule(response(game()), TARGET, datetime(2026, 10, 3, 12))
        with self.assertRaises(PublicScheduleError):
            parse_public_schedule(b"", TARGET, OBSERVED)
        with self.assertRaises(PublicScheduleError):
            parse_public_schedule(b"x" * (8 * 1024 * 1024 + 1), TARGET, OBSERVED)


if __name__ == "__main__":
    unittest.main()
