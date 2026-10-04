"""Tests du rapprochement hors ligne des identités de matchs NHL."""
from __future__ import annotations

import json
import unittest
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from src.nhl.public_schedule_capture import BASE_URL, capture_public_schedule
from src.nhl.schedule_capture_readiness import (
    ScheduleCaptureReadinessError,
    audit_schedule_capture,
)

TARGET = date(2026, 10, 3)
STARTED = datetime(2026, 10, 3, 19, 0, tzinfo=timezone.utc)
OBSERVED = datetime(2026, 10, 3, 19, 0, 1, tzinfo=timezone.utc)


class FakeResponse(BytesIO):
    status = 200
    headers = {"Content-Type": "application/json"}

    def geturl(self) -> str:
        return BASE_URL + TARGET.isoformat()


def payload(*, away_id=16, home_id=7, add_past=False) -> bytes:
    future = {
        "id": 2026020022,
        "season": 20262027,
        "gameType": 2,
        "gameState": "FUT",
        "gameScheduleState": "OK",
        "startTimeUTC": "2026-10-03T23:00:00Z",
        "awayTeam": {"id": away_id, "abbrev": "CHI", "odds": [{"value": "+190"}]},
        "homeTeam": {"id": home_id, "abbrev": "BUF", "score": 99},
    }
    games = [future]
    if add_past:
        games.append({
            "id": 2026020001,
            "season": 20262027,
            "gameType": 2,
            "gameState": "OFF",
            "gameScheduleState": "OK",
            "startTimeUTC": "2026-10-03T16:00:00Z",
            "awayTeam": {"abbrev": "BOS"},
            "homeTeam": {"abbrev": "NYR"},
        })
    return json.dumps({"gameWeek": [{"date": TARGET.isoformat(), "games": games}]}).encode()


class ScheduleCaptureReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def capture(self, raw: bytes) -> Path:
        ticks = iter((STARTED, OBSERVED))
        result = capture_public_schedule(
            TARGET,
            explicit_manual_run=True,
            root=self.root,
            transport=lambda url, timeout: FakeResponse(raw),
            now=lambda: next(ticks),
        )
        return result.path

    def test_verified_capture_exposes_only_future_game_identity(self) -> None:
        report = audit_schedule_capture(self.capture(payload(add_past=True)))
        self.assertEqual(report["status"], "SCHEDULE_IDENTITY_ONLY")
        self.assertEqual(report["game_count"], 1)
        self.assertEqual(report["games"][0]["away_team_id"], 16)
        self.assertEqual(report["games"][0]["home_team_id"], 7)
        self.assertFalse(report["historical_team_history_verified"])
        self.assertFalse(report["training_permitted"])
        self.assertFalse(report["prediction_publication_permitted"])
        self.assertNotIn("odds", json.dumps(report))
        self.assertNotIn("score", json.dumps(report))

    def test_missing_or_boolean_team_id_is_rejected(self) -> None:
        for away_id in (None, True, 0):
            with self.subTest(away_id=away_id):
                slot = self.capture(payload(away_id=away_id))
                with self.assertRaises(ScheduleCaptureReadinessError):
                    audit_schedule_capture(slot)

    def test_same_team_ids_are_rejected(self) -> None:
        slot = self.capture(payload(away_id=7, home_id=7))
        with self.assertRaises(ScheduleCaptureReadinessError):
            audit_schedule_capture(slot)

    def test_tampered_capture_is_rejected(self) -> None:
        slot = self.capture(payload())
        (slot / "response.json").write_bytes(payload(home_id=8))
        with self.assertRaises(ScheduleCaptureReadinessError):
            audit_schedule_capture(slot)


if __name__ == "__main__":
    unittest.main()
