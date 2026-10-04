"""Tests du contrôle pré-match NHL, sans données ni API réelles."""
from __future__ import annotations

import hashlib
import json
import unittest
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from src.nhl.database import NHLDatabaseError, initialize_nhl_database
from src.nhl.pregame_stats_readiness import audit_pregame_stats_availability
from src.nhl.public_schedule_capture import BASE_URL, capture_public_schedule

TARGET = date(2026, 10, 3)
STARTED = datetime(2026, 10, 3, 19, 0, tzinfo=timezone.utc)
OBSERVED = datetime(2026, 10, 3, 19, 0, 1, tzinfo=timezone.utc)


class FakeResponse(BytesIO):
    status = 200
    headers = {"Content-Type": "application/json"}

    def geturl(self) -> str:
        return BASE_URL + TARGET.isoformat()


def schedule_body() -> bytes:
    return json.dumps({"gameWeek": [{"date": TARGET.isoformat(), "games": [{
        "id": 2026020022,
        "season": 20262027,
        "gameType": 2,
        "gameState": "FUT",
        "gameScheduleState": "OK",
        "startTimeUTC": "2026-10-03T23:00:00Z",
        "awayTeam": {"id": 16, "abbrev": "CHI", "odds": [{"value": "+190"}]},
        "homeTeam": {"id": 7, "abbrev": "BUF"},
    }]}]}).encode()


class PregameStatsReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "nhl"
        self.root.mkdir()
        self.database = self.root / "test_nhl.db"
        ticks = iter((STARTED, OBSERVED))
        self.slot = capture_public_schedule(
            TARGET,
            explicit_manual_run=True,
            root=self.root / "captures",
            transport=lambda url, timeout: FakeResponse(schedule_body()),
            now=lambda: next(ticks),
        ).path

    def audit(self, *, lead_minutes: int = 120, min_games_per_team: int = 5):
        return audit_pregame_stats_availability(
            self.slot,
            database_path=self.database,
            allowed_root=self.root,
            lead_minutes=lead_minutes,
            min_games_per_team=min_games_per_team,
        )

    def test_no_local_database_is_reported_without_creating_one(self) -> None:
        report = self.audit()
        self.assertEqual(report["game_count"], 1)
        self.assertEqual(report["history_sample_present_count"], 0)
        self.assertEqual(report["games"][0]["status"], "NO_LOCAL_NHL_DATABASE")
        self.assertEqual(report["games"][0]["information_cutoff_utc"], "2026-10-03T21:00:00Z")
        self.assertFalse(report["historical_asof_availability_independently_proven"])
        self.assertFalse(report["training_permitted"])
        self.assertFalse(report["prediction_publication_permitted"])
        self.assertFalse(self.database.exists())
        self.assertNotIn("odds", json.dumps(report))

    def test_schedule_captured_after_cutoff_is_rejected(self) -> None:
        row = self.audit(lead_minutes=300)["games"][0]
        self.assertEqual(row["status"], "CAPTURE_AFTER_CUTOFF")
        self.assertFalse(row["capture_before_cutoff"])
        self.assertFalse(self.database.exists())

    def test_empty_valid_database_is_read_only_and_not_ready(self) -> None:
        initialize_nhl_database(self.database, allowed_root=self.root)
        before = hashlib.sha256(self.database.read_bytes()).hexdigest()
        row = self.audit()["games"][0]
        self.assertEqual(row["status"], "NO_VALID_ASOF_VIEW")
        self.assertEqual(before, hashlib.sha256(self.database.read_bytes()).hexdigest())

    def test_sample_presence_never_certifies_training_or_publication(self) -> None:
        initialize_nhl_database(self.database, allowed_root=self.root)
        for enough in (False, True):
            with self.subTest(enough=enough):
                history = SimpleNamespace(
                    away=SimpleNamespace(complete_game_count=5),
                    home=SimpleNamespace(complete_game_count=5 if enough else 4),
                    minimum_sample_reached=enough,
                )
                bundle = SimpleNamespace(history=history, feature_sha256="a" * 64)
                with patch("src.nhl.pregame_stats_readiness.build_asof_feature_bundle", return_value=bundle):
                    report = self.audit()
                self.assertEqual(report["games"][0]["history_sample_present"], enough)
                self.assertEqual(report["history_sample_present_count"], int(enough))
                self.assertFalse(report["training_permitted"])
                self.assertFalse(report["prediction_publication_permitted"])

    def test_invalid_database_does_not_become_missing_data(self) -> None:
        self.database.write_bytes(b"not a database")
        with self.assertRaises(NHLDatabaseError):
            self.audit()

    def test_invalid_cutoff_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.audit(lead_minutes=0)


if __name__ == "__main__":
    unittest.main()
