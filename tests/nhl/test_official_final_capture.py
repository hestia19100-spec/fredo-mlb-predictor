"""NHL-32: a later official result cannot rewrite pregame evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.nhl.official_final_capture import (
    OfficialFinalCaptureError, capture_official_final, verify_official_final_capture,
)
from src.nhl.prospective_checkpoint import build_checkpoint, verify_checkpoint

UTC = timezone.utc
NOW = datetime(2026, 10, 8, 4, tzinfo=UTC)
GAME_ID = 2026020053


def _checkpoint(path: Path) -> None:
    game = {
        "game_id": GAME_ID, "away_abbr": "PIT", "home_abbr": "WSH",
        "scheduled_start_utc": "2026-10-07T23:30:00Z",
        "information_cutoff_utc": "2026-10-07T22:30:00Z",
        "status": "CURRENT_SEASON_FORM_DESCRIPTIVE_ONLY",
        "feature_sha256": "a" * 64,
        "current_season_import_id": "verified-import",
        "history_capture_id": "verified-history",
        "history_response_sha256": "b" * 64,
        "form_effective_available_at_utc": "2026-10-07T13:03:00+00:00",
        "away_current_season_games": 3, "home_current_season_games": 2,
        "schedule_before_cutoff": True,
        "current_season_import_selection": "LATEST_BEFORE_CUTOFF",
        "current_season_import_before_cutoff": True,
    }
    report = {
        "schema_version": "nhl26_prospective_readiness_v1",
        "schedule_acquisition_mode": "direct_https",
        "schedule_observed_at_utc": "2026-10-07T18:52:00Z",
        "schedule_response_sha256": "c" * 64,
        "target_date": "2026-10-07", "lead_minutes": 60,
        "current_season_import_verified": True,
        "game_count": 1, "games": [game],
        "training_permitted": False, "prediction_publication_permitted": False,
    }
    path.write_text(json.dumps(build_checkpoint(report, datetime(2026, 10, 7, 20, tzinfo=UTC))))


def _landing(**changes: object) -> bytes:
    document = {
        "id": GAME_ID, "season": 20262027, "gameType": 2,
        "gameState": "OFF", "gameScheduleState": "OK",
        "startTimeUTC": "2026-10-07T23:30:00Z",
        "awayTeam": {"abbrev": "PIT", "score": 4},
        "homeTeam": {"abbrev": "WSH", "score": 3},
    }
    document.update(changes)
    return json.dumps(document).encode("utf-8")


class _Response:
    status = 200
    headers = {"Content-Type": "application/json; charset=utf-8"}

    def __init__(self, url: str, raw: bytes, *, redirect: bool = False) -> None:
        self.url = url + ("?redirected" if redirect else "")
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def geturl(self) -> str:
        return self.url

    def read(self, size: int) -> bytes:
        return self.raw[:size]


class OfficialFinalCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "checkpoint.json"
        _checkpoint(self.checkpoint)
        self.proof_digest = verify_checkpoint(self.checkpoint)["proof_sha256"]
        self.calls = []

    def transport(self, raw: bytes, *, redirect: bool = False):
        def open_url(url: str, timeout: int):
            self.calls.append((url, timeout))
            return _Response(url, raw, redirect=redirect)
        return open_url

    def capture(self, raw: bytes, *, now=NOW, redirect: bool = False):
        return capture_official_final(
            self.checkpoint, GAME_ID, explicit_manual_run=True,
            transport=self.transport(raw, redirect=redirect),
            root=self.root / "captures", now=lambda: now,
        )

    def test_final_is_append_only_and_binds_to_unchanged_checkpoint(self) -> None:
        receipt = self.capture(_landing())
        self.assertEqual(self.calls, [("https://api-web.nhle.com/v1/gamecenter/2026020053/landing", 15)])
        self.assertEqual((receipt.away_score, receipt.home_score, receipt.winner_abbr), (4, 3, "PIT"))
        self.assertEqual(verify_official_final_capture(receipt.path, self.checkpoint), receipt)
        self.assertEqual(verify_checkpoint(self.checkpoint)["proof_sha256"], self.proof_digest)
        saved = json.loads((receipt.path / "receipt.json").read_text())
        self.assertEqual(saved["checkpoint_sha256"], self.proof_digest)
        self.assertFalse(saved["training_permitted"])
        self.assertFalse(saved["prediction_publication_permitted"])
        with self.assertRaises(OfficialFinalCaptureError):
            self.capture(_landing())

    def test_explicit_manual_gate_and_exact_game_gate(self) -> None:
        with self.assertRaises(OfficialFinalCaptureError):
            capture_official_final(self.checkpoint, GAME_ID, transport=self.transport(_landing()))
        with self.assertRaises(OfficialFinalCaptureError):
            capture_official_final(self.checkpoint, GAME_ID + 1,
                                   explicit_manual_run=True, transport=self.transport(_landing()))
        self.assertEqual(self.calls, [])

    def test_future_nonfinal_and_early_observation_are_rejected_without_archive(self) -> None:
        for raw, now in ((_landing(gameState="FUT"), NOW),
                         (_landing(), datetime(2026, 10, 7, 22, tzinfo=UTC))):
            with self.subTest(raw=raw, now=now), self.assertRaises(OfficialFinalCaptureError):
                self.capture(raw, now=now)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse((self.root / "captures").exists())

    def test_changed_start_wrong_teams_and_tied_score_require_review(self) -> None:
        for raw in (_landing(startTimeUTC="2026-10-08T23:30:00Z"),
                    _landing(awayTeam={"abbrev": "BOS", "score": 4}),
                    _landing(homeTeam={"abbrev": "WSH", "score": 4}),
                    _landing(gameType=3)):
            with self.subTest(raw=raw), self.assertRaises(OfficialFinalCaptureError):
                self.capture(raw)
        self.assertFalse((self.root / "captures").exists())

    def test_redirect_and_tampering_are_rejected(self) -> None:
        with self.assertRaises(OfficialFinalCaptureError):
            self.capture(_landing(), redirect=True)
        receipt = self.capture(_landing())
        (receipt.path / "response.json").write_bytes(_landing(homeTeam={"abbrev": "WSH", "score": 2}))
        with self.assertRaises(OfficialFinalCaptureError):
            verify_official_final_capture(receipt.path, self.checkpoint)


if __name__ == "__main__":
    unittest.main()
