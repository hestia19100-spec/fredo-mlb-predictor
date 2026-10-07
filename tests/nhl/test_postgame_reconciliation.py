"""NHL-33: only captured finals resolve a sealed pregame game."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.nhl.official_final_capture import capture_official_final
from src.nhl.postgame_reconciliation import (
    NHLPostgameReconciliationError, build_postgame_reconciliation,
    seal_postgame_reconciliation, verify_postgame_reconciliation,
)
from src.nhl.prospective_checkpoint import build_checkpoint, verify_checkpoint

UTC = timezone.utc
NOW = datetime(2026, 10, 8, 5, tzinfo=UTC)


def _checkpoint(path: Path) -> None:
    def game(game_id: int, away: str, home: str) -> dict[str, object]:
        return {
            "game_id": game_id, "away_abbr": away, "home_abbr": home,
            "scheduled_start_utc": "2026-10-07T23:30:00Z",
            "information_cutoff_utc": "2026-10-07T22:30:00Z",
            "status": "CURRENT_SEASON_FORM_DESCRIPTIVE_ONLY",
            "feature_sha256": "a" * 64,
            "current_season_import_id": "verified-import",
            "history_capture_id": "verified-history",
            "history_response_sha256": "b" * 64,
            "form_effective_available_at_utc": "2026-10-07T13:03:00Z",
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
        "game_count": 2,
        "games": [game(2026020053, "PIT", "WSH"), game(2026020054, "COL", "WPG")],
        "training_permitted": False, "prediction_publication_permitted": False,
    }
    path.write_text(json.dumps(build_checkpoint(report, datetime(2026, 10, 7, 20, tzinfo=UTC))))


class _Response:
    status = 200
    headers = {"Content-Type": "application/json"}

    def __init__(self, url: str, raw: bytes) -> None:
        self.url, self.raw = url, raw

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def geturl(self) -> str:
        return self.url

    def read(self, size: int) -> bytes:
        return self.raw[:size]


class PostgameReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "checkpoint.json"
        _checkpoint(self.checkpoint)
        self.proof_sha = verify_checkpoint(self.checkpoint)["proof_sha256"]

    def capture(self, game_id: int, away: str, home: str,
                away_score: int, home_score: int) -> Path:
        raw = json.dumps({
            "id": game_id, "season": 20262027, "gameType": 2,
            "gameState": "OFF", "gameScheduleState": "OK",
            "startTimeUTC": "2026-10-07T23:30:00Z",
            "awayTeam": {"abbrev": away, "score": away_score},
            "homeTeam": {"abbrev": home, "score": home_score},
        }).encode()
        receipt = capture_official_final(
            self.checkpoint, game_id, explicit_manual_run=True,
            transport=lambda url, timeout: _Response(url, raw),
            root=self.root / "captures", now=lambda: NOW,
        )
        return receipt.path

    def test_no_capture_means_pending_without_invented_result(self) -> None:
        report = build_postgame_reconciliation(self.checkpoint, (), checked_at_utc=NOW)
        self.assertEqual((report["game_count"], report["verified_final_count"], report["pending_count"]),
                         (2, 0, 2))
        self.assertEqual([game["status"] for game in report["games"]],
                         ["PENDING_OFFICIAL_FINAL"] * 2)
        self.assertTrue(all(game["winner_abbr"] is None for game in report["games"]))
        self.assertFalse(report["training_permitted"])
        self.assertFalse(report["prediction_publication_permitted"])

    def test_one_official_final_leaves_other_game_pending(self) -> None:
        slot = self.capture(2026020053, "PIT", "WSH", 4, 3)
        report = build_postgame_reconciliation(self.checkpoint, (slot,), checked_at_utc=NOW)
        self.assertEqual((report["verified_final_count"], report["pending_count"]), (1, 1))
        self.assertEqual(report["games"][0]["winner_abbr"], "PIT")
        self.assertEqual(report["games"][1]["status"], "PENDING_OFFICIAL_FINAL")
        self.assertEqual(report["checkpoint_sha256"], self.proof_sha)
        self.assertEqual(verify_checkpoint(self.checkpoint)["proof_sha256"], self.proof_sha)

    def test_conflicting_repeats_duplicates_and_future_proofs_are_blocked(self) -> None:
        slot = self.capture(2026020053, "PIT", "WSH", 4, 3)
        with self.assertRaises(NHLPostgameReconciliationError):
            build_postgame_reconciliation(self.checkpoint, (slot, slot), checked_at_utc=NOW)
        conflict = self.capture(2026020053, "PIT", "WSH", 2, 3)
        with self.assertRaises(NHLPostgameReconciliationError):
            build_postgame_reconciliation(self.checkpoint, (slot, conflict), checked_at_utc=NOW)
        with self.assertRaises(NHLPostgameReconciliationError):
            build_postgame_reconciliation(
                self.checkpoint, (slot,), checked_at_utc=datetime(2026, 10, 8, 4, tzinfo=UTC))

    def test_append_only_manifest_recomputes_from_sources(self) -> None:
        slot = self.capture(2026020053, "PIT", "WSH", 4, 3)
        path = seal_postgame_reconciliation(
            self.checkpoint, (slot,), explicit_manual_run=True,
            root=self.root / "reports", now=lambda: NOW,
        )
        verified = verify_postgame_reconciliation(path, self.checkpoint, (slot,))
        self.assertEqual(verified["verified_final_count"], 1)
        with self.assertRaises(NHLPostgameReconciliationError):
            seal_postgame_reconciliation(
                self.checkpoint, (slot,), explicit_manual_run=True,
                root=self.root / "reports", now=lambda: NOW,
            )
        document = json.loads(path.read_text())
        document["games"][0]["winner_abbr"] = "WSH"
        path.write_text(json.dumps(document))
        with self.assertRaises(NHLPostgameReconciliationError):
            verify_postgame_reconciliation(path, self.checkpoint, (slot,))

    def test_manual_seal_gate(self) -> None:
        with self.assertRaises(NHLPostgameReconciliationError):
            seal_postgame_reconciliation(self.checkpoint, (), root=self.root / "reports")
        self.assertFalse((self.root / "reports").exists())


if __name__ == "__main__":
    unittest.main()
