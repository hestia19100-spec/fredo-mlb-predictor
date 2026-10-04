"""NHL-23: provenance conservatrice de l'historique MoneyPuck."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.nhl.moneypuck_team_capture import (
    NHLTeamCaptureError, assess_history_before_game, capture_team_history,
    verify_team_capture,
)
from src.nhl.moneypuck_team_import import TEAM_GAME_DOWNLOAD, NHLMoneyPuckImportError, import_team_games
from src.nhl.public_schedule_candidates import ScheduledGame

HEADER = (
    "team,season,name,gameId,playerTeam,opposingTeam,home_or_away,"
    "gameDate,position,situation,xGoalsFor,xGoalsAgainst,goalsFor,"
    "goalsAgainst,shotsOnGoalFor,shotsOnGoalAgainst"
)

def _csv() -> bytes:
    rows = [HEADER]
    for situation in ("all", "5on5"):
        rows.extend((
            f"NYR,2021,NYR,2021020001,NYR,BOS,AWAY,20211013,Team Level,{situation},2.1,1.4,4,2,30,24",
            f"BOS,2021,BOS,2021020001,BOS,NYR,HOME,20211013,Team Level,{situation},1.4,2.1,2,4,24,30",
        ))
    return ("\n".join(rows) + "\n").encode("utf-8")

class Response(BytesIO):
    status = 200
    headers = {"Content-Type": "text/csv"}
    def geturl(self) -> str:
        return TEAM_GAME_DOWNLOAD

class TeamCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "nhl" / "captures"
        self.time = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)
        self.times = iter((self.time, self.time + timedelta(seconds=1)))
        self.transport = lambda url, timeout: Response(_csv())

    def _capture(self):
        return capture_team_history(
            explicit_manual_run=True, transport=self.transport,
            code_commit="a" * 40, root=self.root, now=lambda: next(self.times),
        )

    def test_capture_is_immutable_and_verifiable(self) -> None:
        capture = self._capture()
        verified = verify_team_capture(capture.path, allowed_root=self.root)
        self.assertEqual(verified.response_sha256, capture.response_sha256)
        self.assertEqual(len(verified.history.regular_rows), 4)
        receipt = json.loads((capture.path / "receipt.json").read_text())
        self.assertFalse(receipt["historical_asof_availability_proven"])
        self.assertFalse(receipt["training_permitted"])
        self.assertFalse(receipt["prediction_publication_permitted"])
        self.assertFalse(receipt["regular_coverage_complete"])

    def test_prior_rows_can_be_counted_only_for_future_cutoff(self) -> None:
        capture = self._capture()
        game = ScheduledGame(2026020001, 2026,
            datetime(2026, 10, 6, 21, tzinfo=timezone.utc), "NYR", "BOS")
        report = assess_history_before_game(capture, game, lead_minutes=120)
        self.assertTrue(report["captured_before_cutoff"])
        self.assertEqual(report["prior_regular_games_by_team"], {"NYR": 1, "BOS": 1})
        self.assertFalse(report["historical_backtest_asof_proven"])
        self.assertFalse(report["training_permitted"])

    def test_late_capture_cannot_backfill_pregame_history(self) -> None:
        capture = self._capture()
        game = ScheduledGame(2026020001, 2026,
            datetime(2026, 10, 4, 13, tzinfo=timezone.utc), "NYR", "BOS")
        report = assess_history_before_game(capture, game, lead_minutes=120)
        self.assertFalse(report["captured_before_cutoff"])
        self.assertEqual(report["prior_regular_games_by_team"], {"NYR": 0, "BOS": 0})

    def test_manual_flag_and_commit_are_required(self) -> None:
        with self.assertRaises(NHLTeamCaptureError):
            capture_team_history(transport=self.transport, code_commit="a" * 40,
                                 root=self.root)
        with self.assertRaises(NHLTeamCaptureError):
            capture_team_history(explicit_manual_run=True, transport=self.transport,
                                 code_commit="short", root=self.root)
        self.assertFalse(self.root.exists())

    def test_tampering_and_incomplete_slots_fail_closed(self) -> None:
        capture = self._capture()
        (capture.path / "response.csv").write_bytes(b"altered")
        with self.assertRaises(NHLTeamCaptureError):
            verify_team_capture(capture.path, allowed_root=self.root)
        (capture.path / "COMPLETED").unlink()
        with self.assertRaises(NHLTeamCaptureError):
            verify_team_capture(capture.path, allowed_root=self.root)

    def test_local_file_intake_records_unverified_origin(self) -> None:
        source = Path(self.directory.name) / "download.csv"
        source.write_bytes(_csv())
        capture = capture_team_history(explicit_manual_run=True, source_file=source,
            code_commit="a" * 40, root=self.root, now=lambda: next(self.times))
        receipt = json.loads((capture.path / "receipt.json").read_text())
        self.assertEqual(receipt["acquisition_mode"], "local_file_intake")
        self.assertFalse(receipt["network_source_independently_verified"])
        self.assertEqual(verify_team_capture(capture.path, allowed_root=self.root).response_sha256, capture.response_sha256)

    def test_invalid_older_season_is_excluded_without_weakening_selected_rows(self) -> None:
        source = Path(self.directory.name) / "combined.csv"
        invalid = "ARI,2011,ARI,2011020005,ARI,ARI,AWAY,20111015,Team Level,all,2,2,1,1,20,20"
        source.write_bytes(_csv() + (invalid + "\n").encode("utf-8"))
        snapshot = import_team_games(source, seasons=(2021, 2022, 2023, 2024, 2025))
        self.assertEqual(len(snapshot.rows), 4)
        with self.assertRaises(NHLMoneyPuckImportError):
            import_team_games(source)

    def test_outside_root_rejected(self) -> None:
        capture = self._capture()
        with self.assertRaises(NHLTeamCaptureError):
            verify_team_capture(capture.path, allowed_root=self.root / "other")

if __name__ == "__main__":
    unittest.main()
