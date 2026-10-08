"""NHL-36: official historical finals stay tied to verified MoneyPuck games."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.nhl.historical_official_final import (
    HistoricalOfficialFinalError, capture_historical_final,
    reconcile_historical_finals, verify_historical_final,
)
from src.nhl.moneypuck_team_capture import capture_team_history
from tests.nhl.test_moneypuck_team_capture import _csv

GAME_ID = 2021020001
SOURCE_TIME = datetime(2026, 10, 7, 13, tzinfo=timezone.utc)
FINAL_TIME = datetime(2026, 10, 8, 13, tzinfo=timezone.utc)


def _landing(**changes: object) -> bytes:
    document = {
        "id": GAME_ID, "season": 20212022, "gameType": 2,
        "gameDate": "2021-10-13", "startTimeUTC": "2021-10-13T23:00:00Z",
        "gameState": "OFF", "gameScheduleState": "OK",
        "awayTeam": {"abbrev": "NYR", "score": 2},
        "homeTeam": {"abbrev": "BOS", "score": 3},
        "periodDescriptor": {"periodType": "SO"},
    }
    document.update(changes)
    return json.dumps(document).encode("utf-8")


class _Response:
    status = 200
    headers = {"Content-Type": "application/json; charset=utf-8"}

    def __init__(self, url: str, raw: bytes, *, redirect: bool = False):
        self.url = url + ("?redirected" if redirect else "")
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def geturl(self):
        return self.url

    def read(self, size):
        return self.raw[:size]


class HistoricalOfficialFinalTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.team_root = self.root / "team"
        self.final_root = self.root / "final"
        self.csv_path = self.root / "source.csv"
        source = _csv().replace(b",4,2,", b",2,2,").replace(b",2,4,", b",2,2,")
        self.csv_path.write_bytes(source)
        self.capture = capture_team_history(
            explicit_manual_run=True, source_file=self.csv_path,
            code_commit="a" * 40, root=self.team_root, now=lambda: SOURCE_TIME,
        )
        self.calls = []

    def _transport(self, raw: bytes, *, redirect: bool = False):
        def open_url(url: str, timeout: int):
            self.calls.append((url, timeout))
            return _Response(url, raw, redirect=redirect)
        return open_url

    def _capture(self, raw: bytes | None = None):
        return capture_historical_final(
            self.capture.path, GAME_ID, explicit_manual_run=True,
            transport=self._transport(_landing() if raw is None else raw),
            root=self.final_root, team_root=self.team_root,
            now=lambda: FINAL_TIME,
        )

    def test_tied_statistical_score_resolves_to_official_shootout_winner(self) -> None:
        receipt = self._capture()
        verified = verify_historical_final(
            receipt.path, self.capture.path, root=self.final_root,
            team_root=self.team_root)
        self.assertEqual(receipt, verified)
        self.assertEqual((receipt.winner_abbr, receipt.home_score,
                          receipt.away_score, receipt.final_type), ("BOS", 3, 2, "SO"))
        self.assertTrue(receipt.source_was_tied)
        self.assertEqual(self.calls, [(
            "https://api-web.nhle.com/v1/gamecenter/2021020001/landing", 15)])
        report = reconcile_historical_finals(
            self.capture.path, [receipt.path], root=self.final_root,
            team_root=self.team_root)
        self.assertEqual(report.resolved_tied_game_ids, (GAME_ID,))
        self.assertEqual(report.unresolved_tied_game_ids, ())
        self.assertTrue(report.final_capture_coverage_complete)
        self.assertFalse(report.origin_independently_verified)
        self.assertFalse(report.historical_pregame_availability_proven)
        self.assertFalse(report.training_permitted)
        self.assertFalse(report.prediction_publication_permitted)

    def test_no_manual_flag_no_network(self) -> None:
        with self.assertRaises(HistoricalOfficialFinalError):
            capture_historical_final(
                self.capture.path, GAME_ID, transport=self._transport(_landing()),
                root=self.final_root, team_root=self.team_root)
        self.assertEqual(self.calls, [])

    def test_redirect_nonfinal_wrong_identity_and_wrong_period_fail_closed(self) -> None:
        with self.assertRaises(HistoricalOfficialFinalError):
            capture_historical_final(
                self.capture.path, GAME_ID, explicit_manual_run=True,
                transport=self._transport(_landing(), redirect=True),
                root=self.final_root, team_root=self.team_root,
                now=lambda: FINAL_TIME)
        for raw in (
            _landing(gameState="LIVE"), _landing(id=2021020002),
            _landing(gameDate="2021-10-14"),
            _landing(homeTeam={"abbrev": "CHI", "score": 3}),
            _landing(periodDescriptor={"periodType": "REG"}),
            _landing(homeTeam={"abbrev": "BOS", "score": 4}),
        ):
            with self.subTest(raw=raw), self.assertRaises(HistoricalOfficialFinalError):
                self._capture(raw)
        self.assertFalse(self.final_root.exists())

    def test_tampered_bytes_or_duplicate_receipts_are_rejected(self) -> None:
        receipt = self._capture()
        with self.assertRaises(HistoricalOfficialFinalError):
            reconcile_historical_finals(
                self.capture.path, [receipt.path, receipt.path],
                root=self.final_root, team_root=self.team_root)
        (receipt.path / "response.json").write_bytes(_landing(
            awayTeam={"abbrev": "NYR", "score": 3},
            homeTeam={"abbrev": "BOS", "score": 2}))
        with self.assertRaises(HistoricalOfficialFinalError):
            verify_historical_final(
                receipt.path, self.capture.path, root=self.final_root,
                team_root=self.team_root)

    def test_empty_reconciliation_keeps_source_tie_unresolved(self) -> None:
        report = reconcile_historical_finals(
            self.capture.path, [], root=self.final_root,
            team_root=self.team_root)
        self.assertEqual(report.verified_final_count, 0)
        self.assertEqual(report.unresolved_tied_game_ids, (GAME_ID,))
        self.assertFalse(report.final_capture_coverage_complete)
