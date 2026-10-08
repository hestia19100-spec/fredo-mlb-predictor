"""Offline tests for the bounded NHL historical-final pilot."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.nhl.historical_final_backlog import inspect_historical_final_backlog
from src.nhl.historical_final_pilot import (
    capture_historical_final_pilot, plan_historical_final_pilot,
)
from src.nhl.historical_official_final import HistoricalOfficialFinalError
from src.nhl.moneypuck_team_capture import capture_team_history
from tests.nhl.test_historical_official_final import (
    FINAL_TIME, GAME_ID, SOURCE_TIME, _Response, _csv, _landing,
)

SECOND_ID = GAME_ID + 1


class HistoricalFinalPilotTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.team_root = self.root / "team"
        self.final_root = self.root / "final"
        source = _csv().decode("utf-8")
        rows = source.strip().splitlines()
        second = [row.replace(str(GAME_ID), str(SECOND_ID)).replace(
            "20211013", "20211014") for row in rows[1:]]
        tied = (chr(10).join([rows[0], *rows[1:], *second]) + chr(10)).encode("utf-8")
        tied = tied.replace(b",4,2,", b",2,2,").replace(b",2,4,", b",2,2,")
        source_file = self.root / "source.csv"
        source_file.write_bytes(tied)
        self.source = capture_team_history(
            explicit_manual_run=True, source_file=source_file,
            code_commit="a" * 40, root=self.team_root,
            now=lambda: SOURCE_TIME,
        )
        self.calls: list[int] = []
        self.fail_id: int | None = None

    def _transport(self, url: str, timeout: int):
        game_id = int(url.split("/")[-2])
        self.calls.append(game_id)
        changes = {}
        if game_id == SECOND_ID:
            changes = {"id": SECOND_ID, "gameDate": "2021-10-14",
                       "startTimeUTC": "2021-10-14T23:00:00Z"}
        if game_id == self.fail_id:
            changes["gameState"] = "LIVE"
        return _Response(url, _landing(**changes))

    def _plan(self, limit: int = 10):
        return plan_historical_final_pilot(
            self.source.path, request_limit=limit,
            root=self.final_root, team_root=self.team_root,
        )

    def _capture(self, digest: str, *, limit: int = 10,
                 manual: bool = True, transport=None):
        return capture_historical_final_pilot(
            self.source.path, expected_backlog_sha256=digest,
            explicit_manual_run=manual, request_limit=limit,
            transport=self._transport if transport is None else transport,
            root=self.final_root, team_root=self.team_root,
            now=lambda: FINAL_TIME,
        )

    def test_plan_is_read_only_and_limits_requests(self) -> None:
        plan = self._plan()
        self.assertEqual(plan.game_ids, (GAME_ID, SECOND_ID))
        self.assertEqual(plan.request_limit, 10)
        self.assertFalse(plan.training_permitted)
        self.assertFalse(plan.prediction_publication_permitted)
        self.assertFalse(self.final_root.exists())
        self.assertEqual(self._plan(1).game_ids, (GAME_ID,))
        for bad in (0, 11, True, 1.0):
            with self.subTest(bad=bad), self.assertRaises(HistoricalOfficialFinalError):
                self._plan(bad)
        self.assertEqual(self.calls, [])

    def test_manual_flag_and_fresh_hash_required_before_network(self) -> None:
        plan = self._plan()
        for digest, manual, transport in (
            (plan.backlog_sha256, False, self._transport),
            ("0" * 64, True, self._transport),
            ("bad", True, self._transport),
            (plan.backlog_sha256, True, None),
        ):
            with self.subTest(digest=digest, manual=manual, transport=transport),                     self.assertRaises(HistoricalOfficialFinalError):
                capture_historical_final_pilot(
                    self.source.path, expected_backlog_sha256=digest,
                    explicit_manual_run=manual, transport=transport,
                    root=self.final_root, team_root=self.team_root,
                )
        self.assertEqual(self.calls, [])
        self.assertFalse(self.final_root.exists())

    def test_two_requests_are_verified_and_resume_skips_them(self) -> None:
        plan = self._plan()
        receipts = self._capture(plan.backlog_sha256)
        self.assertEqual(tuple(item.game_id for item in receipts),
                         (GAME_ID, SECOND_ID))
        self.assertEqual(self.calls, [GAME_ID, SECOND_ID])
        refreshed = self._plan()
        self.assertEqual(refreshed.game_ids, ())
        self.assertEqual(self._capture(refreshed.backlog_sha256), ())
        self.assertEqual(self.calls, [GAME_ID, SECOND_ID])
        backlog = inspect_historical_final_backlog(
            self.source.path, root=self.final_root, team_root=self.team_root,
        )
        self.assertEqual(backlog.verified_game_ids, (GAME_ID, SECOND_ID))

    def test_failure_stops_without_retry_and_next_plan_resumes(self) -> None:
        self.fail_id = SECOND_ID
        initial = self._plan()
        with self.assertRaises(HistoricalOfficialFinalError):
            self._capture(initial.backlog_sha256)
        self.assertEqual(self.calls, [GAME_ID, SECOND_ID])
        resumed = self._plan()
        self.assertEqual(resumed.game_ids, (SECOND_ID,))
        self.assertNotEqual(resumed.backlog_sha256, initial.backlog_sha256)
        with self.assertRaises(HistoricalOfficialFinalError):
            self._capture(initial.backlog_sha256)
        self.assertEqual(self.calls, [GAME_ID, SECOND_ID])
        self.fail_id = None
        receipts = self._capture(resumed.backlog_sha256)
        self.assertEqual(tuple(item.game_id for item in receipts), (SECOND_ID,))
        self.assertEqual(self.calls, [GAME_ID, SECOND_ID, SECOND_ID])

    def test_bad_policy_blocks_before_network(self) -> None:
        bad_policy = self.root / "policy.json"
        bad_policy.write_text("{}", encoding="utf-8")
        with patch("src.nhl.historical_final_pilot.POLICY", bad_policy):
            with self.assertRaises(HistoricalOfficialFinalError):
                self._capture("0" * 64)
        self.assertEqual(self.calls, [])
