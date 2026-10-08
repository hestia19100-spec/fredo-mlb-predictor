"""Fail-closed, offline tests of the one-match historical-final backlog."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.nhl.historical_final_backlog import (
    capture_next_tied_final, inspect_historical_final_backlog,
)
from src.nhl.historical_official_final import (
    HistoricalOfficialFinalError, capture_historical_final,
)
from src.nhl.moneypuck_team_capture import capture_team_history
from tests.nhl.test_historical_official_final import (
    FINAL_TIME, GAME_ID, SOURCE_TIME, _Response, _csv, _landing,
)


class HistoricalFinalBacklogTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.team_root = self.root / "team"
        self.final_root = self.root / "final"
        source_file = self.root / "source.csv"
        source_file.write_bytes(_csv().replace(b",4,2,", b",2,2,").replace(b",2,4,", b",2,2,"))
        self.source = capture_team_history(
            explicit_manual_run=True, source_file=source_file,
            code_commit="a" * 40, root=self.team_root,
            now=lambda: SOURCE_TIME,
        )
        self.calls = []

    def _transport(self, url: str, timeout: int):
        self.calls.append((url, timeout))
        return _Response(url, _landing())

    def _inspect(self):
        return inspect_historical_final_backlog(
            self.source.path, root=self.final_root, team_root=self.team_root,
        )

    def _next(self, **kwargs):
        return capture_next_tied_final(
            self.source.path, root=self.final_root, team_root=self.team_root,
            now=lambda: FINAL_TIME, **kwargs,
        )

    def test_read_only_inventory_and_single_explicit_request(self):
        first = self._inspect()
        self.assertEqual(first.pending_tied_game_ids, (GAME_ID,))
        self.assertEqual(first.next_game_id, GAME_ID)
        self.assertEqual(first.verified_game_ids, ())
        self.assertFalse(first.training_permitted)
        self.assertFalse(first.prediction_publication_permitted)
        self.assertFalse(self.final_root.exists())
        with self.assertRaises(HistoricalOfficialFinalError):
            self._next(transport=self._transport)
        self.assertEqual(self.calls, [])
        receipt = self._next(explicit_manual_run=True, transport=self._transport)
        self.assertEqual(receipt.game_id, GAME_ID)
        self.assertEqual(len(self.calls), 1)
        second = self._inspect()
        self.assertEqual(second.pending_tied_game_ids, ())
        self.assertEqual(second.verified_game_ids, (GAME_ID,))
        self.assertNotEqual(first.backlog_sha256, second.backlog_sha256)
        self.assertIsNone(self._next(explicit_manual_run=True, transport=self._transport))
        self.assertEqual(len(self.calls), 1)

    def test_incomplete_archive_blocks_request(self):
        incomplete = self.final_root / "2021" / str(GAME_ID) / "incomplete"
        incomplete.mkdir(parents=True)
        with self.assertRaises(HistoricalOfficialFinalError):
            self._next(explicit_manual_run=True, transport=self._transport)
        self.assertEqual(self.calls, [])

    def test_duplicate_or_tampered_proof_blocks_inventory(self):
        first = self._next(explicit_manual_run=True, transport=self._transport)
        capture_historical_final(
            self.source.path, GAME_ID, explicit_manual_run=True,
            transport=self._transport, root=self.final_root,
            team_root=self.team_root, now=lambda: FINAL_TIME + timedelta(seconds=1),
        )
        with self.assertRaises(HistoricalOfficialFinalError):
            self._inspect()
        self.assertEqual(len(self.calls), 2)
        # A malformed completed archive is also rejected, not silently skipped.
        (first.path / "response.json").write_bytes(b"{}")
        with self.assertRaises(HistoricalOfficialFinalError):
            self._inspect()

    def test_incompatible_policy_blocks_without_network(self):
        bad_policy = self.root / "policy.json"
        bad_policy.write_text("{}", encoding="utf-8")
        with patch("src.nhl.historical_final_backlog.POLICY", bad_policy):
            with self.assertRaises(HistoricalOfficialFinalError):
                self._next(explicit_manual_run=True, transport=self._transport)
        self.assertEqual(self.calls, [])

    def test_empty_game_directory_blocks_inventory(self):
        (self.final_root / "2021" / str(GAME_ID)).mkdir(parents=True)
        with self.assertRaises(HistoricalOfficialFinalError):
            self._next(explicit_manual_run=True, transport=self._transport)
        self.assertEqual(self.calls, [])

    def test_unexpected_season_blocks_inventory(self):
        (self.final_root / "1999").mkdir(parents=True)
        with self.assertRaises(HistoricalOfficialFinalError):
            self._inspect()
