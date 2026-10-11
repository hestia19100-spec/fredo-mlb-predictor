"""Tests for the immutable historical NHL source selection."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.nhl.frozen_historical_source import (
    FrozenHistoricalSourceError,
    audit_frozen_historical_source,
)

SOURCE = "a" * 64
OTHER = "b" * 64


class FrozenHistoricalSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.team_root = self.root / "teams"
        self.final_root = self.root / "finals"
        self.team_root.mkdir()
        self.final_root.mkdir()

    def _team(self, name: str, digest: str) -> Path:
        slot = self.team_root / name
        slot.mkdir()
        (slot / "COMPLETED").touch()
        (slot / "receipt.json").write_text(
            json.dumps({"response_sha256": digest}), encoding="utf-8"
        )
        return slot

    def _final(self, name: str, digest: str) -> Path:
        slot = self.final_root / name
        slot.mkdir()
        (slot / "receipt.json").write_text(
            json.dumps({"source_capture_sha256": digest}), encoding="utf-8"
        )
        return slot

    def _audit(self, final_slots: list[Path]):
        source = SimpleNamespace(
            path=self.team_root / "older",
            response_sha256=SOURCE,
            observed_at_utc=datetime(2026, 10, 7, tzinfo=timezone.utc),
            history=object(),
        )
        backlog = SimpleNamespace(
            source_capture_sha256=SOURCE,
            verified_game_ids=(123,),
            pending_tied_game_ids=(),
            backlog_sha256="c" * 64,
        )
        readiness = SimpleNamespace(
            regular_coverage_complete=True,
            inconsistent_game_ids=(),
            unresolved_tied_game_ids=(123,),
            audit_sha256="d" * 64,
            seasons=(
                SimpleNamespace(role="TRAIN", both_teams_have_prior_form=8),
                SimpleNamespace(role="VALIDATION", both_teams_have_prior_form=3),
                SimpleNamespace(role="HOLDOUT", both_teams_have_prior_form=2),
            ),
        )
        with (
            patch("src.nhl.frozen_historical_source._slots", return_value=final_slots),
            patch("src.nhl.frozen_historical_source.verify_team_capture", return_value=source) as verify,
            patch("src.nhl.frozen_historical_source.inspect_historical_final_backlog", return_value=backlog),
            patch("src.nhl.frozen_historical_source.audit_historical_backtest_readiness", return_value=readiness),
        ):
            result = audit_frozen_historical_source(
                team_root=self.team_root, final_root=self.final_root
            )
        verify.assert_called_once_with(self.team_root / "older", allowed_root=self.team_root)
        return result

    def test_selects_bound_capture_not_newest(self) -> None:
        self._team("older", SOURCE)
        self._team("newer", OTHER)
        final = self._final("123", SOURCE)
        result = self._audit([final])
        self.assertEqual(result.team_slot, self.team_root / "older")
        self.assertEqual(result.verified_official_tie_finals, 1)
        self.assertEqual(
            (result.candidate_train_games, result.candidate_validation_games,
             result.candidate_holdout_games), (8, 3, 2)
        )
        self.assertFalse(result.training_permitted)
        self.assertFalse(result.prediction_publication_permitted)

    def test_rejects_mixed_source_receipts(self) -> None:
        self._team("older", SOURCE)
        first = self._final("123", SOURCE)
        second = self._final("456", OTHER)
        with patch("src.nhl.frozen_historical_source._slots", return_value=[first, second]):
            with self.assertRaisesRegex(FrozenHistoricalSourceError, "plusieurs captures"):
                audit_frozen_historical_source(
                    team_root=self.team_root, final_root=self.final_root
                )

    def test_rejects_missing_reference_capture(self) -> None:
        self._team("newer", OTHER)
        final = self._final("123", SOURCE)
        with patch("src.nhl.frozen_historical_source._slots", return_value=[final]):
            with self.assertRaisesRegex(FrozenHistoricalSourceError, "absente ou ambiguë"):
                audit_frozen_historical_source(
                    team_root=self.team_root, final_root=self.final_root
                )


if __name__ == "__main__":
    unittest.main()
