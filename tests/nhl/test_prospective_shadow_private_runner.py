"""Offline checks for the private NHL shadow scheduler."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.nhl.prospective_shadow_observation import _digest
from tests.nhl.test_prospective_shadow_seal import observation
from src.nhl.prospective_shadow_private_runner import (
    NHLPrivateShadowRunnerError, run_private_shadow,
)

NOW = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
GAME = {
    "game_id": 2026020099, "season": 20262027,
    "scheduled_start_utc": "2026-10-12T20:00:00Z",
    "away_abbr": "NYR", "home_abbr": "BOS",
}


class PrivateShadowRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.private = root / "fredo-lpf-edge-private-ops"
        self.private.mkdir()
        (self.private / ".git").mkdir()
        self.schedules = root / "schedules"
        self.schedules.mkdir()
        self.imports = root / "imports"
        self.imports.mkdir()
        self.capture_root = root / "team-captures"
        self.capture_root.mkdir()

    def run_it(self):
        return run_private_shadow(
            private_repository_root=self.private, now=lambda: NOW,
            schedule_root=self.schedules, import_root=self.imports,
            capture_root=self.capture_root)

    def schedule(self):
        slot = self.schedules / "slot"
        slot.mkdir()
        (slot / "COMPLETED").touch()
        return slot

    def imported(self):
        slot = self.imports / "2026" / "slot"
        slot.mkdir(parents=True)
        (slot / "COMPLETED").touch()
        return slot

    def test_without_verified_future_schedule_no_prediction_or_write(self):
        with patch(
            "src.nhl.prospective_shadow_private_runner.build_prospective_shadow_observation"
        ) as builder:
            self.assertEqual(self.run_it().future_games, 0)
            builder.assert_not_called()

    def test_future_schedule_without_import_stays_inactive(self):
        self.schedule()
        audit = {
            "observed_at_utc": "2026-10-11T10:00:00Z",
            "games": [GAME],
        }
        with (
            patch("src.nhl.prospective_shadow_private_runner.audit_schedule_capture",
                  return_value=audit),
            patch("src.nhl.prospective_shadow_private_runner.build_prospective_shadow_observation"
                  ) as builder,
        ):
            summary = self.run_it()
        self.assertEqual(summary.missing_evidence_games, 1)
        self.assertEqual(summary.sealed_games, 0)
        builder.assert_not_called()

    def test_verified_inputs_seal_only_to_private_checkout(self):
        self.schedule()
        self.imported()
        audit = {
            "observed_at_utc": "2026-10-11T10:00:00Z",
            "games": [GAME],
        }
        current = SimpleNamespace(
            season=2026, effective_available_at_utc=
            datetime(2026, 10, 11, 11, tzinfo=timezone.utc))
        observation = SimpleNamespace(scheduled_start_utc=GAME["scheduled_start_utc"])
        with (
            patch("src.nhl.prospective_shadow_private_runner.audit_schedule_capture",
                  return_value=audit),
            patch("src.nhl.prospective_shadow_private_runner.verify_current_season_import",
                  return_value=current),
            patch("src.nhl.prospective_shadow_private_runner.build_prospective_shadow_observation",
                  return_value=observation) as builder,
            patch("src.nhl.prospective_shadow_private_runner.seal_private_shadow_observation"
                  ) as seal,
        ):
            summary = self.run_it()
        self.assertEqual((summary.future_games, summary.eligible_games,
                          summary.sealed_games), (1, 1, 1))
        self.assertIs(builder.call_args.kwargs["explicit_shadow_run"], True)
        self.assertEqual(seal.call_args.kwargs["private_repository_root"], self.private)
        self.assertIs(seal.call_args.kwargs["explicit_private_seal"], True)

    def test_existing_private_receipt_is_reused_without_retraining(self):
        self.schedule()
        self.imported()
        slot = (self.private / "captures" / "nhl" / "shadow"
                / "2026-10-12" / str(GAME["game_id"]))
        slot.mkdir(parents=True)
        fields = asdict(observation())
        fields.pop("observation_sha256")
        fields["scheduled_start_utc"] = "2026-10-12T20:00:00.000000Z"
        fields["observed_at_utc"] = "2026-10-11T11:00:00.000000Z"
        fields["information_cutoff_utc"] = "2026-10-12T19:00:00.000000Z"
        fields["observation_sha256"] = _digest(fields)
        (slot / "observation.json").write_text(json.dumps(fields), encoding="utf-8")
        (slot / "COMPLETED").touch()
        audit = {
            "observed_at_utc": "2026-10-11T10:00:00Z",
            "games": [GAME],
        }
        current = SimpleNamespace(
            season=2026, effective_available_at_utc=
            datetime(2026, 10, 11, 11, tzinfo=timezone.utc))
        with (
            patch("src.nhl.prospective_shadow_private_runner.audit_schedule_capture",
                  return_value=audit),
            patch("src.nhl.prospective_shadow_private_runner.verify_current_season_import",
                  return_value=current),
            patch("src.nhl.prospective_shadow_private_runner.build_prospective_shadow_observation"
                  ) as builder,
            patch("src.nhl.prospective_shadow_private_runner.seal_private_shadow_observation"
                  ) as seal,
        ):
            summary = self.run_it()
        self.assertEqual(summary.already_sealed_games, 1)
        self.assertEqual(summary.sealed_games, 0)
        builder.assert_not_called()
        seal.assert_not_called()

    def test_one_hour_cutoff_and_old_games_are_excluded(self):
        self.schedule()
        boundary = dict(GAME, scheduled_start_utc="2026-10-11T13:00:00Z")
        old = dict(GAME, game_id=2026020001,
                   scheduled_start_utc="2026-10-11T11:00:00Z")
        audit = {"observed_at_utc": "2026-10-11T10:00:00Z",
                 "games": [boundary, old]}
        with patch("src.nhl.prospective_shadow_private_runner.audit_schedule_capture",
                   return_value=audit):
            self.assertEqual(self.run_it().future_games, 0)

    def test_reject_wrong_private_checkout(self):
        wrong = self.private.parent / "public"
        wrong.mkdir()
        (wrong / ".git").mkdir()
        with self.assertRaisesRegex(NHLPrivateShadowRunnerError, "distinct"):
            run_private_shadow(
                private_repository_root=wrong, now=lambda: NOW,
                schedule_root=self.schedules, import_root=self.imports)


if __name__ == "__main__":
    unittest.main()
