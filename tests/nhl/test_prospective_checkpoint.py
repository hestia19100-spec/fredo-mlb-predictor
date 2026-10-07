"""NHL-31: only pre-cutoff descriptive evidence can be sealed."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.nhl.prospective_checkpoint import (
    NHLCheckpointError, build_checkpoint, seal_checkpoint, verify_checkpoint,
)

UTC = timezone.utc
SEALED = datetime(2026, 10, 7, 20, tzinfo=UTC)


def report() -> dict[str, object]:
    game = {
        "game_id": 2026020053, "away_abbr": "PIT", "home_abbr": "WSH",
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
    return {
        "schema_version": "nhl26_prospective_readiness_v1",
        "schedule_acquisition_mode": "direct_https",
        "schedule_observed_at_utc": "2026-10-07T18:52:00Z",
        "schedule_response_sha256": "c" * 64,
        "target_date": "2026-10-07", "lead_minutes": 60,
        "current_season_import_verified": True,
        "game_count": 1, "games": [game],
        "training_permitted": False, "prediction_publication_permitted": False,
    }


class ProspectiveCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_projection_contains_no_probabilities_odds_or_results(self) -> None:
        proof = build_checkpoint(report(), SEALED)
        self.assertEqual(proof["lead_minutes"], 60)
        self.assertEqual(proof["status"], "DESCRIPTIVE_ONLY")
        self.assertEqual(proof["games"][0]["feature_sha256"], "a" * 64)
        self.assertEqual(proof["games"][0]["form_effective_available_at_utc"],
                         "2026-10-07T13:03:00.000000Z")
        self.assertNotIn("probability", json.dumps(proof).lower())
        self.assertNotIn("goals_for", json.dumps(proof))
        self.assertFalse(proof["training_permitted"])
        self.assertFalse(proof["prediction_publication_permitted"])

    def test_late_seal_or_two_hour_cutoff_is_rejected(self) -> None:
        with self.assertRaises(NHLCheckpointError):
            build_checkpoint(report(), datetime(2026, 10, 7, 22, 31, tzinfo=UTC))
        wrong = report()
        wrong["games"][0]["information_cutoff_utc"] = "2026-10-07T21:30:00Z"
        with self.assertRaises(NHLCheckpointError):
            build_checkpoint(wrong, SEALED)
        wrong = report()
        wrong["lead_minutes"] = 120
        with self.assertRaises(NHLCheckpointError):
            build_checkpoint(wrong, SEALED)

    def test_late_source_and_untrusted_schedule_are_rejected(self) -> None:
        wrong = report()
        wrong["games"][0]["current_season_import_before_cutoff"] = False
        with self.assertRaises(NHLCheckpointError):
            build_checkpoint(wrong, SEALED)
        wrong = report()
        wrong["schedule_acquisition_mode"] = "user_supplied_browser_copy"
        with self.assertRaises(NHLCheckpointError):
            build_checkpoint(wrong, SEALED)
        wrong = report()
        wrong["training_permitted"] = True
        with self.assertRaises(NHLCheckpointError):
            build_checkpoint(wrong, SEALED)

    def test_append_only_receipt_and_tamper_detection(self) -> None:
        with patch("src.nhl.prospective_checkpoint.audit_real_pregame_readiness",
                   return_value=report()) as audit:
            path = seal_checkpoint(self.root / "schedule", (self.root / "import",),
                                   explicit_manual_run=True, root=self.root / "proofs",
                                   now=lambda: SEALED)
            self.assertEqual(audit.call_args.kwargs["lead_minutes"], 60)
            self.assertEqual(verify_checkpoint(path)["games"][0]["game_id"], 2026020053)
            with self.assertRaises(NHLCheckpointError):
                seal_checkpoint(self.root / "schedule", (self.root / "import",),
                                explicit_manual_run=True, root=self.root / "proofs",
                                now=lambda: SEALED)
        document = json.loads(path.read_text())
        document["games"][0]["feature_sha256"] = "d" * 64
        path.write_text(json.dumps(document))
        with self.assertRaises(NHLCheckpointError):
            verify_checkpoint(path)

    def test_manual_gate_and_policy_gate(self) -> None:
        with self.assertRaises(NHLCheckpointError):
            seal_checkpoint(self.root / "schedule", (), root=self.root)
        from src.nhl.prospective_checkpoint import POLICY
        unsafe = self.root / "unsafe.json"
        document = json.loads(POLICY.read_text())
        document["training_permitted"] = True
        unsafe.write_text(json.dumps(document))
        with patch("src.nhl.prospective_checkpoint.POLICY", unsafe):
            with self.assertRaises(NHLCheckpointError):
                build_checkpoint(report(), SEALED)


if __name__ == "__main__":
    unittest.main()
