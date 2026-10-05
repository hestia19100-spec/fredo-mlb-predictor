"""NHL-27: user-supplied schedule copies never impersonate HTTPS captures."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import nhl_schedule_capture_cli as cli
from src.nhl.public_schedule_capture import (
    PublicScheduleCaptureError, ScheduleCaptureReceipt, USER_COPY_POLICY_PATH,
    ingest_user_supplied_schedule, verify_public_schedule_capture,
)
from src.nhl.schedule_capture_readiness import audit_schedule_capture
from src.nhl.prospective_real_readiness import audit_real_pregame_readiness

TARGET = date(2026, 10, 6)
STARTED = datetime(2026, 10, 5, 10, tzinfo=timezone.utc)
OBSERVED = STARTED + timedelta(seconds=2)


def body() -> bytes:
    return json.dumps({"gameWeek": [{"date": TARGET.isoformat(), "games": [{
        "id": 2026020044, "season": 20262027, "gameType": 2,
        "gameState": "FUT", "gameScheduleState": "OK",
        "startTimeUTC": "2026-10-06T23:00:00Z",
        "awayTeam": {"id": 1, "abbrev": "NJD", "odds": [{"value": "+115"}]},
        "homeTeam": {"id": 2, "abbrev": "NYI", "score": 0},
    }]}]}).encode()


class UserScheduleCopyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "user_browser_copy.json"
        self.source.write_bytes(body())

    def ingest(self, **changes):
        ticks = iter((STARTED, OBSERVED))
        kwargs = {"explicit_manual_run": True, "root": self.root / "captures",
                  "now": lambda: next(ticks)}
        kwargs.update(changes)
        return ingest_user_supplied_schedule(TARGET, self.source, **kwargs)

    def test_copy_is_sealed_with_honest_provenance(self) -> None:
        result = self.ingest()
        receipt = verify_public_schedule_capture(result.path)
        self.assertEqual(result.game_count, 1)
        self.assertEqual(receipt["schema_version"], "nhl_public_schedule_user_copy_v1")
        self.assertEqual(receipt["acquisition_mode"], "user_supplied_browser_copy")
        self.assertFalse(receipt["source_origin_independently_verified"])
        self.assertEqual(receipt["observed_at_utc"], "2026-10-05T10:00:02.000000Z")
        self.assertFalse(receipt["training_permitted"])
        self.assertFalse(receipt["prediction_publication_permitted"])
        self.assertNotIn("source_url", receipt)
        self.assertNotIn("odds", json.dumps(receipt))
        self.assertNotIn("score", json.dumps(receipt))
        audit = audit_schedule_capture(result.path)
        self.assertEqual(audit["acquisition_mode"], "user_supplied_browser_copy")
        self.assertEqual(audit["games"][0]["game_id"], 2026020044)
        prospective = audit_real_pregame_readiness(
            result.path, database_path=self.root / "missing.db", allowed_root=self.root,
        )
        self.assertEqual(prospective["status"], "AUDIT_ONLY_USER_SUPPLIED_COPY")
        self.assertEqual(prospective["schedule_acquisition_mode"], "user_supplied_browser_copy")
        self.assertFalse(prospective["prediction_publication_permitted"])

    def test_cli_user_copy_never_invokes_network_capture(self) -> None:
        fake = ScheduleCaptureReceipt(self.root / "sealed", "a" * 64, OBSERVED, 1)
        with (patch("sys.argv", ["nhl_schedule_capture_cli.py", "2026-10-06",
                                 "--user-copy", str(self.source)]),
              patch.object(cli, "ingest_user_supplied_schedule", return_value=fake) as imported,
              patch.object(cli, "capture_public_schedule", side_effect=AssertionError("network")),
              redirect_stdout(StringIO()) as output):
            cli.main()
        imported.assert_called_once_with(TARGET, self.source, explicit_manual_run=True)
        self.assertIn("Matchs réguliers futurs: 1", output.getvalue())

    def test_manual_permission_and_protocol_are_enforced(self) -> None:
        with self.assertRaises(PublicScheduleCaptureError):
            self.ingest(explicit_manual_run=False)
        policy = json.loads(USER_COPY_POLICY_PATH.read_text(encoding="utf-8"))
        policy["prediction_publication_permitted"] = True
        altered = self.root / "unsafe_policy.json"
        altered.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaises(PublicScheduleCaptureError):
            self.ingest(policy_path=altered)
        self.assertFalse((self.root / "captures").exists())

    def test_tampering_with_origin_or_body_is_rejected(self) -> None:
        result = self.ingest()
        path = result.path / "receipt.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["source_origin_independently_verified"] = True
        path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaises(PublicScheduleCaptureError):
            verify_public_schedule_capture(result.path)
        receipt["source_origin_independently_verified"] = False
        path.write_text(json.dumps(receipt), encoding="utf-8")
        (result.path / "response.json").write_bytes(b"{}")
        with self.assertRaises(PublicScheduleCaptureError):
            verify_public_schedule_capture(result.path)

    def test_intake_after_match_start_cannot_backdate_future_game(self) -> None:
        after = datetime(2026, 10, 7, tzinfo=timezone.utc)
        result = self.ingest(now=lambda: after)
        self.assertEqual(result.game_count, 0)
        self.assertEqual(verify_public_schedule_capture(result.path)["future_regular_games"], [])


if __name__ == "__main__":
    unittest.main()
