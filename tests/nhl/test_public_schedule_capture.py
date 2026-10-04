"""Tests de la capture NHL-20 sans accès au réseau."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.nhl.public_schedule_capture import (
    BASE_URL,
    POLICY_PATH,
    PublicScheduleCaptureError,
    capture_public_schedule,
    verify_public_schedule_capture,
)

TARGET = date(2026, 10, 3)
STARTED = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
OBSERVED = STARTED + timedelta(seconds=1)


def body() -> bytes:
    game = {
        "id": 2026020022,
        "season": 20262027,
        "gameType": 2,
        "gameState": "FUT",
        "gameScheduleState": "OK",
        "startTimeUTC": "2026-10-03T23:00:00Z",
        "awayTeam": {"abbrev": "CHI", "odds": [{"value": "+190"}]},
        "homeTeam": {"abbrev": "BUF", "score": 0},
    }
    return json.dumps({"gameWeek": [{"date": "2026-10-03", "games": [game]}]}).encode()


class FakeResponse(BytesIO):
    def __init__(self, payload: bytes, *, url: str = BASE_URL + "2026-10-03",
                 content_type: str = "application/json", status: int = 200) -> None:
        super().__init__(payload)
        self.url = url
        self.headers = {"Content-Type": content_type}
        self.status = status

    def geturl(self) -> str:
        return self.url


class ScheduleCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.requests = []

    def transport(self, request, timeout):
        self.requests.append((request, timeout))
        return FakeResponse(body())

    def clock(self):
        values = iter((STARTED, OBSERVED))
        return lambda: next(values)

    def capture(self, **changes):
        kwargs = {
            "explicit_manual_run": True,
            "root": self.root,
            "transport": self.transport,
            "now": self.clock(),
        }
        kwargs.update(changes)
        return capture_public_schedule(TARGET, **kwargs)

    def test_manual_capture_is_private_complete_and_verifiable(self) -> None:
        result = self.capture()
        self.assertEqual(result.game_count, 1)
        self.assertEqual(len(self.requests), 1)
        url, timeout = self.requests[0]
        self.assertEqual(url, BASE_URL + "2026-10-03")
        self.assertEqual(timeout, 15)
        self.assertEqual((result.path / "response.json").read_bytes(), body())
        self.assertTrue((result.path / "COMPLETED").is_file())
        receipt = verify_public_schedule_capture(result.path)
        self.assertEqual(receipt["response_sha256"], hashlib.sha256(body()).hexdigest())
        self.assertEqual(receipt["observed_at_utc"], "2026-10-03T12:00:01.000000Z")
        self.assertEqual(receipt["future_regular_games"][0]["game_id"], 2026020022)
        self.assertFalse(receipt["training_permitted"])
        self.assertFalse(receipt["prediction_publication_permitted"])
        self.assertNotIn("odds", json.dumps(receipt))
        self.assertNotIn("score", json.dumps(receipt))

    def test_no_implicit_network_or_second_write(self) -> None:
        with self.assertRaises(PublicScheduleCaptureError):
            capture_public_schedule(TARGET, root=self.root, transport=self.transport)
        self.assertEqual(self.requests, [])
        first = self.capture()
        with self.assertRaises(PublicScheduleCaptureError):
            self.capture()
        self.assertEqual((first.path / "response.json").read_bytes(), body())
        self.assertEqual(len(self.requests), 2)

    def test_redirect_content_type_or_failed_status_rejected_before_archive(self) -> None:
        for changes in (
            {"url": "https://example.org/elsewhere"},
            {"content_type": "text/html"},
            {"status": 503},
        ):
            with self.subTest(changes=changes):
                def bad_transport(request, timeout):
                    return FakeResponse(body(), **changes)
                with self.assertRaises(PublicScheduleCaptureError):
                    self.capture(transport=bad_transport)
        self.assertFalse(self.root.joinpath(TARGET.isoformat()).exists())

    def test_tampering_is_detected_offline(self) -> None:
        result = self.capture()
        (result.path / "response.json").write_bytes(body() + b" ")
        with self.assertRaises(PublicScheduleCaptureError):
            verify_public_schedule_capture(result.path)

    def test_disabled_policy_rejects_before_request(self) -> None:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        self.assertFalse(policy["automatic_daily_collection_allowed"])
        self.assertFalse(policy["source_registry_v1_provider_enabled"])
        policy["training_permitted_from_this_capture"] = True
        mutated = self.root / "mutated_policy.json"
        mutated.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaises(PublicScheduleCaptureError):
            self.capture(policy_path=mutated)
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
