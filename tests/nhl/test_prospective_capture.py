"""Tests de la capture locale NHL à provenance non certifiée."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from src.nhl.prospective_capture import (
    NHLPregameCaptureError,
    REGISTRY_PATH,
    capture_local_response,
    verify_local_capture,
)

UTC = timezone.utc
OBSERVED = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
RAW = b'{"gameId": 2026020013, "gameState": "FUT"}'
URL = "https://api-web.nhle.com/v1/schedule/2026-10-03"


class ProspectiveCaptureTests(TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.kwargs = {
            "raw_response": RAW,
            "provider_id": "nhl_public_web_api",
            "declared_source_url": URL,
            "target_game_id": 2026020013,
            "scheduled_start_utc": OBSERVED + timedelta(hours=5),
            "information_cutoff_utc": OBSERVED + timedelta(hours=4),
            "root": self.root,
        }

    def capture(self, **changes):
        kwargs = dict(self.kwargs)
        kwargs.update(changes)
        with mock.patch("src.nhl.prospective_capture._now_utc", return_value=OBSERVED):
            return capture_local_response(**kwargs)

    def test_precutoff_archive_is_verifiable_but_never_model_eligible(self):
        receipt = self.capture()
        self.assertTrue(receipt.path.is_file())
        self.assertEqual(receipt.status, "PRE_CUTOFF_LOCAL_COPY_UNVERIFIED")
        document = verify_local_capture(receipt.path)
        self.assertEqual(document["observed_at_utc"], "2026-10-03T12:00:00.000000Z")
        self.assertEqual(document["declared_source_url"], URL)
        self.assertEqual(document["target_game_id_asserted"], 2026020013)
        self.assertEqual(document["source_registry_sha256"], __import__("hashlib").sha256(REGISTRY_PATH.read_bytes()).hexdigest())
        self.assertFalse(document["provider_enabled_at_capture"])
        self.assertFalse(document["source_origin_authenticated"])
        self.assertFalse(document["target_game_in_response_verified"])
        self.assertFalse(document["historical_pregame_availability_proven"])
        self.assertFalse(document["training_permitted"])
        self.assertFalse(document["prediction_publication_permitted"])

    def test_late_copy_is_explicitly_late(self):
        receipt = self.capture(information_cutoff_utc=OBSERVED - timedelta(seconds=1))
        self.assertEqual(receipt.status, "LATE_LOCAL_COPY")
        self.assertFalse(verify_local_capture(receipt.path)["captured_before_cutoff"])

    def test_duplicate_cannot_overwrite_immutable_capture(self):
        first = self.capture()
        original = first.path.read_bytes()
        with self.assertRaisesRegex(NHLPregameCaptureError, "écrasement"):
            self.capture()
        self.assertEqual(first.path.read_bytes(), original)

    def test_rejects_unknown_provider_and_dangerous_url(self):
        for change in (
            {"provider_id": "invented"},
            {"declared_source_url": "http://api-web.nhle.com/v1/schedule"},
            {"declared_source_url": "https://me:secret@api-web.nhle.com/v1/schedule"},
            {"declared_source_url": "https://api-web.nhle.com/v1/schedule?key=secret"},
        ):
            with self.subTest(change=change), self.assertRaises(NHLPregameCaptureError):
                self.capture(**change)
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_rejects_bad_dates_and_non_json(self):
        for change in (
            {"target_game_id": True},
            {"scheduled_start_utc": OBSERVED.replace(tzinfo=None)},
            {"information_cutoff_utc": OBSERVED + timedelta(hours=5)},
            {"raw_response": b"not json"},
            {"raw_response": b"null"},
            {"raw_response": b""},
        ):
            with self.subTest(change=change), self.assertRaises(NHLPregameCaptureError):
                self.capture(**change)
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_detects_content_and_gate_tampering(self):
        receipt = self.capture()
        clean = json.loads(receipt.path.read_text())
        for edit in (
            {"response_base64": "e30="},
            {"training_permitted": True},
            {"captured_before_cutoff": False},
            {"status": "VERIFIED"},
        ):
            modified = dict(clean, **edit)
            receipt.path.write_text(json.dumps(modified))
            with self.subTest(edit=edit), self.assertRaises(NHLPregameCaptureError):
                verify_local_capture(receipt.path)
        receipt.path.write_text(json.dumps(clean))
        self.assertEqual(verify_local_capture(receipt.path)["status"], receipt.status)

    def test_protocol_keeps_all_operational_gates_closed(self):
        protocol = json.loads((REGISTRY_PATH.parent / "nhl19_local_capture_protocol_v1.json").read_text())
        self.assertFalse(protocol["network_collection_enabled"])
        self.assertFalse(protocol["historical_pregame_availability_proven"])
        self.assertFalse(protocol["training_permitted"])
        self.assertFalse(protocol["prediction_publication_permitted"])
