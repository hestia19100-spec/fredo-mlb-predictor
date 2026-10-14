"""Vérifie le scellement privé des cotes avec le candidat, hors réseau."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.mlb_2027_candidate_archive import (
    MLB2027ArchiveError,
    archive_reference_candidate,
    verify_candidate_archive,
)
from src.mlb_2027_market_match import match_mlb_2027_market
from src.odds_api import OddsFetchResult, _parse_payload
from tests.test_mlb_2027_candidate_archive import _candidate


RECEIVED = datetime(2027, 1, 20, 20, 30, tzinfo=timezone.utc)
ISSUED = datetime(2027, 1, 20, 20, 45, tzinfo=timezone.utc)


def _odds() -> OddsFetchResult:
    raw = json.dumps([{
        "id": "provider-game-200", "sport_key": "baseball_mlb",
        "commence_time": "2027-01-20T23:00:00Z",
        "away_team": "Away", "home_team": "Home",
        "bookmakers": [{
            "key": "french-book", "title": "Book FR",
            "last_update": "2027-01-20T20:29:00Z",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": "Away", "price": 1.9},
                {"name": "Home", "price": 1.95},
            ]}],
        }],
    }], separators=(",", ":")).encode("utf-8")
    return OddsFetchResult(
        provider="the_odds_api_v4", sport_key="baseball_mlb",
        region="fr", market="h2h", odds_format="decimal",
        response_received_at_utc=RECEIVED,
        response_body_sha256=hashlib.sha256(raw).hexdigest(),
        quota_remaining=499, quota_used=1, quota_last_cost=1,
        raw_content=raw, events=_parse_payload(raw),
    )


class MarketArchiveTests(unittest.TestCase):
    def test_raw_market_and_candidate_share_one_manifest(self):
        source, calculation = _candidate()
        odds = _odds()
        market = match_mlb_2027_market(source, odds)
        with TemporaryDirectory() as folder:
            archive = archive_reference_candidate(
                Path(folder), source, calculation, issued_at_utc=ISSUED,
                market=market, odds=odds,
            )
            manifest = verify_candidate_archive(
                archive.path, expected_manifest_sha256=archive.manifest_sha256,
            )
            self.assertEqual((archive.path / "odds-response.json").read_bytes(), odds.raw_content)
            self.assertIn("market.json", manifest["files"])
            candidate = json.loads((archive.path / "candidate.json").read_bytes())
            self.assertEqual(candidate["market"]["matched_game_count"], 1)
            self.assertEqual(candidate["status"], "LOCAL_ONLY_UNPUBLISHED")
            (archive.path / "odds-response.json").write_bytes(b"tampered")
            with self.assertRaises(MLB2027ArchiveError):
                verify_candidate_archive(
                    archive.path, expected_manifest_sha256=archive.manifest_sha256,
                )

    def test_rejects_market_without_its_original_odds(self):
        source, calculation = _candidate()
        market = match_mlb_2027_market(source, _odds())
        with TemporaryDirectory() as folder:
            with self.assertRaises(MLB2027ArchiveError):
                archive_reference_candidate(
                    Path(folder), source, calculation,
                    issued_at_utc=ISSUED, market=market,
                )


if __name__ == "__main__":
    unittest.main()
