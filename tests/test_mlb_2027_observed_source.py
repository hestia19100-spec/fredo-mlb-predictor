"""Contrôles source MLB 2027 et chronologie T−60, sans publication."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from email.utils import format_datetime
import hashlib
import json
from urllib.parse import urlencode
import unittest

from src.mlb_api import (
    MLB_SCHEDULE_URL, ScheduleFetchResult, _parse_schedule_payload,
)
from src.mlb_2027_observed_source import (
    MLB2027SourceError,
    candidate_has_prospective_timing,
    verify_2027_observed_source,
)
from src.mlb_2027_reference_predictor import calculate_reference_probabilities


TARGET = date(2027, 1, 20)


def _game(game_id: int, when: str, status: str, *, score: bool) -> dict:
    teams = {
        "away": {"team": {"id": 1, "name": "Away"}},
        "home": {"team": {"id": 2, "name": "Home"}},
    }
    if score:
        teams["away"]["score"] = 4
        teams["home"]["score"] = 2
    return {
        "gamePk": game_id, "season": 2027, "officialDate": when,
        "gameDate": when + "T23:00:00Z", "gameType": "R",
        "status": {
            "statusCode": "F" if score else "S",
            "detailedState": status,
            "abstractGameState": "Final" if score else "Preview",
        },
        "teams": teams,
    }


def _observed(*, target_status: str = "Scheduled") -> ScheduleFetchResult:
    prior = _game(100, "2027-01-10", "Final", score=True)
    target = _game(200, "2027-01-20", target_status, score=False)
    payload = {
        "dates": [
            {"date": "2027-01-10", "games": [prior]},
            {"date": "2027-01-20", "games": [target]},
        ],
        "totalGames": 2,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    parameters = {
        "sportId": 1, "startDate": "2027-01-01",
        "endDate": "2027-01-20", "gameTypes": "R",
        "hydrate": "probablePitcher",
    }
    observed_at = datetime(2027, 1, 20, 19, tzinfo=timezone.utc)
    return ScheduleFetchResult(
        start_date=date(2027, 1, 1), end_date=TARGET,
        game_types=("R",), request_parameters=parameters,
        raw_content=raw, games=tuple(_parse_schedule_payload(
            payload, reject_duplicate_game_ids=True,
        )),
        response_effective_url=MLB_SCHEDULE_URL + "?" + urlencode(parameters),
        response_status_code=200, response_redirect_count=0,
        mlb_http_date_header_raw=format_datetime(observed_at, usegmt=True),
        mlb_http_date_utc="2027-01-20T19:00:00Z",
        mlb_http_response_received_at_utc="2027-01-20T19:00:01Z",
        response_body_sha256=hashlib.sha256(raw).hexdigest(),
    )


class ObservedSourceTests(unittest.TestCase):
    def test_links_raw_mlb_response_to_candidate_and_timing(self) -> None:
        source = verify_2027_observed_source(TARGET, (_observed(),))
        self.assertEqual(len(source.upcoming_games), 1)
        self.assertEqual(len(source.prior_final_games), 1)
        self.assertEqual(
            source.receipt_sha256,
            hashlib.sha256(source.receipt_bytes).hexdigest(),
        )
        candidate = calculate_reference_probabilities(
            TARGET, source.upcoming_games, source.prior_final_games,
            source_receipt_sha256=source.receipt_sha256,
        )
        self.assertEqual(candidate.abstained_count, 1)
        self.assertTrue(candidate_has_prospective_timing(
            source, candidate,
            issued_at_utc=datetime(2027, 1, 20, 20, tzinfo=timezone.utc),
            completed_at_utc=datetime(2027, 1, 20, 21, tzinfo=timezone.utc),
            remote_http_date_utc=datetime(
                2027, 1, 20, 21, 1, tzinfo=timezone.utc,
            ),
        ))

    def test_late_remote_publication_is_not_prospective(self) -> None:
        source = verify_2027_observed_source(TARGET, (_observed(),))
        candidate = calculate_reference_probabilities(
            TARGET, source.upcoming_games, source.prior_final_games,
            source_receipt_sha256=source.receipt_sha256,
        )
        self.assertFalse(candidate_has_prospective_timing(
            source, candidate,
            issued_at_utc=datetime(2027, 1, 20, 20, tzinfo=timezone.utc),
            completed_at_utc=datetime(2027, 1, 20, 21, tzinfo=timezone.utc),
            remote_http_date_utc=datetime(
                2027, 1, 20, 22, 1, tzinfo=timezone.utc,
            ),
        ))

    def test_rejects_tampered_raw_body(self) -> None:
        observed = _observed()
        with self.assertRaises(MLB2027SourceError):
            verify_2027_observed_source(
                TARGET, (replace(observed, raw_content=b"{}"),),
            )

    def test_rejects_missing_start_of_season(self) -> None:
        observed = _observed()
        with self.assertRaises(MLB2027SourceError):
            verify_2027_observed_source(
                TARGET, (replace(observed, start_date=date(2027, 1, 2)),),
            )

    def test_rejects_started_or_unknown_target_status(self) -> None:
        with self.assertRaises(MLB2027SourceError):
            verify_2027_observed_source(
                TARGET, (_observed(target_status="In Progress"),),
            )

    def test_rejects_capture_after_issuance(self) -> None:
        source = verify_2027_observed_source(TARGET, (_observed(),))
        candidate = calculate_reference_probabilities(
            TARGET, source.upcoming_games, source.prior_final_games,
            source_receipt_sha256=source.receipt_sha256,
        )
        with self.assertRaises(MLB2027SourceError):
            candidate_has_prospective_timing(
                source, candidate,
                issued_at_utc=datetime(
                    2027, 1, 20, 18, tzinfo=timezone.utc,
                ),
                completed_at_utc=datetime(
                    2027, 1, 20, 21, tzinfo=timezone.utc,
                ),
                remote_http_date_utc=datetime(
                    2027, 1, 20, 21, 1, tzinfo=timezone.utc,
                ),
            )


if __name__ == "__main__":
    unittest.main()
