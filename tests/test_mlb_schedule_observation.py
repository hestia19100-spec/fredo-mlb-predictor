"""Teste les preuves temporelles du calendrier MLB prospectif."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.ingestion_service import run_observed_schedule_ingestion
from src.mlb_api import (
    MLBAPIError,
    ScheduledGame,
    ScheduleFetchResult,
    fetch_schedule,
    fetch_schedule_range_observed,
)


HTTP_DATE = "Mon, 31 Aug 2026 06:30:00 GMT"
HTTP_DATE_UTC = "2026-08-31T06:30:00Z"
RECEIVED_AT = datetime(2026, 8, 31, 6, 30, 2, tzinfo=timezone.utc)
RECEIVED_AT_UTC = "2026-08-31T06:30:02Z"
EFFECTIVE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule?"
    "sportId=1&startDate=2026-09-01&endDate=2026-09-01&"
    "gameTypes=R&hydrate=probablePitcher"
)


def _payload() -> dict[str, object]:
    """Construit un match futur minimal avec son état abstrait."""
    return {
        "totalGames": 1,
        "dates": [
            {
                "date": "2026-09-01",
                "games": [
                    {
                        "gamePk": 123456,
                        "season": "2026",
                        "officialDate": "2026-09-01",
                        "gameDate": "2026-09-01T23:10:00Z",
                        "gameType": "R",
                        "status": {
                            "abstractGameState": "Preview",
                            "statusCode": "S",
                            "detailedState": "Scheduled",
                        },
                        "teams": {
                            "away": {
                                "team": {"id": 100, "name": "Away Team"},
                            },
                            "home": {
                                "team": {"id": 200, "name": "Home Team"},
                            },
                        },
                        "venue": {"id": 300, "name": "Test Ballpark"},
                        "doubleHeader": "N",
                        "gameNumber": 1,
                    }
                ],
            }
        ],
    }


def _raw_payload() -> bytes:
    """Sérialise la réponse simulée sans la modifier."""
    return json.dumps(
        _payload(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _configure_response(
    mocked_get: Mock,
    *,
    http_date: object = HTTP_DATE,
    effective_url: object = EFFECTIVE_URL,
    status_code: object = 200,
    history: object = (),
) -> bytes:
    """Configure une réponse requests minimale."""
    raw_content = _raw_payload()
    response = Mock()
    response.raise_for_status.return_value = None
    response.content = raw_content
    response.headers = {"Date": http_date}
    response.url = effective_url
    response.status_code = status_code
    response.history = history
    mocked_get.return_value = response
    return raw_content


class MLBScheduleObservationTests(unittest.TestCase):
    """Verrouille les preuves fournies par le même appel HTTP."""

    @patch("src.mlb_api._utc_now", return_value=RECEIVED_AT)
    @patch("src.mlb_api.requests.get")
    def test_observed_fetch_keeps_http_time_and_abstract_state(
        self,
        mocked_get: Mock,
        mocked_now: Mock,
    ) -> None:
        """L'appel prospectif doit conserver ses deux horodatages."""
        expected_raw = _configure_response(mocked_get)

        result = fetch_schedule_range_observed(
            date(2026, 9, 1),
            date(2026, 9, 1),
        )

        self.assertEqual(result.raw_content, expected_raw)
        self.assertEqual(result.response_effective_url, EFFECTIVE_URL)
        self.assertEqual(result.response_status_code, 200)
        self.assertEqual(result.response_redirect_count, 0)
        self.assertEqual(result.mlb_http_date_header_raw, HTTP_DATE)
        self.assertEqual(result.mlb_http_date_utc, HTTP_DATE_UTC)
        self.assertEqual(
            result.mlb_http_response_received_at_utc,
            RECEIVED_AT_UTC,
        )
        self.assertEqual(
            result.response_body_sha256,
            hashlib.sha256(expected_raw).hexdigest(),
        )
        self.assertEqual(result.games[0].abstract_state, "Preview")
        mocked_now.assert_called_once_with()
        self.assertFalse(mocked_get.call_args.kwargs["allow_redirects"])

    @patch("src.mlb_api._utc_now", return_value=RECEIVED_AT)
    @patch("src.mlb_api.requests.get")
    def test_observed_fetch_rejects_missing_or_invalid_http_date(
        self,
        mocked_get: Mock,
        mocked_now: Mock,
    ) -> None:
        """Une preuve HTTP absente ou illisible interdit le shadow."""
        for invalid_value in (
            None,
            "",
            "date invalide",
            "Mon, 31 Aug 2026 06:30:00 +0000",
        ):
            with self.subTest(invalid_value=invalid_value):
                _configure_response(mocked_get, http_date=invalid_value)
                with self.assertRaises(MLBAPIError):
                    fetch_schedule_range_observed(
                        date(2026, 9, 1),
                        date(2026, 9, 1),
                    )

        self.assertEqual(mocked_now.call_count, 4)

    @patch("src.mlb_api._utc_now", return_value=RECEIVED_AT)
    @patch("src.mlb_api.requests.get")
    def test_observed_fetch_rejects_non_exact_http_response(
        self,
        mocked_get: Mock,
        mocked_now: Mock,
    ) -> None:
        """Statut, redirection, endpoint et query doivent être exacts."""
        invalid_responses = (
            {"status_code": 302},
            {"history": (Mock(),)},
            {
                "effective_url": EFFECTIVE_URL.replace(
                    "https://statsapi.mlb.com",
                    "http://statsapi.mlb.com",
                )
            },
            {
                "effective_url": EFFECTIVE_URL.replace(
                    "statsapi.mlb.com",
                    "example.com",
                )
            },
            {
                "effective_url": EFFECTIVE_URL.replace(
                    "/api/v1/schedule",
                    "/api/v1/people",
                )
            },
            {"effective_url": EFFECTIVE_URL + "&sportId=1"},
            {
                "effective_url": EFFECTIVE_URL.replace(
                    "gameTypes=R",
                    "gameTypes=S",
                )
            },
        )

        for overrides in invalid_responses:
            with self.subTest(overrides=overrides):
                _configure_response(mocked_get, **overrides)
                with self.assertRaises(MLBAPIError):
                    fetch_schedule_range_observed(
                        date(2026, 9, 1),
                        date(2026, 9, 1),
                    )

        self.assertEqual(mocked_now.call_count, len(invalid_responses))

    @patch("src.mlb_api._utc_now", return_value=RECEIVED_AT)
    @patch("src.mlb_api.requests.get")
    def test_observed_fetch_rejects_excessive_clock_skew(
        self,
        mocked_get: Mock,
        mocked_now: Mock,
    ) -> None:
        """La preuve serveur et l'horloge locale doivent rester proches."""
        _configure_response(
            mocked_get,
            http_date="Mon, 31 Aug 2026 06:20:00 GMT",
        )

        with self.assertRaisesRegex(MLBAPIError, "300 secondes"):
            fetch_schedule_range_observed(
                date(2026, 9, 1),
                date(2026, 9, 1),
            )

        mocked_now.assert_called_once_with()

    @patch(
        "src.mlb_api._utc_now",
        return_value=datetime(
            2026,
            8,
            31,
            6,
            35,
            0,
            999999,
            tzinfo=timezone.utc,
        ),
    )
    @patch("src.mlb_api.requests.get")
    def test_observed_fetch_accepts_inclusive_300_second_boundary(
        self,
        mocked_get: Mock,
        mocked_now: Mock,
    ) -> None:
        """La limite enregistrée de 300 secondes reste inclusive."""
        _configure_response(mocked_get)

        result = fetch_schedule_range_observed(
            date(2026, 9, 1),
            date(2026, 9, 1),
        )

        self.assertEqual(
            result.mlb_http_response_received_at_utc,
            "2026-08-31T06:35:00Z",
        )
        mocked_now.assert_called_once_with()

    @patch("src.mlb_api._utc_now", return_value=RECEIVED_AT)
    @patch("src.mlb_api.requests.get")
    def test_observed_fetch_accepts_missing_start_time_as_none(
        self,
        mocked_get: Mock,
        mocked_now: Mock,
    ) -> None:
        """Un horaire absent reste nullable pour être exclu plus tard."""
        payload = _payload()
        raw_game = payload["dates"][0]["games"][0]  # type: ignore[index]
        del raw_game["gameDate"]  # type: ignore[index]
        raw_content = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        _configure_response(mocked_get)
        mocked_get.return_value.content = raw_content

        result = fetch_schedule_range_observed(
            date(2026, 9, 1),
            date(2026, 9, 1),
        )

        self.assertIsNone(result.games[0].game_datetime_utc)
        mocked_now.assert_called_once_with()

    @patch("src.mlb_api._utc_now", return_value=RECEIVED_AT)
    @patch("src.mlb_api.requests.get")
    def test_observed_fetch_rejects_duplicate_game_ids(
        self,
        mocked_get: Mock,
        mocked_now: Mock,
    ) -> None:
        """Un snapshot prospectif ne fusionne jamais deux occurrences."""
        payload = _payload()
        duplicate = json.loads(
            json.dumps(payload["dates"][0]["games"][0])  # type: ignore[index]
        )
        payload["dates"][0]["games"].append(duplicate)  # type: ignore[index]
        payload["totalGames"] = 2
        raw_content = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        _configure_response(mocked_get)
        mocked_get.return_value.content = raw_content

        with self.assertRaisesRegex(MLBAPIError, "doublon"):
            fetch_schedule_range_observed(
                date(2026, 9, 1),
                date(2026, 9, 1),
            )

        mocked_now.assert_called_once_with()

    @patch("src.mlb_api.requests.get")
    def test_legacy_fetch_remains_compatible_without_http_date(
        self,
        mocked_get: Mock,
    ) -> None:
        """L'interface historique ne devient pas plus stricte."""
        _configure_response(mocked_get, http_date=None)
        games = fetch_schedule(date(2026, 9, 1))
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0].abstract_state, "Preview")


class ObservedIngestionTests(unittest.TestCase):
    """Vérifie la propagation des preuves jusqu'au service audité."""

    @patch("src.ingestion_service.fetch_schedule_range_observed")
    def test_observed_ingestion_propagates_same_http_evidence(
        self,
        mocked_fetch: Mock,
    ) -> None:
        """Le résultat final doit exposer les preuves du fetch archivé."""
        game = ScheduledGame(
            game_id=123456,
            season=2026,
            official_date="2026-09-01",
            game_datetime_utc="2026-09-01T23:10:00Z",
            game_type="R",
            status_code="S",
            status_detail="Scheduled",
            away_team_id=100,
            away_team_name="Away Team",
            home_team_id=200,
            home_team_name="Home Team",
            away_score=None,
            home_score=None,
            venue_id=300,
            venue_name="Test Ballpark",
            doubleheader="N",
            game_number=1,
            away_probable_pitcher_id=None,
            away_probable_pitcher_name=None,
            home_probable_pitcher_id=None,
            home_probable_pitcher_name=None,
            abstract_state="Preview",
        )
        parameters = {
            "sportId": 1,
            "startDate": "2026-09-01",
            "endDate": "2026-09-01",
            "gameTypes": "R",
            "hydrate": "probablePitcher",
        }
        mocked_fetch.return_value = ScheduleFetchResult(
            start_date=date(2026, 9, 1),
            end_date=date(2026, 9, 1),
            game_types=("R",),
            request_parameters=parameters,
            raw_content=_raw_payload(),
            games=(game,),
            response_effective_url=EFFECTIVE_URL,
            response_status_code=200,
            response_redirect_count=0,
            mlb_http_date_header_raw=HTTP_DATE,
            mlb_http_date_utc=HTTP_DATE_UTC,
            mlb_http_response_received_at_utc=RECEIVED_AT_UTC,
            response_body_sha256=hashlib.sha256(
                _raw_payload()
            ).hexdigest(),
        )

        with TemporaryDirectory(
            ignore_cleanup_errors=True,
        ) as temporary_directory:
            root = Path(temporary_directory)
            result = run_observed_schedule_ingestion(
                start_date=date(2026, 9, 1),
                end_date=date(2026, 9, 1),
                database_path=root / "test.db",
                data_directory=root / "data",
                code_version="a" * 40,
            )

        self.assertEqual(result.mlb_http_date_utc, HTTP_DATE_UTC)
        self.assertEqual(
            result.mlb_http_response_received_at_utc,
            RECEIVED_AT_UTC,
        )
        self.assertEqual(result.games, (game,))
        self.assertEqual(result.response_effective_url, EFFECTIVE_URL)
        self.assertEqual(result.response_status_code, 200)
        self.assertEqual(result.response_redirect_count, 0)
        self.assertEqual(result.mlb_http_date_header_raw, HTTP_DATE)
        self.assertEqual(
            result.response_body_sha256,
            hashlib.sha256(_raw_payload()).hexdigest(),
        )
        mocked_fetch.assert_called_once_with(
            date(2026, 9, 1),
            date(2026, 9, 1),
            game_types=("R",),
        )

    @patch("src.ingestion_service.fetch_schedule_range_observed")
    def test_observed_ingestion_rejects_missing_evidence(
        self,
        mocked_fetch: Mock,
    ) -> None:
        """Une collecte sans preuve HTTP doit finir en erreur auditée."""
        mocked_fetch.return_value = ScheduleFetchResult(
            start_date=date(2026, 9, 1),
            end_date=date(2026, 9, 1),
            game_types=("R",),
            request_parameters={
                "sportId": 1,
                "startDate": "2026-09-01",
                "endDate": "2026-09-01",
                "gameTypes": "R",
                "hydrate": "probablePitcher",
            },
            raw_content=_raw_payload(),
            games=(),
        )

        with TemporaryDirectory(
            ignore_cleanup_errors=True,
        ) as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(
                RuntimeError,
                "preuves HTTP",
            ):
                run_observed_schedule_ingestion(
                    start_date=date(2026, 9, 1),
                    end_date=date(2026, 9, 1),
                    database_path=root / "test.db",
                    data_directory=root / "data",
                    code_version="a" * 40,
                )

    @patch("src.ingestion_service.fetch_schedule_range_observed")
    def test_observed_ingestion_rejects_body_archive_hash_mismatch(
        self,
        mocked_fetch: Mock,
    ) -> None:
        """Le corps observé doit être exactement celui de l'archive."""
        mocked_fetch.return_value = ScheduleFetchResult(
            start_date=date(2026, 9, 1),
            end_date=date(2026, 9, 1),
            game_types=("R",),
            request_parameters={
                "sportId": 1,
                "startDate": "2026-09-01",
                "endDate": "2026-09-01",
                "gameTypes": "R",
                "hydrate": "probablePitcher",
            },
            raw_content=_raw_payload(),
            games=(),
            response_effective_url=EFFECTIVE_URL,
            response_status_code=200,
            response_redirect_count=0,
            mlb_http_date_header_raw=HTTP_DATE,
            mlb_http_date_utc=HTTP_DATE_UTC,
            mlb_http_response_received_at_utc=RECEIVED_AT_UTC,
            response_body_sha256="0" * 64,
        )

        with TemporaryDirectory(
            ignore_cleanup_errors=True,
        ) as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(
                RuntimeError,
                "diffère de l'archive brute",
            ):
                run_observed_schedule_ingestion(
                    start_date=date(2026, 9, 1),
                    end_date=date(2026, 9, 1),
                    database_path=root / "test.db",
                    data_directory=root / "data",
                    code_version="a" * 40,
                )


if __name__ == "__main__":
    unittest.main()
