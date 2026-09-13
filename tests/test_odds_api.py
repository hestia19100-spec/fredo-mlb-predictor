"""Tests sans réseau du client The Odds API pour la MLB."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
import unittest
from unittest import mock

from src import odds_api


API_KEY = "secret-test-key-123456789"


def sample_payload() -> list[dict[str, object]]:
    return [
        {
            "id": "event-123",
            "sport_key": "baseball_mlb",
            "sport_title": "MLB",
            "commence_time": "2026-09-13T18:10:00Z",
            "home_team": "Chicago Cubs",
            "away_team": "Pittsburgh Pirates",
            "bookmakers": [
                {
                    "key": "unibet_fr",
                    "title": "Unibet (FR)",
                    "last_update": "2026-09-13T12:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {
                                    "name": "Pittsburgh Pirates",
                                    "price": 2.15,
                                },
                                {
                                    "name": "Chicago Cubs",
                                    "price": 1.74,
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    ]


def response_for(payload: object) -> mock.Mock:
    response = mock.Mock()
    response.status_code = 200
    response.history = []
    response.content = json.dumps(
        payload,
        separators=(",", ":"),
    ).encode("utf-8")
    response.headers = {
        "x-requests-remaining": "499",
        "x-requests-used": "1",
        "x-requests-last": "1",
    }
    response.raise_for_status.return_value = None
    return response


class OddsAPITests(unittest.TestCase):
    def _fetch(self, payload: object | None = None) -> odds_api.OddsFetchResult:
        response = response_for(sample_payload() if payload is None else payload)
        with (
            mock.patch.dict(
                os.environ,
                {odds_api.ODDS_API_KEY_ENVIRONMENT_VARIABLE: API_KEY},
                clear=True,
            ),
            mock.patch.object(
                odds_api.requests,
                "get",
                return_value=response,
                create=True,
            ) as request,
            mock.patch.object(
                odds_api,
                "_utc_now",
                return_value=datetime(
                    2026,
                    9,
                    13,
                    12,
                    0,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
        ):
            result = odds_api.fetch_mlb_moneyline_odds()
        self.request = request
        return result

    def test_request_is_fixed_to_mlb_fr_moneyline_decimal(self) -> None:
        result = self._fetch()
        self.request.assert_called_once_with(
            odds_api.ODDS_API_URL,
            params={
                "apiKey": API_KEY,
                "regions": "fr",
                "markets": "h2h",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
            headers={"User-Agent": "fredo-mlb-predictor/0.1"},
            timeout=30,
            allow_redirects=False,
        )
        self.assertEqual(result.provider, "the_odds_api_v4")
        self.assertEqual(result.sport_key, "baseball_mlb")
        self.assertEqual(result.region, "fr")
        self.assertEqual(result.market, "h2h")
        self.assertEqual(result.odds_format, "decimal")

    def test_valid_response_preserves_exact_prices_and_audit_data(self) -> None:
        result = self._fetch()
        event = result.events[0]
        bookmaker = event.bookmakers[0]
        self.assertEqual(event.provider_event_id, "event-123")
        self.assertEqual(event.away_team_name, "Pittsburgh Pirates")
        self.assertEqual(event.home_team_name, "Chicago Cubs")
        self.assertEqual(
            event.commence_time_utc,
            datetime(2026, 9, 13, 18, 10, tzinfo=timezone.utc),
        )
        self.assertEqual(bookmaker.key, "unibet_fr")
        self.assertEqual(bookmaker.away_decimal_odds, Decimal("2.15"))
        self.assertEqual(bookmaker.home_decimal_odds, Decimal("1.74"))
        self.assertEqual(result.quota_remaining, 499)
        self.assertEqual(result.quota_used, 1)
        self.assertEqual(result.quota_last_cost, 1)
        self.assertEqual(
            result.response_body_sha256,
            hashlib.sha256(result.raw_content).hexdigest(),
        )

    def test_secret_is_required_before_any_network_call(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                odds_api.requests,
                "get",
                create=True,
            ) as request,
        ):
            with self.assertRaisesRegex(
                odds_api.OddsAPIConfigurationError,
                "n’est pas configurée",
            ):
                odds_api.fetch_mlb_moneyline_odds()
        request.assert_not_called()

    def test_configuration_status_never_returns_the_secret(self) -> None:
        """Le contrôle de configuration expose uniquement un booléen."""
        with mock.patch.dict(
            os.environ,
            {odds_api.ODDS_API_KEY_ENVIRONMENT_VARIABLE: "x" * 24},
            clear=True,
        ):
            configured = odds_api.odds_api_key_is_configured()
        self.assertIs(configured, True)

        with mock.patch.dict(os.environ, {}, clear=True):
            missing = odds_api.odds_api_key_is_configured()
        self.assertIs(missing, False)

    def test_secret_never_appears_in_result(self) -> None:
        result = self._fetch()
        self.assertNotIn(API_KEY, repr(result))
        self.assertNotIn(API_KEY.encode("utf-8"), result.raw_content)

    def test_foreign_sport_is_rejected(self) -> None:
        payload = sample_payload()
        payload[0]["sport_key"] = "icehockey_nhl"
        with self.assertRaisesRegex(odds_api.OddsAPIError, "étrangère à la MLB"):
            self._fetch(payload)

    def test_mismatched_outcome_team_is_rejected(self) -> None:
        payload = sample_payload()
        payload[0]["bookmakers"][0]["markets"][0]["outcomes"][0][
            "name"
        ] = "Other Team"
        with self.assertRaisesRegex(
            odds_api.OddsAPIError,
            "ne correspondent pas au match",
        ):
            self._fetch(payload)

    def test_duplicate_event_is_rejected(self) -> None:
        payload = sample_payload()
        payload.append(deepcopy(payload[0]))
        with self.assertRaisesRegex(odds_api.OddsAPIError, "événement.*répété"):
            self._fetch(payload)

    def test_duplicate_bookmaker_is_rejected(self) -> None:
        payload = sample_payload()
        payload[0]["bookmakers"].append(
            deepcopy(payload[0]["bookmakers"][0])
        )
        with self.assertRaisesRegex(odds_api.OddsAPIError, "bookmaker est répété"):
            self._fetch(payload)

    def test_exchange_lay_market_is_explicitly_ignored(self) -> None:
        payload = sample_payload()
        payload[0]["bookmakers"][0]["markets"].append(
            {
                "key": "h2h_lay",
                "outcomes": [
                    {"name": "Pittsburgh Pirates", "price": 2.16},
                    {"name": "Chicago Cubs", "price": 1.75},
                ],
            }
        )

        result = self._fetch(payload)

        bookmaker = result.events[0].bookmakers[0]
        self.assertEqual(bookmaker.away_decimal_odds, Decimal("2.15"))
        self.assertEqual(bookmaker.home_decimal_odds, Decimal("1.74"))

    def test_duplicate_moneyline_market_is_rejected(self) -> None:
        payload = sample_payload()
        payload[0]["bookmakers"][0]["markets"].append(
            deepcopy(payload[0]["bookmakers"][0]["markets"][0])
        )
        with self.assertRaisesRegex(
            odds_api.OddsAPIError,
            "répète le marché h2h",
        ):
            self._fetch(payload)

    def test_duplicate_lay_market_is_rejected(self) -> None:
        payload = sample_payload()
        lay_market = {
            "key": "h2h_lay",
            "outcomes": [],
        }
        payload[0]["bookmakers"][0]["markets"].extend(
            [lay_market, deepcopy(lay_market)]
        )
        with self.assertRaisesRegex(
            odds_api.OddsAPIError,
            "répète le marché h2h_lay",
        ):
            self._fetch(payload)

    def test_unexpected_market_is_rejected(self) -> None:
        payload = sample_payload()
        payload[0]["bookmakers"][0]["markets"].append(
            {"key": "spreads", "outcomes": []}
        )
        with self.assertRaisesRegex(
            odds_api.OddsAPIError,
            "marché inattendu spreads",
        ):
            self._fetch(payload)

    def test_lay_market_without_moneyline_is_rejected(self) -> None:
        payload = sample_payload()
        payload[0]["bookmakers"][0]["markets"] = [
            {"key": "h2h_lay", "outcomes": []}
        ]
        with self.assertRaisesRegex(
            odds_api.OddsAPIError,
            "ne contient pas le marché Moneyline h2h",
        ):
            self._fetch(payload)

    def test_non_decimal_or_even_price_is_rejected(self) -> None:
        for price in (1.0, "2.15", True):
            with self.subTest(price=price):
                payload = sample_payload()
                payload[0]["bookmakers"][0]["markets"][0]["outcomes"][0][
                    "price"
                ] = price
                with self.assertRaises(odds_api.OddsAPIError):
                    self._fetch(payload)

    def test_invalid_quota_header_is_rejected(self) -> None:
        response = response_for(sample_payload())
        response.headers["x-requests-remaining"] = "unknown"
        with (
            mock.patch.dict(
                os.environ,
                {odds_api.ODDS_API_KEY_ENVIRONMENT_VARIABLE: API_KEY},
                clear=True,
            ),
            mock.patch.object(
                odds_api.requests,
                "get",
                return_value=response,
                create=True,
            ),
            mock.patch.object(
                odds_api,
                "_utc_now",
                return_value=datetime.now(timezone.utc),
            ),
        ):
            with self.assertRaisesRegex(
                odds_api.OddsAPIError,
                "x-requests-remaining",
            ):
                odds_api.fetch_mlb_moneyline_odds()

    def test_unexpected_quota_cost_is_rejected(self) -> None:
        response = response_for(sample_payload())
        response.headers["x-requests-last"] = "2"
        with (
            mock.patch.dict(
                os.environ,
                {odds_api.ODDS_API_KEY_ENVIRONMENT_VARIABLE: API_KEY},
                clear=True,
            ),
            mock.patch.object(
                odds_api.requests,
                "get",
                return_value=response,
                create=True,
            ),
            mock.patch.object(
                odds_api,
                "_utc_now",
                return_value=datetime.now(timezone.utc),
            ),
        ):
            with self.assertRaisesRegex(
                odds_api.OddsAPIError,
                "nombre inattendu de crédits",
            ):
                odds_api.fetch_mlb_moneyline_odds()

    def test_http_error_never_exposes_secret(self) -> None:
        class RequestException(Exception):
            pass

        class Timeout(RequestException):
            pass

        class ConnectionError(RequestException):
            pass

        class HTTPError(RequestException):
            def __init__(self) -> None:
                super().__init__(f"429 URL apiKey={API_KEY}")
                self.response = mock.Mock(status_code=429)

        response = response_for(sample_payload())
        response.raise_for_status.side_effect = HTTPError()
        with (
            mock.patch.dict(
                os.environ,
                {odds_api.ODDS_API_KEY_ENVIRONMENT_VARIABLE: API_KEY},
                clear=True,
            ),
            mock.patch.object(
                odds_api.requests,
                "get",
                return_value=response,
                create=True,
            ),
            mock.patch.object(
                odds_api.requests,
                "RequestException",
                RequestException,
                create=True,
            ),
            mock.patch.object(
                odds_api.requests,
                "Timeout",
                Timeout,
                create=True,
            ),
            mock.patch.object(
                odds_api.requests,
                "ConnectionError",
                ConnectionError,
                create=True,
            ),
            mock.patch.object(
                odds_api.requests,
                "HTTPError",
                HTTPError,
                create=True,
            ),
        ):
            with self.assertRaises(odds_api.OddsAPIRetryableError) as caught:
                odds_api.fetch_mlb_moneyline_odds()
        self.assertNotIn(API_KEY, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
