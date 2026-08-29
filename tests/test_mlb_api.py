"""Tests automatiques du lecteur de l’API MLB."""

from copy import deepcopy
from datetime import date
import json
from typing import Any
import unittest
from unittest.mock import Mock, patch

import requests

from src.mlb_api import (
    MLBAPIError,
    MLBAPIRetryableError,
    fetch_schedule,
    fetch_schedule_range,
)


def build_sample_payload() -> dict[str, Any]:
    """Construit une réponse MLB minimale et stable."""
    return {
        "totalGames": 1,
        "dates": [
            {
                "date": "2026-08-28",
                "games": [
                    {
                        "gamePk": 123456,
                        "season": "2026",
                        "officialDate": "2026-08-28",
                        "gameDate": "2026-08-28T23:10:00Z",
                        "gameType": "R",
                        "status": {
                            "statusCode": "S",
                            "detailedState": "Scheduled",
                        },
                        "teams": {
                            "away": {
                                "team": {
                                    "id": 100,
                                    "name": "Away Team",
                                },
                                "probablePitcher": {
                                    "id": 400,
                                    "fullName": "Away Pitcher",
                                },
                            },
                            "home": {
                                "team": {
                                    "id": 200,
                                    "name": "Home Team",
                                },
                                "probablePitcher": {
                                    "id": 500,
                                    "fullName": "Home Pitcher",
                                },
                            },
                        },
                        "venue": {
                            "id": 300,
                            "name": "Test Ballpark",
                        },
                        "doubleHeader": "N",
                        "gameNumber": 1,
                    }
                ],
            }
        ],
    }


def build_rescheduled_payload() -> dict[str, Any]:
    """Construit un match reporté puis terminé le lendemain."""
    sample_payload = build_sample_payload()
    sample_game = sample_payload["dates"][0]["games"][0]

    postponed_game = deepcopy(sample_game)
    postponed_game["season"] = "2021"
    postponed_game["officialDate"] = "2021-04-01"
    postponed_game["gameDate"] = "2021-04-01T18:10:00Z"
    postponed_game["status"] = {
        "statusCode": "D",
        "detailedState": "Postponed",
    }

    final_game = deepcopy(postponed_game)
    final_game["officialDate"] = "2021-04-02"
    final_game["gameDate"] = "2021-04-02T18:10:00Z"
    final_game["status"] = {
        "statusCode": "F",
        "detailedState": "Final",
    }
    final_game["teams"]["away"]["score"] = 3
    final_game["teams"]["home"]["score"] = 0

    return {
        "totalGames": 2,
        "dates": [
            {
                "date": "2021-04-01",
                "games": [postponed_game],
            },
            {
                "date": "2021-04-02",
                "games": [final_game],
            },
        ],
    }


def configure_mocked_payload(
    mocked_get: Mock,
    payload: dict[str, Any],
) -> bytes:
    """Configure une réponse HTTP simulée avec un contenu précis."""
    raw_content = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    mocked_response = Mock()
    mocked_response.raise_for_status.return_value = None
    mocked_response.content = raw_content
    mocked_get.return_value = mocked_response

    return raw_content


def configure_mocked_response(
    mocked_get: Mock,
) -> bytes:
    """Configure une réponse HTTP simulée."""
    return configure_mocked_payload(
        mocked_get,
        build_sample_payload(),
    )


class MLBAPITests(unittest.TestCase):
    """Vérifie la transformation des réponses MLB."""

    @patch("src.mlb_api.requests.get")
    def test_fetch_schedule_parses_one_game(
        self,
        mocked_get: Mock,
    ) -> None:
        """Une journée valide doit produire un match contrôlé."""
        configure_mocked_response(mocked_get)

        games = fetch_schedule(date(2026, 8, 28))

        self.assertEqual(len(games), 1)

        game = games[0]

        self.assertEqual(game.game_id, 123456)
        self.assertEqual(game.season, 2026)
        self.assertEqual(game.away_team_id, 100)
        self.assertEqual(game.away_team_name, "Away Team")
        self.assertEqual(game.home_team_id, 200)
        self.assertEqual(game.home_team_name, "Home Team")
        self.assertEqual(game.status_detail, "Scheduled")
        self.assertEqual(game.venue_name, "Test Ballpark")
        self.assertIsNone(game.away_score)
        self.assertIsNone(game.home_score)
        self.assertEqual(game.away_probable_pitcher_id, 400)
        self.assertEqual(
            game.away_probable_pitcher_name,
            "Away Pitcher",
        )
        self.assertEqual(game.home_probable_pitcher_id, 500)
        self.assertEqual(
            game.home_probable_pitcher_name,
            "Home Pitcher",
        )

        parameters = mocked_get.call_args.kwargs["params"]

        self.assertEqual(parameters["sportId"], 1)
        self.assertEqual(parameters["date"], "2026-08-28")
        self.assertEqual(
            parameters["hydrate"],
            "probablePitcher",
        )

    @patch("src.mlb_api.requests.get")
    def test_fetch_schedule_range_preserves_raw_response(
        self,
        mocked_get: Mock,
    ) -> None:
        """Une période doit conserver exactement la réponse brute."""
        expected_raw_content = configure_mocked_response(
            mocked_get
        )

        result = fetch_schedule_range(
            date(2026, 8, 27),
            date(2026, 8, 28),
            game_types=("R",),
        )

        self.assertEqual(result.start_date, date(2026, 8, 27))
        self.assertEqual(result.end_date, date(2026, 8, 28))
        self.assertEqual(result.game_types, ("R",))
        self.assertEqual(result.raw_content, expected_raw_content)
        self.assertEqual(len(result.games), 1)

        parameters = mocked_get.call_args.kwargs["params"]

        self.assertEqual(parameters["sportId"], 1)
        self.assertEqual(parameters["startDate"], "2026-08-27")
        self.assertEqual(parameters["endDate"], "2026-08-28")
        self.assertEqual(parameters["gameTypes"], "R")

    @patch("src.mlb_api.requests.get")
    def test_rescheduled_game_uses_final_occurrence(
        self,
        mocked_get: Mock,
    ) -> None:
        """Un report suivi d’une finale doit produire un match unique."""
        payload = build_rescheduled_payload()
        expected_raw_content = configure_mocked_payload(
            mocked_get,
            payload,
        )

        result = fetch_schedule_range(
            date(2021, 3, 4),
            date(2021, 4, 3),
        )

        self.assertEqual(result.raw_content, expected_raw_content)
        self.assertEqual(len(result.games), 1)

        game = result.games[0]

        self.assertEqual(game.game_id, 123456)
        self.assertEqual(game.official_date, "2021-04-02")
        self.assertEqual(
            game.game_datetime_utc,
            "2021-04-02T18:10:00Z",
        )
        self.assertEqual(game.status_code, "F")
        self.assertEqual(game.status_detail, "Final")
        self.assertEqual(game.away_score, 3)
        self.assertEqual(game.home_score, 0)

    @patch("src.mlb_api.requests.get")
    def test_duplicate_with_different_identity_is_rejected(
        self,
        mocked_get: Mock,
    ) -> None:
        """Un identifiant partagé par deux équipes doit être refusé."""
        payload = build_rescheduled_payload()
        second_game = payload["dates"][1]["games"][0]
        second_game["teams"]["home"]["team"] = {
            "id": 201,
            "name": "Other Home Team",
        }
        configure_mocked_payload(mocked_get, payload)

        with self.assertRaisesRegex(
            MLBAPIError,
            "Occurrences contradictoires",
        ):
            fetch_schedule_range(
                date(2021, 3, 4),
                date(2021, 4, 3),
            )

    @patch("src.mlb_api.requests.get")
    def test_conflicting_final_occurrences_are_rejected(
        self,
        mocked_get: Mock,
    ) -> None:
        """Deux résultats finaux différents doivent être refusés."""
        payload = build_rescheduled_payload()
        first_game = payload["dates"][0]["games"][0]
        first_game["status"] = {
            "statusCode": "F",
            "detailedState": "Final",
        }
        first_game["teams"]["away"]["score"] = 2
        first_game["teams"]["home"]["score"] = 1
        configure_mocked_payload(mocked_get, payload)

        with self.assertRaisesRegex(
            MLBAPIError,
            "Occurrences contradictoires",
        ):
            fetch_schedule_range(
                date(2021, 3, 4),
                date(2021, 4, 3),
            )

    @patch("src.mlb_api.requests.get")
    def test_timeout_is_retryable(
        self,
        mocked_get: Mock,
    ) -> None:
        """Un délai réseau dépassé doit autoriser une reprise."""
        mocked_get.side_effect = requests.Timeout(
            "Délai dépassé."
        )

        with self.assertRaises(MLBAPIRetryableError):
            fetch_schedule(date(2026, 8, 28))

    @patch("src.mlb_api.requests.get")
    def test_http_503_is_retryable(
        self,
        mocked_get: Mock,
    ) -> None:
        """Une erreur serveur MLB doit autoriser une reprise."""
        mocked_response = Mock()
        mocked_response.status_code = 503
        mocked_response.raise_for_status.side_effect = (
            requests.HTTPError(
                "Service indisponible.",
                response=mocked_response,
            )
        )
        mocked_get.return_value = mocked_response

        with self.assertRaises(MLBAPIRetryableError):
            fetch_schedule(date(2026, 8, 28))

    @patch("src.mlb_api.requests.get")
    def test_http_404_is_not_retryable(
        self,
        mocked_get: Mock,
    ) -> None:
        """Une erreur permanente ne doit pas être retentée."""
        mocked_response = Mock()
        mocked_response.status_code = 404
        mocked_response.raise_for_status.side_effect = (
            requests.HTTPError(
                "Ressource absente.",
                response=mocked_response,
            )
        )
        mocked_get.return_value = mocked_response

        with self.assertRaises(MLBAPIError) as raised:
            fetch_schedule(date(2026, 8, 28))

        self.assertNotIsInstance(
            raised.exception,
            MLBAPIRetryableError,
        )

    def test_fetch_schedule_range_rejects_more_than_31_days(
        self,
    ) -> None:
        """Une requête trop grande doit être refusée localement."""
        with self.assertRaisesRegex(ValueError, "31 jours"):
            fetch_schedule_range(
                date(2026, 1, 1),
                date(2026, 2, 1),
            )


if __name__ == "__main__":
    unittest.main()