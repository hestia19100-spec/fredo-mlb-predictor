"""Tests automatiques du lecteur de l’API MLB."""

from datetime import date
import json
import unittest
from unittest.mock import Mock, patch

from src.mlb_api import (
    fetch_schedule,
    fetch_schedule_range,
)


def build_sample_payload() -> dict[str, object]:
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


def configure_mocked_response(
    mocked_get: Mock,
) -> bytes:
    """Configure une réponse HTTP simulée."""
    payload = build_sample_payload()
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