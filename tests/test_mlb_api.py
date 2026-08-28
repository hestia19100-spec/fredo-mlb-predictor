"""Tests automatiques du lecteur de l’API MLB."""

from datetime import date
import unittest
from unittest.mock import Mock, patch

from src.mlb_api import fetch_schedule


class MLBAPITests(unittest.TestCase):
    """Vérifie la transformation des réponses MLB."""

    @patch("src.mlb_api.requests.get")
    def test_fetch_schedule_parses_one_game(
        self,
        mocked_get: Mock,
    ) -> None:
        """Une réponse MLB valide doit produire un match contrôlé."""
        mocked_response = Mock()
        mocked_response.raise_for_status.return_value = None
        mocked_response.json.return_value = {
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
                                    }
                                },
                                "home": {
                                    "team": {
                                        "id": 200,
                                        "name": "Home Team",
                                    }
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
            ]
        }
        mocked_get.return_value = mocked_response

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

        request_parameters = mocked_get.call_args.kwargs["params"]

        self.assertEqual(request_parameters["sportId"], 1)
        self.assertEqual(request_parameters["date"], "2026-08-28")
        mocked_response.raise_for_status.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()