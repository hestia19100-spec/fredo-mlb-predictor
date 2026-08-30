"""Tests des matchs MLB suspendus puis terminés."""

from __future__ import annotations

import unittest

from src.mlb_api import (
    MLBAPIError,
    _parse_schedule_payload,
)


class MLBSuspendedGameTests(unittest.TestCase):
    """Contrôle les occurrences finales d’un match suspendu."""

    @staticmethod
    def _final_occurrence(
        game_datetime_utc: str,
    ) -> dict[str, object]:
        """Construit une occurrence du match Arizona-Cincinnati."""
        return {
            "gamePk": 634408,
            "season": 2021,
            "officialDate": "2021-04-20",
            "gameDate": game_datetime_utc,
            "gameType": "R",
            "status": {
                "statusCode": "F",
                "codedGameState": "F",
                "detailedState": "Final",
            },
            "teams": {
                "away": {
                    "team": {
                        "id": 109,
                        "name": "Arizona Diamondbacks",
                    },
                    "score": 5,
                    "probablePitcher": {
                        "id": 605200,
                        "fullName": "Zac Gallen",
                    },
                },
                "home": {
                    "team": {
                        "id": 113,
                        "name": "Cincinnati Reds",
                    },
                    "score": 4,
                    "probablePitcher": {
                        "id": 622491,
                        "fullName": "Luis Castillo",
                    },
                },
            },
            "venue": {
                "id": 2602,
                "name": "Great American Ball Park",
            },
            "doubleHeader": "N",
            "gameNumber": 1,
        }

    def _suspended_game_payload(
        self,
    ) -> dict[str, object]:
        """Reproduit les deux occurrences renvoyées par MLB."""
        first_occurrence = self._final_occurrence(
            "2021-04-20T22:40:00Z"
        )

        resumed_occurrence = self._final_occurrence(
            "2021-04-21T21:10:00Z"
        )

        return {
            "totalGames": 2,
            "dates": [
                {
                    "date": "2021-04-20",
                    "games": [
                        first_occurrence,
                    ],
                },
                {
                    "date": "2021-04-21",
                    "games": [
                        resumed_occurrence,
                    ],
                },
            ],
        }

    def test_suspended_final_game_keeps_original_start_time(
        self,
    ) -> None:
        """
        Deux finales identiques sauf l’horaire doivent
        produire un seul match avec son début initial.
        """
        games = _parse_schedule_payload(
            self._suspended_game_payload()
        )

        self.assertEqual(
            len(games),
            1,
        )

        game = games[0]

        self.assertEqual(
            game.game_id,
            634408,
        )

        self.assertEqual(
            game.official_date,
            "2021-04-20",
        )

        self.assertEqual(
            game.game_datetime_utc,
            "2021-04-20T22:40:00Z",
        )

        self.assertEqual(
            game.away_score,
            5,
        )

        self.assertEqual(
            game.home_score,
            4,
        )

    def test_other_final_metadata_difference_is_rejected(
        self,
    ) -> None:
        """
        Une différence autre que l’horaire doit
        continuer à bloquer la fusion.
        """
        payload = self._suspended_game_payload()

        second_game = (
            payload["dates"][1]["games"][0]
        )

        second_game["venue"]["name"] = (
            "Stade contradictoire"
        )

        with self.assertRaisesRegex(
            MLBAPIError,
            "Occurrences contradictoires",
        ):
            _parse_schedule_payload(
                payload
            )


if __name__ == "__main__":
    unittest.main()