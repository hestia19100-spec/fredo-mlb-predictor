"""Tests des reprogrammations successives d’un match MLB reporté."""

from dataclasses import replace
import unittest

from src.mlb_api import (
    MLBAPIError,
    ScheduledGame,
    _select_canonical_game,
)


def _postponed_game() -> ScheduledGame:
    """Reproduit la première occurrence reportée de 747090."""
    return ScheduledGame(
        game_id=747090,
        season=2024,
        official_date="2024-09-09",
        game_datetime_utc="2024-07-23T23:20:00Z",
        game_type="R",
        status_code="DR",
        status_detail="Postponed",
        away_team_id=113,
        away_team_name="Cincinnati Reds",
        home_team_id=144,
        home_team_name="Atlanta Braves",
        away_score=None,
        home_score=None,
        venue_id=4705,
        venue_name="Truist Park",
        doubleheader="N",
        game_number=1,
        away_probable_pitcher_id=607259,
        away_probable_pitcher_name="Nick Martinez",
        home_probable_pitcher_id=519242,
        home_probable_pitcher_name="Chris Sale",
    )


class MLBRepeatedPostponementTests(unittest.TestCase):
    """Contrôle le cas 747090 sans relâcher les autres garde-fous."""

    def test_latest_postponed_schedule_is_kept(self) -> None:
        """Le dernier horaire reporté doit devenir l’occurrence canonique."""
        first = _postponed_game()
        second = replace(
            first,
            game_datetime_utc="2024-07-24T22:05:00Z",
            doubleheader="S",
            game_number=2,
        )

        selected = _select_canonical_game(first, second)
        selected_in_reverse_order = _select_canonical_game(
            second,
            first,
        )

        self.assertEqual(selected, second)
        self.assertEqual(selected_in_reverse_order, second)

    def test_other_postponed_difference_is_rejected(self) -> None:
        """Une différence métier autre que la programmation reste interdite."""
        first = _postponed_game()
        conflicting = replace(
            first,
            game_datetime_utc="2024-07-24T22:05:00Z",
            doubleheader="S",
            game_number=2,
            venue_name="Autre stade",
        )

        with self.assertRaises(MLBAPIError):
            _select_canonical_game(first, conflicting)


if __name__ == "__main__":
    unittest.main()
