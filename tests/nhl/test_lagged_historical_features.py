"""Tests de la forme rétrospective NHL sans fuite de la date cible."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
import unittest

from src.nhl.lagged_historical_features import (
    NHLLaggedHistoryError,
    build_lagged_regular_features,
)
from src.nhl.moneypuck_five_season import select_five_season_history
from tests.nhl.test_moneypuck_five_season import _game, _snapshot


def _regular(season: int, number: int, day: date, home_xg: str = "2"):
    game_id = int(f"{season}02{number:04d}")
    return tuple(replace(
        row, game_id=game_id, game_date=day,
        x_goals_for=Decimal(home_xg if row.team == "BOS" else "1"),
        x_goals_against=Decimal("1" if row.team == "BOS" else home_xg),
    ) for row in _game(season, 2))


def _history(rows):
    return select_five_season_history(_snapshot(rows))


class NHLLaggedHistoryTests(unittest.TestCase):
    def test_same_day_games_cannot_influence_one_another(self) -> None:
        first = date(2022, 1, 2)
        second = date(2022, 1, 3)
        source = _regular(2021, 1, first, "2") + _regular(2021, 2, first, "4")
        source += _regular(2021, 3, second, "8")
        candidate = build_lagged_regular_features(_history(source))
        bos = [row for row in candidate.rows if row.team == "BOS"]
        self.assertEqual([row.prior_games for row in bos], [0, 0, 2])
        self.assertEqual([row.sample_games for row in bos], [0, 0, 2])
        self.assertIsNone(bos[0].x_goals_for_mean)
        self.assertEqual(bos[2].last_prior_date, first)
        self.assertLess(bos[2].last_prior_date, bos[2].game_date)
        self.assertEqual(bos[2].x_goals_for_mean, Decimal("3"))
        self.assertEqual(bos[2].five_on_five_x_goals_for_mean, Decimal("3"))
        self.assertEqual(candidate.file_sha256, "0" * 64)
        self.assertFalse(candidate.labels_included)
        self.assertFalse(candidate.training_permitted)
        self.assertFalse(hasattr(bos[2], "winner"))

    def test_lookback_cap_uses_only_latest_previous_game(self) -> None:
        first = date(2022, 1, 2)
        source = _regular(2021, 1, first, "2") + _regular(2021, 2, first, "4")
        source += _regular(2021, 3, date(2022, 1, 3), "8")
        bos = [row for row in build_lagged_regular_features(
            _history(source), lookback_games=1
        ).rows if row.team == "BOS"]
        self.assertEqual(bos[2].prior_games, 2)
        self.assertEqual(bos[2].sample_games, 1)
        self.assertEqual(bos[2].x_goals_for_mean, Decimal("4"))

    def test_seasons_reset_and_playoffs_never_enter_form(self) -> None:
        source = _regular(2021, 1, date(2022, 1, 2))
        source += _game(2021, 3)
        source += _regular(2022, 1, date(2023, 1, 2))
        candidate = build_lagged_regular_features(_history(source))
        self.assertEqual(len(candidate.rows), 4)
        self.assertEqual([row.prior_games for row in candidate.rows], [0, 0, 0, 0])
        self.assertEqual([row.season for row in candidate.rows], [2021, 2021, 2022, 2022])

    def test_missing_situation_and_playoff_contamination_fail_closed(self) -> None:
        source = _history(_regular(2021, 1, date(2022, 1, 2)))
        with self.assertRaises(NHLLaggedHistoryError):
            build_lagged_regular_features(replace(source, regular_rows=source.regular_rows[:-1]))
        playoff = _history(_game(2021, 3))
        with self.assertRaises(NHLLaggedHistoryError):
            build_lagged_regular_features(replace(playoff, regular_rows=playoff.playoff_rows))

    def test_invalid_window_is_rejected(self) -> None:
        history = _history(_regular(2021, 1, date(2022, 1, 2)))
        for value in (0, 31, True, 1.5):
            with self.subTest(value=value), self.assertRaises(NHLLaggedHistoryError):
                build_lagged_regular_features(history, lookback_games=value)


if __name__ == "__main__":
    unittest.main()
