"""Tests de la fenêtre MoneyPuck 2021-2025 et de sa séparation sportive."""

from __future__ import annotations

import ast
import unittest
from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.nhl.moneypuck_five_season import (
    EXPECTED_REGULAR_GAMES_PER_SEASON,
    NHLFiveSeasonAuditError,
    REFERENCE_SEASONS,
    select_five_season_history,
)
from src.nhl.moneypuck_team_import import MoneyPuckTeamSnapshot, TeamGameRow


def _game(season: int, game_type: int) -> tuple[TeamGameRow, ...]:
    game_id = int(f"{season}{game_type:02d}0001")
    result = []
    for situation in ("all", "5on5"):
        for side, team, opponent, scored, conceded in (
            ("HOME", "BOS", "NYR", 2, 1),
            ("AWAY", "NYR", "BOS", 1, 2),
        ):
            result.append(TeamGameRow(
                game_id=game_id, season=season, game_date=date(season + 1, 1, 2),
                team=team, opponent=opponent, home_or_away=side,
                situation=situation, x_goals_for=Decimal(scored),
                x_goals_against=Decimal(conceded), goals_for=scored,
                goals_against=conceded, shots_on_goal_for=Decimal(20),
                shots_on_goal_against=Decimal(18),
            ))
    return tuple(result)


def _snapshot(rows: tuple[TeamGameRow, ...]) -> MoneyPuckTeamSnapshot:
    return MoneyPuckTeamSnapshot(
        observed_at_utc=datetime(2026, 10, 2, tzinfo=timezone.utc),
        file_sha256="0" * 64,
        source_page="https://moneypuck.com/data.htm",
        download_url="https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv",
        attribution="Données : MoneyPuck.com", rows=rows,
    )


class MoneyPuckFiveSeasonTests(unittest.TestCase):
    def test_separates_five_seasons_regular_and_playoffs(self) -> None:
        rows = (
            _game(2020, 2) + _game(2021, 2) + _game(2021, 3)
            + _game(2025, 2) + _game(2026, 2)
        )
        selected = select_five_season_history(_snapshot(rows))
        self.assertEqual(REFERENCE_SEASONS, (2021, 2022, 2023, 2024, 2025))
        self.assertEqual(len(selected.regular_rows), 8)
        self.assertEqual(len(selected.playoff_rows), 4)
        self.assertEqual([item.regular_games for item in selected.coverage], [1, 0, 0, 0, 1])
        self.assertEqual([item.playoff_games for item in selected.coverage], [1, 0, 0, 0, 0])
        self.assertFalse(selected.regular_coverage_complete)
        self.assertFalse(selected.training_permitted)
        self.assertEqual(selected.file_sha256, "0" * 64)
        self.assertEqual(EXPECTED_REGULAR_GAMES_PER_SEASON, 1312)

    def test_excludes_preseason(self) -> None:
        selected = select_five_season_history(_snapshot(_game(2021, 1)))
        self.assertEqual(selected.regular_rows, ())
        self.assertEqual(selected.playoff_rows, ())
        self.assertEqual(selected.coverage[0].regular_games, 0)

    def test_rejects_incomplete_pair_and_missing_situation(self) -> None:
        rows = _game(2021, 2)
        with self.assertRaises(NHLFiveSeasonAuditError):
            select_five_season_history(_snapshot(rows[:-1]))
        with self.assertRaises(NHLFiveSeasonAuditError):
            select_five_season_history(_snapshot(rows[:2]))

    def test_rejects_incoherent_opponent_or_season(self) -> None:
        rows = list(_game(2021, 2))
        rows[0] = replace(rows[0], opponent="CHI")
        with self.assertRaises(NHLFiveSeasonAuditError):
            select_five_season_history(_snapshot(tuple(rows)))
        rows = list(_game(2021, 2))
        rows[0] = replace(rows[0], season=2022)
        with self.assertRaises(NHLFiveSeasonAuditError):
            select_five_season_history(_snapshot(tuple(rows)))

    def test_rejects_duplicate_team_row(self) -> None:
        rows = _game(2021, 2)
        with self.assertRaises(NHLFiveSeasonAuditError):
            select_five_season_history(_snapshot(rows + rows[:1]))

    def test_selection_has_no_network_database_or_model_client(self) -> None:
        path = Path(__file__).resolve().parents[2] / "src/nhl/moneypuck_five_season.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(name=node.module or "")])
        }
        self.assertFalse(imports & {"requests", "urllib", "http", "sqlite3", "sklearn", "joblib"})


if __name__ == "__main__":
    unittest.main()
