"""NHL-11: imported public downloads stay offline and non-predictive."""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.nhl.moneypuck_team_import import (
    ATTRIBUTION, NHLMoneyPuckImportError, import_team_games,
)

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "nhl" / "moneypuck_team_import.py"
HEADER = (
    "team,season,name,gameId,playerTeam,opposingTeam,home_or_away,"
    "gameDate,position,situation,xGoalsFor,xGoalsAgainst,goalsFor,"
    "goalsAgainst,shotsOnGoalFor,shotsOnGoalAgainst"
)
NYR = "NYR,2008,NYR,2008020001,NYR,T.B,AWAY,20081004,Team Level,all,2.1,1.4,4.0,2.0,30,24"
BOS = "BOS,2008,BOS,2008020001,BOS,NYR,HOME,20081004,Team Level,all,1.4,2.1,2.0,4.0,24,30"


class MoneyPuckTeamImportTests(unittest.TestCase):
    def _load(self, *lines: str, season: int | None = 2008):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "teams.csv"
            source.write_text("\n".join((HEADER, *lines, "")), encoding="utf-8")
            before = source.read_bytes()
            snapshot = import_team_games(source, season=season)
            self.assertEqual(source.read_bytes(), before)
            return snapshot

    def _reject(self, *lines: str, season: int | None = 2008):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "teams.csv"
            source.write_text("\n".join((HEADER, *lines, "")), encoding="utf-8")
            with self.assertRaises(NHLMoneyPuckImportError):
                import_team_games(source, season=season)

    def test_reads_team_game_rows_with_explicit_attribution(self) -> None:
        before = datetime.now(timezone.utc)
        snapshot = self._load(BOS, NYR)
        after = datetime.now(timezone.utc)
        self.assertTrue(before <= snapshot.observed_at_utc <= after)
        self.assertEqual(len(snapshot.rows), 2)
        self.assertEqual(snapshot.rows[0].team, "BOS")
        self.assertEqual(snapshot.rows[1].goals_for, 4)
        self.assertEqual(snapshot.attribution, ATTRIBUTION)
        self.assertFalse(snapshot.training_permitted)
        self.assertEqual(len(snapshot.file_sha256), 64)
        self.assertFalse(snapshot.available_by(before - timedelta(days=1)))
        self.assertTrue(snapshot.available_by(after))

    def test_skips_other_situations_and_other_seasons(self) -> None:
        other = NYR.replace(",all,", ",other,")
        later = NYR.replace(",2008,", ",2009,").replace(",2008020001,", ",2009020001,")
        snapshot = self._load(other, later, BOS)
        self.assertEqual([row.team for row in snapshot.rows], ["BOS"])

    def test_rejects_duplicates_and_invalid_statistics(self) -> None:
        self._reject(NYR, NYR)
        self._reject(NYR.replace(",2.1,", ",NaN,"))
        self._reject(NYR.replace(",4.0,", ",4.5,"))
        self._reject(NYR.replace("NYR,T.B,AWAY", "NYR,NYR,AWAY"))
        self._reject(NYR.replace("Team Level", "Skater"))

    def test_rejects_missing_columns_or_empty_season(self) -> None:
        self._reject(NYR, season=2026)
        with TemporaryDirectory() as directory:
            source = Path(directory) / "teams.csv"
            source.write_text("gameId,team\n2008020001,NYR\n", encoding="utf-8")
            with self.assertRaises(NHLMoneyPuckImportError):
                import_team_games(source)

    def test_no_network_model_or_database_client(self) -> None:
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertFalse(imported & {
            "requests", "httpx", "urllib", "socket", "subprocess",
            "sklearn", "xgboost", "lightgbm", "sqlite3",
        })


if __name__ == "__main__":
    unittest.main()
