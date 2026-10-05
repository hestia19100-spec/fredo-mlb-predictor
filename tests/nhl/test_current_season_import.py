"""NHL-28: current-season imports are append-only, as-of and model-disabled."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.nhl.current_season_import import (
    NHLCurrentSeasonImportError, assess_current_season_before_game,
    import_current_season_capture, verify_current_season_import,
)
from src.nhl.moneypuck_team_capture import capture_team_history
from src.nhl.moneypuck_team_import import TEAM_GAME_DOWNLOAD
from src.nhl.public_schedule_candidates import ScheduledGame

UTC = timezone.utc
HEADER = (
    "team,season,name,gameId,playerTeam,opposingTeam,home_or_away,"
    "gameDate,position,situation,xGoalsFor,xGoalsAgainst,goalsFor,"
    "goalsAgainst,shotsOnGoalFor,shotsOnGoalAgainst"
)


def csv_body(*, current_date: str = "20260929", broken_pair: bool = False) -> bytes:
    lines = [HEADER]
    for season, game_id, day in ((2021, 2021020001, "20211013"),
                                 (2026, 2026020001, current_date)):
        for situation in ("all", "5on5"):
            lines.append(f"NYR,{season},NYR,{game_id},NYR,BOS,AWAY,{day},Team Level,{situation},2.1,1.4,4,2,30,24")
            if not (broken_pair and season == 2026 and situation == "5on5"):
                lines.append(f"BOS,{season},BOS,{game_id},BOS,NYR,HOME,{day},Team Level,{situation},1.4,2.1,2,4,24,30")
    return ("\n".join(lines) + "\n").encode()


class Response(BytesIO):
    status = 200
    headers = {"Content-Type": "text/csv"}

    def geturl(self) -> str:
        return TEAM_GAME_DOWNLOAD


class CurrentSeasonImportTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "nhl"
        self.capture_root = self.root / "captures"
        self.import_root = self.root / "current"
        self.observed = datetime(2026, 10, 4, 12, tzinfo=UTC)
        self.imported = datetime(2026, 10, 5, 10, tzinfo=UTC)

    def capture(self, raw: bytes | None = None) -> Path:
        ticks = iter((self.observed, self.observed + timedelta(seconds=1)))
        result = capture_team_history(
            explicit_manual_run=True,
            transport=lambda url, timeout: Response(raw or csv_body()),
            code_commit="a" * 40, root=self.capture_root,
            now=lambda: next(ticks),
        )
        return result.path

    def run_import(self, slot: Path, **changes):
        options = {"season": 2026, "explicit_manual_run": True,
                   "root": self.import_root, "capture_root": self.capture_root,
                   "now": lambda: self.imported}
        options.update(changes)
        return import_current_season_capture(slot, **options)

    def verify(self, slot: Path):
        return verify_current_season_import(
            slot, root=self.import_root, capture_root=self.capture_root,
        )

    def test_real_source_rows_are_imported_without_database_or_model(self) -> None:
        source = self.capture()
        first = self.run_import(source)
        second = self.run_import(source, now=lambda: self.imported + timedelta(days=1))
        self.assertEqual(first.path, second.path)
        checked = self.verify(first.path)
        self.assertEqual(checked.regular_game_count, 1)
        self.assertEqual(len(checked.regular_rows), 4)
        self.assertEqual(checked.effective_available_at_utc, self.imported)
        self.assertFalse(checked.training_permitted)
        self.assertFalse(checked.prediction_publication_permitted)
        self.assertFalse((self.root / "team_history.db").exists())
        self.assertFalse((self.root / "fredo_mlb.db").exists())
        receipt = json.loads((first.path / "receipt.json").read_text())
        self.assertEqual(receipt["source_response_sha256"], first.source_response_sha256)
        self.assertFalse(receipt["historical_backtest_asof_proven"])

    def test_counts_only_prior_regular_games_when_import_precedes_cutoff(self) -> None:
        imported = self.run_import(self.capture())
        future = ScheduledGame(2026020002, 20262027,
                               datetime(2026, 10, 6, 23, tzinfo=UTC), "NYR", "BOS")
        report = assess_current_season_before_game(imported, future, lead_minutes=120)
        self.assertTrue(report["current_season_import_before_cutoff"])
        self.assertEqual(report["prior_regular_games_by_team"], {"NYR": 1, "BOS": 1})
        early = ScheduledGame(2026020002, 20262027,
                              datetime(2026, 10, 5, 11, tzinfo=UTC), "NYR", "BOS")
        late = assess_current_season_before_game(imported, early, lead_minutes=120)
        self.assertFalse(late["current_season_import_before_cutoff"])
        self.assertEqual(late["prior_regular_games_by_team"], {"NYR": 0, "BOS": 0})

    def test_manual_mode_pairing_and_capture_date_are_enforced(self) -> None:
        source = self.capture()
        with self.assertRaises(NHLCurrentSeasonImportError):
            self.run_import(source, explicit_manual_run=False)
        self.assertFalse(self.import_root.exists())
        with self.assertRaisesRegex(NHLCurrentSeasonImportError, "Paire"):
            self.run_import(self.capture(csv_body(broken_pair=True)))
        with self.assertRaisesRegex(NHLCurrentSeasonImportError, "non antérieur"):
            self.run_import(self.capture(csv_body(current_date="20261004")))
        self.assertFalse(self.import_root.exists())

    def test_tampering_and_outside_root_are_rejected(self) -> None:
        imported = self.run_import(self.capture())
        (imported.path / "rows.json").write_bytes(b"[]")
        with self.assertRaises(NHLCurrentSeasonImportError):
            self.verify(imported.path)
        with self.assertRaises(NHLCurrentSeasonImportError):
            self.verify(self.root / "elsewhere")


if __name__ == "__main__":
    unittest.main()
