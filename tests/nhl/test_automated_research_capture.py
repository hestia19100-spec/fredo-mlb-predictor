"""Offline checks for explicit automated NHL research captures."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nhl_operations.automatic_research_capture import (
    NHLAutomaticResearchError, prepare_automatic_research,
)
from src.nhl.current_season_import import (
    NHLCurrentSeasonImportError, import_current_season_capture,
    verify_current_season_import,
)
from src.nhl.moneypuck_team_capture import (
    NHLTeamCaptureError, capture_team_history, verify_team_capture,
)
from src.nhl.moneypuck_team_import import TEAM_GAME_DOWNLOAD
from src.nhl.public_schedule_capture import (
    BASE_URL, PublicScheduleCaptureError, capture_public_schedule,
    verify_public_schedule_capture,
)
from src.nhl.schedule_capture_readiness import audit_schedule_capture


TARGET = date(2026, 10, 15)
OBSERVED = datetime(2026, 10, 15, 10, tzinfo=timezone.utc)
HEADER = (
    "team,season,name,gameId,playerTeam,opposingTeam,home_or_away,"
    "gameDate,position,situation,xGoalsFor,xGoalsAgainst,goalsFor,"
    "goalsAgainst,shotsOnGoalFor,shotsOnGoalAgainst"
)


class ScheduleResponse(BytesIO):
    status = 200
    headers = {"Content-Type": "application/json"}

    def geturl(self) -> str:
        return BASE_URL + TARGET.isoformat()


class TeamResponse(BytesIO):
    status = 200
    headers = {"Content-Type": "text/csv"}

    def geturl(self) -> str:
        return TEAM_GAME_DOWNLOAD


def schedule_bytes() -> bytes:
    game = {
        "id": 2026020108, "season": 20262027, "gameType": 2,
        "gameState": "FUT", "gameScheduleState": "OK",
        "startTimeUTC": "2026-10-15T23:00:00Z",
        "awayTeam": {"id": 7, "abbrev": "BUF"},
        "homeTeam": {"id": 10, "abbrev": "TOR"},
    }
    return json.dumps({"gameWeek": [
        {"date": TARGET.isoformat(), "games": [game]}
    ]}).encode()


def team_bytes() -> bytes:
    rows = [HEADER]
    for season, game_id, day in ((2021, 2021020001, "20211013"),
                                 (2026, 2026020001, "20261014")):
        for situation in ("all", "5on5"):
            rows.extend((
                f"BUF,{season},BUF,{game_id},BUF,TOR,AWAY,{day},Team Level,{situation},2.1,1.4,4,2,30,24",
                f"TOR,{season},TOR,{game_id},TOR,BUF,HOME,{day},Team Level,{situation},1.4,2.1,2,4,24,30",
            ))
    return ("\n".join(rows) + "\n").encode()


class AutomatedResearchCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.schedule_root = self.root / "schedule"
        self.team_root = self.root / "team"
        self.import_root = self.root / "import"

    def test_schedule_is_proven_automatic_but_not_publishable(self) -> None:
        ticks = iter((OBSERVED, OBSERVED + timedelta(seconds=1)))
        slot = capture_public_schedule(
            TARGET, explicit_automated_run=True, root=self.schedule_root,
            transport=lambda url, timeout: ScheduleResponse(schedule_bytes()),
            now=lambda: next(ticks),
        ).path
        receipt = verify_public_schedule_capture(slot)
        audit = audit_schedule_capture(slot)
        self.assertEqual(receipt["acquisition_mode"], "automated_https_get")
        self.assertEqual(audit["acquisition_mode"], "automated_https_get")
        self.assertEqual(audit["game_count"], 1)
        self.assertFalse(receipt["prediction_publication_permitted"])
        self.assertFalse(receipt["training_permitted"])
        receipt["policy_sha256"] = "0" * 64
        (slot / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaises(PublicScheduleCaptureError):
            verify_public_schedule_capture(slot)

    def test_automatic_team_capture_and_import_remain_research_only(self) -> None:
        ticks = iter((OBSERVED, OBSERVED + timedelta(seconds=1)))
        capture = capture_team_history(
            explicit_automated_run=True,
            transport=lambda url, timeout: TeamResponse(team_bytes()),
            code_commit="a" * 40, root=self.team_root,
            now=lambda: next(ticks),
        )
        verified = verify_team_capture(capture.path, allowed_root=self.team_root)
        self.assertEqual(verified.acquisition_mode, "automated_https_get")
        imported = import_current_season_capture(
            capture.path, explicit_automated_run=True,
            root=self.import_root, capture_root=self.team_root,
            now=lambda: OBSERVED + timedelta(minutes=1),
        )
        receipt = json.loads((imported.path / "receipt.json").read_text())
        self.assertEqual(receipt["import_mode"], "automated_research")
        self.assertEqual(
            verify_current_season_import(
                imported.path, root=self.import_root,
                capture_root=self.team_root,
            ).regular_game_count, 1,
        )
        self.assertFalse(receipt["training_permitted"])
        self.assertFalse(receipt["prediction_publication_permitted"])

    def test_preparation_uses_only_official_schedule_and_team_csv(self) -> None:
        calls = []

        def transport(url: str, timeout: int):
            calls.append(url)
            if url == BASE_URL + TARGET.isoformat():
                return ScheduleResponse(schedule_bytes())
            if url == TEAM_GAME_DOWNLOAD:
                return TeamResponse(team_bytes())
            raise AssertionError("Unexpected network source")

        result = prepare_automatic_research(
            code_commit="a" * 40, explicit_automated_run=True,
            schedule_root=self.schedule_root,
            team_root=self.team_root, import_root=self.import_root,
            transport=transport, now=lambda: OBSERVED,
        )
        self.assertEqual(calls, [BASE_URL + TARGET.isoformat(), TEAM_GAME_DOWNLOAD])
        self.assertEqual(result.nhl_date, TARGET)
        self.assertEqual(result.future_games, 1)
        self.assertEqual(result.current_regular_games, 1)
        self.assertFalse(result.prediction_publication_permitted)
        self.assertEqual(audit_schedule_capture(result.schedule_slot)["game_count"], 1)
        self.assertEqual(
            verify_current_season_import(
                result.import_slot, root=self.import_root,
                capture_root=self.team_root,
            ).regular_game_count, 1,
        )

        with self.assertRaises(NHLAutomaticResearchError):
            prepare_automatic_research(
                code_commit="a" * 40, transport=transport,
                schedule_root=self.schedule_root,
                team_root=self.team_root, import_root=self.import_root,
                now=lambda: OBSERVED,
            )
        self.assertEqual(len(calls), 2)

    def test_automatic_mode_cannot_be_spoofed_from_manual_evidence(self) -> None:
        with self.assertRaises(PublicScheduleCaptureError):
            capture_public_schedule(
                TARGET, explicit_manual_run=True, explicit_automated_run=True,
                root=self.schedule_root,
                transport=lambda url, timeout: ScheduleResponse(schedule_bytes()),
            )
        with self.assertRaises(NHLTeamCaptureError):
            capture_team_history(
                explicit_manual_run=True, explicit_automated_run=True,
                transport=lambda url, timeout: TeamResponse(team_bytes()),
                code_commit="a" * 40, root=self.team_root,
            )
        source = self.root / "source.csv"
        source.write_bytes(team_bytes())
        with self.assertRaises(NHLTeamCaptureError):
            capture_team_history(
                explicit_automated_run=True, source_file=source,
                code_commit="a" * 40, root=self.team_root,
            )
        ticks = iter((OBSERVED, OBSERVED + timedelta(seconds=1)))
        manual = capture_team_history(
            explicit_manual_run=True, source_file=source,
            code_commit="a" * 40, root=self.team_root,
            now=lambda: next(ticks),
        )
        with self.assertRaises(NHLCurrentSeasonImportError):
            import_current_season_capture(
                manual.path, explicit_automated_run=True,
                root=self.import_root, capture_root=self.team_root,
                now=lambda: OBSERVED + timedelta(minutes=1),
            )


if __name__ == "__main__":
    unittest.main()
