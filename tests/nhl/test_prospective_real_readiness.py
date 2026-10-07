"""NHL-26: verified prospective evidence remains descriptive and read-only."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.nhl.current_season_import import CurrentSeasonImport
from src.nhl.moneypuck_team_import import TeamGameRow
from src.nhl.prospective_real_readiness import (
    NHLProspectiveReadinessError, POLICY_PATH, audit_real_pregame_readiness,
)
from src.nhl.team_history_store import NHLHistoryStoreError, PregameTeamHistory

UTC = timezone.utc
START = "2026-10-06T21:00:00Z"
OBSERVED = "2026-10-05T10:00:00Z"


def schedule(*, observed: str = OBSERVED, games: bool = True) -> dict[str, object]:
    return {
        "target_date": "2026-10-06", "observed_at_utc": observed,
        "response_sha256": "b" * 64,
        "games": [{
            "game_id": 2026020001, "season": 20262027,
            "scheduled_start_utc": START,
            "away_abbr": "NYR", "home_abbr": "BOS",
        }] if games else [],
    }


def _row(team: str, season: int, day: date, situation: str) -> TeamGameRow:
    return TeamGameRow(
        game_id=season * 1_000_000 + 20001 + day.day,
        season=season, game_date=day, team=team,
        opponent="BOS" if team == "NYR" else "NYR",
        home_or_away="AWAY" if team == "NYR" else "HOME",
        situation=situation, x_goals_for=Decimal("2.5"),
        x_goals_against=Decimal("1.4"), goals_for=3,
        goals_against=2, shots_on_goal_for=Decimal("30"),
        shots_on_goal_against=Decimal("28"),
    )


def history(*, season: int = 2025) -> PregameTeamHistory:
    day = date(season, 10, 1)
    def rows(team: str) -> tuple[TeamGameRow, ...]:
        return (_row(team, season, day, "all"),
                _row(team, season, day, "5on5"))
    return PregameTeamHistory(
        target_game_id=2026020001,
        information_cutoff_utc=datetime(2026, 10, 6, 19, tzinfo=UTC),
        capture_id="capture-1", response_sha256="a" * 64,
        effective_available_at_utc=datetime(2026, 10, 5, 10, tzinfo=UTC),
        away_rows=rows("NYR"), home_rows=rows("BOS"),
        regular_coverage_complete=False,
    )


class ProspectiveReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database = self.root / "team_history.db"
        self.slot = self.root / "schedule"

    def audit(self, *, min_games: int = 1):
        return audit_real_pregame_readiness(
            self.slot, database_path=self.database, allowed_root=self.root,
            min_games_per_team=min_games,
        )

    def test_absent_database_is_visible_without_creating_one(self) -> None:
        with patch("src.nhl.prospective_real_readiness.audit_schedule_capture", return_value=schedule()):
            result = self.audit()
        self.assertEqual(result["games"][0]["status"], "NO_REAL_HISTORY_DATABASE")
        self.assertFalse(self.database.exists())
        self.assertFalse(result["training_permitted"])
        self.assertFalse(result["prediction_publication_permitted"])

    def test_schedule_after_cutoff_is_never_used(self) -> None:
        late = schedule(observed="2026-10-06T20:00:00Z")
        with patch("src.nhl.prospective_real_readiness.audit_schedule_capture", return_value=late):
            result = self.audit()
        self.assertEqual(result["games"][0]["status"], "SCHEDULE_AFTER_CUTOFF")
        self.assertFalse(result["games"][0]["schedule_before_cutoff"])

    def test_no_eligible_historical_capture_is_reported(self) -> None:
        self.database.touch()
        with (patch("src.nhl.prospective_real_readiness.audit_schedule_capture", return_value=schedule()),
              patch("src.nhl.prospective_real_readiness.verify_history_database"),
              patch("src.nhl.prospective_real_readiness.load_pregame_team_history",
                    side_effect=NHLHistoryStoreError("Aucune capture historique avant le cutoff."))):
            result = self.audit()
        self.assertEqual(result["games"][0]["status"], "NO_PRE_CUTOFF_HISTORY_CAPTURE")

    def test_reference_only_history_is_not_mistaken_for_current_season(self) -> None:
        self.database.touch()
        with (patch("src.nhl.prospective_real_readiness.audit_schedule_capture", return_value=schedule()),
              patch("src.nhl.prospective_real_readiness.verify_history_database"),
              patch("src.nhl.prospective_real_readiness.load_pregame_team_history", return_value=history())):
            result = self.audit()
        game = result["games"][0]
        self.assertEqual(game["status"], "CURRENT_SEASON_HISTORY_NOT_IMPORTED")
        self.assertEqual((game["away_current_season_games"], game["home_current_season_games"]), (0, 0))
        self.assertTrue(game["minimum_sample_reached"])
        self.assertEqual(result["descriptive_form_count"], 0)
        self.assertFalse(result["training_permitted"])

    def test_current_season_form_is_still_not_a_prediction(self) -> None:
        self.database.touch()
        with (patch("src.nhl.prospective_real_readiness.audit_schedule_capture", return_value=schedule()),
              patch("src.nhl.prospective_real_readiness.verify_history_database"),
              patch("src.nhl.prospective_real_readiness.load_pregame_team_history",
                    return_value=history(season=2026))):
            result = self.audit()
        game = result["games"][0]
        self.assertEqual(game["status"], "DESCRIPTIVE_FORM_ONLY")
        self.assertEqual((game["away_current_season_games"], game["home_current_season_games"]), (1, 1))
        self.assertEqual(result["descriptive_form_count"], 1)
        self.assertNotIn("probability", json.dumps(result))
        self.assertNotIn("odds", json.dumps(result))
        self.assertFalse(result["prediction_publication_permitted"])

    def test_verified_current_season_sidecar_is_joined_descriptively(self) -> None:
        self.database.touch()
        current = history(season=2026)
        imported = CurrentSeasonImport(
            self.root / "current", 2026, "capture-current", "c" * 64,
            datetime(2026, 10, 4, tzinfo=UTC),
            datetime(2026, 10, 5, 12, tzinfo=UTC),
            datetime(2026, 10, 5, 12, tzinfo=UTC),
            current.away_rows + current.home_rows, 1,
        )
        with (patch("src.nhl.prospective_real_readiness.audit_schedule_capture", return_value=schedule()),
              patch("src.nhl.prospective_real_readiness.verify_history_database"),
              patch("src.nhl.prospective_real_readiness.load_pregame_team_history", return_value=history()),
              patch("src.nhl.prospective_real_readiness.verify_current_season_import", return_value=imported)):
            result = audit_real_pregame_readiness(
                self.slot, database_path=self.database, allowed_root=self.root,
                current_season_import_slot=self.root / "current", min_games_per_team=1,
            )
            self.assertEqual(result["games"][0]["status"],
                             "CURRENT_SEASON_FORM_DESCRIPTIVE_ONLY")
            self.assertEqual(result["games"][0]["away_current_season_imported_games"], 1)
            self.assertEqual(result["games"][0]["home_current_season_imported_games"], 1)
            self.assertEqual(result["games"][0]["away_current_season_games"], 1)
            self.assertEqual(result["games"][0]["home_current_season_games"], 1)
            self.assertTrue(result["games"][0]["current_season_form_joined"])
            self.assertEqual(result["descriptive_form_count"], 1)
            self.assertEqual(result["games"][0]["history_effective_available_at_utc"],
                             datetime(2026, 10, 5, 10, tzinfo=UTC).isoformat())
            self.assertEqual(result["games"][0]["form_effective_available_at_utc"],
                             datetime(2026, 10, 5, 12, tzinfo=UTC).isoformat())
            self.assertTrue(result["current_season_import_verified"])
            self.assertFalse(result["training_permitted"])
            late = replace(imported, effective_available_at_utc=datetime(2026, 10, 6, 20, tzinfo=UTC))
            with patch("src.nhl.prospective_real_readiness.verify_current_season_import", return_value=late):
                blocked = audit_real_pregame_readiness(
                    self.slot, database_path=self.database, allowed_root=self.root,
                    current_season_import_slot=self.root / "current", min_games_per_team=1,
                )
            self.assertEqual(blocked["games"][0]["status"],
                             "CURRENT_SEASON_IMPORT_AFTER_CUTOFF")

    def test_insufficient_sample_precedes_season_status(self) -> None:
        self.database.touch()
        with (patch("src.nhl.prospective_real_readiness.audit_schedule_capture", return_value=schedule()),
              patch("src.nhl.prospective_real_readiness.verify_history_database"),
              patch("src.nhl.prospective_real_readiness.load_pregame_team_history", return_value=history())):
            result = self.audit(min_games=2)
        self.assertEqual(result["games"][0]["status"], "INSUFFICIENT_HISTORICAL_SAMPLE")

    def test_empty_schedule_is_not_invented(self) -> None:
        with patch("src.nhl.prospective_real_readiness.audit_schedule_capture", return_value=schedule(games=False)):
            result = self.audit()
        self.assertEqual(result["game_count"], 0)
        self.assertEqual(result["games"], [])

    def test_tampered_schedule_or_database_fails_closed(self) -> None:
        with patch("src.nhl.prospective_real_readiness.audit_schedule_capture", side_effect=ValueError("tampered")):
            with self.assertRaisesRegex(ValueError, "tampered"):
                self.audit()
        self.database.touch()
        with (patch("src.nhl.prospective_real_readiness.audit_schedule_capture", return_value=schedule()),
              patch("src.nhl.prospective_real_readiness.verify_history_database",
                    side_effect=NHLHistoryStoreError("invalid schema"))):
            with self.assertRaisesRegex(NHLHistoryStoreError, "invalid schema"):
                self.audit()

    def test_sealed_schedule_is_audited_without_network_or_database_write(self) -> None:
        from io import BytesIO
        from src.nhl.public_schedule_capture import BASE_URL, capture_public_schedule

        payload = json.dumps({"gameWeek": [{"date": "2026-10-06", "games": [{
            "id": 2026020001, "season": 20262027, "gameType": 2,
            "gameState": "FUT", "gameScheduleState": "OK",
            "startTimeUTC": START,
            "awayTeam": {"id": 3, "abbrev": "NYR"},
            "homeTeam": {"id": 6, "abbrev": "BOS"},
        }]}]}).encode()

        class Response(BytesIO):
            status = 200
            headers = {"Content-Type": "application/json"}

            def geturl(self) -> str:
                return BASE_URL + "2026-10-06"

        ticks = iter((datetime(2026, 10, 5, 10, tzinfo=UTC),
                      datetime(2026, 10, 5, 10, 0, 1, tzinfo=UTC)))
        receipt = capture_public_schedule(
            date(2026, 10, 6), explicit_manual_run=True,
            root=self.root / "captures",
            transport=lambda url, timeout: Response(payload),
            now=lambda: next(ticks),
        )
        result = audit_real_pregame_readiness(
            receipt.path, database_path=self.database, allowed_root=self.root,
        )
        self.assertEqual(result["game_count"], 1)
        self.assertEqual(result["games"][0]["status"], "NO_REAL_HISTORY_DATABASE")
        self.assertFalse(self.database.exists())
        (receipt.path / "response.json").write_bytes(b"{}")
        with self.assertRaises(ValueError):
            audit_real_pregame_readiness(receipt.path, database_path=self.database,
                                         allowed_root=self.root)

    def test_protocol_change_cannot_enable_training(self) -> None:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        policy["training_permitted"] = True
        unsafe = self.root / "unsafe.json"
        unsafe.write_text(json.dumps(policy), encoding="utf-8")
        with patch("src.nhl.prospective_real_readiness.POLICY_PATH", unsafe):
            with self.assertRaisesRegex(NHLProspectiveReadinessError, "Protocole"):
                self.audit()


if __name__ == "__main__":
    unittest.main()
