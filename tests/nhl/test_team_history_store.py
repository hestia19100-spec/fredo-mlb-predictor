"""NHL-24: import local append-only et lecture de l'historique au cutoff."""
from __future__ import annotations

from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
import json
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.nhl.moneypuck_team_capture import capture_team_history
from src.nhl.moneypuck_team_import import TEAM_GAME_DOWNLOAD
from src.nhl.public_schedule_candidates import ScheduledGame
from src.nhl.public_schedule_capture import BASE_URL, capture_public_schedule
from src.nhl.team_history_store import (
    NHLHistoryStoreError, POLICY_PATH, import_verified_team_capture,
    load_pregame_team_history, load_verified_schedule_team_history, verify_history_database,
)

HEADER = (
    "team,season,name,gameId,playerTeam,opposingTeam,home_or_away,"
    "gameDate,position,situation,xGoalsFor,xGoalsAgainst,goalsFor,"
    "goalsAgainst,shotsOnGoalFor,shotsOnGoalAgainst"
)


def _csv(game_date: str = "20211013") -> bytes:
    rows = [HEADER]
    for situation in ("all", "5on5"):
        rows.extend((
            f"NYR,2021,NYR,2021020001,NYR,BOS,AWAY,{game_date},Team Level,{situation},2.1,1.4,4,2,30,24",
            f"BOS,2021,BOS,2021020001,BOS,NYR,HOME,{game_date},Team Level,{situation},1.4,2.1,2,4,24,30",
        ))
    return ("\n".join(rows) + "\n").encode("utf-8")


class Response(BytesIO):
    status = 200
    headers = {"Content-Type": "text/csv"}

    def geturl(self) -> str:
        return TEAM_GAME_DOWNLOAD


class HistoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "nhl"
        self.capture_root = self.root / "captures"
        self.database = self.root / "team_history.db"
        self.mlb_database = Path(directory.name) / "fredo_mlb.db"
        self.synthetic_database = self.root / "fredo_nhl.db"
        self.observed = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)
        self.imported = self.observed + timedelta(minutes=2)

    def _capture(self, *, game_date: str = "20211013", observed: datetime | None = None):
        started = observed or self.observed
        times = iter((started, started + timedelta(seconds=1)))
        return capture_team_history(
            explicit_manual_run=True,
            transport=lambda url, timeout: Response(_csv(game_date)),
            code_commit="a" * 40, root=self.capture_root,
            now=lambda: next(times),
        )

    def _import(self, slot: Path, *, when: datetime | None = None):
        with patch("src.nhl.team_history_store._clock_utc", return_value=when or self.imported):
            return import_verified_team_capture(
                slot, database_path=self.database, allowed_root=self.root,
                capture_root=self.capture_root,
            )

    def _game(self, start: datetime) -> ScheduledGame:
        return ScheduledGame(2026020001, 2026, start, "NYR", "BOS")

    def _history(self, game: ScheduledGame, schedule_observed: datetime):
        return load_pregame_team_history(
            game, schedule_observed_at_utc=schedule_observed,
            lead_minutes=120, database_path=self.database, allowed_root=self.root,
        )

    def test_explicit_import_is_idempotent_and_separate_from_mlb(self) -> None:
        capture = self._capture()
        self.assertFalse(self.database.exists())
        first = self._import(capture.path)
        second = self._import(capture.path, when=self.imported + timedelta(days=1))
        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual(first.row_count, 4)
        self.assertEqual(first.effective_available_at_utc, second.effective_available_at_utc)
        verify_history_database(self.database, allowed_root=self.root)
        self.assertFalse(self.mlb_database.exists())
        self.assertFalse(self.synthetic_database.exists())

    def test_future_game_reads_prior_regular_rows_only(self) -> None:
        self._import(self._capture().path)
        game = self._game(datetime(2026, 10, 6, 21, tzinfo=timezone.utc))
        history = self._history(game, self.observed)
        self.assertEqual(history.target_game_id, game.game_id)
        self.assertEqual(len(history.away_rows), 2)
        self.assertEqual(len(history.home_rows), 2)
        self.assertEqual({row.situation for row in history.away_rows}, {"all", "5on5"})
        self.assertFalse(history.regular_coverage_complete)
        self.assertFalse(history.historical_backtest_asof_proven)
        self.assertFalse(history.training_permitted)
        self.assertFalse(history.prediction_publication_permitted)

    def test_import_after_cutoff_cannot_be_used_retroactively(self) -> None:
        self._import(self._capture().path)
        game = self._game(datetime(2026, 10, 4, 13, tzinfo=timezone.utc))
        with self.assertRaisesRegex(NHLHistoryStoreError, "avant le cutoff"):
            self._history(game, self.observed - timedelta(hours=2))

    def test_schedule_after_cutoff_is_rejected(self) -> None:
        self._import(self._capture().path)
        game = self._game(datetime(2026, 10, 6, 21, tzinfo=timezone.utc))
        with self.assertRaisesRegex(NHLHistoryStoreError, "Calendrier reçu après"):
            self._history(game, game.start_utc)

    def test_same_day_source_row_is_rejected_before_database_creation(self) -> None:
        capture = self._capture(game_date="20261004")
        with self.assertRaisesRegex(NHLHistoryStoreError, "ligne source"):
            self._import(capture.path)
        self.assertFalse(self.database.exists())

    def test_tampered_receipt_cannot_change_an_existing_import(self) -> None:
        capture = self._capture()
        self._import(capture.path)
        receipt = capture.path / "receipt.json"
        receipt.write_text(receipt.read_text() + " ")
        with self.assertRaisesRegex(NHLHistoryStoreError, "divergente"):
            self._import(capture.path)
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM nhl_history_captures").fetchone()[0], 1)

    def test_append_only_triggers_and_read_only_audit(self) -> None:
        capture = self._capture()
        self._import(capture.path)
        with closing(sqlite3.connect(self.database)) as connection:
            for table in ("nhl_history_captures", "nhl_history_team_rows"):
                with self.subTest(table=table):
                    with self.assertRaises(sqlite3.IntegrityError):
                        connection.execute(f"DELETE FROM {table}")
                    with self.assertRaises(sqlite3.IntegrityError):
                        connection.execute(f"UPDATE {table} SET capture_id = capture_id")
        verify_history_database(self.database, allowed_root=self.root)

    def test_later_capture_does_not_rewrite_earlier_view(self) -> None:
        first = self._capture()
        self._import(first.path)
        later_observed = self.observed + timedelta(days=1)
        later = self._capture(observed=later_observed)
        self._import(later.path, when=later_observed + timedelta(minutes=2))
        early_game = self._game(datetime(2026, 10, 5, 10, tzinfo=timezone.utc))
        late_game = self._game(datetime(2026, 10, 6, 20, tzinfo=timezone.utc))
        self.assertEqual(self._history(early_game, self.observed).capture_id, first.path.name)
        self.assertEqual(self._history(late_game, self.observed).capture_id, later.path.name)

    def _schedule_capture(self, observed: datetime) -> Path:
        target = date(2026, 10, 6)
        body = json.dumps({"gameWeek": [{"date": target.isoformat(), "games": [{
            "id": 2026020001, "season": 20262027, "gameType": 2,
            "gameState": "FUT", "gameScheduleState": "OK",
            "startTimeUTC": "2026-10-06T21:00:00Z",
            "awayTeam": {"id": 3, "abbrev": "NYR"},
            "homeTeam": {"id": 6, "abbrev": "BOS"},
        }]}]}).encode()

        class ScheduleResponse(BytesIO):
            status = 200
            headers = {"Content-Type": "application/json"}

            def geturl(self) -> str:
                return BASE_URL + target.isoformat()

        times = iter((observed, observed + timedelta(seconds=1)))
        return capture_public_schedule(
            target, explicit_manual_run=True, root=self.root / "schedules",
            transport=lambda url, timeout: ScheduleResponse(body),
            now=lambda: next(times),
        ).path

    def test_bridge_requires_a_verified_schedule_capture(self) -> None:
        self._import(self._capture().path)
        schedule = self._schedule_capture(self.imported + timedelta(minutes=1))
        history = load_verified_schedule_team_history(
            schedule, target_game_id=2026020001, lead_minutes=120,
            database_path=self.database, allowed_root=self.root,
        )
        self.assertEqual(len(history.away_rows), 2)
        self.assertFalse(history.training_permitted)
        with self.assertRaises(NHLHistoryStoreError):
            load_verified_schedule_team_history(
                schedule, target_game_id=999999, lead_minutes=120,
                database_path=self.database, allowed_root=self.root,
            )
        (schedule / "response.json").write_bytes(b"{}")
        with self.assertRaises(ValueError):
            load_verified_schedule_team_history(
                schedule, target_game_id=2026020001, lead_minutes=120,
                database_path=self.database, allowed_root=self.root,
            )

    def test_bridge_refuses_schedule_after_cutoff(self) -> None:
        self._import(self._capture().path)
        schedule = self._schedule_capture(datetime(2026, 10, 6, 20, tzinfo=timezone.utc))
        with self.assertRaisesRegex(NHLHistoryStoreError, "Calendrier reçu après"):
            load_verified_schedule_team_history(
                schedule, target_game_id=2026020001, lead_minutes=120,
                database_path=self.database, allowed_root=self.root,
            )

    def test_disabled_protocol_cannot_be_relaxed(self) -> None:
        capture = self._capture()
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        policy["training_permitted"] = True
        changed = self.root / "unsafe_protocol.json"
        changed.write_text(json.dumps(policy), encoding="utf-8")
        with patch("src.nhl.team_history_store.POLICY_PATH", changed):
            with self.assertRaisesRegex(NHLHistoryStoreError, "Protocole"):
                self._import(capture.path)
        self.assertFalse(self.database.exists())
        self._import(capture.path)
        with patch("src.nhl.team_history_store.POLICY_PATH", changed):
            with self.assertRaisesRegex(NHLHistoryStoreError, "Protocole"):
                self._history(self._game(datetime(2026, 10, 6, 21, tzinfo=timezone.utc)), self.observed)

    def test_foreign_preexisting_database_is_not_adopted(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("CREATE TABLE unrelated(value TEXT)")
        capture = self._capture()
        with self.assertRaisesRegex(NHLHistoryStoreError, "Schéma"):
            self._import(capture.path)
        with closing(sqlite3.connect(self.database)) as connection:
            names = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )}
        self.assertEqual(names, {"unrelated"})

    def test_wrong_database_path_is_rejected(self) -> None:
        capture = self._capture()
        with self.assertRaises(NHLHistoryStoreError):
            with patch("src.nhl.team_history_store._clock_utc", return_value=self.imported):
                import_verified_team_capture(capture.path, database_path=self.synthetic_database,
                    allowed_root=self.root, capture_root=self.capture_root)
        self.assertFalse(self.synthetic_database.exists())


if __name__ == "__main__":
    unittest.main()
