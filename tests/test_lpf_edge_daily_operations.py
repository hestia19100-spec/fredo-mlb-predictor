"""Tests de la fondation locale du centre d'actions LPF Edge."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from src import lpf_edge_daily_operations as operations


TARGET = date(2026, 9, 13)
NOW = datetime(2026, 9, 13, 8, 0, tzinfo=timezone.utc)
FIRST_START = datetime(2026, 9, 13, 18, 0, tzinfo=timezone.utc)


class LPFEdgeDailyOperationsTests(unittest.TestCase):
    def _git(self, *, ready: bool = True) -> operations.GitWorkspaceState:
        return operations.GitWorkspaceState(
            available=True,
            branch="main" if ready else "travail",
            head_commit="a" * 40,
            origin_main_commit="a" * 40,
            clean=True,
            synchronized=True,
        )

    def _games(
        self,
        *,
        count: int = 15,
        first_start: datetime | None = FIRST_START,
    ) -> operations.LocalGameDayState:
        return operations.LocalGameDayState(count, 0, first_start)

    def _overview(self, **changes: object) -> operations.DailyOperationsOverview:
        values: dict[str, object] = {
            "target_date": TARGET,
            "paris_today": TARGET,
            "now_utc": NOW,
            "game_day": self._games(),
            "git": self._git(),
            "prediction_slot_state": operations.PredictionSlotState.ABSENT,
            "prediction_certified": False,
            "results_target_date": TARGET - timedelta(days=1),
            "has_past_certified": True,
            "latest_score_pending_count": None,
            "checkpoint_slot_exists": False,
            "integrity_errors": (),
        }
        values.update(changes)
        return operations.build_daily_operations_overview(**values)

    def test_ready_day_exposes_three_executable_actions(self) -> None:
        overview = self._overview()
        self.assertTrue(overview.data_action.can_execute)
        self.assertTrue(overview.prediction_action.can_execute)
        self.assertTrue(overview.results_action.can_execute)
        self.assertEqual(
            {
                overview.data_action.state,
                overview.prediction_action.state,
                overview.results_action.state,
            },
            {operations.DailyActionState.READY},
        )

    def test_missing_games_require_data_before_prediction(self) -> None:
        overview = self._overview(game_day=self._games(count=0, first_start=None))
        self.assertEqual(
            overview.prediction_action.state,
            operations.DailyActionState.NEED_DATA,
        )
        self.assertFalse(overview.prediction_action.can_execute)
        self.assertTrue(overview.data_action.can_execute)

    def test_two_hour_prediction_boundary_is_exact(self) -> None:
        boundary = NOW.replace(hour=10)
        ready = self._overview(
            game_day=self._games(first_start=boundary),
        )
        late = self._overview(
            game_day=self._games(
                first_start=boundary.replace(minute=0) - timedelta(seconds=1)
            ),
        )
        self.assertEqual(ready.prediction_action.state, operations.DailyActionState.READY)
        self.assertEqual(late.prediction_action.state, operations.DailyActionState.TOO_LATE)

    def test_certified_prediction_is_done(self) -> None:
        overview = self._overview(
            prediction_slot_state=operations.PredictionSlotState.COMPLETED,
            prediction_certified=True,
        )
        self.assertEqual(overview.prediction_action.state, operations.DailyActionState.DONE)

    def test_completed_uncertified_prediction_requires_publication(self) -> None:
        overview = self._overview(
            prediction_slot_state=operations.PredictionSlotState.COMPLETED,
        )
        self.assertEqual(
            overview.prediction_action.state,
            operations.DailyActionState.ACTION_REQUIRED,
        )

    def test_failed_or_partial_prediction_is_never_reopened(self) -> None:
        for state in (
            operations.PredictionSlotState.FAILED,
            operations.PredictionSlotState.PARTIAL,
        ):
            with self.subTest(state=state):
                overview = self._overview(prediction_slot_state=state)
                self.assertEqual(
                    overview.prediction_action.state,
                    operations.DailyActionState.BLOCKED,
                )
                self.assertFalse(overview.prediction_action.can_execute)

    def test_results_wait_until_six_utc(self) -> None:
        overview = self._overview(
            now_utc=datetime(2026, 9, 13, 5, 59, 59, tzinfo=timezone.utc)
        )
        self.assertEqual(overview.results_action.state, operations.DailyActionState.TOO_EARLY)

    def test_morning_routine_is_exactly_the_results_decision(self) -> None:
        overview = self._overview()
        self.assertEqual(
            overview.morning_action.state,
            overview.results_action.state,
        )
        self.assertEqual(
            overview.morning_action.can_execute,
            overview.results_action.can_execute,
        )
        self.assertIn("matin", overview.morning_action.label.lower())

    def test_afternoon_routine_waits_until_noon_in_paris(self) -> None:
        before_noon = self._overview(
            now_utc=datetime(2026, 9, 13, 9, 59, 59, tzinfo=timezone.utc),
        )
        at_noon = self._overview(
            now_utc=datetime(2026, 9, 13, 10, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            before_noon.afternoon_action.state,
            operations.DailyActionState.TOO_EARLY,
        )
        self.assertEqual(
            at_noon.afternoon_action.state,
            operations.DailyActionState.READY,
        )

    def test_afternoon_routine_can_fetch_missing_games_itself(self) -> None:
        overview = self._overview(
            now_utc=datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc),
            game_day=self._games(count=0, first_start=None),
        )
        self.assertEqual(
            overview.prediction_action.state,
            operations.DailyActionState.NEED_DATA,
        )
        self.assertEqual(
            overview.afternoon_action.state,
            operations.DailyActionState.READY,
        )

    def test_afternoon_routine_preserves_terminal_prediction_state(self) -> None:
        overview = self._overview(
            now_utc=datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc),
            prediction_slot_state=operations.PredictionSlotState.FAILED,
        )
        self.assertEqual(
            overview.afternoon_action.state,
            operations.DailyActionState.BLOCKED,
        )

    def test_paris_midnight_does_not_open_previous_results_early(self) -> None:
        overview = self._overview(
            now_utc=datetime(2026, 9, 12, 22, 30, tzinfo=timezone.utc)
        )
        self.assertEqual(overview.results_action.state, operations.DailyActionState.TOO_EARLY)

    def test_completed_results_are_not_run_again(self) -> None:
        overview = self._overview(
            results_target_date=None,
            latest_score_pending_count=None,
        )
        self.assertEqual(overview.results_action.state, operations.DailyActionState.DONE)

    def test_consumed_checkpoint_is_never_reopened(self) -> None:
        overview = self._overview(checkpoint_slot_exists=True)
        self.assertEqual(overview.results_action.state, operations.DailyActionState.BLOCKED)

    def test_missing_past_certified_day_disables_results(self) -> None:
        overview = self._overview(
            results_target_date=None,
            has_past_certified=False,
        )
        self.assertEqual(
            overview.results_action.state,
            operations.DailyActionState.NOT_AVAILABLE,
        )

    def test_older_pending_certified_day_can_be_selected(self) -> None:
        older = TARGET - timedelta(days=2)
        overview = self._overview(
            results_target_date=older,
            latest_score_pending_count=2,
        )
        self.assertEqual(overview.results_target_date, older)
        self.assertEqual(overview.results_action.state, operations.DailyActionState.READY)
        self.assertIn(older.strftime("%d/%m/%Y"), overview.results_action.message)

    def test_git_problem_blocks_prediction_and_results_not_data(self) -> None:
        overview = self._overview(git=self._git(ready=False))
        self.assertEqual(overview.prediction_action.state, operations.DailyActionState.BLOCKED)
        self.assertEqual(overview.results_action.state, operations.DailyActionState.BLOCKED)
        self.assertTrue(overview.data_action.can_execute)

    def test_integrity_error_fails_closed(self) -> None:
        overview = self._overview(integrity_errors=("preuve invalide",))
        self.assertEqual(overview.prediction_action.state, operations.DailyActionState.BLOCKED)
        self.assertEqual(overview.results_action.state, operations.DailyActionState.BLOCKED)

    def test_non_today_target_cannot_mutate_prediction_or_results(self) -> None:
        overview = self._overview(paris_today=TARGET + timedelta(days=1))
        self.assertEqual(
            overview.prediction_action.state,
            operations.DailyActionState.NOT_AVAILABLE,
        )
        self.assertEqual(
            overview.results_action.state,
            operations.DailyActionState.NOT_AVAILABLE,
        )

    def test_missing_database_is_read_without_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "missing.db"
            state = operations.load_local_game_day_state(
                TARGET,
                database_path=database,
            )
            self.assertEqual(state, operations.LocalGameDayState(0, 0, None))
            self.assertFalse(database.exists())

    def test_sqlite_summary_counts_final_games(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "games.db"
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    """
                    CREATE TABLE games (
                        game_id INTEGER PRIMARY KEY,
                        official_date TEXT NOT NULL,
                        game_datetime_utc TEXT NOT NULL,
                        status_code TEXT,
                        status_detail TEXT
                    )
                    """
                )
                connection.executemany(
                    "INSERT INTO games VALUES (?, ?, ?, ?, ?)",
                    [
                        (1, TARGET.isoformat(), "2026-09-13T18:00:00Z", "F", "Final"),
                        (2, TARGET.isoformat(), "2026-09-13T19:00:00Z", "S", "Scheduled"),
                    ],
                )
                connection.commit()
            finally:
                connection.close()
            state = operations.load_local_game_day_state(
                TARGET,
                database_path=database,
            )
            self.assertEqual((state.game_count, state.final_game_count), (2, 1))
            self.assertEqual(state.first_start_utc, FIRST_START)

    def test_prediction_slot_states_are_presence_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            self.assertEqual(
                operations.inspect_prediction_slot(TARGET, project_directory=project),
                operations.PredictionSlotState.ABSENT,
            )
            slot = project / operations.PREDICTION_ROOT / TARGET.isoformat()
            slot.mkdir(parents=True)
            (slot / "RESERVED").write_text("reserved", encoding="utf-8")
            self.assertEqual(
                operations.inspect_prediction_slot(TARGET, project_directory=project),
                operations.PredictionSlotState.PARTIAL,
            )
            (slot / "FAILED.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                operations.inspect_prediction_slot(TARGET, project_directory=project),
                operations.PredictionSlotState.FAILED,
            )

    def test_git_inspection_is_local_and_read_only(self) -> None:
        outputs = iter(["main", "a" * 40, "a" * 40, ""])
        with mock.patch.object(
            operations,
            "_run_git",
            side_effect=lambda *_args, **_kwargs: next(outputs),
        ) as run_git:
            state = operations.inspect_git_workspace(
                project_directory=Path("project")
            )
        self.assertTrue(state.ready_for_publication)
        self.assertEqual(run_git.call_count, 4)

    def test_inspector_never_invokes_network_model_or_daily_engines(self) -> None:
        forbidden = AssertionError("effet externe interdit")
        with (
            mock.patch.object(
                operations,
                "load_local_game_day_state",
                return_value=self._games(),
            ),
            mock.patch.object(
                operations,
                "inspect_git_workspace",
                return_value=self._git(),
            ),
            mock.patch.object(
                operations,
                "inspect_prediction_slot",
                return_value=operations.PredictionSlotState.ABSENT,
            ),
            mock.patch.object(
                operations,
                "list_certified_prediction_dates",
                return_value=[],
            ),
            mock.patch(
                "requests.get",
                side_effect=forbidden,
                create=True,
            ) as network,
        ):
            overview = operations.inspect_daily_operations(
                TARGET,
                now_utc=NOW,
                project_directory=Path("project"),
                database_path=Path("database"),
            )
        self.assertEqual(overview.target_date, TARGET)
        network.assert_not_called()

    def test_inspector_selects_oldest_certified_day_still_pending(self) -> None:
        oldest = TARGET - timedelta(days=3)
        pending = TARGET - timedelta(days=2)
        newest = TARGET - timedelta(days=1)
        summaries = {
            oldest: SimpleNamespace(pending_count=0),
            pending: SimpleNamespace(pending_count=2),
        }

        def load_day(value, **_kwargs):
            return SimpleNamespace(predictions=(value.isoformat(),))

        def load_summary(value, **_kwargs):
            return summaries[value]

        with (
            mock.patch.object(
                operations,
                "load_local_game_day_state",
                return_value=self._games(),
            ),
            mock.patch.object(
                operations,
                "inspect_git_workspace",
                return_value=self._git(),
            ),
            mock.patch.object(
                operations,
                "inspect_prediction_slot",
                return_value=operations.PredictionSlotState.ABSENT,
            ),
            mock.patch.object(
                operations,
                "list_certified_prediction_dates",
                return_value=[oldest, pending, newest],
            ),
            mock.patch.object(
                operations,
                "load_certified_prediction_day",
                side_effect=load_day,
            ),
            mock.patch.object(
                operations,
                "load_latest_score_summary",
                side_effect=load_summary,
            ) as scores,
        ):
            overview = operations.inspect_daily_operations(
                TARGET,
                now_utc=NOW,
                project_directory=Path("project"),
                database_path=Path("database"),
            )
        self.assertEqual(overview.results_target_date, pending)
        self.assertEqual(overview.latest_score_pending_count, 2)
        self.assertEqual(scores.call_count, 2)

    def test_daily_data_refresh_uses_only_audited_ingestion_service(self) -> None:
        expected = SimpleNamespace(run_id=42, games_received=15, games_saved=15)
        database = Path("daily.db")
        data_directory = Path("daily-data")
        with mock.patch.object(
            operations,
            "run_schedule_ingestion",
            return_value=expected,
        ) as ingestion:
            result = operations.refresh_daily_mlb_data(
                TARGET,
                now_utc=NOW,
                database_path=database,
                data_directory=data_directory,
            )
        self.assertIs(result, expected)
        ingestion.assert_called_once_with(
            start_date=TARGET,
            end_date=TARGET,
            database_path=database,
            data_directory=data_directory,
        )

    def test_daily_data_refresh_rejects_another_day_before_ingestion(self) -> None:
        with mock.patch.object(operations, "run_schedule_ingestion") as ingestion:
            with self.assertRaisesRegex(
                operations.DailyOperationsError,
                "seulement actualiser les matchs d'aujourd'hui",
            ):
                operations.refresh_daily_mlb_data(
                    TARGET - timedelta(days=1),
                    now_utc=NOW,
                )
        ingestion.assert_not_called()

    def test_daily_data_refresh_rejects_a_naive_clock(self) -> None:
        with mock.patch.object(operations, "run_schedule_ingestion") as ingestion:
            with self.assertRaisesRegex(
                operations.DailyOperationsError,
                "horloge de la collecte",
            ):
                operations.refresh_daily_mlb_data(
                    TARGET,
                    now_utc=datetime(2026, 9, 13, 8, 0),
                )
        ingestion.assert_not_called()


if __name__ == "__main__":
    unittest.main()
