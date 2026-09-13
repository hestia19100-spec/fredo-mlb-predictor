"""Tests fermes des routines separees du matin et de l'apres-midi."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import lpf_edge_daily_operations as operations
from src.ingestion_service import ScheduleIngestionResult


TODAY = date(2026, 9, 13)
NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)


class LPFEdgeSplitDailyRoutinesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.database = self.project / "data/fredo_mlb.db"
        self.data_directory = self.project / "data"
        self.refresh = ScheduleIngestionResult(
            run_id=91,
            start_date=TODAY,
            end_date=TODAY,
            games_received=15,
            games_saved=15,
            archive_relative_path="data/raw/mlb_schedule/day.json.gz",
            response_sha256="a" * 64,
            code_version="b" * 40,
        )
        self.prediction = operations.DailyPredictionPublication(
            target_date=TODAY,
            batch_id="c" * 64,
            results_commit="d" * 40,
            certification_commit="e" * 40,
            results_paths=("result",),
            certification_paths=("certification",),
        )
        self.odds = operations.DailyAfternoonOddsOutcome(
            status=operations.DailyAfternoonOddsStatus.REUSED,
            run_id=7,
            events_matched=15,
            bookmaker_quotes_saved=120,
            quota_remaining=None,
            message="Collecte réutilisée.",
        )

    def _action(
        self,
        key: str,
        state: operations.DailyActionState,
        message: str = "Prêt.",
    ) -> operations.DailyAction:
        return operations.DailyAction(
            key=key,
            label="Routine",
            state=state,
            message=message,
            can_execute=state is operations.DailyActionState.READY,
        )

    def _initial_overview(
        self,
        state: operations.DailyActionState = operations.DailyActionState.READY,
        message: str = "Prêt.",
    ) -> mock.Mock:
        return mock.Mock(
            afternoon_action=self._action("afternoon", state, message),
        )

    def _refreshed_overview(
        self,
        state: operations.DailyActionState = operations.DailyActionState.READY,
        message: str = "Prêt.",
    ) -> mock.Mock:
        return mock.Mock(
            prediction_action=self._action("predictions", state, message),
        )

    def _run(self) -> operations.DailyAfternoonPublication:
        return operations.execute_afternoon_prediction_routine(
            TODAY,
            project_directory=self.project,
            database_path=self.database,
            data_directory=self.data_directory,
        )

    def test_afternoon_success_refreshes_before_prediction(self) -> None:
        parent = mock.Mock()
        with (
            mock.patch.object(
                operations,
                "_utc_now",
                side_effect=(NOW, NOW + timedelta(seconds=1)),
            ),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                side_effect=(self._initial_overview(), self._refreshed_overview()),
            ) as inspect_state,
            mock.patch.object(
                operations,
                "refresh_daily_mlb_data",
                return_value=self.refresh,
            ) as refresh,
            mock.patch.object(
                operations,
                "_prepare_afternoon_odds",
                return_value=self.odds,
            ) as prepare_odds,
            mock.patch.object(
                operations,
                "execute_daily_prediction_publication",
                return_value=self.prediction,
            ) as predict,
        ):
            parent.attach_mock(refresh, "refresh")
            parent.attach_mock(prepare_odds, "odds")
            parent.attach_mock(predict, "predict")
            result = self._run()

        self.assertEqual(result.target_date, TODAY)
        self.assertIs(result.data_refresh, self.refresh)
        self.assertIs(result.odds, self.odds)
        self.assertIs(result.prediction, self.prediction)
        self.assertEqual(
            [call[0] for call in parent.method_calls],
            ["refresh", "odds", "predict"],
        )
        self.assertEqual(inspect_state.call_count, 2)
        refresh.assert_called_once_with(
            TODAY,
            now_utc=NOW,
            database_path=self.database,
            data_directory=self.data_directory,
        )
        predict.assert_called_once_with(
            TODAY,
            project_directory=self.project.resolve(),
            database_path=self.database,
            data_directory=self.data_directory,
        )
        prepare_odds.assert_called_once_with(
            TODAY,
            now_utc=NOW + timedelta(seconds=1),
            database_path=self.database,
            data_directory=self.data_directory,
        )

    def test_blocked_preflight_never_contacts_mlb_or_model(self) -> None:
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._initial_overview(
                    operations.DailyActionState.TOO_EARLY,
                    "Attends midi.",
                ),
            ),
            mock.patch.object(operations, "refresh_daily_mlb_data") as refresh,
            mock.patch.object(
                operations,
                "_prepare_afternoon_odds",
            ) as prepare_odds,
            mock.patch.object(
                operations,
                "execute_daily_prediction_publication",
            ) as predict,
        ):
            with self.assertRaises(operations.DailyAfternoonAutomationError) as raised:
                self._run()
        self.assertEqual(raised.exception.stage, operations.DailyAfternoonStage.PREFLIGHT)
        self.assertIsNone(raised.exception.data_refresh)
        refresh.assert_not_called()
        prepare_odds.assert_not_called()
        predict.assert_not_called()

    def test_refresh_failure_stops_before_prediction(self) -> None:
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._initial_overview(),
            ),
            mock.patch.object(
                operations,
                "refresh_daily_mlb_data",
                side_effect=OSError("MLB indisponible"),
            ),
            mock.patch.object(
                operations,
                "execute_daily_prediction_publication",
            ) as predict,
        ):
            with self.assertRaises(operations.DailyAfternoonAutomationError) as raised:
                self._run()
        self.assertEqual(raised.exception.stage, operations.DailyAfternoonStage.DATA_REFRESH)
        self.assertIsNone(raised.exception.data_refresh)
        predict.assert_not_called()

    def test_second_preflight_can_stop_after_successful_refresh(self) -> None:
        with (
            mock.patch.object(
                operations,
                "_utc_now",
                side_effect=(NOW, NOW + timedelta(seconds=1)),
            ),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                side_effect=(
                    self._initial_overview(),
                    self._refreshed_overview(
                        operations.DailyActionState.TOO_LATE,
                        "Délai dépassé.",
                    ),
                ),
            ),
            mock.patch.object(
                operations,
                "refresh_daily_mlb_data",
                return_value=self.refresh,
            ),
            mock.patch.object(
                operations,
                "_prepare_afternoon_odds",
            ) as prepare_odds,
            mock.patch.object(
                operations,
                "execute_daily_prediction_publication",
            ) as predict,
        ):
            with self.assertRaises(operations.DailyAfternoonAutomationError) as raised:
                self._run()
        self.assertEqual(
            raised.exception.stage,
            operations.DailyAfternoonStage.PREDICTION_PREFLIGHT,
        )
        self.assertIs(raised.exception.data_refresh, self.refresh)
        prepare_odds.assert_not_called()
        predict.assert_not_called()

    def test_prediction_failure_reports_that_data_was_refreshed(self) -> None:
        with (
            mock.patch.object(
                operations,
                "_utc_now",
                side_effect=(NOW, NOW + timedelta(seconds=1)),
            ),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                side_effect=(self._initial_overview(), self._refreshed_overview()),
            ),
            mock.patch.object(
                operations,
                "refresh_daily_mlb_data",
                return_value=self.refresh,
            ),
            mock.patch.object(
                operations,
                "execute_daily_prediction_publication",
                side_effect=operations.DailyPredictionAutomationError(
                    operations.DailyPredictionStage.PREDICTION,
                    "échec contrôlé",
                ),
            ),
        ):
            with self.assertRaises(operations.DailyAfternoonAutomationError) as raised:
                self._run()
        self.assertEqual(raised.exception.stage, operations.DailyAfternoonStage.PREDICTION)
        self.assertIs(raised.exception.data_refresh, self.refresh)

    def test_unexpected_programming_error_is_not_hidden(self) -> None:
        with (
            mock.patch.object(
                operations,
                "_utc_now",
                side_effect=(NOW, NOW + timedelta(seconds=1)),
            ),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                side_effect=(self._initial_overview(), self._refreshed_overview()),
            ),
            mock.patch.object(
                operations,
                "refresh_daily_mlb_data",
                return_value=self.refresh,
            ),
            mock.patch.object(
                operations,
                "execute_daily_prediction_publication",
                side_effect=RuntimeError("erreur de programmation"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "programmation"):
                self._run()

    def test_another_date_is_rejected_before_inspection(self) -> None:
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(operations, "inspect_daily_operations") as inspect_state,
        ):
            with self.assertRaises(operations.DailyAfternoonAutomationError):
                operations.execute_afternoon_prediction_routine(
                    TODAY - timedelta(days=1),
                    project_directory=self.project,
                )
        inspect_state.assert_not_called()

    def test_public_signature_exposes_no_clock_or_engine_override(self) -> None:
        parameters = inspect.signature(
            operations.execute_afternoon_prediction_routine
        ).parameters
        self.assertNotIn("now_utc", parameters)
        self.assertNotIn("refresh_function", parameters)
        self.assertNotIn("prediction_function", parameters)

    def test_keyboard_interrupt_is_never_hidden(self) -> None:
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._initial_overview(),
            ),
            mock.patch.object(
                operations,
                "refresh_daily_mlb_data",
                side_effect=KeyboardInterrupt,
            ),
        ):
            with self.assertRaises(KeyboardInterrupt):
                self._run()


if __name__ == "__main__":
    unittest.main()
