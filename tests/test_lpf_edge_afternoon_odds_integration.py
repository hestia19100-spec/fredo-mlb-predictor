"""Tests de l’intégration des cotes dans la routine de l’après-midi."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import lpf_edge_daily_operations as operations
from src.ingestion_service import ScheduleIngestionResult
from src.odds_ingestion_service import OddsIngestionResult


TODAY = date(2026, 9, 13)
NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)


class LPFEdgeAfternoonOddsIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.database = self.project / "data/fredo_mlb.db"
        self.data_directory = self.project / "data"

    def _ready_action(self) -> operations.DailyAction:
        return operations.DailyAction(
            key="odds",
            label="Cotes",
            state=operations.DailyActionState.READY,
            message="Un appel sera effectué.",
            can_execute=True,
        )

    def _blocked_action(self) -> operations.DailyAction:
        return operations.DailyAction(
            key="odds",
            label="Cotes",
            state=operations.DailyActionState.BLOCKED,
            message="Clé API absente.",
            can_execute=False,
        )

    def _empty_display(self) -> mock.Mock:
        return mock.Mock(
            run_id=None,
            quoted_game_count=0,
            quote_count=0,
        )

    def _overview(self, action: operations.DailyAction) -> mock.Mock:
        return mock.Mock(
            action=action,
            latest_run_id=None,
            events_matched=None,
            bookmaker_quotes_saved=None,
            quota_remaining=None,
        )

    def _prepare(self) -> operations.DailyAfternoonOddsOutcome:
        return operations._prepare_afternoon_odds(
            TODAY,
            now_utc=NOW,
            database_path=self.database,
            data_directory=self.data_directory,
        )

    def test_existing_french_collection_is_reused_without_api_call(self) -> None:
        display = mock.Mock(
            run_id=41,
            quoted_game_count=15,
            quote_count=306,
        )
        with (
            mock.patch.object(
                operations,
                "load_latest_moneyline_odds_display",
                return_value=display,
            ) as load_display,
            mock.patch.object(
                operations, "inspect_daily_odds_collection"
            ) as inspect_odds,
            mock.patch.object(
                operations, "execute_daily_odds_collection"
            ) as collect,
        ):
            outcome = self._prepare()

        self.assertEqual(
            outcome.status,
            operations.DailyAfternoonOddsStatus.REUSED,
        )
        self.assertEqual(outcome.run_id, 41)
        self.assertEqual(outcome.events_matched, 15)
        self.assertEqual(outcome.bookmaker_quotes_saved, 306)
        load_display.assert_called_once_with(
            TODAY,
            database_path=self.database,
            required_region="fr",
        )
        inspect_odds.assert_not_called()
        collect.assert_not_called()

    def test_missing_collection_triggers_exactly_one_api_collection(self) -> None:
        result = OddsIngestionResult(
            run_id=42,
            target_date=TODAY,
            events_received=15,
            events_matched=15,
            unmatched_events=0,
            bookmaker_quotes_saved=208,
            raw_response_path="data/raw/cotes.json.gz",
            response_sha256="a" * 64,
            quota_remaining=496,
            quota_used=4,
            quota_last_cost=1,
            code_version="b" * 40,
        )
        with (
            mock.patch.object(
                operations,
                "load_latest_moneyline_odds_display",
                return_value=self._empty_display(),
            ),
            mock.patch.object(
                operations,
                "inspect_daily_odds_collection",
                return_value=self._overview(self._ready_action()),
            ),
            mock.patch.object(
                operations,
                "execute_daily_odds_collection",
                return_value=result,
            ) as collect,
        ):
            outcome = self._prepare()

        self.assertEqual(
            outcome.status,
            operations.DailyAfternoonOddsStatus.COLLECTED,
        )
        self.assertEqual(outcome.run_id, 42)
        self.assertEqual(outcome.bookmaker_quotes_saved, 208)
        self.assertEqual(outcome.quota_remaining, 496)
        collect.assert_called_once_with(
            TODAY,
            now_utc=NOW,
            database_path=self.database,
            data_directory=self.data_directory,
        )

    def test_controlled_collection_failure_is_not_retried(self) -> None:
        with (
            mock.patch.object(
                operations,
                "load_latest_moneyline_odds_display",
                return_value=self._empty_display(),
            ),
            mock.patch.object(
                operations,
                "inspect_daily_odds_collection",
                return_value=self._overview(self._ready_action()),
            ),
            mock.patch.object(
                operations,
                "execute_daily_odds_collection",
                side_effect=operations.DailyOddsCollectionError(
                    operations.DailyOddsCollectionStage.COLLECTION,
                    "Fournisseur indisponible.",
                ),
            ) as collect,
        ):
            outcome = self._prepare()

        self.assertEqual(
            outcome.status,
            operations.DailyAfternoonOddsStatus.FAILED,
        )
        self.assertIn("continue", outcome.message)
        collect.assert_called_once()

    def test_unavailable_collection_consumes_no_credit(self) -> None:
        with (
            mock.patch.object(
                operations,
                "load_latest_moneyline_odds_display",
                return_value=self._empty_display(),
            ),
            mock.patch.object(
                operations,
                "inspect_daily_odds_collection",
                return_value=self._overview(self._blocked_action()),
            ),
            mock.patch.object(
                operations, "execute_daily_odds_collection"
            ) as collect,
        ):
            outcome = self._prepare()

        self.assertEqual(
            outcome.status,
            operations.DailyAfternoonOddsStatus.UNAVAILABLE,
        )
        self.assertEqual(outcome.message, "Clé API absente.")
        collect.assert_not_called()

    def test_unexpected_odds_error_is_never_hidden(self) -> None:
        with mock.patch.object(
            operations,
            "load_latest_moneyline_odds_display",
            side_effect=RuntimeError("erreur de programmation"),
        ):
            with self.assertRaisesRegex(RuntimeError, "programmation"):
                self._prepare()

    def test_controlled_odds_failure_does_not_block_prediction(self) -> None:
        refresh = ScheduleIngestionResult(
            run_id=91,
            start_date=TODAY,
            end_date=TODAY,
            games_received=15,
            games_saved=15,
            archive_relative_path="data/raw/mlb.json.gz",
            response_sha256="c" * 64,
            code_version="d" * 40,
        )
        failed_odds = operations.DailyAfternoonOddsOutcome(
            status=operations.DailyAfternoonOddsStatus.FAILED,
            run_id=None,
            events_matched=None,
            bookmaker_quotes_saved=None,
            quota_remaining=None,
            message="Échec contrôlé ; la prédiction continue.",
        )
        prediction = operations.DailyPredictionPublication(
            target_date=TODAY,
            batch_id="e" * 64,
            results_commit="f" * 40,
            certification_commit="1" * 40,
            results_paths=("result",),
            certification_paths=("certification",),
        )
        ready = mock.Mock(
            afternoon_action=self._ready_action(),
            prediction_action=self._ready_action(),
        )
        with (
            mock.patch.object(
                operations,
                "_utc_now",
                side_effect=(NOW, NOW + timedelta(seconds=1)),
            ),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                side_effect=(ready, ready),
            ),
            mock.patch.object(
                operations,
                "refresh_daily_mlb_data",
                return_value=refresh,
            ),
            mock.patch.object(
                operations,
                "_prepare_afternoon_odds",
                return_value=failed_odds,
            ),
            mock.patch.object(
                operations,
                "execute_daily_prediction_publication",
                return_value=prediction,
            ) as predict,
        ):
            publication = operations.execute_afternoon_prediction_routine(
                TODAY,
                project_directory=self.project,
                database_path=self.database,
                data_directory=self.data_directory,
            )

        self.assertIs(publication.odds, failed_odds)
        self.assertIs(publication.prediction, prediction)
        predict.assert_called_once()

    def test_prediction_failure_keeps_the_odds_outcome(self) -> None:
        refresh = mock.Mock()
        odds = operations.DailyAfternoonOddsOutcome(
            status=operations.DailyAfternoonOddsStatus.REUSED,
            run_id=7,
            events_matched=15,
            bookmaker_quotes_saved=100,
            quota_remaining=None,
            message="Réutilisée.",
        )
        ready = mock.Mock(
            afternoon_action=self._ready_action(),
            prediction_action=self._ready_action(),
        )
        with (
            mock.patch.object(
                operations,
                "_utc_now",
                side_effect=(NOW, NOW + timedelta(seconds=1)),
            ),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                side_effect=(ready, ready),
            ),
            mock.patch.object(
                operations,
                "refresh_daily_mlb_data",
                return_value=refresh,
            ),
            mock.patch.object(
                operations,
                "_prepare_afternoon_odds",
                return_value=odds,
            ),
            mock.patch.object(
                operations,
                "execute_daily_prediction_publication",
                side_effect=operations.DailyPredictionAutomationError(
                    operations.DailyPredictionStage.PREDICTION,
                    "Échec.",
                ),
            ),
        ):
            with self.assertRaises(
                operations.DailyAfternoonAutomationError
            ) as raised:
                operations.execute_afternoon_prediction_routine(
                    TODAY,
                    project_directory=self.project,
                    database_path=self.database,
                    data_directory=self.data_directory,
                )

        self.assertIs(raised.exception.data_refresh, refresh)
        self.assertIs(raised.exception.odds, odds)


if __name__ == "__main__":
    unittest.main()
