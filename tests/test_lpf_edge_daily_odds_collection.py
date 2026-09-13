"""Tests du bouton quotidien de collecte des cotes dans LPF Edge."""

from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from src import lpf_edge_daily_operations as operations
from src.database import get_connection
from src.game_repository import save_schedule
from src.mlb_api import ScheduledGame
from src.odds_api import OddsAPIError
from src.odds_ingestion_service import OddsIngestionResult
from src.odds_repository import start_odds_ingestion_run


TODAY = date(2026, 9, 13)
NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
SECRET = "x" * 24


class LPFEdgeDailyOddsCollectionTests(unittest.TestCase):
    """Vérifie les contrôles locaux sans utiliser le réseau."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.database = self.root / "data" / "fredo_mlb.db"
        self.data_directory = self.root / "data"

    def _save_game(self) -> None:
        save_schedule(
            (
                ScheduledGame(
                    game_id=9100,
                    season=2026,
                    official_date=TODAY.isoformat(),
                    game_datetime_utc="2026-09-13T18:10:00Z",
                    game_type="R",
                    status_code="S",
                    status_detail="Scheduled",
                    away_team_id=10,
                    away_team_name="New York Mets",
                    home_team_id=20,
                    home_team_name="Chicago Cubs",
                    away_score=None,
                    home_score=None,
                    venue_id=1,
                    venue_name="Test Park",
                    doubleheader="N",
                    game_number=1,
                    away_probable_pitcher_id=None,
                    away_probable_pitcher_name=None,
                    home_probable_pitcher_id=None,
                    home_probable_pitcher_name=None,
                ),
            ),
            self.database,
        )

    def _inspect(self) -> operations.DailyOddsCollectionOverview:
        return operations.inspect_daily_odds_collection(
            TODAY,
            now_utc=NOW,
            database_path=self.database,
        )

    def _result(self) -> OddsIngestionResult:
        return OddsIngestionResult(
            run_id=17,
            target_date=TODAY,
            events_received=15,
            events_matched=15,
            unmatched_events=0,
            bookmaker_quotes_saved=45,
            raw_response_path="data/raw/the_odds_api_mlb_moneyline/day.json.gz",
            response_sha256="a" * 64,
            quota_remaining=484,
            quota_used=16,
            quota_last_cost=1,
            code_version="b" * 40,
        )

    def test_missing_key_is_blocked_without_network(self) -> None:
        """Une clé absente désactive le bouton sans appeler le fournisseur."""
        self._save_game()
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch(
                "requests.get",
                side_effect=AssertionError("réseau"),
                create=True,
            ) as get,
        ):
            overview = self._inspect()
        self.assertFalse(overview.api_configured)
        self.assertEqual(overview.action.state, operations.DailyActionState.BLOCKED)
        self.assertIn("THE_ODDS_API_KEY", overview.action.message)
        get.assert_not_called()

    def test_missing_database_is_read_without_creation(self) -> None:
        """L’inspection locale ne crée pas la base quand elle est absente."""
        with mock.patch.dict(
            "os.environ",
            {"THE_ODDS_API_KEY": SECRET},
            clear=True,
        ):
            overview = self._inspect()
        self.assertEqual(
            overview.action.state,
            operations.DailyActionState.NEED_DATA,
        )
        self.assertFalse(self.database.exists())

    def test_latest_success_exposes_counts_and_quota_only(self) -> None:
        """L’écran relit les compteurs utiles sans exposer le secret."""
        self._save_game()
        run_id = start_odds_ingestion_run(
            target_date=TODAY,
            code_version="b" * 40,
            database_path=self.database,
        )
        with get_connection(self.database) as connection:
            connection.execute(
                """
                UPDATE odds_ingestion_runs
                SET
                    status = 'success',
                    completed_at_utc = '2026-09-13T12:05:00+00:00',
                    events_received = 15,
                    events_matched = 14,
                    bookmaker_quotes_saved = 42,
                    quota_remaining = 484,
                    quota_used = 16,
                    quota_last_cost = 1
                WHERE run_id = ?
                """,
                (run_id,),
            )

        with mock.patch.dict(
            "os.environ",
            {"THE_ODDS_API_KEY": SECRET},
            clear=True,
        ):
            overview = self._inspect()

        self.assertTrue(overview.api_configured)
        self.assertEqual(overview.latest_run_id, run_id)
        self.assertEqual(overview.events_received, 15)
        self.assertEqual(overview.events_matched, 14)
        self.assertEqual(overview.bookmaker_quotes_saved, 42)
        self.assertEqual(overview.quota_remaining, 484)
        self.assertEqual(overview.quota_last_cost, 1)
        self.assertEqual(overview.action.state, operations.DailyActionState.READY)
        self.assertNotIn(SECRET, repr(overview))

    def test_active_collection_disables_second_click(self) -> None:
        """Un journal en cours empêche une collecte concurrente."""
        self._save_game()
        start_odds_ingestion_run(
            target_date=TODAY,
            code_version="b" * 40,
            database_path=self.database,
        )
        with mock.patch.dict(
            "os.environ",
            {"THE_ODDS_API_KEY": SECRET},
            clear=True,
        ):
            overview = self._inspect()
        self.assertEqual(overview.latest_status, "started")
        self.assertEqual(overview.action.state, operations.DailyActionState.BLOCKED)
        self.assertFalse(overview.action.can_execute)

    def test_success_calls_the_audited_service_once(self) -> None:
        """Le bouton délègue une seule fois au service audité existant."""
        self._save_game()
        receipt = self._result()
        with (
            mock.patch.dict(
                "os.environ",
                {"THE_ODDS_API_KEY": SECRET},
                clear=True,
            ),
            mock.patch.object(
                operations,
                "run_odds_ingestion",
                return_value=receipt,
            ) as run,
        ):
            result = operations.execute_daily_odds_collection(
                TODAY,
                now_utc=NOW,
                database_path=self.database,
                data_directory=self.data_directory,
            )
        self.assertIs(result, receipt)
        run.assert_called_once_with(
            target_date=TODAY,
            database_path=self.database,
            data_directory=self.data_directory,
        )

    def test_foreign_date_stops_before_service(self) -> None:
        """Le bouton ne peut jamais collecter une autre journée."""
        with (
            mock.patch.dict(
                "os.environ",
                {"THE_ODDS_API_KEY": SECRET},
                clear=True,
            ),
            mock.patch.object(operations, "run_odds_ingestion") as run,
        ):
            with self.assertRaises(operations.DailyOddsCollectionError) as caught:
                operations.execute_daily_odds_collection(
                    date(2026, 9, 12),
                    now_utc=NOW,
                    database_path=self.database,
                    data_directory=self.data_directory,
                )
        self.assertEqual(
            caught.exception.stage,
            operations.DailyOddsCollectionStage.PREFLIGHT,
        )
        run.assert_not_called()

    def test_provider_failure_is_controlled_without_retry(self) -> None:
        """Une erreur fournisseur remonte une fois sans relance cachée."""
        self._save_game()
        with (
            mock.patch.dict(
                "os.environ",
                {"THE_ODDS_API_KEY": SECRET},
                clear=True,
            ),
            mock.patch.object(
                operations,
                "run_odds_ingestion",
                side_effect=OddsAPIError("Erreur fournisseur contrôlée."),
            ) as run,
        ):
            with self.assertRaises(operations.DailyOddsCollectionError) as caught:
                operations.execute_daily_odds_collection(
                    TODAY,
                    now_utc=NOW,
                    database_path=self.database,
                    data_directory=self.data_directory,
                )
        self.assertEqual(
            caught.exception.stage,
            operations.DailyOddsCollectionStage.COLLECTION,
        )
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
