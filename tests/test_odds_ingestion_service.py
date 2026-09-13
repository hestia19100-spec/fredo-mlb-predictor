"""Tests de l’archivage et du stockage audité des cotes MLB."""

from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from src import odds_repository
from src.database import get_connection, list_applied_migrations
from src.game_repository import save_schedule
from src.mlb_api import ScheduledGame
from src.odds_api import (
    MoneylineBookmaker,
    MoneylineEvent,
    OddsAPIConfigurationError,
    OddsFetchResult,
)
from src.odds_ingestion_service import (
    OddsIngestionError,
    run_odds_ingestion,
)
from src.odds_repository import (
    OddsRepositoryError,
    get_odds_ingestion_run,
    initialize_odds_storage,
    list_moneyline_odds_for_date,
)
from src.raw_archive import load_raw_archive


UTC = timezone.utc


class OddsIngestionServiceTests(unittest.TestCase):
    """Contrôle une collecte complète sans aucun appel réseau réel."""

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory(
            ignore_cleanup_errors=True
        )
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / "data" / "fredo_mlb.db"
        self.data_directory = self.root / "data"
        self.target_date = date(2026, 9, 13)
        self.start = datetime(2026, 9, 13, 17, 10, tzinfo=UTC)
        self._save_game(game_id=9001, start=self.start)

    def _save_game(
        self,
        *,
        game_id: int,
        start: datetime,
        away_name: str = "New York Mets",
        home_name: str = "Chicago Cubs",
    ) -> None:
        save_schedule(
            (
                ScheduledGame(
                    game_id=game_id,
                    season=2026,
                    official_date=self.target_date.isoformat(),
                    game_datetime_utc=(
                        start.astimezone(UTC).isoformat().replace("+00:00", "Z")
                    ),
                    game_type="R",
                    status_code="S",
                    status_detail="Scheduled",
                    away_team_id=game_id * 10 + 1,
                    away_team_name=away_name,
                    home_team_id=game_id * 10 + 2,
                    home_team_name=home_name,
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
            self.database_path,
        )

    def _event(
        self,
        *,
        event_id: str = "event-1",
        start: datetime | None = None,
        away_name: str = "New York Mets",
        home_name: str = "Chicago Cubs",
        bookmakers: bool = True,
    ) -> MoneylineEvent:
        bookmaker_values = (
            MoneylineBookmaker(
                key="unibet",
                title="Unibet",
                last_update_utc=datetime(2026, 9, 13, 15, 1, tzinfo=UTC),
                away_decimal_odds=Decimal("2.15"),
                home_decimal_odds=Decimal("1.72"),
            ),
            MoneylineBookmaker(
                key="betclic",
                title="Betclic",
                last_update_utc=datetime(2026, 9, 13, 15, 2, tzinfo=UTC),
                away_decimal_odds=Decimal("2.10"),
                home_decimal_odds=Decimal("1.75"),
            ),
        ) if bookmakers else ()
        return MoneylineEvent(
            provider_event_id=event_id,
            commence_time_utc=start or self.start,
            away_team_name=away_name,
            home_team_name=home_name,
            bookmakers=bookmaker_values,
        )

    def _fetch_result(
        self,
        *events: MoneylineEvent,
        raw_content: bytes = b'[{"audited":"odds"}]',
        declared_sha256: str | None = None,
    ) -> OddsFetchResult:
        return OddsFetchResult(
            provider="the_odds_api_v4",
            sport_key="baseball_mlb",
            region="fr",
            market="h2h",
            odds_format="decimal",
            response_received_at_utc=datetime(
                2026, 9, 13, 15, 5, tzinfo=UTC
            ),
            response_body_sha256=(
                declared_sha256
                or hashlib.sha256(raw_content).hexdigest()
            ),
            quota_remaining=497,
            quota_used=3,
            quota_last_cost=1,
            raw_content=raw_content,
            events=tuple(events),
        )

    def _run(self, fetch_result: OddsFetchResult):
        with mock.patch(
            "src.odds_ingestion_service.fetch_mlb_moneyline_odds",
            return_value=fetch_result,
        ) as fetch:
            result = run_odds_ingestion(
                target_date=self.target_date,
                database_path=self.database_path,
                data_directory=self.data_directory,
                code_version="abc123",
            )
        fetch.assert_called_once_with()
        return result

    def test_success_archives_matches_and_saves_exact_decimal_odds(self) -> None:
        """Le parcours nominal conserve preuve, journal et cotes exactes."""
        fetch_result = self._fetch_result(self._event())

        result = self._run(fetch_result)

        self.assertEqual(result.events_received, 1)
        self.assertEqual(result.events_matched, 1)
        self.assertEqual(result.unmatched_events, 0)
        self.assertEqual(result.bookmaker_quotes_saved, 2)
        self.assertEqual(result.quota_remaining, 497)
        self.assertEqual(result.quota_last_cost, 1)
        self.assertEqual(result.code_version, "abc123")
        archived_path = self.root / result.raw_response_path
        self.assertEqual(load_raw_archive(archived_path), fetch_result.raw_content)

        journal = get_odds_ingestion_run(result.run_id, self.database_path)
        self.assertEqual(journal["status"], "success")
        self.assertEqual(journal["events_received"], 1)
        self.assertEqual(journal["events_matched"], 1)
        self.assertEqual(journal["bookmaker_quotes_saved"], 2)
        self.assertEqual(journal["response_sha256"], result.response_sha256)
        self.assertEqual(journal["raw_response_path"], result.raw_response_path)
        self.assertEqual(journal["region"], "fr")
        self.assertEqual(list_applied_migrations(self.database_path), [3, 4])
        with get_connection(self.database_path) as connection:
            odds_migrations = [
                tuple(row)
                for row in connection.execute(
                    """
                    SELECT version, name, length(checksum)
                    FROM odds_schema_migrations
                    ORDER BY version
                    """
                ).fetchall()
            ]
        self.assertEqual(
            odds_migrations,
            [
                (1, "audited_moneyline_odds", 64),
                (2, "allow_french_bookmaker_region", 64),
            ],
        )

        quotes = list_moneyline_odds_for_date(
            self.target_date,
            self.database_path,
        )
        self.assertEqual(
            [
                (
                    quote["bookmaker_key"],
                    quote["away_decimal_odds"],
                    quote["home_decimal_odds"],
                )
                for quote in quotes
            ],
            [
                ("betclic", "2.10", "1.75"),
                ("unibet", "2.15", "1.72"),
            ],
        )

    def test_event_without_local_game_is_audited_without_quotes(self) -> None:
        """Une affiche inconnue reste visible mais n’alimente aucune cote locale."""
        event = self._event(
            away_name="Seattle Mariners",
            home_name="Texas Rangers",
        )

        result = self._run(self._fetch_result(event))

        self.assertEqual(result.events_matched, 0)
        self.assertEqual(result.unmatched_events, 1)
        self.assertEqual(result.bookmaker_quotes_saved, 0)
        with get_connection(self.database_path) as connection:
            row = connection.execute(
                "SELECT match_status, matched_game_id FROM odds_events"
            ).fetchone()
        self.assertEqual(row["match_status"], "NO_LOCAL_GAME")
        self.assertIsNone(row["matched_game_id"])

    def test_start_time_mismatch_is_never_forced(self) -> None:
        """Une affiche trop éloignée dans le temps n’est pas rapprochée."""
        event = self._event(
            start=datetime(2026, 9, 13, 18, 0, tzinfo=UTC)
        )

        self._run(self._fetch_result(event))

        with get_connection(self.database_path) as connection:
            status = connection.execute(
                "SELECT match_status FROM odds_events"
            ).fetchone()[0]
        self.assertEqual(status, "START_TIME_MISMATCH")

    def test_equal_doubleheader_candidates_are_marked_ambiguous(self) -> None:
        """Une égalité entre deux matchs locaux ne choisit jamais au hasard."""
        self._save_game(
            game_id=9002,
            start=datetime(2026, 9, 13, 17, 20, tzinfo=UTC),
        )
        event = self._event(
            start=datetime(2026, 9, 13, 17, 15, tzinfo=UTC)
        )

        result = self._run(self._fetch_result(event))

        self.assertEqual(result.events_matched, 0)
        with get_connection(self.database_path) as connection:
            row = connection.execute(
                "SELECT match_status, matched_game_id FROM odds_events"
            ).fetchone()
        self.assertEqual(row["match_status"], "AMBIGUOUS_LOCAL_GAME")
        self.assertIsNone(row["matched_game_id"])

    def test_two_provider_events_cannot_share_one_local_game(self) -> None:
        """Le rapprochement reste strictement un événement pour un match."""
        first = self._event(event_id="event-1")
        second = self._event(event_id="event-2")

        result = self._run(self._fetch_result(first, second))

        self.assertEqual(result.events_received, 2)
        self.assertEqual(result.events_matched, 0)
        self.assertEqual(result.bookmaker_quotes_saved, 0)
        with get_connection(self.database_path) as connection:
            statuses = connection.execute(
                "SELECT match_status FROM odds_events ORDER BY odds_event_id"
            ).fetchall()
        self.assertEqual(
            [row[0] for row in statuses],
            ["AMBIGUOUS_LOCAL_GAME", "AMBIGUOUS_LOCAL_GAME"],
        )

    def test_api_failure_closes_journal_without_archive_or_quotes(self) -> None:
        """Une clé absente produit un échec audité sans données partielles."""
        with mock.patch(
            "src.odds_ingestion_service.fetch_mlb_moneyline_odds",
            side_effect=OddsAPIConfigurationError(
                "La clé THE_ODDS_API_KEY n’est pas configurée."
            ),
        ):
            with self.assertRaises(OddsAPIConfigurationError):
                run_odds_ingestion(
                    target_date=self.target_date,
                    database_path=self.database_path,
                    data_directory=self.data_directory,
                    code_version="abc123",
                )

        with get_connection(self.database_path) as connection:
            journal = connection.execute(
                "SELECT * FROM odds_ingestion_runs"
            ).fetchone()
            quote_count = connection.execute(
                "SELECT COUNT(*) FROM moneyline_odds"
            ).fetchone()[0]
        self.assertEqual(journal["status"], "error")
        self.assertIn("OddsAPIConfigurationError", journal["error_message"])
        self.assertNotIn("secret-value", journal["error_message"])
        self.assertEqual(quote_count, 0)
        self.assertEqual(list((self.data_directory / "raw").glob("**/*")), [])

    def test_tampered_response_hash_is_rejected_before_archiving(self) -> None:
        """Une empreinte incohérente ferme le journal avant publication raw."""
        fetch_result = self._fetch_result(
            self._event(),
            declared_sha256="0" * 64,
        )

        with mock.patch(
            "src.odds_ingestion_service.fetch_mlb_moneyline_odds",
            return_value=fetch_result,
        ):
            with self.assertRaisesRegex(OddsIngestionError, "empreinte"):
                run_odds_ingestion(
                    target_date=self.target_date,
                    database_path=self.database_path,
                    data_directory=self.data_directory,
                    code_version="abc123",
                )

        with get_connection(self.database_path) as connection:
            journal = connection.execute(
                "SELECT status, error_message FROM odds_ingestion_runs"
            ).fetchone()
        self.assertEqual(journal["status"], "error")
        self.assertIn("OddsIngestionError", journal["error_message"])
        self.assertFalse((self.data_directory / "raw").exists())

    def test_tampered_odds_migration_is_rejected(self) -> None:
        """La migration propre aux cotes est contrôlée à chaque ouverture."""
        initialize_odds_storage(self.database_path)
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                UPDATE odds_schema_migrations
                SET checksum = ?
                WHERE version = 1
                """,
                ("0" * 64,),
            )

        with self.assertRaisesRegex(OddsRepositoryError, "modifiée"):
            initialize_odds_storage(self.database_path)

    def test_french_region_migration_preserves_existing_eu_collection(self) -> None:
        """Le passage à la France conserve les anciennes preuves européennes."""
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE odds_schema_migrations (
                    version INTEGER PRIMARY KEY CHECK (version > 0),
                    name TEXT NOT NULL UNIQUE,
                    checksum TEXT NOT NULL,
                    applied_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            for statement in odds_repository.ODDS_STORAGE_BASELINE_STATEMENTS:
                connection.execute(statement)
            connection.execute(
                """
                INSERT INTO odds_schema_migrations (version, name, checksum)
                VALUES (?, ?, ?)
                """,
                (
                    odds_repository.ODDS_STORAGE_BASELINE_VERSION,
                    odds_repository.ODDS_STORAGE_BASELINE_NAME,
                    odds_repository.ODDS_STORAGE_BASELINE_CHECKSUM,
                ),
            )
            connection.execute(
                """
                INSERT INTO odds_ingestion_runs (
                    run_id, provider, target_official_date, sport_key,
                    region, market, odds_format, completed_at_utc, status,
                    events_received, events_matched, bookmaker_quotes_saved
                ) VALUES (
                    1, 'the_odds_api_v4', '2026-09-13', 'baseball_mlb',
                    'eu', 'h2h', 'decimal', '2026-09-13T10:00:00+00:00',
                    'success', 1, 1, 1
                )
                """
            )
            connection.execute(
                """
                INSERT INTO odds_events (
                    odds_event_id, run_id, provider_event_id,
                    commence_time_utc, away_team_name, home_team_name,
                    matched_game_id, match_status
                ) VALUES (
                    10, 1, 'old-event', '2026-09-13T17:10:00+00:00',
                    'New York Mets', 'Chicago Cubs', 9001, 'MATCHED'
                )
                """
            )
            connection.execute(
                """
                INSERT INTO moneyline_odds (
                    odds_quote_id, odds_event_id, run_id, game_id,
                    bookmaker_key, bookmaker_title,
                    bookmaker_last_update_utc, observed_at_utc,
                    away_decimal_odds, home_decimal_odds
                ) VALUES (
                    20, 10, 1, 9001, 'legacy_eu', 'Ancien bookmaker UE',
                    '2026-09-13T09:59:00+00:00',
                    '2026-09-13T10:00:00+00:00', '2.10', '1.75'
                )
                """
            )

        initialize_odds_storage(self.database_path)

        with get_connection(self.database_path) as connection:
            old_run = connection.execute(
                "SELECT region, status FROM odds_ingestion_runs WHERE run_id = 1"
            ).fetchone()
            old_quote = connection.execute(
                "SELECT bookmaker_title FROM moneyline_odds WHERE run_id = 1"
            ).fetchone()
            migration_versions = [
                row[0]
                for row in connection.execute(
                    "SELECT version FROM odds_schema_migrations ORDER BY version"
                ).fetchall()
            ]
            foreign_key_errors = connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()

        self.assertEqual(tuple(old_run), ("eu", "success"))
        self.assertEqual(old_quote[0], "Ancien bookmaker UE")
        self.assertEqual(migration_versions, [1, 2])
        self.assertEqual(foreign_key_errors, [])


if __name__ == "__main__":
    unittest.main()
