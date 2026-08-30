"""Tests du rapport de qualité des données MLB."""

from __future__ import annotations

from contextlib import closing
from datetime import date
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.data_quality import DataQualityError, run_data_quality_audit


def _sha256(path: Path) -> str:
    """Calcule l’empreinte d’un fichier de test."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DataQualityTests(unittest.TestCase):
    """Contrôle les règles essentielles sans toucher à la vraie base."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name) / "quality.db"
        )
        self._create_database()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_database(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE teams (
                    team_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                );

                CREATE TABLE pitchers (
                    pitcher_id INTEGER PRIMARY KEY,
                    full_name TEXT NOT NULL
                );

                CREATE TABLE games (
                    game_id INTEGER PRIMARY KEY,
                    season INTEGER NOT NULL,
                    official_date TEXT NOT NULL,
                    game_datetime_utc TEXT NOT NULL,
                    game_type TEXT NOT NULL,
                    status_code TEXT NOT NULL,
                    status_detail TEXT NOT NULL,
                    away_team_id INTEGER NOT NULL,
                    home_team_id INTEGER NOT NULL,
                    away_score INTEGER,
                    home_score INTEGER,
                    venue_id INTEGER,
                    venue_name TEXT,
                    doubleheader TEXT,
                    game_number INTEGER,
                    updated_at TEXT,
                    away_probable_pitcher_id INTEGER,
                    home_probable_pitcher_id INTEGER,
                    FOREIGN KEY (away_team_id)
                        REFERENCES teams(team_id),
                    FOREIGN KEY (home_team_id)
                        REFERENCES teams(team_id),
                    FOREIGN KEY (away_probable_pitcher_id)
                        REFERENCES pitchers(pitcher_id),
                    FOREIGN KEY (home_probable_pitcher_id)
                        REFERENCES pitchers(pitcher_id)
                );

                CREATE TABLE ingestion_runs (
                    run_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL
                );

                INSERT INTO teams(team_id, name)
                VALUES (1, 'Visitors'), (2, 'Home');

                INSERT INTO pitchers(pitcher_id, full_name)
                VALUES (10, 'Away Pitcher'), (20, 'Home Pitcher');

                INSERT INTO ingestion_runs(run_id, status)
                VALUES (1, 'success'), (2, 'error');
                """
            )
            connection.commit()

    def _insert_game(
        self,
        *,
        game_id: int,
        season: int = 2025,
        official_date: str = "2025-04-01",
        game_datetime_utc: str = "2025-04-01T18:00:00Z",
        status_code: str = "F",
        status_detail: str = "Final",
        away_team_id: int = 1,
        home_team_id: int = 2,
        away_score: int | None = 4,
        home_score: int | None = 2,
        away_pitcher_id: int | None = 10,
        home_pitcher_id: int | None = 20,
    ) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                """
                INSERT INTO games(
                    game_id,
                    season,
                    official_date,
                    game_datetime_utc,
                    game_type,
                    status_code,
                    status_detail,
                    away_team_id,
                    home_team_id,
                    away_score,
                    home_score,
                    away_probable_pitcher_id,
                    home_probable_pitcher_id
                )
                VALUES (?, ?, ?, ?, 'R', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    season,
                    official_date,
                    game_datetime_utc,
                    status_code,
                    status_detail,
                    away_team_id,
                    home_team_id,
                    away_score,
                    home_score,
                    away_pitcher_id,
                    home_pitcher_id,
                ),
            )
            connection.commit()

    def test_healthy_snapshot_passes_without_being_modified(self) -> None:
        """L’audit doit accepter les données et rester en lecture seule."""
        self._insert_game(game_id=100)
        self._insert_game(
            game_id=101,
            season=2026,
            official_date="2026-09-22",
            game_datetime_utc="2026-05-23T17:35:00Z",
            status_code="DR",
            status_detail="Postponed",
            away_score=None,
            home_score=None,
            away_pitcher_id=None,
            home_pitcher_id=None,
        )
        self._insert_game(
            game_id=102,
            season=2026,
            official_date="2026-08-20",
            game_datetime_utc="2026-08-20T18:00:00Z",
            status_code="CI",
            status_detail="Cancelled",
            away_score=None,
            home_score=None,
            away_pitcher_id=None,
            home_pitcher_id=None,
        )
        hash_before = _sha256(self.database_path)

        report = run_data_quality_audit(
            cutoff_date=date(2026, 8, 29),
            database_path=self.database_path,
        )

        self.assertEqual(report.integrity_verdict, "PASS")
        self.assertEqual(report.temporal_verdict, "UNVERIFIABLE")
        self.assertEqual(report.total_games, 3)
        self.assertEqual(report.unique_game_ids, 3)
        self.assertEqual(report.target_games, 1)
        self.assertEqual(report.successful_ingestions, 1)
        self.assertEqual(report.failed_ingestions, 1)
        self.assertEqual(report.latest_successful_run_id, 1)
        self.assertEqual(
            {finding.code for finding in report.findings},
            {"postponed_games", "cancelled_games"},
        )
        self.assertEqual(_sha256(self.database_path), hash_before)

    def test_missing_pitcher_is_information_not_failure(self) -> None:
        """Un lanceur manquant ne doit pas invalider une cible valide."""
        self._insert_game(game_id=200, home_pitcher_id=None)

        report = run_data_quality_audit(
            cutoff_date=date(2026, 8, 29),
            database_path=self.database_path,
        )

        finding = next(
            item
            for item in report.findings
            if item.code == "missing_probable_pitcher"
        )
        self.assertEqual(report.integrity_verdict, "PASS")
        self.assertEqual(report.target_games, 1)
        self.assertEqual(finding.severity, "INFO")
        self.assertEqual(finding.count, 1)
        self.assertEqual(finding.sample_game_ids, (200,))

    def test_final_without_scores_fails(self) -> None:
        """Un match final sans résultat complet doit bloquer l’audit."""
        self._insert_game(
            game_id=300,
            away_score=None,
            home_score=None,
        )

        report = run_data_quality_audit(
            cutoff_date=date(2026, 8, 29),
            database_path=self.database_path,
        )

        self.assertEqual(report.integrity_verdict, "FAIL")
        self.assertEqual(report.target_games, 0)
        self.assertIn(
            "final_without_valid_scores",
            {finding.code for finding in report.findings},
        )

    def test_final_result_after_cutoff_fails(self) -> None:
        """Un résultat postérieur à la coupure doit être signalé."""
        self._insert_game(
            game_id=400,
            season=2026,
            official_date="2026-08-30",
            game_datetime_utc="2026-08-30T18:00:00Z",
        )

        report = run_data_quality_audit(
            cutoff_date=date(2026, 8, 29),
            database_path=self.database_path,
        )

        codes = {finding.code for finding in report.findings}
        self.assertEqual(report.integrity_verdict, "FAIL")
        self.assertEqual(report.target_games, 0)
        self.assertIn("final_after_cutoff", codes)
        self.assertIn("score_after_cutoff", codes)

    def test_same_team_on_both_sides_fails(self) -> None:
        """Une équipe ne peut pas être son propre adversaire."""
        self._insert_game(
            game_id=500,
            away_team_id=1,
            home_team_id=1,
        )

        report = run_data_quality_audit(
            cutoff_date=date(2026, 8, 29),
            database_path=self.database_path,
        )

        self.assertEqual(report.integrity_verdict, "FAIL")
        self.assertIn(
            "invalid_team_ids",
            {finding.code for finding in report.findings},
        )

    def test_naive_datetime_is_rejected(self) -> None:
        """Un horaire sans fuseau ne prouve pas qu’il est en UTC."""
        self._insert_game(
            game_id=600,
            game_datetime_utc="2025-04-01T18:00:00",
        )

        report = run_data_quality_audit(
            cutoff_date=date(2026, 8, 29),
            database_path=self.database_path,
        )

        self.assertEqual(report.integrity_verdict, "FAIL")
        self.assertIn(
            "invalid_game_datetime",
            {finding.code for finding in report.findings},
        )

    def test_stale_scheduled_game_before_cutoff_fails(self) -> None:
        """Un ancien match encore planifié doit être examiné."""
        self._insert_game(game_id=700)
        self._insert_game(
            game_id=701,
            status_code="S",
            status_detail="Scheduled",
            away_score=None,
            home_score=None,
            away_pitcher_id=None,
            home_pitcher_id=None,
        )

        report = run_data_quality_audit(
            cutoff_date=date(2026, 8, 29),
            database_path=self.database_path,
        )

        self.assertEqual(report.integrity_verdict, "FAIL")
        self.assertIn(
            "stale_non_final_before_cutoff",
            {finding.code for finding in report.findings},
        )

    def test_active_ingestion_fails(self) -> None:
        """Une collecte active empêche de déclarer le jeu prêt."""
        self._insert_game(game_id=800)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                "INSERT INTO ingestion_runs(run_id, status) VALUES (3, 'started')"
            )
            connection.commit()

        report = run_data_quality_audit(
            cutoff_date=date(2026, 8, 29),
            database_path=self.database_path,
        )

        self.assertEqual(report.integrity_verdict, "FAIL")
        self.assertEqual(report.active_ingestions, 1)
        self.assertIn(
            "active_ingestion",
            {finding.code for finding in report.findings},
        )

    def test_empty_games_table_fails(self) -> None:
        """Une base vide ne doit pas être déclarée prête à modéliser."""
        report = run_data_quality_audit(
            cutoff_date=date(2026, 8, 29),
            database_path=self.database_path,
        )

        codes = {finding.code for finding in report.findings}
        self.assertEqual(report.integrity_verdict, "FAIL")
        self.assertIn("empty_games", codes)

    def test_missing_required_table_is_rejected(self) -> None:
        """Un schéma incomplet doit interrompre clairement l’audit."""
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("DROP TABLE pitchers")
            connection.commit()

        with self.assertRaisesRegex(
            DataQualityError,
            "Tables obligatoires absentes : pitchers",
        ):
            run_data_quality_audit(
                cutoff_date=date(2026, 8, 29),
                database_path=self.database_path,
            )


if __name__ == "__main__":
    unittest.main()
