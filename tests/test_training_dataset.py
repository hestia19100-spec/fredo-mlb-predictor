"""Tests du premier jeu d’apprentissage MLB."""

from __future__ import annotations

from contextlib import closing
from datetime import date
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.training_dataset import (
    FEATURE_COLUMNS,
    TrainingDatasetError,
    build_training_dataset,
    render_training_csv,
    write_training_dataset,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrainingDatasetTests(unittest.TestCase):
    """Vérifie surtout l’antériorité stricte des variables."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / "training.db"
        self._create_database()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_database(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.executescript(
                """
                CREATE TABLE games (
                    game_id INTEGER PRIMARY KEY,
                    season INTEGER NOT NULL,
                    official_date TEXT NOT NULL,
                    game_type TEXT NOT NULL,
                    status_code TEXT NOT NULL,
                    status_detail TEXT NOT NULL,
                    away_team_id INTEGER NOT NULL,
                    home_team_id INTEGER NOT NULL,
                    away_score INTEGER,
                    home_score INTEGER
                );

                CREATE TABLE ingestion_runs (
                    run_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL
                );

                CREATE TABLE teams (
                    team_id INTEGER PRIMARY KEY
                );

                INSERT INTO teams(team_id)
                VALUES (1), (2);

                INSERT INTO ingestion_runs(run_id, status)
                VALUES (1, 'success');
                """
            )
            connection.commit()

    def _insert_game(
        self,
        *,
        game_id: int,
        official_date: str,
        away_team_id: int = 1,
        home_team_id: int = 2,
        away_score: int | None = 5,
        home_score: int | None = 3,
        season: int | None = None,
        game_type: str = "R",
        status_code: str = "F",
        status_detail: str = "Final",
    ) -> None:
        resolved_season = season or int(official_date[:4])
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """
                INSERT INTO games(
                    game_id,
                    season,
                    official_date,
                    game_type,
                    status_code,
                    status_detail,
                    away_team_id,
                    home_team_id,
                    away_score,
                    home_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    resolved_season,
                    official_date,
                    game_type,
                    status_code,
                    status_detail,
                    away_team_id,
                    home_team_id,
                    away_score,
                    home_score,
                ),
            )
            connection.commit()

    def test_previous_date_builds_binary_target_and_team_form(self) -> None:
        """Le match de la veille nourrit uniquement le match suivant."""
        self._insert_game(game_id=1, official_date="2025-04-01")
        self._insert_game(
            game_id=2,
            official_date="2025-04-02",
            away_team_id=2,
            home_team_id=1,
            away_score=2,
            home_score=4,
        )

        dataset = build_training_dataset(
            cutoff_date=date(2025, 4, 2),
            minimum_history_games=1,
            database_path=self.database_path,
        )

        self.assertEqual(dataset.eligible_games, 2)
        self.assertEqual(dataset.skipped_insufficient_history, 1)
        self.assertEqual(len(dataset.rows), 1)
        row = dataset.rows[0]
        self.assertEqual(row.game_id, 2)
        self.assertEqual(row.feature_as_of_date, date(2025, 4, 1))
        self.assertEqual(row.away_max_source_date, date(2025, 4, 1))
        self.assertEqual(row.home_max_source_date, date(2025, 4, 1))
        self.assertEqual(row.away_games_before, 1)
        self.assertEqual(row.away_win_pct_before, 0.0)
        self.assertEqual(row.away_runs_scored_per_game_before, 3.0)
        self.assertEqual(row.away_runs_allowed_per_game_before, 5.0)
        self.assertEqual(row.home_games_before, 1)
        self.assertEqual(row.home_win_pct_before, 1.0)
        self.assertEqual(row.home_runs_scored_per_game_before, 5.0)
        self.assertEqual(row.home_runs_allowed_per_game_before, 3.0)
        self.assertEqual(row.home_win, 1)

    def test_same_day_games_share_the_same_morning_snapshot(self) -> None:
        """Le second match d’un double programme ne voit pas le premier."""
        self._insert_game(game_id=10, official_date="2025-04-01")
        self._insert_game(
            game_id=11,
            official_date="2025-04-02",
            away_score=1,
            home_score=4,
        )
        self._insert_game(
            game_id=12,
            official_date="2025-04-02",
            away_score=8,
            home_score=2,
        )
        self._insert_game(
            game_id=13,
            official_date="2025-04-03",
            away_score=2,
            home_score=3,
        )

        dataset = build_training_dataset(
            cutoff_date=date(2025, 4, 3),
            minimum_history_games=1,
            database_path=self.database_path,
        )

        rows = {row.game_id: row for row in dataset.rows}
        self.assertEqual(rows[11].away_games_before, 1)
        self.assertEqual(rows[12].away_games_before, 1)
        self.assertEqual(
            rows[11].away_win_pct_before,
            rows[12].away_win_pct_before,
        )
        self.assertEqual(
            rows[11].home_runs_scored_per_game_before,
            rows[12].home_runs_scored_per_game_before,
        )
        self.assertEqual(rows[13].away_games_before, 3)
        self.assertEqual(rows[13].home_games_before, 3)

    def test_new_season_resets_every_team_history(self) -> None:
        """Les cumuls 2025 ne doivent jamais alimenter la saison 2026."""
        self._insert_game(game_id=20, official_date="2025-04-01")
        self._insert_game(game_id=21, official_date="2025-04-02")
        self._insert_game(
            game_id=22,
            official_date="2026-04-01",
            away_score=1,
            home_score=9,
        )
        self._insert_game(
            game_id=23,
            official_date="2026-04-02",
            away_score=2,
            home_score=3,
        )

        dataset = build_training_dataset(
            cutoff_date=date(2026, 4, 2),
            minimum_history_games=1,
            database_path=self.database_path,
        )

        rows = {row.game_id: row for row in dataset.rows}
        self.assertEqual(rows[23].away_games_before, 1)
        self.assertEqual(rows[23].home_games_before, 1)
        self.assertEqual(rows[23].away_runs_scored_per_game_before, 1.0)
        self.assertEqual(rows[23].home_runs_scored_per_game_before, 9.0)

    def test_non_final_non_regular_and_future_games_are_excluded(self) -> None:
        """Seuls les résultats R finaux au plus tard à la coupure comptent."""
        self._insert_game(game_id=30, official_date="2025-04-01")
        self._insert_game(
            game_id=31,
            official_date="2025-04-02",
            game_type="S",
        )
        self._insert_game(
            game_id=32,
            official_date="2025-04-02",
            status_code="S",
            status_detail="Scheduled",
            away_score=None,
            home_score=None,
        )
        self._insert_game(game_id=33, official_date="2025-04-03")
        self._insert_game(game_id=34, official_date="2025-04-04")

        dataset = build_training_dataset(
            cutoff_date=date(2025, 4, 3),
            minimum_history_games=1,
            database_path=self.database_path,
        )

        self.assertEqual(dataset.eligible_games, 2)
        self.assertEqual([row.game_id for row in dataset.rows], [33])

    def test_completed_early_is_accepted(self) -> None:
        """Completed Early constitue bien un résultat final exploitable."""
        self._insert_game(
            game_id=40,
            official_date="2025-04-01",
            status_code="FR",
            status_detail="Completed Early",
        )
        self._insert_game(game_id=41, official_date="2025-04-02")

        dataset = build_training_dataset(
            cutoff_date=date(2025, 4, 2),
            minimum_history_games=1,
            database_path=self.database_path,
        )

        self.assertEqual(dataset.eligible_games, 2)
        self.assertEqual(len(dataset.rows), 1)

    def test_skipped_early_games_still_feed_later_history(self) -> None:
        """Une cible écartée doit rester une source pour les jours suivants."""
        self._insert_game(game_id=42, official_date="2025-04-01")
        self._insert_game(game_id=43, official_date="2025-04-02")
        self._insert_game(game_id=44, official_date="2025-04-03")

        dataset = build_training_dataset(
            cutoff_date=date(2025, 4, 3),
            minimum_history_games=2,
            database_path=self.database_path,
        )

        self.assertEqual(dataset.eligible_games, 3)
        self.assertEqual(dataset.skipped_insufficient_history, 2)
        self.assertEqual(len(dataset.rows), 1)
        self.assertEqual(dataset.rows[0].game_id, 44)
        self.assertEqual(dataset.rows[0].away_games_before, 2)
        self.assertEqual(dataset.rows[0].home_games_before, 2)

    def test_target_score_never_changes_its_own_features(self) -> None:
        """Le score cible peut changer la cible, jamais ses variables."""
        self._insert_game(game_id=45, official_date="2025-04-01")
        self._insert_game(game_id=46, official_date="2025-04-02")
        first = build_training_dataset(
            cutoff_date=date(2025, 4, 2),
            minimum_history_games=1,
            database_path=self.database_path,
        ).rows[0]

        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """
                UPDATE games
                SET away_score = 1, home_score = 7
                WHERE game_id = 46
                """
            )
            connection.commit()

        second = build_training_dataset(
            cutoff_date=date(2025, 4, 2),
            minimum_history_games=1,
            database_path=self.database_path,
        ).rows[0]

        self.assertEqual(first.home_win, 0)
        self.assertEqual(second.home_win, 1)
        for feature_name in FEATURE_COLUMNS:
            self.assertEqual(
                getattr(first, feature_name),
                getattr(second, feature_name),
            )

    def test_invalid_final_score_is_rejected(self) -> None:
        """Une égalité finale ne doit jamais devenir une cible binaire."""
        self._insert_game(
            game_id=50,
            official_date="2025-04-01",
            away_score=4,
            home_score=4,
        )

        with self.assertRaisesRegex(
            TrainingDatasetError,
            "Scores finaux invalides pour le match 50",
        ):
            build_training_dataset(
                cutoff_date=date(2025, 4, 1),
                minimum_history_games=1,
                database_path=self.database_path,
            )

    def test_build_is_deterministic_and_read_only(self) -> None:
        """Deux constructions doivent produire les mêmes octets sans écrire."""
        self._insert_game(game_id=60, official_date="2025-04-01")
        self._insert_game(game_id=61, official_date="2025-04-02")
        hash_before = _sha256(self.database_path)

        first = build_training_dataset(
            cutoff_date=date(2025, 4, 2),
            minimum_history_games=1,
            database_path=self.database_path,
        )
        second = build_training_dataset(
            cutoff_date=date(2025, 4, 2),
            minimum_history_games=1,
            database_path=self.database_path,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.csv_sha256, second.csv_sha256)
        self.assertEqual(
            render_training_csv(first.rows),
            render_training_csv(second.rows),
        )
        self.assertEqual(_sha256(self.database_path), hash_before)

    def test_csv_is_created_then_verified_without_overwrite(self) -> None:
        """Un export identique est accepté, un autre contenu est refusé."""
        self._insert_game(game_id=70, official_date="2025-04-01")
        self._insert_game(game_id=71, official_date="2025-04-02")
        dataset = build_training_dataset(
            cutoff_date=date(2025, 4, 2),
            minimum_history_games=1,
            database_path=self.database_path,
        )
        output_path = self.root / "processed" / "dataset.csv"

        created = write_training_dataset(dataset, output_path)
        verified = write_training_dataset(dataset, output_path)

        self.assertFalse(created.already_existed)
        self.assertTrue(verified.already_existed)
        self.assertEqual(created.sha256, dataset.csv_sha256)
        self.assertEqual(output_path.read_bytes(), render_training_csv(dataset.rows))

        output_path.write_text("different", encoding="utf-8")
        with self.assertRaisesRegex(
            TrainingDatasetError,
            "contenu différent",
        ):
            write_training_dataset(dataset, output_path)

    def test_active_ingestion_is_rejected(self) -> None:
        """Une collecte active doit empêcher la capture du snapshot."""
        self._insert_game(game_id=80, official_date="2025-04-01")
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                "INSERT INTO ingestion_runs(run_id, status) VALUES (2, 'started')"
            )
            connection.commit()

        with self.assertRaisesRegex(
            TrainingDatasetError,
            "collecte est encore active",
        ):
            build_training_dataset(
                cutoff_date=date(2025, 4, 1),
                minimum_history_games=1,
                database_path=self.database_path,
            )

    def test_unknown_team_is_rejected(self) -> None:
        """Un identifiant absent de teams ne doit pas créer un cumul."""
        self._insert_game(
            game_id=90,
            official_date="2025-04-01",
            away_team_id=999,
        )

        with self.assertRaisesRegex(
            TrainingDatasetError,
            "Équipe inconnue pour le match 90",
        ):
            build_training_dataset(
                cutoff_date=date(2025, 4, 1),
                minimum_history_games=1,
                database_path=self.database_path,
            )

    def test_invalid_cutoff_type_is_rejected(self) -> None:
        """L’API publique doit demander une vraie date explicite."""
        with self.assertRaisesRegex(
            ValueError,
            "cutoff_date doit être une date",
        ):
            build_training_dataset(
                cutoff_date="2025-04-01",  # type: ignore[arg-type]
                minimum_history_games=1,
                database_path=self.database_path,
            )


if __name__ == "__main__":
    unittest.main()
