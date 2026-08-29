from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.ingestion_repository import (
    find_successful_ingestion_run,
    mark_ingestion_error,
    mark_ingestion_success,
    start_ingestion_run,
)


class IngestionResumeTests(unittest.TestCase):
    """Vérifie qu’un lot réussi peut être repris sans téléchargement."""

    SOURCE = "mlb_stats_api_schedule"
    START_DATE = date(2026, 7, 1)
    END_DATE = date(2026, 7, 31)
    GAME_TYPES = ("R",)

    def setUp(self) -> None:
        """Crée une base SQLite isolée pour chaque test."""
        self.temporary_directory = TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name) / "resume_test.db"
        )

    def tearDown(self) -> None:
        """Supprime la base temporaire."""
        self.temporary_directory.cleanup()

    def request_parameters(self) -> dict[str, object]:
        """Retourne les paramètres contrôlés du lot de test."""
        return {
            "sportId": 1,
            "startDate": self.START_DATE.isoformat(),
            "endDate": self.END_DATE.isoformat(),
            "gameTypes": "R",
            "hydrate": "probablePitcher",
        }

    def start_run(self) -> int:
        """Crée un journal correspondant au lot de test."""
        return start_ingestion_run(
            source=self.SOURCE,
            start_date=self.START_DATE,
            end_date=self.END_DATE,
            game_types=self.GAME_TYPES,
            request_parameters=self.request_parameters(),
            code_version="test-version",
            database_path=self.database_path,
        )

    def mark_run_successful(
        self,
        run_id: int,
        hash_character: str = "a",
    ) -> None:
        """Clôture un journal de test avec des données valides."""
        mark_ingestion_success(
            run_id=run_id,
            records_received=10,
            records_saved=10,
            raw_response_path=(
                f"data/raw/mlb_schedule/test-{run_id}.json.gz"
            ),
            response_sha256=hash_character * 64,
            database_path=self.database_path,
        )

    def find_matching_run(self) -> dict[str, object] | None:
        """Recherche le lot exact dans la base temporaire."""
        reversed_parameters = dict(
            reversed(tuple(self.request_parameters().items()))
        )

        return find_successful_ingestion_run(
            source=self.SOURCE,
            start_date=self.START_DATE,
            end_date=self.END_DATE,
            game_types=self.GAME_TYPES,
            request_parameters=reversed_parameters,
            database_path=self.database_path,
        )

    def test_exact_successful_run_is_found(self) -> None:
        """Une réussite strictement identique doit être retrouvée."""
        run_id = self.start_run()
        self.mark_run_successful(run_id)

        found_run = self.find_matching_run()

        self.assertIsNotNone(found_run)
        assert found_run is not None
        self.assertEqual(found_run["run_id"], run_id)
        self.assertEqual(found_run["status"], "success")
        self.assertEqual(found_run["records_received"], 10)
        self.assertEqual(found_run["records_saved"], 10)

    def test_started_and_error_runs_are_not_reusable(self) -> None:
        """Un lot incomplet ou en erreur doit être retenté."""
        run_id = self.start_run()

        self.assertIsNone(self.find_matching_run())

        mark_ingestion_error(
            run_id=run_id,
            error_message="Erreur simulée",
            database_path=self.database_path,
        )

        self.assertIsNone(self.find_matching_run())

    def test_different_request_is_not_reused(self) -> None:
        """Une différence de période, type, source ou paramètres compte."""
        run_id = self.start_run()
        self.mark_run_successful(run_id)

        different_queries = (
            {
                "source": "autre_source",
                "start_date": self.START_DATE,
                "end_date": self.END_DATE,
                "game_types": self.GAME_TYPES,
                "request_parameters": self.request_parameters(),
            },
            {
                "source": self.SOURCE,
                "start_date": date(2026, 7, 2),
                "end_date": self.END_DATE,
                "game_types": self.GAME_TYPES,
                "request_parameters": self.request_parameters(),
            },
            {
                "source": self.SOURCE,
                "start_date": self.START_DATE,
                "end_date": self.END_DATE,
                "game_types": ("P",),
                "request_parameters": self.request_parameters(),
            },
            {
                "source": self.SOURCE,
                "start_date": self.START_DATE,
                "end_date": self.END_DATE,
                "game_types": self.GAME_TYPES,
                "request_parameters": {
                    **self.request_parameters(),
                    "hydrate": "team",
                },
            },
        )

        for query in different_queries:
            with self.subTest(query=query):
                found_run = find_successful_ingestion_run(
                    **query,
                    database_path=self.database_path,
                )
                self.assertIsNone(found_run)

    def test_latest_matching_success_is_returned(self) -> None:
        """La réussite la plus récente doit servir de référence."""
        first_run_id = self.start_run()
        self.mark_run_successful(first_run_id, "a")

        second_run_id = self.start_run()
        self.mark_run_successful(second_run_id, "b")

        found_run = self.find_matching_run()

        self.assertIsNotNone(found_run)
        assert found_run is not None
        self.assertEqual(found_run["run_id"], second_run_id)
        self.assertEqual(found_run["response_sha256"], "b" * 64)


if __name__ == "__main__":
    unittest.main()