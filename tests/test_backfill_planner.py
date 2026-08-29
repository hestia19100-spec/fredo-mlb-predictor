from datetime import date, timedelta
import unittest

from src.backfill_planner import (
    MAX_CHUNK_DAYS,
    BackfillPlanError,
    build_date_chunks,
)


class BackfillPlannerTests(unittest.TestCase):
    """Vérifie le découpage sûr des périodes historiques."""

    def test_single_day_creates_one_chunk(self) -> None:
        """Une seule journée doit produire un seul lot."""
        selected_date = date(2026, 8, 29)

        chunks = build_date_chunks(
            selected_date,
            selected_date,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_date, selected_date)
        self.assertEqual(chunks[0].end_date, selected_date)
        self.assertEqual(chunks[0].day_count, 1)

    def test_thirty_one_days_create_one_chunk(self) -> None:
        """La limite de 31 jours doit rester dans un seul lot."""
        chunks = build_date_chunks(
            date(2026, 1, 1),
            date(2026, 1, 31),
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].day_count, MAX_CHUNK_DAYS)

    def test_thirty_two_days_create_two_chunks(self) -> None:
        """Le trente-deuxième jour doit ouvrir un nouveau lot."""
        chunks = build_date_chunks(
            date(2026, 1, 1),
            date(2026, 2, 1),
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].start_date, date(2026, 1, 1))
        self.assertEqual(chunks[0].end_date, date(2026, 1, 31))
        self.assertEqual(chunks[1].start_date, date(2026, 2, 1))
        self.assertEqual(chunks[1].end_date, date(2026, 2, 1))

    def test_historical_period_has_no_gap_or_overlap(self) -> None:
        """L’historique complet doit couvrir chaque journée une fois."""
        start_date = date(2021, 1, 1)
        end_date = date(2026, 8, 29)

        chunks = build_date_chunks(start_date, end_date)

        self.assertEqual(len(chunks), 67)
        self.assertEqual(chunks[0].start_date, start_date)
        self.assertEqual(chunks[-1].end_date, end_date)
        self.assertTrue(
            all(
                chunk.day_count <= MAX_CHUNK_DAYS
                for chunk in chunks
            )
        )

        for previous_chunk, current_chunk in zip(
            chunks,
            chunks[1:],
        ):
            self.assertEqual(
                current_chunk.start_date,
                previous_chunk.end_date + timedelta(days=1),
            )

        covered_days = sum(chunk.day_count for chunk in chunks)
        expected_days = (end_date - start_date).days + 1
        self.assertEqual(covered_days, expected_days)

    def test_end_before_start_is_rejected(self) -> None:
        """Une période inversée doit être refusée."""
        with self.assertRaises(BackfillPlanError):
            build_date_chunks(
                date(2026, 8, 29),
                date(2026, 8, 28),
            )

    def test_invalid_chunk_sizes_are_rejected(self) -> None:
        """Une taille hors limites ne doit jamais être acceptée."""
        invalid_sizes = (0, 32, True, 1.5)

        for invalid_size in invalid_sizes:
            with self.subTest(invalid_size=invalid_size):
                with self.assertRaises(BackfillPlanError):
                    build_date_chunks(
                        date(2026, 8, 1),
                        date(2026, 8, 2),
                        max_days=invalid_size,
                    )


if __name__ == "__main__":
    unittest.main()