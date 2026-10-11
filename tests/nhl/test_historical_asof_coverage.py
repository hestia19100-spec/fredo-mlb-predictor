"""Tests for the retrospective NHL source-time audit."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from types import SimpleNamespace
import unittest

from src.nhl.historical_asof_coverage import (
    NHLHistoricalAsOfCoverageError,
    classify_frozen_source_timing,
)
from src.nhl.moneypuck_five_season import (
    FiveSeasonHistory, REFERENCE_SEASONS, SeasonCoverage,
)

OBSERVED = datetime(2026, 10, 7, 13, 3, tzinfo=timezone.utc)


def history() -> FiveSeasonHistory:
    rows = tuple(
        SimpleNamespace(
            game_id=season * 1_000_000 + game_no,
            season=season,
            game_date=date(season + 1, 4, 1),
        )
        for season in REFERENCE_SEASONS
        for game_no in range(1, 1313)
    )
    return FiveSeasonHistory(
        observed_at_utc=OBSERVED,
        file_sha256="a" * 64,
        source_page="test",
        attribution="test",
        regular_rows=rows,
        playoff_rows=(),
        coverage=tuple(SeasonCoverage(season, 1312, 0)
                       for season in REFERENCE_SEASONS),
    )


class HistoricalAsOfCoverageTests(unittest.TestCase):
    def audit(self, source: FiveSeasonHistory):
        return classify_frozen_source_timing(
            source, source_observed_at_utc=OBSERVED,
            frozen_audit_sha256="b" * 64,
        )

    def test_all_five_seasons_are_postgame_not_pregame_proof(self) -> None:
        result = self.audit(history())
        self.assertEqual(result.historical_games, 6560)
        self.assertEqual(result.source_received_after_game, 6560)
        self.assertEqual(result.source_received_on_game_date, 0)
        self.assertEqual(len(result.seasons), 5)
        self.assertEqual(len(result.audit_sha256), 64)
        self.assertFalse(result.historical_pregame_availability_proven)
        self.assertFalse(result.training_permitted)
        self.assertFalse(result.prediction_publication_permitted)

    def test_same_date_is_unproven_not_accepted(self) -> None:
        source = history()
        rows = list(source.regular_rows)
        rows[-1] = SimpleNamespace(
            game_id=rows[-1].game_id, season=2025,
            game_date=OBSERVED.date(),
        )
        result = self.audit(replace(source, regular_rows=tuple(rows)))
        self.assertEqual(result.source_received_after_game, 6559)
        self.assertEqual(result.source_received_on_game_date, 1)
        self.assertFalse(result.training_permitted)

    def test_future_game_in_received_final_csv_fails_closed(self) -> None:
        source = history()
        rows = list(source.regular_rows)
        rows[-1] = SimpleNamespace(
            game_id=rows[-1].game_id, season=2025,
            game_date=date(2026, 10, 8),
        )
        with self.assertRaisesRegex(
            NHLHistoricalAsOfCoverageError, "non encore joué"
        ):
            self.audit(replace(source, regular_rows=tuple(rows)))

    def test_conflicting_dates_for_one_game_fail_closed(self) -> None:
        source = history()
        first = source.regular_rows[0]
        duplicate = SimpleNamespace(
            game_id=first.game_id, season=first.season,
            game_date=date(2022, 4, 2),
        )
        with self.assertRaisesRegex(
            NHLHistoricalAsOfCoverageError, "divergentes"
        ):
            self.audit(replace(
                source, regular_rows=source.regular_rows + (duplicate,)
            ))


if __name__ == "__main__":
    unittest.main()
