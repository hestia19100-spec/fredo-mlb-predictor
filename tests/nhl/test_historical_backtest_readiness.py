"""NHL-35: descriptive chronological split, not a fitted backtest."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
import unittest

from src.nhl.historical_backtest_readiness import (
    NHLHistoricalBacktestReadinessError, audit_historical_backtest_readiness,
)
from src.nhl.moneypuck_five_season import select_five_season_history
from tests.nhl.test_lagged_historical_features import _regular
from tests.nhl.test_moneypuck_five_season import _snapshot


def _history(*, tie_holdout: bool = False):
    rows = ()
    for season in (2021, 2022, 2023, 2024, 2025):
        rows += _regular(season, 1, date(season + 1, 1, 2))
        second = _regular(season, 2, date(season + 1, 1, 3))
        if tie_holdout and season == 2025:
            second = tuple(replace(row, goals_for=2, goals_against=2)
                           if row.situation == "all" else row for row in second)
        rows += second
    return select_five_season_history(_snapshot(rows))


class HistoricalBacktestReadinessTests(unittest.TestCase):
    def test_chronological_roles_and_warmup_candidates(self) -> None:
        report = audit_historical_backtest_readiness(
            _history(), lookback_games=1, min_prior_games=1)
        self.assertEqual([(item.season, item.role) for item in report.seasons], [
            (2021, "TRAIN"), (2022, "TRAIN"), (2023, "TRAIN"),
            (2024, "VALIDATION"), (2025, "HOLDOUT"),
        ])
        self.assertEqual([item.provisional_binary_candidates
                          for item in report.seasons], [1] * 5)
        self.assertEqual((report.provisional_train_games,
                          report.provisional_validation_games,
                          report.provisional_holdout_games), (3, 1, 1))
        self.assertFalse(report.historical_asof_proven)
        self.assertFalse(report.independent_final_labels_proven)
        self.assertFalse(report.backtest_authorized)
        self.assertFalse(report.training_permitted)
        self.assertFalse(report.prediction_publication_permitted)
        self.assertEqual(report.audit_sha256, audit_historical_backtest_readiness(
            _history(), lookback_games=1, min_prior_games=1).audit_sha256)

    def test_tied_score_is_not_a_binary_candidate(self) -> None:
        report = audit_historical_backtest_readiness(
            _history(tie_holdout=True), lookback_games=1, min_prior_games=1)
        holdout = report.seasons[-1]
        self.assertEqual((holdout.tied_source_scores,
                          holdout.provisional_binary_candidates), (1, 0))
        self.assertEqual(report.unresolved_tied_game_ids, (2025020002,))

    def test_same_date_game_is_not_prior_form(self) -> None:
        rows = (_regular(2021, 1, date(2022, 1, 2))
                + _regular(2021, 2, date(2022, 1, 2))
                + _regular(2021, 3, date(2022, 1, 3)))
        report = audit_historical_backtest_readiness(
            select_five_season_history(_snapshot(rows)),
            lookback_games=1, min_prior_games=1)
        self.assertEqual(report.seasons[0].both_teams_have_prior_form, 1)
        self.assertEqual(report.seasons[0].provisional_binary_candidates, 1)

    def test_invalid_threshold_and_order_are_rejected(self) -> None:
        history = _history()
        for threshold in (0, 2, True):
            with self.subTest(threshold=threshold), self.assertRaises(
                    NHLHistoricalBacktestReadinessError):
                audit_historical_backtest_readiness(
                    history, lookback_games=1, min_prior_games=threshold)
        overlapping = (_regular(2021, 1, date(2023, 1, 2))
                       + _regular(2022, 1, date(2023, 1, 2)))
        with self.assertRaises(NHLHistoricalBacktestReadinessError):
            audit_historical_backtest_readiness(
                select_five_season_history(_snapshot(overlapping)),
                lookback_games=1, min_prior_games=1)

    def test_inconsistent_source_score_cannot_be_candidate(self) -> None:
        rows = _regular(2021, 1, date(2022, 1, 2))
        second = _regular(2021, 2, date(2022, 1, 3))
        broken = tuple(replace(row, goals_against=9)
                       if row.team == "BOS" and row.situation == "all"
                       else row for row in second)
        report = audit_historical_backtest_readiness(
            select_five_season_history(_snapshot(rows + broken)),
            lookback_games=1, min_prior_games=1)
        self.assertEqual(report.seasons[0].inconsistent_source_scores, 1)
        self.assertEqual(report.seasons[0].provisional_binary_candidates, 0)

    def test_capture_time_participates_in_audit_identity(self) -> None:
        history = _history()
        later = replace(history, observed_at_utc=datetime(2026, 10, 3, tzinfo=timezone.utc))
        original = audit_historical_backtest_readiness(
            history, lookback_games=1, min_prior_games=1)
        revised = audit_historical_backtest_readiness(
            later, lookback_games=1, min_prior_games=1)
        self.assertEqual(original.source_observed_at_utc, history.observed_at_utc)
        self.assertFalse(original.regular_coverage_complete)
        self.assertNotEqual(original.audit_sha256, revised.audit_sha256)
