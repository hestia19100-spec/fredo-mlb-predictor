"""Résultats NHL rétrospectifs, séparés des formes pré-match candidates."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
import unittest

from src.nhl.lagged_historical_features import build_lagged_regular_features
from src.nhl.moneypuck_five_season import select_five_season_history
from src.nhl.retrospective_labels import (
    NHLRetrospectiveLabelError, build_retrospective_regular_labels,
)
from tests.nhl.test_lagged_historical_features import _regular
from tests.nhl.test_moneypuck_five_season import _game, _snapshot


def _build(rows):
    history = select_five_season_history(_snapshot(rows))
    candidates = build_lagged_regular_features(history)
    return history, candidates


class NHLRetrospectiveLabelTests(unittest.TestCase):
    def test_regular_result_is_joined_without_training_permission(self) -> None:
        history, candidates = _build(_regular(2021, 1, date(2022, 1, 2)))
        audit = build_retrospective_regular_labels(history, candidates)
        self.assertEqual(len(audit.rows), 1)
        row = audit.rows[0]
        self.assertEqual((row.result.home_team, row.result.away_team), ("BOS", "NYR"))
        self.assertEqual((row.result.home_goals, row.result.away_goals), (2, 1))
        self.assertEqual(row.result.winner, "BOS")
        self.assertEqual(row.home_form.prior_games, 0)
        self.assertEqual(audit.games_by_season[0], (2021, 1))
        self.assertFalse(audit.regular_coverage_complete)
        self.assertFalse(audit.historical_pregame_availability_proven)
        self.assertFalse(audit.training_permitted)
        self.assertFalse(audit.prediction_publication_permitted)
        self.assertFalse(hasattr(row.home_form, "winner"))
        self.assertEqual(audit.audit_sha256, build_retrospective_regular_labels(
            history, candidates).audit_sha256)

    def test_target_score_changes_label_not_its_prior_form(self) -> None:
        first = _regular(2021, 1, date(2022, 1, 2))
        target = _regular(2021, 2, date(2022, 1, 3))
        changed = tuple(replace(
            row,
            goals_for=1 if row.team == "BOS" else 2,
            goals_against=2 if row.team == "BOS" else 1,
        ) for row in target)
        before, before_candidates = _build(first + target)
        after, after_candidates = _build(first + changed)
        self.assertEqual(before_candidates.rows, after_candidates.rows)
        first_audit = build_retrospective_regular_labels(before, before_candidates)
        second_audit = build_retrospective_regular_labels(after, after_candidates)
        self.assertEqual(first_audit.rows[1].home_form, second_audit.rows[1].home_form)
        self.assertEqual(first_audit.rows[1].result.winner, "BOS")
        self.assertEqual(second_audit.rows[1].result.winner, "NYR")
        self.assertNotEqual(first_audit.audit_sha256, second_audit.audit_sha256)

    def test_playoffs_are_not_labels_or_priors(self) -> None:
        history, candidates = _build(
            _regular(2021, 1, date(2022, 1, 2)) + _game(2021, 3)
        )
        audit = build_retrospective_regular_labels(history, candidates)
        self.assertEqual(len(audit.rows), 1)
        self.assertEqual(audit.rows[0].result.game_id, 2021020001)

    def test_tampered_form_or_provenance_is_rejected(self) -> None:
        history, candidates = _build(_regular(2021, 1, date(2022, 1, 2)))
        changed = replace(candidates.rows[0], goals_for_mean=Decimal("99"))
        with self.assertRaises(NHLRetrospectiveLabelError):
            build_retrospective_regular_labels(
                history, replace(candidates, rows=(changed,) + candidates.rows[1:]))
        with self.assertRaises(NHLRetrospectiveLabelError):
            build_retrospective_regular_labels(history, replace(
                candidates, training_permitted=True))
        with self.assertRaises(NHLRetrospectiveLabelError):
            build_retrospective_regular_labels(replace(
                history, file_sha256="1" * 64), candidates)

    def test_mirrored_score_and_no_tie_are_required(self) -> None:
        rows = _regular(2021, 1, date(2022, 1, 2))
        broken = tuple(replace(row, goals_against=8) if
                       row.team == "BOS" and row.situation == "all" else row
                       for row in rows)
        history, candidates = _build(broken)
        with self.assertRaises(NHLRetrospectiveLabelError):
            build_retrospective_regular_labels(history, candidates)
        tied = tuple(replace(row, goals_for=1, goals_against=1) if
                     row.situation == "all" else row for row in rows)
        history, candidates = _build(tied)
        with self.assertRaises(NHLRetrospectiveLabelError):
            build_retrospective_regular_labels(history, candidates)

    def test_source_must_be_observed_after_the_game(self) -> None:
        history, candidates = _build(_regular(2021, 1, date(2022, 1, 2)))
        early = replace(history, observed_at_utc=datetime(
            2022, 1, 2, tzinfo=timezone.utc))
        matching = build_lagged_regular_features(early)
        with self.assertRaises(NHLRetrospectiveLabelError):
            build_retrospective_regular_labels(early, matching)


if __name__ == "__main__":
    unittest.main()
