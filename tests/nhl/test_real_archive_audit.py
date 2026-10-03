"""Contrôles synthétiques et fichier local pour l'audit réel NHL-15."""
from __future__ import annotations

import csv
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.nhl.moneypuck_five_season import REFERENCE_SEASONS, select_five_season_history
from src.nhl.real_archive_audit import (
    NHLRealArchiveAuditError, audit_local_reference_archive,
    audit_reference_history,
)
from tests.nhl.test_lagged_historical_features import _regular
from tests.nhl.test_moneypuck_five_season import _game, _snapshot


def _history(rows):
    return select_five_season_history(_snapshot(tuple(rows)))


class NHLRealArchiveAuditTests(unittest.TestCase):
    def test_decisive_score_and_playoffs_are_separated(self) -> None:
        history = _history(_regular(2021, 1, date(2022, 1, 2)) + _game(2021, 3))
        audit = audit_reference_history(history)
        self.assertEqual(audit.regular_games, 1)
        self.assertEqual(audit.feature_candidate_rows, 2)
        self.assertEqual(audit.mirrored_decisive_scores, 1)
        self.assertEqual(audit.seasons[0].playoff_games_excluded, 1)
        self.assertEqual(audit.tied_score_game_ids, ())
        self.assertEqual(audit.inconsistent_score_game_ids, ())
        self.assertFalse(audit.regular_coverage_complete)
        self.assertTrue(audit.prior_date_order_verified)
        self.assertFalse(audit.historical_pregame_availability_proven)
        self.assertFalse(audit.independent_final_labels_proven)
        self.assertFalse(audit.training_permitted)
        self.assertFalse(audit.prediction_publication_permitted)
        self.assertEqual(audit.audit_sha256,
                         audit_reference_history(history).audit_sha256)

    def test_tied_source_score_is_unresolved_not_a_win(self) -> None:
        rows = _regular(2021, 1, date(2022, 1, 2))
        tied = tuple(replace(row, goals_for=2, goals_against=2) for row in rows)
        audit = audit_reference_history(_history(tied))
        self.assertEqual(audit.mirrored_decisive_scores, 0)
        self.assertEqual(audit.tied_score_game_ids, (2021020001,))
        self.assertEqual(audit.seasons[0].tied_score_games, 1)
        self.assertFalse(audit.independent_final_labels_proven)

    def test_mismatched_mirror_is_reported_without_silent_label(self) -> None:
        rows = _regular(2021, 1, date(2022, 1, 2))
        broken = tuple(replace(row, goals_against=9)
                       if row.team == "BOS" and row.situation == "all"
                       else row for row in rows)
        audit = audit_reference_history(_history(broken))
        self.assertEqual(audit.inconsistent_score_game_ids, (2021020001,))
        self.assertEqual(audit.mirrored_decisive_scores, 0)
        self.assertEqual(audit.seasons[0].inconsistent_score_games, 1)

    def test_same_day_games_never_count_as_prior(self) -> None:
        first_day = date(2022, 1, 2)
        rows = (_regular(2021, 1, first_day)
                + _regular(2021, 2, first_day)
                + _regular(2021, 3, date(2022, 1, 3)))
        audit = audit_reference_history(_history(rows), lookback_games=1)
        self.assertEqual(audit.regular_games, 3)
        self.assertEqual(audit.full_lookback_games, 1)

    def test_digest_changes_with_score_even_if_claimed_file_hash_does_not(self) -> None:
        rows = _regular(2021, 1, date(2022, 1, 2))
        changed = tuple(replace(
            row, goals_for=1 if row.team == "BOS" else 2,
            goals_against=2 if row.team == "BOS" else 1,
        ) for row in rows)
        self.assertNotEqual(audit_reference_history(_history(rows)).audit_sha256,
                            audit_reference_history(_history(changed)).audit_sha256)

    def test_missing_provenance_or_premature_observation_fails(self) -> None:
        history = _history(_regular(2021, 1, date(2022, 1, 2)))
        with self.assertRaises(NHLRealArchiveAuditError):
            audit_reference_history(replace(history, file_sha256="bad"))
        with self.assertRaises(NHLRealArchiveAuditError):
            audit_reference_history(replace(history, training_permitted=True))
        with self.assertRaises(NHLRealArchiveAuditError):
            audit_reference_history(replace(history, observed_at_utc=datetime(
                2022, 1, 2, tzinfo=timezone.utc)))
        with self.assertRaises(NHLRealArchiveAuditError):
            audit_reference_history(_history(_game(2021, 3)))

    def test_local_csv_filters_all_other_seasons(self) -> None:
        columns = [
            "team", "season", "name", "gameId", "playerTeam",
            "opposingTeam", "home_or_away", "gameDate", "position",
            "situation", "xGoalsFor", "xGoalsAgainst", "goalsFor",
            "goalsAgainst", "shotsOnGoalFor", "shotsOnGoalAgainst",
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "all_teams.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for season in (2011,) + REFERENCE_SEASONS:
                    for row in _game(season, 2):
                        writer.writerow({
                            "team": row.team, "season": season, "name": row.team,
                            "gameId": row.game_id, "playerTeam": row.team,
                            "opposingTeam": row.team if season == 2011 else row.opponent,
                            "home_or_away": row.home_or_away,
                            "gameDate": row.game_date.strftime("%Y%m%d"),
                            "position": "Team Level", "situation": row.situation,
                            "xGoalsFor": row.x_goals_for,
                            "xGoalsAgainst": row.x_goals_against,
                            "goalsFor": row.goals_for,
                            "goalsAgainst": row.goals_against,
                            "shotsOnGoalFor": row.shots_on_goal_for,
                            "shotsOnGoalAgainst": row.shots_on_goal_against,
                        })
            audit = audit_local_reference_archive(path)
        self.assertEqual(audit.regular_games, 5)
        self.assertEqual(len(audit.seasons), 5)
        self.assertEqual([item.regular_games for item in audit.seasons], [1] * 5)
        self.assertFalse(audit.training_permitted)


if __name__ == "__main__":
    unittest.main()
