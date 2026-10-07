"""NHL-30: source growth and revisions stay observable without backdating."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.nhl.current_season_import import CurrentSeasonImport
from src.nhl.current_season_quality import (
    NHLCurrentSeasonQualityError, POLICY_PATH,
    audit_current_season_import_quality, compare_verified_current_season_imports,
)
from src.nhl.moneypuck_team_import import TeamGameRow

UTC = timezone.utc


def row(game_id: int, team: str, day: int, situation: str) -> TeamGameRow:
    return TeamGameRow(
        game_id=game_id, season=2026, game_date=date(2026, 10, day),
        team=team, opponent="BOS" if team == "OTT" else "OTT",
        home_or_away="AWAY" if team == "OTT" else "HOME",
        situation=situation, x_goals_for=Decimal("2"),
        x_goals_against=Decimal("1"), goals_for=2, goals_against=1,
        shots_on_goal_for=Decimal("30"), shots_on_goal_against=Decimal("25"),
    )


def game(game_id: int, day: int) -> tuple[TeamGameRow, ...]:
    return tuple(row(game_id, team, day, situation)
                 for team in ("OTT", "BOS") for situation in ("all", "5on5"))


class QualityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        earlier = datetime(2026, 10, 4, 7, tzinfo=UTC)
        later = datetime(2026, 10, 7, 13, tzinfo=UTC)
        self.previous = CurrentSeasonImport(
            self.root / "old", 2026, "old-capture", "a" * 64,
            earlier, earlier, earlier, game(2026020001, 2), 1,
        )
        self.latest = CurrentSeasonImport(
            self.root / "new", 2026, "new-capture", "b" * 64,
            later, later, later, game(2026020001, 2) + game(2026020002, 3), 2,
        )

    def test_growth_prior_dated_rows_and_team_coverage(self) -> None:
        report = compare_verified_current_season_imports(self.previous, self.latest)
        self.assertEqual(report["status"], "COVERAGE_EXPANDED")
        self.assertEqual(report["added_game_ids"], [2026020002])
        self.assertEqual(report["prior_dated_new_game_ids"], [2026020002])
        self.assertEqual(report["added_row_count"], 4)
        self.assertEqual(report["removed_row_count"], 0)
        self.assertEqual(report["revised_row_count"], 0)
        self.assertEqual(report["team_regular_games"], [
            {"team": "BOS", "previous": 1, "latest": 2},
            {"team": "OTT", "previous": 1, "latest": 2},
        ])
        self.assertFalse(report["source_revision_review_required"])
        self.assertFalse(report["current_season_coverage_complete_proven"])
        self.assertFalse(report["training_permitted"])
        self.assertFalse(report["prediction_publication_permitted"])
        self.assertFalse(self.root.joinpath("old").exists())

    def test_revised_row_is_reported_without_overwriting_old(self) -> None:
        new_rows = list(self.latest.regular_rows)
        new_rows[0] = replace(new_rows[0], x_goals_for=Decimal("2.4"))
        latest = replace(self.latest, regular_rows=tuple(new_rows))
        report = compare_verified_current_season_imports(self.previous, latest)
        self.assertEqual(report["status"], "SOURCE_REVISIONS_REVIEW")
        self.assertEqual(report["revised_row_count"], 1)
        self.assertEqual(report["revisions"], [{
            "game_id": 2026020001, "team": "OTT", "situation": "all",
            "changed_fields": ["x_goals_for"],
        }])
        self.assertTrue(report["source_revision_review_required"])
        self.assertEqual(self.previous.regular_rows[0].x_goals_for, Decimal("2"))

    def test_regression_and_no_change_are_distinct(self) -> None:
        missing = replace(self.latest, regular_rows=game(2026020002, 3), regular_game_count=1)
        report = compare_verified_current_season_imports(self.previous, missing)
        self.assertEqual(report["status"], "SOURCE_REGRESSION_REVIEW")
        self.assertEqual(report["removed_game_ids"], [2026020001])
        self.assertTrue(report["source_revision_review_required"])
        unchanged = replace(self.latest, regular_rows=self.previous.regular_rows,
                            regular_game_count=1)
        self.assertEqual(compare_verified_current_season_imports(
            self.previous, unchanged)["status"], "NO_CHANGE")

    def test_wrong_order_or_permissions_are_rejected(self) -> None:
        for latest in (
            replace(self.latest, season=2025),
            replace(self.latest, source_observed_at_utc=self.previous.source_observed_at_utc),
            replace(self.latest, imported_at_utc=self.previous.imported_at_utc),
            replace(self.latest, training_permitted=True),
        ):
            with self.subTest(latest=latest):
                with self.assertRaises(NHLCurrentSeasonQualityError):
                    compare_verified_current_season_imports(self.previous, latest)

    def test_public_entry_verifies_both_slots(self) -> None:
        def verified(path, **_kwargs):
            return self.previous if path == self.previous.path else self.latest
        with patch("src.nhl.current_season_quality.verify_current_season_import",
                   side_effect=verified) as check:
            report = audit_current_season_import_quality(self.previous.path, self.latest.path)
        self.assertEqual(report["added_row_count"], 4)
        self.assertEqual(check.call_count, 2)

    def test_policy_cannot_authorize_training(self) -> None:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        policy["training_permitted"] = True
        unsafe = self.root / "unsafe.json"
        unsafe.write_text(json.dumps(policy), encoding="utf-8")
        with patch("src.nhl.current_season_quality.POLICY_PATH", unsafe):
            with self.assertRaises(NHLCurrentSeasonQualityError):
                compare_verified_current_season_imports(self.previous, self.latest)


if __name__ == "__main__":
    unittest.main()
