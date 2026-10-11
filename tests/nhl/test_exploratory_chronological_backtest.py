"""Safety tests for the research-only NHL chronological experiment."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.nhl.exploratory_chronological_backtest import (
    ExploratoryExample, NHLExploratoryBacktestError,
    build_exploratory_examples, evaluate_exploratory_examples,
)
from src.nhl.moneypuck_five_season import FiveSeasonHistory

SOURCE = "a" * 64


def examples() -> tuple[ExploratoryExample, ...]:
    items = []
    for season in range(2021, 2026):
        for offset in range(8):
            sign = 1 if offset % 2 else -1
            items.append(ExploratoryExample(
                game_id=season * 1_000_000 + offset + 1,
                season=season,
                features=(float(sign), float(offset % 3), 0.2 * sign,
                          0.1 * offset, 0.3 * sign, -0.1 * offset),
                home_won=int(sign > 0),
                label_source="SOURCE_DECISIVE",
            ))
    return tuple(items)


class ExploratoryEvaluationTests(unittest.TestCase):
    def evaluate(self, items):
        return evaluate_exploratory_examples(
            items, source_capture_sha256=SOURCE,
            historical_games=len(items) + 10, official_tie_labels=0,
        )

    def test_fixed_chronological_splits_and_no_publication(self) -> None:
        result = self.evaluate(examples())
        self.assertEqual(tuple((item.role, item.games) for item in result.splits),
                         (("TRAIN", 24), ("VALIDATION", 8), ("HOLDOUT", 8)))
        self.assertEqual(result.examples, 40)
        self.assertTrue(result.retrospective_only)
        self.assertFalse(result.historical_asof_proven)
        self.assertFalse(result.production_training_permitted)
        self.assertFalse(result.prediction_publication_permitted)
        self.assertEqual(len(result.audit_sha256), 64)

    def test_validation_and_holdout_labels_never_change_fitted_predictions(self) -> None:
        original = examples()
        first = self.evaluate(original)
        changed = tuple(
            replace(item, home_won=1 - item.home_won)
            if item.season >= 2024 else item for item in original
        )
        second = self.evaluate(changed)
        self.assertEqual(
            [item.prediction_sha256 for item in first.splits],
            [item.prediction_sha256 for item in second.splits],
        )
        self.assertEqual(first.splits[0], second.splits[0])
        self.assertNotEqual(first.splits[2].model_brier,
                            second.splits[2].model_brier)

    def test_duplicate_game_and_invalid_label_fail_closed(self) -> None:
        original = examples()
        with self.assertRaisesRegex(NHLExploratoryBacktestError, "dupliqué"):
            self.evaluate(original + (original[0],))
        with self.assertRaisesRegex(NHLExploratoryBacktestError, "invalide"):
            self.evaluate((replace(original[0], home_won=2),) + original[1:])


class ExploratoryLabelJoinTests(unittest.TestCase):
    def setUp(self) -> None:
        self.day = date(2022, 1, 5)
        self.history = FiveSeasonHistory(
            observed_at_utc=datetime(2026, 10, 7, tzinfo=timezone.utc),
            file_sha256=SOURCE, source_page="test", attribution="test",
            regular_rows=(
                self.row(2021020001, "HOME", "AAA", "BBB", 2, 2),
                self.row(2021020001, "AWAY", "BBB", "AAA", 2, 2),
                self.row(2021020002, "HOME", "CCC", "DDD", 1, 3),
                self.row(2021020002, "AWAY", "DDD", "CCC", 3, 1),
            ),
            playoff_rows=(), coverage=(),
        )
        self.forms = (
            self.form(2021020001, "HOME", "AAA"),
            self.form(2021020001, "AWAY", "BBB"),
            self.form(2021020002, "HOME", "CCC"),
            self.form(2021020002, "AWAY", "DDD"),
        )

    def row(self, game_id, side, team, opponent, goals_for, goals_against):
        return SimpleNamespace(
            game_id=game_id, situation="all", home_or_away=side,
            team=team, opponent=opponent, season=2021,
            game_date=self.day, goals_for=goals_for,
            goals_against=goals_against,
        )

    def form(self, game_id, side, team):
        return SimpleNamespace(
            game_id=game_id, home_or_away=side, team=team,
            season=2021, game_date=self.day, sample_games=5,
            last_prior_date=date(2022, 1, 4),
            goals_for_mean=Decimal("2.0"),
            goals_against_mean=Decimal("2.0"),
            x_goals_for_mean=Decimal("2.1"),
            x_goals_against_mean=Decimal("2.2"),
            five_on_five_x_goals_for_mean=Decimal("1.8"),
            five_on_five_x_goals_against_mean=Decimal("1.9"),
        )

    def build(self, winners):
        with patch(
            "src.nhl.exploratory_chronological_backtest.build_lagged_regular_features",
            return_value=SimpleNamespace(rows=self.forms),
        ):
            return build_exploratory_examples(self.history, winners)

    def test_official_winner_resolves_tie_and_decisive_score_stays_separate(self):
        joined = self.build({2021020001: "AAA"})
        self.assertEqual(
            [(item.home_won, item.label_source) for item in joined],
            [(1, "OFFICIAL_TIE"), (0, "SOURCE_DECISIVE")],
        )
        self.assertEqual(joined[0].features, (0.0,) * 6)

    def test_missing_official_winner_fails_closed(self):
        with self.assertRaisesRegex(NHLExploratoryBacktestError, "sans vainqueur"):
            self.build({})

    def test_same_day_prior_form_fails_closed(self):
        bad = replace(self.forms[0], last_prior_date=self.day) if hasattr(
            self.forms[0], "__dataclass_fields__"
        ) else SimpleNamespace(**{
            **vars(self.forms[0]), "last_prior_date": self.day
        })
        self.forms = (bad,) + self.forms[1:]
        with self.assertRaisesRegex(NHLExploratoryBacktestError, "chronologie"):
            self.build({2021020001: "AAA"})


if __name__ == "__main__":
    unittest.main()
