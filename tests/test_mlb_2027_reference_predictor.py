"""Tests du calcul candidat MLB 2027, sans déclencher de publication."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.mlb_2027_reference_predictor import (
    MLB2027PredictionError,
    calculate_reference_probabilities,
)
from src.mlb_2027_reference_readiness import (
    PriorFinalGame,
    UpcomingGame,
    REFERENCE_ARTIFACT_SHA256,
)
from src.training_dataset import FEATURE_COLUMNS


TARGET = date(2027, 4, 20)
RECEIPT = "a" * 64


def _upcoming(away: int = 2, home: int = 1) -> tuple[UpcomingGame, ...]:
    return (UpcomingGame(
        game_id=500, official_date=TARGET, game_type="R",
        away_team_id=away, home_team_id=home,
        scheduled_start_utc=datetime(2027, 4, 20, 23, tzinfo=timezone.utc),
        status="SCHEDULED",
    ),)


def _history(count: int = 10) -> tuple[PriorFinalGame, ...]:
    rows = []
    for index in range(count):
        when = date(2027, 4, index + 1)
        rows.append(PriorFinalGame(
            game_id=100 + index, official_date=when, game_type="R",
            away_team_id=1, home_team_id=3,
            away_score=5 if index < 6 else 2, home_score=3,
            status="FINAL",
        ))
        rows.append(PriorFinalGame(
            game_id=200 + index, official_date=when, game_type="R",
            away_team_id=2, home_team_id=4,
            away_score=1, home_score=2, status="FINAL",
        ))
    return tuple(rows)


class FakeClassifier:
    classes_ = (0, 1)

    def __init__(self, values: object = ((0.3, 0.7),)) -> None:
        self.values = values
        self.seen_matrix = None

    def predict_proba(self, matrix: np.ndarray) -> object:
        self.seen_matrix = matrix.copy()
        return self.values


class ReferencePredictorTests(unittest.TestCase):
    def test_abstains_without_loading_model(self) -> None:
        with patch(
            "src.mlb_2027_reference_predictor.load_calibrated_artifact"
        ) as load:
            result = calculate_reference_probabilities(
                TARGET, _upcoming(), _history(9),
                source_receipt_sha256=RECEIPT,
            )
        self.assertEqual(result.probabilities, ())
        self.assertEqual(result.abstained_count, 1)
        load.assert_not_called()

    def test_same_season_j_minus_one_features_and_probability(self) -> None:
        classifier = FakeClassifier()
        artifact = SimpleNamespace(
            calibrated_model_version="logistic_team_form_v1_platt",
            feature_columns=FEATURE_COLUMNS,
            calibrated_classifier=classifier,
        )
        with patch(
            "src.mlb_2027_reference_predictor.load_calibrated_artifact",
            return_value=artifact,
        ) as load:
            result = calculate_reference_probabilities(
                TARGET, _upcoming(), _history(),
                source_receipt_sha256=RECEIPT,
            )
        load.assert_called_once()
        self.assertEqual(
            load.call_args.kwargs["expected_sha256"],
            REFERENCE_ARTIFACT_SHA256,
        )
        self.assertEqual(result.abstained_count, 0)
        prediction = result.probabilities[0]
        self.assertEqual(
            prediction.feature_values,
            (10.0, 0.0, 1.0, 2.0, 10.0, 0.6, 3.8, 3.0),
        )
        self.assertEqual(classifier.seen_matrix.shape, (1, 8))
        self.assertEqual(prediction.p_home_win, 0.7)
        self.assertEqual(prediction.p_away_win, 0.3)

    def test_invalid_probability_fails_closed(self) -> None:
        artifact = SimpleNamespace(
            calibrated_model_version="logistic_team_form_v1_platt",
            feature_columns=FEATURE_COLUMNS,
            calibrated_classifier=FakeClassifier(((float("nan"), 0.5),)),
        )
        with patch(
            "src.mlb_2027_reference_predictor.load_calibrated_artifact",
            return_value=artifact,
        ):
            with self.assertRaises(MLB2027PredictionError):
                calculate_reference_probabilities(
                    TARGET, _upcoming(), _history(),
                    source_receipt_sha256=RECEIPT,
                )

    def test_verified_real_artifact_runs_without_fitting(self) -> None:
        result = calculate_reference_probabilities(
            TARGET, _upcoming(), _history(),
            source_receipt_sha256=RECEIPT,
        )
        prediction = result.probabilities[0]
        self.assertGreaterEqual(prediction.p_home_win, 0.0)
        self.assertLessEqual(prediction.p_home_win, 1.0)
        self.assertAlmostEqual(
            prediction.p_away_win + prediction.p_home_win, 1.0,
        )

    def test_rejects_target_day_results(self) -> None:
        bad = PriorFinalGame(
            game_id=999, official_date=TARGET, game_type="R",
            away_team_id=1, home_team_id=3, away_score=4, home_score=3,
            status="FINAL",
        )
        with self.assertRaises(ValueError):
            calculate_reference_probabilities(
                TARGET, _upcoming(), _history() + (bad,),
                source_receipt_sha256=RECEIPT,
            )


if __name__ == "__main__":
    unittest.main()
