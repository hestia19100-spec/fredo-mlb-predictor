"""Tests du protocole figé avant l'ouverture de la saison 2025."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from src.baseline_model import (
    MODEL_VERSION,
    RANDOM_SEED,
    SEALED_RECENT_SEASONS,
    SEALED_TEST_SEASONS,
    WALK_FORWARD_FOLDS,
    build_logistic_pipeline,
)
from src.training_dataset import (
    CALENDAR_POLICY,
    DATASET_VERSION,
    FEATURE_COLUMNS,
    TEMPORAL_VERDICT,
)


PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "model_protocols"
    / "logistic_team_form_v1.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "c4cb1af750619967514d37ae3a5a47a6"
    "a04255aeaccb20c5e94533dc4d138451"
)
EXPECTED_DATASET_SHA256 = (
    "2a24c1a22a919acfc4ea545f8025f59c"
    "86d15873aaf09cdbdcef66236cd0a73e"
)


class ModelProtocolTests(unittest.TestCase):
    """Empêche de modifier silencieusement les règles après coup."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_bytes = PROTOCOL_PATH.read_bytes()
        cls.protocol = json.loads(cls.protocol_bytes.decode("utf-8"))

    def test_protocol_file_is_frozen_before_sealed_test(self) -> None:
        """L'empreinte et le statut doivent rester ceux enregistrés."""
        self.assertEqual(
            hashlib.sha256(self.protocol_bytes).hexdigest(),
            EXPECTED_PROTOCOL_SHA256,
        )
        self.assertEqual(self.protocol["protocol_version"], 1)
        self.assertEqual(self.protocol["registered_on"], "2026-08-30")
        self.assertEqual(
            self.protocol["registered_after_commit"],
            "4f512ba",
        )
        self.assertEqual(
            self.protocol["status"],
            "REGISTERED_BEFORE_SEALED_TEST",
        )

    def test_dataset_identity_matches_the_exported_snapshot(self) -> None:
        """Version, empreinte, coupure et taille du CSV sont immuables."""
        dataset = self.protocol["dataset"]

        self.assertEqual(dataset["dataset_version"], DATASET_VERSION)
        self.assertEqual(
            dataset["path"],
            "data/processed/mlb_team_form_v1_2026-08-29.csv",
        )
        self.assertEqual(dataset["sha256"], EXPECTED_DATASET_SHA256)
        self.assertEqual(dataset["cutoff_date"], "2026-08-29")
        self.assertEqual(dataset["calendar_policy"], CALENDAR_POLICY)
        self.assertEqual(
            dataset["temporal_availability_verdict"],
            TEMPORAL_VERDICT,
        )
        self.assertEqual(dataset["target"], "home_win")
        self.assertEqual(dataset["row_count"], 13253)

    def test_feature_contract_matches_the_baseline_code(self) -> None:
        """Aucune variable ne peut être ajoutée après consultation du test."""
        self.assertEqual(
            tuple(self.protocol["features"]),
            tuple(FEATURE_COLUMNS),
        )
        self.assertEqual(len(self.protocol["features"]), 8)

    def test_estimator_parameters_match_the_baseline_code(self) -> None:
        """Le pipeline réel doit correspondre au modèle enregistré."""
        model = self.protocol["model"]
        parameters = model["parameters"]
        preprocessing = model["preprocessing"]
        pipeline = build_logistic_pipeline()
        scaler = pipeline.named_steps["standard_scaler"]
        estimator = pipeline.named_steps["logistic_regression"]

        self.assertEqual(model["model_version"], MODEL_VERSION)
        self.assertEqual(
            model["estimator"],
            "sklearn.linear_model.LogisticRegression",
        )
        self.assertEqual(parameters["C"], estimator.C)
        self.assertEqual(parameters["solver"], estimator.solver)
        self.assertEqual(parameters["max_iter"], estimator.max_iter)
        self.assertEqual(parameters["random_state"], RANDOM_SEED)
        self.assertEqual(estimator.random_state, RANDOM_SEED)
        self.assertIsNone(parameters["class_weight"])
        self.assertIsNone(estimator.class_weight)
        self.assertEqual(scaler.__class__.__name__, "StandardScaler")
        self.assertEqual(preprocessing["imputation"], "NONE")
        self.assertEqual(
            preprocessing["scaler_fit_scope"],
            "TRAINING_SEASONS_ONLY",
        )

    def test_development_folds_match_the_evaluated_code(self) -> None:
        """Les trois contrôles chronologiques ne doivent pas changer."""
        checks = self.protocol["chronology"]["development_checks"]
        actual = tuple(
            (
                tuple(check["train_seasons"]),
                check["evaluation_season"],
            )
            for check in checks
        )

        self.assertEqual(actual, WALK_FORWARD_FOLDS)
        for train_seasons, evaluation_season in actual:
            self.assertLess(max(train_seasons), evaluation_season)

    def test_final_chronology_keeps_2025_and_2026_separate(self) -> None:
        """Entraînement, calibration, test et récent ne se chevauchent pas."""
        chronology = self.protocol["chronology"]
        training = set(
            chronology["final_base_model_training_seasons"]
        )
        calibration = {chronology["calibration_season"]}
        sealed_test = {chronology["sealed_test_season"]}
        recent = {chronology["recent_retrospective_season"]}

        self.assertEqual(training, {2021, 2022, 2023})
        self.assertEqual(calibration, {2024})
        self.assertEqual(sealed_test, set(SEALED_TEST_SEASONS))
        self.assertEqual(recent, set(SEALED_RECENT_SEASONS))
        self.assertFalse(training & calibration)
        self.assertFalse(training & sealed_test)
        self.assertFalse(training & recent)
        self.assertFalse(calibration & sealed_test)
        self.assertFalse(calibration & recent)
        self.assertFalse(sealed_test & recent)
        self.assertLess(max(training), min(calibration))
        self.assertLess(max(calibration), min(sealed_test))
        self.assertLess(max(sealed_test), min(recent))

    def test_calibration_method_is_fixed_to_platt(self) -> None:
        """Isotonic ne doit pas être choisi après avoir vu 2025."""
        calibration = self.protocol["calibration"]

        self.assertEqual(calibration["method"], "PLATT_SIGMOID")
        self.assertEqual(calibration["fit_season"], 2024)
        self.assertTrue(
            calibration["selection_was_fixed_before_sealed_test"]
        )
        self.assertFalse(
            calibration["isotonic_is_allowed_for_this_version"]
        )
        self.assertFalse(
            calibration["in_sample_calibration_metrics_are_final_evidence"]
        )

    def test_metrics_and_uncertainty_are_pre_registered(self) -> None:
        """La métrique primaire et le bootstrap sont choisis avant 2025."""
        metrics = self.protocol["metrics"]
        uncertainty = self.protocol["uncertainty"]

        self.assertEqual(metrics["primary"], "LOG_LOSS")
        self.assertIn("BRIER_SCORE", metrics["secondary"])
        self.assertIn(
            "CALIBRATION_IN_THE_LARGE",
            metrics["secondary"],
        )
        self.assertIn("CALIBRATION_SLOPE", metrics["secondary"])
        self.assertTrue(metrics["accuracy_is_descriptive_only"])
        self.assertTrue(metrics["roc_auc_is_descriptive_only"])
        self.assertEqual(
            uncertainty["method"],
            "PAIRED_MOVING_CALENDAR_BLOCK_BOOTSTRAP",
        )
        self.assertEqual(uncertainty["calendar_block_days"], 7)
        self.assertEqual(uncertainty["replications"], 5000)
        self.assertEqual(uncertainty["confidence_level"], 0.95)
        self.assertEqual(uncertainty["random_seed"], RANDOM_SEED)

    def test_verdict_thresholds_and_guardrails_are_fixed(self) -> None:
        """Le verdict final ne peut pas être embelli après le test."""
        verdict = self.protocol["sealed_test_verdict"]
        guardrails = set(self.protocol["guardrails"])

        self.assertEqual(
            verdict["minimum_relative_log_loss_improvement"],
            0.005,
        )
        self.assertTrue(
            verdict["lower_confidence_bound_must_be_above_zero"]
        )
        self.assertTrue(verdict["model_brier_must_not_exceed_baseline"])
        self.assertEqual(
            verdict["maximum_absolute_calibration_in_the_large"],
            0.02,
        )
        self.assertEqual(verdict["minimum_calibration_slope"], 0.8)
        self.assertEqual(verdict["maximum_calibration_slope"], 1.2)
        self.assertEqual(verdict["labels"]["validated"], "SIGNAL_VALIDE")
        self.assertEqual(verdict["labels"]["rejected"], "REJETE_V1")
        self.assertIn(
            "OPEN_2025_ONLY_ONCE_AFTER_THE_PIPELINE_IS_FROZEN",
            guardrails,
        )
        self.assertIn(
            "DO_NOT_REPORT_BETTING_PROFITABILITY_WITHOUT_"
            "TIMESTAMPED_PREMATCH_ODDS",
            guardrails,
        )
        self.assertIn(
            "ANY_CHANGE_AFTER_OPENING_2025_CREATES_A_NEW_"
            "EXPLORATORY_MODEL_VERSION",
            guardrails,
        )


if __name__ == "__main__":
    unittest.main()
