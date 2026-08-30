"""Verrouille le protocole opérationnel avant l'ouverture de 2025."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path, PurePosixPath
import unittest


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
EVALUATION_PROTOCOL_PATH = (
    PROJECT_DIRECTORY
    / "evaluation_protocols"
    / "logistic_team_form_v1_platt_2025.json"
)
ARTIFACT_MANIFEST_PATH = (
    PROJECT_DIRECTORY
    / "model_artifacts"
    / "logistic_team_form_v1_platt.json"
)
MODEL_PROTOCOL_PATH = (
    PROJECT_DIRECTORY
    / "model_protocols"
    / "logistic_team_form_v1.json"
)

EXPECTED_EVALUATION_PROTOCOL_SHA256 = (
    "f6dbace5d25d92c5d0ec3c9ae16962ab"
    "03e022439177342bd28d607afba7b1a9"
)
EXPECTED_ARTIFACT_MANIFEST_SHA256 = (
    "a7375d5376baa043b3365cff713cb717"
    "010ad812efee472b594fa454306c96ae"
)
EXPECTED_MODEL_PROTOCOL_SHA256 = (
    "c4cb1af750619967514d37ae3a5a47a6"
    "a04255aeaccb20c5e94533dc4d138451"
)
EXPECTED_ARTIFACT_SHA256 = (
    "e0d4d2421ba076072c7ef8b3bc97dd9"
    "a341e26c62828a0ad9ba43f30da15ff55"
)
EXPECTED_DATASET_SHA256 = (
    "2a24c1a22a919acfc4ea545f8025f59c"
    "86d15873aaf09cdbdcef66236cd0a73e"
)


def _read_json(path: Path) -> tuple[bytes, dict[str, object]]:
    """Lit un objet JSON suivi par Git."""
    content = path.read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise AssertionError(f"Objet JSON attendu dans {path}.")
    return content, payload


class EvaluationProtocolTests(unittest.TestCase):
    """Empêche tout choix méthodologique postérieur aux résultats."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluation_bytes, cls.evaluation = _read_json(
            EVALUATION_PROTOCOL_PATH
        )
        cls.manifest_bytes, cls.manifest = _read_json(
            ARTIFACT_MANIFEST_PATH
        )
        cls.model_protocol_bytes, cls.model_protocol = _read_json(
            MODEL_PROTOCOL_PATH
        )

    def test_protocol_is_exact_and_registered_before_opening(self) -> None:
        """Le protocole doit rester identique et déclarer 2025 fermé."""
        self.assertEqual(
            hashlib.sha256(self.evaluation_bytes).hexdigest(),
            EXPECTED_EVALUATION_PROTOCOL_SHA256,
        )
        self.assertEqual(
            set(self.evaluation),
            {
                "evaluation_protocol_version",
                "registered_on",
                "status",
                "sealed_test_opened_at_registration",
                "purpose",
                "inputs",
                "sealed_cohort",
                "model_use",
                "reference_baseline",
                "probability_handling",
                "metrics",
                "primary_comparison",
                "uncertainty",
                "verdict",
                "preflight_invariants",
                "opening_control",
                "outputs",
                "prohibited_claims",
                "registered_limitations",
            },
        )
        self.assertEqual(self.evaluation["evaluation_protocol_version"], 1)
        self.assertEqual(
            self.evaluation["status"],
            "REGISTERED_BEFORE_SEALED_TEST",
        )
        self.assertIs(
            self.evaluation["sealed_test_opened_at_registration"],
            False,
        )

    def test_every_input_identity_matches_its_frozen_source(self) -> None:
        """Artefact, manifeste, protocole et dataset doivent concorder."""
        inputs = self.evaluation["inputs"]
        artifact = inputs["model_artifact"]
        manifest_reference = inputs["artifact_manifest"]
        protocol_reference = inputs["model_protocol"]
        dataset = inputs["dataset"]

        self.assertEqual(
            hashlib.sha256(self.manifest_bytes).hexdigest(),
            EXPECTED_ARTIFACT_MANIFEST_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(self.model_protocol_bytes).hexdigest(),
            EXPECTED_MODEL_PROTOCOL_SHA256,
        )
        self.assertEqual(
            manifest_reference["sha256"],
            EXPECTED_ARTIFACT_MANIFEST_SHA256,
        )
        self.assertEqual(
            protocol_reference["sha256"],
            EXPECTED_MODEL_PROTOCOL_SHA256,
        )
        self.assertEqual(artifact["sha256"], EXPECTED_ARTIFACT_SHA256)
        self.assertEqual(
            artifact["sha256"],
            self.manifest["artifact"]["sha256"],
        )
        self.assertEqual(
            artifact["size_bytes"],
            self.manifest["artifact"]["size_bytes"],
        )
        self.assertEqual(
            artifact["code_version"],
            self.manifest["artifact"]["code_version"],
        )
        self.assertEqual(dataset["sha256"], EXPECTED_DATASET_SHA256)
        self.assertEqual(
            dataset["sha256"],
            self.manifest["dataset"]["sha256"],
        )
        self.assertEqual(dataset["row_count"], 13253)

    def test_sealed_cohort_is_exactly_2025_and_excludes_2026(self) -> None:
        """La sélection future ne peut contenir que les 2 276 lignes 2025."""
        cohort = self.evaluation["sealed_cohort"]
        first_date = date.fromisoformat(
            cohort["expected_first_official_date"]
        )
        last_date = date.fromisoformat(
            cohort["expected_last_official_date"]
        )

        self.assertEqual(cohort["season"], 2025)
        self.assertEqual(cohort["expected_rows"], 2276)
        self.assertEqual(cohort["target"], "home_win")
        self.assertEqual(cohort["positive_class"], 1)
        self.assertEqual(cohort["selection_rule"], "season == 2025")
        self.assertEqual(cohort["recent_season_excluded"], 2026)
        self.assertEqual(first_date, date(2025, 4, 7))
        self.assertEqual(last_date, date(2025, 9, 28))
        self.assertEqual((last_date - first_date).days + 1, 175)
        self.assertEqual(cohort["expected_calendar_days_inclusive"], 175)

    def test_model_is_prediction_only_with_fixed_features(self) -> None:
        """Aucun ajustement ou choix de variable ne reste autorisé."""
        model_use = self.evaluation["model_use"]
        self.assertEqual(
            model_use["probability_source"],
            "calibrated_classifier.predict_proba(X) for class 1",
        )
        self.assertEqual(model_use["expected_classes"], [0, 1])
        self.assertEqual(
            model_use["feature_order_source"],
            "artifact_manifest.model.feature_columns",
        )
        for field in (
            "fit_allowed",
            "partial_fit_allowed",
            "recalibration_allowed",
            "threshold_tuning_allowed",
            "feature_selection_allowed",
        ):
            self.assertIs(model_use[field], False)

    def test_reference_baseline_is_fixed_before_2024(self) -> None:
        """La constante ne doit jamais être réestimée sur le test."""
        baseline = self.evaluation["reference_baseline"]
        self.assertEqual(baseline["home_wins"], 3623)
        self.assertEqual(baseline["games"], 6815)
        self.assertEqual(baseline["probability_formula"], "3623 / 6815")
        self.assertEqual(
            baseline["training_seasons"],
            [2021, 2022, 2023],
        )
        self.assertIs(baseline["uses_2024"], False)
        self.assertIs(baseline["uses_2025"], False)
        self.assertIs(baseline["uses_2026"], False)
        self.assertIs(baseline["reestimated_inside_bootstrap"], False)

    def test_probability_and_metric_formulas_are_fully_specified(self) -> None:
        """Clipping, métriques et diagnostics doivent être reproductibles."""
        handling = self.evaluation["probability_handling"]
        metrics = self.evaluation["metrics"]
        calibration = metrics["secondary"][
            "calibration_intercept_and_slope"
        ]

        self.assertEqual(handling["required_range_inclusive"], [0.0, 1.0])
        self.assertEqual(handling["log_loss_clip_epsilon"], 1e-15)
        self.assertIs(handling["brier_uses_unclipped_probability"], True)
        self.assertEqual(handling["classification_threshold"], 0.5)
        self.assertIs(handling["threshold_tie_is_positive_class"], True)
        self.assertEqual(metrics["primary"]["name"], "LOG_LOSS")
        self.assertEqual(metrics["primary"]["logarithm"], "NATURAL")
        self.assertIs(metrics["primary"]["lower_is_better"], True)
        self.assertEqual(
            metrics["secondary"]["calibration_in_the_large"]["formula"],
            "mean(y) - mean(p)",
        )
        self.assertEqual(
            calibration["parameters"],
            {
                "C": "numpy.inf",
                "solver": "lbfgs",
                "fit_intercept": True,
                "max_iter": 10000,
                "tol": 1e-12,
            },
        )
        self.assertEqual(
            calibration["convergence_warning_policy"],
            "ABORT_EVALUATION",
        )

    def test_primary_gain_direction_and_threshold_are_unambiguous(self) -> None:
        """Une valeur positive doit toujours signifier modèle meilleur."""
        comparison = self.evaluation["primary_comparison"]
        registered_uncertainty = self.model_protocol["uncertainty"]
        registered_verdict = self.model_protocol["sealed_test_verdict"]

        self.assertEqual(
            comparison["formula"],
            "(baseline_log_loss - model_log_loss) / baseline_log_loss",
        )
        self.assertEqual(
            comparison["positive_means"],
            "CALIBRATED_MODEL_IS_BETTER",
        )
        self.assertEqual(
            comparison["clarifies_registered_comparison_text"],
            registered_uncertainty["comparison"],
        )
        self.assertEqual(
            comparison["threshold"],
            registered_verdict[
                "minimum_relative_log_loss_improvement"
            ],
        )
        self.assertEqual(comparison["threshold_operator"], ">=")
        self.assertIs(comparison["no_threshold_was_changed"], True)

    def test_bootstrap_is_paired_deterministic_and_calendar_based(self) -> None:
        """Les 5 000 tirages sont définis sans choix restant."""
        uncertainty = self.evaluation["uncertainty"]
        registered = self.model_protocol["uncertainty"]

        self.assertEqual(uncertainty["method"], registered["method"])
        self.assertEqual(uncertainty["unit"], "CALENDAR_DAY")
        self.assertIs(uncertainty["include_calendar_days_without_games"], True)
        self.assertEqual(uncertainty["expected_calendar_axis_days"], 175)
        self.assertEqual(
            uncertainty["block_length_days"],
            registered["calendar_block_days"],
        )
        self.assertIn("non-circular", uncertainty["block_start_rule"])
        self.assertIs(uncertainty["source_blocks_are_always_complete"], True)
        self.assertEqual(uncertainty["replications"], 5000)
        self.assertEqual(uncertainty["replications"], registered["replications"])
        self.assertEqual(uncertainty["random_seed"], 42)
        self.assertEqual(uncertainty["random_seed"], registered["random_seed"])
        self.assertEqual(uncertainty["expected_complete_blocks_per_replicate"], 25)
        self.assertEqual(uncertainty["expected_final_partial_block_days"], 0)
        self.assertIs(uncertainty["baseline_probability_reestimated"], False)
        self.assertEqual(uncertainty["confidence_level"], 0.95)
        self.assertEqual(uncertainty["lower_quantile"], 0.025)
        self.assertEqual(uncertainty["upper_quantile"], 0.975)
        self.assertEqual(uncertainty["quantile_method"], "linear")
        self.assertEqual(uncertainty["required_lower_bound_operator"], "> 0")
        self.assertEqual(
            uncertainty["published_point_estimate"],
            "ORIGINAL_SEALED_COHORT_NOT_BOOTSTRAP_MEAN",
        )

    def test_four_verdicts_follow_one_exhaustive_conservative_order(self) -> None:
        """Chaque combinaison valide doit produire le verdict préenregistré."""
        verdict = self.evaluation["verdict"]
        labels = self.model_protocol["sealed_test_verdict"]["labels"]
        expected_rules = [
            {
                "when": "primary_supported and probability_guardrails_pass",
                "label": labels["validated"],
            },
            {
                "when": "primary_supported and not probability_guardrails_pass",
                "label": labels["signal_but_bad_probabilities"],
            },
            {
                "when": "relative_log_loss_improvement >= 0.005 and bootstrap_lower_bound <= 0 and probability_guardrails_pass",
                "label": labels["positive_but_uncertain"],
            },
            {
                "when": "otherwise",
                "label": labels["rejected"],
            },
        ]

        self.assertEqual(verdict["decision_order"], expected_rules)
        self.assertIs(verdict["decision_order_is_exhaustive"], True)
        self.assertEqual(
            verdict["structural_or_temporal_failure"],
            "ABORT_WITHOUT_VERDICT",
        )
        self.assertEqual(
            verdict["missing_or_non_finite_metric"],
            "ABORT_WITHOUT_VERDICT",
        )
        self.assertEqual(verdict["comparison_precision"], "UNROUNDED_VALUES")

    def test_opening_requires_explicit_one_time_clean_commit(self) -> None:
        """Un aperçu ou un dépôt sale ne peut jamais ouvrir le test."""
        invariants = set(self.evaluation["preflight_invariants"])
        opening = self.evaluation["opening_control"]

        self.assertIn("REQUIRE_CLEAN_GIT_WORKTREE_BEFORE_OPENING", invariants)
        self.assertIn("REQUIRE_EVALUATION_CODE_COMMITTED_BEFORE_OPENING", invariants)
        self.assertIn("REQUIRE_NO_ROW_FROM_SEASON_2026", invariants)
        self.assertEqual(
            opening["explicit_cli_flag_required"],
            "--open-sealed-test",
        )
        self.assertIs(opening["preview_must_not_compute_2025_predictions"], True)
        self.assertIs(opening["preview_must_not_compute_2025_metrics"], True)
        self.assertIs(opening["open_only_once"], True)
        self.assertIs(opening["rerun_for_model_selection_forbidden"], True)

    def test_outputs_are_relative_exclusive_and_auditable(self) -> None:
        """Les résultats officiels ne doivent jamais être remplacés."""
        outputs = self.evaluation["outputs"]
        output_directory = PurePosixPath(outputs["directory"])

        self.assertFalse(output_directory.is_absolute())
        self.assertNotIn("..", output_directory.parts)
        self.assertEqual(
            output_directory,
            PurePosixPath(
                "evaluation_results/logistic_team_form_v1_platt_2025"
            ),
        )
        self.assertIs(outputs["directory_must_not_exist_before_opening"], True)
        self.assertEqual(outputs["write_mode"], "EXCLUSIVE_NO_OVERWRITE")
        self.assertEqual(outputs["predictions_file"], "predictions.csv")
        self.assertEqual(outputs["report_file"], "report.json")
        self.assertIs(outputs["report_must_include_input_hashes"], True)
        self.assertIs(outputs["report_must_include_evaluation_code_commit"], True)
        self.assertIs(outputs["report_must_include_predictions_sha256"], True)
        self.assertIs(outputs["results_must_be_committed_without_model_changes"], True)

    def test_limits_forbid_roi_2026_and_prospective_claims(self) -> None:
        """Un verdict statistique ne constitue jamais une preuve de pari."""
        prohibited = set(self.evaluation["prohibited_claims"])
        limitations = set(self.evaluation["registered_limitations"])

        self.assertIn(
            "NO_BETTING_PROFITABILITY_WITHOUT_TIMESTAMPED_PREMATCH_ODDS",
            prohibited,
        )
        self.assertIn("NO_ROI_WITHOUT_TIMESTAMPED_PREMATCH_ODDS", prohibited)
        self.assertIn("NO_2026_METRIC_BEFORE_THE_2025_VERDICT", prohibited)
        self.assertIn(
            "NO_PROSPECTIVE_CLAIM_WHILE_TEMPORAL_AVAILABILITY_REMAINS_UNVERIFIABLE",
            prohibited,
        )
        self.assertIn(
            "NON_CIRCULAR_BLOCKS_UNDERREPRESENT_CALENDAR_EDGES",
            limitations,
        )
        self.assertIn(
            "BOOTSTRAP_DOES_NOT_CAPTURE_MODEL_TRAINING_UNCERTAINTY",
            limitations,
        )


if __name__ == "__main__":
    unittest.main()
