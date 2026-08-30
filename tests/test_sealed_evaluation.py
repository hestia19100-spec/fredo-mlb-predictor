"""Tests synthétiques du programme d'ouverture scellée 2025."""

from __future__ import annotations

import ast
import csv
from dataclasses import replace
from datetime import date, timedelta
import hashlib
import inspect
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.exceptions import ConvergenceWarning

try:
    import src.sealed_evaluation as module
except ModuleNotFoundError:  # Adaptateur du dossier local de validation.
    import sealed_evaluation as module


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
EVALUATION_PROTOCOL_PATH = (
    PROJECT_DIRECTORY
    / "evaluation_protocols"
    / "logistic_team_form_v1_platt_2025.json"
)


def _row(game_id: int, game_date: date, target: int) -> module.ModelRow:
    return module.ModelRow(
        game_id=game_id,
        season=2025,
        official_date=game_date,
        feature_as_of_date=game_date - timedelta(days=1),
        away_team_id=100 + game_id,
        home_team_id=200 + game_id,
        away_max_source_date=game_date - timedelta(days=1),
        home_max_source_date=game_date - timedelta(days=2),
        features=(10.0, 0.5, 4.0, 4.1, 11.0, 0.55, 4.2, 3.9),
        home_win=target,
    )


def _synthetic_rows() -> tuple[module.ModelRow, ...]:
    first = module.EXPECTED_FIRST_DATE
    offsets = (0, 30, 60, 90, 120, 174)
    return tuple(
        _row(index + 1, first + timedelta(days=offset), index % 2)
        for index, offset in enumerate(offsets)
    )


def _dummy_result() -> module.SealedEvaluationResult:
    rows = _synthetic_rows()
    metrics = module.ProbabilityMetrics(
        samples=6,
        observed_home_win_rate=0.5,
        mean_predicted_home_win_probability=0.5,
        log_loss=0.6,
        brier_score=0.2,
        calibration_in_the_large=0.0,
        roc_auc=0.6,
        accuracy_at_0_5=0.5,
    )
    bootstrap = module.BootstrapInterval(
        replications=50,
        block_length_days=7,
        random_seed=42,
        calendar_days=175,
        blocks_per_replicate=25,
        lower=0.001,
        upper=0.02,
        statistics_sha256="1" * 64,
    )
    verdict = module.decide_verdict(
        relative_log_loss_improvement=0.01,
        bootstrap_lower_bound=0.001,
        model_brier=0.2,
        baseline_brier=0.25,
        calibration_in_the_large=0.0,
        calibration_slope=1.0,
    )
    predictions = tuple(
        module.RowPrediction(
            row=row,
            model_probability=0.4 + index * 0.04,
            baseline_probability=module.BASELINE_PROBABILITY,
            model_log_loss=0.6,
            baseline_log_loss=0.69,
        )
        for index, row in enumerate(rows)
    )
    return module.SealedEvaluationResult(
        season=2025,
        rows=6,
        first_official_date=rows[0].official_date,
        last_official_date=rows[-1].official_date,
        baseline_probability=module.BASELINE_PROBABILITY,
        model_metrics=metrics,
        baseline_metrics=replace(metrics, log_loss=0.61, brier_score=0.21),
        calibration=module.CalibrationDiagnostics(intercept=0.0, slope=1.0),
        relative_log_loss_improvement=0.01,
        bootstrap=bootstrap,
        verdict=verdict,
        predictions=predictions,
    )


class SealedEvaluationTests(unittest.TestCase):
    """Aucun test de ce fichier ne lit le vrai CSV ou le vrai joblib."""

    def test_preview_keeps_csv_and_artifact_sealed(self) -> None:
        """L'aperçu doit réussir même si toute lecture scellée est empoisonnée."""
        with (
            patch.object(
                module,
                "_load_artifact_from_verified_bytes",
                side_effect=AssertionError("joblib interdit"),
            ) as artifact_loader,
            patch.object(
                module,
                "_load_sealed_rows_from_verified_csv",
                side_effect=AssertionError("CSV interdit"),
            ) as dataset_loader,
            patch.object(
                module.joblib,
                "load",
                side_effect=AssertionError("désérialisation interdite"),
            ) as joblib_loader,
        ):
            preview = module.preview_sealed_evaluation(
                EVALUATION_PROTOCOL_PATH,
                project_directory=PROJECT_DIRECTORY,
            )
        self.assertFalse(preview.predictions_computed)
        self.assertFalse(preview.metrics_computed)
        self.assertFalse(preview.artifact_deserialized)
        self.assertFalse(preview.dataset_parsed)
        artifact_loader.assert_not_called()
        dataset_loader.assert_not_called()
        joblib_loader.assert_not_called()

    def test_configuration_rejects_wrong_protocol_hash_before_parsing(self) -> None:
        """Un protocole altéré doit être refusé avant toute interprétation."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "protocol.json"
            path.write_text("{}", encoding="utf-8")
            with patch.object(
                module,
                "_parse_json_object",
                side_effect=AssertionError("parsing interdit"),
            ) as parser:
                with self.assertRaises(module.SealedEvaluationError):
                    module.load_evaluation_configuration(
                        path,
                        project_directory=PROJECT_DIRECTORY,
                    )
            parser.assert_not_called()

    def test_probability_metrics_match_registered_formulas(self) -> None:
        """Log loss, Brier, calibration et classification restent exacts."""
        targets = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])
        metrics = module.compute_probability_metrics(targets, probabilities)
        expected_log_loss = float(
            np.mean(
                -(
                    targets * np.log(probabilities)
                    + (1 - targets) * np.log1p(-probabilities)
                )
            )
        )
        self.assertAlmostEqual(metrics.log_loss, expected_log_loss, places=15)
        self.assertAlmostEqual(metrics.brier_score, 0.085, places=15)
        self.assertAlmostEqual(metrics.calibration_in_the_large, 0.0, places=15)
        self.assertEqual(metrics.accuracy_at_0_5, 1.0)
        self.assertEqual(metrics.roc_auc, 1.0)

    def test_log_loss_clips_zero_and_one_probabilities(self) -> None:
        """Les extrêmes restent finis uniquement pour la log loss."""
        metrics = module.compute_probability_metrics([0, 1], [0.0, 1.0])
        self.assertTrue(np.isfinite(metrics.log_loss))
        self.assertEqual(metrics.brier_score, 0.0)

    def test_probability_metrics_reject_invalid_inputs(self) -> None:
        """NaN, valeur hors intervalle et classe absente interrompent tout."""
        invalid_cases = (
            ([0, 1], [0.2, np.nan]),
            ([0, 1], [-0.1, 0.7]),
            ([1, 1], [0.4, 0.6]),
        )
        for targets, probabilities in invalid_cases:
            with self.subTest(targets=targets, probabilities=probabilities):
                with self.assertRaises(module.SealedEvaluationError):
                    module.compute_probability_metrics(targets, probabilities)

    def test_calibration_diagnostic_is_finite(self) -> None:
        """Le fit diagnostique accepte un échantillon synthétique non séparé."""
        probabilities = np.linspace(0.1, 0.9, 200)
        targets = np.asarray(
            [1 if (index * 37) % 100 < probability * 100 else 0
             for index, probability in enumerate(probabilities)]
        )
        diagnostic = module.fit_calibration_diagnostics(
            targets,
            probabilities,
        )
        self.assertTrue(np.isfinite(diagnostic.intercept))
        self.assertTrue(np.isfinite(diagnostic.slope))

    def test_calibration_convergence_warning_aborts(self) -> None:
        """Un avertissement de convergence interdit tout verdict."""
        with patch.object(
            module.LogisticRegression,
            "fit",
            side_effect=ConvergenceWarning("échec"),
        ):
            with self.assertRaises(module.SealedEvaluationError):
                module.fit_calibration_diagnostics(
                    [0, 1, 0, 1],
                    [0.2, 0.8, 0.4, 0.6],
                )

    def test_calendar_bootstrap_is_paired_and_deterministic(self) -> None:
        """Le même PCG64 doit produire le même intervalle et la même empreinte."""
        first = date(2025, 1, 1)
        dates = [first + timedelta(days=index) for index in range(14)]
        model_losses = np.linspace(0.5, 0.7, 14)
        baseline_losses = np.full(14, 0.72)
        first_run = module.paired_calendar_block_bootstrap(
            dates,
            model_losses,
            baseline_losses,
            first_date=first,
            last_date=first + timedelta(days=13),
            replications=100,
            block_length_days=7,
            random_seed=42,
        )
        second_run = module.paired_calendar_block_bootstrap(
            dates,
            model_losses,
            baseline_losses,
            first_date=first,
            last_date=first + timedelta(days=13),
            replications=100,
            block_length_days=7,
            random_seed=42,
        )
        self.assertEqual(first_run, second_run)
        self.assertEqual(first_run.blocks_per_replicate, 2)
        self.assertGreater(first_run.lower, 0.0)
        self.assertRegex(first_run.statistics_sha256, r"^[0-9a-f]{64}$")

    def test_calendar_bootstrap_rejects_non_finite_loss(self) -> None:
        """Une perte non finie doit interrompre l'incertitude."""
        with self.assertRaises(module.SealedEvaluationError):
            module.paired_calendar_block_bootstrap(
                [date(2025, 1, 1), date(2025, 1, 2)],
                [0.5, np.inf],
                [0.7, 0.7],
                first_date=date(2025, 1, 1),
                last_date=date(2025, 1, 7),
                replications=10,
            )

    def test_verdict_signal_valide(self) -> None:
        verdict = module.decide_verdict(
            relative_log_loss_improvement=0.005,
            bootstrap_lower_bound=1e-12,
            model_brier=0.25,
            baseline_brier=0.25,
            calibration_in_the_large=0.02,
            calibration_slope=0.8,
        )
        self.assertEqual(verdict.label, module.VERDICT_VALIDATED)

    def test_verdict_signal_present_but_bad_probabilities(self) -> None:
        verdict = module.decide_verdict(
            relative_log_loss_improvement=0.01,
            bootstrap_lower_bound=0.001,
            model_brier=0.3,
            baseline_brier=0.25,
            calibration_in_the_large=0.0,
            calibration_slope=1.0,
        )
        self.assertEqual(verdict.label, module.VERDICT_BAD_PROBABILITIES)

    def test_verdict_promising_but_inconclusive(self) -> None:
        verdict = module.decide_verdict(
            relative_log_loss_improvement=0.01,
            bootstrap_lower_bound=0.0,
            model_brier=0.2,
            baseline_brier=0.25,
            calibration_in_the_large=0.0,
            calibration_slope=1.2,
        )
        self.assertEqual(verdict.label, module.VERDICT_PROMISING)

    def test_verdict_rejected(self) -> None:
        verdict = module.decide_verdict(
            relative_log_loss_improvement=0.004999999,
            bootstrap_lower_bound=0.001,
            model_brier=0.2,
            baseline_brier=0.25,
            calibration_in_the_large=0.0,
            calibration_slope=1.0,
        )
        self.assertEqual(verdict.label, module.VERDICT_REJECTED)

    def test_sealed_rows_validate_exact_structure(self) -> None:
        """Une petite cohorte synthétique peut exercer toutes les invariants."""
        rows = _synthetic_rows()
        validated = module.validate_sealed_rows(rows, expected_rows=6)
        self.assertEqual(tuple(row.game_id for row in validated), (1, 2, 3, 4, 5, 6))

    def test_sealed_rows_reject_structural_or_temporal_leakage(self) -> None:
        """Doublon, mauvaise saison, fuite et NaN sont tous bloquants."""
        rows = list(_synthetic_rows())
        invalid_variants = (
            rows[:-1] + [replace(rows[-1], game_id=rows[0].game_id)],
            rows[:-1] + [replace(rows[-1], season=2026)],
            rows[:-1] + [replace(
                rows[-1],
                home_max_source_date=rows[-1].official_date,
            )],
            rows[:-1] + [replace(
                rows[-1],
                features=(np.nan, *rows[-1].features[1:]),
            )],
        )
        for invalid in invalid_variants:
            with self.subTest(invalid=invalid[-1]):
                with self.assertRaises(module.SealedEvaluationError):
                    module.validate_sealed_rows(invalid, expected_rows=6)

    def test_unsorted_rows_cannot_misalign_probabilities(self) -> None:
        """Le programme refuse de deviner comment réordonner les probabilités."""
        rows = tuple(reversed(_synthetic_rows()))
        with self.assertRaises(module.SealedEvaluationError):
            module.evaluate_sealed_rows(
                rows,
                [0.4, 0.6, 0.45, 0.55, 0.3, 0.7],
                expected_rows=6,
                bootstrap_replications=10,
            )

    def test_2026_poisoned_features_are_never_converted(self) -> None:
        """Après le SHA, seules les variables des lignes 2025 sont converties."""
        fieldnames = list(module._EXPECTED_CSV_COLUMNS)
        valid = {name: "1" for name in fieldnames}
        valid.update(
            {
                "game_id": "1",
                "season": "2025",
                "official_date": "2025-04-07",
                "feature_as_of_date": "2025-04-06",
                "away_team_id": "100",
                "home_team_id": "101",
                "away_max_source_date": "2025-04-06",
                "home_max_source_date": "2025-04-05",
                "home_win": "0",
            }
        )
        poisoned = dict(valid)
        poisoned.update(
            {
                "game_id": "2",
                "season": "2026",
                "official_date": "POISON",
                "home_win": "POISON",
            }
        )
        for feature in module.FEATURE_COLUMNS:
            poisoned[feature] = "POISON"
        destination = io.StringIO(newline="")
        writer = csv.DictWriter(destination, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(valid)
        writer.writerow(poisoned)
        with (
            patch.object(module, "EXPECTED_DATASET_ROWS", 2),
            patch.object(
                module,
                "validate_sealed_rows",
                side_effect=lambda rows: tuple(rows),
            ),
        ):
            parsed = module._load_sealed_rows_from_verified_csv(
                destination.getvalue().encode("utf-8")
            )
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].season, 2025)

    def test_opening_preflight_failure_occurs_before_sealed_read(self) -> None:
        """Une erreur Git doit survenir avant le CSV, le joblib et la réservation."""
        configuration = module.load_evaluation_configuration(
            EVALUATION_PROTOCOL_PATH,
            project_directory=PROJECT_DIRECTORY,
        )
        with (
            patch.object(
                module,
                "load_evaluation_configuration",
                return_value=configuration,
            ),
            patch.object(
                module,
                "_verify_git_opening_preconditions",
                side_effect=module.SealedEvaluationError("git sale"),
            ),
            patch.object(module, "_read_verified_bytes") as reader,
            patch.object(module, "_reserve_output_directory") as reserve,
        ):
            with self.assertRaises(module.SealedEvaluationError):
                module.open_sealed_evaluation(
                    expected_evaluation_code_commit="a" * 40,
                )
        reader.assert_not_called()
        reserve.assert_not_called()

    def test_main_module_joblib_artifact_is_loaded_safely(self) -> None:
        """Le joblib historique créé avec python -m reste relisible."""
        import __main__ as main_module

        configuration = module.load_evaluation_configuration(
            EVALUATION_PROTOCOL_PATH,
            project_directory=PROJECT_DIRECTORY,
        )
        manifest = json.loads(json.dumps(configuration.artifact_manifest))
        manifest["runtime"] = module._runtime_versions()
        configuration = replace(
            configuration,
            artifact_manifest=manifest,
        )
        artifact = module.CalibratedModelArtifact(
            artifact_format_version=1,
            calibrated_model_version="logistic_team_form_v1_platt",
            base_model_version="logistic_team_form_v1",
            code_version=module.EXPECTED_ARTIFACT_CODE_VERSION,
            dataset_version="mlb_team_form_v1",
            dataset_sha256=module.EXPECTED_DATASET_SHA256,
            protocol_sha256=module.EXPECTED_MODEL_PROTOCOL_SHA256,
            feature_columns=tuple(module.FEATURE_COLUMNS),
            base_training_seasons=(2021, 2022, 2023),
            calibration_season=2024,
            sealed_test_seasons=(2025,),
            recent_seasons=(2026,),
            calibration_method="sigmoid",
            sklearn_version="1.9.0",
            numpy_version="2.5.2",
            calibrated_classifier=SimpleNamespace(
                classes_=np.asarray([0, 1])
            ),
        )
        artifact_class = module.CalibratedModelArtifact
        original_module = artifact_class.__module__
        missing = object()
        previous = getattr(main_module, "CalibratedModelArtifact", missing)
        buffer = io.BytesIO()
        try:
            artifact_class.__module__ = "__main__"
            setattr(main_module, "CalibratedModelArtifact", artifact_class)
            module.joblib.dump(artifact, buffer, compress=3)
        finally:
            artifact_class.__module__ = original_module
            if previous is missing:
                delattr(main_module, "CalibratedModelArtifact")
            else:
                setattr(main_module, "CalibratedModelArtifact", previous)
        loaded = module._load_artifact_from_verified_bytes(
            buffer.getvalue(),
            configuration,
        )
        self.assertIsInstance(loaded, module.CalibratedModelArtifact)
        self.assertEqual(
            loaded.calibrated_classifier.classes_.tolist(),
            [0, 1],
        )

    def test_output_directory_reservation_is_exclusive(self) -> None:
        """Deux ouvertures concurrentes ne peuvent pas réserver le même dossier."""
        configuration = module.load_evaluation_configuration(
            EVALUATION_PROTOCOL_PATH,
            project_directory=PROJECT_DIRECTORY,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "sealed"
            temporary_configuration = replace(
                configuration,
                output_directory=output,
            )
            marker = module._reserve_output_directory(
                temporary_configuration,
                evaluation_code_commit="a" * 40,
                opening_id="b" * 32,
                opened_at_utc="2026-08-30T20:00:00+00:00",
            )
            self.assertTrue(marker.is_file())
            with self.assertRaises(module.SealedEvaluationError):
                module._reserve_output_directory(
                    temporary_configuration,
                    evaluation_code_commit="a" * 40,
                    opening_id="b" * 32,
                    opened_at_utc="2026-08-30T20:00:00+00:00",
                )

    def test_synthetic_opening_completes_end_to_end(self) -> None:
        """L'orchestration publie une fois deux sorties et un état terminé."""
        configuration = module.load_evaluation_configuration(
            EVALUATION_PROTOCOL_PATH,
            project_directory=PROJECT_DIRECTORY,
        )

        class FakeClassifier:
            classes_ = np.asarray([0, 1])

            def __init__(self) -> None:
                self.calls = 0

            def predict_proba(self, features: np.ndarray) -> np.ndarray:
                self.calls += 1
                probabilities = np.linspace(0.35, 0.65, len(features))
                return np.column_stack((1.0 - probabilities, probabilities))

        classifier = FakeClassifier()
        artifact = SimpleNamespace(calibrated_classifier=classifier)
        artifact_bytes = b"synthetic artifact"
        dataset_bytes = b"synthetic dataset"

        with tempfile.TemporaryDirectory(dir=PROJECT_DIRECTORY) as temporary:
            output = Path(temporary) / "sealed"
            synthetic_configuration = replace(
                configuration,
                output_directory=output,
                artifact_size_bytes=len(artifact_bytes),
                dataset_size_bytes=len(dataset_bytes),
            )

            def verified_bytes(path: Path, **_: object):
                if Path(path) == synthetic_configuration.artifact_path:
                    return (
                        Path(path),
                        artifact_bytes,
                        synthetic_configuration.artifact_sha256,
                    )
                if Path(path) == synthetic_configuration.dataset_path:
                    return (
                        Path(path),
                        dataset_bytes,
                        synthetic_configuration.dataset_sha256,
                    )
                raise AssertionError(f"Lecture inattendue : {path}")

            with (
                patch.object(
                    module,
                    "load_evaluation_configuration",
                    return_value=synthetic_configuration,
                ),
                patch.object(
                    module,
                    "_verify_git_opening_preconditions",
                    return_value="a" * 40,
                ),
                patch.object(
                    module,
                    "_verify_git_unchanged_after_evaluation",
                ),
                patch.object(
                    module,
                    "_read_verified_bytes",
                    side_effect=verified_bytes,
                ),
                patch.object(
                    module,
                    "_load_artifact_from_verified_bytes",
                    return_value=artifact,
                ),
                patch.object(
                    module,
                    "_load_sealed_rows_from_verified_csv",
                    return_value=_synthetic_rows(),
                ),
                patch.object(
                    module,
                    "_artifact_state_sha256",
                    return_value="f" * 64,
                ),
                patch.object(
                    module,
                    "evaluate_sealed_rows",
                    return_value=_dummy_result(),
                ),
                patch.object(module, "EXPECTED_SEALED_ROWS", 6),
            ):
                opened = module.open_sealed_evaluation(
                    expected_evaluation_code_commit="a" * 40,
                )

            self.assertEqual(classifier.calls, 1)
            self.assertTrue(opened.export.predictions_path.is_file())
            self.assertTrue(opened.export.report_path.is_file())
            self.assertTrue((output / "OPENING_COMPLETED").is_file())
            self.assertFalse((output / "OPENING_IN_PROGRESS").exists())
            report = json.loads(opened.export.report_path.read_bytes())
            self.assertEqual(report["status"], "COMPLETED_AND_VERIFIED")
            self.assertEqual(report["verdict"]["label"], module.VERDICT_VALIDATED)

    def test_exclusive_publication_never_overwrites(self) -> None:
        """Une sortie déjà présente reste byte pour byte inchangée."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.csv"
            digest, size = module._publish_exclusive_bytes(path, b"first\n")
            self.assertEqual(digest, hashlib.sha256(b"first\n").hexdigest())
            self.assertEqual(size, 6)
            with self.assertRaises(module.SealedEvaluationError):
                module._publish_exclusive_bytes(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\n")

    def test_prediction_csv_is_canonical_and_2025_only(self) -> None:
        """L'ordre, les colonnes et les fins de ligne sont déterministes."""
        content = module._prediction_csv_bytes(_dummy_result())
        self.assertNotIn(b"\r\n", content)
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["season"] for row in rows}, {"2025"})
        self.assertEqual(tuple(rows[0]), module.PREDICTION_COLUMNS)
        self.assertEqual(content, module._prediction_csv_bytes(_dummy_result()))

    def test_report_records_audit_invariants(self) -> None:
        """Le rapport expose hashes, appel unique et exclusion de 2026."""
        configuration = module.load_evaluation_configuration(
            EVALUATION_PROTOCOL_PATH,
            project_directory=PROJECT_DIRECTORY,
        )
        content = module._report_json_bytes(
            configuration=configuration,
            result=_dummy_result(),
            evaluation_code_commit="a" * 40,
            evaluation_module_sha256="b" * 64,
            opening_id="c" * 32,
            opened_at_utc="2026-08-30T20:00:00+00:00",
            artifact_state_before="d" * 64,
            artifact_state_after="d" * 64,
            predictions_sha256="e" * 64,
            predictions_size_bytes=123,
        )
        report = json.loads(content)
        invariants = report["execution_invariants"]
        self.assertEqual(invariants["predict_proba_calls"], 1)
        self.assertTrue(invariants["artifact_state_unchanged"])
        self.assertEqual(invariants["season_2026_predictions"], 0)
        self.assertEqual(invariants["season_2026_metrics"], 0)
        self.assertEqual(report["verdict"]["label"], module.VERDICT_VALIDATED)

    def test_only_diagnostic_function_contains_a_fit_call(self) -> None:
        """Le code ne peut pas réentraîner l'artefact par accident."""
        tree = ast.parse(inspect.getsource(module))
        functions_with_fit: set[str] = set()
        for function in (
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ):
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "fit"
                ):
                    functions_with_fit.add(function.name)
        self.assertEqual(
            functions_with_fit,
            {"fit_calibration_diagnostics"},
        )
        self.assertNotIn(".partial_fit(", inspect.getsource(module))

    def test_git_commit_must_be_full_lowercase_sha(self) -> None:
        """Un commit abrégé ou ambigu ne peut pas ouvrir 2025."""
        self.assertEqual(module._normalize_git_commit("a" * 40), "a" * 40)
        for invalid in ("a" * 39, "A" * 40, "main", ""):
            with self.subTest(invalid=invalid):
                with self.assertRaises(module.SealedEvaluationError):
                    module._normalize_git_commit(invalid)


if __name__ == "__main__":
    unittest.main()
