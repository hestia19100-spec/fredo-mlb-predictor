"""Tests de la calibration Platt sans ouverture de 2025 ou 2026."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.baseline_model import load_training_dataset
from src.calibrated_model import (
    ARTIFACT_FORMAT_VERSION,
    BASE_TRAINING_SEASONS,
    CALIBRATED_MODEL_VERSION,
    CALIBRATION_METHOD,
    CALIBRATION_SEASON,
    CalibratedModelError,
    load_calibrated_artifact,
    load_model_protocol,
    prepare_calibrated_model,
    prepare_loaded_calibrated_model,
    write_calibrated_artifact,
)
from src.training_dataset import (
    FEATURE_COLUMNS,
    TrainingRow,
    render_training_csv,
)


TRACKED_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "model_protocols"
    / "logistic_team_form_v1.json"
)
CODE_VERSION = "a" * 40


TARGETS_BY_SEASON = {
    2021: (1, 0, 0, 1, 0, 0, 1, 0),
    2022: (1, 1, 0, 1, 0, 1, 0, 1),
    2023: (0, 1, 0, 1, 1, 0, 1, 0),
    2024: (1, 0, 1, 0, 1, 1, 0, 1, 0, 0),
    2025: (1, 1, 1, 0, 1, 1, 1, 0),
    2026: (0, 0, 1, 0, 0, 1, 0, 0),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_rows() -> tuple[TrainingRow, ...]:
    """Crée six saisons distinctes et strictement ordonnées."""
    rows: list[TrainingRow] = []

    for season, targets in TARGETS_BY_SEASON.items():
        for index, home_win in enumerate(targets):
            official_date = date(season, 4, 10) + timedelta(days=index)
            seasonal_offset = season - 2021
            rows.append(
                TrainingRow(
                    game_id=season * 100 + index + 1,
                    season=season,
                    official_date=official_date,
                    feature_as_of_date=official_date - timedelta(days=1),
                    away_team_id=100 + index,
                    home_team_id=200 + index,
                    away_max_source_date=(
                        official_date - timedelta(days=2)
                    ),
                    home_max_source_date=(
                        official_date - timedelta(days=1)
                    ),
                    away_games_before=10 + seasonal_offset + index,
                    away_win_pct_before=(
                        0.20 + 0.07 * ((index + seasonal_offset) % 8)
                    ),
                    away_runs_scored_per_game_before=(
                        2.50 + 0.17 * index + 0.03 * seasonal_offset
                    ),
                    away_runs_allowed_per_game_before=(
                        3.10 + 0.11 * index + 0.02 * seasonal_offset
                    ),
                    home_games_before=12 + seasonal_offset + index,
                    home_win_pct_before=(
                        0.25 + 0.06 * ((2 * index + seasonal_offset) % 8)
                    ),
                    home_runs_scored_per_game_before=(
                        3.20 + 0.13 * index + 0.04 * seasonal_offset
                    ),
                    home_runs_allowed_per_game_before=(
                        2.80 + 0.19 * index + 0.01 * seasonal_offset
                    ),
                    home_win=home_win,
                )
            )

    return tuple(rows)


class CalibratedModelTests(unittest.TestCase):
    """Vérifie la séparation apprentissage, calibration et test."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.rows = _training_rows()
        self.dataset_path = self.root / "training.csv"
        self.dataset_path.write_bytes(render_training_csv(self.rows))
        self.dataset_sha256 = _sha256(self.dataset_path)

        protocol_data = json.loads(
            TRACKED_PROTOCOL_PATH.read_text(encoding="utf-8")
        )
        protocol_data["dataset"]["path"] = self.dataset_path.name
        protocol_data["dataset"]["sha256"] = self.dataset_sha256
        protocol_data["dataset"]["row_count"] = len(self.rows)
        self.protocol_path = self.root / "protocol.json"
        self.protocol_path.write_text(
            json.dumps(
                protocol_data,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="",
        )
        self.protocol_sha256 = _sha256(self.protocol_path)
        self.dataset = load_training_dataset(
            self.dataset_path,
            expected_sha256=self.dataset_sha256,
        )
        self.protocol = load_model_protocol(
            self.protocol_path,
            expected_sha256=self.protocol_sha256,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _prepare(self):
        return prepare_loaded_calibrated_model(
            self.dataset,
            self.protocol,
            code_version=CODE_VERSION,
        )

    def test_protocol_checksum_and_contract_are_required(self) -> None:
        """Une empreinte ou une méthode différente doit échouer."""
        with self.assertRaises(CalibratedModelError):
            load_model_protocol(
                self.protocol_path,
                expected_sha256="0" * 64,
            )

        altered = json.loads(
            self.protocol_path.read_text(encoding="utf-8")
        )
        altered["calibration"]["method"] = "ISOTONIC"
        altered_path = self.root / "altered_protocol.json"
        altered_path.write_text(
            json.dumps(altered, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="",
        )
        with self.assertRaises(CalibratedModelError):
            load_model_protocol(
                altered_path,
                expected_sha256=_sha256(altered_path),
            )

    def test_training_calibration_and_sealed_counts_are_exact(self) -> None:
        """Seules 2021-2023 apprennent et seule 2024 calibre."""
        preparation = self._prepare()

        self.assertEqual(BASE_TRAINING_SEASONS, (2021, 2022, 2023))
        self.assertEqual(CALIBRATION_SEASON, 2024)
        self.assertEqual(preparation.base_training_rows, 24)
        self.assertEqual(preparation.calibration_rows, 10)
        self.assertEqual(preparation.sealed_test_rows, 8)
        self.assertEqual(preparation.recent_rows, 8)
        self.assertTrue(
            preparation.base_model_unchanged_during_calibration
        )
        self.assertFalse(preparation.sealed_predictions_computed)
        self.assertFalse(hasattr(preparation, "sealed_test_metrics"))
        self.assertFalse(hasattr(preparation, "recent_metrics"))

    def test_scaler_and_base_model_are_fitted_on_2021_2023_only(self) -> None:
        """Les moyennes du scaler ne doivent jamais inclure 2024."""
        preparation = self._prepare()
        training_rows = [
            row for row in self.rows if row.season in {2021, 2022, 2023}
        ]
        expected_means = tuple(
            sum(float(getattr(row, feature)) for row in training_rows)
            / len(training_rows)
            for feature in FEATURE_COLUMNS
        )

        self.assertEqual(len(preparation.scaler_mean), 8)
        self.assertEqual(len(preparation.scaler_scale), 8)
        self.assertEqual(len(preparation.base_coefficients), 8)
        for actual, expected in zip(
            preparation.scaler_mean,
            expected_means,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_mutating_sealed_seasons_cannot_change_fitted_model(self) -> None:
        """Des valeurs 2025-2026 différentes restent sans effet."""
        original = self._prepare()
        mutated_rows = tuple(
            replace(
                row,
                features=tuple(value + 1000.0 for value in row.features),
                home_win=1 - row.home_win,
            )
            if row.season in {2025, 2026}
            else row
            for row in self.dataset.rows
        )
        mutated_dataset = replace(self.dataset, rows=mutated_rows)
        mutated = prepare_loaded_calibrated_model(
            mutated_dataset,
            self.protocol,
            code_version=CODE_VERSION,
        )
        x_check = np.asarray(
            [
                row.features
                for row in self.dataset.rows
                if row.season == 2024
            ],
            dtype=np.float64,
        )
        original_probabilities = (
            original.artifact.calibrated_classifier.predict_proba(x_check)
        )
        mutated_probabilities = (
            mutated.artifact.calibrated_classifier.predict_proba(x_check)
        )

        self.assertEqual(original.scaler_mean, mutated.scaler_mean)
        self.assertEqual(
            original.base_coefficients,
            mutated.base_coefficients,
        )
        np.testing.assert_allclose(
            original_probabilities,
            mutated_probabilities,
            rtol=0.0,
            atol=0.0,
        )

    def test_repeated_preparation_is_deterministic_and_read_only(self) -> None:
        """Deux ajustements produisent les mêmes probabilités."""
        dataset_before = self.dataset_path.read_bytes()
        protocol_before = self.protocol_path.read_bytes()
        first = self._prepare()
        second = self._prepare()
        x_check = np.asarray(
            [self.dataset.rows[24].features],
            dtype=np.float64,
        )

        first_probability = (
            first.artifact.calibrated_classifier.predict_proba(x_check)
        )
        second_probability = (
            second.artifact.calibrated_classifier.predict_proba(x_check)
        )
        np.testing.assert_allclose(
            first_probability,
            second_probability,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(first.scaler_mean, second.scaler_mean)
        self.assertEqual(
            first.base_coefficients,
            second.base_coefficients,
        )
        self.assertEqual(self.dataset_path.read_bytes(), dataset_before)
        self.assertEqual(self.protocol_path.read_bytes(), protocol_before)

    def test_public_preparation_verifies_both_source_hashes(self) -> None:
        """Les deux sources doivent correspondre à leurs empreintes."""
        preparation = prepare_calibrated_model(
            dataset_path=self.dataset_path,
            expected_dataset_sha256=self.dataset_sha256,
            protocol_path=self.protocol_path,
            expected_protocol_sha256=self.protocol_sha256,
            code_version=CODE_VERSION,
        )
        self.assertEqual(preparation.base_training_rows, 24)

        with self.assertRaises(CalibratedModelError):
            prepare_calibrated_model(
                dataset_path=self.dataset_path,
                expected_dataset_sha256="0" * 64,
                protocol_path=self.protocol_path,
                expected_protocol_sha256=self.protocol_sha256,
                code_version=CODE_VERSION,
            )

    def test_artifact_round_trip_preserves_predictions_and_metadata(self) -> None:
        """Le joblib relu doit conserver modèle, empreintes et sortie."""
        preparation = self._prepare()
        artifact_path = self.root / "model.joblib"
        export = write_calibrated_artifact(
            preparation,
            output_path=artifact_path,
        )
        loaded = load_calibrated_artifact(
            artifact_path,
            expected_sha256=export.sha256,
        )
        x_check = np.asarray(
            [self.dataset.rows[24].features],
            dtype=np.float64,
        )

        expected_probability = (
            preparation.artifact.calibrated_classifier.predict_proba(
                x_check
            )
        )
        actual_probability = loaded.calibrated_classifier.predict_proba(
            x_check
        )
        np.testing.assert_allclose(
            actual_probability,
            expected_probability,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(
            loaded.artifact_format_version,
            ARTIFACT_FORMAT_VERSION,
        )
        self.assertEqual(
            loaded.calibrated_model_version,
            CALIBRATED_MODEL_VERSION,
        )
        self.assertEqual(loaded.dataset_sha256, self.dataset_sha256)
        self.assertEqual(loaded.protocol_sha256, self.protocol_sha256)
        self.assertEqual(loaded.calibration_method, CALIBRATION_METHOD)
        self.assertGreater(export.size_bytes, 0)

    def test_existing_artifact_is_never_overwritten(self) -> None:
        """Une version existante doit rester strictement identique."""
        preparation = self._prepare()
        artifact_path = self.root / "model.joblib"
        write_calibrated_artifact(
            preparation,
            output_path=artifact_path,
        )
        content_before = artifact_path.read_bytes()

        with self.assertRaises(CalibratedModelError):
            write_calibrated_artifact(
                preparation,
                output_path=artifact_path,
            )

        self.assertEqual(artifact_path.read_bytes(), content_before)

    def test_invalid_artifact_hash_and_code_version_are_rejected(self) -> None:
        """Une provenance incomplète ne doit jamais être acceptée."""
        with self.assertRaises(ValueError):
            prepare_loaded_calibrated_model(
                self.dataset,
                self.protocol,
                code_version="court",
            )

        preparation = self._prepare()
        artifact_path = self.root / "model.joblib"
        write_calibrated_artifact(
            preparation,
            output_path=artifact_path,
        )
        with self.assertRaises(CalibratedModelError):
            load_calibrated_artifact(
                artifact_path,
                expected_sha256="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
