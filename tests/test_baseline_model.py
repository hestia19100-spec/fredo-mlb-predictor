"""Tests du premier modèle logistique MLB et de son étanchéité temporelle."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import date, timedelta
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

from src.baseline_model import (
    BaselineModelError,
    load_training_dataset,
    run_baseline_evaluation,
)
from src.training_dataset import (
    CSV_COLUMNS,
    DATASET_VERSION,
    FEATURE_COLUMNS,
    TrainingRow,
    render_training_csv,
)


TARGETS_BY_SEASON = {
    2021: (1, 0, 0, 1, 0, 0, 1, 0),
    2022: (1, 1, 0, 1, 0, 1, 0, 1),
    2023: (0, 1, 0, 1, 1, 0, 1, 0),
    2024: (1, 0, 1, 0, 1, 1, 0, 1),
    2025: (1, 1, 1, 0, 1, 1, 1, 0),
    2026: (0, 0, 1, 0, 0, 1, 0, 0),
}


def _sha256(path: Path) -> str:
    """Calcule l'empreinte des octets réellement présents sur disque."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_rows() -> tuple[TrainingRow, ...]:
    """Crée six saisons triées dont les distributions sont distinctes."""
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


def _read_records(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Relit le CSV dans un format facile à altérer pour un test."""
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise AssertionError("Le CSV de test doit avoir un en-tête.")
        return list(reader.fieldnames), list(reader)


def _write_records(
    path: Path,
    fieldnames: list[str],
    records: list[dict[str, str]],
) -> None:
    """Écrit une variante contrôlée sans dépendre du module testé."""
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=fieldnames,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)


class BaselineModelTests(unittest.TestCase):
    """Vérifie le modèle sans jamais ouvrir les cohortes scellées."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.rows = _training_rows()
        self.dataset_path = self.root / "training.csv"
        self.dataset_path.write_bytes(render_training_csv(self.rows))
        self.dataset_sha256 = _sha256(self.dataset_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _load(self, path: Path | None = None):
        resolved_path = path or self.dataset_path
        return load_training_dataset(
            resolved_path,
            expected_sha256=_sha256(resolved_path),
        )

    def _evaluate(self, path: Path | None = None):
        resolved_path = path or self.dataset_path
        return run_baseline_evaluation(
            dataset_path=resolved_path,
            expected_sha256=_sha256(resolved_path),
        )

    def _variant_path(
        self,
        name: str,
        *,
        fieldnames: list[str] | None = None,
        records: list[dict[str, str]] | None = None,
    ) -> Path:
        original_fields, original_records = _read_records(self.dataset_path)
        path = self.root / name
        _write_records(
            path,
            fieldnames or original_fields,
            records if records is not None else original_records,
        )
        return path

    def test_sha256_is_required_and_mismatch_is_rejected(self) -> None:
        """Aucun CSV ne doit être accepté sans empreinte attendue exacte."""
        with self.assertRaises(TypeError):
            load_training_dataset(self.dataset_path)  # type: ignore[call-arg]

        with self.assertRaises(BaselineModelError):
            load_training_dataset(
                self.dataset_path,
                expected_sha256="0" * 64,
            )

    def test_valid_dataset_is_loaded_without_modifying_source(self) -> None:
        """Le chargeur conserve les octets et les métadonnées d'audit."""
        source_before = self.dataset_path.read_bytes()

        dataset = self._load()

        self.assertEqual(dataset.version, DATASET_VERSION)
        self.assertEqual(dataset.path, self.dataset_path.resolve())
        self.assertEqual(dataset.sha256, self.dataset_sha256)
        self.assertEqual(len(dataset.rows), len(self.rows))
        self.assertEqual(
            tuple(row.game_id for row in dataset.rows),
            tuple(row.game_id for row in self.rows),
        )
        self.assertEqual(self.dataset_path.read_bytes(), source_before)

    def test_schema_must_be_exact_complete_and_ordered(self) -> None:
        """Une colonne absente, ajoutée ou déplacée invalide le contrat."""
        fields, records = _read_records(self.dataset_path)

        missing_fields = [
            field for field in fields if field != "home_win"
        ]
        extra_fields = fields + ["target_score_leak"]
        extra_records = [dict(record) for record in records]
        for record in extra_records:
            record["target_score_leak"] = "9"
        reordered_fields = list(fields)
        reordered_fields[8], reordered_fields[9] = (
            reordered_fields[9],
            reordered_fields[8],
        )

        variants = (
            ("missing.csv", missing_fields, records),
            ("extra.csv", extra_fields, extra_records),
            ("reordered.csv", reordered_fields, records),
        )
        for name, variant_fields, variant_records in variants:
            with self.subTest(name=name):
                path = self._variant_path(
                    name,
                    fieldnames=variant_fields,
                    records=variant_records,
                )
                with self.assertRaises(BaselineModelError):
                    self._load(path)

        self.assertEqual(tuple(fields), CSV_COLUMNS)

    def test_non_finite_values_and_temporal_leaks_are_rejected(self) -> None:
        """NaN, infini ou date source du jour cible doivent échouer."""
        fields, source_records = _read_records(self.dataset_path)
        mutations = (
            ("nan.csv", "away_win_pct_before", "nan"),
            ("infinity.csv", "home_runs_scored_per_game_before", "inf"),
            (
                "as_of_target_day.csv",
                "feature_as_of_date",
                source_records[0]["official_date"],
            ),
            (
                "source_target_day.csv",
                "away_max_source_date",
                source_records[0]["official_date"],
            ),
        )

        for name, field_name, replacement in mutations:
            with self.subTest(name=name):
                records = [dict(record) for record in source_records]
                records[0][field_name] = replacement
                path = self._variant_path(
                    name,
                    fieldnames=fields,
                    records=records,
                )
                with self.assertRaises(BaselineModelError):
                    self._load(path)

    def test_unsorted_and_duplicate_rows_are_rejected(self) -> None:
        """Chaque match doit apparaître une fois dans l'ordre canonique."""
        fields, source_records = _read_records(self.dataset_path)

        unsorted_records = [dict(record) for record in source_records]
        unsorted_records[0], unsorted_records[1] = (
            unsorted_records[1],
            unsorted_records[0],
        )
        duplicate_records = [dict(record) for record in source_records]
        duplicate_records.insert(1, dict(duplicate_records[0]))

        for name, records in (
            ("unsorted.csv", unsorted_records),
            ("duplicate.csv", duplicate_records),
        ):
            with self.subTest(name=name):
                path = self._variant_path(
                    name,
                    fieldnames=fields,
                    records=records,
                )
                with self.assertRaises(BaselineModelError):
                    self._load(path)

    def test_walk_forward_folds_and_sealed_counts_are_exact(self) -> None:
        """Seules 2022, 2023 et 2024 sont évaluées à cette étape."""
        report = self._evaluate()

        self.assertEqual(
            tuple(
                (fold.train_seasons, fold.evaluation_season)
                for fold in report.folds
            ),
            (
                ((2021,), 2022),
                ((2021, 2022), 2023),
                ((2021, 2022, 2023), 2024),
            ),
        )
        self.assertEqual(
            tuple(fold.training_rows for fold in report.folds),
            (8, 16, 24),
        )
        self.assertEqual(
            tuple(fold.evaluation_rows for fold in report.folds),
            (8, 8, 8),
        )
        self.assertEqual(report.feature_columns, FEATURE_COLUMNS)
        self.assertEqual(report.sealed_test_seasons, (2025,))
        self.assertEqual(report.sealed_test_rows, 8)
        self.assertEqual(report.sealed_recent_seasons, (2026,))
        self.assertEqual(report.sealed_recent_rows, 8)

        for fold in report.folds:
            for metrics in (fold.model_metrics, fold.constant_metrics):
                self.assertEqual(metrics.samples, 8)
                self.assertGreaterEqual(metrics.accuracy, 0.0)
                self.assertLessEqual(metrics.accuracy, 1.0)
                self.assertGreaterEqual(metrics.brier_score, 0.0)
                self.assertLessEqual(metrics.brier_score, 1.0)
                self.assertTrue(math.isfinite(metrics.log_loss))
                self.assertIsNotNone(metrics.roc_auc)
                self.assertTrue(math.isfinite(metrics.roc_auc or 0.0))

    def test_constant_prior_comes_from_training_seasons_only(self) -> None:
        """La probabilité naïve est la moyenne des cibles d'entraînement."""
        report = self._evaluate()
        expected_priors = (3 / 8, 8 / 16, 12 / 24)

        for fold, expected_prior in zip(
            report.folds,
            expected_priors,
            strict=True,
        ):
            self.assertAlmostEqual(
                fold.training_home_win_probability,
                expected_prior,
            )
            self.assertAlmostEqual(
                fold.constant_metrics.mean_predicted_home_win_probability,
                expected_prior,
            )

        self.assertNotAlmostEqual(
            report.folds[0].training_home_win_probability,
            sum(self.rows[index].home_win for index in range(len(self.rows)))
            / len(self.rows),
        )

    def test_scaler_is_fitted_on_each_training_window_only(self) -> None:
        """Les moyennes du scaler excluent l'année évaluée et le futur."""
        report = self._evaluate()

        for fold in report.folds:
            training_rows = [
                row
                for row in self.rows
                if row.season in fold.train_seasons
            ]
            expected_means = tuple(
                sum(float(getattr(row, feature)) for row in training_rows)
                / len(training_rows)
                for feature in FEATURE_COLUMNS
            )
            self.assertEqual(len(fold.scaler_mean), len(FEATURE_COLUMNS))
            self.assertEqual(len(fold.scaler_scale), len(FEATURE_COLUMNS))
            for actual, expected in zip(
                fold.scaler_mean,
                expected_means,
                strict=True,
            ):
                self.assertAlmostEqual(actual, expected, places=12)
            for scale in fold.scaler_scale:
                self.assertTrue(math.isfinite(scale))
                self.assertGreater(scale, 0.0)

    def test_mutating_2025_and_2026_cannot_change_fold_metrics(self) -> None:
        """Les cohortes scellées ne sont ni ajustées ni prédites."""
        original_report = self._evaluate()
        mutated_rows = tuple(
            replace(
                row,
                away_games_before=row.away_games_before + 100,
                away_win_pct_before=1.0 - row.away_win_pct_before,
                away_runs_scored_per_game_before=(
                    row.away_runs_scored_per_game_before + 50.0
                ),
                away_runs_allowed_per_game_before=(
                    row.away_runs_allowed_per_game_before + 60.0
                ),
                home_games_before=row.home_games_before + 100,
                home_win_pct_before=1.0 - row.home_win_pct_before,
                home_runs_scored_per_game_before=(
                    row.home_runs_scored_per_game_before + 70.0
                ),
                home_runs_allowed_per_game_before=(
                    row.home_runs_allowed_per_game_before + 80.0
                ),
                home_win=1 - row.home_win,
            )
            if row.season in {2025, 2026}
            else row
            for row in self.rows
        )
        mutated_path = self.root / "mutated_sealed.csv"
        mutated_path.write_bytes(render_training_csv(mutated_rows))

        mutated_report = self._evaluate(mutated_path)

        self.assertNotEqual(
            original_report.dataset_sha256,
            mutated_report.dataset_sha256,
        )
        self.assertEqual(original_report.folds, mutated_report.folds)
        self.assertEqual(mutated_report.sealed_test_rows, 8)
        self.assertEqual(mutated_report.sealed_recent_rows, 8)

    def test_repeated_evaluation_is_deterministic_and_read_only(self) -> None:
        """Même entrée, mêmes métriques, sans un seul octet source modifié."""
        source_before = self.dataset_path.read_bytes()
        hash_before = _sha256(self.dataset_path)

        first = self._evaluate()
        second = self._evaluate()

        self.assertEqual(first, second)
        self.assertEqual(self.dataset_path.read_bytes(), source_before)
        self.assertEqual(_sha256(self.dataset_path), hash_before)


if __name__ == "__main__":
    unittest.main()
