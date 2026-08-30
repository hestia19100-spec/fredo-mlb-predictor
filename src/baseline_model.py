"""Évalue un premier baseline logistique MLB sans ouvrir les saisons test."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import hmac
import io
import math
from pathlib import Path
import re
from typing import Iterable, Sequence
import warnings

import numpy as np
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss as sklearn_log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training_dataset import (
    CALENDAR_POLICY,
    CSV_COLUMNS,
    DATASET_VERSION,
    FEATURE_COLUMNS,
    TEMPORAL_REASON,
    TEMPORAL_VERDICT,
)


MODEL_VERSION = "logistic_team_form_v1"
RANDOM_SEED = 42

WALK_FORWARD_FOLDS = (
    ((2021,), 2022),
    ((2021, 2022), 2023),
    ((2021, 2022, 2023), 2024),
)
SEALED_TEST_SEASONS = (2025,)
SEALED_RECENT_SEASONS = (2026,)
SUPPORTED_SEASONS = frozenset(
    season
    for train_seasons, evaluation_season in WALK_FORWARD_FOLDS
    for season in (*train_seasons, evaluation_season)
) | frozenset(SEALED_TEST_SEASONS) | frozenset(
    SEALED_RECENT_SEASONS
)

_INTEGER_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_DECIMAL_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z"
)
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


class BaselineModelError(RuntimeError):
    """Erreur empêchant une évaluation reproductible du baseline."""


@dataclass(frozen=True, slots=True)
class ModelRow:
    """Ligne validée dont seules les huit variables entrent au modèle."""

    game_id: int
    season: int
    official_date: date
    feature_as_of_date: date
    away_team_id: int
    home_team_id: int
    away_max_source_date: date
    home_max_source_date: date
    features: tuple[float, ...]
    home_win: int


@dataclass(frozen=True, slots=True)
class LoadedTrainingDataset:
    """CSV contrôlé et informations qui permettent de le reproduire."""

    version: str
    calendar_policy: str
    temporal_verdict: str
    temporal_reason: str
    path: Path
    sha256: str
    rows: tuple[ModelRow, ...]


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """Mesures probabilistes et classification au seuil fixe de 50 %."""

    samples: int
    observed_home_win_rate: float
    mean_predicted_home_win_probability: float
    accuracy: float
    brier_score: float
    log_loss: float
    roc_auc: float | None
    calibration_in_the_large: float


@dataclass(frozen=True, slots=True)
class FoldEvaluation:
    """Résultat d’un repli avec apprentissage strictement antérieur."""

    train_seasons: tuple[int, ...]
    evaluation_season: int
    training_rows: int
    evaluation_rows: int
    training_home_win_probability: float
    model_metrics: BinaryMetrics
    constant_metrics: BinaryMetrics
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BaselineReport:
    """Rapport sans prédiction pour les cohortes encore scellées."""

    model_version: str
    dataset_version: str
    dataset_path: Path
    dataset_sha256: str
    calendar_policy: str
    temporal_verdict: str
    temporal_reason: str
    sklearn_version: str
    feature_columns: tuple[str, ...]
    folds: tuple[FoldEvaluation, ...]
    sealed_test_seasons: tuple[int, ...]
    sealed_test_rows: int
    sealed_recent_seasons: tuple[int, ...]
    sealed_recent_rows: int


def _normalize_expected_sha256(expected_sha256: str) -> str:
    """Valide et normalise l’empreinte exigée par l’appelant."""
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise ValueError(
            "expected_sha256 doit contenir exactement 64 caractères "
            "hexadécimaux."
        )

    return expected_sha256.lower()


def _read_dataset_bytes(dataset_path: Path) -> tuple[Path, bytes]:
    """Lit le CSV une seule fois sans jamais le modifier."""
    try:
        path = Path(dataset_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise BaselineModelError(
            f"CSV d’apprentissage introuvable : {dataset_path}."
        ) from error

    if not path.is_file():
        raise BaselineModelError(
            f"Le chemin du dataset n’est pas un fichier : {path}."
        )

    try:
        content = path.read_bytes()
    except OSError as error:
        raise BaselineModelError(
            f"Lecture du CSV impossible : {error}"
        ) from error

    if not content:
        raise BaselineModelError("Le CSV d’apprentissage est vide.")

    return path, content


def _parse_integer(
    value: object,
    *,
    field_name: str,
    line_number: int,
    minimum: int = 0,
) -> int:
    """Lit un entier canonique positif ou nul."""
    if (
        not isinstance(value, str)
        or _INTEGER_PATTERN.fullmatch(value) is None
    ):
        raise BaselineModelError(
            f"Ligne {line_number} : {field_name} n’est pas un entier "
            "canonique."
        )

    parsed = int(value)
    if parsed < minimum:
        raise BaselineModelError(
            f"Ligne {line_number} : {field_name} doit être supérieur "
            f"ou égal à {minimum}."
        )

    return parsed


def _parse_decimal(
    value: object,
    *,
    field_name: str,
    line_number: int,
) -> float:
    """Lit un décimal fini, non négatif et sans notation ambiguë."""
    if (
        not isinstance(value, str)
        or _DECIMAL_PATTERN.fullmatch(value) is None
    ):
        raise BaselineModelError(
            f"Ligne {line_number} : {field_name} n’est pas un décimal "
            "canonique non négatif."
        )

    parsed = float(value)
    if not math.isfinite(parsed):
        raise BaselineModelError(
            f"Ligne {line_number} : {field_name} n’est pas fini."
        )

    return parsed


def _parse_iso_date(
    value: object,
    *,
    field_name: str,
    line_number: int,
) -> date:
    """Lit une date ISO stricte sans heure ni fuseau implicite."""
    if not isinstance(value, str):
        raise BaselineModelError(
            f"Ligne {line_number} : {field_name} est absent."
        )

    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise BaselineModelError(
            f"Ligne {line_number} : {field_name} n’est pas une date ISO."
        ) from error

    if value != parsed.isoformat():
        raise BaselineModelError(
            f"Ligne {line_number} : {field_name} n’est pas canonique."
        )

    return parsed


def _require_exact_fields(
    raw_row: dict[str | None, str | list[str] | None],
    *,
    line_number: int,
) -> None:
    """Refuse les colonnes en trop et les valeurs manquantes."""
    if None in raw_row:
        raise BaselineModelError(
            f"Ligne {line_number} : des valeurs dépassent le schéma CSV."
        )

    missing_values = [
        column
        for column in CSV_COLUMNS
        if raw_row.get(column) in (None, "")
    ]
    if missing_values:
        raise BaselineModelError(
            f"Ligne {line_number} : valeur absente pour "
            + ", ".join(missing_values)
            + "."
        )


def _parse_model_row(
    raw_row: dict[str | None, str | list[str] | None],
    *,
    line_number: int,
) -> ModelRow:
    """Transforme une ligne CSV après tous les contrôles métier."""
    _require_exact_fields(raw_row, line_number=line_number)

    game_id = _parse_integer(
        raw_row["game_id"],
        field_name="game_id",
        line_number=line_number,
        minimum=1,
    )
    season = _parse_integer(
        raw_row["season"],
        field_name="season",
        line_number=line_number,
        minimum=1,
    )
    official_date = _parse_iso_date(
        raw_row["official_date"],
        field_name="official_date",
        line_number=line_number,
    )
    feature_as_of_date = _parse_iso_date(
        raw_row["feature_as_of_date"],
        field_name="feature_as_of_date",
        line_number=line_number,
    )
    away_team_id = _parse_integer(
        raw_row["away_team_id"],
        field_name="away_team_id",
        line_number=line_number,
        minimum=1,
    )
    home_team_id = _parse_integer(
        raw_row["home_team_id"],
        field_name="home_team_id",
        line_number=line_number,
        minimum=1,
    )
    away_max_source_date = _parse_iso_date(
        raw_row["away_max_source_date"],
        field_name="away_max_source_date",
        line_number=line_number,
    )
    home_max_source_date = _parse_iso_date(
        raw_row["home_max_source_date"],
        field_name="home_max_source_date",
        line_number=line_number,
    )

    away_games_before = _parse_integer(
        raw_row["away_games_before"],
        field_name="away_games_before",
        line_number=line_number,
        minimum=1,
    )
    home_games_before = _parse_integer(
        raw_row["home_games_before"],
        field_name="home_games_before",
        line_number=line_number,
        minimum=1,
    )
    away_win_pct = _parse_decimal(
        raw_row["away_win_pct_before"],
        field_name="away_win_pct_before",
        line_number=line_number,
    )
    home_win_pct = _parse_decimal(
        raw_row["home_win_pct_before"],
        field_name="home_win_pct_before",
        line_number=line_number,
    )
    away_runs_scored = _parse_decimal(
        raw_row["away_runs_scored_per_game_before"],
        field_name="away_runs_scored_per_game_before",
        line_number=line_number,
    )
    away_runs_allowed = _parse_decimal(
        raw_row["away_runs_allowed_per_game_before"],
        field_name="away_runs_allowed_per_game_before",
        line_number=line_number,
    )
    home_runs_scored = _parse_decimal(
        raw_row["home_runs_scored_per_game_before"],
        field_name="home_runs_scored_per_game_before",
        line_number=line_number,
    )
    home_runs_allowed = _parse_decimal(
        raw_row["home_runs_allowed_per_game_before"],
        field_name="home_runs_allowed_per_game_before",
        line_number=line_number,
    )
    home_win = _parse_integer(
        raw_row["home_win"],
        field_name="home_win",
        line_number=line_number,
    )

    if season not in SUPPORTED_SEASONS:
        raise BaselineModelError(
            f"Ligne {line_number} : saison non prévue par ce protocole "
            f"({season})."
        )
    if official_date.year != season:
        raise BaselineModelError(
            f"Ligne {line_number} : official_date ne correspond pas à "
            "la saison."
        )
    if feature_as_of_date != official_date - timedelta(days=1):
        raise BaselineModelError(
            f"Ligne {line_number} : feature_as_of_date ne respecte pas J-1."
        )
    if away_max_source_date.year != season:
        raise BaselineModelError(
            f"Ligne {line_number} : l’historique extérieur appartient à "
            "une autre saison."
        )
    if home_max_source_date.year != season:
        raise BaselineModelError(
            f"Ligne {line_number} : l’historique domicile appartient à "
            "une autre saison."
        )
    if away_max_source_date > feature_as_of_date:
        raise BaselineModelError(
            f"Ligne {line_number} : fuite temporelle dans l’historique "
            "extérieur."
        )
    if home_max_source_date > feature_as_of_date:
        raise BaselineModelError(
            f"Ligne {line_number} : fuite temporelle dans l’historique "
            "domicile."
        )
    if away_team_id == home_team_id:
        raise BaselineModelError(
            f"Ligne {line_number} : une équipe ne peut pas s’affronter "
            "elle-même."
        )
    if not 0.0 <= away_win_pct <= 1.0:
        raise BaselineModelError(
            f"Ligne {line_number} : away_win_pct_before est hors de [0, 1]."
        )
    if not 0.0 <= home_win_pct <= 1.0:
        raise BaselineModelError(
            f"Ligne {line_number} : home_win_pct_before est hors de [0, 1]."
        )
    if home_win not in (0, 1):
        raise BaselineModelError(
            f"Ligne {line_number} : home_win doit valoir 0 ou 1."
        )

    features = (
        float(away_games_before),
        away_win_pct,
        away_runs_scored,
        away_runs_allowed,
        float(home_games_before),
        home_win_pct,
        home_runs_scored,
        home_runs_allowed,
    )
    if len(features) != len(FEATURE_COLUMNS):
        raise BaselineModelError(
            "Le nombre de variables ne correspond plus au schéma attendu."
        )

    return ModelRow(
        game_id=game_id,
        season=season,
        official_date=official_date,
        feature_as_of_date=feature_as_of_date,
        away_team_id=away_team_id,
        home_team_id=home_team_id,
        away_max_source_date=away_max_source_date,
        home_max_source_date=home_max_source_date,
        features=features,
        home_win=home_win,
    )


def load_training_dataset(
    dataset_path: Path,
    *,
    expected_sha256: str,
) -> LoadedTrainingDataset:
    """Charge le CSV seulement si son empreinte et son schéma sont exacts."""
    normalized_expected_sha256 = _normalize_expected_sha256(
        expected_sha256
    )
    path, content = _read_dataset_bytes(dataset_path)
    actual_sha256 = hashlib.sha256(content).hexdigest()

    if not hmac.compare_digest(actual_sha256, normalized_expected_sha256):
        raise BaselineModelError(
            "Le SHA-256 du dataset ne correspond pas à l’empreinte attendue : "
            f"attendu {normalized_expected_sha256}, obtenu {actual_sha256}."
        )

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BaselineModelError(
            "Le CSV d’apprentissage n’est pas encodé en UTF-8 valide."
        ) from error

    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None:
        raise BaselineModelError("L’en-tête du CSV est absent.")
    if tuple(reader.fieldnames) != tuple(CSV_COLUMNS):
        raise BaselineModelError(
            "Le schéma ou l’ordre des colonnes du CSV est différent de "
            f"{DATASET_VERSION}."
        )

    rows: list[ModelRow] = []
    seen_game_ids: set[int] = set()
    previous_key: tuple[int, date, int] | None = None

    for raw_row in reader:
        line_number = reader.line_num
        parsed = _parse_model_row(
            raw_row,
            line_number=line_number,
        )
        if parsed.game_id in seen_game_ids:
            raise BaselineModelError(
                f"Ligne {line_number} : game_id dupliqué "
                f"({parsed.game_id})."
            )

        current_key = (
            parsed.season,
            parsed.official_date,
            parsed.game_id,
        )
        if previous_key is not None and current_key <= previous_key:
            raise BaselineModelError(
                f"Ligne {line_number} : les lignes ne sont pas triées par "
                "saison, date officielle et game_id."
            )

        seen_game_ids.add(parsed.game_id)
        previous_key = current_key
        rows.append(parsed)

    if not rows:
        raise BaselineModelError(
            "Le CSV ne contient aucune ligne d’apprentissage."
        )

    return LoadedTrainingDataset(
        version=DATASET_VERSION,
        calendar_policy=CALENDAR_POLICY,
        temporal_verdict=TEMPORAL_VERDICT,
        temporal_reason=TEMPORAL_REASON,
        path=path,
        sha256=actual_sha256,
        rows=tuple(rows),
    )


def build_logistic_pipeline() -> Pipeline:
    """Crée un pipeline neuf ; aucune étape n’est pré-ajustée."""
    return Pipeline(
        steps=(
            ("standard_scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        )
    )


def _as_binary_targets(values: Iterable[int]) -> np.ndarray:
    """Convertit et valide un vecteur de cibles binaires."""
    try:
        target = np.asarray(tuple(values), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise BaselineModelError(
            "Les cibles ne peuvent pas être converties en nombres."
        ) from error

    if target.ndim != 1 or target.size == 0:
        raise BaselineModelError(
            "Les cibles doivent former un vecteur non vide."
        )
    if not np.all(np.isfinite(target)):
        raise BaselineModelError("Les cibles contiennent une valeur non finie.")
    if not np.all(np.isin(target, (0.0, 1.0))):
        raise BaselineModelError("Les cibles doivent valoir 0 ou 1.")

    return target.astype(np.int64, copy=False)


def _as_probabilities(values: Iterable[float]) -> np.ndarray:
    """Convertit et valide des probabilités du domicile."""
    try:
        probabilities = np.asarray(tuple(values), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise BaselineModelError(
            "Les probabilités ne peuvent pas être converties en nombres."
        ) from error

    if probabilities.ndim != 1 or probabilities.size == 0:
        raise BaselineModelError(
            "Les probabilités doivent former un vecteur non vide."
        )
    if not np.all(np.isfinite(probabilities)):
        raise BaselineModelError(
            "Les probabilités contiennent une valeur non finie."
        )
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise BaselineModelError(
            "Les probabilités doivent rester comprises entre 0 et 1."
        )

    return probabilities


def evaluate_probabilities(
    y_true: Iterable[int],
    probabilities: Iterable[float],
) -> BinaryMetrics:
    """Calcule les métriques sans ajuster ni recalibrer les probabilités."""
    target = _as_binary_targets(y_true)
    predicted_probability = _as_probabilities(probabilities)
    if target.shape[0] != predicted_probability.shape[0]:
        raise BaselineModelError(
            "Le nombre de probabilités diffère du nombre de cibles."
        )

    observed_rate = float(np.mean(target))
    mean_probability = float(np.mean(predicted_probability))
    predicted_class = (predicted_probability >= 0.5).astype(np.int64)
    accuracy = float(np.mean(predicted_class == target))
    brier = float(brier_score_loss(target, predicted_probability))
    loss = float(
        sklearn_log_loss(
            target,
            predicted_probability,
            labels=(0, 1),
        )
    )

    if np.unique(target).size == 2:
        auc: float | None = float(
            roc_auc_score(target, predicted_probability)
        )
    else:
        auc = None

    calibration_in_the_large = observed_rate - mean_probability
    numeric_metrics = (
        observed_rate,
        mean_probability,
        accuracy,
        brier,
        loss,
        calibration_in_the_large,
    )
    if not all(math.isfinite(metric) for metric in numeric_metrics):
        raise BaselineModelError("Une métrique calculée n’est pas finie.")
    if auc is not None and not math.isfinite(auc):
        raise BaselineModelError("Le ROC-AUC calculé n’est pas fini.")

    return BinaryMetrics(
        samples=int(target.size),
        observed_home_win_rate=observed_rate,
        mean_predicted_home_win_probability=mean_probability,
        accuracy=accuracy,
        brier_score=brier,
        log_loss=loss,
        roc_auc=auc,
        calibration_in_the_large=calibration_in_the_large,
    )


def _feature_matrix(rows: Sequence[ModelRow]) -> np.ndarray:
    """Construit la matrice uniquement à partir des huit variables autorisées."""
    if not rows:
        raise BaselineModelError(
            "Impossible de créer une matrice de variables vide."
        )

    matrix = np.asarray(
        [row.features for row in rows],
        dtype=np.float64,
    )
    expected_shape = (len(rows), len(FEATURE_COLUMNS))
    if matrix.shape != expected_shape:
        raise BaselineModelError(
            "La matrice des variables ne respecte pas le schéma attendu."
        )
    if not np.all(np.isfinite(matrix)):
        raise BaselineModelError(
            "La matrice des variables contient une valeur non finie."
        )

    return matrix


def _target_vector(rows: Sequence[ModelRow]) -> np.ndarray:
    """Extrait la cible binaire des lignes sélectionnées."""
    return _as_binary_targets(row.home_win for row in rows)


def _validate_evaluation_dataset(dataset: LoadedTrainingDataset) -> None:
    """Vérifie que toutes les cohortes du protocole sont présentes."""
    if dataset.version != DATASET_VERSION:
        raise BaselineModelError(
            f"Version de dataset inattendue : {dataset.version}."
        )
    if dataset.calendar_policy != CALENDAR_POLICY:
        raise BaselineModelError(
            f"Politique calendrier inattendue : {dataset.calendar_policy}."
        )
    if tuple(FEATURE_COLUMNS) != (
        "away_games_before",
        "away_win_pct_before",
        "away_runs_scored_per_game_before",
        "away_runs_allowed_per_game_before",
        "home_games_before",
        "home_win_pct_before",
        "home_runs_scored_per_game_before",
        "home_runs_allowed_per_game_before",
    ):
        raise BaselineModelError(
            "La liste des variables autorisées a changé ; audit requis."
        )

    observed_seasons = {row.season for row in dataset.rows}
    missing_seasons = SUPPORTED_SEASONS - observed_seasons
    unexpected_seasons = observed_seasons - SUPPORTED_SEASONS
    if missing_seasons:
        raise BaselineModelError(
            "Saisons requises absentes du CSV : "
            + ", ".join(str(season) for season in sorted(missing_seasons))
            + "."
        )
    if unexpected_seasons:
        raise BaselineModelError(
            "Saisons non prévues dans le CSV : "
            + ", ".join(
                str(season) for season in sorted(unexpected_seasons)
            )
            + "."
        )


def _rows_for_seasons(
    rows: Sequence[ModelRow],
    seasons: Iterable[int],
) -> tuple[ModelRow, ...]:
    """Sélectionne des saisons sans modifier l’ordre chronologique."""
    selected = frozenset(seasons)
    return tuple(row for row in rows if row.season in selected)


def _evaluate_fold(
    rows: Sequence[ModelRow],
    *,
    train_seasons: tuple[int, ...],
    evaluation_season: int,
) -> FoldEvaluation:
    """Ajuste scaler et régression uniquement sur les saisons antérieures."""
    training_rows = _rows_for_seasons(rows, train_seasons)
    evaluation_rows = _rows_for_seasons(rows, (evaluation_season,))
    if not training_rows or not evaluation_rows:
        raise BaselineModelError(
            "Un repli chronologique ne contient pas assez de lignes."
        )

    latest_training_date = max(row.official_date for row in training_rows)
    earliest_evaluation_date = min(
        row.official_date for row in evaluation_rows
    )
    if latest_training_date >= earliest_evaluation_date:
        raise BaselineModelError(
            f"Le repli vers {evaluation_season} n’est pas chronologique."
        )

    x_train = _feature_matrix(training_rows)
    y_train = _target_vector(training_rows)
    if np.unique(y_train).size != 2:
        raise BaselineModelError(
            f"L’apprentissage {train_seasons} ne contient pas les deux classes."
        )

    x_evaluation = _feature_matrix(evaluation_rows)
    y_evaluation = _target_vector(evaluation_rows)
    pipeline = build_logistic_pipeline()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(x_train, y_train)
        logistic_regression = pipeline.named_steps["logistic_regression"]
        if (
            not np.all(np.isfinite(logistic_regression.coef_))
            or not np.all(np.isfinite(logistic_regression.intercept_))
        ):
            raise BaselineModelError(
                "La régression logistique contient un coefficient non fini."
            )
        class_positions = np.flatnonzero(logistic_regression.classes_ == 1)
        if class_positions.size != 1:
            raise BaselineModelError(
                "La classe domicile=1 est absente du modèle ajusté."
            )
        model_probabilities = pipeline.predict_proba(x_evaluation)[
            :, int(class_positions[0])
        ]
    except BaselineModelError:
        raise
    except (
        TypeError,
        ValueError,
        FloatingPointError,
        ConvergenceWarning,
    ) as error:
        raise BaselineModelError(
            f"Ajustement impossible pour le repli {train_seasons} vers "
            f"{evaluation_season} : {error}"
        ) from error

    training_home_win_probability = float(np.mean(y_train))
    constant_probabilities = np.full(
        shape=y_evaluation.shape,
        fill_value=training_home_win_probability,
        dtype=np.float64,
    )

    scaler = pipeline.named_steps["standard_scaler"]
    scaler_mean = tuple(float(value) for value in scaler.mean_)
    scaler_scale = tuple(float(value) for value in scaler.scale_)

    return FoldEvaluation(
        train_seasons=train_seasons,
        evaluation_season=evaluation_season,
        training_rows=len(training_rows),
        evaluation_rows=len(evaluation_rows),
        training_home_win_probability=training_home_win_probability,
        model_metrics=evaluate_probabilities(
            y_evaluation,
            model_probabilities,
        ),
        constant_metrics=evaluate_probabilities(
            y_evaluation,
            constant_probabilities,
        ),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
    )


def evaluate_loaded_dataset(
    dataset: LoadedTrainingDataset,
) -> BaselineReport:
    """Exécute les replis autorisés sans jamais prédire 2025 ou 2026."""
    _validate_evaluation_dataset(dataset)

    folds = tuple(
        _evaluate_fold(
            dataset.rows,
            train_seasons=train_seasons,
            evaluation_season=evaluation_season,
        )
        for train_seasons, evaluation_season in WALK_FORWARD_FOLDS
    )

    sealed_test_rows = sum(
        row.season in SEALED_TEST_SEASONS for row in dataset.rows
    )
    sealed_recent_rows = sum(
        row.season in SEALED_RECENT_SEASONS for row in dataset.rows
    )

    return BaselineReport(
        model_version=MODEL_VERSION,
        dataset_version=dataset.version,
        dataset_path=dataset.path,
        dataset_sha256=dataset.sha256,
        calendar_policy=dataset.calendar_policy,
        temporal_verdict=dataset.temporal_verdict,
        temporal_reason=dataset.temporal_reason,
        sklearn_version=SKLEARN_VERSION,
        feature_columns=tuple(FEATURE_COLUMNS),
        folds=folds,
        sealed_test_seasons=SEALED_TEST_SEASONS,
        sealed_test_rows=sealed_test_rows,
        sealed_recent_seasons=SEALED_RECENT_SEASONS,
        sealed_recent_rows=sealed_recent_rows,
    )


def run_baseline_evaluation(
    *,
    dataset_path: Path,
    expected_sha256: str,
) -> BaselineReport:
    """Charge, vérifie puis évalue le baseline de manière reproductible."""
    dataset = load_training_dataset(
        dataset_path,
        expected_sha256=expected_sha256,
    )
    return evaluate_loaded_dataset(dataset)


def _format_percentage(value: float) -> str:
    """Formate une proportion pour la sortie humaine."""
    return f"{100.0 * value:.2f}%"


def _format_auc(value: float | None) -> str:
    """Affiche explicitement un AUC impossible à calculer."""
    return "indisponible" if value is None else f"{value:.6f}"


def _display_metrics(label: str, metrics: BinaryMetrics) -> None:
    """Affiche une ligne homogène de mesures."""
    print(
        f"  {label} | Brier {metrics.brier_score:.6f} | "
        f"Log loss {metrics.log_loss:.6f} | "
        f"Exactitude {_format_percentage(metrics.accuracy)} | "
        f"ROC-AUC {_format_auc(metrics.roc_auc)} | "
        "Calibration globale "
        f"{metrics.calibration_in_the_large:+.6f}"
    )


def display_report(report: BaselineReport) -> None:
    """Présente le protocole, les replis autorisés et les cohortes scellées."""
    print("Baseline probabiliste MLB")
    print(f"Version du modèle : {report.model_version}")
    print(f"Version du dataset : {report.dataset_version}")
    print(f"Dataset : {report.dataset_path}")
    print(f"SHA-256 vérifié : {report.dataset_sha256}")
    print(f"scikit-learn : {report.sklearn_version}")
    print(f"Politique calendrier : {report.calendar_policy}")
    print(f"Disponibilité historique réelle : {report.temporal_verdict}")
    print(f"Variables utilisées : {len(report.feature_columns)}")
    for feature_name in report.feature_columns:
        print(f"- {feature_name}")

    print("\nValidation chronologique walk-forward")
    for fold in report.folds:
        training_label = ", ".join(
            str(season) for season in fold.train_seasons
        )
        print(
            f"\nApprentissage {training_label} -> évaluation "
            f"{fold.evaluation_season}"
        )
        print(
            f"Lignes : {fold.training_rows} apprentissage, "
            f"{fold.evaluation_rows} évaluation"
        )
        print(
            "Probabilité domicile de la baseline constante, apprise "
            f"uniquement sur le passé : "
            f"{_format_percentage(fold.training_home_win_probability)}"
        )
        print(
            "Taux domicile observé pendant l’évaluation : "
            f"{_format_percentage(fold.model_metrics.observed_home_win_rate)}"
        )
        _display_metrics("Régression logistique", fold.model_metrics)
        _display_metrics("Constante historique", fold.constant_metrics)

    print("\nCohortes scellées")
    print(
        f"Saison(s) test {', '.join(map(str, report.sealed_test_seasons))} : "
        f"{report.sealed_test_rows} lignes, aucune prédiction ni métrique."
    )
    print(
        "Saison(s) récente(s) "
        f"{', '.join(map(str, report.sealed_recent_seasons))} : "
        f"{report.sealed_recent_rows} lignes, aucune prédiction ni métrique."
    )

    print("\nLecture des métriques")
    print("Un Brier et une log loss plus faibles sont meilleurs.")
    print(
        "La calibration globale vaut taux observé moins probabilité moyenne : "
        "zéro est idéal, une valeur positive indique une sous-estimation des "
        "victoires à domicile."
    )
    print(
        "L’exactitude et le ROC-AUC sont secondaires ; aucune conclusion de "
        "pari, de cote ou de rentabilité n’est calculée ici."
    )

    print("\nLimite temporelle")
    print(report.temporal_reason)
    print(
        "Aucun modèle, recalibrage, fichier de prédictions ou résultat des "
        "saisons scellées n’a été enregistré."
    )


def main() -> None:
    """Point d’entrée de l’évaluation locale en lecture seule."""
    parser = argparse.ArgumentParser(
        description=(
            "Évalue le baseline logistique MLB jusqu’en 2024, sans ouvrir "
            "les cohortes 2025 et 2026."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help=f"CSV exact produit par {DATASET_VERSION}.",
    )
    parser.add_argument(
        "--expected-sha256",
        required=True,
        help="SHA-256 attendu du CSV ; l’évaluation s’arrête s’il diffère.",
    )
    arguments = parser.parse_args()

    try:
        report = run_baseline_evaluation(
            dataset_path=arguments.dataset,
            expected_sha256=arguments.expected_sha256,
        )
        display_report(report)
    except (BaselineModelError, ValueError) as error:
        parser.exit(
            status=1,
            message=f"Échec du baseline MLB : {error}\n",
        )


if __name__ == "__main__":
    main()
