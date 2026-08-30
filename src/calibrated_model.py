"""Prépare la calibration Platt MLB sans ouvrir les saisons scellées."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import hmac
import io
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence
import warnings

import joblib
import numpy as np
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator

from src.baseline_model import (
    BaselineModelError,
    LoadedTrainingDataset,
    ModelRow,
    SEALED_RECENT_SEASONS,
    SEALED_TEST_SEASONS,
    build_logistic_pipeline,
    load_training_dataset,
)
from src.training_dataset import (
    DATASET_VERSION,
    FEATURE_COLUMNS,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = (
    PROJECT_DIRECTORY
    / "model_protocols"
    / "logistic_team_form_v1.json"
)
DEFAULT_ARTIFACT_PATH = (
    PROJECT_DIRECTORY
    / "models"
    / "logistic_team_form_v1_platt.joblib"
)

ARTIFACT_FORMAT_VERSION = 1
CALIBRATED_MODEL_VERSION = "logistic_team_form_v1_platt"
CALIBRATION_METHOD = "sigmoid"
BASE_TRAINING_SEASONS = (2021, 2022, 2023)
CALIBRATION_SEASON = 2024
EXPECTED_PROTOCOL_SHA256 = (
    "c4cb1af750619967514d37ae3a5a47a6"
    "a04255aeaccb20c5e94533dc4d138451"
)

_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


class CalibratedModelError(RuntimeError):
    """Erreur empêchant de produire un modèle calibré fiable."""


@dataclass(frozen=True, slots=True)
class ModelProtocol:
    """Partie exécutable du contrat enregistré avant le test final."""

    path: Path
    sha256: str
    model_version: str
    dataset_version: str
    dataset_path: str
    dataset_sha256: str
    dataset_rows: int
    feature_columns: tuple[str, ...]
    base_training_seasons: tuple[int, ...]
    calibration_season: int
    sealed_test_seasons: tuple[int, ...]
    recent_seasons: tuple[int, ...]
    calibration_method: str


@dataclass(frozen=True, slots=True)
class CalibratedModelArtifact:
    """Objet versionné qui contient le classifieur prêt à prédire."""

    artifact_format_version: int
    calibrated_model_version: str
    base_model_version: str
    code_version: str
    dataset_version: str
    dataset_sha256: str
    protocol_sha256: str
    feature_columns: tuple[str, ...]
    base_training_seasons: tuple[int, ...]
    calibration_season: int
    sealed_test_seasons: tuple[int, ...]
    recent_seasons: tuple[int, ...]
    calibration_method: str
    sklearn_version: str
    numpy_version: str
    calibrated_classifier: CalibratedClassifierCV


@dataclass(frozen=True, slots=True)
class CalibrationPreparation:
    """Résultat auditable sans métrique sur 2025 ou 2026."""

    artifact: CalibratedModelArtifact
    dataset_path: Path
    protocol_path: Path
    base_training_rows: int
    calibration_rows: int
    base_training_home_win_rate: float
    calibration_home_win_rate: float
    sealed_test_rows: int
    recent_rows: int
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    base_coefficients: tuple[float, ...]
    base_intercept: float
    base_model_unchanged_during_calibration: bool
    sealed_predictions_computed: bool


@dataclass(frozen=True, slots=True)
class ArtifactExport:
    """Empreinte d'un artefact nouvellement écrit et revérifié."""

    path: Path
    sha256: str
    size_bytes: int


def _normalize_sha256(value: str, *, field_name: str) -> str:
    """Valide une empreinte SHA-256 fournie par l'appelant."""
    if (
        not isinstance(value, str)
        or _SHA256_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(
            f"{field_name} doit contenir exactement 64 caractères "
            "hexadécimaux."
        )

    return value.lower()


def _read_verified_bytes(
    path: Path,
    *,
    expected_sha256: str,
    description: str,
) -> tuple[Path, bytes, str]:
    """Lit un fichier seulement si son empreinte est celle attendue."""
    normalized_expected = _normalize_sha256(
        expected_sha256,
        field_name=f"SHA-256 attendu du {description}",
    )
    try:
        resolved_path = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CalibratedModelError(
            f"{description.capitalize()} introuvable : {path}."
        ) from error

    if not resolved_path.is_file():
        raise CalibratedModelError(
            f"Le chemin du {description} n'est pas un fichier : "
            f"{resolved_path}."
        )

    try:
        content = resolved_path.read_bytes()
    except OSError as error:
        raise CalibratedModelError(
            f"Lecture du {description} impossible : {error}"
        ) from error

    if not content:
        raise CalibratedModelError(
            f"Le {description} est vide."
        )

    actual_sha256 = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(actual_sha256, normalized_expected):
        raise CalibratedModelError(
            f"Le SHA-256 du {description} diffère : attendu "
            f"{normalized_expected}, obtenu {actual_sha256}."
        )

    return resolved_path, content, actual_sha256


def _require_mapping(
    value: object,
    *,
    field_name: str,
) -> dict[str, Any]:
    """Exige un objet JSON dont les clés sont textuelles."""
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise CalibratedModelError(
            f"Le champ du protocole {field_name} doit être un objet."
        )

    return value


def _require_integer_list(
    value: object,
    *,
    field_name: str,
) -> tuple[int, ...]:
    """Lit une liste JSON non vide d'années entières distinctes."""
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not int for item in value)
    ):
        raise CalibratedModelError(
            f"Le champ du protocole {field_name} doit être une liste "
            "non vide d'entiers."
        )

    years = tuple(value)
    if len(years) != len(set(years)):
        raise CalibratedModelError(
            f"Le champ du protocole {field_name} contient un doublon."
        )

    return years


def load_model_protocol(
    protocol_path: Path,
    *,
    expected_sha256: str,
) -> ModelProtocol:
    """Charge et contrôle les règles qui gouvernent la calibration."""
    path, content, actual_sha256 = _read_verified_bytes(
        protocol_path,
        expected_sha256=expected_sha256,
        description="protocole du modèle",
    )
    try:
        raw_protocol = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibratedModelError(
            "Le protocole n'est pas un document JSON UTF-8 valide."
        ) from error

    protocol = _require_mapping(
        raw_protocol,
        field_name="racine",
    )
    model = _require_mapping(protocol.get("model"), field_name="model")
    dataset = _require_mapping(
        protocol.get("dataset"),
        field_name="dataset",
    )
    chronology = _require_mapping(
        protocol.get("chronology"),
        field_name="chronology",
    )
    calibration = _require_mapping(
        protocol.get("calibration"),
        field_name="calibration",
    )

    features = protocol.get("features")
    if (
        not isinstance(features, list)
        or not all(isinstance(item, str) for item in features)
    ):
        raise CalibratedModelError(
            "Le protocole doit contenir une liste de variables textuelles."
        )

    base_training_seasons = _require_integer_list(
        chronology.get("final_base_model_training_seasons"),
        field_name="final_base_model_training_seasons",
    )
    calibration_season = chronology.get("calibration_season")
    sealed_test_season = chronology.get("sealed_test_season")
    recent_season = chronology.get("recent_retrospective_season")
    if any(
        type(value) is not int
        for value in (
            calibration_season,
            sealed_test_season,
            recent_season,
        )
    ):
        raise CalibratedModelError(
            "Les saisons de calibration, test et récente doivent être "
            "des entiers."
        )

    dataset_rows = dataset.get("row_count")
    if type(dataset_rows) is not int or dataset_rows <= 0:
        raise CalibratedModelError(
            "Le nombre de lignes du dataset dans le protocole est invalide."
        )

    dataset_sha256 = _normalize_sha256(
        dataset.get("sha256"),
        field_name="SHA-256 du dataset dans le protocole",
    )

    loaded = ModelProtocol(
        path=path,
        sha256=actual_sha256,
        model_version=str(model.get("model_version", "")),
        dataset_version=str(dataset.get("dataset_version", "")),
        dataset_path=str(dataset.get("path", "")),
        dataset_sha256=dataset_sha256,
        dataset_rows=dataset_rows,
        feature_columns=tuple(features),
        base_training_seasons=base_training_seasons,
        calibration_season=calibration_season,
        sealed_test_seasons=(sealed_test_season,),
        recent_seasons=(recent_season,),
        calibration_method=str(calibration.get("method", "")),
    )
    _validate_protocol_contract(loaded, raw_protocol=protocol)
    return loaded


def _validate_protocol_contract(
    protocol: ModelProtocol,
    *,
    raw_protocol: dict[str, Any],
) -> None:
    """Refuse toute déviation par rapport au protocole v1 enregistré."""
    if raw_protocol.get("protocol_version") != 1:
        raise CalibratedModelError(
            "Version de protocole inattendue."
        )
    if raw_protocol.get("status") != "REGISTERED_BEFORE_SEALED_TEST":
        raise CalibratedModelError(
            "Le protocole n'est pas enregistré avant le test scellé."
        )
    if protocol.model_version != "logistic_team_form_v1":
        raise CalibratedModelError("Version du modèle de base inattendue.")
    if protocol.dataset_version != DATASET_VERSION:
        raise CalibratedModelError("Version du dataset inattendue.")
    if protocol.feature_columns != tuple(FEATURE_COLUMNS):
        raise CalibratedModelError(
            "Les variables du protocole diffèrent du dataset."
        )
    if protocol.base_training_seasons != BASE_TRAINING_SEASONS:
        raise CalibratedModelError(
            "Les saisons d'apprentissage du protocole ont changé."
        )
    if protocol.calibration_season != CALIBRATION_SEASON:
        raise CalibratedModelError(
            "La saison de calibration du protocole a changé."
        )
    if protocol.sealed_test_seasons != SEALED_TEST_SEASONS:
        raise CalibratedModelError(
            "La saison de test scellée du protocole a changé."
        )
    if protocol.recent_seasons != SEALED_RECENT_SEASONS:
        raise CalibratedModelError(
            "La saison récente du protocole a changé."
        )
    if protocol.calibration_method != "PLATT_SIGMOID":
        raise CalibratedModelError(
            "La méthode de calibration n'est plus Platt/sigmoid."
        )

    chronology = _require_mapping(
        raw_protocol.get("chronology"),
        field_name="chronology",
    )
    if chronology.get("sealed_test_rule") != (
        "NO_PREDICTION_OR_METRIC_BEFORE_PROTOCOL_AND_"
        "CALIBRATION_ARE_FROZEN"
    ):
        raise CalibratedModelError(
            "La règle de fermeture du test 2025 a changé."
        )

    calibration = _require_mapping(
        raw_protocol.get("calibration"),
        field_name="calibration",
    )
    if calibration.get("isotonic_is_allowed_for_this_version") is not False:
        raise CalibratedModelError(
            "Isotonic ne doit pas être autorisé pour cette version."
        )


def _rows_for_seasons(
    rows: Sequence[ModelRow],
    seasons: Sequence[int],
) -> tuple[ModelRow, ...]:
    """Sélectionne uniquement les saisons explicitement autorisées."""
    selected = frozenset(seasons)
    return tuple(row for row in rows if row.season in selected)


def _feature_matrix(rows: Sequence[ModelRow]) -> np.ndarray:
    """Construit une matrice finie avec les huit variables figées."""
    if not rows:
        raise CalibratedModelError(
            "Impossible de construire une matrice de variables vide."
        )

    matrix = np.asarray(
        [row.features for row in rows],
        dtype=np.float64,
    )
    if matrix.shape != (len(rows), len(FEATURE_COLUMNS)):
        raise CalibratedModelError(
            "La matrice ne respecte pas le nombre de variables figé."
        )
    if not np.all(np.isfinite(matrix)):
        raise CalibratedModelError(
            "La matrice contient une valeur non finie."
        )

    return matrix


def _target_vector(rows: Sequence[ModelRow]) -> np.ndarray:
    """Construit la cible binaire d'une cohorte autorisée."""
    if not rows:
        raise CalibratedModelError(
            "Impossible de construire une cible vide."
        )

    target = np.asarray(
        [row.home_win for row in rows],
        dtype=np.int64,
    )
    if target.shape != (len(rows),):
        raise CalibratedModelError("La cible n'est pas un vecteur.")
    if not np.all(np.isin(target, (0, 1))):
        raise CalibratedModelError("La cible doit valoir 0 ou 1.")
    if np.unique(target).size != 2:
        raise CalibratedModelError(
            "La cohorte doit contenir les deux classes."
        )

    return target


def _detect_code_version(project_directory: Path) -> str:
    """Relit le commit Git qui rend l'entraînement reproductible."""
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=project_directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CalibratedModelError(
            "Impossible d'identifier la version Git du code."
        ) from error

    commit = result.stdout.strip().lower()
    if _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise CalibratedModelError(
            "La version Git du code n'est pas un commit complet."
        )

    return commit


def _validate_code_version(code_version: str) -> str:
    """Exige un identifiant Git complet en minuscules."""
    if not isinstance(code_version, str):
        raise ValueError("code_version doit être une chaîne.")

    normalized = code_version.strip().lower()
    if _GIT_COMMIT_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "code_version doit contenir exactement 40 caractères "
            "hexadécimaux."
        )

    return normalized


def _validate_dataset_against_protocol(
    dataset: LoadedTrainingDataset,
    protocol: ModelProtocol,
) -> None:
    """Vérifie que le CSV chargé est exactement celui enregistré."""
    if dataset.version != protocol.dataset_version:
        raise CalibratedModelError(
            "La version du CSV diffère du protocole."
        )
    if dataset.sha256 != protocol.dataset_sha256:
        raise CalibratedModelError(
            "L'empreinte du CSV diffère du protocole."
        )
    if len(dataset.rows) != protocol.dataset_rows:
        raise CalibratedModelError(
            "Le nombre de lignes du CSV diffère du protocole."
        )

    expected_name = Path(protocol.dataset_path).name
    if dataset.path.name != expected_name:
        raise CalibratedModelError(
            "Le nom du CSV diffère du protocole."
        )


def _pipeline_state(
    pipeline: Any,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    float,
]:
    """Capture les paramètres ajustés du modèle de base."""
    scaler = pipeline.named_steps["standard_scaler"]
    estimator = pipeline.named_steps["logistic_regression"]

    mean = tuple(float(value) for value in scaler.mean_)
    scale = tuple(float(value) for value in scaler.scale_)
    coefficients = tuple(float(value) for value in estimator.coef_[0])
    intercept = float(estimator.intercept_[0])
    numeric_values = (*mean, *scale, *coefficients, intercept)
    if not all(math.isfinite(value) for value in numeric_values):
        raise CalibratedModelError(
            "Le modèle de base contient un paramètre non fini."
        )
    if len(mean) != len(FEATURE_COLUMNS):
        raise CalibratedModelError(
            "Le scaler n'a pas le nombre de moyennes attendu."
        )
    if len(scale) != len(FEATURE_COLUMNS) or any(
        value <= 0.0 for value in scale
    ):
        raise CalibratedModelError(
            "Le scaler contient une échelle invalide."
        )
    if len(coefficients) != len(FEATURE_COLUMNS):
        raise CalibratedModelError(
            "La régression n'a pas le nombre de coefficients attendu."
        )

    return mean, scale, coefficients, intercept


def prepare_loaded_calibrated_model(
    dataset: LoadedTrainingDataset,
    protocol: ModelProtocol,
    *,
    code_version: str,
) -> CalibrationPreparation:
    """Ajuste 2021-2023 puis Platt sur 2024, jamais sur 2025-2026."""
    normalized_code_version = _validate_code_version(code_version)
    _validate_dataset_against_protocol(dataset, protocol)

    training_rows = _rows_for_seasons(
        dataset.rows,
        protocol.base_training_seasons,
    )
    calibration_rows = _rows_for_seasons(
        dataset.rows,
        (protocol.calibration_season,),
    )
    if not training_rows or not calibration_rows:
        raise CalibratedModelError(
            "Les cohortes d'apprentissage ou de calibration sont vides."
        )

    latest_training_date = max(row.official_date for row in training_rows)
    earliest_calibration_date = min(
        row.official_date for row in calibration_rows
    )
    if latest_training_date >= earliest_calibration_date:
        raise CalibratedModelError(
            "La calibration n'est pas strictement postérieure à "
            "l'apprentissage."
        )

    x_training = _feature_matrix(training_rows)
    y_training = _target_vector(training_rows)
    x_calibration = _feature_matrix(calibration_rows)
    y_calibration = _target_vector(calibration_rows)

    base_pipeline = build_logistic_pipeline()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            base_pipeline.fit(x_training, y_training)
    except (TypeError, ValueError, FloatingPointError, ConvergenceWarning) as error:
        raise CalibratedModelError(
            f"Ajustement du modèle de base impossible : {error}"
        ) from error

    state_before_calibration = _pipeline_state(base_pipeline)
    calibrated_classifier = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_pipeline),
        method=CALIBRATION_METHOD,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            calibrated_classifier.fit(x_calibration, y_calibration)
    except (TypeError, ValueError, FloatingPointError, ConvergenceWarning) as error:
        raise CalibratedModelError(
            f"Calibration Platt impossible : {error}"
        ) from error

    state_after_calibration = _pipeline_state(base_pipeline)
    base_unchanged = state_before_calibration == state_after_calibration
    if not base_unchanged:
        raise CalibratedModelError(
            "Le calibrateur a modifié le modèle de base."
        )
    if tuple(int(value) for value in calibrated_classifier.classes_) != (
        0,
        1,
    ):
        raise CalibratedModelError(
            "Le calibrateur ne contient pas les deux classes attendues."
        )
    if len(calibrated_classifier.calibrated_classifiers_) != 1:
        raise CalibratedModelError(
            "La calibration doit contenir un seul couple figé."
        )

    calibration_probabilities = calibrated_classifier.predict_proba(
        x_calibration
    )
    if calibration_probabilities.shape != (len(calibration_rows), 2):
        raise CalibratedModelError(
            "Le calibrateur produit une matrice de probabilités invalide."
        )
    if (
        not np.all(np.isfinite(calibration_probabilities))
        or np.any(calibration_probabilities < 0.0)
        or np.any(calibration_probabilities > 1.0)
        or not np.allclose(
            np.sum(calibration_probabilities, axis=1),
            1.0,
        )
    ):
        raise CalibratedModelError(
            "Le calibrateur produit une probabilité invalide."
        )

    sealed_test_rows = sum(
        row.season in protocol.sealed_test_seasons
        for row in dataset.rows
    )
    recent_rows = sum(
        row.season in protocol.recent_seasons
        for row in dataset.rows
    )
    if sealed_test_rows <= 0 or recent_rows <= 0:
        raise CalibratedModelError(
            "Une cohorte scellée attendue est absente."
        )

    mean, scale, coefficients, intercept = state_after_calibration
    artifact = CalibratedModelArtifact(
        artifact_format_version=ARTIFACT_FORMAT_VERSION,
        calibrated_model_version=CALIBRATED_MODEL_VERSION,
        base_model_version=protocol.model_version,
        code_version=normalized_code_version,
        dataset_version=dataset.version,
        dataset_sha256=dataset.sha256,
        protocol_sha256=protocol.sha256,
        feature_columns=tuple(FEATURE_COLUMNS),
        base_training_seasons=protocol.base_training_seasons,
        calibration_season=protocol.calibration_season,
        sealed_test_seasons=protocol.sealed_test_seasons,
        recent_seasons=protocol.recent_seasons,
        calibration_method=CALIBRATION_METHOD,
        sklearn_version=SKLEARN_VERSION,
        numpy_version=np.__version__,
        calibrated_classifier=calibrated_classifier,
    )

    return CalibrationPreparation(
        artifact=artifact,
        dataset_path=dataset.path,
        protocol_path=protocol.path,
        base_training_rows=len(training_rows),
        calibration_rows=len(calibration_rows),
        base_training_home_win_rate=float(np.mean(y_training)),
        calibration_home_win_rate=float(np.mean(y_calibration)),
        sealed_test_rows=sealed_test_rows,
        recent_rows=recent_rows,
        scaler_mean=mean,
        scaler_scale=scale,
        base_coefficients=coefficients,
        base_intercept=intercept,
        base_model_unchanged_during_calibration=base_unchanged,
        sealed_predictions_computed=False,
    )


def prepare_calibrated_model(
    *,
    dataset_path: Path,
    expected_dataset_sha256: str,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    expected_protocol_sha256: str = EXPECTED_PROTOCOL_SHA256,
    code_version: str | None = None,
) -> CalibrationPreparation:
    """Charge les sources vérifiées puis prépare le modèle calibré."""
    protocol = load_model_protocol(
        protocol_path,
        expected_sha256=expected_protocol_sha256,
    )
    try:
        dataset = load_training_dataset(
            dataset_path,
            expected_sha256=expected_dataset_sha256,
        )
    except (BaselineModelError, ValueError) as error:
        raise CalibratedModelError(str(error)) from error

    resolved_code_version = (
        _detect_code_version(PROJECT_DIRECTORY)
        if code_version is None
        else _validate_code_version(code_version)
    )
    return prepare_loaded_calibrated_model(
        dataset,
        protocol,
        code_version=resolved_code_version,
    )


def _validate_artifact_metadata(
    artifact: object,
) -> CalibratedModelArtifact:
    """Contrôle les métadonnées avant toute utilisation future."""
    if not isinstance(artifact, CalibratedModelArtifact):
        raise CalibratedModelError(
            "Le fichier ne contient pas un artefact calibré reconnu."
        )
    if artifact.artifact_format_version != ARTIFACT_FORMAT_VERSION:
        raise CalibratedModelError(
            "Version de format de l'artefact inattendue."
        )
    if artifact.calibrated_model_version != CALIBRATED_MODEL_VERSION:
        raise CalibratedModelError(
            "Version du modèle calibré inattendue."
        )
    if artifact.feature_columns != tuple(FEATURE_COLUMNS):
        raise CalibratedModelError(
            "Variables inattendues dans l'artefact."
        )
    if artifact.base_training_seasons != BASE_TRAINING_SEASONS:
        raise CalibratedModelError(
            "Saisons d'apprentissage inattendues dans l'artefact."
        )
    if artifact.calibration_season != CALIBRATION_SEASON:
        raise CalibratedModelError(
            "Saison de calibration inattendue dans l'artefact."
        )
    if artifact.sealed_test_seasons != SEALED_TEST_SEASONS:
        raise CalibratedModelError(
            "Saison test inattendue dans l'artefact."
        )
    if artifact.recent_seasons != SEALED_RECENT_SEASONS:
        raise CalibratedModelError(
            "Saison récente inattendue dans l'artefact."
        )
    if artifact.calibration_method != CALIBRATION_METHOD:
        raise CalibratedModelError(
            "Méthode de calibration inattendue dans l'artefact."
        )
    _normalize_sha256(
        artifact.dataset_sha256,
        field_name="SHA-256 du dataset dans l'artefact",
    )
    _normalize_sha256(
        artifact.protocol_sha256,
        field_name="SHA-256 du protocole dans l'artefact",
    )
    _validate_code_version(artifact.code_version)
    if not isinstance(
        artifact.calibrated_classifier,
        CalibratedClassifierCV,
    ):
        raise CalibratedModelError(
            "Le classifieur calibré est absent de l'artefact."
        )

    return artifact


def load_calibrated_artifact(
    artifact_path: Path,
    *,
    expected_sha256: str,
) -> CalibratedModelArtifact:
    """Relit uniquement un artefact local dont l'empreinte est connue.

    Les fichiers joblib reposent sur pickle. Cette fonction ne doit jamais
    recevoir un fichier non fiable ou téléchargé sans empreinte approuvée.
    """
    _, content, _ = _read_verified_bytes(
        artifact_path,
        expected_sha256=expected_sha256,
        description="artefact calibré",
    )
    try:
        with warnings.catch_warnings():
            # joblib 1.5.3 utilise encore une ancienne affectation de forme
            # avec NumPy 2.5 ; elle ne modifie ni les valeurs ni l'artefact.
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Setting the shape on a NumPy array has been "
                    "deprecated.*"
                ),
                category=DeprecationWarning,
                module=r"joblib\.numpy_pickle",
            )
            artifact = joblib.load(io.BytesIO(content))
    except Exception as error:
        raise CalibratedModelError(
            f"L'artefact calibré est illisible : {error}"
        ) from error

    return _validate_artifact_metadata(artifact)


def write_calibrated_artifact(
    preparation: CalibrationPreparation,
    *,
    output_path: Path,
) -> ArtifactExport:
    """Crée un joblib neuf, sans jamais remplacer un modèle existant."""
    artifact = _validate_artifact_metadata(preparation.artifact)
    path = Path(output_path).expanduser()
    if path.suffix.lower() != ".joblib":
        raise ValueError("L'artefact doit utiliser l'extension .joblib.")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path = path.resolve()
    except OSError as error:
        raise CalibratedModelError(
            f"Préparation du dossier de modèles impossible : {error}"
        ) from error

    created = False
    try:
        with resolved_path.open("xb") as destination:
            created = True
            joblib.dump(artifact, destination, compress=3)
    except FileExistsError as error:
        raise CalibratedModelError(
            f"L'artefact existe déjà et ne sera pas remplacé : "
            f"{resolved_path}."
        ) from error
    except Exception as error:
        if created:
            try:
                resolved_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise CalibratedModelError(
            f"Écriture de l'artefact impossible : {error}"
        ) from error

    try:
        content = resolved_path.read_bytes()
        artifact_sha256 = hashlib.sha256(content).hexdigest()
        reloaded = load_calibrated_artifact(
            resolved_path,
            expected_sha256=artifact_sha256,
        )
        if reloaded.dataset_sha256 != artifact.dataset_sha256:
            raise CalibratedModelError(
                "L'artefact relu ne référence plus le même dataset."
            )
        if reloaded.protocol_sha256 != artifact.protocol_sha256:
            raise CalibratedModelError(
                "L'artefact relu ne référence plus le même protocole."
            )
    except Exception as error:
        try:
            resolved_path.unlink(missing_ok=True)
        except OSError:
            pass
        if isinstance(error, CalibratedModelError):
            raise
        raise CalibratedModelError(
            f"Vérification de l'artefact impossible : {error}"
        ) from error

    return ArtifactExport(
        path=resolved_path,
        sha256=artifact_sha256,
        size_bytes=len(content),
    )


def display_preparation(
    preparation: CalibrationPreparation,
    *,
    export: ArtifactExport | None = None,
) -> None:
    """Affiche uniquement la préparation, jamais un résultat scellé."""
    artifact = preparation.artifact
    print("Préparation du modèle MLB calibré")
    print(f"Version calibrée : {artifact.calibrated_model_version}")
    print(f"Version du modèle de base : {artifact.base_model_version}")
    print(f"Version du code : {artifact.code_version}")
    print(f"Dataset : {preparation.dataset_path}")
    print(f"SHA-256 du dataset : {artifact.dataset_sha256}")
    print(f"Protocole : {preparation.protocol_path}")
    print(f"SHA-256 du protocole : {artifact.protocol_sha256}")
    print(f"scikit-learn : {artifact.sklearn_version}")
    print(f"NumPy : {artifact.numpy_version}")
    print(f"Variables : {len(artifact.feature_columns)}")
    print(
        "Apprentissage du modèle de base : saisons "
        + ", ".join(map(str, artifact.base_training_seasons))
        + f" | {preparation.base_training_rows} lignes"
    )
    print(
        f"Calibration Platt : saison {artifact.calibration_season} | "
        f"{preparation.calibration_rows} lignes"
    )
    print(
        "Taux domicile de l'apprentissage : "
        f"{100.0 * preparation.base_training_home_win_rate:.2f}%"
    )
    print(
        "Taux domicile de la calibration : "
        f"{100.0 * preparation.calibration_home_win_rate:.2f}%"
    )
    print(
        "Modèle de base inchangé pendant la calibration : "
        + (
            "OUI"
            if preparation.base_model_unchanged_during_calibration
            else "NON"
        )
    )

    print("\nCohortes toujours scellées")
    print(
        f"Saison(s) test {', '.join(map(str, artifact.sealed_test_seasons))} "
        f": {preparation.sealed_test_rows} lignes, aucune prédiction."
    )
    print(
        f"Saison(s) récente(s) {', '.join(map(str, artifact.recent_seasons))} "
        f": {preparation.recent_rows} lignes, aucune prédiction."
    )
    print(
        "Prédiction calculée sur une cohorte scellée : "
        + ("OUI" if preparation.sealed_predictions_computed else "NON")
    )

    if export is None:
        print("\nMode aperçu : aucun fichier joblib n'a été écrit.")
    else:
        print(f"\nArtefact créé et revérifié : {export.path}")
        print(f"Taille : {export.size_bytes} octets")
        print(f"SHA-256 : {export.sha256}")

    print("\nInterprétation")
    print(
        "2024 sert uniquement à ajuster la calibration Platt. Aucune "
        "métrique calculée sur cette même saison ne constitue une preuve "
        "finale."
    )
    print(
        "Le test 2025 restera fermé jusqu'à l'enregistrement définitif "
        "et à la vérification de cet artefact."
    )


def main() -> None:
    """Point d'entrée de la préparation reproductible."""
    parser = argparse.ArgumentParser(
        description=(
            "Ajuste le modèle MLB sur 2021-2023 puis Platt sur 2024, "
            "sans prédire 2025 ou 2026."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="CSV exact du dataset MLB enregistré.",
    )
    parser.add_argument(
        "--expected-dataset-sha256",
        required=True,
        help="SHA-256 attendu du CSV.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help="Protocole JSON enregistré avant le test final.",
    )
    parser.add_argument(
        "--expected-protocol-sha256",
        default=EXPECTED_PROTOCOL_SHA256,
        help="SHA-256 attendu du protocole JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Chemin .joblib à créer. Sans cette option, aucun fichier "
            "n'est écrit."
        ),
    )
    arguments = parser.parse_args()

    try:
        preparation = prepare_calibrated_model(
            dataset_path=arguments.dataset,
            expected_dataset_sha256=(
                arguments.expected_dataset_sha256
            ),
            protocol_path=arguments.protocol,
            expected_protocol_sha256=(
                arguments.expected_protocol_sha256
            ),
        )
        export = (
            write_calibrated_artifact(
                preparation,
                output_path=arguments.output,
            )
            if arguments.output is not None
            else None
        )
        display_preparation(preparation, export=export)
    except (CalibratedModelError, ValueError) as error:
        parser.exit(
            status=1,
            message=f"Échec de la calibration MLB : {error}\n",
        )


if __name__ == "__main__":
    main()
