"""Évalue une seule fois la cohorte MLB 2025 selon le protocole figé."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import hmac
import io
import json
import math
import os
import platform
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence
import uuid
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.baseline_model import ModelRow
from src.calibrated_model import CalibratedModelArtifact
from src.training_dataset import FEATURE_COLUMNS


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_PROTOCOL_PATH = (
    PROJECT_DIRECTORY
    / "evaluation_protocols"
    / "logistic_team_form_v1_platt_2025.json"
)

EVALUATION_REPORT_VERSION = 1
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
EXPECTED_ARTIFACT_CODE_VERSION = (
    "1d13f8f361de0865f722a8560f1efda8e95badbc"
)
REQUIRED_ANCESTOR_REVISIONS = (
    EXPECTED_ARTIFACT_CODE_VERSION,
    "5719eb6",  # protocole du modèle
    "6e5d439",  # manifeste de l'artefact
    "4fbee75",  # protocole d'évaluation 2025
)

SEALED_SEASON = 2025
EXCLUDED_RECENT_SEASON = 2026
EXPECTED_SEALED_ROWS = 2276
EXPECTED_DATASET_ROWS = 13253
EXPECTED_FIRST_DATE = date(2025, 4, 7)
EXPECTED_LAST_DATE = date(2025, 9, 28)
EXPECTED_CALENDAR_DAYS = 175

BASELINE_HOME_WINS = 3623
BASELINE_GAMES = 6815
BASELINE_PROBABILITY = BASELINE_HOME_WINS / BASELINE_GAMES

LOG_LOSS_EPSILON = 1e-15
BOOTSTRAP_BLOCK_DAYS = 7
BOOTSTRAP_REPLICATIONS = 5000
BOOTSTRAP_SEED = 42
PRIMARY_GAIN_THRESHOLD = 0.005
CALIBRATION_IN_LARGE_LIMIT = 0.02
CALIBRATION_SLOPE_MINIMUM = 0.8
CALIBRATION_SLOPE_MAXIMUM = 1.2

VERDICT_VALIDATED = "SIGNAL_VALIDE"
VERDICT_PROMISING = "PROMETTEUR_MAIS_INCONCLUSIF"
VERDICT_REJECTED = "REJETE_V1"
VERDICT_BAD_PROBABILITIES = (
    "SIGNAL_PRESENT_MAIS_PROBABILITES_NON_EXPLOITABLES"
)

PREDICTION_COLUMNS = (
    "game_id",
    "season",
    "official_date",
    "feature_as_of_date",
    "away_team_id",
    "home_team_id",
    "away_max_source_date",
    "home_max_source_date",
    "home_win",
    "model_home_win_probability",
    "baseline_home_win_probability",
    "model_log_loss",
    "baseline_log_loss",
)

_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


class SealedEvaluationError(RuntimeError):
    """Erreur empêchant une ouverture fiable et auditable de 2025."""


@dataclass(frozen=True, slots=True)
class EvaluationConfiguration:
    """Entrées vérifiées sans désérialiser le modèle ni parser le CSV."""

    project_directory: Path
    evaluation_protocol_path: Path
    evaluation_protocol_sha256: str
    evaluation_protocol: dict[str, Any]
    artifact_manifest_path: Path
    artifact_manifest_sha256: str
    artifact_manifest: dict[str, Any]
    model_protocol_path: Path
    model_protocol_sha256: str
    model_protocol: dict[str, Any]
    artifact_path: Path
    artifact_sha256: str
    artifact_size_bytes: int
    dataset_path: Path
    dataset_sha256: str
    dataset_size_bytes: int
    output_directory: Path


@dataclass(frozen=True, slots=True)
class EvaluationPreview:
    """Aperçu qui atteste qu'aucun résultat scellé n'a été calculé."""

    evaluation_protocol_sha256: str
    artifact_manifest_sha256: str
    model_protocol_sha256: str
    artifact_sha256: str
    artifact_size_bytes: int
    dataset_sha256: str
    dataset_size_bytes: int
    sealed_season: int
    expected_sealed_rows: int
    predictions_computed: bool
    metrics_computed: bool
    artifact_deserialized: bool
    dataset_parsed: bool


@dataclass(frozen=True, slots=True)
class ProbabilityMetrics:
    """Mesures binaires calculées à pleine précision."""

    samples: int
    observed_home_win_rate: float
    mean_predicted_home_win_probability: float
    log_loss: float
    brier_score: float
    calibration_in_the_large: float
    roc_auc: float
    accuracy_at_0_5: float


@dataclass(frozen=True, slots=True)
class CalibrationDiagnostics:
    """Régression diagnostique logit(y) sur logit(p), non prédictive."""

    intercept: float
    slope: float


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Intervalle percentile du gain relatif de log loss."""

    replications: int
    block_length_days: int
    random_seed: int
    calendar_days: int
    blocks_per_replicate: int
    lower: float
    upper: float
    statistics_sha256: str


@dataclass(frozen=True, slots=True)
class VerdictResult:
    """Verdict et détail des cinq conditions préenregistrées."""

    label: str
    relative_gain_meets_threshold: bool
    bootstrap_lower_bound_above_zero: bool
    model_brier_not_worse: bool
    calibration_in_the_large_within_limit: bool
    calibration_slope_within_range: bool
    primary_supported: bool
    probability_guardrails_pass: bool
    failed_conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RowPrediction:
    """Prédiction et pertes d'un match, conservées pour l'audit."""

    row: ModelRow
    model_probability: float
    baseline_probability: float
    model_log_loss: float
    baseline_log_loss: float


@dataclass(frozen=True, slots=True)
class SealedEvaluationResult:
    """Résultat scientifique complet avant toute écriture disque."""

    season: int
    rows: int
    first_official_date: date
    last_official_date: date
    baseline_probability: float
    model_metrics: ProbabilityMetrics
    baseline_metrics: ProbabilityMetrics
    calibration: CalibrationDiagnostics
    relative_log_loss_improvement: float
    bootstrap: BootstrapInterval
    verdict: VerdictResult
    predictions: tuple[RowPrediction, ...]


@dataclass(frozen=True, slots=True)
class EvaluationExport:
    """Deux sorties écrites une fois puis relues et vérifiées."""

    directory: Path
    predictions_path: Path
    predictions_sha256: str
    predictions_size_bytes: int
    report_path: Path
    report_sha256: str
    report_size_bytes: int


@dataclass(frozen=True, slots=True)
class OpenedSealedEvaluation:
    """Ouverture terminée seulement après vérification des deux sorties."""

    evaluation_code_commit: str
    result: SealedEvaluationResult
    export: EvaluationExport


def _normalize_sha256(value: str, *, field_name: str) -> str:
    """Valide une empreinte exigée avant lecture d'une source."""
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
    """Vérifie l'empreinte complète avant tout parsing ou chargement."""
    expected = _normalize_sha256(
        expected_sha256,
        field_name=f"SHA-256 attendu de {description}",
    )
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SealedEvaluationError(
            f"{description.capitalize()} introuvable : {path}."
        ) from error
    if not resolved.is_file():
        raise SealedEvaluationError(
            f"Le chemin de {description} n'est pas un fichier : "
            f"{resolved}."
        )
    try:
        content = resolved.read_bytes()
    except OSError as error:
        raise SealedEvaluationError(
            f"Lecture de {description} impossible : {error}"
        ) from error
    if not content:
        raise SealedEvaluationError(f"{description.capitalize()} vide.")
    actual = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise SealedEvaluationError(
            f"SHA-256 différent pour {description} : attendu {expected}, "
            f"obtenu {actual}."
        )
    return resolved, content, actual


def _parse_json_object(content: bytes, *, description: str) -> dict[str, Any]:
    """Décode un objet JSON après sa vérification cryptographique."""
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealedEvaluationError(
            f"{description.capitalize()} n'est pas un JSON valide."
        ) from error
    if not isinstance(payload, dict):
        raise SealedEvaluationError(
            f"{description.capitalize()} doit contenir un objet JSON."
        )
    return payload


def _require_mapping(
    parent: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> dict[str, Any]:
    """Exige un sous-objet JSON nommé."""
    value = parent.get(key)
    if not isinstance(value, dict):
        raise SealedEvaluationError(
            f"Le champ {context}.{key} doit être un objet."
        )
    return value


def _require_string(
    parent: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> str:
    """Exige une chaîne JSON non vide."""
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise SealedEvaluationError(
            f"Le champ {context}.{key} doit être une chaîne non vide."
        )
    return value


def _safe_project_path(
    project_directory: Path,
    raw_relative_path: str,
    *,
    description: str,
) -> Path:
    """Refuse tout chemin absolu ou sortant du projet."""
    pure = PurePosixPath(raw_relative_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise SealedEvaluationError(
            f"Chemin dangereux pour {description} : {raw_relative_path}."
        )
    project = project_directory.expanduser().resolve()
    candidate = project.joinpath(*pure.parts).resolve()
    if not candidate.is_relative_to(project):
        raise SealedEvaluationError(
            f"Le chemin de {description} sort du projet."
        )
    return candidate


def _assert_equal(
    actual: object,
    expected: object,
    *,
    description: str,
) -> None:
    """Produit une erreur de contrat explicite."""
    if actual != expected:
        raise SealedEvaluationError(
            f"Contrat inattendu pour {description} : "
            f"attendu {expected!r}, obtenu {actual!r}."
        )


def _require_int(
    parent: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> int:
    """Exige un entier JSON, sans accepter les booléens."""
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SealedEvaluationError(
            f"Le champ {context}.{key} doit être un entier."
        )
    return value


def _validate_configuration_payloads(
    evaluation: Mapping[str, Any],
    manifest: Mapping[str, Any],
    model_protocol: Mapping[str, Any],
) -> None:
    """Contrôle les règles exécutables figées dans les trois JSON."""
    _assert_equal(
        evaluation.get("evaluation_protocol_version"),
        1,
        description="version du protocole d'évaluation",
    )
    _assert_equal(
        evaluation.get("status"),
        "REGISTERED_BEFORE_SEALED_TEST",
        description="statut du protocole d'évaluation",
    )
    _assert_equal(
        evaluation.get("sealed_test_opened_at_registration"),
        False,
        description="état initial du test scellé",
    )

    inputs = _require_mapping(evaluation, "inputs", context="evaluation")
    artifact_input = _require_mapping(
        inputs,
        "model_artifact",
        context="evaluation.inputs",
    )
    manifest_input = _require_mapping(
        inputs,
        "artifact_manifest",
        context="evaluation.inputs",
    )
    protocol_input = _require_mapping(
        inputs,
        "model_protocol",
        context="evaluation.inputs",
    )
    dataset_input = _require_mapping(
        inputs,
        "dataset",
        context="evaluation.inputs",
    )
    _assert_equal(
        _normalize_sha256(
            _require_string(
                artifact_input,
                "sha256",
                context="evaluation.inputs.model_artifact",
            ),
            field_name="SHA-256 de l'artefact",
        ),
        EXPECTED_ARTIFACT_SHA256,
        description="SHA-256 de l'artefact",
    )
    _assert_equal(
        _require_int(
            artifact_input,
            "size_bytes",
            context="evaluation.inputs.model_artifact",
        ),
        1589,
        description="taille de l'artefact",
    )
    _assert_equal(
        artifact_input.get("code_version"),
        EXPECTED_ARTIFACT_CODE_VERSION,
        description="commit du code ayant créé l'artefact",
    )
    _assert_equal(
        _normalize_sha256(
            _require_string(
                manifest_input,
                "sha256",
                context="evaluation.inputs.artifact_manifest",
            ),
            field_name="SHA-256 du manifeste",
        ),
        EXPECTED_ARTIFACT_MANIFEST_SHA256,
        description="SHA-256 du manifeste",
    )
    _assert_equal(
        _normalize_sha256(
            _require_string(
                protocol_input,
                "sha256",
                context="evaluation.inputs.model_protocol",
            ),
            field_name="SHA-256 du protocole du modèle",
        ),
        EXPECTED_MODEL_PROTOCOL_SHA256,
        description="SHA-256 du protocole du modèle",
    )
    _assert_equal(
        _normalize_sha256(
            _require_string(
                dataset_input,
                "sha256",
                context="evaluation.inputs.dataset",
            ),
            field_name="SHA-256 du dataset",
        ),
        EXPECTED_DATASET_SHA256,
        description="SHA-256 du dataset",
    )
    _assert_equal(
        _require_int(
            dataset_input,
            "row_count",
            context="evaluation.inputs.dataset",
        ),
        EXPECTED_DATASET_ROWS,
        description="nombre total de lignes du dataset",
    )

    cohort = _require_mapping(
        evaluation,
        "sealed_cohort",
        context="evaluation",
    )
    expected_cohort = {
        "season": SEALED_SEASON,
        "expected_rows": EXPECTED_SEALED_ROWS,
        "expected_first_official_date": EXPECTED_FIRST_DATE.isoformat(),
        "expected_last_official_date": EXPECTED_LAST_DATE.isoformat(),
        "expected_calendar_days_inclusive": EXPECTED_CALENDAR_DAYS,
        "target": "home_win",
        "positive_class": 1,
        "selection_rule": "season == 2025",
        "recent_season_excluded": EXCLUDED_RECENT_SEASON,
    }
    for key, expected in expected_cohort.items():
        _assert_equal(
            cohort.get(key),
            expected,
            description=f"cohorte scellée : {key}",
        )

    model_use = _require_mapping(
        evaluation,
        "model_use",
        context="evaluation",
    )
    _assert_equal(
        model_use.get("expected_classes"),
        [0, 1],
        description="classes du modèle",
    )
    for forbidden_fit in (
        "fit_allowed",
        "partial_fit_allowed",
        "recalibration_allowed",
        "threshold_tuning_allowed",
        "feature_selection_allowed",
    ):
        _assert_equal(
            model_use.get(forbidden_fit),
            False,
            description=f"interdiction {forbidden_fit}",
        )

    baseline = _require_mapping(
        evaluation,
        "reference_baseline",
        context="evaluation",
    )
    _assert_equal(
        baseline.get("home_wins"),
        BASELINE_HOME_WINS,
        description="victoires domicile de la référence",
    )
    _assert_equal(
        baseline.get("games"),
        BASELINE_GAMES,
        description="matchs de la référence",
    )
    _assert_equal(
        baseline.get("training_seasons"),
        [2021, 2022, 2023],
        description="saisons de la référence",
    )
    _assert_equal(
        baseline.get("reestimated_inside_bootstrap"),
        False,
        description="non-réestimation de la référence",
    )

    handling = _require_mapping(
        evaluation,
        "probability_handling",
        context="evaluation",
    )
    _assert_equal(
        handling.get("log_loss_clip_epsilon"),
        LOG_LOSS_EPSILON,
        description="epsilon de log loss",
    )
    _assert_equal(
        handling.get("classification_threshold"),
        0.5,
        description="seuil de classification",
    )

    primary = _require_mapping(
        evaluation,
        "primary_comparison",
        context="evaluation",
    )
    _assert_equal(
        primary.get("threshold"),
        PRIMARY_GAIN_THRESHOLD,
        description="seuil principal",
    )
    _assert_equal(
        primary.get("threshold_operator"),
        ">=",
        description="opérateur du seuil principal",
    )

    uncertainty = _require_mapping(
        evaluation,
        "uncertainty",
        context="evaluation",
    )
    expected_uncertainty = {
        "method": "PAIRED_MOVING_CALENDAR_BLOCK_BOOTSTRAP",
        "expected_calendar_axis_days": EXPECTED_CALENDAR_DAYS,
        "block_length_days": BOOTSTRAP_BLOCK_DAYS,
        "replications": BOOTSTRAP_REPLICATIONS,
        "random_seed": BOOTSTRAP_SEED,
        "expected_complete_blocks_per_replicate": 25,
        "expected_final_partial_block_days": 0,
        "quantile_method": "linear",
        "baseline_probability_reestimated": False,
    }
    for key, expected in expected_uncertainty.items():
        _assert_equal(
            uncertainty.get(key),
            expected,
            description=f"bootstrap : {key}",
        )

    opening = _require_mapping(
        evaluation,
        "opening_control",
        context="evaluation",
    )
    _assert_equal(
        opening.get("explicit_cli_flag_required"),
        "--open-sealed-test",
        description="drapeau d'ouverture",
    )
    _assert_equal(
        opening.get("open_only_once"),
        True,
        description="ouverture unique",
    )

    outputs = _require_mapping(
        evaluation,
        "outputs",
        context="evaluation",
    )
    _assert_equal(
        outputs.get("directory"),
        "evaluation_results/logistic_team_form_v1_platt_2025",
        description="dossier de sortie",
    )
    _assert_equal(
        outputs.get("write_mode"),
        "EXCLUSIVE_NO_OVERWRITE",
        description="mode d'écriture",
    )

    _assert_equal(
        manifest.get("manifest_version"),
        1,
        description="version du manifeste",
    )
    _assert_equal(
        manifest.get("status"),
        "FROZEN_BEFORE_SEALED_TEST",
        description="statut du manifeste",
    )
    manifest_artifact = _require_mapping(
        manifest,
        "artifact",
        context="manifest",
    )
    manifest_dataset = _require_mapping(
        manifest,
        "dataset",
        context="manifest",
    )
    manifest_model = _require_mapping(
        manifest,
        "model",
        context="manifest",
    )
    manifest_protocol = _require_mapping(
        manifest,
        "protocol",
        context="manifest",
    )
    chronology = _require_mapping(
        manifest,
        "chronology",
        context="manifest",
    )
    _assert_equal(
        manifest_artifact.get("sha256"),
        EXPECTED_ARTIFACT_SHA256,
        description="artefact du manifeste",
    )
    _assert_equal(
        manifest_artifact.get("size_bytes"),
        1589,
        description="taille d'artefact du manifeste",
    )
    _assert_equal(
        manifest_artifact.get("code_version"),
        EXPECTED_ARTIFACT_CODE_VERSION,
        description="code de l'artefact du manifeste",
    )
    _assert_equal(
        manifest_dataset.get("sha256"),
        EXPECTED_DATASET_SHA256,
        description="dataset du manifeste",
    )
    _assert_equal(
        manifest_dataset.get("size_bytes"),
        1680593,
        description="taille du dataset du manifeste",
    )
    _assert_equal(
        manifest_dataset.get("row_count"),
        EXPECTED_DATASET_ROWS,
        description="lignes du dataset du manifeste",
    )
    _assert_equal(
        manifest_model.get("feature_columns"),
        list(FEATURE_COLUMNS),
        description="ordre des variables du manifeste",
    )
    _assert_equal(
        manifest_protocol.get("sha256"),
        EXPECTED_MODEL_PROTOCOL_SHA256,
        description="protocole référencé par le manifeste",
    )
    _assert_equal(
        chronology.get("sealed_predictions_computed"),
        False,
        description="absence initiale de prédictions scellées",
    )

    _assert_equal(
        model_protocol.get("protocol_version"),
        1,
        description="version du protocole du modèle",
    )
    _assert_equal(
        model_protocol.get("status"),
        "REGISTERED_BEFORE_SEALED_TEST",
        description="statut du protocole du modèle",
    )
    _assert_equal(
        model_protocol.get("features"),
        list(FEATURE_COLUMNS),
        description="ordre des variables du protocole",
    )
    protocol_dataset = _require_mapping(
        model_protocol,
        "dataset",
        context="model_protocol",
    )
    _assert_equal(
        protocol_dataset.get("sha256"),
        EXPECTED_DATASET_SHA256,
        description="dataset du protocole du modèle",
    )
    protocol_chronology = _require_mapping(
        model_protocol,
        "chronology",
        context="model_protocol",
    )
    _assert_equal(
        protocol_chronology.get("sealed_test_season"),
        SEALED_SEASON,
        description="saison scellée du protocole du modèle",
    )
    _assert_equal(
        protocol_chronology.get("recent_retrospective_season"),
        EXCLUDED_RECENT_SEASON,
        description="saison récente exclue",
    )


def load_evaluation_configuration(
    evaluation_protocol_path: Path = DEFAULT_EVALUATION_PROTOCOL_PATH,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
    expected_evaluation_protocol_sha256: str = (
        EXPECTED_EVALUATION_PROTOCOL_SHA256
    ),
) -> EvaluationConfiguration:
    """Lit uniquement les trois contrats JSON publics et vérifiés."""
    project = Path(project_directory).expanduser().resolve()
    if not project.is_dir():
        raise SealedEvaluationError(
            f"Dossier du projet introuvable : {project}."
        )
    evaluation_path, evaluation_bytes, evaluation_sha = (
        _read_verified_bytes(
            Path(evaluation_protocol_path),
            expected_sha256=expected_evaluation_protocol_sha256,
            description="protocole d'évaluation",
        )
    )
    evaluation = _parse_json_object(
        evaluation_bytes,
        description="protocole d'évaluation",
    )
    inputs = _require_mapping(evaluation, "inputs", context="evaluation")
    manifest_input = _require_mapping(
        inputs,
        "artifact_manifest",
        context="evaluation.inputs",
    )
    protocol_input = _require_mapping(
        inputs,
        "model_protocol",
        context="evaluation.inputs",
    )
    artifact_input = _require_mapping(
        inputs,
        "model_artifact",
        context="evaluation.inputs",
    )
    dataset_input = _require_mapping(
        inputs,
        "dataset",
        context="evaluation.inputs",
    )
    outputs = _require_mapping(
        evaluation,
        "outputs",
        context="evaluation",
    )

    manifest_path = _safe_project_path(
        project,
        _require_string(
            manifest_input,
            "path",
            context="evaluation.inputs.artifact_manifest",
        ),
        description="manifeste de l'artefact",
    )
    manifest_path, manifest_bytes, manifest_sha = _read_verified_bytes(
        manifest_path,
        expected_sha256=EXPECTED_ARTIFACT_MANIFEST_SHA256,
        description="manifeste de l'artefact",
    )
    manifest = _parse_json_object(
        manifest_bytes,
        description="manifeste de l'artefact",
    )

    model_protocol_path = _safe_project_path(
        project,
        _require_string(
            protocol_input,
            "path",
            context="evaluation.inputs.model_protocol",
        ),
        description="protocole du modèle",
    )
    model_protocol_path, model_protocol_bytes, model_protocol_sha = (
        _read_verified_bytes(
            model_protocol_path,
            expected_sha256=EXPECTED_MODEL_PROTOCOL_SHA256,
            description="protocole du modèle",
        )
    )
    model_protocol = _parse_json_object(
        model_protocol_bytes,
        description="protocole du modèle",
    )
    _validate_configuration_payloads(
        evaluation,
        manifest,
        model_protocol,
    )

    manifest_dataset = _require_mapping(
        manifest,
        "dataset",
        context="manifest",
    )
    artifact_path = _safe_project_path(
        project,
        _require_string(
            artifact_input,
            "path",
            context="evaluation.inputs.model_artifact",
        ),
        description="artefact calibré",
    )
    dataset_path = _safe_project_path(
        project,
        _require_string(
            dataset_input,
            "path",
            context="evaluation.inputs.dataset",
        ),
        description="dataset",
    )
    output_directory = _safe_project_path(
        project,
        _require_string(
            outputs,
            "directory",
            context="evaluation.outputs",
        ),
        description="dossier de résultats",
    )
    return EvaluationConfiguration(
        project_directory=project,
        evaluation_protocol_path=evaluation_path,
        evaluation_protocol_sha256=evaluation_sha,
        evaluation_protocol=evaluation,
        artifact_manifest_path=manifest_path,
        artifact_manifest_sha256=manifest_sha,
        artifact_manifest=manifest,
        model_protocol_path=model_protocol_path,
        model_protocol_sha256=model_protocol_sha,
        model_protocol=model_protocol,
        artifact_path=artifact_path,
        artifact_sha256=EXPECTED_ARTIFACT_SHA256,
        artifact_size_bytes=1589,
        dataset_path=dataset_path,
        dataset_sha256=EXPECTED_DATASET_SHA256,
        dataset_size_bytes=_require_int(
            manifest_dataset,
            "size_bytes",
            context="manifest.dataset",
        ),
        output_directory=output_directory,
    )


def preview_sealed_evaluation(
    evaluation_protocol_path: Path = DEFAULT_EVALUATION_PROTOCOL_PATH,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> EvaluationPreview:
    """Prépare l'ouverture sans toucher au CSV, au joblib ou à 2025."""
    configuration = load_evaluation_configuration(
        evaluation_protocol_path,
        project_directory=project_directory,
    )
    return EvaluationPreview(
        evaluation_protocol_sha256=(
            configuration.evaluation_protocol_sha256
        ),
        artifact_manifest_sha256=configuration.artifact_manifest_sha256,
        model_protocol_sha256=configuration.model_protocol_sha256,
        artifact_sha256=configuration.artifact_sha256,
        artifact_size_bytes=configuration.artifact_size_bytes,
        dataset_sha256=configuration.dataset_sha256,
        dataset_size_bytes=configuration.dataset_size_bytes,
        sealed_season=SEALED_SEASON,
        expected_sealed_rows=EXPECTED_SEALED_ROWS,
        predictions_computed=False,
        metrics_computed=False,
        artifact_deserialized=False,
        dataset_parsed=False,
    )


def _parse_int(value: object, *, field_name: str) -> int:
    """Convertit un entier CSV strict."""
    if not isinstance(value, str) or not value.strip():
        raise SealedEvaluationError(f"Valeur absente pour {field_name}.")
    try:
        parsed = int(value)
    except ValueError as error:
        raise SealedEvaluationError(
            f"Entier invalide pour {field_name} : {value!r}."
        ) from error
    if str(parsed) != value.strip():
        raise SealedEvaluationError(
            f"Écriture non canonique pour {field_name} : {value!r}."
        )
    return parsed


def _parse_date(value: object, *, field_name: str) -> date:
    """Convertit une date ISO CSV stricte."""
    if not isinstance(value, str):
        raise SealedEvaluationError(f"Date absente pour {field_name}.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise SealedEvaluationError(
            f"Date invalide pour {field_name} : {value!r}."
        ) from error
    if parsed.isoformat() != value:
        raise SealedEvaluationError(
            f"Date non canonique pour {field_name} : {value!r}."
        )
    return parsed


def _parse_float(value: object, *, field_name: str) -> float:
    """Convertit une variable numérique finie."""
    if not isinstance(value, str) or not value.strip():
        raise SealedEvaluationError(f"Valeur absente pour {field_name}.")
    try:
        parsed = float(value)
    except ValueError as error:
        raise SealedEvaluationError(
            f"Nombre invalide pour {field_name} : {value!r}."
        ) from error
    if not math.isfinite(parsed):
        raise SealedEvaluationError(
            f"Nombre non fini pour {field_name}."
        )
    return parsed


_EXPECTED_CSV_COLUMNS = (
    "game_id",
    "season",
    "official_date",
    "feature_as_of_date",
    "away_team_id",
    "home_team_id",
    "away_max_source_date",
    "home_max_source_date",
    *FEATURE_COLUMNS,
    "home_win",
)


def _load_sealed_rows_from_verified_csv(
    content: bytes,
) -> tuple[ModelRow, ...]:
    """Parse seulement 2025; les variables de 2026 restent intouchées."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SealedEvaluationError("Le dataset n'est pas en UTF-8.") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != _EXPECTED_CSV_COLUMNS:
        raise SealedEvaluationError(
            "Le schéma du dataset ne correspond pas au protocole."
        )
    sealed_rows: list[ModelRow] = []
    total_rows = 0
    try:
        for csv_row in reader:
            total_rows += 1
            if None in csv_row:
                raise SealedEvaluationError(
                    f"Colonnes supplémentaires à la ligne {reader.line_num}."
                )
            season = _parse_int(
                csv_row["season"],
                field_name=f"season ligne {reader.line_num}",
            )
            if season != SEALED_SEASON:
                continue
            features = tuple(
                _parse_float(
                    csv_row[column],
                    field_name=f"{column} ligne {reader.line_num}",
                )
                for column in FEATURE_COLUMNS
            )
            sealed_rows.append(
                ModelRow(
                    game_id=_parse_int(
                        csv_row["game_id"],
                        field_name=f"game_id ligne {reader.line_num}",
                    ),
                    season=season,
                    official_date=_parse_date(
                        csv_row["official_date"],
                        field_name=(
                            f"official_date ligne {reader.line_num}"
                        ),
                    ),
                    feature_as_of_date=_parse_date(
                        csv_row["feature_as_of_date"],
                        field_name=(
                            f"feature_as_of_date ligne {reader.line_num}"
                        ),
                    ),
                    away_team_id=_parse_int(
                        csv_row["away_team_id"],
                        field_name=(
                            f"away_team_id ligne {reader.line_num}"
                        ),
                    ),
                    home_team_id=_parse_int(
                        csv_row["home_team_id"],
                        field_name=(
                            f"home_team_id ligne {reader.line_num}"
                        ),
                    ),
                    away_max_source_date=_parse_date(
                        csv_row["away_max_source_date"],
                        field_name=(
                            "away_max_source_date ligne "
                            f"{reader.line_num}"
                        ),
                    ),
                    home_max_source_date=_parse_date(
                        csv_row["home_max_source_date"],
                        field_name=(
                            "home_max_source_date ligne "
                            f"{reader.line_num}"
                        ),
                    ),
                    features=features,
                    home_win=_parse_int(
                        csv_row["home_win"],
                        field_name=f"home_win ligne {reader.line_num}",
                    ),
                )
            )
    except csv.Error as error:
        raise SealedEvaluationError(
            f"CSV invalide : {error}."
        ) from error
    if total_rows != EXPECTED_DATASET_ROWS:
        raise SealedEvaluationError(
            "Nombre total de lignes inattendu dans le dataset : "
            f"attendu {EXPECTED_DATASET_ROWS}, obtenu {total_rows}."
        )
    return validate_sealed_rows(sealed_rows)


def validate_sealed_rows(
    rows: Sequence[ModelRow],
    *,
    expected_rows: int = EXPECTED_SEALED_ROWS,
    expected_first_date: date = EXPECTED_FIRST_DATE,
    expected_last_date: date = EXPECTED_LAST_DATE,
) -> tuple[ModelRow, ...]:
    """Refuse toute cohorte différente de celle qui a été enregistrée."""
    ordered = tuple(sorted(rows, key=lambda row: (row.official_date, row.game_id)))
    if len(ordered) != expected_rows:
        raise SealedEvaluationError(
            f"Cohorte 2025 incomplète : attendu {expected_rows} matchs, "
            f"obtenu {len(ordered)}."
        )
    if not ordered:
        raise SealedEvaluationError("La cohorte scellée est vide.")
    if ordered[0].official_date != expected_first_date:
        raise SealedEvaluationError("Première date 2025 inattendue.")
    if ordered[-1].official_date != expected_last_date:
        raise SealedEvaluationError("Dernière date 2025 inattendue.")
    if (
        expected_last_date - expected_first_date
    ).days + 1 != EXPECTED_CALENDAR_DAYS:
        raise SealedEvaluationError("Axe calendrier 2025 inattendu.")
    game_ids: set[int] = set()
    targets: set[int] = set()
    for row in ordered:
        if row.season != SEALED_SEASON:
            raise SealedEvaluationError(
                f"Saison interdite dans l'évaluation : {row.season}."
            )
        if row.game_id in game_ids:
            raise SealedEvaluationError(
                f"game_id dupliqué dans 2025 : {row.game_id}."
            )
        game_ids.add(row.game_id)
        if row.home_win not in {0, 1}:
            raise SealedEvaluationError(
                f"Cible non binaire pour le match {row.game_id}."
            )
        targets.add(row.home_win)
        if row.feature_as_of_date != row.official_date - timedelta(days=1):
            raise SealedEvaluationError(
                f"Date J-1 invalide pour le match {row.game_id}."
            )
        if (
            row.away_max_source_date >= row.official_date
            or row.home_max_source_date >= row.official_date
        ):
            raise SealedEvaluationError(
                f"Fuite temporelle pour le match {row.game_id}."
            )
        if len(row.features) != len(FEATURE_COLUMNS):
            raise SealedEvaluationError(
                f"Nombre de variables invalide pour le match {row.game_id}."
            )
        if not all(math.isfinite(value) for value in row.features):
            raise SealedEvaluationError(
                f"Variable non finie pour le match {row.game_id}."
            )
        if row.away_team_id == row.home_team_id:
            raise SealedEvaluationError(
                f"Équipe opposée à elle-même au match {row.game_id}."
            )
    if targets != {0, 1}:
        raise SealedEvaluationError(
            "La cohorte doit contenir les deux classes de la cible."
        )
    return ordered


def _validated_binary_arrays(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    require_both_classes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise deux vecteurs binaires sans tolérer NaN ou infini."""
    y = np.asarray(targets)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape:
        raise SealedEvaluationError(
            "Cibles et probabilités doivent être deux vecteurs de même taille."
        )
    if y.size == 0:
        raise SealedEvaluationError("Aucune observation à évaluer.")
    if not np.all(np.isfinite(p)):
        raise SealedEvaluationError("Probabilité non finie détectée.")
    if np.any((p < 0.0) | (p > 1.0)):
        raise SealedEvaluationError(
            "Les probabilités doivent appartenir à [0, 1]."
        )
    try:
        y_float = y.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise SealedEvaluationError("Cible binaire invalide.") from error
    if not np.all(np.isfinite(y_float)):
        raise SealedEvaluationError("Cible non finie détectée.")
    if not np.all((y_float == 0.0) | (y_float == 1.0)):
        raise SealedEvaluationError("La cible doit être strictement binaire.")
    y_int = y_float.astype(np.int8)
    if require_both_classes and set(y_int.tolist()) != {0, 1}:
        raise SealedEvaluationError(
            "Les deux classes sont indispensables à l'évaluation."
        )
    return y_int, p


def _individual_log_losses(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Calcule les pertes individuelles avec le clipping préenregistré."""
    clipped = np.clip(
        probabilities,
        LOG_LOSS_EPSILON,
        1.0 - LOG_LOSS_EPSILON,
    )
    return -(
        targets * np.log(clipped)
        + (1 - targets) * np.log1p(-clipped)
    )


def compute_probability_metrics(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> ProbabilityMetrics:
    """Applique exactement les métriques probabilistes enregistrées."""
    y, p = _validated_binary_arrays(targets, probabilities)
    losses = _individual_log_losses(y, p)
    values = {
        "observed_home_win_rate": float(np.mean(y)),
        "mean_predicted_home_win_probability": float(np.mean(p)),
        "log_loss": float(np.mean(losses)),
        "brier_score": float(np.mean(np.square(p - y))),
        "roc_auc": float(roc_auc_score(y, p)),
        "accuracy_at_0_5": float(np.mean((p >= 0.5) == y)),
    }
    values["calibration_in_the_large"] = (
        values["observed_home_win_rate"]
        - values["mean_predicted_home_win_probability"]
    )
    if not all(math.isfinite(value) for value in values.values()):
        raise SealedEvaluationError("Métrique non finie détectée.")
    return ProbabilityMetrics(samples=int(y.size), **values)


def fit_calibration_diagnostics(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> CalibrationDiagnostics:
    """Effectue l'unique fit autorisé : le diagnostic logit(p)."""
    y, p = _validated_binary_arrays(targets, probabilities)
    clipped = np.clip(p, LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    diagnostic = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=10000,
        tol=1e-12,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            diagnostic.fit(logits, y)
    except ConvergenceWarning as error:
        raise SealedEvaluationError(
            "La régression diagnostique de calibration n'a pas convergé."
        ) from error
    except (TypeError, ValueError) as error:
        raise SealedEvaluationError(
            f"Diagnostic de calibration impossible : {error}"
        ) from error
    intercept = float(diagnostic.intercept_[0])
    slope = float(diagnostic.coef_[0, 0])
    if not math.isfinite(intercept) or not math.isfinite(slope):
        raise SealedEvaluationError(
            "Diagnostic de calibration non fini."
        )
    return CalibrationDiagnostics(intercept=intercept, slope=slope)


def paired_calendar_block_bootstrap(
    official_dates: Sequence[date],
    model_losses: Sequence[float] | np.ndarray,
    baseline_losses: Sequence[float] | np.ndarray,
    *,
    first_date: date,
    last_date: date,
    replications: int = BOOTSTRAP_REPLICATIONS,
    block_length_days: int = BOOTSTRAP_BLOCK_DAYS,
    random_seed: int = BOOTSTRAP_SEED,
) -> BootstrapInterval:
    """Bootstrap apparié par blocs complets de jours calendaires."""
    dates = tuple(official_dates)
    model = np.asarray(model_losses, dtype=np.float64)
    baseline = np.asarray(baseline_losses, dtype=np.float64)
    if (
        model.ndim != 1
        or baseline.ndim != 1
        or model.shape != baseline.shape
        or model.size != len(dates)
        or model.size == 0
    ):
        raise SealedEvaluationError(
            "Entrées incompatibles pour le bootstrap apparié."
        )
    if not np.all(np.isfinite(model)) or not np.all(np.isfinite(baseline)):
        raise SealedEvaluationError("Perte non finie dans le bootstrap.")
    if np.any(baseline <= 0.0):
        raise SealedEvaluationError(
            "La log loss de référence doit être strictement positive."
        )
    if isinstance(replications, bool) or replications <= 0:
        raise SealedEvaluationError(
            "Le nombre de réplications doit être positif."
        )
    if isinstance(block_length_days, bool) or block_length_days <= 0:
        raise SealedEvaluationError(
            "La longueur de bloc doit être positive."
        )
    if last_date < first_date:
        raise SealedEvaluationError("Axe calendrier inversé.")
    calendar_days = (last_date - first_date).days + 1
    if block_length_days > calendar_days:
        raise SealedEvaluationError(
            "La longueur de bloc dépasse l'axe calendrier."
        )
    daily_model_sum = np.zeros(calendar_days, dtype=np.float64)
    daily_baseline_sum = np.zeros(calendar_days, dtype=np.float64)
    daily_count = np.zeros(calendar_days, dtype=np.int64)
    for game_date, model_loss, baseline_loss in zip(
        dates,
        model,
        baseline,
        strict=True,
    ):
        day_index = (game_date - first_date).days
        if day_index < 0 or day_index >= calendar_days:
            raise SealedEvaluationError(
                "Match situé hors de l'axe calendrier du bootstrap."
            )
        daily_model_sum[day_index] += model_loss
        daily_baseline_sum[day_index] += baseline_loss
        daily_count[day_index] += 1

    full_blocks, remainder = divmod(calendar_days, block_length_days)
    blocks_per_replicate = full_blocks + (1 if remainder else 0)
    possible_full_starts = calendar_days - block_length_days + 1
    possible_partial_starts = (
        calendar_days - remainder + 1 if remainder else 0
    )
    rng = np.random.Generator(np.random.PCG64(random_seed))
    statistics = np.empty(replications, dtype=np.float64)
    for replicate_index in range(replications):
        model_sum = 0.0
        baseline_sum = 0.0
        games = 0
        if full_blocks:
            starts = rng.integers(
                0,
                possible_full_starts,
                size=full_blocks,
            )
            for start in starts:
                stop = int(start) + block_length_days
                model_sum += float(np.sum(daily_model_sum[start:stop]))
                baseline_sum += float(
                    np.sum(daily_baseline_sum[start:stop])
                )
                games += int(np.sum(daily_count[start:stop]))
        if remainder:
            start = int(rng.integers(0, possible_partial_starts))
            stop = start + remainder
            model_sum += float(np.sum(daily_model_sum[start:stop]))
            baseline_sum += float(np.sum(daily_baseline_sum[start:stop]))
            games += int(np.sum(daily_count[start:stop]))
        if games == 0:
            raise SealedEvaluationError(
                "Une réplication bootstrap ne contient aucun match."
            )
        model_mean = model_sum / games
        baseline_mean = baseline_sum / games
        if baseline_mean <= 0.0:
            raise SealedEvaluationError(
                "Log loss de référence bootstrap invalide."
            )
        statistics[replicate_index] = (
            baseline_mean - model_mean
        ) / baseline_mean
    if not np.all(np.isfinite(statistics)):
        raise SealedEvaluationError(
            "Statistique bootstrap non finie détectée."
        )
    lower, upper = np.quantile(
        statistics,
        [0.025, 0.975],
        method="linear",
    )
    canonical_statistics = (
        "\n".join(format(float(value), ".17g") for value in statistics)
        + "\n"
    ).encode("ascii")
    return BootstrapInterval(
        replications=replications,
        block_length_days=block_length_days,
        random_seed=random_seed,
        calendar_days=calendar_days,
        blocks_per_replicate=blocks_per_replicate,
        lower=float(lower),
        upper=float(upper),
        statistics_sha256=hashlib.sha256(canonical_statistics).hexdigest(),
    )


def decide_verdict(
    *,
    relative_log_loss_improvement: float,
    bootstrap_lower_bound: float,
    model_brier: float,
    baseline_brier: float,
    calibration_in_the_large: float,
    calibration_slope: float,
) -> VerdictResult:
    """Applique dans l'ordre les quatre branches préenregistrées."""
    values = (
        relative_log_loss_improvement,
        bootstrap_lower_bound,
        model_brier,
        baseline_brier,
        calibration_in_the_large,
        calibration_slope,
    )
    if not all(math.isfinite(value) for value in values):
        raise SealedEvaluationError(
            "Valeur non finie : aucun verdict ne peut être rendu."
        )
    gain_ok = relative_log_loss_improvement >= PRIMARY_GAIN_THRESHOLD
    interval_ok = bootstrap_lower_bound > 0.0
    brier_ok = model_brier <= baseline_brier
    calibration_level_ok = (
        abs(calibration_in_the_large) <= CALIBRATION_IN_LARGE_LIMIT
    )
    calibration_slope_ok = (
        CALIBRATION_SLOPE_MINIMUM
        <= calibration_slope
        <= CALIBRATION_SLOPE_MAXIMUM
    )
    primary_supported = gain_ok and interval_ok
    probability_guardrails_pass = (
        brier_ok and calibration_level_ok and calibration_slope_ok
    )
    if primary_supported and probability_guardrails_pass:
        label = VERDICT_VALIDATED
    elif primary_supported and not probability_guardrails_pass:
        label = VERDICT_BAD_PROBABILITIES
    elif gain_ok and not interval_ok and probability_guardrails_pass:
        label = VERDICT_PROMISING
    else:
        label = VERDICT_REJECTED
    failed: list[str] = []
    conditions = (
        ("relative_gain_meets_threshold", gain_ok),
        ("bootstrap_lower_bound_above_zero", interval_ok),
        ("model_brier_not_worse", brier_ok),
        ("calibration_in_the_large_within_limit", calibration_level_ok),
        ("calibration_slope_within_range", calibration_slope_ok),
    )
    failed.extend(name for name, passed in conditions if not passed)
    return VerdictResult(
        label=label,
        relative_gain_meets_threshold=gain_ok,
        bootstrap_lower_bound_above_zero=interval_ok,
        model_brier_not_worse=brier_ok,
        calibration_in_the_large_within_limit=calibration_level_ok,
        calibration_slope_within_range=calibration_slope_ok,
        primary_supported=primary_supported,
        probability_guardrails_pass=probability_guardrails_pass,
        failed_conditions=tuple(failed),
    )


def evaluate_sealed_rows(
    rows: Sequence[ModelRow],
    model_probabilities: Sequence[float] | np.ndarray,
    *,
    expected_rows: int = EXPECTED_SEALED_ROWS,
    expected_first_date: date = EXPECTED_FIRST_DATE,
    expected_last_date: date = EXPECTED_LAST_DATE,
    bootstrap_replications: int = BOOTSTRAP_REPLICATIONS,
    bootstrap_block_days: int = BOOTSTRAP_BLOCK_DAYS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> SealedEvaluationResult:
    """Évalue des probabilités déjà produites, sans entraîner le modèle."""
    input_rows = tuple(rows)
    ordered = validate_sealed_rows(
        input_rows,
        expected_rows=expected_rows,
        expected_first_date=expected_first_date,
        expected_last_date=expected_last_date,
    )
    if input_rows != ordered:
        raise SealedEvaluationError(
            "Les lignes doivent être fournies dans l'ordre canonique "
            "(official_date, game_id) afin de préserver l'appariement des "
            "probabilités."
        )
    probabilities = np.asarray(model_probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size != len(ordered):
        raise SealedEvaluationError(
            "Une probabilité modèle est requise pour chaque match."
        )
    targets = np.asarray([row.home_win for row in ordered], dtype=np.int8)
    _, probabilities = _validated_binary_arrays(targets, probabilities)
    baseline_probabilities = np.full(
        len(ordered),
        BASELINE_PROBABILITY,
        dtype=np.float64,
    )
    model_metrics = compute_probability_metrics(targets, probabilities)
    baseline_metrics = compute_probability_metrics(
        targets,
        baseline_probabilities,
    )
    calibration = fit_calibration_diagnostics(targets, probabilities)
    model_losses = _individual_log_losses(targets, probabilities)
    baseline_losses = _individual_log_losses(
        targets,
        baseline_probabilities,
    )
    relative_gain = (
        baseline_metrics.log_loss - model_metrics.log_loss
    ) / baseline_metrics.log_loss
    bootstrap = paired_calendar_block_bootstrap(
        [row.official_date for row in ordered],
        model_losses,
        baseline_losses,
        first_date=expected_first_date,
        last_date=expected_last_date,
        replications=bootstrap_replications,
        block_length_days=bootstrap_block_days,
        random_seed=bootstrap_seed,
    )
    verdict = decide_verdict(
        relative_log_loss_improvement=relative_gain,
        bootstrap_lower_bound=bootstrap.lower,
        model_brier=model_metrics.brier_score,
        baseline_brier=baseline_metrics.brier_score,
        calibration_in_the_large=(
            model_metrics.calibration_in_the_large
        ),
        calibration_slope=calibration.slope,
    )
    predictions = tuple(
        RowPrediction(
            row=row,
            model_probability=float(probability),
            baseline_probability=BASELINE_PROBABILITY,
            model_log_loss=float(model_loss),
            baseline_log_loss=float(baseline_loss),
        )
        for row, probability, model_loss, baseline_loss in zip(
            ordered,
            probabilities,
            model_losses,
            baseline_losses,
            strict=True,
        )
    )
    return SealedEvaluationResult(
        season=SEALED_SEASON,
        rows=len(ordered),
        first_official_date=ordered[0].official_date,
        last_official_date=ordered[-1].official_date,
        baseline_probability=BASELINE_PROBABILITY,
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        calibration=calibration,
        relative_log_loss_improvement=float(relative_gain),
        bootstrap=bootstrap,
        verdict=verdict,
        predictions=predictions,
    )


def _load_artifact_from_verified_bytes(
    content: bytes,
    configuration: EvaluationConfiguration,
) -> CalibratedModelArtifact:
    """Désérialise les mêmes octets dont le SHA-256 vient d'être validé."""
    # L'artefact historique a été créé avec ``python -m``. Si joblib a
    # enregistré la dataclass sous ``__main__``, ce lien contrôlé lui permet
    # de retrouver exactement le type approuvé, sans accepter un autre type.
    import __main__ as main_module

    missing = object()
    previous_main_class = getattr(
        main_module,
        "CalibratedModelArtifact",
        missing,
    )
    setattr(main_module, "CalibratedModelArtifact", CalibratedModelArtifact)
    try:
        with warnings.catch_warnings():
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
        raise SealedEvaluationError(
            f"L'artefact calibré est illisible : {error}"
        ) from error
    finally:
        if previous_main_class is missing:
            delattr(main_module, "CalibratedModelArtifact")
        else:
            setattr(
                main_module,
                "CalibratedModelArtifact",
                previous_main_class,
            )
    if not isinstance(artifact, CalibratedModelArtifact):
        raise SealedEvaluationError(
            "Type inattendu dans l'artefact calibré."
        )
    expected_fields: tuple[tuple[str, object], ...] = (
        ("artifact_format_version", 1),
        ("calibrated_model_version", "logistic_team_form_v1_platt"),
        ("base_model_version", "logistic_team_form_v1"),
        ("code_version", EXPECTED_ARTIFACT_CODE_VERSION),
        ("dataset_version", "mlb_team_form_v1"),
        ("dataset_sha256", EXPECTED_DATASET_SHA256),
        ("protocol_sha256", EXPECTED_MODEL_PROTOCOL_SHA256),
        ("feature_columns", tuple(FEATURE_COLUMNS)),
        ("base_training_seasons", (2021, 2022, 2023)),
        ("calibration_season", 2024),
        ("sealed_test_seasons", (2025,)),
        ("recent_seasons", (2026,)),
        ("calibration_method", "sigmoid"),
        ("sklearn_version", "1.9.0"),
        ("numpy_version", "2.5.2"),
    )
    for field_name, expected in expected_fields:
        _assert_equal(
            getattr(artifact, field_name, None),
            expected,
            description=f"artefact.{field_name}",
        )
    classifier = artifact.calibrated_classifier
    classes = getattr(classifier, "classes_", None)
    if classes is None or np.asarray(classes).tolist() != [0, 1]:
        raise SealedEvaluationError(
            "Les classes de l'artefact ne sont pas exactement [0, 1]."
        )

    runtime = _require_mapping(
        configuration.artifact_manifest,
        "runtime",
        context="manifest",
    )
    actual_runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    for key, actual in actual_runtime.items():
        _assert_equal(
            actual,
            runtime.get(key),
            description=f"runtime.{key}",
        )
    return artifact


def _artifact_state_sha256(artifact: CalibratedModelArtifact) -> str:
    """Empreinte l'état complet afin de détecter toute mutation en mémoire."""
    buffer = io.BytesIO()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Setting the shape on a NumPy array has been "
                    "deprecated.*"
                ),
                category=DeprecationWarning,
                module=r"joblib\.numpy_pickle",
            )
            joblib.dump(artifact, buffer, compress=3)
    except Exception as error:
        raise SealedEvaluationError(
            f"Empreinte de l'état du modèle impossible : {error}"
        ) from error
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _run_git(
    project_directory: Path,
    arguments: Sequence[str],
    *,
    allow_return_codes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    """Exécute une lecture Git déterministe sans shell intermédiaire."""
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_directory,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError as error:
        raise SealedEvaluationError(
            f"Git est indisponible : {error}"
        ) from error
    if completed.returncode not in allow_return_codes:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SealedEvaluationError(
            f"Contrôle Git impossible ({' '.join(arguments)}) : {detail}."
        )
    return completed


def _git_relative_path(project_directory: Path, path: Path) -> str:
    """Convertit un chemin vérifié en chemin Git POSIX."""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(project_directory.resolve())
    except ValueError as error:
        raise SealedEvaluationError(
            f"Chemin hors du dépôt Git : {resolved}."
        ) from error
    return relative.as_posix()


def _normalize_git_commit(value: str) -> str:
    """Exige un commit complet pour éviter toute ouverture ambiguë."""
    if not isinstance(value, str) or _GIT_COMMIT_PATTERN.fullmatch(value) is None:
        raise SealedEvaluationError(
            "--expected-evaluation-code-commit doit contenir "
            "exactement 40 caractères hexadécimaux minuscules."
        )
    return value


def _verify_git_opening_preconditions(
    configuration: EvaluationConfiguration,
    *,
    expected_evaluation_code_commit: str,
) -> str:
    """Bloque toute ouverture sur du code non figé ou déjà évalué."""
    expected_commit = _normalize_git_commit(
        expected_evaluation_code_commit
    )
    project = configuration.project_directory
    head = _run_git(project, ["rev-parse", "HEAD"]).stdout.strip()
    if head != expected_commit:
        raise SealedEvaluationError(
            "Le commit courant ne correspond pas au commit d'évaluation "
            f"attendu : {head}."
        )
    status = _run_git(
        project,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    ).stdout
    if status.strip():
        raise SealedEvaluationError(
            "Le dépôt Git doit être parfaitement propre avant l'ouverture."
        )
    required_paths = (
        project / "src" / "sealed_evaluation.py",
        configuration.evaluation_protocol_path,
        configuration.artifact_manifest_path,
        configuration.model_protocol_path,
    )
    expected_module_path = (project / "src" / "sealed_evaluation.py").resolve()
    if Path(__file__).resolve() != expected_module_path:
        raise SealedEvaluationError(
            "Le module exécuté n'est pas src/sealed_evaluation.py dans le "
            "dépôt contrôlé."
        )
    for required_path in required_paths:
        relative = _git_relative_path(project, required_path)
        _run_git(project, ["ls-files", "--error-unmatch", "--", relative])
        _run_git(project, ["cat-file", "-e", f"HEAD:{relative}"])
    module_relative = _git_relative_path(project, expected_module_path)
    working_blob = _run_git(
        project,
        ["hash-object", "--", module_relative],
    ).stdout.strip()
    committed_blob = _run_git(
        project,
        ["rev-parse", f"HEAD:{module_relative}"],
    ).stdout.strip()
    if working_blob != committed_blob:
        raise SealedEvaluationError(
            "Le module exécuté ne correspond pas exactement au blob Git "
            "enregistré dans HEAD."
        )

    for revision in REQUIRED_ANCESTOR_REVISIONS:
        resolved_revision = _run_git(
            project,
            ["rev-parse", f"{revision}^{{commit}}"],
        ).stdout.strip()
        ancestor = _run_git(
            project,
            ["merge-base", "--is-ancestor", resolved_revision, head],
            allow_return_codes=(0, 1),
        )
        if ancestor.returncode != 0:
            raise SealedEvaluationError(
                "Un commit figé requis n'appartient pas à l'historique : "
                f"{resolved_revision}."
            )

    output_relative = _git_relative_path(
        project,
        configuration.output_directory,
    )
    history = _run_git(
        project,
        ["log", "--all", "--format=%H", "--", output_relative],
    ).stdout.strip()
    if history:
        raise SealedEvaluationError(
            "Un résultat 2025 existe déjà dans l'historique Git."
        )
    if configuration.output_directory.exists():
        raise SealedEvaluationError(
            "Le dossier de résultats 2025 existe déjà : ouverture refusée."
        )
    ignored_probe = f"{output_relative}/report.json"
    ignored = _run_git(
        project,
        ["check-ignore", "--no-index", "--quiet", "--", ignored_probe],
        allow_return_codes=(0, 1),
    )
    if ignored.returncode == 0:
        raise SealedEvaluationError(
            "Le dossier de résultats est ignoré par Git."
        )
    return head


def _verify_git_unchanged_after_evaluation(
    configuration: EvaluationConfiguration,
    *,
    expected_commit: str,
) -> None:
    """Revérifie HEAD et tous les fichiers suivis avant publication."""
    project = configuration.project_directory
    head = _run_git(project, ["rev-parse", "HEAD"]).stdout.strip()
    if head != expected_commit:
        raise SealedEvaluationError(
            "Le commit Git a changé pendant l'évaluation."
        )
    expected_module_path = (project / "src" / "sealed_evaluation.py").resolve()
    if Path(__file__).resolve() != expected_module_path:
        raise SealedEvaluationError(
            "Le module exécuté a changé d'identité pendant l'évaluation."
        )
    module_relative = _git_relative_path(project, expected_module_path)
    working_blob = _run_git(
        project,
        ["hash-object", "--", module_relative],
    ).stdout.strip()
    committed_blob = _run_git(
        project,
        ["rev-parse", f"HEAD:{module_relative}"],
    ).stdout.strip()
    if working_blob != committed_blob:
        raise SealedEvaluationError(
            "Le blob du module a changé pendant l'évaluation."
        )
    unstaged = _run_git(
        project,
        ["diff", "--quiet"],
        allow_return_codes=(0, 1),
    )
    staged = _run_git(
        project,
        ["diff", "--cached", "--quiet"],
        allow_return_codes=(0, 1),
    )
    if unstaged.returncode != 0 or staged.returncode != 0:
        raise SealedEvaluationError(
            "Un fichier suivi a changé pendant l'évaluation."
        )
    output_relative = _git_relative_path(
        project,
        configuration.output_directory,
    )
    status_lines = _run_git(
        project,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    ).stdout.splitlines()
    allowed_prefix = f"?? {output_relative}/"
    unexpected = [
        line for line in status_lines if not line.startswith(allowed_prefix)
    ]
    if unexpected:
        raise SealedEvaluationError(
            "Un fichier imprévu est apparu pendant l'évaluation."
        )


def _reserve_output_directory(
    configuration: EvaluationConfiguration,
    *,
    evaluation_code_commit: str,
    opening_id: str,
    opened_at_utc: str,
) -> Path:
    """Réserve atomiquement l'unique ouverture avant lecture des données."""
    output = configuration.output_directory
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise SealedEvaluationError(
            "Le dossier de résultats existe déjà : aucune réouverture."
        ) from error
    except OSError as error:
        raise SealedEvaluationError(
            f"Réservation du dossier de résultats impossible : {error}"
        ) from error
    marker = output / "OPENING_IN_PROGRESS"
    marker_payload = (
        "État : OPENING_IN_PROGRESS\n"
        f"opening_id={opening_id}\n"
        f"opened_at_utc={opened_at_utc}\n"
        f"evaluation_code_commit={evaluation_code_commit}\n"
        "Toute erreur après ce point consomme l'ouverture.\n"
    ).encode("utf-8")
    try:
        with marker.open("xb") as destination:
            destination.write(marker_payload)
            destination.flush()
            os.fsync(destination.fileno())
    except OSError as error:
        raise SealedEvaluationError(
            "Le dossier a été réservé mais son marqueur d'ouverture "
            f"n'a pas pu être écrit : {error}"
        ) from error
    return marker


def _complete_opening_marker(
    in_progress_marker: Path,
    *,
    opening_id: str,
    opened_at_utc: str,
    evaluation_code_commit: str,
    report_sha256: str,
) -> Path:
    """Remplace le marqueur seulement après le rapport final revérifié."""
    completed_marker = in_progress_marker.with_name("OPENING_COMPLETED")
    payload = (
        "État : OPENING_COMPLETED\n"
        f"opening_id={opening_id}\n"
        f"opened_at_utc={opened_at_utc}\n"
        f"evaluation_code_commit={evaluation_code_commit}\n"
        f"report_sha256={report_sha256}\n"
    ).encode("utf-8")
    _publish_exclusive_bytes(completed_marker, payload)
    try:
        in_progress_marker.unlink()
    except OSError as error:
        raise SealedEvaluationError(
            "Le rapport est publié, mais le marqueur d'ouverture n'a pas "
            f"pu être finalisé : {error}"
        ) from error
    return completed_marker


def _prediction_csv_bytes(result: SealedEvaluationResult) -> bytes:
    """Produit le CSV canonique, ordonné et non arrondi."""
    destination = io.StringIO(newline="")
    writer = csv.writer(destination, lineterminator="\n")
    writer.writerow(PREDICTION_COLUMNS)
    for prediction in result.predictions:
        row = prediction.row
        if row.season != SEALED_SEASON:
            raise SealedEvaluationError(
                "Une saison interdite a atteint le fichier de prédictions."
            )
        writer.writerow(
            (
                row.game_id,
                row.season,
                row.official_date.isoformat(),
                row.feature_as_of_date.isoformat(),
                row.away_team_id,
                row.home_team_id,
                row.away_max_source_date.isoformat(),
                row.home_max_source_date.isoformat(),
                row.home_win,
                format(prediction.model_probability, ".17g"),
                format(prediction.baseline_probability, ".17g"),
                format(prediction.model_log_loss, ".17g"),
                format(prediction.baseline_log_loss, ".17g"),
            )
        )
    return destination.getvalue().encode("utf-8")


def _runtime_versions() -> dict[str, str]:
    """Versions réellement utilisées pendant l'ouverture."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _report_json_bytes(
    *,
    configuration: EvaluationConfiguration,
    result: SealedEvaluationResult,
    evaluation_code_commit: str,
    evaluation_module_sha256: str,
    opening_id: str,
    opened_at_utc: str,
    artifact_state_before: str,
    artifact_state_after: str,
    predictions_sha256: str,
    predictions_size_bytes: int,
) -> bytes:
    """Construit le rapport complet; le verdict utilise les valeurs brutes."""
    report = {
        "evaluation_report_version": EVALUATION_REPORT_VERSION,
        "status": "COMPLETED_AND_VERIFIED",
        "opening": {
            "opening_id": opening_id,
            "opened_at_utc": opened_at_utc,
            "open_only_once": True,
            "evaluation_code_commit": evaluation_code_commit,
            "evaluation_module_sha256": evaluation_module_sha256,
        },
        "inputs": {
            "evaluation_protocol": {
                "path": _git_relative_path(
                    configuration.project_directory,
                    configuration.evaluation_protocol_path,
                ),
                "sha256": configuration.evaluation_protocol_sha256,
            },
            "artifact_manifest": {
                "path": _git_relative_path(
                    configuration.project_directory,
                    configuration.artifact_manifest_path,
                ),
                "sha256": configuration.artifact_manifest_sha256,
            },
            "model_protocol": {
                "path": _git_relative_path(
                    configuration.project_directory,
                    configuration.model_protocol_path,
                ),
                "sha256": configuration.model_protocol_sha256,
            },
            "model_artifact": {
                "path": _git_relative_path(
                    configuration.project_directory,
                    configuration.artifact_path,
                ),
                "sha256": configuration.artifact_sha256,
                "size_bytes": configuration.artifact_size_bytes,
                "code_version": EXPECTED_ARTIFACT_CODE_VERSION,
            },
            "dataset": {
                "path": _git_relative_path(
                    configuration.project_directory,
                    configuration.dataset_path,
                ),
                "sha256": configuration.dataset_sha256,
                "size_bytes": configuration.dataset_size_bytes,
                "row_count": EXPECTED_DATASET_ROWS,
            },
        },
        "runtime": _runtime_versions(),
        "sealed_cohort": {
            "season": result.season,
            "rows": result.rows,
            "first_official_date": result.first_official_date.isoformat(),
            "last_official_date": result.last_official_date.isoformat(),
            "calendar_days_inclusive": EXPECTED_CALENDAR_DAYS,
            "target": "home_win",
            "feature_columns": list(FEATURE_COLUMNS),
        },
        "reference_baseline": {
            "home_wins": BASELINE_HOME_WINS,
            "games": BASELINE_GAMES,
            "probability_formula": "3623 / 6815",
            "probability": result.baseline_probability,
            "reestimated": False,
        },
        "metrics": {
            "model": asdict(result.model_metrics),
            "reference_baseline": asdict(result.baseline_metrics),
            "calibration_diagnostic": asdict(result.calibration),
            "relative_log_loss_improvement": (
                result.relative_log_loss_improvement
            ),
        },
        "uncertainty": asdict(result.bootstrap),
        "verdict": {
            **asdict(result.verdict),
            "comparison_precision": "UNROUNDED_VALUES",
        },
        "predictions": {
            "file": "predictions.csv",
            "columns": list(PREDICTION_COLUMNS),
            "rows": result.rows,
            "sha256": predictions_sha256,
            "size_bytes": predictions_size_bytes,
        },
        "execution_invariants": {
            "predict_proba_calls": 1,
            "artifact_state_sha256_before": artifact_state_before,
            "artifact_state_sha256_after": artifact_state_after,
            "artifact_state_unchanged": (
                artifact_state_before == artifact_state_after
            ),
            "season_2026_predictions": 0,
            "season_2026_metrics": 0,
            "model_fit_calls": 0,
            "diagnostic_fit_calls": 1,
        },
        "registered_limitations": configuration.evaluation_protocol.get(
            "registered_limitations",
            [],
        ),
        "prohibited_claims": configuration.evaluation_protocol.get(
            "prohibited_claims",
            [],
        ),
        "interpretation": (
            "Ce rapport évalue un signal statistique, pas une rentabilité "
            "de pari. Aucun ROI n'est calculable sans cotes pré-match "
            "horodatées. La disponibilité temporelle historique reste "
            "UNVERIFIABLE."
        ),
    }
    try:
        text = json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as error:
        raise SealedEvaluationError(
            f"Rapport JSON non sérialisable : {error}"
        ) from error
    return text.encode("utf-8")


def _publish_exclusive_bytes(path: Path, content: bytes) -> tuple[str, int]:
    """Publie sans remplacement après une écriture temporaire synchronisée."""
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise SealedEvaluationError(
            f"Le fichier existe déjà et ne sera pas remplacé : {path}."
        ) from error
    except OSError as error:
        raise SealedEvaluationError(
            f"Publication exclusive impossible pour {path.name} : {error}"
        ) from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        reloaded = path.read_bytes()
    except OSError as error:
        raise SealedEvaluationError(
            f"Relecture impossible pour {path.name} : {error}"
        ) from error
    if reloaded != content:
        raise SealedEvaluationError(
            f"Le fichier publié diffère des octets préparés : {path.name}."
        )
    return hashlib.sha256(reloaded).hexdigest(), len(reloaded)


def open_sealed_evaluation(
    *,
    expected_evaluation_code_commit: str,
    evaluation_protocol_path: Path = DEFAULT_EVALUATION_PROTOCOL_PATH,
    project_directory: Path = PROJECT_DIRECTORY,
) -> OpenedSealedEvaluation:
    """Ouvre 2025 une seule fois, puis publie deux sorties vérifiées."""
    configuration = load_evaluation_configuration(
        evaluation_protocol_path,
        project_directory=project_directory,
    )
    evaluation_code_commit = _verify_git_opening_preconditions(
        configuration,
        expected_evaluation_code_commit=expected_evaluation_code_commit,
    )
    opening_id = uuid.uuid4().hex
    opened_at_utc = datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    opening_marker = _reserve_output_directory(
        configuration,
        evaluation_code_commit=evaluation_code_commit,
        opening_id=opening_id,
        opened_at_utc=opened_at_utc,
    )

    # À partir d'ici, l'ouverture est consommée même si une erreur survient.
    artifact_path, artifact_bytes, artifact_sha256 = _read_verified_bytes(
        configuration.artifact_path,
        expected_sha256=configuration.artifact_sha256,
        description="artefact calibré",
    )
    dataset_path, dataset_bytes, dataset_sha256 = _read_verified_bytes(
        configuration.dataset_path,
        expected_sha256=configuration.dataset_sha256,
        description="dataset scellé",
    )
    if len(artifact_bytes) != configuration.artifact_size_bytes:
        raise SealedEvaluationError(
            "Taille inattendue pour l'artefact calibré."
        )
    if len(dataset_bytes) != configuration.dataset_size_bytes:
        raise SealedEvaluationError(
            "Taille inattendue pour le dataset scellé."
        )
    _assert_equal(
        artifact_sha256,
        configuration.artifact_sha256,
        description="artefact relu",
    )
    _assert_equal(
        dataset_sha256,
        configuration.dataset_sha256,
        description="dataset relu",
    )
    artifact = _load_artifact_from_verified_bytes(
        artifact_bytes,
        configuration,
    )
    rows = _load_sealed_rows_from_verified_csv(dataset_bytes)
    features = np.asarray(
        [row.features for row in rows],
        dtype=np.float64,
    )
    if features.shape != (EXPECTED_SEALED_ROWS, len(FEATURE_COLUMNS)):
        raise SealedEvaluationError(
            "Matrice 2025 de dimensions inattendues."
        )
    if not np.all(np.isfinite(features)):
        raise SealedEvaluationError("Matrice 2025 non finie.")

    artifact_state_before = _artifact_state_sha256(artifact)
    classifier = artifact.calibrated_classifier
    try:
        raw_probabilities = classifier.predict_proba(features)
    except Exception as error:
        raise SealedEvaluationError(
            f"Prédiction 2025 impossible : {error}"
        ) from error
    artifact_state_after = _artifact_state_sha256(artifact)
    if artifact_state_after != artifact_state_before:
        raise SealedEvaluationError(
            "L'état du modèle a changé pendant predict_proba."
        )
    probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    classes = np.asarray(classifier.classes_)
    positive_indices = np.flatnonzero(classes == 1)
    if probabilities.shape != (EXPECTED_SEALED_ROWS, 2):
        raise SealedEvaluationError(
            "Sortie predict_proba de dimensions inattendues."
        )
    if positive_indices.tolist() != [1]:
        raise SealedEvaluationError(
            "La classe positive 1 n'a pas l'indice attendu."
        )
    result = evaluate_sealed_rows(
        rows,
        probabilities[:, int(positive_indices[0])],
    )

    # Les chemins peuvent être modifiés par un autre processus : ils sont
    # revérifiés avant de publier, même si le calcul a utilisé les octets sûrs.
    _, current_artifact_bytes, _ = _read_verified_bytes(
        artifact_path,
        expected_sha256=configuration.artifact_sha256,
        description="artefact calibré après évaluation",
    )
    _, current_dataset_bytes, _ = _read_verified_bytes(
        dataset_path,
        expected_sha256=configuration.dataset_sha256,
        description="dataset après évaluation",
    )
    if current_artifact_bytes != artifact_bytes:
        raise SealedEvaluationError(
            "L'artefact a changé pendant l'évaluation."
        )
    if current_dataset_bytes != dataset_bytes:
        raise SealedEvaluationError(
            "Le dataset a changé pendant l'évaluation."
        )
    _verify_git_unchanged_after_evaluation(
        configuration,
        expected_commit=evaluation_code_commit,
    )
    module_path = Path(__file__).resolve(strict=True)
    try:
        module_bytes = module_path.read_bytes()
    except OSError as error:
        raise SealedEvaluationError(
            f"Lecture du module d'évaluation impossible : {error}"
        ) from error
    evaluation_module_sha256 = hashlib.sha256(module_bytes).hexdigest()

    predictions_bytes = _prediction_csv_bytes(result)
    predictions_sha256 = hashlib.sha256(predictions_bytes).hexdigest()
    report_bytes = _report_json_bytes(
        configuration=configuration,
        result=result,
        evaluation_code_commit=evaluation_code_commit,
        evaluation_module_sha256=evaluation_module_sha256,
        opening_id=opening_id,
        opened_at_utc=opened_at_utc,
        artifact_state_before=artifact_state_before,
        artifact_state_after=artifact_state_after,
        predictions_sha256=predictions_sha256,
        predictions_size_bytes=len(predictions_bytes),
    )
    predictions_path = configuration.output_directory / "predictions.csv"
    report_path = configuration.output_directory / "report.json"
    published_predictions_sha, published_predictions_size = (
        _publish_exclusive_bytes(predictions_path, predictions_bytes)
    )
    if published_predictions_sha != predictions_sha256:
        raise SealedEvaluationError(
            "Empreinte inattendue après publication des prédictions."
        )
    report_sha, report_size = _publish_exclusive_bytes(
        report_path,
        report_bytes,
    )
    try:
        parsed_report = json.loads(report_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealedEvaluationError(
            f"Rapport publié illisible : {error}"
        ) from error
    if (
        not isinstance(parsed_report, dict)
        or parsed_report.get("status") != "COMPLETED_AND_VERIFIED"
        or parsed_report.get("verdict", {}).get("label")
        != result.verdict.label
    ):
        raise SealedEvaluationError(
            "Le rapport publié ne confirme pas le verdict attendu."
        )
    _complete_opening_marker(
        opening_marker,
        opening_id=opening_id,
        opened_at_utc=opened_at_utc,
        evaluation_code_commit=evaluation_code_commit,
        report_sha256=report_sha,
    )
    export = EvaluationExport(
        directory=configuration.output_directory,
        predictions_path=predictions_path,
        predictions_sha256=published_predictions_sha,
        predictions_size_bytes=published_predictions_size,
        report_path=report_path,
        report_sha256=report_sha,
        report_size_bytes=report_size,
    )
    return OpenedSealedEvaluation(
        evaluation_code_commit=evaluation_code_commit,
        result=result,
        export=export,
    )


def _display_preview(preview: EvaluationPreview) -> None:
    """Explique clairement que 2025 est encore fermé."""
    print("Aperçu de l'évaluation MLB 2025")
    print(f"Saison scellée : {preview.sealed_season}")
    print(f"Lignes attendues : {preview.expected_sealed_rows}")
    print(
        "SHA-256 du protocole d'évaluation : "
        f"{preview.evaluation_protocol_sha256}"
    )
    print(
        "SHA-256 du manifeste : "
        f"{preview.artifact_manifest_sha256}"
    )
    print(
        "SHA-256 du protocole du modèle : "
        f"{preview.model_protocol_sha256}"
    )
    print(f"SHA-256 attendu du modèle : {preview.artifact_sha256}")
    print(f"SHA-256 attendu du dataset : {preview.dataset_sha256}")
    print()
    print("Mode aperçu : le test scellé 2025 reste fermé.")
    print("Aucun octet du CSV ou du joblib n'a été lu.")
    print("Aucune prédiction, métrique ou sortie n'a été créée.")


def _display_completed(opened: OpenedSealedEvaluation) -> None:
    """Affiche les résultats seulement après leur publication vérifiée."""
    result = opened.result
    print("Évaluation scellée MLB 2025 terminée et vérifiée")
    print(f"Commit d'évaluation : {opened.evaluation_code_commit}")
    print(f"Matchs évalués : {result.rows}")
    print(
        "Log loss du modèle : "
        f"{result.model_metrics.log_loss:.6f}"
    )
    print(
        "Log loss de la référence : "
        f"{result.baseline_metrics.log_loss:.6f}"
    )
    print(
        "Gain relatif de log loss : "
        f"{result.relative_log_loss_improvement:.4%}"
    )
    print(
        "Intervalle bootstrap à 95 % : "
        f"[{result.bootstrap.lower:.4%}, {result.bootstrap.upper:.4%}]"
    )
    print(f"Verdict préenregistré : {result.verdict.label}")
    print()
    print(f"Prédictions : {opened.export.predictions_path}")
    print(f"SHA-256 : {opened.export.predictions_sha256}")
    print(f"Rapport : {opened.export.report_path}")
    print(f"SHA-256 : {opened.export.report_sha256}")
    print()
    print(
        "Ce verdict ne mesure pas la rentabilité des paris : aucune cote "
        "pré-match horodatée n'est encore utilisée."
    )


def main() -> None:
    """Point d'entrée : aperçu sûr par défaut, ouverture doublement explicite."""
    parser = argparse.ArgumentParser(
        description=(
            "Évalue une seule fois la cohorte MLB 2025 selon le protocole "
            "figé. Sans drapeau, aucun CSV ni modèle n'est lu."
        )
    )
    parser.add_argument(
        "--open-sealed-test",
        action="store_true",
        help="Consomme l'ouverture unique du test scellé 2025.",
    )
    parser.add_argument(
        "--expected-evaluation-code-commit",
        help=(
            "Commit Git complet de 40 caractères qui contient le code "
            "d'évaluation figé."
        ),
    )
    arguments = parser.parse_args()
    if not arguments.open_sealed_test:
        if arguments.expected_evaluation_code_commit is not None:
            parser.error(
                "--expected-evaluation-code-commit nécessite "
                "--open-sealed-test."
            )
        try:
            preview = preview_sealed_evaluation()
        except (SealedEvaluationError, ValueError) as error:
            raise SystemExit(
                f"Aperçu de l'évaluation impossible : {error}"
            ) from error
        _display_preview(preview)
        return
    if arguments.expected_evaluation_code_commit is None:
        parser.error(
            "--open-sealed-test exige "
            "--expected-evaluation-code-commit."
        )
    try:
        opened = open_sealed_evaluation(
            expected_evaluation_code_commit=(
                arguments.expected_evaluation_code_commit
            )
        )
    except (SealedEvaluationError, ValueError) as error:
        raise SystemExit(
            "Évaluation scellée 2025 interrompue sans publication de "
            f"métriques : {error}"
        ) from error
    _display_completed(opened)


if __name__ == "__main__":
    main()
