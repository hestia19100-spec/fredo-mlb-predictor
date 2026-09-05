"""Fondations controlees de la prediction fantome MLB v2.

Ce module ne sait volontairement ni activer ni executer une prediction. Il
valide l'apercu statique, construit les formats canoniques, inspecte les
creneaux, fige leurs preuves distantes et publie leur sous-ensemble source,
puis le registre des candidats et leurs variables J-1 canoniques.
Une primitive separee controle les fichiers figes du modele et les versions
installees sans deserialisation et sans autoriser une execution.
Un chargeur interne peut ensuite deserialiser ces octets sans predire.
Deux primitives internes empreintent son etat en memoire et refusent une
mutation entre deux prises d'empreinte. Une primitive d'appel unique produit
ensuite des probabilites internes; une autre les lie aux cinq preuves du slot
et publie exclusivement le sixieme fichier predictions.csv.
Une primitive interne distincte relit les six fichiers, verifie un contexte
d'execution deja autorise et publie le septieme fichier receipt.json.
Une derniere primitive interne relit les sept preuves, controle la limite
temporelle et publie exclusivement le marqueur terminal COMPLETED.
Les chemins d'apercu restent sans lecture officielle, import ou chargement
de modele.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import closing, contextmanager
import csv
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.utils import format_datetime, parsedate_to_datetime
from enum import Enum
import errno
import gzip
import hashlib
from importlib import import_module, metadata as distribution_metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sqlite3
import stat
import sys
import tempfile
import threading
import time
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit
import zlib
import warnings

import requests

from src.database import DATA_DIR, DATABASE_PATH
from src.ingestion_service import (
    INGESTION_SOURCE,
    ScheduleIngestionResult,
    build_schedule_request_parameters,
    run_observed_schedule_ingestion,
)
from src.mlb_api import MLBAPIRetryableError, ScheduledGame
from src.raw_archive import RawArchiveError, verify_raw_archive
from src.retry_policy import RetryPolicy, run_with_retries


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SHADOW_PROTOCOL_RELATIVE_PATH = PurePosixPath(
    "shadow_protocols/logistic_team_form_v1_platt_shadow_v2.json"
)
DEFAULT_SHADOW_PROTOCOL_PATH = PROJECT_DIRECTORY.joinpath(
    *SHADOW_PROTOCOL_RELATIVE_PATH.parts
)

EXPECTED_SHADOW_PROTOCOL_SHA256 = (
    "4dcae9e85bb9ed5b3f4a9f961491d872"
    "56965c42e97d11e70f31f02f52cc49c9"
)
EXPECTED_SHADOW_PROTOCOL_VERSION = 2
EXPECTED_SHADOW_PROTOCOL_STATUS = (
    "REGISTERED_BEFORE_FIRST_SHADOW_V2_PREDICTION"
)
EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON = date(2026, 8, 31)
EXPECTED_TARGET_SEASON = 2026
EXPECTED_PREVIEW_MODE = "PREVIEW_WITHOUT_MODEL_OR_PREDICTIONS"

SHADOW_RESULT_ROOT_RELATIVE_PATH = PurePosixPath(
    "shadow_results/logistic_team_form_v1_platt_shadow_v2"
)
SHADOW_SLOT_SUCCESS_FILENAMES = (
    "RESERVED",
    "activation_reverification.remote.json.gz",
    "source_snapshot.json.gz",
    "candidate_ledger.csv",
    "features.csv",
    "predictions.csv",
    "receipt.json",
    "COMPLETED",
)

ACTIVATION_REVERIFICATION_FILENAME = (
    "activation_reverification.remote.json.gz"
)
SOURCE_SNAPSHOT_FILENAME = "source_snapshot.json.gz"
CANDIDATE_LEDGER_FILENAME = "candidate_ledger.csv"
FEATURES_FILENAME = "features.csv"
PREDICTIONS_FILENAME = "predictions.csv"
RECEIPT_FILENAME = "receipt.json"
COMPLETED_FILENAME = "COMPLETED"
EXPECTED_CALIBRATED_MODEL_VERSION = "logistic_team_form_v1_platt"
ACTIVATION_RELATIVE_PATH = PurePosixPath(
    "shadow_activations/logistic_team_form_v1_platt_shadow_v2/activation.json"
)

_CANDIDATE_LEDGER_COLUMNS = (
    "batch_id",
    "game_id",
    "occurrence_key",
    "season",
    "official_date",
    "away_team_id",
    "home_team_id",
    "scheduled_start_utc",
    "status_code",
    "abstract_state",
    "detailed_state",
    "eligibility_status",
    "exclusion_reason",
)
GITHUB_COMPARE_URL_TEMPLATE = (
    "https://api.github.com/repos/hestia19100-spec/"
    "fredo-mlb-predictor/compare/{expected_commit}...main"
)
GITHUB_REMOTE_REF = "refs/heads/main"
GITHUB_REQUEST_HEADERS = MappingProxyType(
    {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "fredo-mlb-predictor-shadow-v2/1.0",
    }
)
GITHUB_REQUEST_TIMEOUT_SECONDS = 30

EXPECTED_SERVICE_MODULE_PATH = "src/shadow_prediction.py"
EXPECTED_MODEL_ARTIFACT_PATH = (
    "models/logistic_team_form_v1_platt.joblib"
)
EXPECTED_MODEL_ARTIFACT_SHA256 = (
    "e0d4d2421ba076072c7ef8b3bc97dd9a"
    "341e26c62828a0ad9ba43f30da15ff55"
)
EXPECTED_ARTIFACT_MANIFEST_PATH = (
    "model_artifacts/logistic_team_form_v1_platt.json"
)
EXPECTED_ARTIFACT_MANIFEST_SHA256 = (
    "a7375d5376baa043b3365cff713cb717"
    "010ad812efee472b594fa454306c96ae"
)
EXPECTED_MODEL_PROTOCOL_PATH = "model_protocols/logistic_team_form_v1.json"
EXPECTED_MODEL_PROTOCOL_SHA256 = (
    "c4cb1af750619967514d37ae3a5a47a6"
    "a04255aeaccb20c5e94533dc4d138451"
)
EXPECTED_EVALUATION_PROTOCOL_SHA256 = (
    "f6dbace5d25d92c5d0ec3c9ae16962ab"
    "03e022439177342bd28d607afba7b1a9"
)
EXPECTED_EVALUATION_REPORT_SHA256 = (
    "3f3d71baa122a4ac5f1690356e1c1656"
    "e812f500436dd6838d1421ba36522e70"
)
EXPECTED_EVALUATION_RESULTS_COMMIT = (
    "7d64a6e5c03e36900cbf36f7c10182232c126baa"
)
EXPECTED_MODEL_ARTIFACT_SIZE_BYTES = 1589
EXPECTED_MODEL_ARTIFACT_CODE_COMMIT = (
    "1d13f8f361de0865f722a8560f1efda8e95badbc"
)
_MODEL_RUNTIME_DISTRIBUTIONS = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit_learn", "scikit-learn"),
    ("joblib", "joblib"),
)
_MODEL_RUNTIME_KEYS = frozenset(
    {"python", *(key for key, _ in _MODEL_RUNTIME_DISTRIBUTIONS)}
)
_MODEL_LOADING_LOCK = threading.Lock()
_MODEL_WARNING_POLICY_ID = "VERIFIED_JOBLIB_NUMPY_COMPATIBILITY_V1"
_MODEL_WARNING_MESSAGE = "Setting the shape on a NumPy array has been deprecated.*"
_MODEL_WARNING_MODULE = r"joblib\.numpy_pickle"
_MODEL_PREREQUISITE_PATHS = frozenset({
    SHADOW_PROTOCOL_RELATIVE_PATH.as_posix(),
    EXPECTED_ARTIFACT_MANIFEST_PATH,
    EXPECTED_MODEL_PROTOCOL_PATH,
    EXPECTED_MODEL_ARTIFACT_PATH,
})

_CANONICAL_DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T"
    r"([0-9]{2}):([0-9]{2}):([0-9]{2})Z\Z"
)
_DATABASE_UTC_TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:Z|\+00:00)\Z"
)
_FIXED_6_PATTERN = re.compile(r"[0-9]+\.[0-9]{6}\Z")

_FEATURE_ROW_FIELD_NAMES = (
    "prediction_id",
    "batch_id",
    "game_id",
    "occurrence_key",
    "season",
    "official_date",
    "away_team_id",
    "home_team_id",
    "scheduled_start_utc",
    "feature_as_of_date",
    "away_max_source_date",
    "home_max_source_date",
    "away_games_before",
    "away_win_pct_before",
    "away_runs_scored_per_game_before",
    "away_runs_allowed_per_game_before",
    "home_games_before",
    "home_win_pct_before",
    "home_runs_scored_per_game_before",
    "home_runs_allowed_per_game_before",
)
_FEATURE_ROW_SHA256_INDEXES = (0, 1, 3)
_FEATURE_ROW_INTEGER_INDEXES = (2, 4, 6, 7, 12, 16)
_FEATURE_ROW_DATE_INDEXES = (5, 9, 10, 11)
_FEATURE_ROW_TIMESTAMP_INDEX = 8
_FEATURE_ROW_FIXED_6_INDEXES = (13, 14, 15, 17, 18, 19)
_FEATURES_COLUMNS = (*_FEATURE_ROW_FIELD_NAMES, "feature_row_sha256")
_PREDICTIONS_COLUMNS = (
    "prediction_id",
    "batch_id",
    "game_id",
    "occurrence_key",
    "season",
    "official_date_at_prediction",
    "away_team_id",
    "home_team_id",
    "scheduled_start_utc_at_prediction",
    "information_cutoff_utc",
    "issued_at_utc",
    "feature_as_of_date",
    "away_max_source_date",
    "home_max_source_date",
    "away_games_before",
    "away_win_pct_before",
    "away_runs_scored_per_game_before",
    "away_runs_allowed_per_game_before",
    "home_games_before",
    "home_win_pct_before",
    "home_runs_scored_per_game_before",
    "home_runs_allowed_per_game_before",
    "p_home_win",
    "p_away_win",
    "model_version",
    "artifact_sha256",
    "protocol_sha256",
    "code_commit",
)

_SOURCE_SNAPSHOT_KEYS = frozenset(
    {
        "schema_version",
        "batch_id",
        "target_official_date",
        "created_at_utc",
        "information_cutoff_utc",
        "schedule_ingestion",
        "sqlite_snapshot",
        "teams",
        "target_schedule",
        "source_final_games",
    }
)
_SOURCE_SCHEDULE_INGESTION_KEYS = frozenset(
    {
        "run_id",
        "source",
        "requested_start_date",
        "requested_end_date",
        "game_types",
        "request_parameters_json",
        "completed_at_utc",
        "raw_archive_path",
        "raw_archive_sha256",
        "response_effective_url",
        "response_status_code",
        "response_redirect_count",
        "mlb_http_date_header_raw",
        "mlb_http_date_utc",
        "mlb_http_response_received_at_utc",
        "response_body_sha256",
    }
)
_SOURCE_SQLITE_SNAPSHOT_KEYS = frozenset(
    {
        "source_database_path",
        "sha256",
        "size_bytes",
        "foreign_key_violation_count",
        "active_ingestion_count",
    }
)
_SOURCE_TEAM_KEYS = frozenset({"team_id", "name", "abbreviation"})
_SOURCE_TARGET_SCHEDULE_KEYS = frozenset(
    {
        "game_id",
        "season",
        "official_date",
        "game_datetime_utc",
        "game_type",
        "status_code",
        "abstract_state",
        "detailed_state",
        "away_team_id",
        "home_team_id",
        "doubleheader",
        "game_number",
    }
)
_SOURCE_FINAL_GAME_KEYS = frozenset(
    {
        "game_id",
        "season",
        "official_date",
        "game_type",
        "status_code",
        "status_detail",
        "away_team_id",
        "home_team_id",
        "away_score",
        "home_score",
    }
)
_POSTPONED_STATUS_CODES = frozenset({"D", "DI", "DR"})
_CANCELLED_STATUS_CODES = frozenset({"C", "CI", "CR"})
_ALLOWED_CANDIDATE_EXCLUSION_REASONS = (
    "POSTPONED",
    "CANCELLED",
    "START_TIME_MISSING",
    "INSUFFICIENT_BOTH_HISTORY",
    "INSUFFICIENT_AWAY_HISTORY",
    "INSUFFICIENT_HOME_HISTORY",
)
_MINIMUM_HISTORY_GAMES_PER_TEAM = 10

_RESERVED_MARKER_KEYS = frozenset(
    {
        "marker_schema_version",
        "batch_id",
        "slot_key",
        "target_official_date",
        "reserved_at_utc",
        "shadow_protocol_sha256",
        "execution_manifest_sha256",
        "runtime_code_commit",
    }
)
_FAILED_MARKER_KEYS = frozenset(
    {
        "marker_schema_version",
        "batch_id",
        "slot_key",
        "target_official_date",
        "failed_at_utc",
        "stage",
        "error_type",
        "error_message",
        "shadow_protocol_sha256",
        "execution_manifest_sha256",
        "runtime_code_commit",
    }
)
_RAW_REMOTE_EVIDENCE_KEYS = frozenset(
    {
        "evidence_schema_version",
        "request_url",
        "request_method",
        "application_request_headers",
        "effective_url",
        "response_status_code",
        "response_redirect_count",
        "selected_response_headers",
        "response_received_at_utc",
        "response_body_base64",
        "response_body_sha256",
    }
)
_SELECTED_REMOTE_RESPONSE_HEADER_KEYS = (
    "date",
    "content-type",
    "etag",
    "x-github-request-id",
)
_GITHUB_COMPARE_STATUS_VALUES = frozenset({"ahead", "identical"})
_COMPLETED_MARKER_KEYS = frozenset(
    {
        "marker_schema_version",
        "batch_id",
        "receipt_path",
        "receipt_sha256",
        "completed_at_utc",
    }
)
_RECEIPT_TOP_LEVEL_KEYS = frozenset(
    {
        "receipt_schema_version",
        "batch",
        "activation",
        "times",
        "schedule_http_response",
        "lineage",
        "source",
        "counts",
        "output_hashes",
        "model_invariants",
        "negative_attestations",
        "runtime_versions",
    }
)
_RECEIPT_SECTION_KEYS = {
    "batch": frozenset(
        {
            "batch_id",
            "slot_key",
            "target_official_date",
            "status",
            "earliest_predicted_scheduled_start_utc",
        }
    ),
    "activation": frozenset(
        {
            "execution_manifest_introduction_commit",
            "activation_introduction_commit",
            "activation_path",
            "activation_sha256",
            "activation_verified_at_utc",
            "activation_remote_reverified_at_utc",
            "activation_remote_ref",
            "activation_remote_reverification_query_url",
            "activation_remote_reverification_effective_url",
            "activation_remote_reverification_status_code",
            "activation_remote_reverification_redirect_count",
            "activation_remote_reverification_response_received_at_utc",
            "activation_remote_reverification_response_body_sha256",
            "activation_remote_reverification_evidence_path",
            "activation_remote_reverification_evidence_sha256",
            "minimum_target_official_date",
        }
    ),
    "times": frozenset(
        {
            "started_at_utc",
            "reserved_at_utc",
            "schedule_observed_at_utc",
            "information_cutoff_utc",
            "issued_at_utc",
            "receipt_finalized_at_utc",
            "mlb_http_date_utc",
            "mlb_http_response_received_at_utc",
            "clock_skew_seconds",
            "schedule_age_seconds",
        }
    ),
    "schedule_http_response": frozenset(
        {
            "effective_url",
            "status_code",
            "redirect_count",
            "date_header_raw",
            "date_header_utc",
            "received_at_utc",
            "body_sha256",
        }
    ),
    "lineage": frozenset(
        {
            "runtime_code_commit",
            "shadow_service_module_sha256",
            "shadow_protocol_sha256",
            "execution_manifest_sha256",
            "model_artifact_sha256",
            "artifact_manifest_sha256",
            "model_protocol_sha256",
            "evaluation_protocol_sha256",
            "evaluation_report_sha256",
            "evaluation_results_commit",
        }
    ),
    "source": frozenset(
        {
            "sqlite_snapshot_sha256",
            "sqlite_snapshot_size_bytes",
            "source_snapshot_path",
            "source_snapshot_sha256",
            "schedule_ingestion_run_id",
            "schedule_source",
            "schedule_requested_start_date",
            "schedule_requested_end_date",
            "schedule_game_types",
            "schedule_request_parameters_json",
            "schedule_ingestion_completed_at_utc",
            "schedule_raw_archive_path",
            "schedule_raw_archive_sha256",
        }
    ),
    "counts": frozenset(
        {
            "schedule_games",
            "eligible_games",
            "predicted_games",
            "excluded_games_by_reason",
        }
    ),
    "output_hashes": frozenset(
        {
            "activation_reverification_evidence_sha256",
            "candidate_ledger_sha256",
            "features_sha256",
            "predictions_sha256",
        }
    ),
    "model_invariants": frozenset(
        {
            "fit_calls",
            "partial_fit_calls",
            "recalibration_calls",
            "threshold_tuning_calls",
            "feature_selection_calls",
            "predict_proba_calls",
            "artifact_state_sha256_before",
            "artifact_state_sha256_after",
            "artifact_state_unchanged",
            "warning_policy_id",
            "approved_compatibility_warning_count",
            "unexpected_warning_count",
        }
    ),
    "negative_attestations": frozenset(
        {
            "ALL_TARGET_GAMES_UNSTARTED_AT_INFORMATION_CUTOFF",
            "TARGET_OUTCOMES_NOT_AVAILABLE_AT_INFORMATION_CUTOFF",
            "TARGET_SCORE_VALUES_REDACTED_FROM_TRACKED_TARGET_ROWS",
            "TARGET_SCORES_NOT_USED_AS_FEATURES",
            "ODDS_NOT_READ",
            "BETTING_RECOMMENDATIONS_NOT_COMPUTED",
            "SEASON_2026_METRICS_NOT_COMPUTED",
        }
    ),
    "runtime_versions": frozenset(
        {
            "python",
            "numpy",
            "pandas",
            "scipy",
            "scikit_learn",
            "joblib",
        }
    ),
}
_EXCLUDED_GAMES_BY_REASON_KEYS = frozenset(
    {
        "POSTPONED",
        "CANCELLED",
        "START_TIME_MISSING",
        "INSUFFICIENT_BOTH_HISTORY",
        "INSUFFICIENT_AWAY_HISTORY",
        "INSUFFICIENT_HOME_HISTORY",
    }
)
_BATCH_STATUS_DOMAIN = frozenset(
    {
        "COMPLETED_WITH_PREDICTIONS",
        "COMPLETED_NO_ELIGIBLE_GAMES",
    }
)


class ShadowPredictionError(RuntimeError):
    """Erreur qui interdit de produire meme un apercu fiable."""


class ShadowPublicationConflictError(ShadowPredictionError):
    """Refus attendu lorsqu'un chemin append-only est deja consomme."""


class ShadowPredictionSlotConsumedError(ShadowPublicationConflictError):
    """Refus definitif lorsqu'un slot journalier n'est plus absent."""


class ShadowPredictionSlotState(str, Enum):
    """Etats fermes d'un creneau journalier v2 deja inspecte."""

    ABSENT = "ABSENT"
    COMPLETED_EXACT = "COMPLETED_EXACT"
    COMPLETED_MISMATCH = "COMPLETED_MISMATCH"
    FAILED_CONSUMED = "FAILED_CONSUMED"
    INCOMPLETE_CONSUMED = "INCOMPLETE_CONSUMED"


@dataclass(frozen=True, slots=True)
class ShadowPredictionSlotInspection:
    """Resultat local d'inspection; seul un doublon exact rend son recu."""

    state: ShadowPredictionSlotState
    slot_path: Path
    slot_key: str
    batch_id: str
    receipt: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class ShadowPredictionSlotReservation:
    """Preuve locale retournee apres creation exclusive de RESERVED."""

    slot_path: Path
    slot_key: str
    batch_id: str
    reserved_marker: dict[str, Any]
    reserved_marker_sha256: str


@dataclass(frozen=True, slots=True)
class ShadowPredictionSlotFailure:
    """Preuve locale retournee apres publication exclusive de FAILED.json."""

    slot_path: Path
    failed_marker: dict[str, Any]
    failed_marker_sha256: str


@dataclass(frozen=True, slots=True)
class ShadowActivationReverificationEvidence:
    """Preuve GitHub canonique entierement construite avant reservation."""

    activation_introduction_commit: str
    activation_remote_ref: str
    activation_remote_reverified_at_utc: str
    response_received_at_utc: str
    response_body_sha256: str
    raw_evidence: dict[str, Any]
    canonical_json_bytes: bytes
    canonical_gzip_bytes: bytes
    canonical_gzip_sha256: str


@dataclass(frozen=True, slots=True)
class ShadowActivationReverificationPublication:
    """Preuve locale de publication du deuxieme fichier d'un slot v2."""

    slot_path: Path
    evidence_path: Path
    evidence_relative_path: str
    evidence_sha256: str
    activation_introduction_commit: str
    activation_remote_ref: str
    activation_remote_reverified_at_utc: str
    response_received_at_utc: str
    response_body_sha256: str


@dataclass(frozen=True, slots=True)
class ShadowSourceSnapshotPublication:
    """Preuve de publication du troisieme fichier officiel d'un slot."""

    slot_path: Path
    snapshot_path: Path
    snapshot_relative_path: str
    snapshot_sha256: str
    snapshot_size_bytes: int
    sqlite_snapshot_sha256: str
    sqlite_snapshot_size_bytes: int
    schedule_ingestion_run_id: int
    schedule_attempts: int
    information_cutoff_utc: str


@dataclass(frozen=True, slots=True)
class ShadowCandidateFeaturesPublication:
    """Preuve des quatrieme et cinquieme fichiers officiels du slot."""

    slot_path: Path
    candidate_ledger_path: Path
    candidate_ledger_relative_path: str
    candidate_ledger_sha256: str
    candidate_ledger_size_bytes: int
    candidate_row_count: int
    features_path: Path
    features_relative_path: str
    features_sha256: str
    features_size_bytes: int
    feature_row_count: int
    eligible_game_count: int
    excluded_games_by_reason: tuple[tuple[str, int], ...]
    earliest_eligible_scheduled_start_utc: str | None


@dataclass(frozen=True, slots=True)
class ShadowModelPrerequisites:
    """Octets controles, pas une autorisation d'execution ou de chargement.

    Le futur orchestrateur doit encore verifier le manifeste d'execution,
    l'activation, le slot et son environnement au moment du chargement.
    Aucun objet pickle n'est interprete pour obtenir cette preuve.
    """

    artifact_path: Path
    artifact_relative_path: str
    artifact_sha256: str
    artifact_size_bytes: int
    artifact_bytes: bytes = field(repr=False)
    artifact_manifest_sha256: str
    model_protocol_sha256: str
    shadow_protocol_sha256: str
    runtime_versions: tuple[tuple[str, str], ...]
    validation_scope: str = field(
        default="FROZEN_MODEL_FILES_AND_INSTALLED_RUNTIME_ONLY", init=False
    )
    runtime_versions_source: str = field(
        default="PYTHON_AND_INSTALLED_DISTRIBUTION_METADATA", init=False
    )
    execution_manifest_verified: bool = field(default=False, init=False)
    activation_verified: bool = field(default=False, init=False)
    model_deserialized: bool = field(default=False, init=False)
    predictions_computed: bool = field(default=False, init=False)
    execution_ready: bool = field(default=False, init=False)


@dataclass(frozen=True, slots=True)
class ShadowLoadedModel:
    """Objet charge, sans autorisation de prediction ni garantie d'immuabilite.

    L'enveloppe est figee, mais l'estimateur qu'elle contient reste un objet
    mutable. Ses empreintes d'etat devront encadrer la future prediction.
    """

    artifact: Any = field(repr=False)
    artifact_sha256: str
    runtime_versions: tuple[tuple[str, str], ...]
    approved_deserialization_warning_count: int
    warning_policy_id: str = field(default=_MODEL_WARNING_POLICY_ID, init=False)
    model_deserialized: bool = field(default=True, init=False)
    predictions_computed: bool = field(default=False, init=False)
    execution_manifest_verified: bool = field(default=False, init=False)
    activation_verified: bool = field(default=False, init=False)
    execution_ready: bool = field(default=False, init=False)


@dataclass(frozen=True, slots=True)
class ShadowModelStateSnapshot:
    """Mesure interne liee a une instance, pas une preuve d'execution.

    La reference retient l'objet d'origine ; elle n'en fait pas une copie.
    Seule l'empreinte est une photographie de l'etat serialise a cet instant.
    """

    loaded_model: ShadowLoadedModel = field(repr=False, compare=False)
    artifact_state_sha256: str
    runtime_versions: tuple[tuple[str, str], ...]
    approved_state_serialization_warning_count: int
    warning_policy_id: str = field(default=_MODEL_WARNING_POLICY_ID, init=False)


@dataclass(frozen=True, slots=True)
class ShadowModelStateVerification:
    """Comparaison de deux etats, sans attester un appel de prediction.

    Le compteur concerne uniquement les deux serialisations d'etat.
    Celui de la deserialisation reste sur ShadowLoadedModel, sans doublon.
    """

    artifact_state_sha256_before: str
    artifact_state_sha256_after: str
    approved_state_serialization_warning_count: int
    artifact_state_unchanged: bool = field(default=True, init=False)
    warning_policy_id: str = field(default=_MODEL_WARNING_POLICY_ID, init=False)
    execution_manifest_verified: bool = field(default=False, init=False)
    activation_verified: bool = field(default=False, init=False)
    execution_ready: bool = field(default=False, init=False)


@dataclass(frozen=True, slots=True)
class ShadowModelProbabilities:
    """Resultat interne en memoire, ni CSV officiel ni autorisation de slot.

    Les paires suivent les lignes figees : (p_away, p_home). L'orchestrateur
    devra encore prouver leur origine, l'activation, les delais et l'unicite
    du lot. Aucun de ces faits n'est atteste par cette primitive.
    """

    feature_rows: tuple[tuple[object, ...], ...]
    probabilities: tuple[tuple[float, float], ...]
    artifact_sha256: str
    artifact_state_sha256_before: str
    artifact_state_sha256_after: str
    runtime_versions: tuple[tuple[str, str], ...]
    approved_deserialization_warning_count: int
    approved_state_serialization_warning_count: int
    approved_compatibility_warning_count: int
    predict_proba_calls: int = field(default=1, init=False)
    artifact_state_unchanged: bool = field(default=True, init=False)
    unexpected_warning_count: int = field(default=0, init=False)
    warning_policy_id: str = field(default=_MODEL_WARNING_POLICY_ID, init=False)
    execution_manifest_verified: bool = field(default=False, init=False)
    activation_verified: bool = field(default=False, init=False)
    official_prediction_created: bool = field(default=False, init=False)
    execution_ready: bool = field(default=False, init=False)


@dataclass(frozen=True, slots=True)
class ShadowPredictionsPublication:
    """Preuve du sixieme fichier officiel, sans recu ni fin de slot.

    Le contenu du CSV est deja relu apres publication. La preuve reste
    intermediaire : seul le futur recu suivi de COMPLETED rendra le lot
    terminal et exploitable comme publication shadow complete.
    """

    slot_path: Path
    predictions_path: Path
    predictions_relative_path: str
    predictions_sha256: str
    predictions_size_bytes: int
    prediction_row_count: int
    issued_at_utc: str
    earliest_predicted_scheduled_start_utc: str | None
    model_version: str
    artifact_sha256: str
    protocol_sha256: str
    code_commit: str
    artifact_state_sha256_before: str | None
    artifact_state_sha256_after: str | None
    runtime_versions: tuple[tuple[str, str], ...]
    approved_deserialization_warning_count: int
    approved_state_serialization_warning_count: int
    approved_compatibility_warning_count: int
    predict_proba_calls: int
    artifact_state_unchanged: bool = field(default=True, init=False)
    unexpected_warning_count: int = field(default=0, init=False)
    warning_policy_id: str = field(default=_MODEL_WARNING_POLICY_ID, init=False)
    official_prediction_created: bool = field(default=True, init=False)
    receipt_created: bool = field(default=False, init=False)
    slot_completed: bool = field(default=False, init=False)


@dataclass(frozen=True, slots=True)
class ShadowReceiptExecutionContext:
    """Valeurs issues du futur preflight fige, jamais lues par le recu.

    Cette preuve interne ne constitue pas elle-meme une autorisation publique.
    Le futur orchestrateur sera le seul producteur admis apres verification de
    l'introduction immuable du manifeste et de l'activation distante.
    """

    started_at_utc: str
    execution_manifest_introduction_commit: str
    activation_sha256: str
    activation_verified_at_utc: str
    minimum_target_official_date: str
    shadow_service_module_sha256: str
    runtime_versions: tuple[tuple[str, str], ...]
    execution_manifest_verified: bool = field(default=True, init=False)
    activation_verified: bool = field(default=True, init=False)
    execution_ready: bool = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class ShadowReceiptPublication:
    """Preuve du septieme fichier, encore non terminal sans COMPLETED."""

    slot_path: Path
    receipt_path: Path
    receipt_relative_path: str
    receipt_sha256: str
    receipt_size_bytes: int
    receipt_finalized_at_utc: str
    batch_status: str
    schedule_game_count: int
    eligible_game_count: int
    predicted_game_count: int
    earliest_predicted_scheduled_start_utc: str | None
    execution_manifest_sha256: str
    protocol_sha256: str
    artifact_sha256: str
    official_prediction_created: bool
    receipt_created: bool = field(default=True, init=False)
    slot_completed: bool = field(default=False, init=False)


@dataclass(frozen=True, slots=True)
class ShadowCompletionPublication:
    """Preuve du huitieme fichier et de la fermeture terminale du slot."""

    slot_path: Path
    completed_path: Path
    completed_relative_path: str
    completed_sha256: str
    completed_size_bytes: int
    completed_at_utc: str
    batch_id: str
    receipt_relative_path: str
    receipt_sha256: str
    batch_status: str
    earliest_predicted_scheduled_start_utc: str | None
    local_completion_lead_seconds: int | None
    official_prediction_created: bool
    receipt_created: bool = field(default=True, init=False)
    slot_completed: bool = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class _ShadowReceiptPredecessors:
    activation_evidence: dict[str, Any]
    activation_bytes: bytes
    source_snapshot: dict[str, Any]
    source_bytes: bytes
    candidate_bytes: bytes
    features_bytes: bytes
    predictions_bytes: bytes
    candidate_rows: tuple[tuple[str, ...], ...]
    feature_rows: tuple[tuple[object, ...], ...]
    prediction_rows: tuple[tuple[str, ...], ...]
    exclusions: tuple[tuple[str, int], ...]
    target_official_date: str
    information_cutoff_utc: str


@dataclass(frozen=True, slots=True)
class _ShadowCompletionPredecessors:
    receipt: dict[str, Any]
    receipt_bytes: bytes
    receipt_finalized_at_utc: str
    batch_status: str
    earliest_predicted_scheduled_start_utc: str | None
    receipt_predecessors: _ShadowReceiptPredecessors


def _validate_json_value(
    value: object,
    *,
    path: str = "$",
    ancestors: set[int] | None = None,
) -> None:
    """Refuse les extensions Python ambigues avant canonicalisation JSON."""
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ShadowPredictionError(
                f"Valeur JSON non finie interdite a {path}."
            )
        return

    if value_type not in (list, dict):
        raise ShadowPredictionError(
            f"Type non JSON interdit a {path} : {value_type.__name__}."
        )

    current_id = id(value)
    active_ancestors = ancestors if ancestors is not None else set()
    if current_id in active_ancestors:
        raise ShadowPredictionError(
            f"Reference circulaire interdite a {path}."
        )
    active_ancestors.add(current_id)
    try:
        if value_type is list:
            for index, item in enumerate(value):
                _validate_json_value(
                    item,
                    path=f"{path}[{index}]",
                    ancestors=active_ancestors,
                )
            return

        for key, item in value.items():
            if type(key) is not str:
                raise ShadowPredictionError(
                    f"Cle JSON non textuelle interdite a {path}."
                )
            _validate_json_value(
                item,
                path=f"{path}.{key}",
                ancestors=active_ancestors,
            )
    finally:
        active_ancestors.remove(current_id)


def _canonical_json_bytes(value: object) -> bytes:
    """Applique exactement la canonicalisation de hachage du protocole v2."""
    _validate_json_value(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ShadowPredictionError(
            "Valeur impossible a canonicaliser en JSON UTF-8."
        ) from error


def _canonical_json_file_bytes(value: object) -> bytes:
    """Ajoute l'unique LF terminal impose aux fichiers JSON v2."""
    return _canonical_json_bytes(value) + b"\n"


def _csv_field_text(value: object, *, row_index: int, column: str) -> str:
    """Convertit uniquement les scalaires dont le rendu CSV est explicite."""
    if value is None:
        return ""
    if type(value) is str:
        text = value
    elif type(value) is int:
        text = str(value)
    else:
        raise ShadowPredictionError(
            "Valeur CSV non canonique a la ligne "
            f"{row_index}, colonne {column!r} : "
            f"{type(value).__name__}."
        )
    if "\r" in text:
        raise ShadowPredictionError(
            "Retour chariot interdit dans une valeur CSV canonique a la "
            f"ligne {row_index}, colonne {column!r}."
        )
    return text


def _canonical_csv_bytes(
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> bytes:
    """Serialise un tableau deja ordonne en CSV UTF-8/LF a largeur fixe."""
    if type(columns) not in (list, tuple) or not columns:
        raise ShadowPredictionError(
            "Les colonnes CSV doivent etre une liste ordonnee non vide."
        )
    canonical_columns: list[str] = []
    for index, column in enumerate(columns):
        if type(column) is not str or not column or "\r" in column:
            raise ShadowPredictionError(
                f"Nom de colonne CSV invalide a l'index {index}."
            )
        canonical_columns.append(column)
    if len(set(canonical_columns)) != len(canonical_columns):
        raise ShadowPredictionError(
            "Les noms de colonnes CSV doivent etre uniques."
        )

    if type(rows) not in (list, tuple):
        raise ShadowPredictionError(
            "Les lignes CSV doivent etre une liste ordonnee."
        )

    canonical_rows: list[list[str]] = []
    for row_index, row in enumerate(rows, start=1):
        if type(row) not in (list, tuple):
            raise ShadowPredictionError(
                f"La ligne CSV {row_index} doit etre une sequence ordonnee."
            )
        if len(row) != len(canonical_columns):
            raise ShadowPredictionError(
                f"Largeur CSV invalide a la ligne {row_index} : "
                f"{len(row)}, attendu {len(canonical_columns)}."
            )
        canonical_rows.append(
            [
                _csv_field_text(
                    value,
                    row_index=row_index,
                    column=canonical_columns[column_index],
                )
                for column_index, value in enumerate(row)
            ]
        )

    destination = io.StringIO(newline="")
    writer = csv.writer(
        destination,
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    try:
        writer.writerow(canonical_columns)
        writer.writerows(canonical_rows)
        return destination.getvalue().encode("utf-8")
    except (csv.Error, UnicodeEncodeError) as error:
        raise ShadowPredictionError(
            "Valeur impossible a serialiser en CSV UTF-8 canonique."
        ) from error


def _canonical_gzip_bytes(payload: bytes) -> bytes:
    """Compresse des octets en memoire avec l'en-tete gzip fige v2."""
    if type(payload) is not bytes:
        raise ShadowPredictionError(
            "Le contenu gzip canonique doit etre fourni en octets exacts."
        )
    destination = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=destination,
        mtime=0,
    ) as gzip_file:
        gzip_file.write(payload)
    return destination.getvalue()


def _fsync_parent_directory(directory: Path) -> None:
    """Synchronise l'entree de repertoire sur les plateformes POSIX."""
    if os.name == "nt":
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_descriptor = os.open(directory, flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _publish_exclusive_verified(destination: Path, content: bytes) -> str:
    """Publie une fois, synchronise, relit et renvoie le SHA-256 exact.

    Le parent doit deja exister. Un lien physique cree la destination sans
    aucune possibilite de remplacement, meme si deux processus publient au
    meme instant. Une destination publiee n'est jamais supprimee par ce
    helper, y compris lorsqu'une verification ulterieure echoue.
    """
    if not isinstance(destination, Path):
        raise ShadowPredictionError(
            "La destination append-only doit etre un objet Path."
        )
    if type(content) is not bytes:
        raise ShadowPredictionError(
            "Le contenu append-only doit etre fourni en octets exacts."
        )
    if not destination.name:
        raise ShadowPredictionError(
            "La destination append-only doit nommer un fichier."
        )

    parent = destination.parent
    parent_mode = _lstat_mode(parent)
    if not isinstance(parent_mode, int) or not stat.S_ISDIR(parent_mode):
        raise ShadowPredictionError(
            "Le repertoire parent doit exister, etre local et ne pas etre "
            "symbolique avant toute publication."
        )

    expected_sha256 = hashlib.sha256(content).hexdigest()
    temporary_path: Path | None = None
    published = False

    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary_path = Path(temporary_name)

        try:
            with os.fdopen(descriptor, "wb") as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

        try:
            os.link(temporary_path, destination)
        except FileExistsError as error:
            raise ShadowPublicationConflictError(
                f"La destination append-only existe deja : {destination}."
            ) from error

        published = True
        temporary_path.unlink()
        temporary_path = None
        _fsync_parent_directory(parent)

        persisted = destination.read_bytes()
        persisted_sha256 = hashlib.sha256(persisted).hexdigest()
        if persisted != content or persisted_sha256 != expected_sha256:
            raise ShadowPredictionError(
                "La relecture append-only ne correspond pas aux octets "
                "publies."
            )
        return expected_sha256
    except (ShadowPredictionError, OSError):
        raise
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                if not published:
                    raise


def _format_feature_rate(value: object) -> str:
    """Formate un taux fini non negatif avec exactement six decimales."""
    if type(value) not in (int, float):
        raise ShadowPredictionError(
            "Un taux de variable doit etre un nombre JSON, hors booleen."
        )
    if type(value) is float and not math.isfinite(value):
        raise ShadowPredictionError(
            "Un taux de variable doit etre fini."
        )
    if value < 0:
        raise ShadowPredictionError(
            "Un taux de variable doit etre non negatif."
        )
    rendered = format(value, ".6f")
    if not _FIXED_6_PATTERN.fullmatch(rendered):
        raise ShadowPredictionError(
            "Un taux de variable doit avoir un rendu decimal non negatif."
        )
    return rendered


def _format_probability_float(value: object) -> str:
    """Formate une probabilite finie de [0, 1] avec la regle .17g."""
    if type(value) not in (int, float):
        raise ShadowPredictionError(
            "Une probabilite doit etre un nombre JSON, hors booleen."
        )
    if type(value) is float and not math.isfinite(value):
        raise ShadowPredictionError("Une probabilite doit etre finie.")
    if value < 0 or value > 1:
        raise ShadowPredictionError(
            "Une probabilite doit appartenir a l'intervalle [0, 1]."
        )
    rendered = format(value, ".17g")
    if rendered.startswith("-"):
        raise ShadowPredictionError(
            "Une probabilite ne peut pas utiliser un zero negatif."
        )
    return rendered


def _sha256_identifier(preimage: object) -> str:
    """Hache une preimage JSON-array sans saut de ligne terminal."""
    if type(preimage) is not list or not preimage:
        raise ShadowPredictionError(
            "La preimage d'un identifiant doit etre un tableau JSON non vide."
        )
    if type(preimage[0]) is not str:
        raise ShadowPredictionError(
            "Le domaine d'un identifiant doit etre une chaine JSON."
        )
    return hashlib.sha256(_canonical_json_bytes(preimage)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or not _SHA256_PATTERN.fullmatch(value):
        raise ShadowPredictionError(
            f"{field} doit contenir exactement 64 caracteres hexadecimaux "
            "ASCII minuscules."
        )
    return value


def _require_git_commit(value: object, *, field: str) -> str:
    if type(value) is not str or not _GIT_COMMIT_PATTERN.fullmatch(value):
        raise ShadowPredictionError(
            f"{field} doit contenir exactement 40 caracteres hexadecimaux "
            "ASCII minuscules."
        )
    return value


def _require_nonempty_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ShadowPredictionError(
            f"{field} doit etre une chaine non vide."
        )
    return value


def _require_integer(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise ShadowPredictionError(
            f"{field} doit etre un entier JSON; les booleens sont interdits."
        )
    return value


def _require_date_string(value: object, *, field: str) -> str:
    if type(value) is not str or not _CANONICAL_DATE_PATTERN.fullmatch(
        value
    ):
        raise ShadowPredictionError(
            f"{field} doit respecter exactement YYYY-MM-DD."
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ShadowPredictionError(
            f"{field} n'est pas une date calendaire valide : {value!r}."
        ) from error
    if parsed.isoformat() != value:
        raise ShadowPredictionError(
            f"{field} doit etre une date ISO canonique."
        )
    return value


def _require_utc_timestamp(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise ShadowPredictionError(
            f"{field} doit etre une chaine RFC3339_SECONDS_Z."
        )
    match = _UTC_TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        raise ShadowPredictionError(
            f"{field} doit respecter exactement RFC3339_SECONDS_Z."
        )
    try:
        datetime(*(int(component) for component in match.groups()))
    except ValueError as error:
        raise ShadowPredictionError(
            f"{field} n'est pas un instant UTC valide : {value!r}."
        ) from error
    return value


def _utc_now() -> datetime:
    """Retourne l'instant local UTC; remplace uniquement en test."""
    return datetime.now(timezone.utc)


def _format_utc_seconds(value: object, *, field: str) -> str:
    """Canonise un datetime UTC en RFC3339_SECONDS_Z."""
    if type(value) is not datetime:
        raise ShadowPredictionError(
            f"{field} doit etre un datetime UTC conscient."
        )
    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as error:
        raise ShadowPredictionError(
            f"{field} n'est pas un datetime UTC valide."
        ) from error
    if offset is None or offset.total_seconds() != 0:
        raise ShadowPredictionError(
            f"{field} doit utiliser explicitement le fuseau UTC."
        )
    rendered = value.astimezone(timezone.utc).replace(
        microsecond=0
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return _require_utc_timestamp(rendered, field=field)


def _parse_imf_fixdate_gmt(value: object, *, field: str) -> str:
    """Parse strictement un en-tete HTTP Date au format IMF-fixdate GMT."""
    if type(value) is not str or not value:
        raise ShadowPredictionError(
            f"{field} doit etre un en-tete HTTP Date IMF-fixdate GMT."
        )
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ShadowPredictionError(
            f"{field} n'est pas un en-tete HTTP Date valide."
        ) from error
    if parsed is None or parsed.utcoffset() is None:
        raise ShadowPredictionError(
            f"{field} doit porter explicitement le fuseau GMT."
        )
    parsed_utc = parsed.astimezone(timezone.utc).replace(microsecond=0)
    if format_datetime(parsed_utc, usegmt=True) != value:
        raise ShadowPredictionError(
            f"{field} doit respecter exactement IMF-fixdate GMT."
        )
    return parsed_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _optional_response_header(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ShadowPredictionError(
            f"{field} doit etre une chaine non vide ou JSON null."
        )
    return value


def _validate_github_compare_payload(
    raw_content: bytes,
    *,
    expected_commit: str,
) -> None:
    """Exige que GitHub nomme le commit attendu comme base et merge-base."""
    if type(raw_content) is not bytes or not raw_content:
        raise ShadowPredictionError(
            "GitHub a renvoye une preuve distante vide ou non binaire."
        )
    try:
        payload = json.loads(
            raw_content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (
        UnicodeError,
        ValueError,
        ShadowPredictionError,
        RecursionError,
    ) as error:
        raise ShadowPredictionError(
            "La preuve distante GitHub n'est pas un JSON non ambigu."
        ) from error
    if type(payload) is not dict:
        raise ShadowPredictionError(
            "La preuve distante GitHub doit etre un objet JSON."
        )
    for key in ("base_commit", "merge_base_commit"):
        commit_object = payload.get(key)
        if (
            type(commit_object) is not dict
            or type(commit_object.get("sha")) is not str
            or commit_object["sha"] != expected_commit
        ):
            raise ShadowPredictionError(
                "La reponse GitHub ne nomme pas le commit d'activation "
                f"attendu dans {key}.sha."
            )
    status_value = payload.get("status")
    if (
        type(status_value) is not str
        or status_value not in _GITHUB_COMPARE_STATUS_VALUES
    ):
        raise ShadowPredictionError(
            "Le statut de comparaison GitHub doit etre ahead ou identical."
        )


def _validate_activation_reverification_evidence(
    evidence: ShadowActivationReverificationEvidence,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Revérifie toutes les liaisons d'une preuve preparee en memoire."""
    if type(evidence) is not ShadowActivationReverificationEvidence:
        raise ShadowPredictionError(
            "Une preuve de reverification d'activation exacte est requise."
        )
    expected_commit = _require_git_commit(
        evidence.activation_introduction_commit,
        field="activation_introduction_commit",
    )
    if evidence.activation_remote_ref != GITHUB_REMOTE_REF:
        raise ShadowPredictionError(
            "La preuve distante doit viser exactement refs/heads/main."
        )
    reverified_at = _require_utc_timestamp(
        evidence.activation_remote_reverified_at_utc,
        field="activation_remote_reverified_at_utc",
    )
    received_at = _require_utc_timestamp(
        evidence.response_received_at_utc,
        field="response_received_at_utc",
    )
    body_sha256 = _require_sha256(
        evidence.response_body_sha256,
        field="response_body_sha256",
    )
    gzip_sha256 = _require_sha256(
        evidence.canonical_gzip_sha256,
        field="canonical_gzip_sha256",
    )
    raw = evidence.raw_evidence
    if not _has_exact_keys(raw, _RAW_REMOTE_EVIDENCE_KEYS):
        raise ShadowPredictionError(
            "La preuve distante ne respecte pas le schema exact v2."
        )
    assert isinstance(raw, dict)

    request_url = GITHUB_COMPARE_URL_TEMPLATE.format(
        expected_commit=expected_commit
    )
    exact_values: dict[str, object] = {
        "evidence_schema_version": 1,
        "request_url": request_url,
        "request_method": "GET",
        "application_request_headers": dict(GITHUB_REQUEST_HEADERS),
        "effective_url": request_url,
        "response_status_code": 200,
        "response_redirect_count": 0,
        "response_received_at_utc": received_at,
        "response_body_sha256": body_sha256,
    }
    for field, expected in exact_values.items():
        actual = raw.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise ShadowPredictionError(
                f"Valeur distante invalide pour {field}."
            )

    selected_headers = raw.get("selected_response_headers")
    if (
        type(selected_headers) is not dict
        or frozenset(selected_headers)
        != frozenset(_SELECTED_REMOTE_RESPONSE_HEADER_KEYS)
    ):
        raise ShadowPredictionError(
            "Les en-tetes GitHub selectionnes ne respectent pas l'ordre "
            "et le schema exacts."
        )
    date_header = _optional_response_header(
        selected_headers.get("date"),
        field="selected_response_headers.date",
    )
    if date_header is None:
        raise ShadowPredictionError(
            "La preuve GitHub doit contenir un en-tete Date."
        )
    parsed_date = _parse_imf_fixdate_gmt(
        date_header,
        field="selected_response_headers.date",
    )
    if parsed_date != reverified_at:
        raise ShadowPredictionError(
            "L'instant de reverification ne correspond pas a l'en-tete "
            "Date de GitHub."
        )
    for header_name in _SELECTED_REMOTE_RESPONSE_HEADER_KEYS[1:]:
        _optional_response_header(
            selected_headers.get(header_name),
            field=f"selected_response_headers.{header_name}",
        )

    encoded_body = raw.get("response_body_base64")
    if type(encoded_body) is not str or not encoded_body:
        raise ShadowPredictionError(
            "Le corps GitHub doit etre conserve en base64 non vide."
        )
    try:
        raw_content = base64.b64decode(
            encoded_body.encode("ascii"),
            validate=True,
        )
    except (UnicodeError, ValueError) as error:
        raise ShadowPredictionError(
            "Le corps GitHub conserve n'est pas un base64 canonique."
        ) from error
    if base64.b64encode(raw_content).decode("ascii") != encoded_body:
        raise ShadowPredictionError(
            "Le corps GitHub doit utiliser l'encodage base64 canonique."
        )
    if hashlib.sha256(raw_content).hexdigest() != body_sha256:
        raise ShadowPredictionError(
            "L'empreinte du corps GitHub conserve est incoherente."
        )
    _validate_github_compare_payload(
        raw_content,
        expected_commit=expected_commit,
    )

    canonical_json = _canonical_json_file_bytes(raw)
    if (
        type(evidence.canonical_json_bytes) is not bytes
        or evidence.canonical_json_bytes != canonical_json
    ):
        raise ShadowPredictionError(
            "Les octets JSON de la preuve distante ne sont pas canoniques."
        )
    canonical_gzip = _canonical_gzip_bytes(canonical_json)
    if (
        type(evidence.canonical_gzip_bytes) is not bytes
        or evidence.canonical_gzip_bytes != canonical_gzip
        or hashlib.sha256(canonical_gzip).hexdigest() != gzip_sha256
    ):
        raise ShadowPredictionError(
            "L'archive gzip de reverification n'est pas canonique."
        )
    return raw, canonical_json, canonical_gzip


def fetch_activation_reverification_evidence(
    activation_introduction_commit: str,
) -> ShadowActivationReverificationEvidence:
    """Obtient et fige en memoire la preuve GitHub avant reservation.

    Cette fonction n'effectue qu'un GET HTTPS anonyme vers l'URL GitHub
    exacte du protocole. Elle ne lit ni MLB, ni SQLite, ni modele, ni fichier
    du projet et ne publie aucun octet.
    """
    expected_commit = _require_git_commit(
        activation_introduction_commit,
        field="activation_introduction_commit",
    )
    request_url = GITHUB_COMPARE_URL_TEMPLATE.format(
        expected_commit=expected_commit
    )
    try:
        response = requests.get(
            request_url,
            headers=dict(GITHUB_REQUEST_HEADERS),
            timeout=GITHUB_REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
            verify=True,
        )
    except requests.RequestException as error:
        raise ShadowPredictionError(
            "Impossible d'obtenir la preuve distante d'activation GitHub."
        ) from error
    response_received_at = _format_utc_seconds(
        _utc_now(),
        field="response_received_at_utc",
    )

    status_code = getattr(response, "status_code", None)
    if type(status_code) is not int or status_code != 200:
        raise ShadowPredictionError(
            "GitHub doit repondre exactement avec le statut HTTP 200."
        )
    history = getattr(response, "history", None)
    if type(history) is not list or history:
        raise ShadowPredictionError(
            "La preuve GitHub ne doit contenir aucune redirection."
        )
    effective_url = getattr(response, "url", None)
    if type(effective_url) is not str or effective_url != request_url:
        raise ShadowPredictionError(
            "L'URL GitHub effective doit etre identique a l'URL demandee."
        )
    raw_content = getattr(response, "content", None)
    if type(raw_content) is not bytes or not raw_content:
        raise ShadowPredictionError(
            "GitHub a renvoye une preuve distante vide ou non binaire."
        )
    _validate_github_compare_payload(
        raw_content,
        expected_commit=expected_commit,
    )

    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        raise ShadowPredictionError(
            "La reponse GitHub ne contient pas de table d'en-tetes."
        )
    selected_headers: dict[str, str | None] = {}
    for header_name in _SELECTED_REMOTE_RESPONSE_HEADER_KEYS:
        selected_headers[header_name] = _optional_response_header(
            headers.get(header_name),
            field=f"selected_response_headers.{header_name}",
        )
    date_header = selected_headers["date"]
    if date_header is None:
        raise ShadowPredictionError(
            "La reponse GitHub doit contenir un en-tete Date."
        )
    reverified_at = _parse_imf_fixdate_gmt(
        date_header,
        field="selected_response_headers.date",
    )
    body_sha256 = hashlib.sha256(raw_content).hexdigest()
    raw_evidence: dict[str, Any] = {
        "evidence_schema_version": 1,
        "request_url": request_url,
        "request_method": "GET",
        "application_request_headers": dict(GITHUB_REQUEST_HEADERS),
        "effective_url": effective_url,
        "response_status_code": status_code,
        "response_redirect_count": len(history),
        "selected_response_headers": selected_headers,
        "response_received_at_utc": response_received_at,
        "response_body_base64": base64.b64encode(raw_content).decode(
            "ascii"
        ),
        "response_body_sha256": body_sha256,
    }
    canonical_json = _canonical_json_file_bytes(raw_evidence)
    canonical_gzip = _canonical_gzip_bytes(canonical_json)
    prepared = ShadowActivationReverificationEvidence(
        activation_introduction_commit=expected_commit,
        activation_remote_ref=GITHUB_REMOTE_REF,
        activation_remote_reverified_at_utc=reverified_at,
        response_received_at_utc=response_received_at,
        response_body_sha256=body_sha256,
        raw_evidence=raw_evidence,
        canonical_json_bytes=canonical_json,
        canonical_gzip_bytes=canonical_gzip,
        canonical_gzip_sha256=hashlib.sha256(canonical_gzip).hexdigest(),
    )
    _validate_activation_reverification_evidence(prepared)
    return prepared


def _require_fixed_6_string(value: object, *, field: str) -> str:
    if type(value) is not str or not _FIXED_6_PATTERN.fullmatch(value):
        raise ShadowPredictionError(
            f"{field} doit etre une chaine decimale non negative avec "
            "exactement six decimales."
        )
    return value


def build_slot_key(
    *,
    shadow_protocol_sha256: str,
    target_official_date: str,
) -> str:
    """Construit l'identifiant v2 d'un slot sans aucune entree-sortie."""
    protocol_hash = _require_sha256(
        shadow_protocol_sha256,
        field="shadow_protocol_sha256",
    )
    target_date = _require_date_string(
        target_official_date,
        field="target_official_date",
    )
    return _sha256_identifier(
        ["shadow_slot_v2", protocol_hash, target_date]
    )


def build_batch_id(
    *,
    slot_key: str,
    execution_manifest_sha256: str,
    model_artifact_sha256: str,
) -> str:
    """Construit l'identifiant v2 d'un lot sans consulter ses fichiers."""
    validated_slot_key = _require_sha256(slot_key, field="slot_key")
    manifest_hash = _require_sha256(
        execution_manifest_sha256,
        field="execution_manifest_sha256",
    )
    model_hash = _require_sha256(
        model_artifact_sha256,
        field="model_artifact_sha256",
    )
    return _sha256_identifier(
        [
            "shadow_batch_v2",
            validated_slot_key,
            manifest_hash,
            model_hash,
        ]
    )


def build_occurrence_key(
    *,
    game_id: int,
    official_date_at_snapshot: str,
    scheduled_start_utc_at_snapshot_or_null: str | None,
) -> str:
    """Construit une occurrence v2; seul ce hachage accepte un debut nul."""
    validated_game_id = _require_integer(game_id, field="game_id")
    official_date = _require_date_string(
        official_date_at_snapshot,
        field="official_date_at_snapshot",
    )
    scheduled_start: str | None
    if scheduled_start_utc_at_snapshot_or_null is None:
        scheduled_start = None
    else:
        scheduled_start = _require_utc_timestamp(
            scheduled_start_utc_at_snapshot_or_null,
            field="scheduled_start_utc_at_snapshot_or_null",
        )
    return _sha256_identifier(
        [
            "shadow_occurrence_v2",
            validated_game_id,
            official_date,
            scheduled_start,
        ]
    )


def build_prediction_id(
    *,
    shadow_protocol_sha256: str,
    game_id: int,
    official_date_at_snapshot: str,
    scheduled_start_utc_at_snapshot: str,
) -> str:
    """Construit l'identifiant v2 d'une prediction admissible."""
    protocol_hash = _require_sha256(
        shadow_protocol_sha256,
        field="shadow_protocol_sha256",
    )
    validated_game_id = _require_integer(game_id, field="game_id")
    official_date = _require_date_string(
        official_date_at_snapshot,
        field="official_date_at_snapshot",
    )
    scheduled_start = _require_utc_timestamp(
        scheduled_start_utc_at_snapshot,
        field="scheduled_start_utc_at_snapshot",
    )
    return _sha256_identifier(
        [
            "shadow_prediction_v2",
            protocol_hash,
            validated_game_id,
            official_date,
            scheduled_start,
        ]
    )


def build_feature_row_sha256(
    *,
    feature_row_values_in_features_columns_exact_order_excluding_feature_row_sha256:
        list[object],
) -> str:
    """Hache les 20 valeurs v2 d'une ligne de variables, dans l'ordre fige."""
    source_values = (
        feature_row_values_in_features_columns_exact_order_excluding_feature_row_sha256
    )
    if type(source_values) is not list:
        raise ShadowPredictionError(
            "Les valeurs de feature row doivent etre un tableau JSON."
        )
    if len(source_values) != len(_FEATURE_ROW_FIELD_NAMES):
        raise ShadowPredictionError(
            "Une feature row v2 doit contenir exactement "
            f"{len(_FEATURE_ROW_FIELD_NAMES)} valeurs."
        )

    values = list(source_values)
    for index in _FEATURE_ROW_SHA256_INDEXES:
        _require_sha256(values[index], field=_FEATURE_ROW_FIELD_NAMES[index])
    for index in _FEATURE_ROW_INTEGER_INDEXES:
        _require_integer(values[index], field=_FEATURE_ROW_FIELD_NAMES[index])
    for index in _FEATURE_ROW_DATE_INDEXES:
        _require_date_string(
            values[index],
            field=_FEATURE_ROW_FIELD_NAMES[index],
        )
    _require_utc_timestamp(
        values[_FEATURE_ROW_TIMESTAMP_INDEX],
        field=_FEATURE_ROW_FIELD_NAMES[_FEATURE_ROW_TIMESTAMP_INDEX],
    )
    for index in _FEATURE_ROW_FIXED_6_INDEXES:
        _require_fixed_6_string(
            values[index],
            field=_FEATURE_ROW_FIELD_NAMES[index],
        )

    return _sha256_identifier(["shadow_feature_row_v2", values])


@dataclass(frozen=True, slots=True)
class ShadowPredictionPreview:
    """Resultat deterministe d'un controle statique, sans execution."""

    mode: str
    shadow_protocol_version: int
    shadow_protocol_path: str
    shadow_protocol_sha256: str
    shadow_protocol_registered_on: str
    shadow_protocol_status: str
    target_official_date: str
    target_season: int
    target_validation_scope: str
    execution_ready: bool
    execution_manifest_read: bool
    activation_read: bool
    model_artifact_read: bool
    model_deserialized: bool
    sqlite_read: bool
    network_request_performed: bool
    output_slot_reserved: bool
    output_files_created: bool
    predictions_computed: bool

    def to_dict(self) -> dict[str, object]:
        """Retourne une representation JSON sans etat ni horodatage."""
        return asdict(self)

    def to_canonical_json(self) -> str:
        """Serialise l'apercu avec un encodage stable entre deux appels."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Refuse une ambiguite JSON au lieu de conserver la derniere cle."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ShadowPredictionError(
                f"Cle JSON dupliquee dans le protocole v2 : {key!r}."
            )
        result[key] = value
    return result


def _require_mapping(
    value: object,
    *,
    description: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ShadowPredictionError(
            f"Objet JSON attendu pour {description}."
        )
    return value


def _require_exact(
    mapping: Mapping[str, Any],
    key: str,
    expected: object,
    *,
    context: str,
) -> None:
    actual = mapping.get(key)
    if type(actual) is not type(expected) or actual != expected:
        raise ShadowPredictionError(
            f"Valeur v2 invalide pour {context}.{key} : "
            f"{actual!r}, attendu {expected!r}."
        )


_PATH_MISSING = object()
_PATH_UNREADABLE = object()


def _lstat_mode(path: Path) -> int | object:
    """Retourne le mode sans accepter lien ou reparse point Windows."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return _PATH_MISSING
    except (OSError, ValueError):
        return _PATH_UNREADABLE
    reparse_attribute = getattr(
        stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        0x0400,
    )
    if getattr(metadata, "st_file_attributes", 0) & reparse_attribute:
        return _PATH_UNREADABLE
    return metadata.st_mode


def _has_exact_keys(
    value: object,
    expected_keys: frozenset[str],
) -> bool:
    return type(value) is dict and frozenset(value) == expected_keys


def _read_canonical_json_object(
    path: Path,
) -> tuple[dict[str, Any], bytes] | None:
    """Lit un fichier JSON regulier, canonique et sans cle dupliquee."""
    mode = _lstat_mode(path)
    if not isinstance(mode, int) or not stat.S_ISREG(mode):
        return None
    try:
        content = path.read_bytes()
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
        if type(payload) is not dict:
            return None
        if _canonical_json_file_bytes(payload) != content:
            return None
    except (
        OSError,
        UnicodeError,
        ValueError,
        ShadowPredictionError,
        RecursionError,
    ):
        return None
    return payload, content


def _valid_reserved_marker(
    marker: Mapping[str, Any],
    *,
    batch_id: str,
    slot_key: str,
    target_official_date: str,
    shadow_protocol_sha256: str,
    execution_manifest_sha256: str,
) -> bool:
    if not _has_exact_keys(marker, _RESERVED_MARKER_KEYS):
        return False
    expected_values = {
        "marker_schema_version": 1,
        "batch_id": batch_id,
        "slot_key": slot_key,
        "target_official_date": target_official_date,
        "shadow_protocol_sha256": shadow_protocol_sha256,
        "execution_manifest_sha256": execution_manifest_sha256,
    }
    if any(
        type(marker.get(key)) is not type(expected)
        or marker.get(key) != expected
        for key, expected in expected_values.items()
    ):
        return False
    runtime_code_commit = marker.get("runtime_code_commit")
    if (
        type(runtime_code_commit) is not str
        or not _GIT_COMMIT_PATTERN.fullmatch(runtime_code_commit)
    ):
        return False
    try:
        _require_utc_timestamp(
            marker.get("reserved_at_utc"),
            field="RESERVED.reserved_at_utc",
        )
    except ShadowPredictionError:
        return False
    return True


def _valid_receipt(
    receipt: Mapping[str, Any],
    *,
    reserved: Mapping[str, Any],
    batch_id: str,
    slot_key: str,
    target_official_date: str,
    shadow_protocol_sha256: str,
    execution_manifest_sha256: str,
    model_artifact_sha256: str,
) -> bool:
    if not _has_exact_keys(receipt, _RECEIPT_TOP_LEVEL_KEYS):
        return False
    if type(receipt.get("receipt_schema_version")) is not int:
        return False
    if receipt.get("receipt_schema_version") != 1:
        return False
    for section_name, expected_keys in _RECEIPT_SECTION_KEYS.items():
        if not _has_exact_keys(receipt.get(section_name), expected_keys):
            return False

    batch = receipt["batch"]
    lineage = receipt["lineage"]
    times = receipt["times"]
    counts = receipt["counts"]
    output_hashes = receipt["output_hashes"]
    source = receipt["source"]
    negative_attestations = receipt["negative_attestations"]
    assert isinstance(batch, dict)
    assert isinstance(lineage, dict)
    assert isinstance(times, dict)
    assert isinstance(counts, dict)
    assert isinstance(output_hashes, dict)
    assert isinstance(source, dict)
    assert isinstance(negative_attestations, dict)

    exact_identity_values = {
        ("batch", "batch_id"): batch_id,
        ("batch", "slot_key"): slot_key,
        ("batch", "target_official_date"): target_official_date,
        (
            "lineage",
            "shadow_protocol_sha256",
        ): shadow_protocol_sha256,
        (
            "lineage",
            "execution_manifest_sha256",
        ): execution_manifest_sha256,
        ("lineage", "model_artifact_sha256"): model_artifact_sha256,
    }
    sections = {"batch": batch, "lineage": lineage}
    for (section_name, field), expected in exact_identity_values.items():
        actual = sections[section_name].get(field)
        if type(actual) is not type(expected) or actual != expected:
            return False

    runtime_code_commit = lineage.get("runtime_code_commit")
    if (
        type(runtime_code_commit) is not str
        or not _GIT_COMMIT_PATTERN.fullmatch(runtime_code_commit)
        or runtime_code_commit != reserved.get("runtime_code_commit")
    ):
        return False
    if times.get("reserved_at_utc") != reserved.get("reserved_at_utc"):
        return False

    status = batch.get("status")
    if type(status) is not str or status not in _BATCH_STATUS_DOMAIN:
        return False
    excluded = counts.get("excluded_games_by_reason")
    if not _has_exact_keys(excluded, _EXCLUDED_GAMES_BY_REASON_KEYS):
        return False
    assert isinstance(excluded, dict)
    count_values = [
        counts.get("schedule_games"),
        counts.get("eligible_games"),
        counts.get("predicted_games"),
        *excluded.values(),
    ]
    if any(type(value) is not int or value < 0 for value in count_values):
        return False
    if counts["predicted_games"] != counts["eligible_games"]:
        return False
    if counts["schedule_games"] != (
        counts["eligible_games"] + sum(excluded.values())
    ):
        return False
    predicted_games = counts["predicted_games"]
    expected_status = (
        "COMPLETED_WITH_PREDICTIONS"
        if predicted_games > 0
        else "COMPLETED_NO_ELIGIBLE_GAMES"
    )
    if status != expected_status:
        return False
    earliest_start = batch.get("earliest_predicted_scheduled_start_utc")
    if predicted_games == 0:
        if earliest_start is not None:
            return False
    else:
        try:
            _require_utc_timestamp(
                earliest_start,
                field=(
                    "receipt.batch."
                    "earliest_predicted_scheduled_start_utc"
                ),
            )
        except ShadowPredictionError:
            return False

    for field in _RECEIPT_SECTION_KEYS["output_hashes"]:
        try:
            _require_sha256(
                output_hashes.get(field),
                field=f"receipt.output_hashes.{field}",
            )
        except ShadowPredictionError:
            return False
    for field in (
        "sqlite_snapshot_sha256",
        "source_snapshot_sha256",
        "schedule_raw_archive_sha256",
    ):
        try:
            _require_sha256(
                source.get(field),
                field=f"receipt.source.{field}",
            )
        except ShadowPredictionError:
            return False
    if any(value is not True for value in negative_attestations.values()):
        return False
    try:
        _require_utc_timestamp(
            times.get("receipt_finalized_at_utc"),
            field="receipt.times.receipt_finalized_at_utc",
        )
    except ShadowPredictionError:
        return False
    return True


def _valid_completed_marker(
    marker: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    receipt_bytes: bytes,
    receipt_path: str,
    batch_id: str,
) -> bool:
    if not _has_exact_keys(marker, _COMPLETED_MARKER_KEYS):
        return False
    expected_values = {
        "marker_schema_version": 1,
        "batch_id": batch_id,
        "receipt_path": receipt_path,
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }
    if any(
        type(marker.get(key)) is not type(expected)
        or marker.get(key) != expected
        for key, expected in expected_values.items()
    ):
        return False
    receipt_batch = receipt.get("batch")
    receipt_times = receipt.get("times")
    if not isinstance(receipt_batch, dict) or not isinstance(
        receipt_times,
        dict,
    ):
        return False
    if marker.get("batch_id") != receipt_batch.get("batch_id"):
        return False
    try:
        completed_at = _require_utc_timestamp(
            marker.get("completed_at_utc"),
            field="COMPLETED.completed_at_utc",
        )
        receipt_finalized_at = _require_utc_timestamp(
            receipt_times.get("receipt_finalized_at_utc"),
            field="receipt.times.receipt_finalized_at_utc",
        )
    except ShadowPredictionError:
        return False
    if receipt_finalized_at > completed_at:
        return False
    earliest_start = receipt_batch.get(
        "earliest_predicted_scheduled_start_utc"
    )
    if earliest_start is None:
        return True
    try:
        earliest = _require_utc_timestamp(
            earliest_start,
            field=(
                "receipt.batch.earliest_predicted_scheduled_start_utc"
            ),
        )
    except ShadowPredictionError:
        return False
    completed_datetime = datetime.fromisoformat(
        completed_at.replace("Z", "+00:00")
    )
    earliest_datetime = datetime.fromisoformat(
        earliest.replace("Z", "+00:00")
    )
    return completed_datetime <= earliest_datetime - timedelta(minutes=120)


def inspect_shadow_prediction_slot(
    target_official_date: date | str,
    *,
    shadow_protocol_sha256: str,
    execution_manifest_sha256: str,
    model_artifact_sha256: str,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowPredictionSlotInspection:
    """Inspecte un slot v2 local sans reserver, ecrire ni lire ses donnees.

    Les identifiants attendus sont derives ici a partir des valeurs runtime.
    Les CSV et gzip ne sont jamais ouverts : leur presence comme fichiers
    reguliers suffit a cette inspection d'idempotence.
    """
    target_date = _parse_target_official_date(
        target_official_date
    ).isoformat()
    protocol_hash = _require_sha256(
        shadow_protocol_sha256,
        field="shadow_protocol_sha256",
    )
    manifest_hash = _require_sha256(
        execution_manifest_sha256,
        field="execution_manifest_sha256",
    )
    model_hash = _require_sha256(
        model_artifact_sha256,
        field="model_artifact_sha256",
    )
    slot_key = build_slot_key(
        shadow_protocol_sha256=protocol_hash,
        target_official_date=target_date,
    )
    batch_id = build_batch_id(
        slot_key=slot_key,
        execution_manifest_sha256=manifest_hash,
        model_artifact_sha256=model_hash,
    )

    try:
        project_input = Path(project_directory).expanduser()
        project = Path(os.path.abspath(os.fspath(project_input)))
    except (TypeError, OSError, RuntimeError) as error:
        raise ShadowPredictionError(
            "Dossier du projet invalide pour l'inspection du slot."
        ) from error
    project_mode = _lstat_mode(project)
    if not isinstance(project_mode, int) or not stat.S_ISDIR(project_mode):
        raise ShadowPredictionError(
            "Le dossier du projet doit etre un repertoire local non "
            "symbolique."
        )

    slot_path = project.joinpath(
        *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts,
        target_date,
    )

    def inspection(
        state: ShadowPredictionSlotState,
        receipt: dict[str, Any] | None = None,
    ) -> ShadowPredictionSlotInspection:
        return ShadowPredictionSlotInspection(
            state=state,
            slot_path=slot_path,
            slot_key=slot_key,
            batch_id=batch_id,
            receipt=receipt,
        )

    current = project
    for component in SHADOW_RESULT_ROOT_RELATIVE_PATH.parts:
        current = current / component
        mode = _lstat_mode(current)
        if mode is _PATH_MISSING:
            return inspection(ShadowPredictionSlotState.ABSENT)
        if not isinstance(mode, int) or not stat.S_ISDIR(mode):
            return inspection(
                ShadowPredictionSlotState.INCOMPLETE_CONSUMED
            )

    slot_mode = _lstat_mode(slot_path)
    if slot_mode is _PATH_MISSING:
        return inspection(ShadowPredictionSlotState.ABSENT)
    if not isinstance(slot_mode, int) or not stat.S_ISDIR(slot_mode):
        return inspection(ShadowPredictionSlotState.INCOMPLETE_CONSUMED)

    completed_path = slot_path / COMPLETED_FILENAME
    completed_mode = _lstat_mode(completed_path)
    if completed_mode is _PATH_MISSING:
        failed_mode = _lstat_mode(slot_path / "FAILED.json")
        if isinstance(failed_mode, int) and stat.S_ISREG(failed_mode):
            return inspection(ShadowPredictionSlotState.FAILED_CONSUMED)
        return inspection(ShadowPredictionSlotState.INCOMPLETE_CONSUMED)

    mismatch = ShadowPredictionSlotState.COMPLETED_MISMATCH
    if not isinstance(completed_mode, int) or not stat.S_ISREG(
        completed_mode
    ):
        return inspection(mismatch)

    try:
        entries = list(slot_path.iterdir())
    except OSError:
        return inspection(mismatch)
    expected_names = frozenset(SHADOW_SLOT_SUCCESS_FILENAMES)
    if len(entries) != len(expected_names):
        return inspection(mismatch)
    if frozenset(entry.name for entry in entries) != expected_names:
        return inspection(mismatch)
    if any(
        not isinstance(mode := _lstat_mode(entry), int)
        or not stat.S_ISREG(mode)
        for entry in entries
    ):
        return inspection(mismatch)

    reserved_result = _read_canonical_json_object(slot_path / "RESERVED")
    receipt_result = _read_canonical_json_object(
        slot_path / RECEIPT_FILENAME
    )
    completed_result = _read_canonical_json_object(completed_path)
    if (
        reserved_result is None
        or receipt_result is None
        or completed_result is None
    ):
        return inspection(mismatch)
    reserved, _reserved_bytes = reserved_result
    receipt, receipt_bytes = receipt_result
    completed, _completed_bytes = completed_result
    if not _valid_reserved_marker(
        reserved,
        batch_id=batch_id,
        slot_key=slot_key,
        target_official_date=target_date,
        shadow_protocol_sha256=protocol_hash,
        execution_manifest_sha256=manifest_hash,
    ):
        return inspection(mismatch)
    if not _valid_receipt(
        receipt,
        reserved=reserved,
        batch_id=batch_id,
        slot_key=slot_key,
        target_official_date=target_date,
        shadow_protocol_sha256=protocol_hash,
        execution_manifest_sha256=manifest_hash,
        model_artifact_sha256=model_hash,
    ):
        return inspection(mismatch)
    expected_receipt_path = (
        SHADOW_RESULT_ROOT_RELATIVE_PATH
        / target_date
        / RECEIPT_FILENAME
    ).as_posix()
    if not _valid_completed_marker(
        completed,
        receipt=receipt,
        receipt_bytes=receipt_bytes,
        receipt_path=expected_receipt_path,
        batch_id=batch_id,
    ):
        return inspection(mismatch)
    return inspection(ShadowPredictionSlotState.COMPLETED_EXACT, receipt)


def _require_shadow_result_root(
    *,
    project_directory: Path,
) -> Path:
    """Exige la racine versionnee sans creer ni suivre aucun composant."""
    current = project_directory
    for component in SHADOW_RESULT_ROOT_RELATIVE_PATH.parts:
        current = current / component
        mode = _lstat_mode(current)
        if not isinstance(mode, int) or not stat.S_ISDIR(mode):
            raise ShadowPredictionError(
                "La racine versionnee des resultats fantomes doit deja "
                "exister et ne contenir aucun lien symbolique."
            )
    return current


def reserve_shadow_prediction_slot(
    target_official_date: date | str,
    *,
    reserved_at_utc: str,
    runtime_code_commit: str,
    shadow_protocol_sha256: str,
    execution_manifest_sha256: str,
    model_artifact_sha256: str,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowPredictionSlotReservation:
    """Reserve une fois un slot v2 absent et publie son premier fichier.

    Tous les instants et identifiants variables sont fournis par l'appelant :
    cette primitive ne consulte ni horloge, ni Git, ni reseau, ni SQLite, ni
    modele. Une fois le repertoire du slot cree, aucune erreur ne provoque sa
    suppression ou sa reutilisation.
    """
    target = _parse_target_official_date(target_official_date)
    target_date = target.isoformat()
    if target.year != EXPECTED_TARGET_SEASON:
        raise ShadowPredictionError(
            "Saison cible invalide pour une reservation fantome v2 : "
            f"{target.year}, attendu {EXPECTED_TARGET_SEASON}."
        )
    if target < EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON:
        raise ShadowPredictionError(
            "La date cible ne peut pas preceder l'enregistrement du "
            "protocole fantome v2."
        )

    reserved_at = _require_utc_timestamp(
        reserved_at_utc,
        field="reserved_at_utc",
    )
    if target_date < reserved_at[:10]:
        raise ShadowPredictionError(
            "target_official_date ne peut pas preceder la date UTC de "
            "reserved_at_utc."
        )
    runtime_commit = _require_git_commit(
        runtime_code_commit,
        field="runtime_code_commit",
    )
    protocol_hash = _require_sha256(
        shadow_protocol_sha256,
        field="shadow_protocol_sha256",
    )
    if protocol_hash != EXPECTED_SHADOW_PROTOCOL_SHA256:
        raise ShadowPredictionError(
            "La reservation exige l'empreinte du protocole fantome v2 "
            "fige."
        )
    manifest_hash = _require_sha256(
        execution_manifest_sha256,
        field="execution_manifest_sha256",
    )
    model_hash = _require_sha256(
        model_artifact_sha256,
        field="model_artifact_sha256",
    )
    if model_hash != EXPECTED_MODEL_ARTIFACT_SHA256:
        raise ShadowPredictionError(
            "La reservation exige l'empreinte du modele valide et fige."
        )

    # L'inspection est volontairement la premiere operation sur l'arbre de
    # resultats. Elle derive aussi les identifiants attendus du slot et du lot.
    inspected = inspect_shadow_prediction_slot(
        target_date,
        shadow_protocol_sha256=protocol_hash,
        execution_manifest_sha256=manifest_hash,
        model_artifact_sha256=model_hash,
        project_directory=project_directory,
    )
    if inspected.state is not ShadowPredictionSlotState.ABSENT:
        raise ShadowPredictionSlotConsumedError(
            "Le slot fantome est deja consomme et ne peut jamais etre "
            f"reserve de nouveau : {inspected.state.value}."
        )

    reserved_marker: dict[str, Any] = {
        "marker_schema_version": 1,
        "batch_id": inspected.batch_id,
        "slot_key": inspected.slot_key,
        "target_official_date": target_date,
        "reserved_at_utc": reserved_at,
        "shadow_protocol_sha256": protocol_hash,
        "execution_manifest_sha256": manifest_hash,
        "runtime_code_commit": runtime_commit,
    }
    reserved_bytes = _canonical_json_file_bytes(reserved_marker)

    result_root = _require_shadow_result_root(
        project_directory=inspected.slot_path.parents[
            len(SHADOW_RESULT_ROOT_RELATIVE_PATH.parts)
        ],
    )
    if result_root != inspected.slot_path.parent:
        raise ShadowPredictionError(
            "Le repertoire derive du slot fantome est incoherent."
        )

    try:
        inspected.slot_path.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ShadowPredictionSlotConsumedError(
            "Le slot fantome a ete reserve par une autre execution."
        ) from error
    except OSError as error:
        raise ShadowPredictionError(
            "Impossible de creer le slot fantome de facon exclusive."
        ) from error

    # A partir de mkdir, le slot est consomme meme si une synchronisation ou
    # la publication de RESERVED echoue. Il ne doit donc jamais etre nettoye.
    confirmed_result_root = _require_shadow_result_root(
        project_directory=inspected.slot_path.parents[
            len(SHADOW_RESULT_ROOT_RELATIVE_PATH.parts)
        ],
    )
    if confirmed_result_root != result_root:
        raise ShadowPredictionError(
            "La racine du slot a change pendant sa reservation."
        )
    slot_mode = _lstat_mode(inspected.slot_path)
    if not isinstance(slot_mode, int) or not stat.S_ISDIR(slot_mode):
        raise ShadowPredictionError(
            "Le slot reserve n'est pas un repertoire local non symbolique."
        )
    _fsync_parent_directory(inspected.slot_path.parent)
    marker_sha256 = _publish_exclusive_verified(
        inspected.slot_path / "RESERVED",
        reserved_bytes,
    )

    return ShadowPredictionSlotReservation(
        slot_path=inspected.slot_path,
        slot_key=inspected.slot_key,
        batch_id=inspected.batch_id,
        reserved_marker=reserved_marker,
        reserved_marker_sha256=marker_sha256,
    )


def _validate_reservation_proof_without_disk(
    reservation: ShadowPredictionSlotReservation,
) -> tuple[str, str, str, str, str, bytes]:
    """Valide une preuve RESERVED en memoire sans toucher au slot."""
    if type(reservation) is not ShadowPredictionSlotReservation:
        raise ShadowPredictionError(
            "Une preuve de reservation fantome exacte est requise."
        )
    if not isinstance(reservation.slot_path, Path):
        raise ShadowPredictionError(
            "Le chemin du slot reserve doit etre un objet Path."
        )
    slot_key = _require_sha256(reservation.slot_key, field="slot_key")
    batch_id = _require_sha256(reservation.batch_id, field="batch_id")
    reserved_sha256 = _require_sha256(
        reservation.reserved_marker_sha256,
        field="reserved_marker_sha256",
    )
    marker = reservation.reserved_marker
    if not _has_exact_keys(marker, _RESERVED_MARKER_KEYS):
        raise ShadowPredictionError(
            "La preuve de reservation ne contient pas un marqueur "
            "RESERVED exact."
        )
    assert isinstance(marker, dict)
    target_date = _require_date_string(
        marker.get("target_official_date"),
        field="RESERVED.target_official_date",
    )
    target = _parse_target_official_date(target_date)
    if (
        target.year != EXPECTED_TARGET_SEASON
        or target < EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON
    ):
        raise ShadowPredictionError(
            "La preuve RESERVED ne vise pas une date admissible du "
            "protocole fantome v2."
        )
    reserved_at = _require_utc_timestamp(
        marker.get("reserved_at_utc"),
        field="RESERVED.reserved_at_utc",
    )
    protocol_hash = _require_sha256(
        marker.get("shadow_protocol_sha256"),
        field="RESERVED.shadow_protocol_sha256",
    )
    manifest_hash = _require_sha256(
        marker.get("execution_manifest_sha256"),
        field="RESERVED.execution_manifest_sha256",
    )
    runtime_commit = _require_git_commit(
        marker.get("runtime_code_commit"),
        field="RESERVED.runtime_code_commit",
    )
    if protocol_hash != EXPECTED_SHADOW_PROTOCOL_SHA256:
        raise ShadowPredictionError(
            "La preuve RESERVED exige le protocole fantome v2 fige."
        )
    expected_slot_key = build_slot_key(
        shadow_protocol_sha256=protocol_hash,
        target_official_date=target_date,
    )
    expected_batch_id = build_batch_id(
        slot_key=expected_slot_key,
        execution_manifest_sha256=manifest_hash,
        model_artifact_sha256=EXPECTED_MODEL_ARTIFACT_SHA256,
    )
    expected_marker: dict[str, Any] = {
        "marker_schema_version": 1,
        "batch_id": expected_batch_id,
        "slot_key": expected_slot_key,
        "target_official_date": target_date,
        "reserved_at_utc": reserved_at,
        "shadow_protocol_sha256": protocol_hash,
        "execution_manifest_sha256": manifest_hash,
        "runtime_code_commit": runtime_commit,
    }
    if (
        marker != expected_marker
        or slot_key != expected_slot_key
        or batch_id != expected_batch_id
    ):
        raise ShadowPredictionError(
            "La preuve RESERVED ne correspond pas aux identifiants "
            "figes du slot."
        )
    marker_bytes = _canonical_json_file_bytes(expected_marker)
    if hashlib.sha256(marker_bytes).hexdigest() != reserved_sha256:
        raise ShadowPredictionError(
            "L'empreinte de la preuve RESERVED est incoherente."
        )
    return (
        target_date,
        reserved_at,
        protocol_hash,
        manifest_hash,
        runtime_commit,
        marker_bytes,
    )


def publish_activation_reverification_evidence(
    reservation: ShadowPredictionSlotReservation,
    evidence: ShadowActivationReverificationEvidence,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowActivationReverificationPublication:
    """Publie la preuve GitHub exclusivement juste apres RESERVED.

    La preuve HTTP doit deja etre entierement construite en memoire. Cette
    primitive ne fait aucun appel reseau et refuse tout slot contenant autre
    chose que son marqueur RESERVED canonique.
    """
    _, _, canonical_gzip = _validate_activation_reverification_evidence(
        evidence
    )
    (
        target_date,
        reserved_at,
        _protocol_hash,
        _manifest_hash,
        _runtime_commit,
        reserved_bytes,
    ) = _validate_reservation_proof_without_disk(reservation)
    if not isinstance(project_directory, Path):
        raise ShadowPredictionError(
            "project_directory doit etre un chemin Path."
        )
    if evidence.response_received_at_utc > reserved_at:
        raise ShadowPredictionError(
            "La reponse GitHub doit etre recue avant la reservation."
        )
    if evidence.activation_remote_reverified_at_utc > reserved_at:
        raise ShadowPredictionError(
            "La date distante GitHub ne peut pas suivre la reservation."
        )
    if target_date < evidence.activation_remote_reverified_at_utc[:10]:
        raise ShadowPredictionError(
            "La date cible ne peut pas preceder la date UTC de la preuve "
            "GitHub."
        )

    result_root = _require_shadow_result_root(
        project_directory=project_directory,
    )
    expected_slot_path = result_root / target_date
    if reservation.slot_path != expected_slot_path:
        raise ShadowPredictionError(
            "La preuve de reservation ne vise pas le slot attendu."
        )
    slot_mode = _lstat_mode(expected_slot_path)
    if not isinstance(slot_mode, int) or not stat.S_ISDIR(slot_mode):
        raise ShadowPredictionError(
            "Le slot reserve doit rester un repertoire local non "
            "symbolique."
        )

    try:
        with os.scandir(expected_slot_path) as entries:
            entry_names = frozenset(entry.name for entry in entries)
    except OSError as error:
        raise ShadowPredictionError(
            "Impossible de controler l'ordre d'ecriture du slot."
        ) from error
    if entry_names != frozenset({"RESERVED"}):
        raise ShadowPredictionSlotConsumedError(
            "La preuve d'activation doit etre le premier fichier publie "
            "apres RESERVED."
        )

    persisted_reserved = _read_canonical_json_object(
        expected_slot_path / "RESERVED"
    )
    if persisted_reserved is None:
        raise ShadowPredictionError(
            "Le slot ne contient pas la preuve RESERVED canonique attendue."
        )
    persisted_marker, persisted_bytes = persisted_reserved
    if (
        persisted_marker != reservation.reserved_marker
        or persisted_bytes != reserved_bytes
        or hashlib.sha256(persisted_bytes).hexdigest()
        != reservation.reserved_marker_sha256
    ):
        raise ShadowPredictionError(
            "La preuve RESERVED persistee ne correspond pas a la "
            "reservation fournie."
        )

    # Recontrole immediatement l'ordre apres la lecture de RESERVED. En cas
    # de concurrence, le lien exclusif de publication decide ensuite du seul
    # gagnant sans aucun remplacement ni nettoyage du slot.
    try:
        with os.scandir(expected_slot_path) as entries:
            confirmed_names = frozenset(entry.name for entry in entries)
    except OSError as error:
        raise ShadowPredictionError(
            "Impossible de reconfirmer l'ordre d'ecriture du slot."
        ) from error
    if confirmed_names != frozenset({"RESERVED"}):
        raise ShadowPredictionSlotConsumedError(
            "Le slot a ete modifie avant la publication de la preuve."
        )

    evidence_path = expected_slot_path / ACTIVATION_REVERIFICATION_FILENAME
    try:
        persisted_sha256 = _publish_exclusive_verified(
            evidence_path,
            canonical_gzip,
        )
    except ShadowPublicationConflictError as error:
        raise ShadowPredictionSlotConsumedError(
            "La preuve d'activation du slot a deja ete publiee."
        ) from error
    if persisted_sha256 != evidence.canonical_gzip_sha256:
        raise ShadowPredictionError(
            "L'empreinte publiee de la preuve d'activation est incoherente."
        )

    try:
        relative_path = evidence_path.relative_to(
            project_directory
        ).as_posix()
    except ValueError as error:
        raise ShadowPredictionError(
            "Le chemin de preuve publie doit rester dans le projet."
        ) from error
    expected_relative_path = (
        SHADOW_RESULT_ROOT_RELATIVE_PATH
        / target_date
        / ACTIVATION_REVERIFICATION_FILENAME
    ).as_posix()
    if relative_path != expected_relative_path:
        raise ShadowPredictionError(
            "Le chemin relatif de la preuve d'activation est incoherent."
        )

    return ShadowActivationReverificationPublication(
        slot_path=expected_slot_path,
        evidence_path=evidence_path,
        evidence_relative_path=relative_path,
        evidence_sha256=persisted_sha256,
        activation_introduction_commit=(
            evidence.activation_introduction_commit
        ),
        activation_remote_ref=evidence.activation_remote_ref,
        activation_remote_reverified_at_utc=(
            evidence.activation_remote_reverified_at_utc
        ),
        response_received_at_utc=evidence.response_received_at_utc,
        response_body_sha256=evidence.response_body_sha256,
    )


def _validate_source_snapshot_predecessors(
    reservation: ShadowPredictionSlotReservation,
    activation_publication: ShadowActivationReverificationPublication,
    *,
    project_directory: Path,
    expected_additional_filenames: frozenset[str] = frozenset(),
) -> tuple[str, str, str]:
    """Reverifie les deux preuves et l'ensemble exact des fichiers attendus."""
    if type(expected_additional_filenames) is not frozenset or not all(
        type(name) is str and name
        for name in expected_additional_filenames
    ):
        raise ShadowPredictionError(
            "Les fichiers additionnels attendus doivent etre explicites."
        )
    (
        target_date,
        reserved_at,
        _protocol_hash,
        _manifest_hash,
        runtime_commit,
        reserved_bytes,
    ) = _validate_reservation_proof_without_disk(reservation)
    if type(activation_publication) is not (
        ShadowActivationReverificationPublication
    ):
        raise ShadowPredictionError(
            "Une preuve exacte de publication d'activation est requise."
        )
    if not isinstance(project_directory, Path):
        raise ShadowPredictionError(
            "project_directory doit etre un chemin Path."
        )

    result_root = _require_shadow_result_root(
        project_directory=project_directory,
    )
    slot_path = result_root / target_date
    slot_mode = _lstat_mode(slot_path)
    if not isinstance(slot_mode, int) or not stat.S_ISDIR(slot_mode):
        raise ShadowPredictionError(
            "Le slot source doit rester un repertoire local non "
            "symbolique."
        )
    evidence_path = slot_path / ACTIVATION_REVERIFICATION_FILENAME
    expected_relative_path = (
        SHADOW_RESULT_ROOT_RELATIVE_PATH
        / target_date
        / ACTIVATION_REVERIFICATION_FILENAME
    ).as_posix()
    evidence_sha256 = _require_sha256(
        activation_publication.evidence_sha256,
        field="activation_publication.evidence_sha256",
    )
    activation_commit = _require_git_commit(
        activation_publication.activation_introduction_commit,
        field="activation_publication.activation_introduction_commit",
    )
    reverified_at = _require_utc_timestamp(
        activation_publication.activation_remote_reverified_at_utc,
        field=(
            "activation_publication."
            "activation_remote_reverified_at_utc"
        ),
    )
    received_at = _require_utc_timestamp(
        activation_publication.response_received_at_utc,
        field="activation_publication.response_received_at_utc",
    )
    body_sha256 = _require_sha256(
        activation_publication.response_body_sha256,
        field="activation_publication.response_body_sha256",
    )
    if (
        reservation.slot_path != slot_path
        or activation_publication.slot_path != slot_path
        or activation_publication.evidence_path != evidence_path
        or activation_publication.evidence_relative_path
        != expected_relative_path
        or activation_publication.activation_remote_ref
        != GITHUB_REMOTE_REF
    ):
        raise ShadowPredictionError(
            "Les preuves prealables ne visent pas le meme slot canonique."
        )

    try:
        entries = tuple(slot_path.iterdir())
    except OSError as error:
        raise ShadowPredictionError(
            "Impossible de controler les fichiers prealables du slot."
        ) from error
    expected_names = frozenset(
        {"RESERVED", ACTIVATION_REVERIFICATION_FILENAME}
    ) | expected_additional_filenames
    if (
        len(entries) != len(expected_names)
        or frozenset(entry.name for entry in entries) != expected_names
        or any(
            not isinstance(mode := _lstat_mode(entry), int)
            or not stat.S_ISREG(mode)
            for entry in entries
        )
    ):
        raise ShadowPredictionSlotConsumedError(
            "Le slot ne contient pas exactement les predecesseurs attendus."
        )

    persisted_reserved = _read_canonical_json_object(
        slot_path / "RESERVED"
    )
    if persisted_reserved is None:
        raise ShadowPredictionError(
            "Le marqueur RESERVED prealable n'est pas canonique."
        )
    marker, marker_bytes = persisted_reserved
    if (
        marker != reservation.reserved_marker
        or marker_bytes != reserved_bytes
        or hashlib.sha256(marker_bytes).hexdigest()
        != reservation.reserved_marker_sha256
    ):
        raise ShadowPredictionError(
            "Le marqueur RESERVED ne correspond pas a sa preuve."
        )

    try:
        persisted_gzip = evidence_path.read_bytes()
        evidence_json = gzip.decompress(persisted_gzip)
        raw_evidence = json.loads(
            evidence_json.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (
        OSError,
        EOFError,
        UnicodeError,
        ValueError,
        ShadowPredictionError,
        RecursionError,
    ) as error:
        raise ShadowPredictionError(
            "La preuve d'activation persistee est illisible ou ambigue."
        ) from error
    if type(raw_evidence) is not dict:
        raise ShadowPredictionError(
            "La preuve d'activation persistee doit etre un objet JSON."
        )
    reconstructed = ShadowActivationReverificationEvidence(
        activation_introduction_commit=activation_commit,
        activation_remote_ref=GITHUB_REMOTE_REF,
        activation_remote_reverified_at_utc=reverified_at,
        response_received_at_utc=received_at,
        response_body_sha256=body_sha256,
        raw_evidence=raw_evidence,
        canonical_json_bytes=evidence_json,
        canonical_gzip_bytes=persisted_gzip,
        canonical_gzip_sha256=evidence_sha256,
    )
    _validate_activation_reverification_evidence(reconstructed)
    if hashlib.sha256(persisted_gzip).hexdigest() != evidence_sha256:
        raise ShadowPredictionError(
            "L'empreinte de la preuve d'activation a change."
        )
    if received_at > reserved_at or reverified_at > reserved_at:
        raise ShadowPredictionError(
            "La preuve distante doit preceder la reservation du slot."
        )
    return target_date, reserved_at, runtime_commit


@contextmanager
def _hold_source_snapshot_stage_lock(
    reservation: ShadowPredictionSlotReservation,
    *,
    project_directory: Path,
) -> Iterator[None]:
    """Elit un seul producteur source sans ajouter de fichier au slot.

    Le verrou non bloquant porte sur le premier fichier append-only du
    creneau. Il est libere automatiquement par la fermeture du descripteur,
    y compris en cas d'exception ou d'arret du processus.
    """
    (
        target_date,
        _reserved_at,
        _protocol_hash,
        _manifest_hash,
        _runtime_commit,
        _reserved_bytes,
    ) = _validate_reservation_proof_without_disk(reservation)
    if not isinstance(project_directory, Path):
        raise ShadowPredictionError(
            "project_directory doit etre un chemin Path."
        )
    result_root = _require_shadow_result_root(
        project_directory=project_directory,
    )
    expected_slot_path = result_root / target_date
    if reservation.slot_path != expected_slot_path:
        raise ShadowPredictionError(
            "Le verrou source doit viser le slot canonique reserve."
        )
    reserved_path = expected_slot_path / "RESERVED"
    reserved_mode = _lstat_mode(reserved_path)
    if not isinstance(reserved_mode, int) or not stat.S_ISREG(reserved_mode):
        raise ShadowPredictionError(
            "Le verrou source exige un marqueur RESERVED local regulier."
        )

    flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(reserved_path, flags)
    except OSError as error:
        raise ShadowPredictionError(
            "Impossible d'ouvrir le verrou du snapshot source."
        ) from error

    with os.fdopen(descriptor, "r+b", buffering=0) as reserved_file:
        try:
            opened_metadata = os.fstat(reserved_file.fileno())
            current_metadata = reserved_path.lstat()
        except OSError as error:
            raise ShadowPredictionError(
                "Impossible de verifier l'identite du verrou source."
            ) from error
        if (
            not stat.S_ISREG(opened_metadata.st_mode)
            or (opened_metadata.st_dev, opened_metadata.st_ino)
            != (current_metadata.st_dev, current_metadata.st_ino)
            or _lstat_mode(reserved_path) is _PATH_UNREADABLE
        ):
            raise ShadowPredictionError(
                "Le marqueur RESERVED a change pendant son ouverture."
            )

        try:
            if os.name == "nt":
                import msvcrt

                # read_bytes() peut tenter une lecture au-dela de l'EOF :
                # un verrou a EOF bloque alors aussi son proprietaire.
                # Tous les concurrents verrouillent le meme octet lointain,
                # sans ecrire ni agrandir le petit marqueur canonique.
                reserved_file.seek(0x7fffffff, os.SEEK_SET)
                msvcrt.locking(
                    reserved_file.fileno(),
                    msvcrt.LK_NBLCK,
                    1,
                )
            else:
                import fcntl

                fcntl.flock(
                    reserved_file.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise ShadowPredictionSlotConsumedError(
                    "Un autre processus possede deja le slot source."
                ) from error
            raise ShadowPredictionError(
                "Impossible de verrouiller le snapshot source."
            ) from error

        yield


def _open_sqlite_read_only(path: Path) -> sqlite3.Connection:
    """Ouvre un fichier SQLite existant sans autoriser sa mutation."""
    uri = f"{path.resolve(strict=True).as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _require_no_active_ingestion(database_path: Path) -> None:
    """Refuse de commencer si une autre collecte est deja active."""
    mode = _lstat_mode(database_path)
    if not isinstance(mode, int) or not stat.S_ISREG(mode):
        raise ShadowPredictionError(
            "La base SQLite source doit exister sans lien symbolique."
        )
    try:
        with closing(_open_sqlite_read_only(database_path)) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM ingestion_runs WHERE status = ?",
                ("started",),
            ).fetchone()[0]
    except (OSError, sqlite3.Error) as error:
        raise ShadowPredictionError(
            "Impossible de verifier les collectes actives."
        ) from error
    if type(count) is not int or count != 0:
        raise ShadowPredictionError(
            "Une collecte est deja active; le slot ne peut pas continuer."
        )


def _canonical_database_utc(value: object, *, field: str) -> str:
    """Accepte l'UTC SQLite historique puis produit le format v2 en Z."""
    if (
        type(value) is not str
        or not _DATABASE_UTC_TIMESTAMP_PATTERN.fullmatch(value)
    ):
        raise ShadowPredictionError(
            f"{field} doit etre un instant UTC a la seconde."
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ShadowPredictionError(
            f"{field} n'est pas un instant UTC valide."
        ) from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ShadowPredictionError(f"{field} doit etre en UTC.")
    return _format_utc_seconds(parsed, field=field)


def _require_positive_integer(value: object, *, field: str) -> int:
    parsed = _require_integer(value, field=field)
    if parsed <= 0:
        raise ShadowPredictionError(f"{field} doit etre strictement positif.")
    return parsed


def _require_status_text(value: object, *, field: str) -> tuple[str, str]:
    if type(value) is not str or not value.strip():
        raise ShadowPredictionError(f"{field} doit etre un statut non vide.")
    return value, value.strip().upper()


def _validate_target_schedule(
    games: tuple[ScheduledGame, ...],
    *,
    target_date: str,
) -> list[dict[str, Any]]:
    """Valide le calendrier HTTP, retire les scores puis le trie."""
    rows: list[dict[str, Any]] = []
    game_ids: set[int] = set()
    for index, game in enumerate(games):
        if type(game) is not ScheduledGame:
            raise ShadowPredictionError(
                f"Le match cible {index} n'est pas un ScheduledGame exact."
            )
        game_id = _require_positive_integer(
            game.game_id,
            field=f"target_schedule[{index}].game_id",
        )
        if game_id in game_ids:
            raise ShadowPredictionError(
                "Le calendrier cible contient un game_id en doublon."
            )
        game_ids.add(game_id)
        season = _require_integer(
            game.season,
            field=f"target_schedule[{index}].season",
        )
        official_date = _require_date_string(
            game.official_date,
            field=f"target_schedule[{index}].official_date",
        )
        if season != EXPECTED_TARGET_SEASON or official_date != target_date:
            raise ShadowPredictionError(
                "Un match cible ne correspond pas a la saison ou la date."
            )
        if game.game_type != "R":
            raise ShadowPredictionError(
                "Tous les matchs cibles doivent etre de type R."
            )
        away_team_id = _require_positive_integer(
            game.away_team_id,
            field=f"target_schedule[{index}].away_team_id",
        )
        home_team_id = _require_positive_integer(
            game.home_team_id,
            field=f"target_schedule[{index}].home_team_id",
        )
        if away_team_id == home_team_id:
            raise ShadowPredictionError(
                "Une equipe cible ne peut pas jouer contre elle-meme."
            )
        scores = (game.away_score, game.home_score)
        if scores != (None, None) and not (
            type(scores[0]) is int
            and type(scores[1]) is int
            and scores == (0, 0)
        ):
            raise ShadowPredictionError(
                "Les scores pre-match doivent etre tous deux null ou zero."
            )

        status_code, normalized_code = _require_status_text(
            game.status_code,
            field=f"target_schedule[{index}].status_code",
        )
        abstract_state, normalized_abstract = _require_status_text(
            game.abstract_state,
            field=f"target_schedule[{index}].abstract_state",
        )
        detailed_state, normalized_detail = _require_status_text(
            game.status_detail,
            field=f"target_schedule[{index}].detailed_state",
        )
        postponed = (
            normalized_code in {"D", "DI", "DR"}
            and normalized_detail == "POSTPONED"
        )
        cancelled = (
            normalized_code in {"C", "CI", "CR"}
            and normalized_detail == "CANCELLED"
        )
        if not postponed and not cancelled:
            if normalized_abstract in {"LIVE", "FINAL"}:
                raise ShadowPredictionError(
                    "Un match cible est deja live ou final."
                )
            scheduled = (
                normalized_code == "S"
                and normalized_abstract == "PREVIEW"
                and normalized_detail == "SCHEDULED"
            )
            pre_game = (
                normalized_code == "P"
                and normalized_abstract == "PREVIEW"
                and normalized_detail == "PRE-GAME"
            )
            if not scheduled and not pre_game:
                raise ShadowPredictionError(
                    "Un match cible possede un etat MLB inconnu."
                )

        game_datetime = game.game_datetime_utc
        if game_datetime is not None:
            game_datetime = _require_utc_timestamp(
                game_datetime,
                field=f"target_schedule[{index}].game_datetime_utc",
            )
        if game.doubleheader is not None and (
            type(game.doubleheader) is not str or not game.doubleheader
        ):
            raise ShadowPredictionError(
                "doubleheader doit etre une chaine non vide ou null."
            )
        game_number = game.game_number
        if game_number is not None:
            game_number = _require_positive_integer(
                game_number,
                field=f"target_schedule[{index}].game_number",
            )
        rows.append(
            {
                "game_id": game_id,
                "season": season,
                "official_date": official_date,
                "game_datetime_utc": game_datetime,
                "game_type": "R",
                "status_code": status_code,
                "abstract_state": abstract_state,
                "detailed_state": detailed_state,
                "away_team_id": away_team_id,
                "home_team_id": home_team_id,
                "doubleheader": game.doubleheader,
                "game_number": game_number,
            }
        )
    rows.sort(
        key=lambda row: (
            1 if row["game_datetime_utc"] is None else 0,
            row["game_datetime_utc"] or "",
            row["game_id"],
        )
    )
    return rows


def _verify_schedule_raw_archive(
    *,
    relative_path: object,
    expected_sha256: str,
    data_directory: Path,
) -> str:
    """Verifie l'archive MLB sans accepter de lien symbolique.

    ``verify_raw_archive`` reste la source de verite pour le contenu gzip et
    son empreinte. Ce controle local ferme en plus chaque composant du chemin
    avant et apres cette lecture afin qu'une archive shadow ne puisse jamais
    provenir d'un lien ou d'un reparse point.
    """
    if type(relative_path) is not str or relative_path != relative_path.strip():
        raise ShadowPredictionError(
            "Le chemin de l'archive MLB doit etre un texte canonique."
        )
    if "\\" in relative_path:
        raise ShadowPredictionError(
            "Le chemin de l'archive MLB doit utiliser des barres obliques."
        )
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or relative.parts[:2] != ("data", "raw")
        or len(relative.parts) < 3
        or any(part in {"", ".", ".."} for part in relative.parts)
        or not relative.name.endswith(".json.gz")
        or relative.as_posix() != relative_path
    ):
        raise ShadowPredictionError(
            "Le chemin de l'archive MLB doit rester canonique dans data/raw."
        )

    project_directory = data_directory.parent
    archive_path = project_directory.joinpath(*relative.parts)

    def require_real_path_components() -> None:
        current = project_directory
        for index, part in enumerate(relative.parts):
            current = current / part
            mode = _lstat_mode(current)
            is_final = index == len(relative.parts) - 1
            valid = (
                isinstance(mode, int)
                and (
                    stat.S_ISREG(mode)
                    if is_final
                    else stat.S_ISDIR(mode)
                )
            )
            if not valid:
                raise ShadowPredictionError(
                    "L'archive MLB et tous ses repertoires doivent etre "
                    "locaux, reguliers et non symboliques."
                )

    require_real_path_components()
    try:
        verified = verify_raw_archive(
            relative_path=relative_path,
            expected_sha256=expected_sha256,
            data_directory=data_directory,
        )
    except (RawArchiveError, OSError, ValueError) as error:
        raise ShadowPredictionError(
            "L'archive MLB fraiche est absente, illisible ou incoherente."
        ) from error
    require_real_path_components()
    if verified.absolute_path != archive_path.resolve(strict=True):
        raise ShadowPredictionError(
            "L'archive MLB verifiee ne correspond pas au chemin canonique."
        )
    return relative_path


def _validate_schedule_provenance(
    result: ScheduleIngestionResult,
    ingestion_row: Mapping[str, Any],
    *,
    target_date: str,
    runtime_commit: str,
    data_directory: Path,
) -> tuple[dict[str, Any], str]:
    """Lie le resultat HTTP, son archive et son journal SQLite."""
    if type(result) is not ScheduleIngestionResult:
        raise ShadowPredictionError(
            "La collecte observee n'a pas produit son resultat exact."
        )
    target = date.fromisoformat(target_date)
    parameters = build_schedule_request_parameters(
        start_date=target,
        end_date=target,
        game_types=("R",),
    )
    parameters_json = _canonical_json_bytes(parameters).decode("utf-8")
    run_id = _require_positive_integer(result.run_id, field="run_id")
    response_hash = _require_sha256(
        result.response_sha256,
        field="response_sha256",
    )
    body_hash = _require_sha256(
        result.response_body_sha256,
        field="response_body_sha256",
    )
    if response_hash != body_hash:
        raise ShadowPredictionError(
            "Le corps HTTP MLB et l'archive brute ont des empreintes "
            "differentes."
        )
    if (
        result.start_date != target
        or result.end_date != target
        or result.code_version != runtime_commit
        or type(result.games_received) is not int
        or type(result.games_saved) is not int
        or result.games_received != len(result.games)
        or result.games_saved != len(result.games)
    ):
        raise ShadowPredictionError(
            "Le resultat de collecte MLB ne correspond pas au lot cible."
        )
    if type(result.response_status_code) is not int or (
        result.response_status_code != 200
    ):
        raise ShadowPredictionError("La reponse MLB doit etre HTTP 200.")
    if type(result.response_redirect_count) is not int or (
        result.response_redirect_count != 0
    ):
        raise ShadowPredictionError(
            "La reponse MLB ne doit contenir aucune redirection."
        )
    if type(result.response_effective_url) is not str:
        raise ShadowPredictionError("L'URL effective MLB est absente.")
    parsed_url = urlsplit(result.response_effective_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.netloc != "statsapi.mlb.com"
        or parsed_url.path != "/api/v1/schedule"
        or parsed_url.fragment
    ):
        raise ShadowPredictionError("L'URL effective MLB est invalide.")
    try:
        query_pairs = parse_qsl(
            parsed_url.query,
            keep_blank_values=True,
            strict_parsing=True,
        )
    except ValueError as error:
        raise ShadowPredictionError("La query MLB est invalide.") from error
    if (
        len(query_pairs) != len({key for key, _ in query_pairs})
        or dict(query_pairs)
        != {str(key): str(value) for key, value in parameters.items()}
    ):
        raise ShadowPredictionError(
            "La query effective MLB ne correspond pas aux parametres figes."
        )
    http_date_raw = _require_nonempty_text(
        result.mlb_http_date_header_raw,
        field="mlb_http_date_header_raw",
    )
    http_date_utc = _parse_imf_fixdate_gmt(
        http_date_raw,
        field="mlb_http_date_header_raw",
    )
    if result.mlb_http_date_utc != http_date_utc:
        raise ShadowPredictionError(
            "La date HTTP MLB normalisee est incoherente."
        )
    received_at = _require_utc_timestamp(
        result.mlb_http_response_received_at_utc,
        field="mlb_http_response_received_at_utc",
    )
    if abs(
        (
            datetime.fromisoformat(received_at.replace("Z", "+00:00"))
            - datetime.fromisoformat(http_date_utc.replace("Z", "+00:00"))
        ).total_seconds()
    ) > 300:
        raise ShadowPredictionError(
            "La preuve MLB et l'horloge locale different de plus de 300 "
            "secondes."
        )

    expected_row: dict[str, object] = {
        "run_id": run_id,
        "source": INGESTION_SOURCE,
        "requested_start_date": target_date,
        "requested_end_date": target_date,
        "game_types": "R",
        "request_parameters_json": parameters_json,
        "status": "success",
        "records_received": len(result.games),
        "records_saved": len(result.games),
        "raw_response_path": result.archive_relative_path,
        "response_sha256": response_hash,
        "code_version": runtime_commit,
        "error_message": None,
    }
    for field, expected in expected_row.items():
        actual = ingestion_row.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise ShadowPredictionError(
                f"Le journal de collecte est incoherent pour {field}."
            )
    completed_at = _canonical_database_utc(
        ingestion_row.get("completed_at_utc"),
        field="ingestion.completed_at_utc",
    )
    archive_relative_path = _verify_schedule_raw_archive(
        relative_path=result.archive_relative_path,
        expected_sha256=response_hash,
        data_directory=data_directory,
    )
    return (
        {
            "run_id": run_id,
            "source": INGESTION_SOURCE,
            "requested_start_date": target_date,
            "requested_end_date": target_date,
            "game_types": "R",
            "request_parameters_json": parameters_json,
            "completed_at_utc": completed_at,
            "raw_archive_path": archive_relative_path,
            "raw_archive_sha256": response_hash,
            "response_effective_url": result.response_effective_url,
            "response_status_code": 200,
            "response_redirect_count": 0,
            "mlb_http_date_header_raw": http_date_raw,
            "mlb_http_date_utc": http_date_utc,
            "mlb_http_response_received_at_utc": received_at,
            "response_body_sha256": body_hash,
        },
        completed_at,
    )


def _read_snapshot_rows(
    connection: sqlite3.Connection,
    *,
    target_date: str,
    ingestion_run_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extrait uniquement les lignes permises du backup SQLite fige."""
    active_count = connection.execute(
        "SELECT COUNT(*) FROM ingestion_runs WHERE status = ?",
        ("started",),
    ).fetchone()[0]
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if type(active_count) is not int or active_count != 0:
        raise ShadowPredictionError(
            "Le snapshot SQLite contient une collecte encore active."
        )
    if violations:
        raise ShadowPredictionError(
            "Le snapshot SQLite contient une violation de cle etrangere."
        )
    ingestion = connection.execute(
        "SELECT * FROM ingestion_runs WHERE run_id = ?",
        (ingestion_run_id,),
    ).fetchone()
    if ingestion is None:
        raise ShadowPredictionError(
            "Le journal de la collecte fraiche manque dans le snapshot."
        )

    teams: list[dict[str, Any]] = []
    seen_team_ids: set[int] = set()
    for index, row in enumerate(
        connection.execute(
            "SELECT team_id, name, abbreviation FROM teams ORDER BY team_id"
        )
    ):
        team_id = _require_positive_integer(
            row["team_id"],
            field=f"teams[{index}].team_id",
        )
        if team_id in seen_team_ids:
            raise ShadowPredictionError("Un team_id SQLite est duplique.")
        seen_team_ids.add(team_id)
        name = _require_nonempty_text(
            row["name"],
            field=f"teams[{index}].name",
        )
        abbreviation = row["abbreviation"]
        if abbreviation is not None:
            abbreviation = _require_nonempty_text(
                abbreviation,
                field=f"teams[{index}].abbreviation",
            )
        teams.append(
            {
                "team_id": team_id,
                "name": name,
                "abbreviation": abbreviation,
            }
        )

    source_games: list[dict[str, Any]] = []
    seen_game_ids: set[int] = set()
    query = """
        SELECT
            game_id, season, official_date, game_type, status_code,
            status_detail, away_team_id, home_team_id, away_score, home_score
        FROM games
        WHERE season = ?
          AND official_date < ?
          AND game_type = 'R'
          AND (
              status_code = 'F'
              OR UPPER(TRIM(status_detail)) IN (
                  'FINAL', 'GAME OVER', 'COMPLETED EARLY'
              )
          )
        ORDER BY official_date, game_id
    """
    for index, row in enumerate(
        connection.execute(query, (EXPECTED_TARGET_SEASON, target_date))
    ):
        game_id = _require_positive_integer(
            row["game_id"],
            field=f"source_final_games[{index}].game_id",
        )
        if game_id in seen_game_ids:
            raise ShadowPredictionError("Un game_id source est duplique.")
        seen_game_ids.add(game_id)
        season = _require_integer(
            row["season"],
            field=f"source_final_games[{index}].season",
        )
        official_date = _require_date_string(
            row["official_date"],
            field=f"source_final_games[{index}].official_date",
        )
        status_code, normalized_code = _require_status_text(
            row["status_code"],
            field=f"source_final_games[{index}].status_code",
        )
        status_detail, normalized_detail = _require_status_text(
            row["status_detail"],
            field=f"source_final_games[{index}].status_detail",
        )
        away_team_id = _require_positive_integer(
            row["away_team_id"],
            field=f"source_final_games[{index}].away_team_id",
        )
        home_team_id = _require_positive_integer(
            row["home_team_id"],
            field=f"source_final_games[{index}].home_team_id",
        )
        away_score = _require_integer(
            row["away_score"],
            field=f"source_final_games[{index}].away_score",
        )
        home_score = _require_integer(
            row["home_score"],
            field=f"source_final_games[{index}].home_score",
        )
        if (
            season != EXPECTED_TARGET_SEASON
            or official_date >= target_date
            or row["game_type"] != "R"
            or away_team_id == home_team_id
            or away_score < 0
            or home_score < 0
            or away_score == home_score
            or not (
                normalized_code == "F"
                or normalized_detail
                in {"FINAL", "GAME OVER", "COMPLETED EARLY"}
            )
        ):
            raise ShadowPredictionError(
                "Un match final source viole le contrat temporel v2."
            )
        source_games.append(
            {
                "game_id": game_id,
                "season": season,
                "official_date": official_date,
                "game_type": "R",
                "status_code": status_code,
                "status_detail": status_detail,
                "away_team_id": away_team_id,
                "home_team_id": home_team_id,
                "away_score": away_score,
                "home_score": home_score,
            }
        )
    return dict(ingestion), teams, source_games


def _build_and_publish_source_snapshot(
    *,
    reservation: ShadowPredictionSlotReservation,
    activation_publication: ShadowActivationReverificationPublication,
    ingestion_result: ScheduleIngestionResult,
    ingestion_attempts: int,
    ingestion_row: Mapping[str, Any],
    teams: list[dict[str, Any]],
    source_games: list[dict[str, Any]],
    target_schedule: list[dict[str, Any]],
    sqlite_sha256: str,
    sqlite_size: int,
    database_relative_path: str,
    target_date: str,
    reserved_at: str,
    runtime_commit: str,
    data_directory: Path,
    project_directory: Path,
) -> ShadowSourceSnapshotPublication:
    """Valide les sources figees et publie pendant que le backup existe."""
    schedule_ingestion, ingestion_completed_at = (
        _validate_schedule_provenance(
            ingestion_result,
            ingestion_row,
            target_date=target_date,
            runtime_commit=runtime_commit,
            data_directory=data_directory,
        )
    )
    team_ids = {row["team_id"] for row in teams}
    referenced_team_ids = {
        team_id
        for game in (*target_schedule, *source_games)
        for team_id in (game["away_team_id"], game["home_team_id"])
    }
    if not referenced_team_ids.issubset(team_ids):
        raise ShadowPredictionError(
            "Le snapshot ne contient pas toutes les identites d'equipe."
        )

    information_cutoff = _format_utc_seconds(
        _utc_now(),
        field="information_cutoff_utc",
    )
    schedule_observed_at = schedule_ingestion[
        "mlb_http_response_received_at_utc"
    ]
    assert isinstance(schedule_observed_at, str)
    if not (
        reserved_at <= schedule_observed_at <= information_cutoff
        and schedule_observed_at
        <= ingestion_completed_at
        <= information_cutoff
    ):
        raise ShadowPredictionError(
            "L'ordre temporel de la collecte source est invalide."
        )
    cutoff_datetime = datetime.fromisoformat(
        information_cutoff.replace("Z", "+00:00")
    )
    schedule_age = (
        cutoff_datetime
        - datetime.fromisoformat(
            schedule_observed_at.replace("Z", "+00:00")
        )
    ).total_seconds()
    if schedule_age < 0 or schedule_age > 900:
        raise ShadowPredictionError(
            "Le calendrier MLB doit avoir entre 0 et 900 secondes."
        )
    if any(
        row["game_datetime_utc"] is not None
        and datetime.fromisoformat(
            row["game_datetime_utc"].replace("Z", "+00:00")
        )
        <= cutoff_datetime
        for row in target_schedule
    ):
        raise ShadowPredictionError(
            "Tout horaire cible connu doit suivre information_cutoff_utc."
        )

    source_snapshot: dict[str, Any] = {
        "schema_version": 1,
        "batch_id": reservation.batch_id,
        "target_official_date": target_date,
        "created_at_utc": information_cutoff,
        "information_cutoff_utc": information_cutoff,
        "schedule_ingestion": schedule_ingestion,
        "sqlite_snapshot": {
            "source_database_path": database_relative_path,
            "sha256": sqlite_sha256,
            "size_bytes": sqlite_size,
            "foreign_key_violation_count": 0,
            "active_ingestion_count": 0,
        },
        "teams": teams,
        "target_schedule": target_schedule,
        "source_final_games": source_games,
    }
    canonical_json = _canonical_json_file_bytes(source_snapshot)
    canonical_gzip = _canonical_gzip_bytes(canonical_json)
    expected_sha256 = hashlib.sha256(canonical_gzip).hexdigest()

    _validate_source_snapshot_predecessors(
        reservation,
        activation_publication,
        project_directory=project_directory,
    )
    snapshot_path = reservation.slot_path / SOURCE_SNAPSHOT_FILENAME
    try:
        persisted_sha256 = _publish_exclusive_verified(
            snapshot_path,
            canonical_gzip,
        )
    except ShadowPublicationConflictError as error:
        raise ShadowPredictionSlotConsumedError(
            "Le snapshot source du slot a deja ete publie."
        ) from error
    if persisted_sha256 != expected_sha256:
        raise ShadowPredictionError(
            "L'empreinte du snapshot source publie est incoherente."
        )
    relative_path = snapshot_path.relative_to(project_directory).as_posix()
    expected_relative_path = (
        SHADOW_RESULT_ROOT_RELATIVE_PATH
        / target_date
        / SOURCE_SNAPSHOT_FILENAME
    ).as_posix()
    if relative_path != expected_relative_path:
        raise ShadowPredictionError(
            "Le chemin du snapshot source publie est incoherent."
        )
    return ShadowSourceSnapshotPublication(
        slot_path=reservation.slot_path,
        snapshot_path=snapshot_path,
        snapshot_relative_path=relative_path,
        snapshot_sha256=persisted_sha256,
        snapshot_size_bytes=len(canonical_gzip),
        sqlite_snapshot_sha256=sqlite_sha256,
        sqlite_snapshot_size_bytes=sqlite_size,
        schedule_ingestion_run_id=ingestion_result.run_id,
        schedule_attempts=ingestion_attempts,
        information_cutoff_utc=information_cutoff,
    )


def _capture_and_publish_source_snapshot_under_lock(
    reservation: ShadowPredictionSlotReservation,
    activation_publication: ShadowActivationReverificationPublication,
    *,
    database_path: Path = DATABASE_PATH,
    data_directory: Path = DATA_DIR,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowSourceSnapshotPublication:
    """Collecte et publie le snapshot pendant que le verrou est tenu.

    Les deux preuves deja publiees sont entierement controlees avant la
    premiere lecture MLB ou SQLite. Le modele et le dataset historique ne
    sont jamais charges par cette primitive.
    """
    target_date, reserved_at, runtime_commit = (
        _validate_source_snapshot_predecessors(
            reservation,
            activation_publication,
            project_directory=project_directory,
        )
    )
    if not isinstance(database_path, Path) or not isinstance(
        data_directory, Path
    ):
        raise ShadowPredictionError(
            "Les chemins de donnees doivent etre des objets Path."
        )
    try:
        project_root = project_directory.resolve(strict=True)
        resolved_database_path = database_path.resolve(strict=True)
        expected_data_directory = project_root / "data"
        expected_database_path = expected_data_directory / "fredo_mlb.db"
        database_relative_path = resolved_database_path.relative_to(
            project_root
        ).as_posix()
    except (OSError, ValueError) as error:
        raise ShadowPredictionError(
            "La base SQLite doit rester dans le projet."
        ) from error
    if data_directory.resolve() != expected_data_directory:
        raise ShadowPredictionError(
            "Le repertoire data doit etre le repertoire canonique du projet."
        )
    if resolved_database_path != expected_database_path:
        raise ShadowPredictionError(
            "La base officielle doit etre exactement data/fredo_mlb.db."
        )

    _require_no_active_ingestion(database_path)
    target = date.fromisoformat(target_date)

    outcome = run_with_retries(
        lambda: run_observed_schedule_ingestion(
            start_date=target,
            end_date=target,
            game_types=("R",),
            database_path=database_path,
            data_directory=data_directory,
            code_version=runtime_commit,
        ),
        retry_exceptions=(MLBAPIRetryableError,),
        policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=1,
            backoff_multiplier=2,
        ),
        sleep_function=time.sleep,
    )
    ingestion_result = outcome.value
    if type(ingestion_result) is not ScheduleIngestionResult or type(
        ingestion_result.games
    ) is not tuple:
        raise ShadowPredictionError(
            "La collecte observee doit produire un resultat et un calendrier "
            "de types exacts."
        )
    target_schedule = _validate_target_schedule(
        ingestion_result.games,
        target_date=target_date,
    )

    try:
        with tempfile.TemporaryDirectory(
            prefix="fredo-shadow-sqlite-"
        ) as temporary_directory:
            snapshot_path = Path(temporary_directory) / "snapshot.db"
            with closing(
                _open_sqlite_read_only(database_path)
            ) as source_connection:
                with closing(
                    sqlite3.connect(snapshot_path)
                ) as target_connection:
                    source_connection.backup(target_connection)
            snapshot_bytes = snapshot_path.read_bytes()
            if not snapshot_bytes:
                raise ShadowPredictionError(
                    "Le backup SQLite ne peut pas etre vide."
                )
            sqlite_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
            sqlite_size = len(snapshot_bytes)
            with closing(
                _open_sqlite_read_only(snapshot_path)
            ) as snapshot_connection:
                ingestion_row, teams, source_games = _read_snapshot_rows(
                    snapshot_connection,
                    target_date=target_date,
                    ingestion_run_id=ingestion_result.run_id,
                )
            verified_snapshot_bytes = snapshot_path.read_bytes()
            if (
                verified_snapshot_bytes != snapshot_bytes
                or len(verified_snapshot_bytes) != sqlite_size
                or hashlib.sha256(verified_snapshot_bytes).hexdigest()
                != sqlite_sha256
            ):
                raise ShadowPredictionError(
                    "Le backup SQLite a change pendant son extraction."
                )
            return _build_and_publish_source_snapshot(
                reservation=reservation,
                activation_publication=activation_publication,
                ingestion_result=ingestion_result,
                ingestion_attempts=outcome.attempts,
                ingestion_row=ingestion_row,
                teams=teams,
                source_games=source_games,
                target_schedule=target_schedule,
                sqlite_sha256=sqlite_sha256,
                sqlite_size=sqlite_size,
                database_relative_path=database_relative_path,
                target_date=target_date,
                reserved_at=reserved_at,
                runtime_commit=runtime_commit,
                data_directory=data_directory,
                project_directory=project_directory,
            )
    except (OSError, sqlite3.Error) as error:
        raise ShadowPredictionError(
            "Impossible de produire le backup SQLite coherent."
        ) from error


def capture_and_publish_source_snapshot(
    reservation: ShadowPredictionSlotReservation,
    activation_publication: ShadowActivationReverificationPublication,
    *,
    database_path: Path = DATABASE_PATH,
    data_directory: Path = DATA_DIR,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowSourceSnapshotPublication:
    """Elit, collecte et publie l'unique snapshot source shadow v2."""
    _validate_source_snapshot_predecessors(
        reservation,
        activation_publication,
        project_directory=project_directory,
    )
    with _hold_source_snapshot_stage_lock(
        reservation,
        project_directory=project_directory,
    ):
        _validate_source_snapshot_predecessors(
            reservation,
            activation_publication,
            project_directory=project_directory,
        )
        return _capture_and_publish_source_snapshot_under_lock(
            reservation,
            activation_publication,
            database_path=database_path,
            data_directory=data_directory,
            project_directory=project_directory,
        )


def _require_exact_source_object(
    value: object,
    keys: frozenset[str],
    *,
    field: str,
) -> dict[str, Any]:
    """Refuse toute extension ou omission dans le snapshot deja fige."""
    if type(value) is not dict or frozenset(value) != keys:
        raise ShadowPredictionError(f"Schema exact requis pour {field}.")
    return value


def _validate_candidate_source_proof(
    reservation: ShadowPredictionSlotReservation,
    source_publication: ShadowSourceSnapshotPublication,
    *,
    project_directory: Path,
) -> tuple[str, str]:
    """Controle les identites de source sans lire ni remplacer de fichier."""
    target, reserved_at, _, _, _, _ = (
        _validate_reservation_proof_without_disk(reservation)
    )
    if type(source_publication) is not ShadowSourceSnapshotPublication:
        raise ShadowPredictionError("Une preuve exacte de snapshot est requise.")
    if not isinstance(project_directory, Path):
        raise ShadowPredictionError("project_directory doit etre un Path.")
    expected_slot = project_directory.joinpath(
        *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts, target
    )
    expected_path = expected_slot / SOURCE_SNAPSHOT_FILENAME
    expected_relative = (
        SHADOW_RESULT_ROOT_RELATIVE_PATH / target / SOURCE_SNAPSHOT_FILENAME
    ).as_posix()
    if (
        reservation.slot_path != expected_slot
        or source_publication.slot_path != expected_slot
        or source_publication.snapshot_path != expected_path
        or source_publication.snapshot_relative_path != expected_relative
    ):
        raise ShadowPredictionError("La preuve source ne vise pas le slot exact.")
    for field in ("snapshot_sha256", "sqlite_snapshot_sha256"):
        _require_sha256(getattr(source_publication, field), field=field)
    for field in (
        "snapshot_size_bytes", "sqlite_snapshot_size_bytes",
        "schedule_ingestion_run_id", "schedule_attempts",
    ):
        _require_positive_integer(getattr(source_publication, field), field=field)
    if source_publication.schedule_attempts > 3:
        raise ShadowPredictionError("La preuve source excede trois tentatives.")
    cutoff = _require_utc_timestamp(
        source_publication.information_cutoff_utc,
        field="source_publication.information_cutoff_utc",
    )
    if cutoff < reserved_at:
        raise ShadowPredictionError("Le cutoff source precede la reservation.")
    return target, cutoff


def _read_candidate_source_snapshot(
    reservation: ShadowPredictionSlotReservation,
    source_publication: ShadowSourceSnapshotPublication,
    *,
    project_directory: Path,
) -> tuple[bytes, dict[str, Any]]:
    """Relit uniquement les octets immuables du troisieme fichier officiel."""
    target, cutoff = _validate_candidate_source_proof(
        reservation, source_publication, project_directory=project_directory
    )
    snapshot_path = source_publication.snapshot_path
    mode = _lstat_mode(snapshot_path)
    if not isinstance(mode, int) or not stat.S_ISREG(mode):
        raise ShadowPredictionError("Le snapshot doit etre un fichier non symbolique.")
    try:
        compressed = snapshot_path.read_bytes()
        if (
            len(compressed) != source_publication.snapshot_size_bytes
            or hashlib.sha256(compressed).hexdigest()
            != source_publication.snapshot_sha256
        ):
            raise ShadowPredictionError("L'empreinte ou la taille du snapshot a change.")
        json_bytes = gzip.decompress(compressed)
        payload = json.loads(
            json_bytes.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
        if (
            _canonical_json_file_bytes(payload) != json_bytes
            or _canonical_gzip_bytes(json_bytes) != compressed
        ):
            raise ShadowPredictionError("Le snapshot source n'est pas canonique.")
    except (
        OSError, EOFError, UnicodeError, ValueError, RecursionError, zlib.error
    ) as error:
        raise ShadowPredictionError("Le snapshot source est illisible ou ambigu.") from error
    payload = _require_exact_source_object(
        payload, _SOURCE_SNAPSHOT_KEYS, field="source_snapshot"
    )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["batch_id"] != reservation.batch_id
        or payload["target_official_date"] != target
        or payload["information_cutoff_utc"] != cutoff
        or payload["created_at_utc"] != cutoff
    ):
        raise ShadowPredictionError("L'identite du snapshot source est incoherente.")

    sqlite = _require_exact_source_object(
        payload["sqlite_snapshot"], _SOURCE_SQLITE_SNAPSHOT_KEYS,
        field="sqlite_snapshot",
    )
    expected_sqlite = {
        "source_database_path": "data/fredo_mlb.db",
        "sha256": source_publication.sqlite_snapshot_sha256,
        "size_bytes": source_publication.sqlite_snapshot_size_bytes,
        "foreign_key_violation_count": 0,
        "active_ingestion_count": 0,
    }
    if any(type(sqlite[k]) is not type(v) or sqlite[k] != v
           for k, v in expected_sqlite.items()):
        raise ShadowPredictionError("La preuve SQLite du snapshot est incoherente.")

    ingestion = _require_exact_source_object(
        payload["schedule_ingestion"], _SOURCE_SCHEDULE_INGESTION_KEYS,
        field="schedule_ingestion",
    )
    parameters = build_schedule_request_parameters(
        start_date=date.fromisoformat(target),
        end_date=date.fromisoformat(target), game_types=("R",),
    )
    expected_ingestion = {
        "run_id": source_publication.schedule_ingestion_run_id,
        "source": INGESTION_SOURCE,
        "requested_start_date": target,
        "requested_end_date": target,
        "game_types": "R",
        "request_parameters_json": _canonical_json_bytes(parameters).decode("utf-8"),
        "response_status_code": 200,
        "response_redirect_count": 0,
    }
    if any(type(ingestion[k]) is not type(v) or ingestion[k] != v
           for k, v in expected_ingestion.items()):
        raise ShadowPredictionError("La collecte source ne correspond pas au lot.")
    raw_hash = _require_sha256(ingestion["raw_archive_sha256"], field="raw_archive_sha256")
    if ingestion["response_body_sha256"] != raw_hash:
        raise ShadowPredictionError("Le corps HTTP et l'archive source divergent.")
    raw_path_text = _require_nonempty_text(ingestion["raw_archive_path"], field="raw_archive_path")
    raw_path = PurePosixPath(raw_path_text)
    if (
        "\\" in raw_path_text or raw_path.is_absolute()
        or raw_path.parts[:2] != ("data", "raw")
        or len(raw_path.parts) < 3
        or any(part in {"", ".", ".."} for part in raw_path.parts)
        or raw_path.as_posix() != raw_path_text
        or not raw_path.name.endswith(".json.gz")
    ):
        raise ShadowPredictionError("Chemin d'archive source non canonique.")
    effective_url = _require_nonempty_text(ingestion["response_effective_url"], field="response_effective_url")
    try:
        url = urlsplit(effective_url)
        pairs = parse_qsl(url.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as error:
        raise ShadowPredictionError("URL source MLB invalide.") from error
    if (
        url.scheme != "https" or url.netloc != "statsapi.mlb.com"
        or url.path != "/api/v1/schedule" or url.fragment
        or len(pairs) != len({key for key, _ in pairs})
        or dict(pairs) != {str(k): str(v) for k, v in parameters.items()}
    ):
        raise ShadowPredictionError("URL source MLB hors contrat.")
    http_date = _parse_imf_fixdate_gmt(
        ingestion["mlb_http_date_header_raw"], field="mlb_http_date_header_raw"
    )
    received = _require_utc_timestamp(
        ingestion["mlb_http_response_received_at_utc"], field="mlb_http_response_received_at_utc"
    )
    completed = _require_utc_timestamp(ingestion["completed_at_utc"], field="completed_at_utc")
    reserved_at = reservation.reserved_marker["reserved_at_utc"]
    if (
        ingestion["mlb_http_date_utc"] != http_date
        or not (reserved_at <= received <= completed <= cutoff)
    ):
        raise ShadowPredictionError("Ordre temporel du snapshot invalide.")
    to_datetime = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
    if (
        abs((to_datetime(received) - to_datetime(http_date)).total_seconds()) > 300
        or not 0 <= (to_datetime(cutoff) - to_datetime(received)).total_seconds() <= 900
    ):
        raise ShadowPredictionError("Fraicheur ou horloge du snapshot invalide.")
    for name in ("teams", "target_schedule", "source_final_games"):
        if type(payload[name]) is not list:
            raise ShadowPredictionError(f"{name} doit etre une liste JSON exacte.")
    return compressed, payload


def _build_candidate_feature_rows(
    snapshot: dict[str, Any],
    *,
    shadow_protocol_sha256: str,
) -> tuple[list[list[object]], list[list[object]], dict[str, int]]:
    """Calcule les deux tables en memoire depuis le seul sous-ensemble fige.

    Tous les cumuls sont termines avant le premier candidat. Le parcours du
    calendrier cible ne modifie jamais l'etat d'une equipe.
    """
    _require_sha256(shadow_protocol_sha256, field="shadow_protocol_sha256")
    target = _require_date_string(snapshot["target_official_date"], field="target_official_date")
    batch_id = _require_sha256(snapshot["batch_id"], field="batch_id")
    cutoff = _require_utc_timestamp(snapshot["information_cutoff_utc"], field="information_cutoff_utc")
    feature_as_of = (date.fromisoformat(target) - timedelta(days=1)).isoformat()
    # games, wins, runs_scored, runs_allowed, max_source_date
    states: dict[int, list[Any]] = {}
    previous_team_id = 0
    for row in snapshot["teams"]:
        team = _require_exact_source_object(row, _SOURCE_TEAM_KEYS, field="teams[]")
        team_id = _require_positive_integer(team["team_id"], field="team_id")
        _require_nonempty_text(team["name"], field="team.name")
        if team["abbreviation"] is not None:
            _require_nonempty_text(team["abbreviation"], field="team.abbreviation")
        if team_id <= previous_team_id:
            raise ShadowPredictionError("Equipes dupliquees ou hors ordre canonique.")
        previous_team_id = team_id
        states[team_id] = [0, 0, 0, 0, None]

    source_ids: set[int] = set()
    previous_source_key: tuple[str, int] | None = None
    for raw in snapshot["source_final_games"]:
        row = _require_exact_source_object(raw, _SOURCE_FINAL_GAME_KEYS, field="source_final_games[]")
        game_id = _require_positive_integer(row["game_id"], field="source.game_id")
        season = _require_integer(row["season"], field="source.season")
        official = _require_date_string(row["official_date"], field="source.official_date")
        code, _ = _require_status_text(row["status_code"], field="source.status_code")
        _, detail = _require_status_text(row["status_detail"], field="source.status_detail")
        away = _require_positive_integer(row["away_team_id"], field="source.away_team_id")
        home = _require_positive_integer(row["home_team_id"], field="source.home_team_id")
        away_score = _require_integer(row["away_score"], field="source.away_score")
        home_score = _require_integer(row["home_score"], field="source.home_score")
        key = (official, game_id)
        if (
            season != EXPECTED_TARGET_SEASON or official >= target
            or row["game_type"] != "R" or away == home
            or away not in states or home not in states
            or away_score < 0 or home_score < 0 or away_score == home_score
            or not (code == "F" or detail in {"FINAL", "GAME OVER", "COMPLETED EARLY"})
            or game_id in source_ids
            or (previous_source_key is not None and key <= previous_source_key)
        ):
            raise ShadowPredictionError("Match source malforme, non J-1 ou hors ordre.")
        source_ids.add(game_id)
        previous_source_key = key
        for team_id, scored, allowed in ((away, away_score, home_score), (home, home_score, away_score)):
            state = states[team_id]
            state[0] += 1
            state[1] += int(scored > allowed)
            state[2] += scored
            state[3] += allowed
            state[4] = official if state[4] is None else max(state[4], official)

    ledger: list[list[object]] = []
    features: list[list[object]] = []
    exclusions = {reason: 0 for reason in _ALLOWED_CANDIDATE_EXCLUSION_REASONS}
    target_ids: set[int] = set()
    occurrence_ids: set[str] = set()
    previous_target_key: tuple[int, str, int] | None = None
    for raw in snapshot["target_schedule"]:
        row = _require_exact_source_object(raw, _SOURCE_TARGET_SCHEDULE_KEYS, field="target_schedule[]")
        game_id = _require_positive_integer(row["game_id"], field="target.game_id")
        season = _require_integer(row["season"], field="target.season")
        official = _require_date_string(row["official_date"], field="target.official_date")
        away = _require_positive_integer(row["away_team_id"], field="target.away_team_id")
        home = _require_positive_integer(row["home_team_id"], field="target.home_team_id")
        code, normalized_code = _require_status_text(row["status_code"], field="target.status_code")
        abstract, normalized_abstract = _require_status_text(row["abstract_state"], field="target.abstract_state")
        detail, normalized_detail = _require_status_text(row["detailed_state"], field="target.detailed_state")
        start = row["game_datetime_utc"]
        if start is not None:
            start = _require_utc_timestamp(start, field="target.game_datetime_utc")
            if start <= cutoff:
                raise ShadowPredictionError("Un horaire cible ne suit pas le cutoff fige.")
        if row["doubleheader"] is not None:
            _require_nonempty_text(row["doubleheader"], field="target.doubleheader")
        if row["game_number"] is not None:
            _require_positive_integer(row["game_number"], field="target.game_number")
        key = (int(start is None), start or "", game_id)
        if (
            season != EXPECTED_TARGET_SEASON or official != target
            or row["game_type"] != "R" or away == home
            or away not in states or home not in states
            or game_id in target_ids or game_id in source_ids
            or (previous_target_key is not None and key <= previous_target_key)
        ):
            raise ShadowPredictionError("Identite cible contradictoire ou ordre non canonique.")
        target_ids.add(game_id)
        previous_target_key = key
        occurrence = build_occurrence_key(
            game_id=game_id, official_date_at_snapshot=official,
            scheduled_start_utc_at_snapshot_or_null=start,
        )
        if occurrence in occurrence_ids:
            raise ShadowPredictionError("Une occurrence cible est dupliquee.")
        occurrence_ids.add(occurrence)
        reason: str | None = None
        if normalized_code in _POSTPONED_STATUS_CODES and normalized_detail == "POSTPONED":
            reason = "POSTPONED"
        elif normalized_code in _CANCELLED_STATUS_CODES and normalized_detail == "CANCELLED":
            reason = "CANCELLED"
        else:
            if normalized_abstract in {"LIVE", "FINAL"}:
                raise ShadowPredictionError("Un candidat est deja commence ou termine.")
            if not (
                (normalized_code == "S" and normalized_abstract == "PREVIEW" and normalized_detail == "SCHEDULED")
                or (normalized_code == "P" and normalized_abstract == "PREVIEW" and normalized_detail == "PRE-GAME")
            ):
                raise ShadowPredictionError("Un candidat possede un etat MLB inconnu.")
            away_short = states[away][0] < _MINIMUM_HISTORY_GAMES_PER_TEAM
            home_short = states[home][0] < _MINIMUM_HISTORY_GAMES_PER_TEAM
            if start is None:
                reason = "START_TIME_MISSING"
            elif away_short and home_short:
                reason = "INSUFFICIENT_BOTH_HISTORY"
            elif away_short:
                reason = "INSUFFICIENT_AWAY_HISTORY"
            elif home_short:
                reason = "INSUFFICIENT_HOME_HISTORY"
        ledger.append([
            batch_id, game_id, occurrence, season, official, away, home,
            start, code, abstract, detail,
            "ELIGIBLE" if reason is None else "EXCLUDED", reason,
        ])
        if reason is not None:
            exclusions[reason] += 1
            continue
        assert start is not None
        team_values: list[list[object]] = []
        for team_id in (away, home):
            games, wins, scored, allowed, max_date = states[team_id]
            if games < _MINIMUM_HISTORY_GAMES_PER_TEAM or max_date is None or max_date >= target:
                raise ShadowPredictionError("Historique insuffisant avant division.")
            try:
                values: list[object] = [games] + [
                    _format_feature_rate(numerator / games)
                    for numerator in (wins, scored, allowed)
                ]
            except (OverflowError, ValueError) as error:
                raise ShadowPredictionError("Variable de forme non finie.") from error
            team_values.append(values)
        prediction_id = build_prediction_id(
            shadow_protocol_sha256=shadow_protocol_sha256,
            game_id=game_id, official_date_at_snapshot=official,
            scheduled_start_utc_at_snapshot=start,
        )
        feature_values: list[object] = [
            prediction_id, batch_id, game_id, occurrence, season, official,
            away, home, start, feature_as_of, states[away][4], states[home][4],
            *team_values[0], *team_values[1],
        ]
        feature_hash = build_feature_row_sha256(
            feature_row_values_in_features_columns_exact_order_excluding_feature_row_sha256=feature_values,
        )
        features.append([*feature_values, feature_hash])
    if len(ledger) != len(features) + sum(exclusions.values()):
        raise ShadowPredictionError("Les compteurs du calendrier ne sont pas conserves.")
    if [(r[1], r[2]) for r in ledger if r[11] == "ELIGIBLE"] != [(r[2], r[3]) for r in features]:
        raise ShadowPredictionError("Le registre et les variables ne concordent pas.")
    return ledger, features, exclusions


def build_and_publish_candidate_ledger_and_features(
    reservation: ShadowPredictionSlotReservation,
    activation_publication: ShadowActivationReverificationPublication,
    source_publication: ShadowSourceSnapshotPublication,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowCandidateFeaturesPublication:
    """Publie les fichiers 4 et 5 depuis le snapshot, sans modele ni reseau.

    Toute la validation et les deux serialisations precedent le premier
    lien. Un echec d'E/S conserve les fichiers deja publies et ne permet
    aucune reprise. La fermeture FAILED.json appartient a l'orchestrateur.
    """
    target, _ = _validate_candidate_source_proof(
        reservation, source_publication, project_directory=project_directory
    )
    source_names = frozenset({SOURCE_SNAPSHOT_FILENAME})
    _validate_source_snapshot_predecessors(
        reservation, activation_publication,
        project_directory=project_directory,
        expected_additional_filenames=source_names,
    )
    with _hold_source_snapshot_stage_lock(reservation, project_directory=project_directory):
        _validate_source_snapshot_predecessors(
            reservation, activation_publication,
            project_directory=project_directory,
            expected_additional_filenames=source_names,
        )
        source_bytes, snapshot = _read_candidate_source_snapshot(
            reservation, source_publication, project_directory=project_directory
        )
        ledger, features, exclusions = _build_candidate_feature_rows(
            snapshot,
            shadow_protocol_sha256=reservation.reserved_marker["shadow_protocol_sha256"],
        )
        ledger_bytes = _canonical_csv_bytes(_CANDIDATE_LEDGER_COLUMNS, ledger)
        features_bytes = _canonical_csv_bytes(_FEATURES_COLUMNS, features)
        # Ne jamais publier le registre puis decouvrir une source changee ou
        # une erreur de calcul/serialisation des variables.
        _validate_source_snapshot_predecessors(
            reservation, activation_publication,
            project_directory=project_directory,
            expected_additional_filenames=source_names,
        )
        confirmed_bytes, _ = _read_candidate_source_snapshot(
            reservation, source_publication, project_directory=project_directory
        )
        if confirmed_bytes != source_bytes:
            raise ShadowPredictionError("Le snapshot a change pendant les calculs.")
        ledger_path = reservation.slot_path / CANDIDATE_LEDGER_FILENAME
        features_path = reservation.slot_path / FEATURES_FILENAME
        try:
            ledger_hash = _publish_exclusive_verified(ledger_path, ledger_bytes)
            _validate_source_snapshot_predecessors(
                reservation, activation_publication,
                project_directory=project_directory,
                expected_additional_filenames=source_names | {CANDIDATE_LEDGER_FILENAME},
            )
            if ledger_path.read_bytes() != ledger_bytes or source_publication.snapshot_path.read_bytes() != source_bytes:
                raise ShadowPredictionError("Une source ou le registre a change avant les variables.")
            features_hash = _publish_exclusive_verified(features_path, features_bytes)
            _validate_source_snapshot_predecessors(
                reservation, activation_publication,
                project_directory=project_directory,
                expected_additional_filenames=source_names | {CANDIDATE_LEDGER_FILENAME, FEATURES_FILENAME},
            )
            if (
                ledger_path.read_bytes() != ledger_bytes
                or features_path.read_bytes() != features_bytes
                or source_publication.snapshot_path.read_bytes() != source_bytes
                or ledger_hash != hashlib.sha256(ledger_bytes).hexdigest()
                or features_hash != hashlib.sha256(features_bytes).hexdigest()
            ):
                raise ShadowPredictionError("Les fichiers publies ne sont plus identiques.")
        except ShadowPublicationConflictError as error:
            raise ShadowPredictionSlotConsumedError("Le jalon candidats/variables est deja consomme.") from error
        except OSError as error:
            raise ShadowPredictionError("Publication incomplete; aucune reparation du slot n'est permise.") from error
        relative_root = SHADOW_RESULT_ROOT_RELATIVE_PATH / target
        return ShadowCandidateFeaturesPublication(
            slot_path=reservation.slot_path,
            candidate_ledger_path=ledger_path,
            candidate_ledger_relative_path=(relative_root / CANDIDATE_LEDGER_FILENAME).as_posix(),
            candidate_ledger_sha256=ledger_hash,
            candidate_ledger_size_bytes=len(ledger_bytes),
            candidate_row_count=len(ledger),
            features_path=features_path,
            features_relative_path=(relative_root / FEATURES_FILENAME).as_posix(),
            features_sha256=features_hash,
            features_size_bytes=len(features_bytes),
            feature_row_count=len(features),
            eligible_game_count=len(features),
            excluded_games_by_reason=tuple(exclusions.items()),
            earliest_eligible_scheduled_start_utc=(features[0][8] if features else None),
        )


def _require_model_prerequisite_path(
    project_directory: Path, relative_path: str,
) -> Path:
    """Accepte seulement les quatre chemins figes, sans lien observe."""
    if (
        not isinstance(project_directory, Path)
        or not project_directory.is_absolute()
        or ".." in project_directory.parts
        or relative_path not in _MODEL_PREREQUISITE_PATHS
    ):
        raise ShadowPredictionError("Chemin de controle du modele non autorise.")
    path = project_directory.joinpath(*PurePosixPath(relative_path).parts)
    # Controler aussi les parents du projet, pas seulement le dernier fichier.
    for directory in reversed(path.parents):
        mode = _lstat_mode(directory)
        if not isinstance(mode, int) or not stat.S_ISDIR(mode):
            raise ShadowPredictionError(
                "Un parent des fichiers du modele est absent ou symbolique."
            )
    mode = _lstat_mode(path)
    if not isinstance(mode, int) or not stat.S_ISREG(mode):
        raise ShadowPredictionError(
            f"Fichier du modele absent, non regulier ou symbolique : {relative_path}."
        )
    return path


def _read_pinned_model_prerequisite_file(
    project_directory: Path,
    relative_path: str,
    expected_sha256: str,
    *,
    expected_size: int | None = None,
) -> bytes:
    """Verifie les octets avant toute interpretation, sans deserialisation."""
    _require_sha256(expected_sha256, field="expected_sha256")
    if expected_size is not None:
        _require_positive_integer(expected_size, field="expected_size")
    path = _require_model_prerequisite_path(project_directory, relative_path)
    try:
        before = path.stat(follow_symlinks=False)
        if expected_size is not None and before.st_size != expected_size:
            raise ShadowPredictionError(
                f"Taille du modele incorrecte : {relative_path}."
            )
        content = path.read_bytes()
        _require_model_prerequisite_path(project_directory, relative_path)
        after = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ShadowPredictionError(
            f"Lecture du fichier fige impossible : {relative_path}."
        ) from error
    identity = lambda metadata: (
        metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns
    )
    if (
        identity(before) != identity(after)
        or type(content) is not bytes
        or len(content) != before.st_size
        or (expected_size is not None and len(content) != expected_size)
        or hashlib.sha256(content).hexdigest() != expected_sha256
    ):
        raise ShadowPredictionError(
            f"Fichier fige modifie ou empreinte incorrecte : {relative_path}."
        )
    return content


def _decode_pinned_model_json(content: bytes, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            content.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
        _validate_json_value(payload)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ShadowPredictionError(f"JSON du modele invalide : {name}.") from error
    if type(payload) is not dict:
        raise ShadowPredictionError(f"Objet JSON requis : {name}.")
    return payload


def _validate_model_prerequisite_contracts(
    shadow: dict[str, Any],
    manifest: dict[str, Any],
    model_protocol: dict[str, Any],
) -> dict[str, str]:
    """Reverifie les liaisons des trois documents dont les hashes sont fixes."""
    _validate_protocol_contract(shadow)
    artifact = _require_mapping(manifest.get("artifact"), description="manifest.artifact")
    model = _require_mapping(manifest.get("model"), description="manifest.model")
    protocol_link = _require_mapping(manifest.get("protocol"), description="manifest.protocol")
    policy = _require_mapping(manifest.get("loading_policy"), description="manifest.loading_policy")
    base_model = _require_mapping(model_protocol.get("model"), description="model_protocol.model")
    lineage = shadow["validated_lineage"]["model_artifact"]
    feature_columns = list(_FEATURE_ROW_FIELD_NAMES[12:20])
    for mapping, pairs, context in (
        (manifest, (("manifest_version", 1),
                    ("status", "FROZEN_BEFORE_SEALED_TEST")), "manifest"),
        (artifact, (("path", EXPECTED_MODEL_ARTIFACT_PATH),
                    ("sha256", EXPECTED_MODEL_ARTIFACT_SHA256),
                    ("size_bytes", EXPECTED_MODEL_ARTIFACT_SIZE_BYTES),
                    ("code_version", EXPECTED_MODEL_ARTIFACT_CODE_COMMIT),
                    ("serializer", "joblib"), ("compression_level", 3),
                    ("artifact_format_version", 1),
                    ("round_trip_verified_after_write", True)), "manifest.artifact"),
        (lineage, (("size_bytes", EXPECTED_MODEL_ARTIFACT_SIZE_BYTES),
                   ("code_commit", EXPECTED_MODEL_ARTIFACT_CODE_COMMIT),
                   ("model_version", "logistic_team_form_v1_platt")), "shadow.model_artifact"),
        (model, (("calibrated_model_version", "logistic_team_form_v1_platt"),
                 ("base_model_version", "logistic_team_form_v1"),
                 ("calibration_method", "sigmoid"),
                 ("base_model_unchanged_during_calibration", True),
                 ("feature_columns", feature_columns)), "manifest.model"),
        (protocol_link, (("path", EXPECTED_MODEL_PROTOCOL_PATH),
                         ("sha256", EXPECTED_MODEL_PROTOCOL_SHA256)), "manifest.protocol"),
        (policy, (("pickle_based_format", True),
                  ("verify_sha256_before_deserialization", True),
                  ("accept_untrusted_artifact", False)), "manifest.loading_policy"),
        (model_protocol, (("protocol_version", 1),
                          ("status", "REGISTERED_BEFORE_SEALED_TEST"),
                          ("features", feature_columns)), "model_protocol"),
        (base_model, (("model_version", "logistic_team_form_v1"),),
         "model_protocol.model"),
    ):
        for key, value in pairs:
            _require_exact(mapping, key, value, context=context)
    runtime = manifest.get("runtime")
    if type(runtime) is not dict or frozenset(runtime) != _MODEL_RUNTIME_KEYS:
        raise ShadowPredictionError("Le manifeste doit nommer les six versions exactes.")
    for key, value in runtime.items():
        _require_nonempty_text(value, field=f"runtime.{key}")
        if value != value.strip():
            raise ShadowPredictionError("Une version du manifeste est non canonique.")
    return dict(runtime)


def _installed_model_runtime_versions() -> dict[str, str]:
    """Lit les metadonnees installees sans importer les bibliotheques ML."""
    versions = {"python": platform.python_version()}
    for key, distribution in _MODEL_RUNTIME_DISTRIBUTIONS:
        try:
            versions[key] = distribution_metadata.version(distribution)
        except (distribution_metadata.PackageNotFoundError, OSError, ValueError) as error:
            raise ShadowPredictionError(
                f"Version installee introuvable pour {distribution}."
            ) from error
    for key, value in versions.items():
        _require_nonempty_text(value, field=f"installed_runtime.{key}")
    return versions


def verify_frozen_model_prerequisites(
    *, project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowModelPrerequisites:
    """Controle l'artefact en lecture seule, sans charger aucun objet modele.

    Primitive non branchee sur l'apercu ou un mode d'execution. Les quatre
    chemins et leurs empreintes sont imposes par le protocole deja fige.
    Les octets renvoyes sont immuables et devront etre utilises tels quels
    par un futur chargeur, apres ses controles d'execution supplementaires.
    Le manifeste d'execution et l'activation ne sont PAS certifies ici.
    """
    metadata_specs = (
        (SHADOW_PROTOCOL_RELATIVE_PATH.as_posix(), EXPECTED_SHADOW_PROTOCOL_SHA256),
        (EXPECTED_ARTIFACT_MANIFEST_PATH, EXPECTED_ARTIFACT_MANIFEST_SHA256),
        (EXPECTED_MODEL_PROTOCOL_PATH, EXPECTED_MODEL_PROTOCOL_SHA256),
    )
    # Verifier tous les hashes avant de decoder le premier JSON.
    metadata_bytes = tuple(
        _read_pinned_model_prerequisite_file(project_directory, path, digest)
        for path, digest in metadata_specs
    )
    shadow, manifest, protocol = (
        _decode_pinned_model_json(content, name=spec[0])
        for spec, content in zip(metadata_specs, metadata_bytes)
    )
    expected_runtime = _validate_model_prerequisite_contracts(shadow, manifest, protocol)
    runtime = _installed_model_runtime_versions()
    if runtime != expected_runtime:
        differences = "; ".join(
            f"{key} : installe {runtime[key]}, attendu {value}"
            for key, value in expected_runtime.items() if runtime[key] != value
        )
        raise ShadowPredictionError(f"Environnement du modele incompatible : {differences}.")
    artifact_bytes = _read_pinned_model_prerequisite_file(
        project_directory, EXPECTED_MODEL_ARTIFACT_PATH,
        EXPECTED_MODEL_ARTIFACT_SHA256, expected_size=EXPECTED_MODEL_ARTIFACT_SIZE_BYTES,
    )
    # Recontroler les fichiers et versions pour refuser les changements observes.
    # Ceci n'est pas un verrou global : seuls les octets verifies ci-dessous
    # font foi, pas une promesse que le disque restera inchange apres le retour.
    for (path, digest), original in zip(metadata_specs, metadata_bytes):
        if _read_pinned_model_prerequisite_file(project_directory, path, digest) != original:
            raise ShadowPredictionError("Les metadonnees du modele ont change.")
    if _installed_model_runtime_versions() != runtime:
        raise ShadowPredictionError("L'environnement a change pendant les controles.")
    if _read_pinned_model_prerequisite_file(
        project_directory, EXPECTED_MODEL_ARTIFACT_PATH,
        EXPECTED_MODEL_ARTIFACT_SHA256, expected_size=EXPECTED_MODEL_ARTIFACT_SIZE_BYTES,
    ) != artifact_bytes:
        raise ShadowPredictionError("Les octets du modele ont change.")
    return ShadowModelPrerequisites(
        artifact_path=project_directory.joinpath(*PurePosixPath(EXPECTED_MODEL_ARTIFACT_PATH).parts),
        artifact_relative_path=EXPECTED_MODEL_ARTIFACT_PATH,
        artifact_sha256=EXPECTED_MODEL_ARTIFACT_SHA256,
        artifact_size_bytes=len(artifact_bytes),
        artifact_bytes=artifact_bytes,
        artifact_manifest_sha256=EXPECTED_ARTIFACT_MANIFEST_SHA256,
        model_protocol_sha256=EXPECTED_MODEL_PROTOCOL_SHA256,
        shadow_protocol_sha256=EXPECTED_SHADOW_PROTOCOL_SHA256,
        runtime_versions=tuple(sorted(runtime.items())),
    )


def _validate_shadow_model_loading_proof(
    proof: ShadowModelPrerequisites, project_directory: Path,
) -> None:
    """Une dataclass fabriquee par l'appelant ne constitue pas une confiance."""
    if type(proof) is not ShadowModelPrerequisites:
        raise ShadowPredictionError("Preuve exacte des prerequis du modele requise.")
    if (
        not isinstance(project_directory, Path)
        or not project_directory.is_absolute()
        or ".." in project_directory.parts
        or not isinstance(proof.artifact_path, Path)
        or proof.artifact_path != project_directory.joinpath(
            *PurePosixPath(EXPECTED_MODEL_ARTIFACT_PATH).parts
        )
    ):
        raise ShadowPredictionError("La preuve du modele appartient a un autre projet.")
    for key, expected in (
        ("artifact_relative_path", EXPECTED_MODEL_ARTIFACT_PATH),
        ("artifact_sha256", EXPECTED_MODEL_ARTIFACT_SHA256),
        ("artifact_size_bytes", EXPECTED_MODEL_ARTIFACT_SIZE_BYTES),
        ("artifact_manifest_sha256", EXPECTED_ARTIFACT_MANIFEST_SHA256),
        ("model_protocol_sha256", EXPECTED_MODEL_PROTOCOL_SHA256),
        ("shadow_protocol_sha256", EXPECTED_SHADOW_PROTOCOL_SHA256),
        ("validation_scope", "FROZEN_MODEL_FILES_AND_INSTALLED_RUNTIME_ONLY"),
        ("runtime_versions_source", "PYTHON_AND_INSTALLED_DISTRIBUTION_METADATA"),
        ("execution_manifest_verified", False), ("activation_verified", False),
        ("model_deserialized", False), ("predictions_computed", False),
        ("execution_ready", False),
    ):
        _require_exact({key: getattr(proof, key)}, key, expected, context="prerequisites")
    content = proof.artifact_bytes
    if (
        type(content) is not bytes
        or len(content) != EXPECTED_MODEL_ARTIFACT_SIZE_BYTES
        or hashlib.sha256(content).hexdigest() != EXPECTED_MODEL_ARTIFACT_SHA256
    ):
        raise ShadowPredictionError("Les octets en memoire du modele ne sont pas fiables.")
    if (
        type(proof.runtime_versions) is not tuple
        or any(type(pair) is not tuple or len(pair) != 2
               or any(type(value) is not str for value in pair)
               for pair in proof.runtime_versions)
    ):
        raise ShadowPredictionError("Versions de la preuve non canoniques.")


def _shadow_imported_runtime_versions(modules: Mapping[str, Any]) -> dict[str, str]:
    """Les modules effectivement importes doivent aussi avoir les bonnes versions."""
    versions = {"python": platform.python_version()}
    for key, _ in _MODEL_RUNTIME_DISTRIBUTIONS:
        version = getattr(modules[key], "__version__", None)
        _require_nonempty_text(version, field=f"imported_runtime.{key}")
        versions[key] = version
    return versions


def _import_shadow_model_runtime(expected_runtime: dict[str, str]) -> dict[str, Any]:
    """Import tardif, apres verification des octets et de leurs documents figes."""
    modules = {
        key: import_module("sklearn" if key == "scikit_learn" else key)
        for key, _ in _MODEL_RUNTIME_DISTRIBUTIONS
    }
    if _shadow_imported_runtime_versions(modules) != expected_runtime:
        raise ShadowPredictionError("Versions des modules importes incompatibles.")
    modules["artifact_module"] = import_module("src.calibrated_model")
    modules["calibration_module"] = import_module("sklearn.calibration")
    return modules


def _deserialize_shadow_model_once(
    content: bytes, modules: Mapping[str, Any],
) -> tuple[Any, int]:
    """Un seul joblib.load, avec le filtre exact du protocole et alias temporaire.

    L'appelant detient _MODEL_LOADING_LOCK et a controle les versions exactes.
    Le verrou protege nos chargeurs concurrents, pas du code Python tiers qui
    modifierait lui-meme les filtres globaux ou __main__.
    """
    main_module = sys.modules.get("__main__")
    if main_module is None:
        raise ShadowPredictionError("Module __main__ absent pour la compatibilite.")
    artifact_type = modules["artifact_module"].CalibratedModelArtifact
    missing = object()
    previous = vars(main_module).get("CalibratedModelArtifact", missing)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("error")
        warnings.filterwarnings(
            "always", message=_MODEL_WARNING_MESSAGE,
            category=DeprecationWarning, module=_MODEL_WARNING_MODULE, append=False,
        )
        vars(main_module)["CalibratedModelArtifact"] = artifact_type
        try:
            artifact = modules["joblib"].load(io.BytesIO(content))
        finally:
            if previous is missing:
                vars(main_module).pop("CalibratedModelArtifact", None)
            else:
                vars(main_module)["CalibratedModelArtifact"] = previous
    return artifact, len(recorded)


def _validate_loaded_shadow_model_metadata(
    artifact: Any, manifest: dict[str, Any], modules: Mapping[str, Any],
) -> None:
    """Controle le type, toutes les metadonnees et les classes, sans prediction."""
    if type(artifact) is not modules["artifact_module"].CalibratedModelArtifact:
        raise ShadowPredictionError("Type exact de CalibratedModelArtifact requis.")
    model = manifest["model"]
    chronology = manifest["chronology"]
    expected_fields = {
        "artifact_format_version": manifest["artifact"]["artifact_format_version"],
        "calibrated_model_version": model["calibrated_model_version"],
        "base_model_version": model["base_model_version"],
        "code_version": manifest["artifact"]["code_version"],
        "dataset_version": manifest["dataset"]["version"],
        "dataset_sha256": manifest["dataset"]["sha256"],
        "protocol_sha256": manifest["protocol"]["sha256"],
        "feature_columns": tuple(model["feature_columns"]),
        "base_training_seasons": tuple(chronology["base_training_seasons"]),
        "calibration_season": chronology["calibration_season"],
        "sealed_test_seasons": tuple(chronology["sealed_test_seasons"]),
        "recent_seasons": tuple(chronology["recent_seasons"]),
        "calibration_method": model["calibration_method"],
        "sklearn_version": manifest["runtime"]["scikit_learn"],
        "numpy_version": manifest["runtime"]["numpy"],
    }
    for name, expected in expected_fields.items():
        actual = getattr(artifact, name, None)
        _require_exact({name: actual}, name, expected, context="loaded_artifact")
        if type(expected) is tuple and any(
            type(item) is not type(reference) for item, reference in zip(actual, expected)
        ):
            raise ShadowPredictionError(f"Types de metadonnees inattendus : {name}.")
    classifier = artifact.calibrated_classifier
    if type(classifier) is not modules["calibration_module"].CalibratedClassifierCV:
        raise ShadowPredictionError("Classifieur calibre exact requis.")
    if getattr(classifier, "method", None) != "sigmoid":
        raise ShadowPredictionError("Le classifieur ne declare pas la calibration sigmoid.")
    np = modules["numpy"]
    classes = getattr(classifier, "classes_", None)
    if (
        type(classes) is not np.ndarray or classes.shape != (2,)
        or classes.dtype.kind not in "iu" or classes.tolist() != [0, 1]
    ):
        raise ShadowPredictionError("Les classes doivent etre exactement les entiers [0, 1].")
    n_features = getattr(classifier, "n_features_in_", None)
    if (
        isinstance(n_features, (bool, np.bool_))
        or not isinstance(n_features, (int, np.integer))
        or n_features != len(model["feature_columns"])
    ):
        raise ShadowPredictionError("Le classifieur doit attendre les huit variables figees.")


def _load_frozen_shadow_model(
    prerequisites: ShadowModelPrerequisites,
    *, project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowLoadedModel:
    """Brique interne de chargement, pas un point d'entree d'execution officielle.

    Ne relit jamais le fichier joblib : seul le tampon immuable reverifie est
    transmis au deserialiseur. L'orchestrateur devra verifier auparavant les
    autorisations d'execution/activation, le creneau et l'egalite des versions
    avec le manifeste d'execution. Ces controles restent hors de ce lot.
    """
    if not _MODEL_LOADING_LOCK.acquire(blocking=False):
        raise ShadowPredictionError("Un chargement shadow est deja en cours dans ce processus.")
    try:
        # Hors de joblib.load, aucun avertissement n'est autorise, y compris
        # pendant les controles de fichiers et de provenance avant/apres lui.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_shadow_model_loading_proof(prerequisites, project_directory)
            specs = (
                (SHADOW_PROTOCOL_RELATIVE_PATH.as_posix(), EXPECTED_SHADOW_PROTOCOL_SHA256),
                (EXPECTED_ARTIFACT_MANIFEST_PATH, EXPECTED_ARTIFACT_MANIFEST_SHA256),
                (EXPECTED_MODEL_PROTOCOL_PATH, EXPECTED_MODEL_PROTOCOL_SHA256),
            )
            documents = tuple(
                _read_pinned_model_prerequisite_file(project_directory, path, digest)
                for path, digest in specs
            )
            shadow, manifest, protocol = (
                _decode_pinned_model_json(content, name=spec[0])
                for spec, content in zip(specs, documents)
            )
            runtime = _validate_model_prerequisite_contracts(shadow, manifest, protocol)
            if prerequisites.runtime_versions != tuple(sorted(runtime.items())):
                raise ShadowPredictionError("Les versions de la preuve different du manifeste.")
            if _installed_model_runtime_versions() != runtime:
                raise ShadowPredictionError("L'environnement installe a change avant chargement.")
            modules = _import_shadow_model_runtime(runtime)
            artifact, warning_count = _deserialize_shadow_model_once(
                prerequisites.artifact_bytes, modules
            )
            _validate_loaded_shadow_model_metadata(artifact, manifest, modules)
            if (
                _shadow_imported_runtime_versions(modules) != runtime
                or _installed_model_runtime_versions() != runtime
            ):
                raise ShadowPredictionError("L'environnement a change pendant le chargement.")
            for (path, digest), original in zip(specs, documents):
                if _read_pinned_model_prerequisite_file(project_directory, path, digest) != original:
                    raise ShadowPredictionError("Les documents figes ont change pendant le chargement.")
            return ShadowLoadedModel(
                artifact=artifact, artifact_sha256=EXPECTED_MODEL_ARTIFACT_SHA256,
                runtime_versions=tuple(sorted(runtime.items())),
                approved_deserialization_warning_count=warning_count,
            )
    except ShadowPredictionError:
        raise
    except Exception as error:
        raise ShadowPredictionError(
            f"Chargement du modele fige refuse : {type(error).__name__}: {error}"
        ) from error
    finally:
        _MODEL_LOADING_LOCK.release()


def _validate_shadow_state_loaded_envelope(loaded: ShadowLoadedModel) -> None:
    """Refuse une enveloppe incompatible avant imports et serialisation.

    Les objets internes ne sont pas une frontiere de securite contre du code
    Python arbitraire. L'appelant doit provenir du chargeur controle ; cette
    validation ne prouve pas l'origine d'un estimateur fabrique en memoire.
    """
    if type(loaded) is not ShadowLoadedModel:
        raise ShadowPredictionError("Objet ShadowLoadedModel exact requis.")
    for key, expected in (
        ("artifact_sha256", EXPECTED_MODEL_ARTIFACT_SHA256),
        ("warning_policy_id", _MODEL_WARNING_POLICY_ID),
        ("model_deserialized", True), ("predictions_computed", False),
        ("execution_manifest_verified", False), ("activation_verified", False),
        ("execution_ready", False),
    ):
        _require_exact({key: getattr(loaded, key)}, key, expected, context="loaded_model")
    count = loaded.approved_deserialization_warning_count
    if type(count) is not int or count < 0:
        raise ShadowPredictionError("Compteur d'avertissements du chargement invalide.")
    versions = loaded.runtime_versions
    if (
        type(versions) is not tuple
        or any(type(pair) is not tuple or len(pair) != 2
               or any(type(value) is not str for value in pair) for pair in versions)
        or tuple(sorted(dict(versions).items())) != versions
        or frozenset(dict(versions)) != _MODEL_RUNTIME_KEYS
        or any(not value or value != value.strip() for _, value in versions)
    ):
        raise ShadowPredictionError("Versions de l'objet charge non canoniques.")


def _snapshot_frozen_shadow_model_state(
    loaded_model: ShadowLoadedModel,
    *, project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowModelStateSnapshot:
    """Un seul joblib.dump de l'artefact ENTIER dans BytesIO, compress=3.

    Aucun chargement, fit, predict, fichier modele ou sortie officielle.
    Le verrou du chargeur est partage car les filtres d'avertissement sont
    communs. Il couvre cette operation, pas l'intervalle entre deux mesures.
    Le futur orchestrateur doit posseder exclusivement le modele pendant
    toute la sequence mesure/prediction/mesure, aux versions du manifeste
    d'execution. Aucun de ces controles d'autorisation n'est atteste ici.
    """
    if not _MODEL_LOADING_LOCK.acquire(blocking=False):
        raise ShadowPredictionError("Une operation sur le modele shadow est deja en cours.")
    try:
        return _snapshot_frozen_shadow_model_state_under_lock(
            loaded_model, project_directory=project_directory,
        )
    except ShadowPredictionError:
        raise
    except Exception as error:
        raise ShadowPredictionError(
            f"Empreinte de l'etat du modele refusee : {type(error).__name__}: {error}"
        ) from error
    finally:
        _MODEL_LOADING_LOCK.release()


def _snapshot_frozen_shadow_model_state_under_lock(
    loaded_model: ShadowLoadedModel,
    *, project_directory: Path,
) -> ShadowModelStateSnapshot:
    """Corps du controle existant ; seul un appelant detenant le verrou l'utilise.

    locked() est un garde defensif, pas une preuve de proprietaire de thread.
    Ce helper prive ne remplace jamais l'entree autonome verrouillee.
    """
    if not _MODEL_LOADING_LOCK.locked():
        raise ShadowPredictionError("Le controle d'etat exige le verrou du modele.")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validate_shadow_state_loaded_envelope(loaded_model)
        specs = (
            (SHADOW_PROTOCOL_RELATIVE_PATH.as_posix(), EXPECTED_SHADOW_PROTOCOL_SHA256),
            (EXPECTED_ARTIFACT_MANIFEST_PATH, EXPECTED_ARTIFACT_MANIFEST_SHA256),
            (EXPECTED_MODEL_PROTOCOL_PATH, EXPECTED_MODEL_PROTOCOL_SHA256),
        )
        documents = tuple(
            _read_pinned_model_prerequisite_file(project_directory, path, digest)
            for path, digest in specs
        )
        shadow, manifest, protocol = (
            _decode_pinned_model_json(content, name=spec[0])
            for spec, content in zip(specs, documents)
        )
        runtime = _validate_model_prerequisite_contracts(shadow, manifest, protocol)
        if (
            loaded_model.runtime_versions != tuple(sorted(runtime.items()))
            or _installed_model_runtime_versions() != runtime
        ):
            raise ShadowPredictionError("Environnement incompatible avant empreinte d'etat.")
        modules = _import_shadow_model_runtime(runtime)
        artifact = loaded_model.artifact
        _validate_loaded_shadow_model_metadata(artifact, manifest, modules)
        # Ce filtre approuve uniquement ARTIFACT_STATE_SERIALIZATION.
        # Aucune autre operation n'est incluse dans son contexte.
        with io.BytesIO() as buffer:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("error")
                warnings.filterwarnings(
                    "always", message=_MODEL_WARNING_MESSAGE,
                    category=DeprecationWarning, module=_MODEL_WARNING_MODULE,
                    append=False,
                )
                modules["joblib"].dump(artifact, buffer, compress=3)
            content = buffer.getvalue()
            if not content:
                raise ShadowPredictionError("Serialisation d'etat vide.")
            digest = hashlib.sha256(content).hexdigest()
        _validate_shadow_state_loaded_envelope(loaded_model)
        if loaded_model.artifact is not artifact:
            raise ShadowPredictionError("Objet modele remplace pendant la serialisation.")
        _validate_loaded_shadow_model_metadata(artifact, manifest, modules)
        if (
            _shadow_imported_runtime_versions(modules) != runtime
            or _installed_model_runtime_versions() != runtime
            or loaded_model.runtime_versions != tuple(sorted(runtime.items()))
        ):
            raise ShadowPredictionError("L'environnement a change pendant l'empreinte d'etat.")
        for (path, expected), original in zip(specs, documents):
            if _read_pinned_model_prerequisite_file(project_directory, path, expected) != original:
                raise ShadowPredictionError("Les documents figes ont change pendant l'empreinte.")
        return ShadowModelStateSnapshot(
            loaded_model=loaded_model, artifact_state_sha256=digest,
            runtime_versions=tuple(sorted(runtime.items())),
            approved_state_serialization_warning_count=len(recorded),
        )


def _verify_frozen_shadow_model_state_unchanged(
    loaded_model: ShadowLoadedModel,
    before: ShadowModelStateSnapshot,
    *, project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowModelStateVerification:
    """Recalcule le second etat ; une difference est une erreur, sans reparation.

    Ne prend ni fonction de prediction ni empreinte apres fournie par l'appelant.
    Ne garantit pas qu'une prediction a eu lieu, ni l'absence d'une mutation
    transitoire annulee entre deux mesures. Les objets sont internes au moteur.
    """
    if type(before) is not ShadowModelStateSnapshot:
        raise ShadowPredictionError("Mesure initiale d'etat exacte requise.")
    _validate_shadow_state_loaded_envelope(loaded_model)
    if (
        before.loaded_model is not loaded_model
        or type(before.runtime_versions) is not tuple
        or before.runtime_versions != loaded_model.runtime_versions
        or type(before.warning_policy_id) is not str
        or before.warning_policy_id != _MODEL_WARNING_POLICY_ID
        or type(before.approved_state_serialization_warning_count) is not int
        or before.approved_state_serialization_warning_count < 0
    ):
        raise ShadowPredictionError("Mesure initiale et modele incompatibles.")
    _require_sha256(before.artifact_state_sha256, field="artifact_state_sha256_before")
    after = _snapshot_frozen_shadow_model_state(
        loaded_model, project_directory=project_directory,
    )
    if before.artifact_state_sha256 != after.artifact_state_sha256:
        raise ShadowPredictionError("L'etat serialise du modele a change ; execution refusee.")
    return ShadowModelStateVerification(
        artifact_state_sha256_before=before.artifact_state_sha256,
        artifact_state_sha256_after=after.artifact_state_sha256,
        approved_state_serialization_warning_count=(
            before.approved_state_serialization_warning_count
            + after.approved_state_serialization_warning_count
        ),
    )


def _validate_shadow_prediction_feature_rows(
    feature_rows: object,
) -> tuple[tuple[object, ...], ...]:
    """Copie figee des 21 champs types, sans tri, imputation ou arrondi nouveau.

    Verifie la coherence des lignes, pas leur provenance depuis un fichier
    officiel. L'orchestrateur doit fournir les lignes du jalon deja verifie.
    """
    if type(feature_rows) not in (list, tuple) or not feature_rows:
        raise ShadowPredictionError("Un lot non vide de lignes de variables est requis.")
    rows = []
    previous = None
    game_ids: set[int] = set()
    batch = target = None
    for raw in feature_rows:
        if type(raw) not in (list, tuple) or len(raw) != len(_FEATURES_COLUMNS):
            raise ShadowPredictionError("Une ligne doit contenir les 21 champs exacts.")
        row = tuple(raw)
        expected_hash = build_feature_row_sha256(
            feature_row_values_in_features_columns_exact_order_excluding_feature_row_sha256=list(row[:-1]),
        )
        if _require_sha256(row[20], field="feature_row_sha256") != expected_hash:
            raise ShadowPredictionError("L'empreinte de la ligne de variables est incoherente.")
        game_id, season, official, away, home = row[2], row[4], row[5], row[6], row[7]
        if (
            game_id <= 0 or away <= 0 or home <= 0 or away == home
            or season != EXPECTED_TARGET_SEASON
            or date.fromisoformat(official).year != season
            or date.fromisoformat(official) < EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON
            or row[9] != (date.fromisoformat(official) - timedelta(days=1)).isoformat()
            or any(not f"{season}-01-01" <= row[i] < official for i in (10, 11))
            or any(row[i] < _MINIMUM_HISTORY_GAMES_PER_TEAM for i in (12, 16))
            or (batch is not None and (row[1] != batch or official != target))
        ):
            raise ShadowPredictionError("Identites, dates J-1 ou historique de variables invalides.")
        if row[0] != build_prediction_id(
            shadow_protocol_sha256=EXPECTED_SHADOW_PROTOCOL_SHA256,
            game_id=game_id, official_date_at_snapshot=official,
            scheduled_start_utc_at_snapshot=row[8],
        ) or row[3] != build_occurrence_key(
            game_id=game_id, official_date_at_snapshot=official,
            scheduled_start_utc_at_snapshot_or_null=row[8],
        ):
            raise ShadowPredictionError("Identifiants de prediction ou d'occurrence incoherents.")
        for index in _FEATURE_ROW_FIXED_6_INDEXES:
            value = float(row[index])
            if not math.isfinite(value) or (index in (13, 17) and value > 1):
                raise ShadowPredictionError("Taux non fini ou proportion de victoires invalide.")
        key = (row[8], game_id)
        if game_id in game_ids or (previous is not None and key <= previous):
            raise ShadowPredictionError("Lignes dupliquees ou hors ordre canonique.")
        game_ids.add(game_id)
        previous, batch, target = key, row[1], official
        rows.append(row)
    return tuple(rows)


def _predict_frozen_shadow_model_once(
    loaded_model: ShadowLoadedModel,
    feature_rows: list[list[object]] | tuple[tuple[object, ...], ...],
    *, project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowModelProbabilities:
    """Un seul appel pour un lot NON VIDE, sans chargement, reseau ou ecriture.

    Meme verrou pendant preparation/empreinte/predict_proba/empreinte/validation.
    Toute erreur est terminale pour cet appel, sans retry ni resultat partiel.
    Un lot vide doit eviter le chargeur et cette primitive. Le futur
    orchestrateur garantit l'unicite durable du slot et ses preconditions ;
    cette fonction interne ne constitue ni ce journal ni une autorisation.
    """
    if not _MODEL_LOADING_LOCK.acquire(blocking=False):
        raise ShadowPredictionError("Une operation sur le modele shadow est deja en cours.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_shadow_state_loaded_envelope(loaded_model)
            rows = _validate_shadow_prediction_feature_rows(feature_rows)
            runtime = dict(loaded_model.runtime_versions)
            if _installed_model_runtime_versions() != runtime:
                raise ShadowPredictionError("Environnement incompatible avant preparation de matrice.")
            modules = _import_shadow_model_runtime(runtime)
            np = modules["numpy"]
            # Seules les huit colonnes autorisees, deja quantifiees en .6f,
            # sont converties. Aucun identifiant, horaire ou score cible.
            matrix = np.asarray([
                [row[i] if i in (12, 16) else float(row[i]) for i in range(12, 20)]
                for row in rows
            ], dtype=np.float64)
            if matrix.shape != (len(rows), 8) or not np.isfinite(matrix).all():
                raise ShadowPredictionError("Matrice de variables non finie ou mal dimensionnee.")
            for row, values in zip(rows, matrix):
                if int(values[0]) != row[12] or int(values[4]) != row[16]:
                    raise ShadowPredictionError("Nombre de matchs non representable exactement en float64.")
            matrix.setflags(write=False)
            matrix_bytes = matrix.tobytes()
            artifact = loaded_model.artifact
            load_warning_count = loaded_model.approved_deserialization_warning_count
            before = _snapshot_frozen_shadow_model_state_under_lock(
                loaded_model, project_directory=project_directory,
            )
            # L'exception de compatibilite du dump est deja fermee : MEME
            # l'avertissement approuve est une erreur pendant predict_proba.
            raw = artifact.calibrated_classifier.predict_proba(matrix)
            after = _snapshot_frozen_shadow_model_state_under_lock(
                loaded_model, project_directory=project_directory,
            )
            if (
                before.artifact_state_sha256 != after.artifact_state_sha256
                or loaded_model.artifact is not artifact
                or before.runtime_versions != after.runtime_versions
                or loaded_model.approved_deserialization_warning_count != load_warning_count
            ):
                raise ShadowPredictionError("Le modele ou son enveloppe a change pendant la prediction.")
            if (
                type(matrix) is not np.ndarray or matrix.dtype != np.dtype(np.float64)
                or matrix.shape != (len(rows), 8) or matrix.flags.writeable
                or matrix.tobytes() != matrix_bytes
            ):
                raise ShadowPredictionError("La matrice de variables a ete modifiee.")
            if (
                type(raw) is not np.ndarray or raw.shape != (len(rows), 2)
                or raw.dtype.kind != "f" or not np.isfinite(raw).all()
                or (raw < 0).any() or (raw > 1).any()
            ):
                raise ShadowPredictionError("Sortie predict_proba invalide : tableau fini N x 2 requis.")
            probabilities = []
            for away_raw, home_raw in raw:
                away_column, p_home = float(away_raw), float(home_raw)
                _format_probability_float(away_column)
                _format_probability_float(p_home)
                if abs(away_column + p_home - 1.0) > 1e-12:
                    raise ShadowPredictionError("Les colonnes de probabilites ne somment pas a un.")
                p_away = 1.0 - p_home
                # Aucun clipping, renormalisation, calibrage ou arrondi.
                _format_probability_float(p_away)
                probabilities.append((p_away, p_home))
            state_warning_count = (
                before.approved_state_serialization_warning_count
                + after.approved_state_serialization_warning_count
            )
            return ShadowModelProbabilities(
                feature_rows=rows, probabilities=tuple(probabilities),
                artifact_sha256=loaded_model.artifact_sha256,
                artifact_state_sha256_before=before.artifact_state_sha256,
                artifact_state_sha256_after=after.artifact_state_sha256,
                runtime_versions=before.runtime_versions,
                approved_deserialization_warning_count=load_warning_count,
                approved_state_serialization_warning_count=state_warning_count,
                approved_compatibility_warning_count=load_warning_count + state_warning_count,
            )
    except ShadowPredictionError:
        raise
    except Exception as error:
        raise ShadowPredictionError(
            f"Prediction interne refusee : {type(error).__name__}: {error}"
        ) from error
    finally:
        _MODEL_LOADING_LOCK.release()


def _read_validated_prediction_predecessors(
    reservation: ShadowPredictionSlotReservation,
    activation_publication: ShadowActivationReverificationPublication,
    source_publication: ShadowSourceSnapshotPublication,
    candidate_publication: ShadowCandidateFeaturesPublication,
    *,
    project_directory: Path,
) -> tuple[bytes, bytes, bytes, list[list[object]], str, str]:
    """Relit et reconstruit exactement les cinq fichiers precedents.

    Aucun CSV fourni par l'appelant n'est considere comme une source de
    verite : registre et variables sont recalcules depuis le snapshot source
    immuable, puis compares octet pour octet aux fichiers publies.
    """
    target, cutoff = _validate_candidate_source_proof(
        reservation, source_publication, project_directory=project_directory
    )
    if type(candidate_publication) is not ShadowCandidateFeaturesPublication:
        raise ShadowPredictionError(
            "Une preuve exacte des candidats et variables est requise."
        )
    expected_slot = project_directory.joinpath(
        *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts, target
    )
    relative_root = SHADOW_RESULT_ROOT_RELATIVE_PATH / target
    candidate_path = expected_slot / CANDIDATE_LEDGER_FILENAME
    features_path = expected_slot / FEATURES_FILENAME
    if (
        reservation.slot_path != expected_slot
        or candidate_publication.slot_path != expected_slot
        or candidate_publication.candidate_ledger_path != candidate_path
        or candidate_publication.features_path != features_path
        or candidate_publication.candidate_ledger_relative_path
        != (relative_root / CANDIDATE_LEDGER_FILENAME).as_posix()
        or candidate_publication.features_relative_path
        != (relative_root / FEATURES_FILENAME).as_posix()
    ):
        raise ShadowPredictionError(
            "La preuve candidats/variables ne vise pas le slot exact."
        )

    expected_names = frozenset(
        {
            SOURCE_SNAPSHOT_FILENAME,
            CANDIDATE_LEDGER_FILENAME,
            FEATURES_FILENAME,
        }
    )
    _validate_source_snapshot_predecessors(
        reservation,
        activation_publication,
        project_directory=project_directory,
        expected_additional_filenames=expected_names,
    )
    source_bytes, snapshot = _read_candidate_source_snapshot(
        reservation, source_publication, project_directory=project_directory
    )
    ledger, features, exclusions = _build_candidate_feature_rows(
        snapshot,
        shadow_protocol_sha256=reservation.reserved_marker[
            "shadow_protocol_sha256"
        ],
    )
    candidate_bytes = _canonical_csv_bytes(_CANDIDATE_LEDGER_COLUMNS, ledger)
    features_bytes = _canonical_csv_bytes(_FEATURES_COLUMNS, features)
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    features_sha256 = hashlib.sha256(features_bytes).hexdigest()
    expected_earliest = features[0][8] if features else None
    expected_values = {
        "candidate_ledger_sha256": candidate_sha256,
        "candidate_ledger_size_bytes": len(candidate_bytes),
        "candidate_row_count": len(ledger),
        "features_sha256": features_sha256,
        "features_size_bytes": len(features_bytes),
        "feature_row_count": len(features),
        "eligible_game_count": len(features),
        "excluded_games_by_reason": tuple(exclusions.items()),
        "earliest_eligible_scheduled_start_utc": expected_earliest,
    }
    for name, expected in expected_values.items():
        actual = getattr(candidate_publication, name)
        if type(actual) is not type(expected) or actual != expected:
            raise ShadowPredictionError(
                f"La preuve candidats/variables diverge pour {name}."
            )
    try:
        persisted_candidate = candidate_path.read_bytes()
        persisted_features = features_path.read_bytes()
    except OSError as error:
        raise ShadowPredictionError(
            "Les fichiers candidats/variables sont illisibles."
        ) from error
    if (
        persisted_candidate != candidate_bytes
        or persisted_features != features_bytes
        or hashlib.sha256(persisted_candidate).hexdigest()
        != candidate_publication.candidate_ledger_sha256
        or hashlib.sha256(persisted_features).hexdigest()
        != candidate_publication.features_sha256
    ):
        raise ShadowPredictionError(
            "Les fichiers candidats/variables ne correspondent plus a leur preuve."
        )
    return (
        source_bytes,
        candidate_bytes,
        features_bytes,
        features,
        target,
        cutoff,
    )


def _validate_probabilities_for_publication(
    model_probabilities: ShadowModelProbabilities | None,
    feature_rows: list[list[object]],
) -> tuple[
    tuple[tuple[float, float], ...],
    str | None,
    str | None,
    tuple[tuple[str, str], ...],
    int,
    int,
    int,
    int,
]:
    """Lie le resultat interne exact aux variables, sans nouvel appel modele."""
    if not feature_rows:
        if model_probabilities is not None:
            raise ShadowPredictionError(
                "Un lot vide ne doit porter aucun resultat de modele."
            )
        return (), None, None, (), 0, 0, 0, 0

    if type(model_probabilities) is not ShadowModelProbabilities:
        raise ShadowPredictionError(
            "Le resultat interne exact du modele est requis pour un lot non vide."
        )
    expected_feature_rows = tuple(tuple(row) for row in feature_rows)
    if model_probabilities.feature_rows != expected_feature_rows:
        raise ShadowPredictionError(
            "Les probabilites ne correspondent pas aux variables publiees."
        )
    for name, expected in (
        ("artifact_sha256", EXPECTED_MODEL_ARTIFACT_SHA256),
        ("warning_policy_id", _MODEL_WARNING_POLICY_ID),
        ("predict_proba_calls", 1),
        ("artifact_state_unchanged", True),
        ("unexpected_warning_count", 0),
        ("execution_manifest_verified", False),
        ("activation_verified", False),
        ("official_prediction_created", False),
        ("execution_ready", False),
    ):
        actual = getattr(model_probabilities, name)
        if type(actual) is not type(expected) or actual != expected:
            raise ShadowPredictionError(
                f"Le resultat modele diverge pour {name}."
            )
    before = _require_sha256(
        model_probabilities.artifact_state_sha256_before,
        field="artifact_state_sha256_before",
    )
    after = _require_sha256(
        model_probabilities.artifact_state_sha256_after,
        field="artifact_state_sha256_after",
    )
    if before != after:
        raise ShadowPredictionError(
            "L'etat du modele n'est pas reste identique."
        )
    versions = model_probabilities.runtime_versions
    if (
        type(versions) is not tuple
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(value) is not str for value in pair)
            for pair in versions
        )
        or tuple(sorted(dict(versions).items())) != versions
        or frozenset(dict(versions)) != _MODEL_RUNTIME_KEYS
        or any(not value or value != value.strip() for _, value in versions)
    ):
        raise ShadowPredictionError(
            "Les versions du resultat modele ne sont pas canoniques."
        )
    counts = (
        model_probabilities.approved_deserialization_warning_count,
        model_probabilities.approved_state_serialization_warning_count,
        model_probabilities.approved_compatibility_warning_count,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise ShadowPredictionError(
            "Les compteurs d'avertissements du modele sont invalides."
        )
    if counts[2] != counts[0] + counts[1]:
        raise ShadowPredictionError(
            "Le compteur total d'avertissements du modele est incoherent."
        )
    probabilities = model_probabilities.probabilities
    if type(probabilities) is not tuple or len(probabilities) != len(feature_rows):
        raise ShadowPredictionError(
            "Une paire de probabilites exacte est requise par ligne."
        )
    for pair in probabilities:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(value) is not float for value in pair)
        ):
            raise ShadowPredictionError(
                "Chaque sortie doit etre une paire de flottants exacte."
            )
        p_away, p_home = pair
        _format_probability_float(p_home)
        _format_probability_float(p_away)
        if p_away != 1.0 - p_home or abs(p_away + p_home - 1.0) > 1e-12:
            raise ShadowPredictionError(
                "La probabilite exterieure doit etre le complement exact."
            )
    return (
        probabilities,
        before,
        after,
        versions,
        counts[0],
        counts[1],
        counts[2],
        1,
    )


def _build_and_publish_shadow_predictions(
    reservation: ShadowPredictionSlotReservation,
    activation_publication: ShadowActivationReverificationPublication,
    source_publication: ShadowSourceSnapshotPublication,
    candidate_publication: ShadowCandidateFeaturesPublication,
    model_probabilities: ShadowModelProbabilities | None,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowPredictionsPublication:
    """Publie exclusivement predictions.csv, sans charger ni rappeler le modele.

    La primitive est volontairement interne. Le futur orchestrateur devra
    encore enchainer, sous les gardes d'activation et de manifeste, le calcul
    unique des probabilites avec cette publication, puis creer receipt.json et
    COMPLETED. Une erreur conserve tout fichier deja lie et interdit la
    reparation du slot.
    """
    predecessor_names = frozenset(
        {
            SOURCE_SNAPSHOT_FILENAME,
            CANDIDATE_LEDGER_FILENAME,
            FEATURES_FILENAME,
        }
    )
    _validate_source_snapshot_predecessors(
        reservation,
        activation_publication,
        project_directory=project_directory,
        expected_additional_filenames=predecessor_names,
    )
    with _hold_source_snapshot_stage_lock(
        reservation, project_directory=project_directory
    ):
        (
            source_bytes,
            candidate_bytes,
            features_bytes,
            feature_rows,
            target,
            cutoff,
        ) = _read_validated_prediction_predecessors(
            reservation,
            activation_publication,
            source_publication,
            candidate_publication,
            project_directory=project_directory,
        )
        (
            probabilities,
            state_before,
            state_after,
            runtime_versions,
            load_warning_count,
            state_warning_count,
            total_warning_count,
            predict_calls,
        ) = _validate_probabilities_for_publication(
            model_probabilities, feature_rows
        )
        protocol_sha256 = _require_sha256(
            reservation.reserved_marker["shadow_protocol_sha256"],
            field="RESERVED.shadow_protocol_sha256",
        )
        if protocol_sha256 != EXPECTED_SHADOW_PROTOCOL_SHA256:
            raise ShadowPredictionError(
                "Le protocole reserve n'est pas le protocole shadow v2 fige."
            )
        code_commit = _require_git_commit(
            reservation.reserved_marker["runtime_code_commit"],
            field="RESERVED.runtime_code_commit",
        )
        artifact_sha256 = (
            EXPECTED_MODEL_ARTIFACT_SHA256
            if model_probabilities is None
            else model_probabilities.artifact_sha256
        )
        earliest = feature_rows[0][8] if feature_rows else None

        # Relecture finale avant de figer l'heure inscrite dans chaque ligne.
        confirmed = _read_validated_prediction_predecessors(
            reservation,
            activation_publication,
            source_publication,
            candidate_publication,
            project_directory=project_directory,
        )
        if (
            confirmed[0] != source_bytes
            or confirmed[1] != candidate_bytes
            or confirmed[2] != features_bytes
            or confirmed[3] != feature_rows
            or confirmed[4:] != (target, cutoff)
        ):
            raise ShadowPredictionError(
                "Les predecesseurs ont change avant la construction des predictions."
            )

        issued_at = _format_utc_seconds(
            _utc_now(), field="issued_at_utc"
        )
        if issued_at < cutoff:
            raise ShadowPredictionError(
                "L'heure d'emission precede le cutoff d'information."
            )
        if earliest is not None:
            latest_safe_completion = (
                datetime.fromisoformat(earliest.replace("Z", "+00:00"))
                - timedelta(minutes=120)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            if issued_at > latest_safe_completion:
                raise ShadowPredictionError(
                    "Le delai minimal de deux heures ne peut plus etre respecte."
                )

        prediction_rows: list[list[object]] = []
        for feature_row, (p_away, p_home) in zip(
            feature_rows, probabilities
        ):
            prediction_rows.append(
                [
                    *feature_row[:9],
                    cutoff,
                    issued_at,
                    *feature_row[9:20],
                    _format_probability_float(p_home),
                    _format_probability_float(p_away),
                    EXPECTED_CALIBRATED_MODEL_VERSION,
                    artifact_sha256,
                    protocol_sha256,
                    code_commit,
                ]
            )
        predictions_bytes = _canonical_csv_bytes(
            _PREDICTIONS_COLUMNS, prediction_rows
        )

        # Construire les octets n'autorise aucune source a changer avant le lien.
        confirmed = _read_validated_prediction_predecessors(
            reservation,
            activation_publication,
            source_publication,
            candidate_publication,
            project_directory=project_directory,
        )
        if (
            confirmed[0] != source_bytes
            or confirmed[1] != candidate_bytes
            or confirmed[2] != features_bytes
            or confirmed[3] != feature_rows
        ):
            raise ShadowPredictionError(
                "Les predecesseurs ont change pendant la construction des predictions."
            )

        predictions_path = reservation.slot_path / PREDICTIONS_FILENAME
        try:
            predictions_sha256 = _publish_exclusive_verified(
                predictions_path, predictions_bytes
            )
            _validate_source_snapshot_predecessors(
                reservation,
                activation_publication,
                project_directory=project_directory,
                expected_additional_filenames=(
                    predecessor_names | {PREDICTIONS_FILENAME}
                ),
            )
            persisted = predictions_path.read_bytes()
            if (
                persisted != predictions_bytes
                or hashlib.sha256(persisted).hexdigest()
                != predictions_sha256
                or candidate_publication.candidate_ledger_path.read_bytes()
                != candidate_bytes
                or candidate_publication.features_path.read_bytes()
                != features_bytes
                or source_publication.snapshot_path.read_bytes()
                != source_bytes
            ):
                raise ShadowPredictionError(
                    "Les predictions ou leurs predecesseurs ont change apres publication."
                )
        except ShadowPublicationConflictError as error:
            raise ShadowPredictionSlotConsumedError(
                "Le jalon predictions est deja consomme."
            ) from error
        except OSError as error:
            raise ShadowPredictionError(
                "Publication incomplete; aucune reparation du slot n'est permise."
            ) from error

        relative_path = (
            SHADOW_RESULT_ROOT_RELATIVE_PATH / target / PREDICTIONS_FILENAME
        ).as_posix()
        return ShadowPredictionsPublication(
            slot_path=reservation.slot_path,
            predictions_path=predictions_path,
            predictions_relative_path=relative_path,
            predictions_sha256=predictions_sha256,
            predictions_size_bytes=len(predictions_bytes),
            prediction_row_count=len(prediction_rows),
            issued_at_utc=issued_at,
            earliest_predicted_scheduled_start_utc=earliest,
            model_version=EXPECTED_CALIBRATED_MODEL_VERSION,
            artifact_sha256=artifact_sha256,
            protocol_sha256=protocol_sha256,
            code_commit=code_commit,
            artifact_state_sha256_before=state_before,
            artifact_state_sha256_after=state_after,
            runtime_versions=runtime_versions,
            approved_deserialization_warning_count=load_warning_count,
            approved_state_serialization_warning_count=state_warning_count,
            approved_compatibility_warning_count=total_warning_count,
            predict_proba_calls=predict_calls,
        )


def _read_canonical_csv_rows(
    path: Path,
    columns: tuple[str, ...],
    *,
    field: str,
) -> tuple[bytes, tuple[tuple[str, ...], ...]]:
    """Relit un CSV officiel et exige exactement sa forme canonique."""
    mode = _lstat_mode(path)
    if not isinstance(mode, int) or not stat.S_ISREG(mode):
        raise ShadowPredictionError(
            f"{field} doit rester un fichier regulier non symbolique."
        )
    try:
        content = path.read_bytes()
        decoded = content.decode("utf-8")
        parsed = list(
            csv.reader(
                io.StringIO(decoded, newline=""),
                delimiter=",",
                quotechar='"',
                strict=True,
            )
        )
    except (OSError, UnicodeError, csv.Error) as error:
        raise ShadowPredictionError(
            f"{field} est illisible ou non canonique."
        ) from error
    if not parsed or tuple(parsed[0]) != columns:
        raise ShadowPredictionError(f"Schema CSV exact requis pour {field}.")
    rows = tuple(tuple(row) for row in parsed[1:])
    if any(len(row) != len(columns) for row in rows):
        raise ShadowPredictionError(
            f"Une ligne de {field} ne respecte pas le schema exact."
        )
    if _canonical_csv_bytes(columns, rows) != content:
        raise ShadowPredictionError(f"{field} n'utilise pas les octets canoniques.")
    return content, rows


def _read_receipt_activation_evidence(
    activation_publication: ShadowActivationReverificationPublication,
) -> tuple[dict[str, Any], bytes]:
    """Reconstruit la preuve HTTP depuis le deuxieme fichier persiste."""
    try:
        compressed = activation_publication.evidence_path.read_bytes()
        json_bytes = gzip.decompress(compressed)
        payload = json.loads(
            json_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (
        OSError,
        EOFError,
        UnicodeError,
        ValueError,
        ShadowPredictionError,
        RecursionError,
        zlib.error,
    ) as error:
        raise ShadowPredictionError(
            "La preuve distante d'activation est illisible ou ambigue."
        ) from error
    if type(payload) is not dict:
        raise ShadowPredictionError(
            "La preuve distante d'activation doit etre un objet JSON."
        )
    reconstructed = ShadowActivationReverificationEvidence(
        activation_introduction_commit=(
            activation_publication.activation_introduction_commit
        ),
        activation_remote_ref=activation_publication.activation_remote_ref,
        activation_remote_reverified_at_utc=(
            activation_publication.activation_remote_reverified_at_utc
        ),
        response_received_at_utc=activation_publication.response_received_at_utc,
        response_body_sha256=activation_publication.response_body_sha256,
        raw_evidence=payload,
        canonical_json_bytes=json_bytes,
        canonical_gzip_bytes=compressed,
        canonical_gzip_sha256=activation_publication.evidence_sha256,
    )
    _validate_activation_reverification_evidence(reconstructed)
    if (
        _canonical_json_file_bytes(payload) != json_bytes
        or _canonical_gzip_bytes(json_bytes) != compressed
        or hashlib.sha256(compressed).hexdigest()
        != activation_publication.evidence_sha256
    ):
        raise ShadowPredictionError(
            "La preuve distante d'activation n'est plus canonique."
        )
    return payload, compressed


def _read_validated_receipt_predecessors(
    reservation: ShadowPredictionSlotReservation,
    activation_publication: ShadowActivationReverificationPublication,
    source_publication: ShadowSourceSnapshotPublication,
    candidate_publication: ShadowCandidateFeaturesPublication,
    predictions_publication: ShadowPredictionsPublication,
    *,
    project_directory: Path,
    expected_additional_filenames: frozenset[str] = frozenset(),
) -> _ShadowReceiptPredecessors:
    """Reconstruit et relie les six fichiers qui precedent le recu."""
    if type(expected_additional_filenames) is not frozenset or not all(
        type(name) is str and name
        for name in expected_additional_filenames
    ):
        raise ShadowPredictionError(
            "Les fichiers posterieurs attendus doivent etre explicites."
        )
    expected_names = frozenset(
        {
            SOURCE_SNAPSHOT_FILENAME,
            CANDIDATE_LEDGER_FILENAME,
            FEATURES_FILENAME,
            PREDICTIONS_FILENAME,
        }
    ) | expected_additional_filenames
    _validate_source_snapshot_predecessors(
        reservation,
        activation_publication,
        project_directory=project_directory,
        expected_additional_filenames=expected_names,
    )
    target, cutoff = _validate_candidate_source_proof(
        reservation,
        source_publication,
        project_directory=project_directory,
    )
    if type(candidate_publication) is not ShadowCandidateFeaturesPublication:
        raise ShadowPredictionError(
            "Une preuve exacte des candidats et variables est requise."
        )
    if type(predictions_publication) is not ShadowPredictionsPublication:
        raise ShadowPredictionError(
            "Une preuve exacte de publication des predictions est requise."
        )

    expected_slot = project_directory.joinpath(
        *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts,
        target,
    )
    relative_root = SHADOW_RESULT_ROOT_RELATIVE_PATH / target
    candidate_path = expected_slot / CANDIDATE_LEDGER_FILENAME
    features_path = expected_slot / FEATURES_FILENAME
    predictions_path = expected_slot / PREDICTIONS_FILENAME
    if (
        reservation.slot_path != expected_slot
        or candidate_publication.slot_path != expected_slot
        or candidate_publication.candidate_ledger_path != candidate_path
        or candidate_publication.features_path != features_path
        or candidate_publication.candidate_ledger_relative_path
        != (relative_root / CANDIDATE_LEDGER_FILENAME).as_posix()
        or candidate_publication.features_relative_path
        != (relative_root / FEATURES_FILENAME).as_posix()
        or predictions_publication.slot_path != expected_slot
        or predictions_publication.predictions_path != predictions_path
        or predictions_publication.predictions_relative_path
        != (relative_root / PREDICTIONS_FILENAME).as_posix()
    ):
        raise ShadowPredictionError(
            "Les preuves du recu ne visent pas le meme slot canonique."
        )

    source_bytes, snapshot = _read_candidate_source_snapshot(
        reservation,
        source_publication,
        project_directory=project_directory,
    )
    ledger, features, exclusions = _build_candidate_feature_rows(
        snapshot,
        shadow_protocol_sha256=reservation.reserved_marker[
            "shadow_protocol_sha256"
        ],
    )
    expected_candidate_bytes = _canonical_csv_bytes(
        _CANDIDATE_LEDGER_COLUMNS,
        ledger,
    )
    expected_features_bytes = _canonical_csv_bytes(_FEATURES_COLUMNS, features)
    candidate_bytes, candidate_rows = _read_canonical_csv_rows(
        candidate_path,
        _CANDIDATE_LEDGER_COLUMNS,
        field="candidate_ledger.csv",
    )
    features_bytes, persisted_feature_rows = _read_canonical_csv_rows(
        features_path,
        _FEATURES_COLUMNS,
        field="features.csv",
    )
    typed_feature_rows = tuple(tuple(row) for row in features)
    expected_candidate_proof: dict[str, object] = {
        "candidate_ledger_sha256": hashlib.sha256(
            expected_candidate_bytes
        ).hexdigest(),
        "candidate_ledger_size_bytes": len(expected_candidate_bytes),
        "candidate_row_count": len(ledger),
        "features_sha256": hashlib.sha256(expected_features_bytes).hexdigest(),
        "features_size_bytes": len(expected_features_bytes),
        "feature_row_count": len(features),
        "eligible_game_count": len(features),
        "excluded_games_by_reason": tuple(exclusions.items()),
        "earliest_eligible_scheduled_start_utc": (
            features[0][8] if features else None
        ),
    }
    if (
        candidate_bytes != expected_candidate_bytes
        or features_bytes != expected_features_bytes
        or persisted_feature_rows
        != tuple(tuple(str(value) for value in row) for row in features)
    ):
        raise ShadowPredictionError(
            "Les candidats ou variables ne correspondent plus au snapshot."
        )
    for name, expected in expected_candidate_proof.items():
        actual = getattr(candidate_publication, name)
        if type(actual) is not type(expected) or actual != expected:
            raise ShadowPredictionError(
                f"La preuve candidats/variables diverge pour {name}."
            )

    predictions_bytes, prediction_rows = _read_canonical_csv_rows(
        predictions_path,
        _PREDICTIONS_COLUMNS,
        field="predictions.csv",
    )
    issued_at = _require_utc_timestamp(
        predictions_publication.issued_at_utc,
        field="predictions_publication.issued_at_utc",
    )
    protocol_sha256 = _require_sha256(
        reservation.reserved_marker["shadow_protocol_sha256"],
        field="RESERVED.shadow_protocol_sha256",
    )
    runtime_commit = _require_git_commit(
        reservation.reserved_marker["runtime_code_commit"],
        field="RESERVED.runtime_code_commit",
    )
    expected_earliest = features[0][8] if features else None
    exact_prediction_proof: dict[str, object] = {
        "predictions_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
        "predictions_size_bytes": len(predictions_bytes),
        "prediction_row_count": len(prediction_rows),
        "earliest_predicted_scheduled_start_utc": expected_earliest,
        "model_version": EXPECTED_CALIBRATED_MODEL_VERSION,
        "artifact_sha256": EXPECTED_MODEL_ARTIFACT_SHA256,
        "protocol_sha256": protocol_sha256,
        "code_commit": runtime_commit,
        "artifact_state_unchanged": True,
        "unexpected_warning_count": 0,
        "warning_policy_id": _MODEL_WARNING_POLICY_ID,
        "official_prediction_created": True,
        "receipt_created": False,
        "slot_completed": False,
    }
    for name, expected in exact_prediction_proof.items():
        actual = getattr(predictions_publication, name)
        if type(actual) is not type(expected) or actual != expected:
            raise ShadowPredictionError(
                f"La preuve des predictions diverge pour {name}."
            )
    expected_predict_calls = int(bool(features))
    if (
        type(predictions_publication.predict_proba_calls) is not int
        or predictions_publication.predict_proba_calls != expected_predict_calls
        or predictions_publication.prediction_row_count != len(features)
        or predictions_publication.issued_at_utc != issued_at
        or issued_at < cutoff
    ):
        raise ShadowPredictionError(
            "Les compteurs ou temps de la preuve predictions divergent."
        )
    warning_counts = (
        predictions_publication.approved_deserialization_warning_count,
        predictions_publication.approved_state_serialization_warning_count,
        predictions_publication.approved_compatibility_warning_count,
    )
    if (
        any(type(value) is not int or value < 0 for value in warning_counts)
        or warning_counts[2] != warning_counts[0] + warning_counts[1]
    ):
        raise ShadowPredictionError(
            "Les compteurs d'avertissements des predictions divergent."
        )
    if features:
        before = _require_sha256(
            predictions_publication.artifact_state_sha256_before,
            field="predictions.artifact_state_sha256_before",
        )
        after = _require_sha256(
            predictions_publication.artifact_state_sha256_after,
            field="predictions.artifact_state_sha256_after",
        )
        if before != after:
            raise ShadowPredictionError(
                "L'etat du modele a change dans la preuve predictions."
            )
    elif (
        predictions_publication.artifact_state_sha256_before is not None
        or predictions_publication.artifact_state_sha256_after is not None
        or predictions_publication.runtime_versions != ()
        or warning_counts != (0, 0, 0)
    ):
        raise ShadowPredictionError(
            "Un lot vide porte des traces interdites de chargement du modele."
        )

    prediction_ids: set[str] = set()
    for feature_row, prediction_row in zip(typed_feature_rows, prediction_rows):
        expected_left = tuple(str(value) for value in feature_row[:9])
        expected_features = tuple(str(value) for value in feature_row[9:20])
        if (
            prediction_row[:9] != expected_left
            or prediction_row[9] != cutoff
            or prediction_row[10] != issued_at
            or prediction_row[11:22] != expected_features
            or prediction_row[24] != EXPECTED_CALIBRATED_MODEL_VERSION
            or prediction_row[25] != EXPECTED_MODEL_ARTIFACT_SHA256
            or prediction_row[26] != protocol_sha256
            or prediction_row[27] != runtime_commit
        ):
            raise ShadowPredictionError(
                "Une ligne de predictions ne correspond pas a features.csv."
            )
        try:
            p_home = float(prediction_row[22])
            p_away = float(prediction_row[23])
        except ValueError as error:
            raise ShadowPredictionError(
                "Une probabilite persistee n'est pas un flottant canonique."
            ) from error
        if (
            _format_probability_float(p_home) != prediction_row[22]
            or _format_probability_float(p_away) != prediction_row[23]
            or p_away != 1.0 - p_home
            or abs(p_away + p_home - 1.0) > 1e-12
            or prediction_row[0] in prediction_ids
        ):
            raise ShadowPredictionError(
                "Une paire de probabilites persistee est invalide."
            )
        prediction_ids.add(prediction_row[0])

    raw_activation, activation_bytes = _read_receipt_activation_evidence(
        activation_publication
    )
    return _ShadowReceiptPredecessors(
        activation_evidence=raw_activation,
        activation_bytes=activation_bytes,
        source_snapshot=snapshot,
        source_bytes=source_bytes,
        candidate_bytes=candidate_bytes,
        features_bytes=features_bytes,
        predictions_bytes=predictions_bytes,
        candidate_rows=candidate_rows,
        feature_rows=typed_feature_rows,
        prediction_rows=prediction_rows,
        exclusions=tuple(exclusions.items()),
        target_official_date=target,
        information_cutoff_utc=cutoff,
    )


def _validate_receipt_execution_context(
    context: ShadowReceiptExecutionContext,
    *,
    target_official_date: str,
    reserved_at_utc: str,
    activation_publication: ShadowActivationReverificationPublication,
    predictions_publication: ShadowPredictionsPublication,
) -> tuple[str, str, str, str, str, dict[str, str]]:
    """Valide la preuve interne que le futur preflight devra produire."""
    if type(context) is not ShadowReceiptExecutionContext:
        raise ShadowPredictionError(
            "Un contexte d'execution du recu exact est requis."
        )
    for name in (
        "execution_manifest_verified",
        "activation_verified",
        "execution_ready",
    ):
        if getattr(context, name) is not True:
            raise ShadowPredictionError(
                f"Le contexte d'execution diverge pour {name}."
            )
    started_at = _require_utc_timestamp(
        context.started_at_utc,
        field="context.started_at_utc",
    )
    manifest_commit = _require_git_commit(
        context.execution_manifest_introduction_commit,
        field="context.execution_manifest_introduction_commit",
    )
    activation_sha256 = _require_sha256(
        context.activation_sha256,
        field="context.activation_sha256",
    )
    activation_verified_at = _require_utc_timestamp(
        context.activation_verified_at_utc,
        field="context.activation_verified_at_utc",
    )
    minimum_date = _require_date_string(
        context.minimum_target_official_date,
        field="context.minimum_target_official_date",
    )
    service_sha256 = _require_sha256(
        context.shadow_service_module_sha256,
        field="context.shadow_service_module_sha256",
    )
    runtime_versions = context.runtime_versions
    if (
        type(runtime_versions) is not tuple
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(value) is not str for value in pair)
            for pair in runtime_versions
        )
        or tuple(sorted(dict(runtime_versions).items())) != runtime_versions
        or frozenset(dict(runtime_versions)) != _MODEL_RUNTIME_KEYS
        or any(not value or value != value.strip() for _, value in runtime_versions)
    ):
        raise ShadowPredictionError(
            "Les versions runtime du contexte ne sont pas canoniques."
        )
    if (
        activation_verified_at > started_at
        or started_at > activation_publication.response_received_at_utc
        or activation_publication.response_received_at_utc > reserved_at_utc
    ):
        raise ShadowPredictionError(
            "L'ordre temporel activation, debut et reservation est invalide."
        )
    if (
        minimum_date < EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON.isoformat()
        or minimum_date < activation_verified_at[:10]
        or target_official_date < minimum_date
    ):
        raise ShadowPredictionError(
            "La date cible precede la date minimale active du contexte."
        )
    if predictions_publication.prediction_row_count > 0:
        if predictions_publication.runtime_versions != runtime_versions:
            raise ShadowPredictionError(
                "Les versions runtime du modele et du contexte divergent."
            )
    elif predictions_publication.runtime_versions != ():
        raise ShadowPredictionError(
            "Un lot vide ne doit pas porter de versions issues du modele."
        )
    return (
        started_at,
        manifest_commit,
        activation_sha256,
        activation_verified_at,
        service_sha256,
        dict(runtime_versions),
    )


def _build_and_publish_shadow_receipt(
    reservation: ShadowPredictionSlotReservation,
    activation_publication: ShadowActivationReverificationPublication,
    source_publication: ShadowSourceSnapshotPublication,
    candidate_publication: ShadowCandidateFeaturesPublication,
    predictions_publication: ShadowPredictionsPublication,
    execution_context: ShadowReceiptExecutionContext,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowReceiptPublication:
    """Publie exclusivement receipt.json sans creer le marqueur COMPLETED.

    Cette primitive interne relit et reconstruit tous les fichiers de donnees.
    Le contexte d'execution doit provenir du futur preflight fige; aucune
    activation, lecture Git, MLB, SQLite ou modele n'est effectuee ici.
    """
    predecessor_names = frozenset(
        {
            SOURCE_SNAPSHOT_FILENAME,
            CANDIDATE_LEDGER_FILENAME,
            FEATURES_FILENAME,
            PREDICTIONS_FILENAME,
        }
    )
    _validate_source_snapshot_predecessors(
        reservation,
        activation_publication,
        project_directory=project_directory,
        expected_additional_filenames=predecessor_names,
    )
    with _hold_source_snapshot_stage_lock(
        reservation,
        project_directory=project_directory,
    ):
        predecessors = _read_validated_receipt_predecessors(
            reservation,
            activation_publication,
            source_publication,
            candidate_publication,
            predictions_publication,
            project_directory=project_directory,
        )
        reserved = reservation.reserved_marker
        reserved_at = _require_utc_timestamp(
            reserved["reserved_at_utc"],
            field="RESERVED.reserved_at_utc",
        )
        (
            started_at,
            manifest_introduction_commit,
            activation_sha256,
            activation_verified_at,
            service_sha256,
            runtime_versions,
        ) = _validate_receipt_execution_context(
            execution_context,
            target_official_date=predecessors.target_official_date,
            reserved_at_utc=reserved_at,
            activation_publication=activation_publication,
            predictions_publication=predictions_publication,
        )
        raw_activation = predecessors.activation_evidence
        snapshot = predecessors.source_snapshot
        ingestion = snapshot["schedule_ingestion"]
        sqlite_snapshot = snapshot["sqlite_snapshot"]
        schedule_observed_at = _require_utc_timestamp(
            ingestion["mlb_http_response_received_at_utc"],
            field="schedule_ingestion.mlb_http_response_received_at_utc",
        )
        mlb_http_date = _require_utc_timestamp(
            ingestion["mlb_http_date_utc"],
            field="schedule_ingestion.mlb_http_date_utc",
        )
        cutoff = predecessors.information_cutoff_utc
        issued_at = _require_utc_timestamp(
            predictions_publication.issued_at_utc,
            field="predictions.issued_at_utc",
        )

        # Tous les fichiers de donnees ont ete relus avant cet instant unique.
        receipt_finalized_at = _format_utc_seconds(
            _utc_now(),
            field="receipt_finalized_at_utc",
        )
        if not (
            started_at
            <= activation_publication.response_received_at_utc
            <= reserved_at
            <= schedule_observed_at
            <= cutoff
            <= issued_at
            <= receipt_finalized_at
        ):
            raise ShadowPredictionError(
                "L'ordre temporel complet du recu est invalide."
            )
        earliest = predictions_publication.earliest_predicted_scheduled_start_utc
        if earliest is not None:
            latest_safe_completion = (
                datetime.fromisoformat(earliest.replace("Z", "+00:00"))
                - timedelta(minutes=120)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            if receipt_finalized_at > latest_safe_completion:
                raise ShadowPredictionError(
                    "Le recu est trop tardif pour permettre une fin conforme."
                )

        to_datetime = lambda value: datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
        clock_skew_seconds = int(
            (to_datetime(schedule_observed_at) - to_datetime(mlb_http_date))
            .total_seconds()
        )
        schedule_age_seconds = int(
            (to_datetime(cutoff) - to_datetime(schedule_observed_at))
            .total_seconds()
        )
        excluded = dict(predecessors.exclusions)
        schedule_games = len(snapshot["target_schedule"])
        eligible_games = len(predecessors.feature_rows)
        predicted_games = len(predecessors.prediction_rows)
        batch_status = (
            "COMPLETED_WITH_PREDICTIONS"
            if predicted_games > 0
            else "COMPLETED_NO_ELIGIBLE_GAMES"
        )
        receipt: dict[str, Any] = {
            "receipt_schema_version": 1,
            "batch": {
                "batch_id": reservation.batch_id,
                "slot_key": reservation.slot_key,
                "target_official_date": predecessors.target_official_date,
                "status": batch_status,
                "earliest_predicted_scheduled_start_utc": earliest,
            },
            "activation": {
                "execution_manifest_introduction_commit": (
                    manifest_introduction_commit
                ),
                "activation_introduction_commit": (
                    activation_publication.activation_introduction_commit
                ),
                "activation_path": ACTIVATION_RELATIVE_PATH.as_posix(),
                "activation_sha256": activation_sha256,
                "activation_verified_at_utc": activation_verified_at,
                "activation_remote_reverified_at_utc": (
                    activation_publication.activation_remote_reverified_at_utc
                ),
                "activation_remote_ref": GITHUB_REMOTE_REF,
                "activation_remote_reverification_query_url": (
                    raw_activation["request_url"]
                ),
                "activation_remote_reverification_effective_url": (
                    raw_activation["effective_url"]
                ),
                "activation_remote_reverification_status_code": (
                    raw_activation["response_status_code"]
                ),
                "activation_remote_reverification_redirect_count": (
                    raw_activation["response_redirect_count"]
                ),
                "activation_remote_reverification_response_received_at_utc": (
                    raw_activation["response_received_at_utc"]
                ),
                "activation_remote_reverification_response_body_sha256": (
                    raw_activation["response_body_sha256"]
                ),
                "activation_remote_reverification_evidence_path": (
                    activation_publication.evidence_relative_path
                ),
                "activation_remote_reverification_evidence_sha256": (
                    activation_publication.evidence_sha256
                ),
                "minimum_target_official_date": (
                    execution_context.minimum_target_official_date
                ),
            },
            "times": {
                "started_at_utc": started_at,
                "reserved_at_utc": reserved_at,
                "schedule_observed_at_utc": schedule_observed_at,
                "information_cutoff_utc": cutoff,
                "issued_at_utc": issued_at,
                "receipt_finalized_at_utc": receipt_finalized_at,
                "mlb_http_date_utc": mlb_http_date,
                "mlb_http_response_received_at_utc": schedule_observed_at,
                "clock_skew_seconds": clock_skew_seconds,
                "schedule_age_seconds": schedule_age_seconds,
            },
            "schedule_http_response": {
                "effective_url": ingestion["response_effective_url"],
                "status_code": ingestion["response_status_code"],
                "redirect_count": ingestion["response_redirect_count"],
                "date_header_raw": ingestion["mlb_http_date_header_raw"],
                "date_header_utc": mlb_http_date,
                "received_at_utc": schedule_observed_at,
                "body_sha256": ingestion["response_body_sha256"],
            },
            "lineage": {
                "runtime_code_commit": reserved["runtime_code_commit"],
                "shadow_service_module_sha256": service_sha256,
                "shadow_protocol_sha256": reserved["shadow_protocol_sha256"],
                "execution_manifest_sha256": reserved[
                    "execution_manifest_sha256"
                ],
                "model_artifact_sha256": EXPECTED_MODEL_ARTIFACT_SHA256,
                "artifact_manifest_sha256": (
                    EXPECTED_ARTIFACT_MANIFEST_SHA256
                ),
                "model_protocol_sha256": EXPECTED_MODEL_PROTOCOL_SHA256,
                "evaluation_protocol_sha256": (
                    EXPECTED_EVALUATION_PROTOCOL_SHA256
                ),
                "evaluation_report_sha256": EXPECTED_EVALUATION_REPORT_SHA256,
                "evaluation_results_commit": EXPECTED_EVALUATION_RESULTS_COMMIT,
            },
            "source": {
                "sqlite_snapshot_sha256": sqlite_snapshot["sha256"],
                "sqlite_snapshot_size_bytes": sqlite_snapshot["size_bytes"],
                "source_snapshot_path": source_publication.snapshot_relative_path,
                "source_snapshot_sha256": source_publication.snapshot_sha256,
                "schedule_ingestion_run_id": ingestion["run_id"],
                "schedule_source": ingestion["source"],
                "schedule_requested_start_date": ingestion[
                    "requested_start_date"
                ],
                "schedule_requested_end_date": ingestion[
                    "requested_end_date"
                ],
                "schedule_game_types": ingestion["game_types"],
                "schedule_request_parameters_json": ingestion[
                    "request_parameters_json"
                ],
                "schedule_ingestion_completed_at_utc": ingestion[
                    "completed_at_utc"
                ],
                "schedule_raw_archive_path": ingestion["raw_archive_path"],
                "schedule_raw_archive_sha256": ingestion[
                    "raw_archive_sha256"
                ],
            },
            "counts": {
                "schedule_games": schedule_games,
                "eligible_games": eligible_games,
                "predicted_games": predicted_games,
                "excluded_games_by_reason": excluded,
            },
            "output_hashes": {
                "activation_reverification_evidence_sha256": (
                    activation_publication.evidence_sha256
                ),
                "candidate_ledger_sha256": (
                    candidate_publication.candidate_ledger_sha256
                ),
                "features_sha256": candidate_publication.features_sha256,
                "predictions_sha256": predictions_publication.predictions_sha256,
            },
            "model_invariants": {
                "fit_calls": 0,
                "partial_fit_calls": 0,
                "recalibration_calls": 0,
                "threshold_tuning_calls": 0,
                "feature_selection_calls": 0,
                "predict_proba_calls": predictions_publication.predict_proba_calls,
                "artifact_state_sha256_before": (
                    predictions_publication.artifact_state_sha256_before
                ),
                "artifact_state_sha256_after": (
                    predictions_publication.artifact_state_sha256_after
                ),
                "artifact_state_unchanged": True,
                "warning_policy_id": _MODEL_WARNING_POLICY_ID,
                "approved_compatibility_warning_count": (
                    predictions_publication.approved_compatibility_warning_count
                ),
                "unexpected_warning_count": 0,
            },
            "negative_attestations": {
                name: True
                for name in _RECEIPT_SECTION_KEYS["negative_attestations"]
            },
            "runtime_versions": runtime_versions,
        }
        if (
            not _has_exact_keys(receipt, _RECEIPT_TOP_LEVEL_KEYS)
            or any(
                not _has_exact_keys(receipt[name], keys)
                for name, keys in _RECEIPT_SECTION_KEYS.items()
            )
            or schedule_games != eligible_games + sum(excluded.values())
            or predicted_games != eligible_games
            or not _valid_receipt(
                receipt,
                reserved=reserved,
                batch_id=reservation.batch_id,
                slot_key=reservation.slot_key,
                target_official_date=predecessors.target_official_date,
                shadow_protocol_sha256=reserved["shadow_protocol_sha256"],
                execution_manifest_sha256=reserved[
                    "execution_manifest_sha256"
                ],
                model_artifact_sha256=EXPECTED_MODEL_ARTIFACT_SHA256,
            )
        ):
            raise ShadowPredictionError("Le recu construit viole le contrat v2.")
        receipt_bytes = _canonical_json_file_bytes(receipt)

        # Une derniere reconstruction avant le lien exclut toute derive source.
        confirmed = _read_validated_receipt_predecessors(
            reservation,
            activation_publication,
            source_publication,
            candidate_publication,
            predictions_publication,
            project_directory=project_directory,
        )
        if confirmed != predecessors:
            raise ShadowPredictionError(
                "Les predecesseurs ont change pendant la construction du recu."
            )

        receipt_path = reservation.slot_path / RECEIPT_FILENAME
        try:
            receipt_sha256 = _publish_exclusive_verified(
                receipt_path,
                receipt_bytes,
            )
            _validate_source_snapshot_predecessors(
                reservation,
                activation_publication,
                project_directory=project_directory,
                expected_additional_filenames=(
                    predecessor_names | {RECEIPT_FILENAME}
                ),
            )
            persisted_receipt = _read_canonical_json_object(receipt_path)
            if (
                persisted_receipt is None
                or persisted_receipt[0] != receipt
                or persisted_receipt[1] != receipt_bytes
                or hashlib.sha256(persisted_receipt[1]).hexdigest()
                != receipt_sha256
                or source_publication.snapshot_path.read_bytes()
                != predecessors.source_bytes
                or candidate_publication.candidate_ledger_path.read_bytes()
                != predecessors.candidate_bytes
                or candidate_publication.features_path.read_bytes()
                != predecessors.features_bytes
                or predictions_publication.predictions_path.read_bytes()
                != predecessors.predictions_bytes
                or activation_publication.evidence_path.read_bytes()
                != predecessors.activation_bytes
            ):
                raise ShadowPredictionError(
                    "Le recu ou ses predecesseurs ont change apres publication."
                )
        except ShadowPublicationConflictError as error:
            raise ShadowPredictionSlotConsumedError(
                "Le jalon receipt.json est deja consomme."
            ) from error
        except OSError as error:
            raise ShadowPredictionError(
                "Publication du recu incomplete; aucune reparation n'est permise."
            ) from error

        relative_path = (
            SHADOW_RESULT_ROOT_RELATIVE_PATH
            / predecessors.target_official_date
            / RECEIPT_FILENAME
        ).as_posix()
        return ShadowReceiptPublication(
            slot_path=reservation.slot_path,
            receipt_path=receipt_path,
            receipt_relative_path=relative_path,
            receipt_sha256=receipt_sha256,
            receipt_size_bytes=len(receipt_bytes),
            receipt_finalized_at_utc=receipt_finalized_at,
            batch_status=batch_status,
            schedule_game_count=schedule_games,
            eligible_game_count=eligible_games,
            predicted_game_count=predicted_games,
            earliest_predicted_scheduled_start_utc=earliest,
            execution_manifest_sha256=reserved["execution_manifest_sha256"],
            protocol_sha256=reserved["shadow_protocol_sha256"],
            artifact_sha256=EXPECTED_MODEL_ARTIFACT_SHA256,
            official_prediction_created=predicted_games > 0,
        )


def _read_validated_completion_predecessors(
    reservation: ShadowPredictionSlotReservation,
    activation_publication: ShadowActivationReverificationPublication,
    source_publication: ShadowSourceSnapshotPublication,
    candidate_publication: ShadowCandidateFeaturesPublication,
    predictions_publication: ShadowPredictionsPublication,
    receipt_publication: ShadowReceiptPublication,
    *,
    project_directory: Path,
    expected_additional_filenames: frozenset[str] = frozenset(),
) -> _ShadowCompletionPredecessors:
    """Relit les sept preuves et lie integralement le recu a leur contenu."""
    if type(receipt_publication) is not ShadowReceiptPublication:
        raise ShadowPredictionError(
            "Une preuve exacte de publication du recu est requise."
        )
    if type(expected_additional_filenames) is not frozenset or not all(
        type(name) is str and name
        for name in expected_additional_filenames
    ):
        raise ShadowPredictionError(
            "Les fichiers terminaux attendus doivent etre explicites."
        )

    receipt_predecessors = _read_validated_receipt_predecessors(
        reservation,
        activation_publication,
        source_publication,
        candidate_publication,
        predictions_publication,
        project_directory=project_directory,
        expected_additional_filenames=(
            frozenset({RECEIPT_FILENAME})
            | expected_additional_filenames
        ),
    )
    target = receipt_predecessors.target_official_date
    expected_slot = project_directory.joinpath(
        *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts,
        target,
    )
    receipt_path = expected_slot / RECEIPT_FILENAME
    receipt_relative_path = (
        SHADOW_RESULT_ROOT_RELATIVE_PATH / target / RECEIPT_FILENAME
    ).as_posix()
    if (
        reservation.slot_path != expected_slot
        or receipt_publication.slot_path != expected_slot
        or receipt_publication.receipt_path != receipt_path
        or receipt_publication.receipt_relative_path != receipt_relative_path
        or receipt_publication.receipt_created is not True
        or receipt_publication.slot_completed is not False
    ):
        raise ShadowPredictionError(
            "La preuve du recu ne vise pas le septieme fichier canonique."
        )

    persisted_receipt = _read_canonical_json_object(receipt_path)
    if persisted_receipt is None:
        raise ShadowPredictionError(
            "Le recu persiste n'est pas un objet JSON canonique."
        )
    receipt, receipt_bytes = persisted_receipt
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if (
        type(receipt_publication.receipt_size_bytes) is not int
        or receipt_publication.receipt_size_bytes != len(receipt_bytes)
        or _require_sha256(
            receipt_publication.receipt_sha256,
            field="receipt_publication.receipt_sha256",
        )
        != receipt_sha256
    ):
        raise ShadowPredictionError(
            "La taille ou l'empreinte du recu ne correspond pas a sa preuve."
        )

    reserved = reservation.reserved_marker
    protocol_sha256 = _require_sha256(
        reserved["shadow_protocol_sha256"],
        field="RESERVED.shadow_protocol_sha256",
    )
    manifest_sha256 = _require_sha256(
        reserved["execution_manifest_sha256"],
        field="RESERVED.execution_manifest_sha256",
    )
    if not _valid_receipt(
        receipt,
        reserved=reserved,
        batch_id=reservation.batch_id,
        slot_key=reservation.slot_key,
        target_official_date=target,
        shadow_protocol_sha256=protocol_sha256,
        execution_manifest_sha256=manifest_sha256,
        model_artifact_sha256=EXPECTED_MODEL_ARTIFACT_SHA256,
    ):
        raise ShadowPredictionError("Le recu persiste viole le contrat v2.")

    batch = receipt["batch"]
    activation = receipt["activation"]
    times = receipt["times"]
    schedule_http = receipt["schedule_http_response"]
    lineage = receipt["lineage"]
    source = receipt["source"]
    counts = receipt["counts"]
    output_hashes = receipt["output_hashes"]
    model_invariants = receipt["model_invariants"]
    runtime_versions = receipt["runtime_versions"]
    assert isinstance(batch, dict)
    assert isinstance(activation, dict)
    assert isinstance(times, dict)
    assert isinstance(schedule_http, dict)
    assert isinstance(lineage, dict)
    assert isinstance(source, dict)
    assert isinstance(counts, dict)
    assert isinstance(output_hashes, dict)
    assert isinstance(model_invariants, dict)
    assert isinstance(runtime_versions, dict)

    snapshot = receipt_predecessors.source_snapshot
    ingestion = snapshot["schedule_ingestion"]
    sqlite_snapshot = snapshot["sqlite_snapshot"]
    raw_activation = receipt_predecessors.activation_evidence
    exclusions = dict(receipt_predecessors.exclusions)
    schedule_game_count = len(snapshot["target_schedule"])
    eligible_game_count = len(receipt_predecessors.feature_rows)
    predicted_game_count = len(receipt_predecessors.prediction_rows)
    batch_status = (
        "COMPLETED_WITH_PREDICTIONS"
        if predicted_game_count > 0
        else "COMPLETED_NO_ELIGIBLE_GAMES"
    )
    earliest = predictions_publication.earliest_predicted_scheduled_start_utc

    expected_publication_values: dict[str, object] = {
        "receipt_finalized_at_utc": times["receipt_finalized_at_utc"],
        "batch_status": batch_status,
        "schedule_game_count": schedule_game_count,
        "eligible_game_count": eligible_game_count,
        "predicted_game_count": predicted_game_count,
        "earliest_predicted_scheduled_start_utc": earliest,
        "execution_manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "artifact_sha256": EXPECTED_MODEL_ARTIFACT_SHA256,
        "official_prediction_created": predicted_game_count > 0,
    }
    for name, expected in expected_publication_values.items():
        actual = getattr(receipt_publication, name)
        if type(actual) is not type(expected) or actual != expected:
            raise ShadowPredictionError(
                f"La preuve du recu diverge pour {name}."
            )

    expected_batch = {
        "batch_id": reservation.batch_id,
        "slot_key": reservation.slot_key,
        "target_official_date": target,
        "status": batch_status,
        "earliest_predicted_scheduled_start_utc": earliest,
    }
    expected_counts = {
        "schedule_games": schedule_game_count,
        "eligible_games": eligible_game_count,
        "predicted_games": predicted_game_count,
        "excluded_games_by_reason": exclusions,
    }
    expected_output_hashes = {
        "activation_reverification_evidence_sha256": (
            activation_publication.evidence_sha256
        ),
        "candidate_ledger_sha256": (
            candidate_publication.candidate_ledger_sha256
        ),
        "features_sha256": candidate_publication.features_sha256,
        "predictions_sha256": predictions_publication.predictions_sha256,
    }
    expected_lineage = {
        "runtime_code_commit": reserved["runtime_code_commit"],
        "shadow_protocol_sha256": protocol_sha256,
        "execution_manifest_sha256": manifest_sha256,
        "model_artifact_sha256": EXPECTED_MODEL_ARTIFACT_SHA256,
        "artifact_manifest_sha256": EXPECTED_ARTIFACT_MANIFEST_SHA256,
        "model_protocol_sha256": EXPECTED_MODEL_PROTOCOL_SHA256,
        "evaluation_protocol_sha256": EXPECTED_EVALUATION_PROTOCOL_SHA256,
        "evaluation_report_sha256": EXPECTED_EVALUATION_REPORT_SHA256,
        "evaluation_results_commit": EXPECTED_EVALUATION_RESULTS_COMMIT,
    }
    if (
        batch != expected_batch
        or counts != expected_counts
        or output_hashes != expected_output_hashes
        or any(
            type(lineage.get(name)) is not type(expected)
            or lineage.get(name) != expected
            for name, expected in expected_lineage.items()
        )
        or not _has_exact_keys(
            runtime_versions,
            _RECEIPT_SECTION_KEYS["runtime_versions"],
        )
        or any(
            type(value) is not str or not value or value != value.strip()
            for value in runtime_versions.values()
        )
    ):
        raise ShadowPredictionError(
            "L'identite, les compteurs ou les empreintes du recu divergent."
        )
    if predicted_game_count > 0 and runtime_versions != dict(
        predictions_publication.runtime_versions
    ):
        raise ShadowPredictionError(
            "Les versions runtime du recu divergent des predictions."
        )

    execution_manifest_commit = _require_git_commit(
        activation["execution_manifest_introduction_commit"],
        field="receipt.activation.execution_manifest_introduction_commit",
    )
    activation_sha256 = _require_sha256(
        activation["activation_sha256"],
        field="receipt.activation.activation_sha256",
    )
    activation_verified_at = _require_utc_timestamp(
        activation["activation_verified_at_utc"],
        field="receipt.activation.activation_verified_at_utc",
    )
    minimum_target_date = _require_date_string(
        activation["minimum_target_official_date"],
        field="receipt.activation.minimum_target_official_date",
    )
    service_sha256 = _require_sha256(
        lineage["shadow_service_module_sha256"],
        field="receipt.lineage.shadow_service_module_sha256",
    )
    expected_activation = {
        "execution_manifest_introduction_commit": execution_manifest_commit,
        "activation_introduction_commit": (
            activation_publication.activation_introduction_commit
        ),
        "activation_path": ACTIVATION_RELATIVE_PATH.as_posix(),
        "activation_sha256": activation_sha256,
        "activation_verified_at_utc": activation_verified_at,
        "activation_remote_reverified_at_utc": (
            activation_publication.activation_remote_reverified_at_utc
        ),
        "activation_remote_ref": GITHUB_REMOTE_REF,
        "activation_remote_reverification_query_url": raw_activation[
            "request_url"
        ],
        "activation_remote_reverification_effective_url": raw_activation[
            "effective_url"
        ],
        "activation_remote_reverification_status_code": raw_activation[
            "response_status_code"
        ],
        "activation_remote_reverification_redirect_count": raw_activation[
            "response_redirect_count"
        ],
        "activation_remote_reverification_response_received_at_utc": (
            raw_activation["response_received_at_utc"]
        ),
        "activation_remote_reverification_response_body_sha256": (
            raw_activation["response_body_sha256"]
        ),
        "activation_remote_reverification_evidence_path": (
            activation_publication.evidence_relative_path
        ),
        "activation_remote_reverification_evidence_sha256": (
            activation_publication.evidence_sha256
        ),
        "minimum_target_official_date": minimum_target_date,
    }
    if activation != expected_activation:
        raise ShadowPredictionError(
            "La section activation du recu diverge de la preuve distante."
        )

    schedule_observed_at = _require_utc_timestamp(
        ingestion["mlb_http_response_received_at_utc"],
        field="schedule_ingestion.mlb_http_response_received_at_utc",
    )
    mlb_http_date = _require_utc_timestamp(
        ingestion["mlb_http_date_utc"],
        field="schedule_ingestion.mlb_http_date_utc",
    )
    cutoff = receipt_predecessors.information_cutoff_utc
    issued_at = _require_utc_timestamp(
        predictions_publication.issued_at_utc,
        field="predictions_publication.issued_at_utc",
    )
    receipt_finalized_at = _require_utc_timestamp(
        times["receipt_finalized_at_utc"],
        field="receipt.times.receipt_finalized_at_utc",
    )
    started_at = _require_utc_timestamp(
        times["started_at_utc"],
        field="receipt.times.started_at_utc",
    )
    reserved_at = _require_utc_timestamp(
        reserved["reserved_at_utc"],
        field="RESERVED.reserved_at_utc",
    )
    to_datetime = lambda value: datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )
    expected_times = {
        "started_at_utc": started_at,
        "reserved_at_utc": reserved_at,
        "schedule_observed_at_utc": schedule_observed_at,
        "information_cutoff_utc": cutoff,
        "issued_at_utc": issued_at,
        "receipt_finalized_at_utc": receipt_finalized_at,
        "mlb_http_date_utc": mlb_http_date,
        "mlb_http_response_received_at_utc": schedule_observed_at,
        "clock_skew_seconds": int(
            (to_datetime(schedule_observed_at) - to_datetime(mlb_http_date))
            .total_seconds()
        ),
        "schedule_age_seconds": int(
            (to_datetime(cutoff) - to_datetime(schedule_observed_at))
            .total_seconds()
        ),
    }
    if (
        times != expected_times
        or not (
            started_at
            <= activation_publication.response_received_at_utc
            <= reserved_at
            <= schedule_observed_at
            <= cutoff
            <= issued_at
            <= receipt_finalized_at
        )
        or activation_verified_at > started_at
        or minimum_target_date
        < EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON.isoformat()
        or minimum_target_date < activation_verified_at[:10]
        or target < minimum_target_date
    ):
        raise ShadowPredictionError(
            "Les temps ou la date minimale du recu divergent."
        )

    expected_schedule_http = {
        "effective_url": ingestion["response_effective_url"],
        "status_code": ingestion["response_status_code"],
        "redirect_count": ingestion["response_redirect_count"],
        "date_header_raw": ingestion["mlb_http_date_header_raw"],
        "date_header_utc": mlb_http_date,
        "received_at_utc": schedule_observed_at,
        "body_sha256": ingestion["response_body_sha256"],
    }
    expected_source = {
        "sqlite_snapshot_sha256": sqlite_snapshot["sha256"],
        "sqlite_snapshot_size_bytes": sqlite_snapshot["size_bytes"],
        "source_snapshot_path": source_publication.snapshot_relative_path,
        "source_snapshot_sha256": source_publication.snapshot_sha256,
        "schedule_ingestion_run_id": ingestion["run_id"],
        "schedule_source": ingestion["source"],
        "schedule_requested_start_date": ingestion["requested_start_date"],
        "schedule_requested_end_date": ingestion["requested_end_date"],
        "schedule_game_types": ingestion["game_types"],
        "schedule_request_parameters_json": ingestion[
            "request_parameters_json"
        ],
        "schedule_ingestion_completed_at_utc": ingestion[
            "completed_at_utc"
        ],
        "schedule_raw_archive_path": ingestion["raw_archive_path"],
        "schedule_raw_archive_sha256": ingestion["raw_archive_sha256"],
    }
    expected_model_invariants = {
        "fit_calls": 0,
        "partial_fit_calls": 0,
        "recalibration_calls": 0,
        "threshold_tuning_calls": 0,
        "feature_selection_calls": 0,
        "predict_proba_calls": predictions_publication.predict_proba_calls,
        "artifact_state_sha256_before": (
            predictions_publication.artifact_state_sha256_before
        ),
        "artifact_state_sha256_after": (
            predictions_publication.artifact_state_sha256_after
        ),
        "artifact_state_unchanged": True,
        "warning_policy_id": _MODEL_WARNING_POLICY_ID,
        "approved_compatibility_warning_count": (
            predictions_publication.approved_compatibility_warning_count
        ),
        "unexpected_warning_count": 0,
    }
    if (
        schedule_http != expected_schedule_http
        or source != expected_source
        or model_invariants != expected_model_invariants
        or any(
            value is not True
            for value in receipt["negative_attestations"].values()
        )
        or not service_sha256
    ):
        raise ShadowPredictionError(
            "Le recu ne correspond plus aux preuves source ou modele."
        )

    return _ShadowCompletionPredecessors(
        receipt=receipt,
        receipt_bytes=receipt_bytes,
        receipt_finalized_at_utc=receipt_finalized_at,
        batch_status=batch_status,
        earliest_predicted_scheduled_start_utc=earliest,
        receipt_predecessors=receipt_predecessors,
    )


def _complete_shadow_prediction_slot(
    reservation: ShadowPredictionSlotReservation,
    activation_publication: ShadowActivationReverificationPublication,
    source_publication: ShadowSourceSnapshotPublication,
    candidate_publication: ShadowCandidateFeaturesPublication,
    predictions_publication: ShadowPredictionsPublication,
    receipt_publication: ShadowReceiptPublication,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowCompletionPublication:
    """Publie exclusivement COMPLETED apres relecture des sept preuves."""
    predecessor_names = frozenset(
        {
            SOURCE_SNAPSHOT_FILENAME,
            CANDIDATE_LEDGER_FILENAME,
            FEATURES_FILENAME,
            PREDICTIONS_FILENAME,
            RECEIPT_FILENAME,
        }
    )
    _validate_source_snapshot_predecessors(
        reservation,
        activation_publication,
        project_directory=project_directory,
        expected_additional_filenames=predecessor_names,
    )
    with _hold_source_snapshot_stage_lock(
        reservation,
        project_directory=project_directory,
    ):
        predecessors = _read_validated_completion_predecessors(
            reservation,
            activation_publication,
            source_publication,
            candidate_publication,
            predictions_publication,
            receipt_publication,
            project_directory=project_directory,
        )

        # Cet instant unique suit la relecture fsync du recu et precede
        # immediatement la construction des octets canoniques de COMPLETED.
        completed_at = _format_utc_seconds(
            _utc_now(),
            field="completed_at_utc",
        )
        if completed_at < predecessors.receipt_finalized_at_utc:
            raise ShadowPredictionError(
                "COMPLETED ne peut pas preceder la finalisation du recu."
            )
        earliest = predecessors.earliest_predicted_scheduled_start_utc
        lead_seconds: int | None = None
        if earliest is not None:
            lead_seconds = int(
                (
                    datetime.fromisoformat(earliest.replace("Z", "+00:00"))
                    - datetime.fromisoformat(
                        completed_at.replace("Z", "+00:00")
                    )
                ).total_seconds()
            )
            if lead_seconds < 120 * 60:
                raise ShadowPredictionError(
                    "Le slot ne peut plus etre termine deux heures avant "
                    "le premier match predit."
                )

        receipt_relative_path = receipt_publication.receipt_relative_path
        marker: dict[str, Any] = {
            "marker_schema_version": 1,
            "batch_id": reservation.batch_id,
            "receipt_path": receipt_relative_path,
            "receipt_sha256": receipt_publication.receipt_sha256,
            "completed_at_utc": completed_at,
        }
        if not _valid_completed_marker(
            marker,
            receipt=predecessors.receipt,
            receipt_bytes=predecessors.receipt_bytes,
            receipt_path=receipt_relative_path,
            batch_id=reservation.batch_id,
        ):
            raise ShadowPredictionError(
                "Le marqueur COMPLETED construit viole le contrat v2."
            )
        completed_bytes = _canonical_json_file_bytes(marker)
        completed_path = reservation.slot_path / COMPLETED_FILENAME
        try:
            completed_sha256 = _publish_exclusive_verified(
                completed_path,
                completed_bytes,
            )
            confirmed = _read_validated_completion_predecessors(
                reservation,
                activation_publication,
                source_publication,
                candidate_publication,
                predictions_publication,
                receipt_publication,
                project_directory=project_directory,
                expected_additional_filenames=frozenset(
                    {COMPLETED_FILENAME}
                ),
            )
            persisted_completed = _read_canonical_json_object(completed_path)
            if (
                confirmed != predecessors
                or persisted_completed is None
                or persisted_completed[0] != marker
                or persisted_completed[1] != completed_bytes
                or hashlib.sha256(persisted_completed[1]).hexdigest()
                != completed_sha256
                or not _valid_completed_marker(
                    persisted_completed[0],
                    receipt=confirmed.receipt,
                    receipt_bytes=confirmed.receipt_bytes,
                    receipt_path=receipt_relative_path,
                    batch_id=reservation.batch_id,
                )
            ):
                raise ShadowPredictionError(
                    "COMPLETED ou ses predecesseurs ont change apres publication."
                )
        except ShadowPublicationConflictError as error:
            raise ShadowPredictionSlotConsumedError(
                "Le jalon COMPLETED est deja consomme."
            ) from error
        except OSError as error:
            raise ShadowPredictionError(
                "Publication de COMPLETED incomplete; aucune reparation "
                "n'est permise."
            ) from error

        completed_relative_path = (
            SHADOW_RESULT_ROOT_RELATIVE_PATH
            / predecessors.receipt_predecessors.target_official_date
            / COMPLETED_FILENAME
        ).as_posix()
        return ShadowCompletionPublication(
            slot_path=reservation.slot_path,
            completed_path=completed_path,
            completed_relative_path=completed_relative_path,
            completed_sha256=completed_sha256,
            completed_size_bytes=len(completed_bytes),
            completed_at_utc=completed_at,
            batch_id=reservation.batch_id,
            receipt_relative_path=receipt_relative_path,
            receipt_sha256=receipt_publication.receipt_sha256,
            batch_status=predecessors.batch_status,
            earliest_predicted_scheduled_start_utc=earliest,
            local_completion_lead_seconds=lead_seconds,
            official_prediction_created=(
                receipt_publication.official_prediction_created
            ),
        )


def fail_shadow_prediction_slot(
    reservation: ShadowPredictionSlotReservation,
    *,
    failed_at_utc: str,
    stage: str,
    error_type: str,
    error_message: str,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowPredictionSlotFailure:
    """Ferme definitivement un slot deja reserve avec FAILED.json.

    La primitive n'ecrit que dans le repertoire exact porte par la preuve de
    reservation. Elle exige que RESERVED soit encore canonique et identique,
    puis publie FAILED.json sans remplacement. Aucun fichier du slot n'est
    supprime, y compris si la publication ou sa verification echoue.
    """
    if type(reservation) is not ShadowPredictionSlotReservation:
        raise ShadowPredictionError(
            "Une preuve de reservation fantome exacte est requise."
        )
    if not isinstance(reservation.slot_path, Path):
        raise ShadowPredictionError(
            "Le chemin du slot reserve doit etre un objet Path."
        )

    failed_at = _require_utc_timestamp(
        failed_at_utc,
        field="failed_at_utc",
    )
    failure_stage = _require_nonempty_text(stage, field="stage")
    failure_type = _require_nonempty_text(
        error_type,
        field="error_type",
    )
    failure_message = _require_nonempty_text(
        error_message,
        field="error_message",
    )
    slot_key = _require_sha256(reservation.slot_key, field="slot_key")
    batch_id = _require_sha256(reservation.batch_id, field="batch_id")
    reserved_sha256 = _require_sha256(
        reservation.reserved_marker_sha256,
        field="reserved_marker_sha256",
    )

    reserved = reservation.reserved_marker
    if not _has_exact_keys(reserved, _RESERVED_MARKER_KEYS):
        raise ShadowPredictionError(
            "La preuve de reservation ne contient pas un marqueur RESERVED "
            "exact."
        )
    assert isinstance(reserved, dict)
    target_date = _require_date_string(
        reserved.get("target_official_date"),
        field="RESERVED.target_official_date",
    )
    target = _parse_target_official_date(target_date)
    if (
        target.year != EXPECTED_TARGET_SEASON
        or target < EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON
    ):
        raise ShadowPredictionError(
            "La preuve RESERVED ne vise pas une date admissible du "
            "protocole fantome v2."
        )
    reserved_at = _require_utc_timestamp(
        reserved.get("reserved_at_utc"),
        field="RESERVED.reserved_at_utc",
    )
    protocol_hash = _require_sha256(
        reserved.get("shadow_protocol_sha256"),
        field="RESERVED.shadow_protocol_sha256",
    )
    manifest_hash = _require_sha256(
        reserved.get("execution_manifest_sha256"),
        field="RESERVED.execution_manifest_sha256",
    )
    runtime_commit = _require_git_commit(
        reserved.get("runtime_code_commit"),
        field="RESERVED.runtime_code_commit",
    )
    if protocol_hash != EXPECTED_SHADOW_PROTOCOL_SHA256:
        raise ShadowPredictionError(
            "FAILED.json exige l'empreinte du protocole fantome v2 fige."
        )
    expected_slot_key = build_slot_key(
        shadow_protocol_sha256=protocol_hash,
        target_official_date=target_date,
    )
    expected_batch_id = build_batch_id(
        slot_key=expected_slot_key,
        execution_manifest_sha256=manifest_hash,
        model_artifact_sha256=EXPECTED_MODEL_ARTIFACT_SHA256,
    )
    expected_reserved: dict[str, Any] = {
        "marker_schema_version": 1,
        "batch_id": expected_batch_id,
        "slot_key": expected_slot_key,
        "target_official_date": target_date,
        "reserved_at_utc": reserved_at,
        "shadow_protocol_sha256": protocol_hash,
        "execution_manifest_sha256": manifest_hash,
        "runtime_code_commit": runtime_commit,
    }
    if (
        not _valid_reserved_marker(
            reserved,
            batch_id=expected_batch_id,
            slot_key=expected_slot_key,
            target_official_date=target_date,
            shadow_protocol_sha256=protocol_hash,
            execution_manifest_sha256=manifest_hash,
        )
        or reserved != expected_reserved
        or slot_key != expected_slot_key
        or batch_id != expected_batch_id
    ):
        raise ShadowPredictionError(
            "La preuve de reservation ne correspond pas aux identifiants "
            "figes du slot."
        )
    reserved_bytes = _canonical_json_file_bytes(expected_reserved)
    if hashlib.sha256(reserved_bytes).hexdigest() != reserved_sha256:
        raise ShadowPredictionError(
            "L'empreinte de la preuve RESERVED est incoherente."
        )
    if failed_at < reserved_at:
        raise ShadowPredictionError(
            "failed_at_utc ne peut pas preceder RESERVED.reserved_at_utc."
        )

    failed_marker: dict[str, Any] = {
        "marker_schema_version": 1,
        "batch_id": batch_id,
        "slot_key": slot_key,
        "target_official_date": target_date,
        "failed_at_utc": failed_at,
        "stage": failure_stage,
        "error_type": failure_type,
        "error_message": failure_message,
        "shadow_protocol_sha256": protocol_hash,
        "execution_manifest_sha256": manifest_hash,
        "runtime_code_commit": runtime_commit,
    }
    if not _has_exact_keys(failed_marker, _FAILED_MARKER_KEYS):
        raise AssertionError("Schema interne FAILED.json incoherent.")
    failed_bytes = _canonical_json_file_bytes(failed_marker)

    result_root = _require_shadow_result_root(
        project_directory=project_directory,
    )
    expected_slot_path = result_root / target_date
    if reservation.slot_path != expected_slot_path:
        raise ShadowPredictionError(
            "La preuve de reservation ne vise pas le slot attendu."
        )
    slot_mode = _lstat_mode(expected_slot_path)
    if not isinstance(slot_mode, int) or not stat.S_ISDIR(slot_mode):
        raise ShadowPredictionError(
            "Le slot reserve doit rester un repertoire local non "
            "symbolique."
        )

    for terminal_name in ("COMPLETED", "FAILED.json"):
        if _lstat_mode(expected_slot_path / terminal_name) is not _PATH_MISSING:
            raise ShadowPredictionSlotConsumedError(
                "Le slot fantome possede deja un marqueur terminal et ne "
                "peut pas etre modifie."
            )

    persisted_reserved = _read_canonical_json_object(
        expected_slot_path / "RESERVED"
    )
    if persisted_reserved is None:
        raise ShadowPredictionError(
            "Le slot ne contient pas la preuve RESERVED canonique attendue."
        )
    persisted_marker, persisted_bytes = persisted_reserved
    if (
        persisted_marker != expected_reserved
        or persisted_bytes != reserved_bytes
        or hashlib.sha256(persisted_bytes).hexdigest() != reserved_sha256
    ):
        raise ShadowPredictionError(
            "La preuve RESERVED persistee ne correspond pas a la "
            "reservation fournie."
        )

    try:
        failed_sha256 = _publish_exclusive_verified(
            expected_slot_path / "FAILED.json",
            failed_bytes,
        )
    except ShadowPublicationConflictError as error:
        raise ShadowPredictionSlotConsumedError(
            "Le slot fantome a deja ete ferme par une autre execution."
        ) from error

    return ShadowPredictionSlotFailure(
        slot_path=expected_slot_path,
        failed_marker=failed_marker,
        failed_marker_sha256=failed_sha256,
    )


def _read_frozen_protocol(
    project_directory: Path,
) -> tuple[dict[str, Any], str]:
    """Lit uniquement le chemin local impose, controle avant decodage."""
    project = Path(project_directory).expanduser().resolve()
    if not project.is_dir():
        raise ShadowPredictionError(
            f"Dossier du projet introuvable : {project}."
        )

    protocol_path = project.joinpath(*SHADOW_PROTOCOL_RELATIVE_PATH.parts)
    if protocol_path.is_symlink():
        raise ShadowPredictionError(
            "Le protocole v2 local ne peut pas etre un lien symbolique : "
            f"{SHADOW_PROTOCOL_RELATIVE_PATH.as_posix()}."
        )
    try:
        protocol_bytes = protocol_path.read_bytes()
    except OSError as error:
        raise ShadowPredictionError(
            "Protocole v2 introuvable au chemin fige "
            f"{SHADOW_PROTOCOL_RELATIVE_PATH.as_posix()}."
        ) from error

    actual_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    if actual_sha256 != EXPECTED_SHADOW_PROTOCOL_SHA256:
        raise ShadowPredictionError(
            "SHA-256 invalide pour le protocole v2 local : "
            f"{actual_sha256}, attendu {EXPECTED_SHADOW_PROTOCOL_SHA256}."
        )

    try:
        decoded = protocol_bytes.decode("utf-8")
        payload = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShadowPredictionError(
            "Le protocole v2 fige n'est pas un objet JSON UTF-8 valide."
        ) from error
    protocol = dict(
        _require_mapping(payload, description="le protocole v2")
    )
    return protocol, actual_sha256


def _validate_protocol_contract(protocol: Mapping[str, Any]) -> None:
    """Valide les verrous necessaires au seul mode apercu."""
    _require_exact(
        protocol,
        "shadow_prediction_protocol_version",
        EXPECTED_SHADOW_PROTOCOL_VERSION,
        context="protocol",
    )
    _require_exact(
        protocol,
        "registered_on",
        EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON.isoformat(),
        context="protocol",
    )
    _require_exact(
        protocol,
        "status",
        EXPECTED_SHADOW_PROTOCOL_STATUS,
        context="protocol",
    )
    _require_exact(
        protocol,
        "first_shadow_prediction_created_at_registration",
        False,
        context="protocol",
    )

    freeze = _require_mapping(
        protocol.get("execution_freeze"),
        description="protocol.execution_freeze",
    )
    _require_exact(
        freeze,
        "shadow_service_module_path_exact",
        EXPECTED_SERVICE_MODULE_PATH,
        context="protocol.execution_freeze",
    )
    _require_exact(
        freeze,
        "shadow_protocol_path_exact",
        SHADOW_PROTOCOL_RELATIVE_PATH.as_posix(),
        context="protocol.execution_freeze",
    )
    _require_exact(
        freeze,
        "default_mode",
        EXPECTED_PREVIEW_MODE,
        context="protocol.execution_freeze",
    )
    for key in (
        "preview_must_not_deserialize_model",
        "preview_must_not_call_predict_proba",
        "preview_must_not_reserve_output_slot",
        "preview_must_not_create_output_files",
    ):
        _require_exact(
            freeze,
            key,
            True,
            context="protocol.execution_freeze",
        )

    operating = _require_mapping(
        protocol.get("operating_mode"),
        description="protocol.operating_mode",
    )
    for key, expected in (
        ("strictly_prospective", True),
        ("historical_backfill_allowed", False),
        ("target_season_exact", EXPECTED_TARGET_SEASON),
        ("future_other_seasons_require_new_registered_protocol", True),
        ("time_basis", "UTC"),
    ):
        _require_exact(
            operating,
            key,
            expected,
            context="protocol.operating_mode",
        )

    candidates = _require_mapping(
        protocol.get("candidate_games"),
        description="protocol.candidate_games",
    )
    _require_exact(
        candidates,
        "season_must_equal",
        EXPECTED_TARGET_SEASON,
        context="protocol.candidate_games",
    )

    lineage = _require_mapping(
        protocol.get("validated_lineage"),
        description="protocol.validated_lineage",
    )
    _validate_declared_file(
        lineage,
        "model_artifact",
        expected_path=EXPECTED_MODEL_ARTIFACT_PATH,
        expected_sha256=EXPECTED_MODEL_ARTIFACT_SHA256,
    )
    _validate_declared_file(
        lineage,
        "artifact_manifest",
        expected_path=EXPECTED_ARTIFACT_MANIFEST_PATH,
        expected_sha256=EXPECTED_ARTIFACT_MANIFEST_SHA256,
    )
    _validate_declared_file(
        lineage,
        "model_protocol",
        expected_path=EXPECTED_MODEL_PROTOCOL_PATH,
        expected_sha256=EXPECTED_MODEL_PROTOCOL_SHA256,
    )
    historical = _require_mapping(
        lineage.get("historical_dataset_lineage_only"),
        description=(
            "protocol.validated_lineage.historical_dataset_lineage_only"
        ),
    )
    _require_exact(
        historical,
        "must_not_be_loaded_for_shadow_prediction",
        True,
        context=(
            "protocol.validated_lineage.historical_dataset_lineage_only"
        ),
    )


def _validate_declared_file(
    lineage: Mapping[str, Any],
    key: str,
    *,
    expected_path: str,
    expected_sha256: str,
) -> None:
    declaration = _require_mapping(
        lineage.get(key),
        description=f"protocol.validated_lineage.{key}",
    )
    context = f"protocol.validated_lineage.{key}"
    _require_exact(
        declaration,
        "path",
        expected_path,
        context=context,
    )
    _require_exact(
        declaration,
        "sha256",
        expected_sha256,
        context=context,
    )


def _parse_target_official_date(value: date | str) -> date:
    """Accepte une date ou une chaine ISO canonique, jamais un instant."""
    if isinstance(value, datetime):
        raise ShadowPredictionError(
            "target_official_date doit etre une date sans heure."
        )
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not _CANONICAL_DATE_PATTERN.fullmatch(
        value
    ):
        raise ShadowPredictionError(
            "target_official_date doit respecter exactement YYYY-MM-DD."
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ShadowPredictionError(
            f"target_official_date invalide : {value!r}."
        ) from error
    if parsed.isoformat() != value:
        raise ShadowPredictionError(
            "target_official_date doit etre une date ISO canonique."
        )
    return parsed


def preview_shadow_prediction(
    target_official_date: date | str,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowPredictionPreview:
    """Valide le protocole et la date sans ouvrir aucune entree officielle."""
    protocol, protocol_sha256 = _read_frozen_protocol(project_directory)
    _validate_protocol_contract(protocol)
    target = _parse_target_official_date(target_official_date)

    if target.year != EXPECTED_TARGET_SEASON:
        raise ShadowPredictionError(
            "Saison cible invalide : "
            f"{target.year}, attendu {EXPECTED_TARGET_SEASON}."
        )
    if target < EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON:
        raise ShadowPredictionError(
            "La date cible ne peut pas preceder l'enregistrement du "
            f"protocole v2 ({EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON})."
        )

    return ShadowPredictionPreview(
        mode=EXPECTED_PREVIEW_MODE,
        shadow_protocol_version=EXPECTED_SHADOW_PROTOCOL_VERSION,
        shadow_protocol_path=SHADOW_PROTOCOL_RELATIVE_PATH.as_posix(),
        shadow_protocol_sha256=protocol_sha256,
        shadow_protocol_registered_on=(
            EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON.isoformat()
        ),
        shadow_protocol_status=EXPECTED_SHADOW_PROTOCOL_STATUS,
        target_official_date=target.isoformat(),
        target_season=target.year,
        target_validation_scope="STATIC_PROTOCOL_ONLY",
        execution_ready=False,
        execution_manifest_read=False,
        activation_read=False,
        model_artifact_read=False,
        model_deserialized=False,
        sqlite_read=False,
        network_request_performed=False,
        output_slot_reserved=False,
        output_files_created=False,
        predictions_computed=False,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Valide localement le protocole fantome MLB v2 et une date "
            "cible. Ce jalon est un apercu sans modele ni prediction."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Affiche un JSON deterministe; aucun mode d'execution n'existe ici."""
    parser = _build_argument_parser()
    parser.add_argument(
        "--target-official-date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Date officielle MLB cible, obligatoirement en saison 2026.",
    )
    arguments = parser.parse_args(argv)
    try:
        preview = preview_shadow_prediction(arguments.target_official_date)
    except ShadowPredictionError as error:
        parser.error(str(error))
    print(preview.to_canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
