"""Moteur append-only du scoring prospectif shadow MLB 2026.

Le moteur valide l'autorite du protocole deja preenregistre, reserve
atomiquement un creneau d'observation et archive une reponse MLB fraiche.
Il ne lit ni SQLite ni modele et n'interprete encore aucun score : cette
couche conserve uniquement la preuve brute ou ferme l'observation en echec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from enum import Enum
import base64
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence
import zlib

import requests

from src import shadow_certification as certification
from src import shadow_prediction as shadow
from src import shadow_scoring_registration as scoring_registration


PROJECT_DIRECTORY = shadow.PROJECT_DIRECTORY
SCORING_ENGINE_RELATIVE_PATH = PurePosixPath("src/shadow_scoring.py")
SCORING_OUTPUT_ROOT_RELATIVE_PATH = PurePosixPath(
    "shadow_scores/logistic_team_form_v1_platt_shadow_v2_2026_v1"
)
EXPECTED_PROTOCOL_INTRODUCTION_COMMIT = (
    "64a8d07930f85c5a4101af8868ac6a1fbd7a1ccd"
)
EXPECTED_REGISTRATION_COMMIT = (
    "c48ccd02b9fb7714bf756f3be78c23d5bf8f649b"
)
EXPECTED_REGISTRATION_SHA256 = (
    "ddbd56c3bf827fdceecde6cf3e8aac71"
    "987c88f9573fd5ae992eb7dcd056d94e"
)
EXPECTED_REGISTRATION_REMOTE_EVIDENCE_SHA256 = (
    "daf589053610a0c4975d6a3e4f9cc428"
    "65bbd5f85a8a34c875146a3117e30c41"
)
EXPECTED_REGISTRATION_SERVICE_SHA256 = (
    "53abeaee65f684dde9e54d78080ad976"
    "7653ee347c41cba774474dfeb13d0a69"
)
EXPECTED_HORIZON_DATES = tuple(
    date(2026, 9, day) for day in range(10, 28)
)
FINAL_CHECKPOINT_AT_UTC = datetime(
    2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc
)
ORDINARY_CHECKPOINT_TIME_UTC = time(6, 0, 0)
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
OUTCOME_REQUEST_HEADERS = {
    "User-Agent": "fredo-mlb-predictor-scoring-v1/1.0",
}
OUTCOME_REQUEST_TIMEOUT_SECONDS = 30
OUTCOME_EVIDENCE_FILENAME = "outcome_observation.remote.json.gz"
FAILED_FILENAME = "FAILED.json"

_OUTCOME_EVIDENCE_KEYS = frozenset(
    {
        "evidence_schema_version",
        "target_official_date",
        "checkpoint_utc_date",
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
_OUTCOME_SELECTED_HEADER_KEYS = frozenset({"date", "content-type"})
_FAILED_MARKER_KEYS = frozenset(
    {
        "marker_schema_version",
        "protocol_id",
        "target_official_date",
        "checkpoint_utc_date",
        "observation_id",
        "failed_at_utc",
        "stage",
        "error_type",
        "error_message",
    }
)
_FAILURE_STAGES = frozenset({"OUTCOME_REQUEST", "OUTCOME_VALIDATION"})

_FINAL_STATUS_CODES = frozenset({"F", "FG", "FO", "FR"})
_FINAL_STATUS_DETAILS = frozenset({"COMPLETED EARLY", "FINAL", "GAME OVER"})
_CANCELLED_STATUS_CODES = frozenset({"C", "CI", "CR"})
_CANCELLED_STATUS_DETAILS = frozenset({"CANCELLED"})
_POSTPONED_STATUS_CODES = frozenset({"D", "DI", "DR"})
_POSTPONED_STATUS_DETAILS = frozenset({"POSTPONED"})
_TERMINAL_OUTCOME_STATUSES = frozenset(
    {
        "SCORED_FINAL",
        "VOID_CANCELLED",
        "VOID_RESCHEDULED_OFFICIAL_DATE",
        "VOID_POSTPONED_AT_DEADLINE",
        "VOID_UNRESOLVED_AT_DEADLINE",
    }
)
_VOID_OUTCOME_STATUSES = frozenset(
    {
        "VOID_CANCELLED",
        "VOID_RESCHEDULED_OFFICIAL_DATE",
        "VOID_POSTPONED_AT_DEADLINE",
        "VOID_UNRESOLVED_AT_DEADLINE",
    }
)
_PENDING_OUTCOME_STATUSES = frozenset(
    {
        "PENDING_MISSING_FROM_OBSERVATION",
        "PENDING_NONTERMINAL",
        "PENDING_POSTPONED",
    }
)
_ALL_OUTCOME_STATUSES = frozenset(
    {"SCORED_FINAL"} | _VOID_OUTCOME_STATUSES | _PENDING_OUTCOME_STATUSES
)
_ADJUDICATION_COLUMNS = (
    "prediction_id",
    "batch_id",
    "game_id",
    "occurrence_key",
    "target_official_date",
    "away_team_id",
    "home_team_id",
    "p_away_win",
    "p_home_win",
    "predicted_side",
    "outcome_status",
    "away_score",
    "home_score",
    "home_win",
    "actual_winner",
    "classification_correct",
    "individual_log_loss",
    "individual_brier_score",
    "status_code_normalized",
    "status_detail_normalized",
    "final_official_date",
    "outcome_http_date_utc",
    "outcome_response_received_at_utc",
    "outcome_evidence_sha256",
)
_PROBABILITY_TEXT_PATTERN = re.compile(
    r"(?:0(?:\.[0-9]+)?|1(?:\.0+)?)\Z"
)
_STATUS_WHITESPACE_PATTERN = re.compile(r"\s+")
_FIXED_12_QUANTUM = Decimal("0.000000000001")
_LOG_LOSS_CLIP_EPSILON = 1e-15

_RESERVED_MARKER_KEYS = frozenset(
    {
        "marker_schema_version",
        "protocol_id",
        "scoring_protocol_sha256",
        "target_official_date",
        "checkpoint_utc_date",
        "observation_id",
        "reserved_at_utc",
        "runtime_code_commit",
    }
)


class ScoringObservationSlotState(str, Enum):
    """Etat minimal, presence-first, d'un creneau d'observation."""

    ABSENT = "ABSENT"
    RESERVED_EXACT = "RESERVED_EXACT"
    CONSUMED = "CONSUMED"


class ScoringOutcomeRequestError(shadow.ShadowPredictionError):
    """Echec controle de l'unique requete MLB d'une observation."""


class ScoringOutcomeValidationError(shadow.ShadowPredictionError):
    """Reponse MLB recue mais incompatible avec le protocole fige."""


class ScoringAdjudicationError(shadow.ShadowPredictionError):
    """Conflit structurel interdisant toute adjudication silencieuse."""


class ScoringOutcomeFamily(str, Enum):
    """Famille metier d'une occurrence MLB apres normalisation."""

    FINAL = "FINAL"
    CANCELLED = "CANCELLED"
    NONTERMINAL = "NONTERMINAL"
    POSTPONED = "POSTPONED"


class ScoringPredictionSourceStatus(str, Enum):
    """Statut exact d'une date de l'intention prospective figee."""

    CERTIFIED_NONEMPTY = "CERTIFIED_NONEMPTY"
    COMPLETED_EMPTY = "COMPLETED_EMPTY"
    LOCAL_ONLY = "LOCAL_ONLY"
    FAILED = "FAILED"
    MISSED = "MISSED"


@dataclass(frozen=True, slots=True)
class ScoringExecutionAuthority:
    """Autorite Git locale complete, sans aucune lecture de resultat."""

    protocol: dict[str, Any] = field(repr=False)
    registration: dict[str, Any] = field(repr=False)
    scoring_protocol_sha256: str
    protocol_introduction_commit: str
    registration_commit: str
    registration_sha256: str
    registration_remote_evidence_sha256: str
    runtime_code_commit: str
    scoring_engine_sha256: str


@dataclass(frozen=True, slots=True)
class ScoringObservationSlotInspection:
    """Inspection locale d'un chemin qui est definitivement consommable."""

    state: ScoringObservationSlotState
    slot_path: Path
    target_official_date: str
    checkpoint_utc_date: str
    observation_id: str
    reserved_marker: dict[str, Any] | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ScoringObservationReservation:
    """Preuve en memoire de l'unique publication RESERVED."""

    slot_path: Path
    target_official_date: str
    checkpoint_utc_date: str
    observation_id: str
    reserved_marker: dict[str, Any] = field(repr=False)
    reserved_marker_sha256: str


@dataclass(frozen=True, slots=True)
class ScoringOutcomeObservationEvidence:
    """Reponse MLB validee en memoire, encore non publiee."""

    target_official_date: str
    checkpoint_utc_date: str
    outcome_http_date_utc: str
    response_received_at_utc: str
    response_body_sha256: str
    flattened_occurrence_count: int
    raw_evidence: dict[str, Any] = field(repr=False)
    canonical_json_bytes: bytes = field(repr=False)
    canonical_gzip_bytes: bytes = field(repr=False)
    canonical_gzip_sha256: str


@dataclass(frozen=True, slots=True)
class ScoringOutcomeEvidencePublication:
    """Preuve de publication append-only de la reponse MLB brute."""

    reservation: ScoringObservationReservation = field(repr=False)
    evidence_path: Path
    evidence_sha256: str
    evidence_size_bytes: int
    outcome_http_date_utc: str
    response_received_at_utc: str
    response_body_sha256: str
    flattened_occurrence_count: int
    evidence: ScoringOutcomeObservationEvidence = field(repr=False)


@dataclass(frozen=True, slots=True)
class ScoringObservationFailure:
    """Marqueur terminal d'une requete ou validation MLB echouee."""

    reservation: ScoringObservationReservation = field(repr=False)
    failed_path: Path
    failed_marker: dict[str, Any] = field(repr=False)
    failed_marker_sha256: str
    stage: str
    error_type: str


@dataclass(frozen=True, slots=True)
class CertifiedScoringPrediction:
    """Projection minimale d'une ligne de prediction certifiee et immuable."""

    prediction_id: str
    batch_id: str
    game_id: int
    occurrence_key: str
    season: int
    target_official_date: str
    away_team_id: int
    home_team_id: int
    p_home_win: str
    p_away_win: str


@dataclass(frozen=True, slots=True)
class ImmutableScoringPredictionSource:
    """Cohorte d'une date relue seulement depuis des blobs Git immuables."""

    target_official_date: str
    status: ScoringPredictionSourceStatus
    results_commit: str | None
    certification_commit: str | None
    batch_id: str | None
    predictions: tuple[CertifiedScoringPrediction, ...]
    predictions_sha256: str | None
    receipt_sha256: str | None
    certification_sha256: str | None
    raw_certification_evidence_sha256: str | None
    status_reason: str


@dataclass(frozen=True, slots=True)
class _ValidatedImmutableShadowBatch:
    """Lot terminal controle a partir de ses huit blobs Git uniquement."""

    target_official_date: str
    results_commit: str
    batch_id: str
    status: str
    earliest_predicted_start_utc: str | None
    results_tree_file_hashes: tuple[dict[str, object], ...]
    receipt: dict[str, Any] = field(repr=False)
    receipt_bytes: bytes = field(repr=False)
    predictions_bytes: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class ReducedScoringOutcome:
    """Occurrence canonique issue de la reduction exacte d'un gamePk."""

    game_id: int
    season: int
    game_type: str
    away_team_id: int
    home_team_id: int
    family: ScoringOutcomeFamily
    status_code_normalized: str
    status_detail_normalized: str
    final_official_date: str | None
    away_score: int | None
    home_score: int | None


@dataclass(frozen=True, slots=True)
class ScoringAdjudication:
    """Ligne d'adjudication pure conforme au schema CSV preenregistre."""

    prediction_id: str
    batch_id: str
    game_id: int
    occurrence_key: str
    target_official_date: str
    away_team_id: int
    home_team_id: int
    p_away_win: str
    p_home_win: str
    predicted_side: str
    outcome_status: str
    away_score: int | None
    home_score: int | None
    home_win: int | None
    actual_winner: str | None
    classification_correct: int | None
    individual_log_loss: str | None
    individual_brier_score: str | None
    status_code_normalized: str
    status_detail_normalized: str
    final_official_date: str | None
    outcome_http_date_utc: str
    outcome_response_received_at_utc: str
    outcome_evidence_sha256: str

    def as_csv_row(self) -> tuple[object, ...]:
        """Retourne les valeurs dans l'ordre exact du protocole."""
        return tuple(getattr(self, column) for column in _ADJUDICATION_COLUMNS)


@dataclass(frozen=True, slots=True)
class ScoringAdjudicationBatch:
    """Resultat en memoire d'une adjudication complete, jamais filtree."""

    target_official_date: str
    checkpoint_utc_date: str
    outcome_evidence_sha256: str
    adjudications: tuple[ScoringAdjudication, ...]
    certified_prediction_count: int
    scored_count: int
    void_count: int
    pending_count: int
    correct_count: int
    incorrect_count: int


@dataclass(frozen=True, slots=True)
class _RawScoringOccurrence:
    """Occurrence MLB structurellement validee avant reduction."""

    game_id: int
    raw_season: int | str
    season: int
    game_type: str
    away_team_id: int
    home_team_id: int
    family: ScoringOutcomeFamily
    status_code_normalized: str
    status_detail_normalized: str
    official_date: str | None
    away_score: int | None
    home_score: int | None

    @property
    def exact_identity(self) -> tuple[object, ...]:
        return (
            type(self.raw_season),
            self.raw_season,
            self.game_type,
            self.away_team_id,
            self.home_team_id,
        )


def _parse_horizon_target(value: date | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        parsed = value
    elif type(value) is str:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise shadow.ShadowPredictionError(
                "target_official_date doit etre une date ISO exacte."
            ) from error
        if parsed.isoformat() != value:
            raise shadow.ShadowPredictionError(
                "target_official_date doit etre canonique."
            )
    else:
        raise shadow.ShadowPredictionError(
            "target_official_date doit etre une date exacte."
        )
    if parsed not in EXPECTED_HORIZON_DATES:
        raise shadow.ShadowPredictionError(
            "La date cible est hors de l'horizon prospectif fige."
        )
    return parsed


def _parse_checkpoint_date(value: date | str, *, target: date) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        parsed = value
    elif type(value) is str:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise shadow.ShadowPredictionError(
                "checkpoint_utc_date doit etre une date ISO exacte."
            ) from error
        if parsed.isoformat() != value:
            raise shadow.ShadowPredictionError(
                "checkpoint_utc_date doit etre canonique."
            )
    else:
        raise shadow.ShadowPredictionError(
            "checkpoint_utc_date doit etre une date exacte."
        )
    if parsed < target + timedelta(days=1) or parsed > FINAL_CHECKPOINT_AT_UTC.date():
        raise shadow.ShadowPredictionError(
            "Le checkpoint est hors de la fenetre preenregistree."
        )
    return parsed


def _parse_utc_timestamp(value: object, *, field_name: str) -> datetime:
    canonical = shadow._require_utc_timestamp(value, field=field_name)
    parsed = datetime.fromisoformat(canonical.replace("Z", "+00:00"))
    if parsed.microsecond != 0:
        raise shadow.ShadowPredictionError(
            f"{field_name} doit etre precis a la seconde."
        )
    return parsed


def _checkpoint_instant(checkpoint: date) -> datetime:
    if checkpoint == FINAL_CHECKPOINT_AT_UTC.date():
        return FINAL_CHECKPOINT_AT_UTC
    return datetime.combine(
        checkpoint,
        ORDINARY_CHECKPOINT_TIME_UTC,
        tzinfo=timezone.utc,
    )


def _utc_now() -> datetime:
    """Horloge UTC injectable des orchestrations de scoring."""
    return datetime.now(timezone.utc)


def _format_utc_seconds(value: datetime, *, field_name: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise shadow.ShadowPredictionError(
            f"{field_name} doit etre un datetime UTC conscient."
        )
    normalized = value.astimezone(timezone.utc)
    if normalized.microsecond != 0:
        normalized = normalized.replace(microsecond=0)
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_observation_id(
    *,
    scoring_protocol_sha256: str,
    target_official_date: date | str,
    checkpoint_utc_date: date | str,
) -> str:
    """Applique exactement la formule d'identifiant preenregistree."""
    protocol_hash = shadow._require_sha256(
        scoring_protocol_sha256,
        field="scoring_protocol_sha256",
    )
    if protocol_hash != scoring_registration.EXPECTED_SCORING_PROTOCOL_SHA256:
        raise shadow.ShadowPredictionError(
            "L'identifiant exige le protocole de scoring fige."
        )
    target = _parse_horizon_target(target_official_date)
    checkpoint = _parse_checkpoint_date(checkpoint_utc_date, target=target)
    preimage = (
        f"{protocol_hash}\n{target.isoformat()}\n"
        f"{checkpoint.isoformat()}\n"
    ).encode("utf-8")
    return hashlib.sha256(preimage).hexdigest()


def _observation_slot_path(
    project_directory: Path,
    target: date,
    checkpoint: date,
) -> Path:
    return project_directory.joinpath(
        *SCORING_OUTPUT_ROOT_RELATIVE_PATH.parts,
        target.isoformat(),
        "observations",
        checkpoint.isoformat(),
    )


def inspect_scoring_observation_slot_presence_first(
    target_official_date: date | str,
    checkpoint_utc_date: date | str,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ScoringObservationSlotInspection:
    """Inspecte le chemin cible avant Git, reseau, resultat ou modele."""
    project = shadow._preflight_project_directory(project_directory)
    target = _parse_horizon_target(target_official_date)
    checkpoint = _parse_checkpoint_date(checkpoint_utc_date, target=target)
    observation_id = build_observation_id(
        scoring_protocol_sha256=(
            scoring_registration.EXPECTED_SCORING_PROTOCOL_SHA256
        ),
        target_official_date=target,
        checkpoint_utc_date=checkpoint,
    )
    slot = _observation_slot_path(project, target, checkpoint)
    mode = shadow._lstat_mode(slot)
    state = (
        ScoringObservationSlotState.ABSENT
        if mode is shadow._PATH_MISSING
        else ScoringObservationSlotState.CONSUMED
    )
    return ScoringObservationSlotInspection(
        state=state,
        slot_path=slot,
        target_official_date=target.isoformat(),
        checkpoint_utc_date=checkpoint.isoformat(),
        observation_id=observation_id,
    )


def _registration_evidence(
    raw_blob: bytes,
    registration: Mapping[str, Any],
) -> shadow.ShadowActivationReverificationEvidence:
    try:
        raw_json = gzip.decompress(raw_blob)
    except (OSError, EOFError, zlib.error) as error:
        raise shadow.ShadowPredictionError(
            "La preuve distante du preenregistrement est illisible."
        ) from error
    raw = shadow._read_canonical_json_bytes(
        raw_json,
        description="preuve distante du preenregistrement scoring",
    )
    if shadow._canonical_gzip_bytes(raw_json) != raw_blob:
        raise shadow.ShadowPredictionError(
            "La preuve distante du preenregistrement n'est pas canonique."
        )
    return shadow.ShadowActivationReverificationEvidence(
        activation_introduction_commit=EXPECTED_PROTOCOL_INTRODUCTION_COMMIT,
        activation_remote_ref=shadow.GITHUB_REMOTE_REF,
        activation_remote_reverified_at_utc=registration.get(
            "remote_http_date_utc"
        ),
        response_received_at_utc=registration.get(
            "remote_response_received_at_utc"
        ),
        response_body_sha256=registration.get(
            "remote_response_body_sha256"
        ),
        raw_evidence=raw,
        canonical_json_bytes=raw_json,
        canonical_gzip_bytes=raw_blob,
        canonical_gzip_sha256=hashlib.sha256(raw_blob).hexdigest(),
    )


def _validate_published_registration(
    registration: Mapping[str, Any],
    raw_blob: bytes,
) -> None:
    if (
        type(registration) is not dict
        or frozenset(registration) != scoring_registration._REGISTRATION_KEYS
    ):
        raise shadow.ShadowPredictionError(
            "Le preenregistrement publie ne respecte pas son schema exact."
        )
    expected_values: dict[str, object] = {
        "registration_schema_version": 1,
        "protocol_id": scoring_registration.EXPECTED_PROTOCOL_ID,
        "status": scoring_registration.REGISTRATION_STATUS,
        "claim_level": scoring_registration.REGISTRATION_CLAIM_LEVEL,
        "scoring_protocol_path": (
            scoring_registration.SCORING_PROTOCOL_RELATIVE_PATH.as_posix()
        ),
        "scoring_protocol_sha256": (
            scoring_registration.EXPECTED_SCORING_PROTOCOL_SHA256
        ),
        "scoring_protocol_introduction_commit": (
            EXPECTED_PROTOCOL_INTRODUCTION_COMMIT
        ),
        "registration_service_path": (
            scoring_registration.REGISTRATION_SERVICE_RELATIVE_PATH.as_posix()
        ),
        "registration_service_sha256": EXPECTED_REGISTRATION_SERVICE_SHA256,
        "runtime_code_commit": EXPECTED_PROTOCOL_INTRODUCTION_COMMIT,
        "remote_ref": shadow.GITHUB_REMOTE_REF,
        "remote_response_status_code": 200,
        "remote_response_redirect_count": 0,
        "raw_remote_evidence_path": (
            scoring_registration.RAW_REMOTE_EVIDENCE_RELATIVE_PATH.as_posix()
        ),
        "raw_remote_evidence_sha256": (
            EXPECTED_REGISTRATION_REMOTE_EVIDENCE_SHA256
        ),
        "first_target_official_date": (
            scoring_registration.EXPECTED_FIRST_TARGET_DATE
        ),
        "first_batch_id": scoring_registration.EXPECTED_FIRST_BATCH_ID,
        "first_results_commit": (
            scoring_registration.EXPECTED_FIRST_RESULTS_COMMIT
        ),
        "first_certification_commit": (
            scoring_registration.EXPECTED_FIRST_CERTIFICATION_COMMIT
        ),
        "first_certification_sha256": (
            scoring_registration.EXPECTED_FIRST_CERTIFICATION_SHA256
        ),
        "first_predictions_sha256": (
            scoring_registration.EXPECTED_FIRST_PREDICTIONS_SHA256
        ),
        "first_receipt_sha256": (
            scoring_registration.EXPECTED_FIRST_RECEIPT_SHA256
        ),
        "first_earliest_predicted_start_utc": (
            scoring_registration.EXPECTED_FIRST_START_UTC
        ),
    }
    for key, expected in expected_values.items():
        actual = registration.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise shadow.ShadowPredictionError(
                f"Le preenregistrement publie diverge pour {key}."
            )
    attestations = registration.get("negative_attestations")
    if (
        type(attestations) is not dict
        or frozenset(attestations)
        != scoring_registration._NEGATIVE_ATTESTATION_KEYS
        or any(value is not True for value in attestations.values())
    ):
        raise shadow.ShadowPredictionError(
            "Les attestations du preenregistrement sont invalides."
        )
    if hashlib.sha256(raw_blob).hexdigest() != (
        EXPECTED_REGISTRATION_REMOTE_EVIDENCE_SHA256
    ):
        raise shadow.ShadowPredictionError(
            "L'empreinte de la preuve distante du preenregistrement diverge."
        )
    evidence = _registration_evidence(raw_blob, registration)
    shadow._validate_activation_reverification_evidence(evidence)
    _, lead_minutes = scoring_registration._remote_registration_lead(
        evidence.activation_remote_reverified_at_utc,
        evidence.response_received_at_utc,
    )
    if (
        registration.get("remote_query_url")
        != evidence.raw_evidence["request_url"]
        or registration.get("remote_effective_url")
        != evidence.raw_evidence["effective_url"]
        or registration.get("registered_at_utc")
        != evidence.response_received_at_utc
        or registration.get("remote_publication_lead_minutes")
        != lead_minutes
    ):
        raise shadow.ShadowPredictionError(
            "Le recu et la preuve distante du preenregistrement divergent."
        )


def verify_scoring_execution_authority(
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ScoringExecutionAuthority:
    """Valide Git, protocole et preenregistrement sans lire de resultat."""
    project = shadow._preflight_project_directory(project_directory)
    protocol, protocol_bytes, protocol_sha256 = (
        scoring_registration._read_scoring_protocol(project)
    )
    runtime_commit = shadow._require_exact_clean_git_root(project)
    protocol_commit, _ = shadow._git_immutable_introduction_blob(
        project,
        scoring_registration.SCORING_PROTOCOL_RELATIVE_PATH,
        protocol_bytes,
        expected_introduction_commit=EXPECTED_PROTOCOL_INTRODUCTION_COMMIT,
    )
    shadow._require_git_ancestor(
        project,
        protocol_commit,
        EXPECTED_REGISTRATION_COMMIT,
        strict=True,
        description="protocole de scoring vers preenregistrement",
    )
    shadow._require_git_ancestor(
        project,
        EXPECTED_REGISTRATION_COMMIT,
        runtime_commit,
        strict=True,
        description="preenregistrement vers moteur de scoring",
    )

    registration_bytes = shadow._read_regular_project_file(
        project,
        scoring_registration.REGISTRATION_RELATIVE_PATH,
        description="preenregistrement du scoring",
    )
    raw_bytes = shadow._read_regular_project_file(
        project,
        scoring_registration.RAW_REMOTE_EVIDENCE_RELATIVE_PATH,
        description="preuve distante du preenregistrement du scoring",
    )
    if hashlib.sha256(registration_bytes).hexdigest() != EXPECTED_REGISTRATION_SHA256:
        raise shadow.ShadowPredictionError(
            "L'empreinte du preenregistrement du scoring diverge."
        )
    for relative, content in (
        (scoring_registration.REGISTRATION_RELATIVE_PATH, registration_bytes),
        (scoring_registration.RAW_REMOTE_EVIDENCE_RELATIVE_PATH, raw_bytes),
    ):
        shadow._git_immutable_introduction_blob(
            project,
            relative,
            content,
            expected_introduction_commit=EXPECTED_REGISTRATION_COMMIT,
        )
    registration = shadow._read_canonical_json_bytes(
        registration_bytes,
        description="preenregistrement du scoring",
    )
    _validate_published_registration(registration, raw_bytes)

    engine_bytes = shadow._read_regular_project_file(
        project,
        SCORING_ENGINE_RELATIVE_PATH,
        description="moteur de scoring",
    )
    engine_blob = shadow._git_blob_at_commit(
        project,
        runtime_commit,
        SCORING_ENGINE_RELATIVE_PATH.as_posix(),
    )
    if engine_bytes != engine_blob:
        raise shadow.ShadowPredictionError(
            "Le moteur de scoring local differe de son blob Git runtime."
        )
    shadow._require_tracked_nonignored_root(
        project,
        SCORING_OUTPUT_ROOT_RELATIVE_PATH,
    )
    return ScoringExecutionAuthority(
        protocol=protocol,
        registration=registration,
        scoring_protocol_sha256=protocol_sha256,
        protocol_introduction_commit=protocol_commit,
        registration_commit=EXPECTED_REGISTRATION_COMMIT,
        registration_sha256=EXPECTED_REGISTRATION_SHA256,
        registration_remote_evidence_sha256=(
            EXPECTED_REGISTRATION_REMOTE_EVIDENCE_SHA256
        ),
        runtime_code_commit=runtime_commit,
        scoring_engine_sha256=hashlib.sha256(engine_bytes).hexdigest(),
    )


def _require_prediction_source_authorities(
    authority: ScoringExecutionAuthority,
    target: date,
    *,
    project_directory: Path,
) -> shadow.ShadowExecutionAuthority:
    """Relie l'autorite scoring au preflight shadow sans lire un resultat."""
    if type(authority) is not ScoringExecutionAuthority:
        raise shadow.ShadowPredictionError(
            "Une autorite d'execution scoring exacte est requise."
        )
    if (
        authority.scoring_protocol_sha256
        != scoring_registration.EXPECTED_SCORING_PROTOCOL_SHA256
        or authority.protocol_introduction_commit
        != EXPECTED_PROTOCOL_INTRODUCTION_COMMIT
        or authority.registration_commit != EXPECTED_REGISTRATION_COMMIT
        or authority.registration_sha256 != EXPECTED_REGISTRATION_SHA256
        or authority.registration_remote_evidence_sha256
        != EXPECTED_REGISTRATION_REMOTE_EVIDENCE_SHA256
    ):
        raise shadow.ShadowPredictionError(
            "L'autorite scoring ne correspond pas au preenregistrement fige."
        )
    runtime_commit = shadow._require_git_commit(
        authority.runtime_code_commit,
        field="scoring.runtime_code_commit",
    )
    shadow_authority = shadow.verify_shadow_execution_authority(
        target,
        project_directory=project_directory,
    )
    if (
        type(shadow_authority) is not shadow.ShadowExecutionAuthority
        or shadow_authority.runtime_code_commit != runtime_commit
        or shadow_authority.shadow_protocol_sha256
        != shadow.EXPECTED_SHADOW_PROTOCOL_SHA256
    ):
        raise shadow.ShadowPredictionError(
            "Les autorites scoring et shadow ne designent pas le meme runtime."
        )
    protocol_authorities = authority.protocol.get("authorities")
    if type(protocol_authorities) is not dict:
        raise shadow.ShadowPredictionError(
            "Les autorites du protocole de scoring sont absentes."
        )
    protocol_shadow = protocol_authorities.get("shadow_protocol")
    protocol_manifest = protocol_authorities.get("execution_manifest")
    protocol_model = protocol_authorities.get("model")
    if (
        type(protocol_shadow) is not dict
        or type(protocol_manifest) is not dict
        or type(protocol_model) is not dict
        or protocol_shadow.get("sha256")
        != shadow_authority.shadow_protocol_sha256
        or protocol_manifest.get("sha256")
        != shadow_authority.execution_manifest_sha256
        or protocol_model.get("artifact_sha256")
        != shadow.EXPECTED_MODEL_ARTIFACT_SHA256
    ):
        raise shadow.ShadowPredictionError(
            "Le protocole de scoring et l'autorite shadow divergent."
        )
    return shadow_authority


def _git_history(
    project_directory: Path,
    arguments: tuple[str, ...],
    *,
    field_name: str,
) -> tuple[str, ...]:
    return certification._git_lines(
        project_directory,
        arguments,
        field=field_name,
    )


def _certification_paths_for_target(
    target_text: str,
) -> tuple[PurePosixPath, PurePosixPath]:
    raw_path, certification_path = certification._certification_relative_paths(
        target_text
    )
    return raw_path, certification_path


def _path_exists_without_reading(
    project_directory: Path,
    relative_path: PurePosixPath,
) -> bool:
    return shadow._lstat_mode(
        project_directory.joinpath(*relative_path.parts)
    ) is not shadow._PATH_MISSING


def _require_no_orphan_certification(
    project_directory: Path,
    target_text: str,
) -> None:
    raw_path, certification_path = _certification_paths_for_target(target_text)
    for relative in (certification_path, raw_path):
        history = _git_history(
            project_directory,
            ("log", "--format=%H", "--", relative.as_posix()),
            field_name=f"historique Git de {relative.as_posix()}",
        )
        if history or _path_exists_without_reading(project_directory, relative):
            raise shadow.ShadowPredictionError(
                "Une certification existe sans lot COMPLETED immuable."
            )


def _discover_result_commit(
    project_directory: Path,
    target_text: str,
    *,
    runtime_commit: str,
) -> tuple[ScoringPredictionSourceStatus, str | None]:
    result_root = shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH / target_text
    completed_path = result_root / shadow.COMPLETED_FILENAME
    additions = _git_history(
        project_directory,
        (
            "log",
            "--diff-filter=A",
            "--format=%H",
            "--reverse",
            "--",
            completed_path.as_posix(),
        ),
        field_name=f"introduction Git de {completed_path.as_posix()}",
    )
    root_history = _git_history(
        project_directory,
        ("log", "--format=%H", "--", result_root.as_posix()),
        field_name=f"historique Git de {result_root.as_posix()}",
    )
    if not additions:
        _require_no_orphan_certification(
            project_directory,
            target_text,
        )
        if root_history or _path_exists_without_reading(
            project_directory,
            result_root,
        ):
            return ScoringPredictionSourceStatus.FAILED, None
        return ScoringPredictionSourceStatus.MISSED, None
    if len(additions) != 1:
        raise shadow.ShadowPredictionError(
            "Le marqueur COMPLETED doit avoir un unique commit d'introduction."
        )
    results_commit = shadow._require_git_commit(
        additions[0],
        field="results_commit",
    )
    if root_history != (results_commit,):
        raise shadow.ShadowPredictionError(
            "INVALID_OR_MUTATED_RESULT_OR_CERTIFICATION_GIT_HISTORY"
        )
    shadow._require_git_ancestor(
        project_directory,
        results_commit,
        runtime_commit,
        strict=True,
        description="resultats certifiables vers runtime scoring",
    )
    return ScoringPredictionSourceStatus.CERTIFIED_NONEMPTY, results_commit


def _read_result_blobs(
    project_directory: Path,
    target_text: str,
    results_commit: str,
) -> dict[str, bytes]:
    result_root = shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH / target_text
    prefix = result_root.as_posix()
    expected_paths = tuple(
        f"{prefix}/{filename}"
        for filename in certification.EXPECTED_RESULT_FILENAMES
    )
    actual_paths = _git_history(
        project_directory,
        ("ls-tree", "-r", "--name-only", results_commit, "--", prefix),
        field_name="arbre Git immuable du lot shadow",
    )
    if actual_paths != expected_paths:
        raise shadow.ShadowPredictionError(
            "Le commit de resultats ne contient pas exactement les huit "
            "fichiers attendus."
        )
    return {
        filename: shadow._git_blob_at_commit(
            project_directory,
            results_commit,
            f"{prefix}/{filename}",
        )
        for filename in certification.EXPECTED_RESULT_FILENAMES
    }


def _validate_immutable_result_blobs(
    project_directory: Path,
    target_text: str,
    results_commit: str,
    shadow_authority: shadow.ShadowExecutionAuthority,
    blobs: Mapping[str, bytes],
) -> _ValidatedImmutableShadowBatch:
    """Valide les deux formes terminales, y compris le lot vide."""
    slot_key = shadow.build_slot_key(
        shadow_protocol_sha256=shadow_authority.shadow_protocol_sha256,
        target_official_date=target_text,
    )
    batch_id = shadow.build_batch_id(
        slot_key=slot_key,
        execution_manifest_sha256=(
            shadow_authority.execution_manifest_sha256
        ),
        model_artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
    )
    reserved, _reserved_bytes = certification._read_git_json_blob(
        blobs,
        "RESERVED",
    )
    receipt, receipt_bytes = certification._read_git_json_blob(
        blobs,
        shadow.RECEIPT_FILENAME,
    )
    completed, _completed_bytes = certification._read_git_json_blob(
        blobs,
        shadow.COMPLETED_FILENAME,
    )
    if not shadow._valid_reserved_marker(
        reserved,
        batch_id=batch_id,
        slot_key=slot_key,
        target_official_date=target_text,
        shadow_protocol_sha256=shadow_authority.shadow_protocol_sha256,
        execution_manifest_sha256=(
            shadow_authority.execution_manifest_sha256
        ),
    ):
        raise shadow.ShadowPredictionError(
            "Le marqueur RESERVED du commit de resultats est invalide."
        )
    if not shadow._valid_receipt(
        receipt,
        reserved=reserved,
        batch_id=batch_id,
        slot_key=slot_key,
        target_official_date=target_text,
        shadow_protocol_sha256=shadow_authority.shadow_protocol_sha256,
        execution_manifest_sha256=(
            shadow_authority.execution_manifest_sha256
        ),
        model_artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
    ):
        raise shadow.ShadowPredictionError(
            "Le recu du commit de resultats est invalide."
        )
    receipt_path = (
        shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH
        / target_text
        / shadow.RECEIPT_FILENAME
    ).as_posix()
    if not shadow._valid_completed_marker(
        completed,
        receipt=receipt,
        receipt_bytes=receipt_bytes,
        receipt_path=receipt_path,
        batch_id=batch_id,
    ):
        raise shadow.ShadowPredictionError(
            "Le marqueur COMPLETED du commit de resultats est invalide."
        )

    batch = receipt.get("batch")
    counts = receipt.get("counts")
    output_hashes = receipt.get("output_hashes")
    source = receipt.get("source")
    lineage = receipt.get("lineage")
    if any(
        type(section) is not dict
        for section in (batch, counts, output_hashes, source, lineage)
    ):
        raise shadow.ShadowPredictionError(
            "Les sections du recu Git sont invalides."
        )
    assert isinstance(batch, dict)
    assert isinstance(counts, dict)
    assert isinstance(output_hashes, dict)
    assert isinstance(source, dict)
    assert isinstance(lineage, dict)
    predicted_count = counts.get("predicted_games")
    if type(predicted_count) is not int or predicted_count < 0:
        raise shadow.ShadowPredictionError(
            "Le nombre de predictions Git est invalide."
        )

    for receipt_field, filename in certification._OUTPUT_HASH_TO_FILENAME.items():
        content = blobs.get(filename)
        if (
            type(content) is not bytes
            or output_hashes.get(receipt_field)
            != hashlib.sha256(content).hexdigest()
        ):
            raise shadow.ShadowPredictionError(
                f"L'empreinte Git de {filename} diverge du recu."
            )
    source_snapshot = blobs.get(shadow.SOURCE_SNAPSHOT_FILENAME)
    if (
        type(source_snapshot) is not bytes
        or source.get("source_snapshot_sha256")
        != hashlib.sha256(source_snapshot).hexdigest()
    ):
        raise shadow.ShadowPredictionError(
            "L'empreinte Git du snapshot source diverge du recu."
        )
    predictions_bytes = blobs.get(shadow.PREDICTIONS_FILENAME)
    if type(predictions_bytes) is not bytes:
        raise shadow.ShadowPredictionError(
            "Le blob Git predictions.csv est absent."
        )
    expected_status = (
        "COMPLETED_WITH_PREDICTIONS"
        if predicted_count > 0
        else "COMPLETED_NO_ELIGIBLE_GAMES"
    )
    if batch.get("status") != expected_status:
        raise shadow.ShadowPredictionError(
            "Le statut du lot Git diverge de son nombre de predictions."
        )
    if predicted_count:
        earliest = certification._read_canonical_predictions_blob(
            predictions_bytes,
            batch_id=batch_id,
            target_official_date=target_text,
            expected_row_count=predicted_count,
        )
    else:
        expected_empty = shadow._canonical_csv_bytes(
            shadow._PREDICTIONS_COLUMNS,
            [],
        )
        if predictions_bytes != expected_empty:
            raise shadow.ShadowPredictionError(
                "Le lot vide doit contenir seulement l'en-tete canonique."
            )
        earliest = None
    if batch.get("earliest_predicted_scheduled_start_utc") != earliest:
        raise shadow.ShadowPredictionError(
            "Le premier horaire Git diverge du recu terminal."
        )
    runtime_commit = shadow._require_git_commit(
        lineage.get("runtime_code_commit"),
        field="receipt.lineage.runtime_code_commit",
    )
    shadow._require_git_ancestor(
        project_directory,
        runtime_commit,
        results_commit,
        strict=True,
        description="execution de prediction vers commit de resultats",
    )
    tree_hashes = tuple(
        {
            "path": filename,
            "sha256": hashlib.sha256(blobs[filename]).hexdigest(),
            "size_bytes": len(blobs[filename]),
        }
        for filename in certification.EXPECTED_RESULT_FILENAMES
    )
    return _ValidatedImmutableShadowBatch(
        target_official_date=target_text,
        results_commit=results_commit,
        batch_id=batch_id,
        status=expected_status,
        earliest_predicted_start_utc=earliest,
        results_tree_file_hashes=tree_hashes,
        receipt=receipt,
        receipt_bytes=receipt_bytes,
        predictions_bytes=predictions_bytes,
    )


def _parse_canonical_positive_integer(
    value: object,
    *,
    field_name: str,
) -> int:
    if type(value) is not str or not re.fullmatch(r"[1-9][0-9]*", value):
        raise shadow.ShadowPredictionError(
            f"{field_name} doit etre un entier decimal canonique positif."
        )
    return int(value)


def _read_certified_predictions_from_blob(
    batch: _ValidatedImmutableShadowBatch,
) -> tuple[CertifiedScoringPrediction, ...]:
    try:
        decoded = batch.predictions_bytes.decode("utf-8")
        parsed = list(csv.reader(io.StringIO(decoded, newline="")))
    except (UnicodeError, csv.Error) as error:
        raise shadow.ShadowPredictionError(
            "Le blob Git predictions.csv est illisible."
        ) from error
    if not parsed or tuple(parsed[0]) != shadow._PREDICTIONS_COLUMNS:
        raise shadow.ShadowPredictionError(
            "Le schema Git de predictions.csv est invalide."
        )
    rows = parsed[1:]
    if (
        shadow._canonical_csv_bytes(shadow._PREDICTIONS_COLUMNS, rows)
        != batch.predictions_bytes
    ):
        raise shadow.ShadowPredictionError(
            "Les octets Git de predictions.csv ne sont pas canoniques."
        )
    receipt_lineage = batch.receipt.get("lineage")
    if type(receipt_lineage) is not dict:
        raise shadow.ShadowPredictionError(
            "La provenance du recu Git est invalide."
        )
    code_commit = shadow._require_git_commit(
        receipt_lineage.get("runtime_code_commit"),
        field="receipt.lineage.runtime_code_commit",
    )
    prediction_ids: set[str] = set()
    game_ids: set[int] = set()
    previous_order: tuple[str, int] | None = None
    predictions: list[CertifiedScoringPrediction] = []
    target = date.fromisoformat(batch.target_official_date)
    for row_number, row in enumerate(rows, start=2):
        if len(row) != len(shadow._PREDICTIONS_COLUMNS):
            raise shadow.ShadowPredictionError(
                f"Largeur invalide de predictions.csv a la ligne {row_number}."
            )
        prediction_id = shadow._require_sha256(
            row[0],
            field=f"predictions[{row_number}].prediction_id",
        )
        occurrence_key = shadow._require_sha256(
            row[3],
            field=f"predictions[{row_number}].occurrence_key",
        )
        game_id = _parse_canonical_positive_integer(
            row[2],
            field_name=f"predictions[{row_number}].game_id",
        )
        season = _parse_canonical_positive_integer(
            row[4],
            field_name=f"predictions[{row_number}].season",
        )
        away_team_id = _parse_canonical_positive_integer(
            row[6],
            field_name=f"predictions[{row_number}].away_team_id",
        )
        home_team_id = _parse_canonical_positive_integer(
            row[7],
            field_name=f"predictions[{row_number}].home_team_id",
        )
        scheduled_start = shadow._require_utc_timestamp(
            row[8],
            field=f"predictions[{row_number}].scheduled_start_utc",
        )
        shadow._require_utc_timestamp(
            row[9],
            field=f"predictions[{row_number}].information_cutoff_utc",
        )
        shadow._require_utc_timestamp(
            row[10],
            field=f"predictions[{row_number}].issued_at_utc",
        )
        official_date = shadow._require_date_string(
            row[5],
            field=f"predictions[{row_number}].official_date",
        )
        feature_as_of = shadow._require_date_string(
            row[11],
            field=f"predictions[{row_number}].feature_as_of_date",
        )
        away_source_date = shadow._require_date_string(
            row[12],
            field=f"predictions[{row_number}].away_max_source_date",
        )
        home_source_date = shadow._require_date_string(
            row[13],
            field=f"predictions[{row_number}].home_max_source_date",
        )
        try:
            home_float = float(row[22])
            away_float = float(row[23])
            home_decimal = _parse_probability_text(
                row[22],
                field_name=f"predictions[{row_number}].p_home_win",
            )
            away_decimal = _parse_probability_text(
                row[23],
                field_name=f"predictions[{row_number}].p_away_win",
            )
        except (ValueError, ScoringAdjudicationError) as error:
            raise shadow.ShadowPredictionError(
                "Une probabilite Git certifiee est invalide."
            ) from error
        if (
            row[1] != batch.batch_id
            or season != 2026
            or official_date != batch.target_official_date
            or feature_as_of != (target - timedelta(days=1)).isoformat()
            or away_source_date >= official_date
            or home_source_date >= official_date
            or away_team_id == home_team_id
            or row[24] != shadow.EXPECTED_CALIBRATED_MODEL_VERSION
            or row[25] != shadow.EXPECTED_MODEL_ARTIFACT_SHA256
            or row[26] != shadow.EXPECTED_SHADOW_PROTOCOL_SHA256
            or row[27] != code_commit
            or prediction_id
            != shadow.build_prediction_id(
                shadow_protocol_sha256=shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
                game_id=game_id,
                official_date_at_snapshot=official_date,
                scheduled_start_utc_at_snapshot=scheduled_start,
            )
            or occurrence_key
            != shadow.build_occurrence_key(
                game_id=game_id,
                official_date_at_snapshot=official_date,
                scheduled_start_utc_at_snapshot_or_null=scheduled_start,
            )
            or shadow._format_probability_float(home_float) != row[22]
            or shadow._format_probability_float(away_float) != row[23]
            or away_float != 1.0 - home_float
            or home_decimal + away_decimal != Decimal("1")
        ):
            raise shadow.ShadowPredictionError(
                "Une ligne Git certifiee diverge du contrat shadow v2."
            )
        order_key = (scheduled_start, game_id)
        if (
            prediction_id in prediction_ids
            or game_id in game_ids
            or (previous_order is not None and order_key <= previous_order)
        ):
            raise shadow.ShadowPredictionError(
                "Les predictions Git sont dupliquees ou hors ordre canonique."
            )
        prediction_ids.add(prediction_id)
        game_ids.add(game_id)
        previous_order = order_key
        predictions.append(
            CertifiedScoringPrediction(
                prediction_id=prediction_id,
                batch_id=batch.batch_id,
                game_id=game_id,
                occurrence_key=occurrence_key,
                season=season,
                target_official_date=official_date,
                away_team_id=away_team_id,
                home_team_id=home_team_id,
                p_home_win=row[22],
                p_away_win=row[23],
            )
        )
    expected_count = batch.receipt["counts"]["predicted_games"]
    if len(predictions) != expected_count:
        raise shadow.ShadowPredictionError(
            "Le nombre de predictions certifiees diverge du recu."
        )
    return tuple(predictions)


def _read_and_validate_certification(
    project_directory: Path,
    batch: _ValidatedImmutableShadowBatch,
    *,
    runtime_commit: str,
) -> tuple[str, str, str]:
    raw_path, certification_path = _certification_paths_for_target(
        batch.target_official_date
    )
    certification_additions = _git_history(
        project_directory,
        (
            "log",
            "--diff-filter=A",
            "--format=%H",
            "--reverse",
            "--",
            certification_path.as_posix(),
        ),
        field_name="introduction Git de la certification",
    )
    raw_additions = _git_history(
        project_directory,
        (
            "log",
            "--diff-filter=A",
            "--format=%H",
            "--reverse",
            "--",
            raw_path.as_posix(),
        ),
        field_name="introduction Git de la preuve de certification",
    )
    if not certification_additions and not raw_additions:
        if _path_exists_without_reading(
            project_directory,
            certification_path,
        ) or _path_exists_without_reading(project_directory, raw_path):
            raise shadow.ShadowPredictionError(
                "Une certification non versionnee constitue un conflit structurel."
            )
        return "", "", ""
    if (
        len(certification_additions) != 1
        or raw_additions != certification_additions
    ):
        raise shadow.ShadowPredictionError(
            "INVALID_OR_MUTATED_RESULT_OR_CERTIFICATION_GIT_HISTORY"
        )
    certification_commit = shadow._require_git_commit(
        certification_additions[0],
        field="certification_commit",
    )
    for relative in (certification_path, raw_path):
        history = _git_history(
            project_directory,
            ("log", "--format=%H", "--", relative.as_posix()),
            field_name=f"historique Git de {relative.as_posix()}",
        )
        if history != (certification_commit,):
            raise shadow.ShadowPredictionError(
                "INVALID_OR_MUTATED_RESULT_OR_CERTIFICATION_GIT_HISTORY"
            )
    shadow._require_git_ancestor(
        project_directory,
        batch.results_commit,
        certification_commit,
        strict=True,
        description="resultats vers certification prospective",
    )
    shadow._require_git_ancestor(
        project_directory,
        certification_commit,
        runtime_commit,
        strict=True,
        description="certification prospective vers runtime scoring",
    )
    certification_bytes = shadow._git_blob_at_commit(
        project_directory,
        certification_commit,
        certification_path.as_posix(),
    )
    certification_document = shadow._read_canonical_json_bytes(
        certification_bytes,
        description="certification prospective immuable",
    )
    if certification_document.get("raw_remote_evidence_path") != raw_path.as_posix():
        raise shadow.ShadowPredictionError(
            "La certification designe un chemin de preuve distante inattendu."
        )
    raw_bytes = shadow._git_blob_at_commit(
        project_directory,
        certification_commit,
        raw_path.as_posix(),
    )
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if certification_document.get("raw_remote_evidence_sha256") != raw_sha256:
        raise shadow.ShadowPredictionError(
            "L'empreinte de la preuve distante certifiee diverge."
        )
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes), mode="rb") as archive:
            raw_json_bytes = archive.read()
    except (OSError, EOFError, zlib.error) as error:
        raise shadow.ShadowPredictionError(
            "La preuve distante certifiee est illisible."
        ) from error
    raw_document = shadow._read_canonical_json_bytes(
        raw_json_bytes,
        description="preuve distante certifiee immuable",
    )
    if shadow._canonical_gzip_bytes(raw_json_bytes) != raw_bytes:
        raise shadow.ShadowPredictionError(
            "La preuve distante certifiee n'est pas un gzip canonique."
        )
    evidence = shadow.ShadowActivationReverificationEvidence(
        activation_introduction_commit=batch.results_commit,
        activation_remote_ref=certification_document.get(
            "results_remote_ref"
        ),
        activation_remote_reverified_at_utc=certification_document.get(
            "remote_http_date_utc"
        ),
        response_received_at_utc=certification_document.get(
            "remote_response_received_at_utc"
        ),
        response_body_sha256=certification_document.get(
            "remote_response_body_sha256"
        ),
        raw_evidence=raw_document,
        canonical_json_bytes=raw_json_bytes,
        canonical_gzip_bytes=raw_bytes,
        canonical_gzip_sha256=raw_sha256,
    )
    completed = certification.CompletedShadowBatchCommit(
        target_official_date=batch.target_official_date,
        results_commit=batch.results_commit,
        batch_id=batch.batch_id,
        earliest_predicted_start_utc=(
            batch.earliest_predicted_start_utc or ""
        ),
        results_tree_file_hashes=batch.results_tree_file_hashes,
        receipt=batch.receipt,
    )
    validated_bytes = certification._validate_prepared_certification(
        certification_document,
        batch=completed,
        evidence=evidence,
        raw_evidence_relative_path=raw_path.as_posix(),
    )
    if validated_bytes != certification_bytes:
        raise shadow.ShadowPredictionError(
            "Les octets de certification Git divergent du document valide."
        )
    return (
        certification_commit,
        hashlib.sha256(certification_bytes).hexdigest(),
        raw_sha256,
    )


def _validate_first_prediction_source_anchor(
    source: ImmutableScoringPredictionSource,
) -> None:
    if source.target_official_date != scoring_registration.EXPECTED_FIRST_TARGET_DATE:
        return
    expected = {
        "status": ScoringPredictionSourceStatus.CERTIFIED_NONEMPTY,
        "results_commit": scoring_registration.EXPECTED_FIRST_RESULTS_COMMIT,
        "certification_commit": (
            scoring_registration.EXPECTED_FIRST_CERTIFICATION_COMMIT
        ),
        "batch_id": scoring_registration.EXPECTED_FIRST_BATCH_ID,
        "predictions_sha256": (
            scoring_registration.EXPECTED_FIRST_PREDICTIONS_SHA256
        ),
        "receipt_sha256": scoring_registration.EXPECTED_FIRST_RECEIPT_SHA256,
        "certification_sha256": (
            scoring_registration.EXPECTED_FIRST_CERTIFICATION_SHA256
        ),
    }
    for field_name, expected_value in expected.items():
        if getattr(source, field_name) != expected_value:
            raise shadow.ShadowPredictionError(
                "Le premier lot diverge de l'ancre prospectivement preenregistree."
            )


def load_immutable_scoring_prediction_source(
    authority: ScoringExecutionAuthority,
    target_official_date: date | str,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ImmutableScoringPredictionSource:
    """Charge une cohorte seulement depuis ses commits Git certifies."""
    project = shadow._preflight_project_directory(project_directory)
    target = _parse_horizon_target(target_official_date)
    target_text = target.isoformat()
    shadow_authority = _require_prediction_source_authorities(
        authority,
        target,
        project_directory=project,
    )
    discovery_status, results_commit = _discover_result_commit(
        project,
        target_text,
        runtime_commit=authority.runtime_code_commit,
    )
    if results_commit is None:
        source = ImmutableScoringPredictionSource(
            target_official_date=target_text,
            status=discovery_status,
            results_commit=None,
            certification_commit=None,
            batch_id=None,
            predictions=(),
            predictions_sha256=None,
            receipt_sha256=None,
            certification_sha256=None,
            raw_certification_evidence_sha256=None,
            status_reason=(
                "RESULT_ROOT_PRESENT_OR_HISTORIC_WITHOUT_COMPLETED"
                if discovery_status is ScoringPredictionSourceStatus.FAILED
                else "NO_RESULT_ROOT_EVER_PUBLISHED"
            ),
        )
        _validate_first_prediction_source_anchor(source)
        return source

    blobs = _read_result_blobs(
        project,
        target_text,
        results_commit,
    )
    batch = _validate_immutable_result_blobs(
        project,
        target_text,
        results_commit,
        shadow_authority,
        blobs,
    )
    predictions = _read_certified_predictions_from_blob(batch)
    predictions_sha256 = hashlib.sha256(batch.predictions_bytes).hexdigest()
    receipt_sha256 = hashlib.sha256(batch.receipt_bytes).hexdigest()
    if batch.status == "COMPLETED_NO_ELIGIBLE_GAMES":
        _require_no_orphan_certification(project, target_text)
        source = ImmutableScoringPredictionSource(
            target_official_date=target_text,
            status=ScoringPredictionSourceStatus.COMPLETED_EMPTY,
            results_commit=results_commit,
            certification_commit=None,
            batch_id=batch.batch_id,
            predictions=(),
            predictions_sha256=predictions_sha256,
            receipt_sha256=receipt_sha256,
            certification_sha256=None,
            raw_certification_evidence_sha256=None,
            status_reason="VALID_COMPLETED_NO_ELIGIBLE_GAMES",
        )
        _validate_first_prediction_source_anchor(source)
        return source

    certification_commit, certification_sha256, raw_sha256 = (
        _read_and_validate_certification(
            project,
            batch,
            runtime_commit=authority.runtime_code_commit,
        )
    )
    if not certification_commit:
        source = ImmutableScoringPredictionSource(
            target_official_date=target_text,
            status=ScoringPredictionSourceStatus.LOCAL_ONLY,
            results_commit=results_commit,
            certification_commit=None,
            batch_id=batch.batch_id,
            predictions=(),
            predictions_sha256=predictions_sha256,
            receipt_sha256=receipt_sha256,
            certification_sha256=None,
            raw_certification_evidence_sha256=None,
            status_reason="VALID_NONEMPTY_BATCH_WITHOUT_CERTIFICATION",
        )
        _validate_first_prediction_source_anchor(source)
        return source
    source = ImmutableScoringPredictionSource(
        target_official_date=target_text,
        status=ScoringPredictionSourceStatus.CERTIFIED_NONEMPTY,
        results_commit=results_commit,
        certification_commit=certification_commit,
        batch_id=batch.batch_id,
        predictions=predictions,
        predictions_sha256=predictions_sha256,
        receipt_sha256=receipt_sha256,
        certification_sha256=certification_sha256,
        raw_certification_evidence_sha256=raw_sha256,
        status_reason="VALID_IMMUTABLE_CERTIFIED_NONEMPTY_BATCH",
    )
    _validate_first_prediction_source_anchor(source)
    return source


def _require_local_directory(path: Path, *, description: str) -> None:
    mode = shadow._lstat_mode(path)
    if not isinstance(mode, int) or not stat.S_ISDIR(mode):
        raise shadow.ShadowPredictionError(
            f"{description} doit etre un repertoire local non symbolique."
        )


def _create_or_require_local_directory(path: Path, *, description: str) -> None:
    try:
        path.mkdir(exist_ok=True)
    except OSError as error:
        raise shadow.ShadowPredictionError(
            f"Impossible de creer {description}."
        ) from error
    _require_local_directory(path, description=description)
    shadow._fsync_parent_directory(path.parent)


def reserve_scoring_observation_slot(
    authority: ScoringExecutionAuthority,
    target_official_date: date | str,
    checkpoint_utc_date: date | str,
    *,
    reserved_at_utc: str,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ScoringObservationReservation:
    """Reserve une observation une fois, sans horloge, reseau ou resultat."""
    if type(authority) is not ScoringExecutionAuthority:
        raise shadow.ShadowPredictionError(
            "Une autorite d'execution scoring exacte est requise."
        )
    if (
        authority.scoring_protocol_sha256
        != scoring_registration.EXPECTED_SCORING_PROTOCOL_SHA256
        or authority.protocol_introduction_commit
        != EXPECTED_PROTOCOL_INTRODUCTION_COMMIT
        or authority.registration_commit != EXPECTED_REGISTRATION_COMMIT
        or authority.registration_sha256 != EXPECTED_REGISTRATION_SHA256
        or authority.registration_remote_evidence_sha256
        != EXPECTED_REGISTRATION_REMOTE_EVIDENCE_SHA256
    ):
        raise shadow.ShadowPredictionError(
            "L'autorite d'execution scoring ne correspond pas aux preuves figees."
        )
    runtime_commit = shadow._require_git_commit(
        authority.runtime_code_commit,
        field="runtime_code_commit",
    )
    target = _parse_horizon_target(target_official_date)
    checkpoint = _parse_checkpoint_date(checkpoint_utc_date, target=target)
    reserved_at = _parse_utc_timestamp(
        reserved_at_utc,
        field_name="reserved_at_utc",
    )
    if reserved_at.date() != checkpoint:
        raise shadow.ShadowPredictionError(
            "Un checkpoint manque ne peut pas etre recree retrospectivement."
        )
    if reserved_at < _checkpoint_instant(checkpoint):
        raise shadow.ShadowPredictionError(
            "Le creneau d'observation ne peut pas etre reserve avant son checkpoint."
        )

    project = shadow._preflight_project_directory(project_directory)
    inspected = inspect_scoring_observation_slot_presence_first(
        target,
        checkpoint,
        project_directory=project,
    )
    if inspected.state is not ScoringObservationSlotState.ABSENT:
        raise shadow.ShadowPredictionSlotConsumedError(
            "Le creneau d'observation est deja consomme et ne peut jamais "
            "etre ecrase, repare ou recree."
        )
    marker: dict[str, Any] = {
        "marker_schema_version": 1,
        "protocol_id": scoring_registration.EXPECTED_PROTOCOL_ID,
        "scoring_protocol_sha256": authority.scoring_protocol_sha256,
        "target_official_date": inspected.target_official_date,
        "checkpoint_utc_date": inspected.checkpoint_utc_date,
        "observation_id": inspected.observation_id,
        "reserved_at_utc": reserved_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_code_commit": runtime_commit,
    }
    if frozenset(marker) != _RESERVED_MARKER_KEYS:
        raise shadow.ShadowPredictionError("Le marqueur RESERVED est incomplet.")
    marker_bytes = shadow._canonical_json_file_bytes(marker)

    output_root = project.joinpath(*SCORING_OUTPUT_ROOT_RELATIVE_PATH.parts)
    _require_local_directory(output_root, description="racine de scoring")
    target_root = output_root / target.isoformat()
    _create_or_require_local_directory(
        target_root,
        description="repertoire de date scoring",
    )
    observations_root = target_root / "observations"
    _create_or_require_local_directory(
        observations_root,
        description="repertoire des observations",
    )
    if inspected.slot_path.parent != observations_root:
        raise shadow.ShadowPredictionError(
            "Le chemin derive du creneau d'observation est incoherent."
        )
    try:
        inspected.slot_path.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise shadow.ShadowPredictionSlotConsumedError(
            "Une autre execution a deja reserve ce creneau d'observation."
        ) from error
    except OSError as error:
        raise shadow.ShadowPredictionError(
            "Impossible de creer atomiquement le creneau d'observation."
        ) from error

    # Des ce mkdir, le creneau est definitivement consomme, meme si la suite
    # echoue. Aucun nettoyage ou mecanisme de reparation n'est autorise.
    _require_local_directory(
        inspected.slot_path,
        description="creneau d'observation reserve",
    )
    shadow._fsync_parent_directory(observations_root)
    marker_sha256 = shadow._publish_exclusive_verified(
        inspected.slot_path / "RESERVED",
        marker_bytes,
    )
    return ScoringObservationReservation(
        slot_path=inspected.slot_path,
        target_official_date=inspected.target_official_date,
        checkpoint_utc_date=inspected.checkpoint_utc_date,
        observation_id=inspected.observation_id,
        reserved_marker=marker,
        reserved_marker_sha256=marker_sha256,
    )


def inspect_scoring_observation_slot(
    authority: ScoringExecutionAuthority,
    target_official_date: date | str,
    checkpoint_utc_date: date | str,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ScoringObservationSlotInspection:
    """Reconnait uniquement ABSENT ou un RESERVED exact; tout autre est consomme."""
    if type(authority) is not ScoringExecutionAuthority:
        raise shadow.ShadowPredictionError(
            "Une autorite d'execution scoring exacte est requise."
        )
    basic = inspect_scoring_observation_slot_presence_first(
        target_official_date,
        checkpoint_utc_date,
        project_directory=project_directory,
    )
    if basic.state is ScoringObservationSlotState.ABSENT:
        return basic
    mode = shadow._lstat_mode(basic.slot_path)
    if not isinstance(mode, int) or not stat.S_ISDIR(mode):
        return basic
    try:
        names = sorted(path.name for path in basic.slot_path.iterdir())
    except OSError:
        return basic
    if names != ["RESERVED"]:
        return basic
    result = shadow._read_canonical_json_object(basic.slot_path / "RESERVED")
    if result is None:
        return basic
    marker, _ = result
    expected = {
        "marker_schema_version": 1,
        "protocol_id": scoring_registration.EXPECTED_PROTOCOL_ID,
        "scoring_protocol_sha256": authority.scoring_protocol_sha256,
        "target_official_date": basic.target_official_date,
        "checkpoint_utc_date": basic.checkpoint_utc_date,
        "observation_id": basic.observation_id,
        "reserved_at_utc": marker.get("reserved_at_utc"),
        "runtime_code_commit": authority.runtime_code_commit,
    }
    try:
        reserved_at = _parse_utc_timestamp(
            marker.get("reserved_at_utc"),
            field_name="RESERVED.reserved_at_utc",
        )
    except shadow.ShadowPredictionError:
        return basic
    checkpoint = date.fromisoformat(basic.checkpoint_utc_date)
    if (
        frozenset(marker) != _RESERVED_MARKER_KEYS
        or marker != expected
        or reserved_at.date() != checkpoint
        or reserved_at < _checkpoint_instant(checkpoint)
    ):
        return basic
    return ScoringObservationSlotInspection(
        state=ScoringObservationSlotState.RESERVED_EXACT,
        slot_path=basic.slot_path,
        target_official_date=basic.target_official_date,
        checkpoint_utc_date=basic.checkpoint_utc_date,
        observation_id=basic.observation_id,
        reserved_marker=marker,
    )


def _outcome_query_parameters(target_date: str) -> dict[str, object]:
    return {
        "sportId": 1,
        "startDate": target_date,
        "endDate": target_date,
        "gameTypes": "R",
        "hydrate": "probablePitcher",
    }


def _expected_outcome_request_url(target_date: str) -> str:
    try:
        prepared = requests.Request(
            "GET",
            MLB_SCHEDULE_URL,
            params=_outcome_query_parameters(target_date),
            headers=OUTCOME_REQUEST_HEADERS,
        ).prepare()
    except requests.RequestException as error:
        raise ScoringOutcomeRequestError(
            "Impossible de preparer la requete MLB exacte du scoring."
        ) from error
    if type(prepared.url) is not str or not prepared.url:
        raise ScoringOutcomeRequestError(
            "L'URL MLB preparee du scoring est absente."
        )
    return prepared.url


def _decode_and_validate_outcome_payload(
    content: bytes,
) -> tuple[dict[str, Any], int]:
    if type(content) is not bytes or not content:
        raise ScoringOutcomeValidationError(
            "MLB a renvoye un corps vide ou non binaire."
        )
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=shadow._reject_duplicate_keys,
        )
    except (
        UnicodeError,
        ValueError,
        shadow.ShadowPredictionError,
        RecursionError,
    ) as error:
        raise ScoringOutcomeValidationError(
            "La reponse MLB n'est pas un objet JSON non ambigu."
        ) from error
    if type(payload) is not dict:
        raise ScoringOutcomeValidationError(
            "La reponse MLB doit etre un objet JSON."
        )
    date_blocks = payload.get("dates")
    if type(date_blocks) is not list:
        raise ScoringOutcomeValidationError(
            "La reponse MLB doit contenir un tableau dates."
        )
    occurrences = 0
    for date_index, date_block in enumerate(date_blocks):
        if type(date_block) is not dict:
            raise ScoringOutcomeValidationError(
                f"Le bloc MLB dates[{date_index}] est invalide."
            )
        games = date_block.get("games")
        if type(games) is not list:
            raise ScoringOutcomeValidationError(
                f"Le bloc MLB dates[{date_index}].games est invalide."
            )
        if any(type(game) is not dict for game in games):
            raise ScoringOutcomeValidationError(
                f"Un match MLB de dates[{date_index}] est invalide."
            )
        occurrences += len(games)
    if "totalGames" in payload:
        reported = payload["totalGames"]
        if type(reported) is not int or reported != occurrences:
            raise ScoringOutcomeValidationError(
                "Le totalGames MLB ne correspond pas aux occurrences brutes."
            )
    return payload, occurrences


def _validate_outcome_evidence(
    evidence: ScoringOutcomeObservationEvidence,
) -> tuple[dict[str, Any], bytes, bytes]:
    if type(evidence) is not ScoringOutcomeObservationEvidence:
        raise shadow.ShadowPredictionError(
            "Une preuve MLB d'observation exacte est requise."
        )
    target = _parse_horizon_target(evidence.target_official_date)
    checkpoint = _parse_checkpoint_date(
        evidence.checkpoint_utc_date,
        target=target,
    )
    http_date = shadow._require_utc_timestamp(
        evidence.outcome_http_date_utc,
        field="outcome_http_date_utc",
    )
    received_at = shadow._require_utc_timestamp(
        evidence.response_received_at_utc,
        field="response_received_at_utc",
    )
    body_sha256 = shadow._require_sha256(
        evidence.response_body_sha256,
        field="response_body_sha256",
    )
    gzip_sha256 = shadow._require_sha256(
        evidence.canonical_gzip_sha256,
        field="canonical_gzip_sha256",
    )
    if type(evidence.flattened_occurrence_count) is not int:
        raise shadow.ShadowPredictionError(
            "Le nombre d'occurrences MLB doit etre un entier exact."
        )
    raw = evidence.raw_evidence
    if type(raw) is not dict or frozenset(raw) != _OUTCOME_EVIDENCE_KEYS:
        raise shadow.ShadowPredictionError(
            "La preuve MLB ne respecte pas son schema exact."
        )
    expected_url = _expected_outcome_request_url(target.isoformat())
    exact_values: dict[str, object] = {
        "evidence_schema_version": 1,
        "target_official_date": target.isoformat(),
        "checkpoint_utc_date": checkpoint.isoformat(),
        "request_url": expected_url,
        "request_method": "GET",
        "application_request_headers": dict(OUTCOME_REQUEST_HEADERS),
        "effective_url": expected_url,
        "response_status_code": 200,
        "response_redirect_count": 0,
        "response_received_at_utc": received_at,
        "response_body_sha256": body_sha256,
    }
    for key, expected in exact_values.items():
        actual = raw.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise shadow.ShadowPredictionError(
                f"Valeur de preuve MLB invalide pour {key}."
            )
    headers = raw.get("selected_response_headers")
    if type(headers) is not dict or frozenset(headers) != _OUTCOME_SELECTED_HEADER_KEYS:
        raise shadow.ShadowPredictionError(
            "Les en-tetes MLB selectionnes sont invalides."
        )
    raw_date = headers.get("date")
    if type(raw_date) is not str or not raw_date:
        raise shadow.ShadowPredictionError(
            "L'en-tete Date MLB est obligatoire."
        )
    if shadow._parse_imf_fixdate_gmt(
        raw_date,
        field="selected_response_headers.date",
    ) != http_date:
        raise shadow.ShadowPredictionError(
            "L'en-tete Date MLB diverge de l'instant observe."
        )
    content_type = headers.get("content-type")
    if content_type is not None and (
        type(content_type) is not str or not content_type
    ):
        raise shadow.ShadowPredictionError(
            "L'en-tete Content-Type MLB est invalide."
        )
    encoded = raw.get("response_body_base64")
    if type(encoded) is not str or not encoded:
        raise shadow.ShadowPredictionError(
            "Le corps MLB base64 est absent."
        )
    try:
        body = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeError, ValueError) as error:
        raise shadow.ShadowPredictionError(
            "Le corps MLB base64 est invalide."
        ) from error
    if base64.b64encode(body).decode("ascii") != encoded:
        raise shadow.ShadowPredictionError(
            "Le corps MLB n'utilise pas un base64 canonique."
        )
    if hashlib.sha256(body).hexdigest() != body_sha256:
        raise shadow.ShadowPredictionError(
            "L'empreinte du corps MLB est incoherente."
        )
    _, occurrences = _decode_and_validate_outcome_payload(body)
    if occurrences != evidence.flattened_occurrence_count:
        raise shadow.ShadowPredictionError(
            "Le nombre d'occurrences de la preuve MLB diverge."
        )
    canonical_json = shadow._canonical_json_file_bytes(raw)
    canonical_gzip = shadow._canonical_gzip_bytes(canonical_json)
    if (
        type(evidence.canonical_json_bytes) is not bytes
        or evidence.canonical_json_bytes != canonical_json
        or type(evidence.canonical_gzip_bytes) is not bytes
        or evidence.canonical_gzip_bytes != canonical_gzip
        or hashlib.sha256(canonical_gzip).hexdigest() != gzip_sha256
    ):
        raise shadow.ShadowPredictionError(
            "Les octets canoniques de la preuve MLB sont incoherents."
        )
    return raw, body, canonical_gzip


def fetch_scoring_outcome_evidence(
    target_official_date: date | str,
    checkpoint_utc_date: date | str,
) -> ScoringOutcomeObservationEvidence:
    """Effectue exactement une requete MLB et prepare sa preuve en memoire."""
    target = _parse_horizon_target(target_official_date)
    checkpoint = _parse_checkpoint_date(checkpoint_utc_date, target=target)
    target_text = target.isoformat()
    request_url = _expected_outcome_request_url(target_text)
    try:
        response = requests.get(
            MLB_SCHEDULE_URL,
            params=_outcome_query_parameters(target_text),
            headers=OUTCOME_REQUEST_HEADERS,
            timeout=OUTCOME_REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
            verify=True,
        )
    except requests.RequestException as error:
        raise ScoringOutcomeRequestError(
            f"Impossible de joindre MLB pour l'observation : {error}"
        ) from error
    received_at = _format_utc_seconds(
        _utc_now(),
        field_name="response_received_at_utc",
    )
    if response.status_code != 200:
        raise ScoringOutcomeRequestError(
            "MLB doit repondre exactement avec le statut HTTP 200."
        )
    if len(response.history) != 0:
        raise ScoringOutcomeRequestError(
            "La requete MLB de scoring ne doit suivre aucune redirection."
        )
    if response.url != request_url:
        raise ScoringOutcomeRequestError(
            "L'URL MLB effective diverge de la requete exacte."
        )
    raw_http_date = response.headers.get("Date")
    if type(raw_http_date) is not str or not raw_http_date:
        raise ScoringOutcomeValidationError(
            "La reponse MLB doit fournir un en-tete Date."
        )
    try:
        http_date = shadow._parse_imf_fixdate_gmt(
            raw_http_date,
            field="MLB.Date",
        )
        http_datetime = datetime.fromisoformat(
            http_date.replace("Z", "+00:00")
        )
        received_datetime = datetime.fromisoformat(
            received_at.replace("Z", "+00:00")
        )
    except shadow.ShadowPredictionError as error:
        raise ScoringOutcomeValidationError(str(error)) from error
    if abs((received_datetime - http_datetime).total_seconds()) > 300:
        raise ScoringOutcomeValidationError(
            "L'horloge locale et la Date MLB divergent de plus de 300 secondes."
        )
    content = response.content
    _, occurrences = _decode_and_validate_outcome_payload(content)
    body_sha256 = hashlib.sha256(content).hexdigest()
    content_type = response.headers.get("Content-Type")
    if content_type is not None and (
        type(content_type) is not str or not content_type
    ):
        raise ScoringOutcomeValidationError(
            "Le Content-Type MLB est invalide."
        )
    raw: dict[str, Any] = {
        "evidence_schema_version": 1,
        "target_official_date": target_text,
        "checkpoint_utc_date": checkpoint.isoformat(),
        "request_url": request_url,
        "request_method": "GET",
        "application_request_headers": dict(OUTCOME_REQUEST_HEADERS),
        "effective_url": response.url,
        "response_status_code": response.status_code,
        "response_redirect_count": len(response.history),
        "selected_response_headers": {
            "date": raw_http_date,
            "content-type": content_type,
        },
        "response_received_at_utc": received_at,
        "response_body_base64": base64.b64encode(content).decode("ascii"),
        "response_body_sha256": body_sha256,
    }
    canonical_json = shadow._canonical_json_file_bytes(raw)
    canonical_gzip = shadow._canonical_gzip_bytes(canonical_json)
    evidence = ScoringOutcomeObservationEvidence(
        target_official_date=target_text,
        checkpoint_utc_date=checkpoint.isoformat(),
        outcome_http_date_utc=http_date,
        response_received_at_utc=received_at,
        response_body_sha256=body_sha256,
        flattened_occurrence_count=occurrences,
        raw_evidence=raw,
        canonical_json_bytes=canonical_json,
        canonical_gzip_bytes=canonical_gzip,
        canonical_gzip_sha256=hashlib.sha256(canonical_gzip).hexdigest(),
    )
    _validate_outcome_evidence(evidence)
    return evidence


def _validate_reservation_for_outcome(
    reservation: ScoringObservationReservation,
    *,
    project_directory: Path,
) -> tuple[Path, dict[str, Any], bytes, datetime]:
    if type(reservation) is not ScoringObservationReservation:
        raise shadow.ShadowPredictionError(
            "Une reservation d'observation exacte est requise."
        )
    project = shadow._preflight_project_directory(project_directory)
    target = _parse_horizon_target(reservation.target_official_date)
    checkpoint = _parse_checkpoint_date(
        reservation.checkpoint_utc_date,
        target=target,
    )
    expected_id = build_observation_id(
        scoring_protocol_sha256=(
            scoring_registration.EXPECTED_SCORING_PROTOCOL_SHA256
        ),
        target_official_date=target,
        checkpoint_utc_date=checkpoint,
    )
    expected_path = _observation_slot_path(project, target, checkpoint)
    marker = reservation.reserved_marker
    if (
        reservation.slot_path != expected_path
        or reservation.observation_id != expected_id
        or type(marker) is not dict
        or frozenset(marker) != _RESERVED_MARKER_KEYS
        or marker.get("observation_id") != expected_id
        or marker.get("target_official_date") != target.isoformat()
        or marker.get("checkpoint_utc_date") != checkpoint.isoformat()
        or marker.get("scoring_protocol_sha256")
        != scoring_registration.EXPECTED_SCORING_PROTOCOL_SHA256
    ):
        raise shadow.ShadowPredictionError(
            "La preuve RESERVED ne correspond pas a l'observation attendue."
        )
    reserved_at = _parse_utc_timestamp(
        marker.get("reserved_at_utc"),
        field_name="RESERVED.reserved_at_utc",
    )
    marker_bytes = shadow._canonical_json_file_bytes(marker)
    if hashlib.sha256(marker_bytes).hexdigest() != (
        reservation.reserved_marker_sha256
    ):
        raise shadow.ShadowPredictionError(
            "L'empreinte de la reservation en memoire diverge."
        )
    _require_local_directory(expected_path, description="creneau reserve")
    try:
        names = sorted(path.name for path in expected_path.iterdir())
    except OSError as error:
        raise shadow.ShadowPredictionError(
            "Le creneau reserve ne peut pas etre inspecte."
        ) from error
    if names != ["RESERVED"]:
        raise shadow.ShadowPredictionSlotConsumedError(
            "Le creneau reserve contient deja un second chemin."
        )
    persisted = shadow._read_canonical_json_object(expected_path / "RESERVED")
    if persisted is None or persisted[0] != marker or persisted[1] != marker_bytes:
        raise shadow.ShadowPredictionError(
            "Le marqueur RESERVED persiste diverge de sa preuve."
        )
    return expected_path, marker, marker_bytes, reserved_at


def publish_scoring_outcome_evidence(
    reservation: ScoringObservationReservation,
    evidence: ScoringOutcomeObservationEvidence,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ScoringOutcomeEvidencePublication:
    """Publie la preuve MLB comme deuxieme fichier exact du creneau."""
    _, _, canonical_gzip = _validate_outcome_evidence(evidence)
    slot, _, _, reserved_at = _validate_reservation_for_outcome(
        reservation,
        project_directory=project_directory,
    )
    if (
        evidence.target_official_date != reservation.target_official_date
        or evidence.checkpoint_utc_date != reservation.checkpoint_utc_date
    ):
        raise shadow.ShadowPredictionError(
            "La preuve MLB ne vise pas le creneau reserve."
        )
    received_at = _parse_utc_timestamp(
        evidence.response_received_at_utc,
        field_name="response_received_at_utc",
    )
    if received_at < reserved_at:
        raise shadow.ShadowPredictionError(
            "La reponse MLB ne peut pas preceder la reservation."
        )
    destination = slot / OUTCOME_EVIDENCE_FILENAME
    published_sha256 = shadow._publish_exclusive_verified(
        destination,
        canonical_gzip,
    )
    if published_sha256 != evidence.canonical_gzip_sha256:
        raise shadow.ShadowPredictionError(
            "La preuve MLB publiee diverge de sa preparation."
        )
    return ScoringOutcomeEvidencePublication(
        reservation=reservation,
        evidence_path=destination,
        evidence_sha256=published_sha256,
        evidence_size_bytes=len(canonical_gzip),
        outcome_http_date_utc=evidence.outcome_http_date_utc,
        response_received_at_utc=evidence.response_received_at_utc,
        response_body_sha256=evidence.response_body_sha256,
        flattened_occurrence_count=evidence.flattened_occurrence_count,
        evidence=evidence,
    )


def fail_scoring_observation_slot(
    reservation: ScoringObservationReservation,
    *,
    failed_at_utc: str,
    stage: str,
    error: shadow.ShadowPredictionError,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ScoringObservationFailure:
    """Ferme une requete/validation echouee avec l'unique FAILED.json."""
    slot, marker, _, reserved_at = _validate_reservation_for_outcome(
        reservation,
        project_directory=project_directory,
    )
    if type(stage) is not str or stage not in _FAILURE_STAGES:
        raise shadow.ShadowPredictionError(
            "L'etape d'echec de l'observation est invalide."
        )
    if not isinstance(error, (ScoringOutcomeRequestError, ScoringOutcomeValidationError)):
        raise shadow.ShadowPredictionError(
            "Seul un echec MLB controle peut fermer l'observation."
        )
    failed_datetime = _parse_utc_timestamp(
        failed_at_utc,
        field_name="failed_at_utc",
    )
    if failed_datetime < reserved_at:
        raise shadow.ShadowPredictionError(
            "L'echec ne peut pas preceder la reservation."
        )
    error_message = str(error)
    if not error_message or "\r" in error_message or "\n" in error_message:
        raise shadow.ShadowPredictionError(
            "Le message d'echec doit tenir sur une ligne non vide."
        )
    expected_stage = (
        "OUTCOME_REQUEST"
        if isinstance(error, ScoringOutcomeRequestError)
        else "OUTCOME_VALIDATION"
    )
    if stage != expected_stage:
        raise shadow.ShadowPredictionError(
            "L'etape d'echec ne correspond pas au type d'erreur MLB."
        )
    failed_marker: dict[str, Any] = {
        "marker_schema_version": 1,
        "protocol_id": scoring_registration.EXPECTED_PROTOCOL_ID,
        "target_official_date": marker["target_official_date"],
        "checkpoint_utc_date": marker["checkpoint_utc_date"],
        "observation_id": marker["observation_id"],
        "failed_at_utc": failed_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": error_message,
    }
    if frozenset(failed_marker) != _FAILED_MARKER_KEYS:
        raise shadow.ShadowPredictionError("Le marqueur FAILED est incomplet.")
    failed_bytes = shadow._canonical_json_file_bytes(failed_marker)
    destination = slot / FAILED_FILENAME
    failed_sha256 = shadow._publish_exclusive_verified(
        destination,
        failed_bytes,
    )
    return ScoringObservationFailure(
        reservation=reservation,
        failed_path=destination,
        failed_marker=failed_marker,
        failed_marker_sha256=failed_sha256,
        stage=stage,
        error_type=type(error).__name__,
    )


def capture_and_publish_scoring_outcome_evidence(
    reservation: ScoringObservationReservation,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ScoringOutcomeEvidencePublication | ScoringObservationFailure:
    """Appelle MLB apres RESERVED; ferme les erreurs controlees une fois."""
    _validate_reservation_for_outcome(
        reservation,
        project_directory=project_directory,
    )
    try:
        evidence = fetch_scoring_outcome_evidence(
            reservation.target_official_date,
            reservation.checkpoint_utc_date,
        )
    except (ScoringOutcomeRequestError, ScoringOutcomeValidationError) as error:
        stage = (
            "OUTCOME_REQUEST"
            if isinstance(error, ScoringOutcomeRequestError)
            else "OUTCOME_VALIDATION"
        )
        return fail_scoring_observation_slot(
            reservation,
            failed_at_utc=_format_utc_seconds(
                _utc_now(),
                field_name="failed_at_utc",
            ),
            stage=stage,
            error=error,
            project_directory=project_directory,
        )
    return publish_scoring_outcome_evidence(
        reservation,
        evidence,
        project_directory=project_directory,
    )


def _require_json_object(value: object, *, field_name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ScoringAdjudicationError(
            f"{field_name} doit etre un objet JSON exact."
        )
    return value


def _require_positive_json_integer(value: object, *, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ScoringAdjudicationError(
            f"{field_name} doit etre un entier JSON strictement positif."
        )
    return value


def _require_mlb_season(value: object) -> tuple[int | str, int]:
    if type(value) is int:
        parsed = value
    elif type(value) is str and re.fullmatch(r"[0-9]{4}", value):
        parsed = int(value)
    else:
        raise ScoringAdjudicationError(
            "season MLB doit etre un entier ou une annee JSON exacte."
        )
    if parsed != 2026:
        raise ScoringAdjudicationError(
            "SEASON_MISMATCH : une occurrence MLB n'appartient pas a 2026."
        )
    return value, parsed


def _require_canonical_official_date(value: object) -> str:
    if type(value) is not str:
        raise ScoringAdjudicationError(
            "officialDate final doit etre une date JSON."
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ScoringAdjudicationError(
            "officialDate final n'est pas une date ISO valide."
        ) from error
    if parsed.isoformat() != value:
        raise ScoringAdjudicationError(
            "officialDate final doit etre une date ISO canonique."
        )
    return value


def _normalize_mlb_status(value: object, *, field_name: str) -> str:
    if type(value) is not str:
        raise ScoringAdjudicationError(
            f"{field_name} doit etre une chaine JSON."
        )
    return _STATUS_WHITESPACE_PATTERN.sub(" ", value.strip()).upper()


def _classify_mlb_occurrence(
    status_code: str,
    status_detail: str,
) -> ScoringOutcomeFamily:
    families: set[ScoringOutcomeFamily] = set()
    if status_code in _FINAL_STATUS_CODES or status_detail in _FINAL_STATUS_DETAILS:
        families.add(ScoringOutcomeFamily.FINAL)
    if (
        status_code in _CANCELLED_STATUS_CODES
        or status_detail in _CANCELLED_STATUS_DETAILS
    ):
        families.add(ScoringOutcomeFamily.CANCELLED)
    if (
        status_code in _POSTPONED_STATUS_CODES
        or status_detail in _POSTPONED_STATUS_DETAILS
    ):
        families.add(ScoringOutcomeFamily.POSTPONED)
    if len(families) > 1:
        raise ScoringAdjudicationError(
            "STATUS_FAMILY_CONFLICT_WITHIN_OCCURRENCE"
        )
    if not families:
        return ScoringOutcomeFamily.NONTERMINAL
    return next(iter(families))


def _require_final_score(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ScoringAdjudicationError(
            "FINAL_SCORE_MISSING_NEGATIVE_BOOLEAN_NONINTEGER_OR_TIED : "
            f"{field_name} est invalide."
        )
    return value


def _parse_raw_scoring_occurrence(game: dict[str, Any]) -> _RawScoringOccurrence:
    game_id = _require_positive_json_integer(
        game.get("gamePk"),
        field_name="gamePk",
    )
    raw_season, season = _require_mlb_season(game.get("season"))
    game_type = game.get("gameType")
    if type(game_type) is not str or game_type != "R":
        raise ScoringAdjudicationError(
            "GAME_TYPE_MISMATCH : gameType MLB doit etre exactement R."
        )
    teams = _require_json_object(game.get("teams"), field_name="teams")
    away = _require_json_object(teams.get("away"), field_name="teams.away")
    home = _require_json_object(teams.get("home"), field_name="teams.home")
    away_team = _require_json_object(
        away.get("team"),
        field_name="teams.away.team",
    )
    home_team = _require_json_object(
        home.get("team"),
        field_name="teams.home.team",
    )
    away_team_id = _require_positive_json_integer(
        away_team.get("id"),
        field_name="teams.away.team.id",
    )
    home_team_id = _require_positive_json_integer(
        home_team.get("id"),
        field_name="teams.home.team.id",
    )
    if away_team_id == home_team_id:
        raise ScoringAdjudicationError(
            "TEAM_IDENTITY_MISMATCH : une equipe ne peut pas jouer contre elle-meme."
        )
    status = _require_json_object(game.get("status"), field_name="status")
    status_code = _normalize_mlb_status(
        status.get("statusCode"),
        field_name="status.statusCode",
    )
    status_detail = _normalize_mlb_status(
        status.get("detailedState"),
        field_name="status.detailedState",
    )
    family = _classify_mlb_occurrence(status_code, status_detail)
    official_date: str | None = None
    away_score: int | None = None
    home_score: int | None = None
    if family is ScoringOutcomeFamily.FINAL:
        official_date = _require_canonical_official_date(
            game.get("officialDate")
        )
        away_score = _require_final_score(
            away.get("score"),
            field_name="teams.away.score",
        )
        home_score = _require_final_score(
            home.get("score"),
            field_name="teams.home.score",
        )
        if away_score == home_score:
            raise ScoringAdjudicationError(
                "FINAL_SCORE_MISSING_NEGATIVE_BOOLEAN_NONINTEGER_OR_TIED : "
                "un match MLB final ne peut pas etre a egalite."
            )
    return _RawScoringOccurrence(
        game_id=game_id,
        raw_season=raw_season,
        season=season,
        game_type=game_type,
        away_team_id=away_team_id,
        home_team_id=home_team_id,
        family=family,
        status_code_normalized=status_code,
        status_detail_normalized=status_detail,
        official_date=official_date,
        away_score=away_score,
        home_score=home_score,
    )


def _reduce_occurrence_group(
    occurrences: Sequence[_RawScoringOccurrence],
) -> ReducedScoringOutcome:
    if not occurrences:
        raise ScoringAdjudicationError(
            "Un groupe MLB vide ne peut pas etre reduit."
        )
    first = occurrences[0]
    if any(item.exact_identity != first.exact_identity for item in occurrences[1:]):
        raise ScoringAdjudicationError("CONFLICTING_MLB_OCCURRENCE_IDENTITY")

    final_occurrences = tuple(
        item for item in occurrences if item.family is ScoringOutcomeFamily.FINAL
    )
    if final_occurrences:
        selected = final_occurrences[0]
        final_identity = (
            selected.official_date,
            selected.away_score,
            selected.home_score,
        )
        if any(
            (item.official_date, item.away_score, item.home_score)
            != final_identity
            for item in final_occurrences[1:]
        ):
            raise ScoringAdjudicationError(
                "CONFLICTING_FINAL_SCORES_OR_FINAL_OFFICIAL_DATES"
            )
    else:
        cancelled = tuple(
            item
            for item in occurrences
            if item.family is ScoringOutcomeFamily.CANCELLED
        )
        if cancelled:
            if any(
                item.family
                not in {
                    ScoringOutcomeFamily.CANCELLED,
                    ScoringOutcomeFamily.POSTPONED,
                }
                for item in occurrences
            ):
                raise ScoringAdjudicationError(
                    "Une occurrence annulee contredit un etat non terminal."
                )
            selected = cancelled[0]
        else:
            nonterminal = tuple(
                item
                for item in occurrences
                if item.family is ScoringOutcomeFamily.NONTERMINAL
            )
            selected = nonterminal[0] if nonterminal else occurrences[0]

    return ReducedScoringOutcome(
        game_id=selected.game_id,
        season=selected.season,
        game_type=selected.game_type,
        away_team_id=selected.away_team_id,
        home_team_id=selected.home_team_id,
        family=selected.family,
        status_code_normalized=selected.status_code_normalized,
        status_detail_normalized=selected.status_detail_normalized,
        final_official_date=selected.official_date,
        away_score=selected.away_score,
        home_score=selected.home_score,
    )


def reduce_scoring_outcome_evidence(
    evidence: ScoringOutcomeObservationEvidence,
) -> tuple[ReducedScoringOutcome, ...]:
    """Aplatit puis reduit toutes les occurrences MLB dans l'ordre brut."""
    _, body, _ = _validate_outcome_evidence(evidence)
    payload, _ = _decode_and_validate_outcome_payload(body)
    groups: dict[int, list[_RawScoringOccurrence]] = {}
    for date_block in payload["dates"]:
        for raw_game in date_block["games"]:
            occurrence = _parse_raw_scoring_occurrence(raw_game)
            groups.setdefault(occurrence.game_id, []).append(occurrence)
    return tuple(
        _reduce_occurrence_group(occurrences)
        for occurrences in groups.values()
    )


def _parse_probability_text(value: object, *, field_name: str) -> Decimal:
    if type(value) is not str or _PROBABILITY_TEXT_PATTERN.fullmatch(value) is None:
        raise ScoringAdjudicationError(
            f"{field_name} n'est pas une probabilite decimale canonique."
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ScoringAdjudicationError(
            f"{field_name} n'est pas une probabilite valide."
        ) from error
    if not parsed.is_finite() or parsed < 0 or parsed > 1:
        raise ScoringAdjudicationError(
            f"{field_name} doit appartenir a l'intervalle ferme [0, 1]."
        )
    return parsed


def _validate_certified_prediction(
    prediction: CertifiedScoringPrediction,
    *,
    expected_target: str,
) -> tuple[Decimal, Decimal]:
    if type(prediction) is not CertifiedScoringPrediction:
        raise ScoringAdjudicationError(
            "Une prediction certifiee exacte est requise."
        )
    shadow._require_sha256(prediction.prediction_id, field="prediction_id")
    shadow._require_sha256(prediction.batch_id, field="batch_id")
    shadow._require_sha256(prediction.occurrence_key, field="occurrence_key")
    if type(prediction.game_id) is not int or prediction.game_id <= 0:
        raise ScoringAdjudicationError("game_id de prediction invalide.")
    if type(prediction.season) is not int or prediction.season != 2026:
        raise ScoringAdjudicationError("SEASON_MISMATCH dans la prediction.")
    if prediction.target_official_date != expected_target:
        raise ScoringAdjudicationError(
            "La prediction ne vise pas la date de l'observation."
        )
    if (
        type(prediction.away_team_id) is not int
        or prediction.away_team_id <= 0
        or type(prediction.home_team_id) is not int
        or prediction.home_team_id <= 0
        or prediction.away_team_id == prediction.home_team_id
    ):
        raise ScoringAdjudicationError(
            "Identite des equipes de la prediction invalide."
        )
    home_probability = _parse_probability_text(
        prediction.p_home_win,
        field_name="p_home_win",
    )
    away_probability = _parse_probability_text(
        prediction.p_away_win,
        field_name="p_away_win",
    )
    if home_probability + away_probability != Decimal(1):
        raise ScoringAdjudicationError(
            "Les probabilites domicile et exterieur ne totalisent pas exactement 1."
        )
    return home_probability, away_probability


def _format_fixed_12(value: float | Decimal, *, field_name: str) -> str:
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        formatted = decimal_value.quantize(
            _FIXED_12_QUANTUM,
            rounding=ROUND_HALF_EVEN,
        )
    except (InvalidOperation, ValueError) as error:
        raise ScoringAdjudicationError(
            f"{field_name} ne peut pas etre arrondi a douze decimales."
        ) from error
    if not formatted.is_finite():
        raise ScoringAdjudicationError(f"{field_name} doit etre fini.")
    return format(formatted, "f")


def _individual_probability_metrics(
    home_probability: Decimal,
    home_win: int,
) -> tuple[str, str]:
    probability = float(home_probability)
    clipped = min(
        max(probability, _LOG_LOSS_CLIP_EPSILON),
        1.0 - _LOG_LOSS_CLIP_EPSILON,
    )
    log_loss = -(
        home_win * math.log(clipped)
        + (1 - home_win) * math.log(1.0 - clipped)
    )
    if not math.isfinite(log_loss):
        raise ScoringAdjudicationError("La log loss individuelle n'est pas finie.")
    brier = (home_probability - Decimal(home_win)) ** 2
    return (
        _format_fixed_12(log_loss, field_name="individual_log_loss"),
        _format_fixed_12(brier, field_name="individual_brier_score"),
    )


def adjudicate_scoring_predictions(
    predictions: Sequence[CertifiedScoringPrediction],
    evidence: ScoringOutcomeObservationEvidence,
) -> ScoringAdjudicationBatch:
    """Adjuge chaque prediction sans filtre, SQLite, modele ou nouvel appel MLB."""
    if isinstance(predictions, (str, bytes, bytearray)):
        raise ScoringAdjudicationError(
            "Les predictions doivent former une sequence de lignes exactes."
        )
    try:
        prediction_rows = tuple(predictions)
    except TypeError as error:
        raise ScoringAdjudicationError(
            "Les predictions doivent etre iterables."
        ) from error
    target = _parse_horizon_target(evidence.target_official_date)
    checkpoint = _parse_checkpoint_date(
        evidence.checkpoint_utc_date,
        target=target,
    )
    reduced = reduce_scoring_outcome_evidence(evidence)
    reduced_by_game_id = {item.game_id: item for item in reduced}
    if len(reduced_by_game_id) != len(reduced):
        raise ScoringAdjudicationError(
            "La reduction MLB contient un game_id duplique."
        )

    seen_prediction_ids: set[str] = set()
    seen_game_ids: set[int] = set()
    batch_ids: set[str] = set()
    validated: list[
        tuple[CertifiedScoringPrediction, Decimal, Decimal]
    ] = []
    for prediction in prediction_rows:
        home_probability, away_probability = _validate_certified_prediction(
            prediction,
            expected_target=target.isoformat(),
        )
        if prediction.prediction_id in seen_prediction_ids:
            raise ScoringAdjudicationError("DUPLICATE_PREDICTION_ID")
        if prediction.game_id in seen_game_ids:
            raise ScoringAdjudicationError("DUPLICATE_GAME_ID_WITHIN_BATCH")
        seen_prediction_ids.add(prediction.prediction_id)
        seen_game_ids.add(prediction.game_id)
        batch_ids.add(prediction.batch_id)
        validated.append((prediction, home_probability, away_probability))
    if len(batch_ids) > 1:
        raise ScoringAdjudicationError(
            "Une observation ne peut pas melanger plusieurs batch_id."
        )
    validated.sort(key=lambda item: item[0].prediction_id)

    at_deadline = checkpoint == FINAL_CHECKPOINT_AT_UTC.date()
    adjudications: list[ScoringAdjudication] = []
    for prediction, home_probability, _away_probability in validated:
        outcome = reduced_by_game_id.get(prediction.game_id)
        status_code = ""
        status_detail = ""
        final_official_date: str | None = None
        away_score: int | None = None
        home_score: int | None = None
        home_win: int | None = None
        actual_winner: str | None = None
        classification_correct: int | None = None
        individual_log_loss: str | None = None
        individual_brier_score: str | None = None

        if outcome is None:
            outcome_status = (
                "VOID_UNRESOLVED_AT_DEADLINE"
                if at_deadline
                else "PENDING_MISSING_FROM_OBSERVATION"
            )
        else:
            if outcome.season != prediction.season:
                raise ScoringAdjudicationError("SEASON_MISMATCH")
            if outcome.game_type != "R":
                raise ScoringAdjudicationError("GAME_TYPE_MISMATCH")
            if (
                outcome.away_team_id != prediction.away_team_id
                or outcome.home_team_id != prediction.home_team_id
            ):
                raise ScoringAdjudicationError("TEAM_IDENTITY_MISMATCH")
            status_code = outcome.status_code_normalized
            status_detail = outcome.status_detail_normalized
            final_official_date = outcome.final_official_date
            if outcome.family is ScoringOutcomeFamily.FINAL:
                away_score = outcome.away_score
                home_score = outcome.home_score
                if away_score is None or home_score is None or final_official_date is None:
                    raise ScoringAdjudicationError(
                        "Une finale reduite est structurellement incomplete."
                    )
                if final_official_date != prediction.target_official_date:
                    outcome_status = "VOID_RESCHEDULED_OFFICIAL_DATE"
                else:
                    outcome_status = "SCORED_FINAL"
                    home_win = int(home_score > away_score)
                    actual_winner = "HOME" if home_win else "AWAY"
                    predicted_side = (
                        "HOME"
                        if home_probability >= Decimal("0.5")
                        else "AWAY"
                    )
                    classification_correct = int(predicted_side == actual_winner)
                    (
                        individual_log_loss,
                        individual_brier_score,
                    ) = _individual_probability_metrics(
                        home_probability,
                        home_win,
                    )
            elif outcome.family is ScoringOutcomeFamily.CANCELLED:
                outcome_status = "VOID_CANCELLED"
            elif outcome.family is ScoringOutcomeFamily.POSTPONED:
                outcome_status = (
                    "VOID_POSTPONED_AT_DEADLINE"
                    if at_deadline
                    else "PENDING_POSTPONED"
                )
            else:
                outcome_status = (
                    "VOID_UNRESOLVED_AT_DEADLINE"
                    if at_deadline
                    else "PENDING_NONTERMINAL"
                )

        predicted_side = (
            "HOME" if home_probability >= Decimal("0.5") else "AWAY"
        )
        if outcome_status not in _ALL_OUTCOME_STATUSES:
            raise ScoringAdjudicationError("Statut d'adjudication non autorise.")
        adjudications.append(
            ScoringAdjudication(
                prediction_id=prediction.prediction_id,
                batch_id=prediction.batch_id,
                game_id=prediction.game_id,
                occurrence_key=prediction.occurrence_key,
                target_official_date=prediction.target_official_date,
                away_team_id=prediction.away_team_id,
                home_team_id=prediction.home_team_id,
                p_away_win=prediction.p_away_win,
                p_home_win=prediction.p_home_win,
                predicted_side=predicted_side,
                outcome_status=outcome_status,
                away_score=away_score,
                home_score=home_score,
                home_win=home_win,
                actual_winner=actual_winner,
                classification_correct=classification_correct,
                individual_log_loss=individual_log_loss,
                individual_brier_score=individual_brier_score,
                status_code_normalized=status_code,
                status_detail_normalized=status_detail,
                final_official_date=final_official_date,
                outcome_http_date_utc=evidence.outcome_http_date_utc,
                outcome_response_received_at_utc=(
                    evidence.response_received_at_utc
                ),
                outcome_evidence_sha256=evidence.canonical_gzip_sha256,
            )
        )

    scored_count = sum(
        row.outcome_status == "SCORED_FINAL" for row in adjudications
    )
    void_count = sum(
        row.outcome_status in _VOID_OUTCOME_STATUSES for row in adjudications
    )
    pending_count = sum(
        row.outcome_status in _PENDING_OUTCOME_STATUSES for row in adjudications
    )
    correct_count = sum(row.classification_correct == 1 for row in adjudications)
    incorrect_count = sum(
        row.classification_correct == 0 for row in adjudications
    )
    if scored_count + void_count + pending_count != len(adjudications):
        raise ScoringAdjudicationError(
            "Les comptes d'adjudication ne couvrent pas toutes les predictions."
        )
    if correct_count + incorrect_count != scored_count:
        raise ScoringAdjudicationError(
            "Les comptes de classification divergent des matchs scores."
        )
    return ScoringAdjudicationBatch(
        target_official_date=target.isoformat(),
        checkpoint_utc_date=checkpoint.isoformat(),
        outcome_evidence_sha256=evidence.canonical_gzip_sha256,
        adjudications=tuple(adjudications),
        certified_prediction_count=len(adjudications),
        scored_count=scored_count,
        void_count=void_count,
        pending_count=pending_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
    )
