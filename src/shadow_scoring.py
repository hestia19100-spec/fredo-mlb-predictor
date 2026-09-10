"""Moteur append-only du scoring prospectif shadow MLB 2026.

Le moteur valide l'autorite du protocole deja preenregistre, reserve
atomiquement un creneau d'observation et archive une reponse MLB fraiche.
Il ne lit ni SQLite ni modele et n'interprete encore aucun score : cette
couche conserve uniquement la preuve brute ou ferme l'observation en echec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
import base64
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping
import zlib

import requests

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
