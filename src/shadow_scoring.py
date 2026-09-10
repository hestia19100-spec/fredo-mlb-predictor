"""Moteur append-only du scoring prospectif shadow MLB 2026.

Ce premier lot valide l'autorite du protocole deja preenregistre et reserve
atomiquement un creneau d'observation. Il ne contacte pas MLB, ne lit ni
SQLite, ni modele, ni resultat et ne calcule aucune metrique.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
import gzip
import hashlib
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping
import zlib

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
