"""Preenregistrement distant du protocole de scoring shadow MLB 2026.

Ce module ne collecte aucun resultat sportif. Il valide le protocole separe,
son unique blob Git d'introduction et les preuves immuables du premier lot,
puis demande a GitHub une attestation anonyme de visibilite du commit exact.
La preuve brute gzip et le recu JSON sont publies une seule fois, sans
ecrasement. Aucun acces MLB, SQLite ou modele n'est effectue.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_EVEN
import gzip
import hashlib
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Mapping, Sequence
import zlib

from src import shadow_certification as certification
from src import shadow_prediction as shadow


PROJECT_DIRECTORY = shadow.PROJECT_DIRECTORY
SCORING_PROTOCOL_RELATIVE_PATH = PurePosixPath(
    "scoring_protocols/"
    "logistic_team_form_v1_platt_shadow_v2_2026_v1.json"
)
REGISTRATION_SERVICE_RELATIVE_PATH = PurePosixPath(
    "src/shadow_scoring_registration.py"
)
SCORING_REGISTRATION_ROOT_RELATIVE_PATH = PurePosixPath(
    "scoring_registrations/"
    "logistic_team_form_v1_platt_shadow_v2_2026_v1"
)
RAW_REMOTE_EVIDENCE_RELATIVE_PATH = (
    SCORING_REGISTRATION_ROOT_RELATIVE_PATH / "protocol.remote.json.gz"
)
REGISTRATION_RELATIVE_PATH = (
    SCORING_REGISTRATION_ROOT_RELATIVE_PATH / "registration.json"
)

EXPECTED_SCORING_PROTOCOL_SHA256 = (
    "2b886357bd79ced0397aec394415fd3e"
    "e8ad061f3a4f770691d9e8b7093f2f96"
)
EXPECTED_PROTOCOL_ID = (
    "logistic_team_form_v1_platt_shadow_v2_2026_scoring_v1"
)
EXPECTED_PROTOCOL_STATUS = "FROZEN_BEFORE_FIRST_2026_OUTCOME_OBSERVATION"
EXPECTED_FIRST_TARGET_DATE = "2026-09-10"
EXPECTED_LAST_TARGET_DATE = "2026-09-27"
EXPECTED_FIRST_START_UTC = "2026-09-10T16:15:00Z"
EXPECTED_FIRST_BATCH_ID = (
    "e482070d9f8f7c4239f9be4c84637f6"
    "b72a983a2b1cee37b49fcfc0562a9fdeb"
)
EXPECTED_FIRST_RESULTS_COMMIT = (
    "4d68c78cebf6bf2d3f364296fcca75cf860bae4b"
)
EXPECTED_FIRST_CERTIFICATION_COMMIT = (
    "23319446260d58673f7eed34ac15a4690d8ca192"
)
EXPECTED_FIRST_CERTIFICATION_PATH = (
    "shadow_certifications/logistic_team_form_v1_platt_shadow_v2/"
    "2026-09-10.json"
)
EXPECTED_FIRST_CERTIFICATION_SHA256 = (
    "de322ded087b77c6cd34b85282fa37e3"
    "b2ab52a2b32e20ba1507cfe9701988e0"
)
EXPECTED_FIRST_PREDICTIONS_SHA256 = (
    "576618f39455889c40fcc17a34a3c3f3"
    "248a63c58b9755424e6437df92bb3e9b"
)
EXPECTED_FIRST_RECEIPT_SHA256 = (
    "f8e13957643b0b03093117a1f6f0587e"
    "0583b6e879b9bdeb355574d9422481b1"
)
MINIMUM_REMOTE_REGISTRATION_LEAD_SECONDS = 3600
REGISTRATION_STATUS = (
    "SCORING_PROTOCOL_REGISTERED_REMOTELY_BEFORE_FIRST_PREDICTED_START"
)
REGISTRATION_CLAIM_LEVEL = (
    "REMOTE_SERVER_ATTESTED_NOT_CRYPTOGRAPHICALLY_TIMESTAMPED"
)

_PROTOCOL_TOP_LEVEL_KEYS = frozenset(
    {
        "aggregate_metrics",
        "authorities",
        "daily_reporting",
        "fixed_horizon",
        "forbidden_actions",
        "identity_and_linkage",
        "intent_to_observe",
        "limitations",
        "outcome_adjudication",
        "outcome_observation",
        "output_publication",
        "per_prediction_output",
        "prediction_cohort",
        "prediction_source",
        "prospective_boundary",
        "protocol_id",
        "protocol_schema_version",
        "purpose",
        "registered_on",
        "remote_registration",
        "settlement",
        "status",
        "uncertainty",
        "verdict",
    }
)
_REGISTRATION_KEYS = frozenset(
    {
        "registration_schema_version",
        "protocol_id",
        "status",
        "claim_level",
        "scoring_protocol_path",
        "scoring_protocol_sha256",
        "scoring_protocol_introduction_commit",
        "registration_service_path",
        "registration_service_sha256",
        "runtime_code_commit",
        "remote_ref",
        "remote_query_url",
        "remote_effective_url",
        "remote_response_status_code",
        "remote_response_redirect_count",
        "remote_http_date_utc",
        "remote_response_received_at_utc",
        "remote_response_body_sha256",
        "raw_remote_evidence_path",
        "raw_remote_evidence_sha256",
        "first_target_official_date",
        "first_batch_id",
        "first_results_commit",
        "first_certification_commit",
        "first_certification_sha256",
        "first_predictions_sha256",
        "first_receipt_sha256",
        "first_earliest_predicted_start_utc",
        "remote_publication_lead_minutes",
        "registered_at_utc",
        "negative_attestations",
    }
)
_NEGATIVE_ATTESTATION_KEYS = frozenset(
    {
        "MLB_NOT_READ",
        "SQLITE_NOT_READ",
        "MODEL_NOT_READ_OR_DESERIALIZED",
        "OUTCOMES_NOT_READ",
        "PREDICTIONS_NOT_RECOMPUTED",
        "EXISTING_SHADOW_FILES_NOT_MODIFIED",
    }
)


@dataclass(frozen=True, slots=True)
class ScoringProtocolAuthority:
    """Autorite locale validee avant l'unique requete GitHub."""

    protocol: dict[str, Any] = field(repr=False)
    protocol_bytes: bytes = field(repr=False)
    protocol_sha256: str
    protocol_introduction_commit: str
    runtime_code_commit: str
    registration_service_sha256: str
    first_batch: certification.CompletedShadowBatchCommit = field(repr=False)
    first_certification: dict[str, Any] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ScoringRegistrationPublication:
    """Preuve locale des deux publications append-only du preenregistrement."""

    raw_evidence_path: Path
    raw_evidence_relative_path: str
    raw_evidence_sha256: str
    raw_evidence_size_bytes: int
    registration_path: Path
    registration_relative_path: str
    registration_sha256: str
    registration_size_bytes: int
    protocol_introduction_commit: str
    runtime_code_commit: str
    remote_http_date_utc: str
    remote_publication_lead_minutes: str
    registration: dict[str, Any] = field(repr=False)


def _require_mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise shadow.ShadowPredictionError(
            f"{field_name} doit etre un objet JSON exact."
        )
    return value


def _validate_scoring_protocol(protocol: Mapping[str, Any]) -> None:
    """Refuse toute derive des decisions preenregistrees essentielles."""
    if type(protocol) is not dict or frozenset(protocol) != _PROTOCOL_TOP_LEVEL_KEYS:
        raise shadow.ShadowPredictionError(
            "Le protocole de scoring ne respecte pas son schema exact."
        )
    exact_values: dict[str, object] = {
        "protocol_schema_version": 1,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "registered_on": EXPECTED_FIRST_TARGET_DATE,
        "status": EXPECTED_PROTOCOL_STATUS,
    }
    for key, expected in exact_values.items():
        actual = protocol.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise shadow.ShadowPredictionError(
                f"Valeur du protocole de scoring invalide pour {key}."
            )

    horizon = _require_mapping(protocol.get("fixed_horizon"), field_name="fixed_horizon")
    if horizon != {
        "calendar_days": 18,
        "game_type": "R",
        "no_early_stop": True,
        "season": 2026,
        "target_official_date_end": EXPECTED_LAST_TARGET_DATE,
        "target_official_date_start": EXPECTED_FIRST_TARGET_DATE,
    }:
        raise shadow.ShadowPredictionError(
            "L'horizon fixe du scoring a change."
        )

    dates = _require_mapping(
        protocol.get("intent_to_observe"), field_name="intent_to_observe"
    ).get("dates")
    expected_dates = tuple(
        f"2026-09-{day:02d}" for day in range(10, 28)
    )
    if type(dates) is not list or tuple(dates) != expected_dates:
        raise shadow.ShadowPredictionError(
            "Les dix-huit dates du scoring ne sont pas exactes."
        )

    prospective = _require_mapping(
        protocol.get("prospective_boundary"), field_name="prospective_boundary"
    )
    if (
        prospective.get("first_predicted_start_utc") != EXPECTED_FIRST_START_UTC
        or prospective.get("minimum_remote_registration_lead_seconds")
        != MINIMUM_REMOTE_REGISTRATION_LEAD_SECONDS
        or prospective.get("outcomes_must_not_be_read_before_remote_registration")
        is not True
        or prospective.get("remote_registration_required") is not True
    ):
        raise shadow.ShadowPredictionError(
            "La frontiere prospective du scoring a change."
        )

    authorities = _require_mapping(
        protocol.get("authorities"), field_name="authorities"
    )
    first = _require_mapping(
        authorities.get("first_certified_batch"),
        field_name="authorities.first_certified_batch",
    )
    expected_first = {
        "batch_id": EXPECTED_FIRST_BATCH_ID,
        "certification_commit": EXPECTED_FIRST_CERTIFICATION_COMMIT,
        "certification_path": EXPECTED_FIRST_CERTIFICATION_PATH,
        "certification_sha256": EXPECTED_FIRST_CERTIFICATION_SHA256,
        "earliest_predicted_start_utc": EXPECTED_FIRST_START_UTC,
        "predictions_sha256": EXPECTED_FIRST_PREDICTIONS_SHA256,
        "receipt_sha256": EXPECTED_FIRST_RECEIPT_SHA256,
        "results_commit": EXPECTED_FIRST_RESULTS_COMMIT,
        "target_official_date": EXPECTED_FIRST_TARGET_DATE,
    }
    if first != expected_first:
        raise shadow.ShadowPredictionError(
            "L'ancrage du premier lot certifie a change."
        )

    registration = _require_mapping(
        protocol.get("remote_registration"), field_name="remote_registration"
    )
    if (
        registration.get("full_ref") != shadow.GITHUB_REMOTE_REF
        or registration.get("status_code") != 200
        or registration.get("redirect_count") != 0
        or registration.get("minimum_lead_seconds_before_first_predicted_start")
        != MINIMUM_REMOTE_REGISTRATION_LEAD_SECONDS
        or registration.get(
            "maximum_github_date_to_local_receive_skew_seconds"
        )
        != 300
        or registration.get("protocol_introduction_commit_must_equal_clean_runtime_head")
        is not True
        or registration.get("remote_evidence_path")
        != RAW_REMOTE_EVIDENCE_RELATIVE_PATH.as_posix()
        or registration.get("registration_path")
        != REGISTRATION_RELATIVE_PATH.as_posix()
    ):
        raise shadow.ShadowPredictionError(
            "Le contrat de preenregistrement distant a change."
        )

    forbidden = protocol.get("forbidden_actions")
    required_forbidden = {
        "READ_OUTCOME_BEFORE_REMOTE_PROTOCOL_REGISTRATION",
        "LOAD_OR_DESERIALIZE_MODEL_DURING_SCORING",
        "STOP_EARLY_USING_OBSERVED_OUTCOMES",
        "USE_ODDS_IMPLIED_PROBABILITY_EDGE_EXPECTED_VALUE_STAKE_PROFIT_ROI_OR_BET_RECOMMENDATION",
    }
    if type(forbidden) is not list or not required_forbidden.issubset(forbidden):
        raise shadow.ShadowPredictionError(
            "Les interdictions anti-fuite du scoring sont incompletes."
        )

    intent = _require_mapping(
        protocol.get("intent_to_observe"), field_name="intent_to_observe"
    )
    if (
        intent.get("ledger_row_count") != 18
        or intent.get("all_calendar_dates_required") is not True
        or "ALL_18_DATES_MUST_BE_CERTIFIED_NONEMPTY_OR_COMPLETED_EMPTY"
        not in str(intent.get("confirmatory_coverage_rule"))
    ):
        raise shadow.ShadowPredictionError(
            "La couverture exhaustive des dix-huit dates a change."
        )

    adjudication = _require_mapping(
        protocol.get("outcome_adjudication"),
        field_name="outcome_adjudication",
    )
    results = _require_mapping(
        adjudication.get("result_by_reduced_family"),
        field_name="outcome_adjudication.result_by_reduced_family",
    )
    deadline = _require_mapping(
        adjudication.get("deadline_conversion"),
        field_name="outcome_adjudication.deadline_conversion",
    )
    if (
        results.get("POSTPONED_BEFORE_DEADLINE") != "PENDING_POSTPONED"
        or results.get("FINAL_SAME_OFFICIAL_DATE") != "SCORED_FINAL"
        or results.get("FINAL_DIFFERENT_OFFICIAL_DATE")
        != "VOID_RESCHEDULED_OFFICIAL_DATE"
        or deadline.get("PENDING_POSTPONED")
        != "VOID_POSTPONED_AT_DEADLINE"
    ):
        raise shadow.ShadowPredictionError(
            "La machine d'etat des resultats a change."
        )

    publication = _require_mapping(
        protocol.get("output_publication"), field_name="output_publication"
    )
    if (
        publication.get("observation_success_write_order_exact")
        != [
            "RESERVED",
            "outcome_observation.remote.json.gz",
            "adjudications.csv",
            "daily_report.json",
            "observation_receipt.json",
            "COMPLETED",
        ]
        or "observations/CHECKPOINT_UTC_DATE"
        not in str(publication.get("observation_path_template"))
        or publication.get("publication_semantics")
        != "EXCLUSIVE_CREATE_FSYNC_REREAD_VERIFY"
    ):
        raise shadow.ShadowPredictionError(
            "Le contrat append-only des observations a change."
        )

    metric_rules = _require_mapping(
        _require_mapping(
            protocol.get("daily_reporting"), field_name="daily_reporting"
        ).get("metric_value_rules"),
        field_name="daily_reporting.metric_value_rules",
    )
    if (
        metric_rules.get(
            "accuracy_mean_log_loss_and_mean_brier_score_when_scored_count_is_zero"
        )
        != "JSON_NULL"
        or "IF_AND_ONLY_IF" not in str(metric_rules.get("null_equivalence"))
    ):
        raise shadow.ShadowPredictionError(
            "L'encodage des metriques quotidiennes sans score a change."
        )

    discovery = _require_mapping(
        protocol.get("prediction_source"), field_name="prediction_source"
    ).get("discovery_algorithm_exact")
    if (
        type(discovery) is not list
        or len(discovery) != 11
        or "raw_remote_evidence_path" not in str(discovery[9])
        or "SAME_UNIQUE_INTRODUCTION_COMMIT" not in str(discovery[9])
        or "git_show" not in str(discovery[9])
        or "CANONICAL_JSON" not in str(discovery[9])
    ):
        raise shadow.ShadowPredictionError(
            "La decouverte des preuves distantes futures a change."
        )

    uncertainty = _require_mapping(
        protocol.get("uncertainty"), field_name="uncertainty"
    )
    if (
        uncertainty.get("replications") != 5000
        or uncertainty.get("random_seed") != 42
        or uncertainty.get("blocks_drawn_per_replicate_exact") != 3
        or uncertainty.get("sampled_day_slots_per_replicate_exact") != 18
        or uncertainty.get("sampled_start_index_domain_inclusive") != [0, 11]
    ):
        raise shadow.ShadowPredictionError(
            "La procedure bootstrap preenregistree a change."
        )

    coverage = _require_mapping(
        _require_mapping(protocol.get("verdict"), field_name="verdict").get(
            "coverage_gate"
        ),
        field_name="verdict.coverage_gate",
    )
    if (
        coverage.get("minimum_operationally_valid_dates") != 18
        or coverage.get("maximum_missed_dates") != 0
        or coverage.get("maximum_failed_dates") != 0
        or coverage.get("maximum_local_only_dates") != 0
        or coverage.get("minimum_scorable_games") != 100
        or coverage.get("minimum_scorable_fraction_of_certified_predictions")
        != 0.95
    ):
        raise shadow.ShadowPredictionError(
            "La porte de couverture prospective a change."
        )


def _read_scoring_protocol(
    project_directory: Path,
) -> tuple[dict[str, Any], bytes, str]:
    content = shadow._read_regular_project_file(
        project_directory,
        SCORING_PROTOCOL_RELATIVE_PATH,
        description="protocole de scoring 2026",
    )
    sha256 = hashlib.sha256(content).hexdigest()
    if sha256 != EXPECTED_SCORING_PROTOCOL_SHA256:
        raise shadow.ShadowPredictionError(
            "L'empreinte du protocole de scoring 2026 est invalide."
        )
    protocol = shadow._read_canonical_json_bytes(
        content,
        description="protocole de scoring 2026",
    )
    _validate_scoring_protocol(protocol)
    return protocol, content, sha256


def _require_absent_registration_paths(
    project_directory: Path,
) -> tuple[Path, Path]:
    """Refuse avant tout reseau un succes, un orphelin ou une substitution."""
    shadow._require_tracked_nonignored_root(
        project_directory,
        SCORING_REGISTRATION_ROOT_RELATIVE_PATH,
    )
    root = project_directory.joinpath(
        *SCORING_REGISTRATION_ROOT_RELATIVE_PATH.parts
    )
    mode = shadow._lstat_mode(root)
    if not isinstance(mode, int) or not stat.S_ISDIR(mode):
        raise shadow.ShadowPredictionError(
            "La racine de preenregistrement doit etre un repertoire local."
        )
    paths = (
        project_directory.joinpath(*RAW_REMOTE_EVIDENCE_RELATIVE_PATH.parts),
        project_directory.joinpath(*REGISTRATION_RELATIVE_PATH.parts),
    )
    for relative, path in zip(
        (RAW_REMOTE_EVIDENCE_RELATIVE_PATH, REGISTRATION_RELATIVE_PATH),
        paths,
        strict=True,
    ):
        return_code, _ = shadow._run_preflight_git(
            project_directory,
            ("check-ignore", "-q", "--", relative.as_posix()),
            accepted_return_codes=frozenset({0, 1}),
        )
        if return_code == 0:
            raise shadow.ShadowPredictionError(
                "Un chemin de preenregistrement est interdit par .gitignore."
            )
        if shadow._lstat_mode(path) is not shadow._PATH_MISSING:
            raise shadow.ShadowPublicationConflictError(
                "Le preenregistrement est deja consomme et ne peut jamais "
                "etre relu, ecrase, repare ou retente."
            )
    return paths


def _git_history(
    project_directory: Path,
    paths: Sequence[str],
) -> tuple[str, ...]:
    _, output = shadow._run_preflight_git(
        project_directory,
        ("log", "--format=%H", "--", *paths),
    )
    return tuple(
        line
        for line in shadow._git_text(output, field="historique immuable").splitlines()
        if line
    )


def _validate_first_certified_batch(
    project_directory: Path,
    runtime_code_commit: str,
    protocol: Mapping[str, Any],
) -> tuple[certification.CompletedShadowBatchCommit, dict[str, Any]]:
    """Relit le premier lot et sa certification depuis leurs seuls blobs Git."""
    shadow._require_git_ancestor(
        project_directory,
        EXPECTED_FIRST_RESULTS_COMMIT,
        EXPECTED_FIRST_CERTIFICATION_COMMIT,
        strict=True,
        description="premier lot vers sa certification",
    )
    shadow._require_git_ancestor(
        project_directory,
        EXPECTED_FIRST_CERTIFICATION_COMMIT,
        runtime_code_commit,
        strict=True,
        description="premiere certification vers protocole de scoring",
    )
    result_prefix = (
        shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH / EXPECTED_FIRST_TARGET_DATE
    ).as_posix()
    if _git_history(project_directory, (result_prefix,)) != (
        EXPECTED_FIRST_RESULTS_COMMIT,
    ):
        raise shadow.ShadowPredictionError(
            "L'historique Git du premier lot n'est plus immuable."
        )

    batch = certification._read_completed_batch_commit(
        EXPECTED_FIRST_TARGET_DATE,
        EXPECTED_FIRST_RESULTS_COMMIT,
        project_directory=project_directory,
    )
    if (
        batch.target_official_date != EXPECTED_FIRST_TARGET_DATE
        or batch.results_commit != EXPECTED_FIRST_RESULTS_COMMIT
        or batch.batch_id != EXPECTED_FIRST_BATCH_ID
        or batch.earliest_predicted_start_utc != EXPECTED_FIRST_START_UTC
    ):
        raise shadow.ShadowPredictionError(
            "Le premier lot Git ne correspond plus au protocole de scoring."
        )

    certificate_blob = shadow._git_blob_at_commit(
        project_directory,
        EXPECTED_FIRST_CERTIFICATION_COMMIT,
        EXPECTED_FIRST_CERTIFICATION_PATH,
    )
    if hashlib.sha256(certificate_blob).hexdigest() != (
        EXPECTED_FIRST_CERTIFICATION_SHA256
    ):
        raise shadow.ShadowPredictionError(
            "L'empreinte de la premiere certification est invalide."
        )
    certificate = shadow._read_canonical_json_bytes(
        certificate_blob,
        description="premiere certification prospective",
    )
    exact_certificate_values = {
        "target_official_date": EXPECTED_FIRST_TARGET_DATE,
        "batch_id": EXPECTED_FIRST_BATCH_ID,
        "results_commit": EXPECTED_FIRST_RESULTS_COMMIT,
        "earliest_predicted_start_utc": EXPECTED_FIRST_START_UTC,
        "status": certification.CERTIFICATION_STATUS,
        "claim_level": certification.CERTIFICATION_CLAIM_LEVEL,
    }
    for field_name, expected in exact_certificate_values.items():
        if certificate.get(field_name) != expected:
            raise shadow.ShadowPredictionError(
                "La premiere certification ne correspond plus au protocole."
            )
    entries = certificate.get("results_tree_file_hashes")
    if type(entries) is not list:
        raise shadow.ShadowPredictionError(
            "Les empreintes du premier lot certifie sont absentes."
        )
    hashes = {
        entry.get("path"): entry.get("sha256")
        for entry in entries
        if type(entry) is dict
    }
    if (
        hashes.get(shadow.PREDICTIONS_FILENAME)
        != EXPECTED_FIRST_PREDICTIONS_SHA256
        or hashes.get(shadow.RECEIPT_FILENAME) != EXPECTED_FIRST_RECEIPT_SHA256
    ):
        raise shadow.ShadowPredictionError(
            "Les empreintes du premier lot divergent de sa certification."
        )

    raw_path = certificate.get("raw_remote_evidence_path")
    raw_sha256 = certificate.get("raw_remote_evidence_sha256")
    if type(raw_path) is not str or type(raw_sha256) is not str:
        raise shadow.ShadowPredictionError(
            "La preuve brute de la premiere certification est absente."
        )
    if _git_history(
        project_directory,
        (EXPECTED_FIRST_CERTIFICATION_PATH, raw_path),
    ) != (EXPECTED_FIRST_CERTIFICATION_COMMIT,):
        raise shadow.ShadowPredictionError(
            "L'historique Git de la premiere certification n'est plus immuable."
        )
    raw_blob = shadow._git_blob_at_commit(
        project_directory,
        EXPECTED_FIRST_CERTIFICATION_COMMIT,
        raw_path,
    )
    if hashlib.sha256(raw_blob).hexdigest() != raw_sha256:
        raise shadow.ShadowPredictionError(
            "La preuve distante de la premiere certification est alteree."
        )
    try:
        raw_json = gzip.decompress(raw_blob)
    except (OSError, EOFError, zlib.error) as error:
        raise shadow.ShadowPredictionError(
            "La preuve distante de la premiere certification est illisible."
        ) from error
    shadow._read_canonical_json_bytes(
        raw_json,
        description="preuve distante de la premiere certification",
    )
    if shadow._canonical_gzip_bytes(raw_json) != raw_blob:
        raise shadow.ShadowPredictionError(
            "La preuve distante de la premiere certification n'est pas canonique."
        )

    first = _require_mapping(
        _require_mapping(protocol.get("authorities"), field_name="authorities").get(
            "first_certified_batch"
        ),
        field_name="authorities.first_certified_batch",
    )
    if first.get("certification_sha256") != hashlib.sha256(
        certificate_blob
    ).hexdigest():
        raise shadow.ShadowPredictionError(
            "Le protocole ne lie plus la premiere certification exacte."
        )
    return batch, certificate


def verify_scoring_protocol_authority(
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ScoringProtocolAuthority:
    """Valide toute l'autorite locale sans reseau ni resultat sportif."""
    project = shadow._preflight_project_directory(project_directory)
    _require_absent_registration_paths(project)
    protocol, protocol_bytes, protocol_sha256 = _read_scoring_protocol(project)
    runtime_commit = shadow._require_exact_clean_git_root(project)
    introduction_commit, introduction_blob = shadow._git_immutable_introduction_blob(
        project,
        SCORING_PROTOCOL_RELATIVE_PATH,
        protocol_bytes,
    )
    if introduction_blob != protocol_bytes or introduction_commit != runtime_commit:
        raise shadow.ShadowPredictionError(
            "Le protocole doit etre execute depuis son commit d'introduction "
            "exact, propre et encore HEAD."
        )

    service_bytes = shadow._read_regular_project_file(
        project,
        REGISTRATION_SERVICE_RELATIVE_PATH,
        description="service de preenregistrement du scoring",
    )
    service_blob = shadow._git_blob_at_commit(
        project,
        runtime_commit,
        REGISTRATION_SERVICE_RELATIVE_PATH.as_posix(),
    )
    if service_blob != service_bytes:
        raise shadow.ShadowPredictionError(
            "Le service local de preenregistrement differe de son blob Git."
        )
    service_sha256 = hashlib.sha256(service_bytes).hexdigest()
    batch, first_certificate = _validate_first_certified_batch(
        project,
        runtime_commit,
        protocol,
    )
    return ScoringProtocolAuthority(
        protocol=dict(protocol),
        protocol_bytes=protocol_bytes,
        protocol_sha256=protocol_sha256,
        protocol_introduction_commit=introduction_commit,
        runtime_code_commit=runtime_commit,
        registration_service_sha256=service_sha256,
        first_batch=batch,
        first_certification=first_certificate,
    )


def _remote_registration_lead(
    remote_http_date_utc: str,
    response_received_at_utc: str | None = None,
) -> tuple[int, str]:
    remote = shadow._require_utc_timestamp(
        remote_http_date_utc,
        field="remote_http_date_utc",
    )
    earliest = datetime.fromisoformat(EXPECTED_FIRST_START_UTC.replace("Z", "+00:00"))
    remote_datetime = datetime.fromisoformat(remote.replace("Z", "+00:00"))
    if response_received_at_utc is not None:
        received = shadow._require_utc_timestamp(
            response_received_at_utc,
            field="response_received_at_utc",
        )
        received_datetime = datetime.fromisoformat(
            received.replace("Z", "+00:00")
        )
        skew_seconds = abs(
            (received_datetime - remote_datetime).total_seconds()
        )
        received_lead_seconds = (
            earliest - received_datetime
        ).total_seconds()
        if (
            not skew_seconds.is_integer()
            or skew_seconds > 300
            or not received_lead_seconds.is_integer()
            or received_lead_seconds
            < MINIMUM_REMOTE_REGISTRATION_LEAD_SECONDS
        ):
            raise shadow.ShadowPredictionError(
                "L'instant local de reception GitHub ne confirme pas la "
                "frontiere prospective du scoring."
            )
    seconds_value = (earliest - remote_datetime).total_seconds()
    if not seconds_value.is_integer():
        raise shadow.ShadowPredictionError(
            "L'avance distante du scoring doit etre un nombre entier de secondes."
        )
    seconds = int(seconds_value)
    if seconds < MINIMUM_REMOTE_REGISTRATION_LEAD_SECONDS:
        raise shadow.ShadowPredictionError(
            "Le protocole de scoring n'est pas visible sur GitHub au moins "
            "60 minutes avant le premier match. Aucun scoring officiel 2026 "
            "n'est autorise sous ce protocole."
        )
    minutes = (Decimal(seconds) / Decimal(60)).quantize(
        Decimal("0.000001"),
        rounding=ROUND_HALF_EVEN,
    )
    return seconds, format(minutes, ".6f")


def _validate_registration(
    registration: Mapping[str, Any],
    *,
    authority: ScoringProtocolAuthority,
    evidence: shadow.ShadowActivationReverificationEvidence,
) -> bytes:
    if type(registration) is not dict or frozenset(registration) != _REGISTRATION_KEYS:
        raise shadow.ShadowPredictionError(
            "Le recu de preenregistrement ne respecte pas son schema exact."
        )
    _, lead_minutes = _remote_registration_lead(
        evidence.activation_remote_reverified_at_utc,
        evidence.response_received_at_utc,
    )
    exact_values: dict[str, object] = {
        "registration_schema_version": 1,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "status": REGISTRATION_STATUS,
        "claim_level": REGISTRATION_CLAIM_LEVEL,
        "scoring_protocol_path": SCORING_PROTOCOL_RELATIVE_PATH.as_posix(),
        "scoring_protocol_sha256": authority.protocol_sha256,
        "scoring_protocol_introduction_commit": authority.protocol_introduction_commit,
        "registration_service_path": REGISTRATION_SERVICE_RELATIVE_PATH.as_posix(),
        "registration_service_sha256": authority.registration_service_sha256,
        "runtime_code_commit": authority.runtime_code_commit,
        "remote_ref": shadow.GITHUB_REMOTE_REF,
        "remote_query_url": evidence.raw_evidence["request_url"],
        "remote_effective_url": evidence.raw_evidence["effective_url"],
        "remote_response_status_code": 200,
        "remote_response_redirect_count": 0,
        "remote_http_date_utc": evidence.activation_remote_reverified_at_utc,
        "remote_response_received_at_utc": evidence.response_received_at_utc,
        "remote_response_body_sha256": evidence.response_body_sha256,
        "raw_remote_evidence_path": RAW_REMOTE_EVIDENCE_RELATIVE_PATH.as_posix(),
        "raw_remote_evidence_sha256": evidence.canonical_gzip_sha256,
        "first_target_official_date": EXPECTED_FIRST_TARGET_DATE,
        "first_batch_id": EXPECTED_FIRST_BATCH_ID,
        "first_results_commit": EXPECTED_FIRST_RESULTS_COMMIT,
        "first_certification_commit": EXPECTED_FIRST_CERTIFICATION_COMMIT,
        "first_certification_sha256": EXPECTED_FIRST_CERTIFICATION_SHA256,
        "first_predictions_sha256": EXPECTED_FIRST_PREDICTIONS_SHA256,
        "first_receipt_sha256": EXPECTED_FIRST_RECEIPT_SHA256,
        "first_earliest_predicted_start_utc": EXPECTED_FIRST_START_UTC,
        "remote_publication_lead_minutes": lead_minutes,
        "registered_at_utc": evidence.response_received_at_utc,
    }
    for key, expected in exact_values.items():
        actual = registration.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise shadow.ShadowPredictionError(
                f"Valeur du recu de preenregistrement invalide pour {key}."
            )
    attestations = registration.get("negative_attestations")
    if (
        type(attestations) is not dict
        or frozenset(attestations) != _NEGATIVE_ATTESTATION_KEYS
        or any(value is not True for value in attestations.values())
    ):
        raise shadow.ShadowPredictionError(
            "Les attestations negatives du preenregistrement sont invalides."
        )
    shadow._validate_activation_reverification_evidence(evidence)
    if evidence.activation_introduction_commit != (
        authority.protocol_introduction_commit
    ):
        raise shadow.ShadowPredictionError(
            "La preuve GitHub ne nomme pas le commit exact du protocole."
        )
    return shadow._canonical_json_file_bytes(dict(registration))


def _publish_registration(
    authority: ScoringProtocolAuthority,
    evidence: shadow.ShadowActivationReverificationEvidence,
    *,
    raw_evidence_path: Path,
    registration_path: Path,
) -> ScoringRegistrationPublication:
    """Publie raw puis JSON; un orphelin est definitif et jamais repare."""
    _, lead_minutes = _remote_registration_lead(
        evidence.activation_remote_reverified_at_utc,
        evidence.response_received_at_utc,
    )
    registration: dict[str, Any] = {
        "registration_schema_version": 1,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "status": REGISTRATION_STATUS,
        "claim_level": REGISTRATION_CLAIM_LEVEL,
        "scoring_protocol_path": SCORING_PROTOCOL_RELATIVE_PATH.as_posix(),
        "scoring_protocol_sha256": authority.protocol_sha256,
        "scoring_protocol_introduction_commit": authority.protocol_introduction_commit,
        "registration_service_path": REGISTRATION_SERVICE_RELATIVE_PATH.as_posix(),
        "registration_service_sha256": authority.registration_service_sha256,
        "runtime_code_commit": authority.runtime_code_commit,
        "remote_ref": shadow.GITHUB_REMOTE_REF,
        "remote_query_url": evidence.raw_evidence["request_url"],
        "remote_effective_url": evidence.raw_evidence["effective_url"],
        "remote_response_status_code": 200,
        "remote_response_redirect_count": 0,
        "remote_http_date_utc": evidence.activation_remote_reverified_at_utc,
        "remote_response_received_at_utc": evidence.response_received_at_utc,
        "remote_response_body_sha256": evidence.response_body_sha256,
        "raw_remote_evidence_path": RAW_REMOTE_EVIDENCE_RELATIVE_PATH.as_posix(),
        "raw_remote_evidence_sha256": evidence.canonical_gzip_sha256,
        "first_target_official_date": EXPECTED_FIRST_TARGET_DATE,
        "first_batch_id": EXPECTED_FIRST_BATCH_ID,
        "first_results_commit": EXPECTED_FIRST_RESULTS_COMMIT,
        "first_certification_commit": EXPECTED_FIRST_CERTIFICATION_COMMIT,
        "first_certification_sha256": EXPECTED_FIRST_CERTIFICATION_SHA256,
        "first_predictions_sha256": EXPECTED_FIRST_PREDICTIONS_SHA256,
        "first_receipt_sha256": EXPECTED_FIRST_RECEIPT_SHA256,
        "first_earliest_predicted_start_utc": EXPECTED_FIRST_START_UTC,
        "remote_publication_lead_minutes": lead_minutes,
        "registered_at_utc": evidence.response_received_at_utc,
        "negative_attestations": {
            key: True for key in sorted(_NEGATIVE_ATTESTATION_KEYS)
        },
    }
    registration_bytes = _validate_registration(
        registration,
        authority=authority,
        evidence=evidence,
    )
    raw_sha256 = shadow._publish_exclusive_verified(
        raw_evidence_path,
        evidence.canonical_gzip_bytes,
    )
    if raw_sha256 != evidence.canonical_gzip_sha256:
        raise shadow.ShadowPredictionError(
            "La preuve GitHub publiee diverge de ses octets prepares."
        )
    registration_sha256 = shadow._publish_exclusive_verified(
        registration_path,
        registration_bytes,
    )
    return ScoringRegistrationPublication(
        raw_evidence_path=raw_evidence_path,
        raw_evidence_relative_path=RAW_REMOTE_EVIDENCE_RELATIVE_PATH.as_posix(),
        raw_evidence_sha256=raw_sha256,
        raw_evidence_size_bytes=len(evidence.canonical_gzip_bytes),
        registration_path=registration_path,
        registration_relative_path=REGISTRATION_RELATIVE_PATH.as_posix(),
        registration_sha256=registration_sha256,
        registration_size_bytes=len(registration_bytes),
        protocol_introduction_commit=authority.protocol_introduction_commit,
        runtime_code_commit=authority.runtime_code_commit,
        remote_http_date_utc=evidence.activation_remote_reverified_at_utc,
        remote_publication_lead_minutes=lead_minutes,
        registration=registration,
    )


def register_scoring_protocol(
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ScoringRegistrationPublication:
    """Preenregistre une seule fois, avant toute lecture de resultat sportif."""
    project = shadow._preflight_project_directory(project_directory)
    authority = verify_scoring_protocol_authority(project_directory=project)
    raw_path, registration_path = _require_absent_registration_paths(project)

    # Unique acces reseau, seulement apres tous les controles locaux. Cette
    # requete vise GitHub, jamais MLB, et ne lit aucun resultat sportif.
    evidence = shadow.fetch_activation_reverification_evidence(
        authority.protocol_introduction_commit
    )
    _remote_registration_lead(
        evidence.activation_remote_reverified_at_utc,
        evidence.response_received_at_utc,
    )
    return _publish_registration(
        authority,
        evidence,
        raw_evidence_path=raw_path,
        registration_path=registration_path,
    )


def _canonical_publication_output(
    publication: ScoringRegistrationPublication,
) -> bytes:
    result = shadow._read_canonical_json_object(publication.registration_path)
    if result is None:
        raise shadow.ShadowPredictionError(
            "Le preenregistrement publie ne peut pas etre relu."
        )
    registration, content = result
    if (
        registration != publication.registration
        or hashlib.sha256(content).hexdigest() != publication.registration_sha256
        or len(content) != publication.registration_size_bytes
    ):
        raise shadow.ShadowPredictionError(
            "Le preenregistrement publie diverge de sa preuve de sortie."
        )
    return content


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.shadow_scoring_registration",
        description=(
            "Preenregistre le protocole de scoring MLB 2026 sur GitHub sans "
            "lire de resultat, SQLite ou modele."
        ),
    )
    parser.add_argument(
        "--register-scoring",
        action="store_true",
        help="Autorise explicitement les deux publications append-only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)
    if not arguments.register_scoring:
        parser.error(
            "--register-scoring est obligatoire pour figer le protocole."
        )
    try:
        publication = register_scoring_protocol()
        output = _canonical_publication_output(publication)
    except shadow.ShadowPredictionError as error:
        parser.error(str(error))
    sys.stdout.buffer.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
