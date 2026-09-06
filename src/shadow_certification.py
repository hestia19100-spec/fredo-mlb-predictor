"""Certification prospective distante des lots fantomes MLB v2.

Ce module est volontairement separe du moteur de prediction active. Le
manifeste d'execution a fige les octets de ``src/shadow_prediction.py`` avant
l'activation : la certification peut donc etre ajoutee sans changer le moteur,
ses variables, son modele ou une preuve deja publiee.

La certification ne s'execute qu'apres qu'un lot COMPLETED a ete commite et
pousse. Elle relit les huit blobs depuis le commit Git fourni, obtient une
preuve HTTPS anonyme que ce commit est visible sur ``main``, controle une
avance distante d'au moins soixante minutes, puis publie exactement deux
fichiers append-only hors du repertoire du lot. Elle ne lit ni MLB, ni SQLite,
ni joblib et ne calcule aucune nouvelle prediction.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import io
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence

from src import shadow_prediction as shadow


PROJECT_DIRECTORY = shadow.PROJECT_DIRECTORY
SHADOW_CERTIFICATION_ROOT_RELATIVE_PATH = (
    shadow.SHADOW_CERTIFICATION_ROOT_RELATIVE_PATH
)
EXPECTED_RESULT_FILENAMES = tuple(sorted(shadow.SHADOW_SLOT_SUCCESS_FILENAMES))
CERTIFICATION_STATUS = "PROSPECTIVELY_CERTIFIED_REMOTE_MAIN_BEFORE_GAMES"
CERTIFICATION_CLAIM_LEVEL = (
    "REMOTE_SERVER_ATTESTED_NOT_CRYPTOGRAPHICALLY_TIMESTAMPED"
)
MINIMUM_REMOTE_PUBLICATION_LEAD_SECONDS = 3600

_CERTIFICATION_KEYS = frozenset(
    {
        "certification_schema_version",
        "target_official_date",
        "batch_id",
        "results_commit",
        "results_remote_ref",
        "results_tree_file_hashes",
        "remote_query_url",
        "remote_effective_url",
        "remote_response_status_code",
        "remote_response_redirect_count",
        "remote_http_date_utc",
        "remote_response_received_at_utc",
        "remote_response_body_sha256",
        "raw_remote_evidence_path",
        "raw_remote_evidence_sha256",
        "earliest_predicted_start_utc",
        "remote_publication_lead_minutes",
        "status",
        "claim_level",
    }
)
_TREE_HASH_KEYS = frozenset({"path", "sha256", "size_bytes"})
_OUTPUT_HASH_TO_FILENAME = {
    "activation_reverification_evidence_sha256": (
        shadow.ACTIVATION_REVERIFICATION_FILENAME
    ),
    "candidate_ledger_sha256": shadow.CANDIDATE_LEDGER_FILENAME,
    "features_sha256": shadow.FEATURES_FILENAME,
    "predictions_sha256": shadow.PREDICTIONS_FILENAME,
}


@dataclass(frozen=True, slots=True)
class CompletedShadowBatchCommit:
    """Lot terminal relu exclusivement depuis ses blobs Git."""

    target_official_date: str
    results_commit: str
    batch_id: str
    earliest_predicted_start_utc: str
    results_tree_file_hashes: tuple[dict[str, object], ...]
    receipt: dict[str, Any] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ShadowCertificationPublication:
    """Preuve locale des deux fichiers de certification append-only."""

    certification_path: Path
    certification_relative_path: str
    certification_sha256: str
    certification_size_bytes: int
    raw_evidence_path: Path
    raw_evidence_relative_path: str
    raw_evidence_sha256: str
    raw_evidence_size_bytes: int
    target_official_date: str
    batch_id: str
    results_commit: str
    remote_http_date_utc: str
    remote_publication_lead_minutes: str
    certification: dict[str, Any] = field(repr=False)


def _certification_relative_paths(
    target_official_date: str,
) -> tuple[PurePosixPath, PurePosixPath]:
    target = shadow._require_date_string(
        target_official_date,
        field="target_official_date",
    )
    return (
        SHADOW_CERTIFICATION_ROOT_RELATIVE_PATH / f"{target}.remote.json.gz",
        SHADOW_CERTIFICATION_ROOT_RELATIVE_PATH / f"{target}.json",
    )


def _require_absent_certification_paths(
    project_directory: Path,
    target_official_date: str,
) -> tuple[Path, Path, str, str]:
    """Refuse avant le reseau tout succes, orphelin ou chemin substitue."""
    raw_relative, certification_relative = _certification_relative_paths(
        target_official_date
    )
    shadow._require_tracked_nonignored_root(
        project_directory,
        SHADOW_CERTIFICATION_ROOT_RELATIVE_PATH,
    )
    for relative in (raw_relative, certification_relative):
        return_code, _ = shadow._run_preflight_git(
            project_directory,
            ("check-ignore", "-q", "--", relative.as_posix()),
            accepted_return_codes=frozenset({0, 1}),
        )
        if return_code == 0:
            raise shadow.ShadowPredictionError(
                "Un chemin de certification est interdit par .gitignore."
            )

    raw_path = project_directory.joinpath(*raw_relative.parts)
    certification_path = project_directory.joinpath(
        *certification_relative.parts
    )
    if (
        shadow._lstat_mode(raw_path) is not shadow._PATH_MISSING
        or shadow._lstat_mode(certification_path) is not shadow._PATH_MISSING
    ):
        raise shadow.ShadowPublicationConflictError(
            "La certification de cette date est deja consommee et ne peut "
            "etre relue, ecrasee, reparee ou retentee."
        )
    return (
        raw_path,
        certification_path,
        raw_relative.as_posix(),
        certification_relative.as_posix(),
    )


def _git_lines(
    project_directory: Path,
    arguments: tuple[str, ...],
    *,
    field: str,
) -> tuple[str, ...]:
    _, output = shadow._run_preflight_git(project_directory, arguments)
    return tuple(
        line
        for line in shadow._git_text(output, field=field).splitlines()
        if line
    )


def _require_exact_results_commit_head(
    project_directory: Path,
    expected_results_commit: str,
) -> str:
    expected = shadow._require_git_commit(
        expected_results_commit,
        field="expected_results_commit",
    )
    head = shadow._require_exact_clean_git_root(project_directory)
    if head != expected:
        raise shadow.ShadowPredictionError(
            "Le commit de resultats attendu doit etre exactement le HEAD "
            "local propre avant certification."
        )
    return head


def _read_git_json_blob(
    blobs: Mapping[str, bytes],
    filename: str,
) -> tuple[dict[str, Any], bytes]:
    content = blobs.get(filename)
    if type(content) is not bytes:
        raise shadow.ShadowPredictionError(
            f"Blob Git absent pour {filename}."
        )
    return (
        shadow._read_canonical_json_bytes(
            content,
            description=f"blob Git {filename}",
        ),
        content,
    )


def _read_canonical_predictions_blob(
    content: bytes,
    *,
    batch_id: str,
    target_official_date: str,
    expected_row_count: int,
) -> str:
    """Relit le CSV Git canonique et retourne son premier horaire."""
    if type(content) is not bytes:
        raise shadow.ShadowPredictionError(
            "Le blob Git predictions.csv doit etre binaire."
        )
    try:
        decoded = content.decode("utf-8")
        parsed = list(csv.reader(io.StringIO(decoded, newline="")))
    except (UnicodeError, csv.Error) as error:
        raise shadow.ShadowPredictionError(
            "Le blob Git predictions.csv est invalide."
        ) from error
    if not parsed or tuple(parsed[0]) != shadow._PREDICTIONS_COLUMNS:
        raise shadow.ShadowPredictionError(
            "Le schema Git de predictions.csv est invalide."
        )
    rows = parsed[1:]
    if (
        type(expected_row_count) is not int
        or expected_row_count <= 0
        or len(rows) != expected_row_count
        or shadow._canonical_csv_bytes(shadow._PREDICTIONS_COLUMNS, rows)
        != content
    ):
        raise shadow.ShadowPredictionError(
            "Le nombre ou les octets Git de predictions.csv sont invalides."
        )

    prediction_ids: set[str] = set()
    game_ids: set[str] = set()
    starts: list[str] = []
    for index, row in enumerate(rows, start=1):
        if len(row) != len(shadow._PREDICTIONS_COLUMNS):
            raise shadow.ShadowPredictionError(
                f"Largeur invalide dans predictions.csv a la ligne {index}."
            )
        if row[1] != batch_id or row[5] != target_official_date:
            raise shadow.ShadowPredictionError(
                "Une prediction Git ne correspond pas au lot certifie."
            )
        if row[0] in prediction_ids or row[2] in game_ids:
            raise shadow.ShadowPredictionError(
                "Les identifiants Git de predictions.csv sont en doublon."
            )
        prediction_ids.add(row[0])
        game_ids.add(row[2])
        starts.append(
            shadow._require_utc_timestamp(
                row[8],
                field=f"predictions[{index}].scheduled_start_utc",
            )
        )
    return min(starts)


def _validate_committed_batch_blobs(
    *,
    project_directory: Path,
    target_official_date: str,
    results_commit: str,
    authority: shadow.ShadowExecutionAuthority,
    blobs: Mapping[str, bytes],
) -> CompletedShadowBatchCommit:
    """Controle les liaisons terminales a partir des seuls blobs Git."""
    slot_key = shadow.build_slot_key(
        shadow_protocol_sha256=authority.shadow_protocol_sha256,
        target_official_date=target_official_date,
    )
    batch_id = shadow.build_batch_id(
        slot_key=slot_key,
        execution_manifest_sha256=authority.execution_manifest_sha256,
        model_artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
    )
    reserved, _reserved_bytes = _read_git_json_blob(blobs, "RESERVED")
    receipt, receipt_bytes = _read_git_json_blob(
        blobs,
        shadow.RECEIPT_FILENAME,
    )
    completed, _completed_bytes = _read_git_json_blob(
        blobs,
        shadow.COMPLETED_FILENAME,
    )
    if not shadow._valid_reserved_marker(
        reserved,
        batch_id=batch_id,
        slot_key=slot_key,
        target_official_date=target_official_date,
        shadow_protocol_sha256=authority.shadow_protocol_sha256,
        execution_manifest_sha256=authority.execution_manifest_sha256,
    ):
        raise shadow.ShadowPredictionError(
            "Le marqueur RESERVED du commit de resultats est invalide."
        )
    if not shadow._valid_receipt(
        receipt,
        reserved=reserved,
        batch_id=batch_id,
        slot_key=slot_key,
        target_official_date=target_official_date,
        shadow_protocol_sha256=authority.shadow_protocol_sha256,
        execution_manifest_sha256=authority.execution_manifest_sha256,
        model_artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
    ):
        raise shadow.ShadowPredictionError(
            "Le recu du commit de resultats est invalide."
        )
    receipt_path = (
        shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH
        / target_official_date
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

    batch = receipt["batch"]
    counts = receipt["counts"]
    output_hashes = receipt["output_hashes"]
    source = receipt["source"]
    lineage = receipt["lineage"]
    assert all(
        isinstance(value, dict)
        for value in (batch, counts, output_hashes, source, lineage)
    )
    if (
        batch.get("status") != "COMPLETED_WITH_PREDICTIONS"
        or type(counts.get("predicted_games")) is not int
        or counts["predicted_games"] <= 0
    ):
        raise shadow.ShadowPredictionError(
            "Un lot vide ne peut jamais recevoir de certification."
        )

    for receipt_field, filename in _OUTPUT_HASH_TO_FILENAME.items():
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

    predictions = blobs.get(shadow.PREDICTIONS_FILENAME)
    if type(predictions) is not bytes:
        raise shadow.ShadowPredictionError(
            "Le blob Git predictions.csv est absent."
        )
    earliest = _read_canonical_predictions_blob(
        predictions,
        batch_id=batch_id,
        target_official_date=target_official_date,
        expected_row_count=counts["predicted_games"],
    )
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
        description="execution locale vers commit de resultats",
    )

    tree_hashes = tuple(
        {
            "path": filename,
            "sha256": hashlib.sha256(blobs[filename]).hexdigest(),
            "size_bytes": len(blobs[filename]),
        }
        for filename in EXPECTED_RESULT_FILENAMES
    )
    return CompletedShadowBatchCommit(
        target_official_date=target_official_date,
        results_commit=results_commit,
        batch_id=batch_id,
        earliest_predicted_start_utc=earliest,
        results_tree_file_hashes=tree_hashes,
        receipt=receipt,
    )


def _read_completed_batch_commit(
    target_official_date: str,
    results_commit: str,
    *,
    project_directory: Path,
) -> CompletedShadowBatchCommit:
    """Relit un lot propre, unique et immuable depuis le commit HEAD."""
    authority = shadow.verify_shadow_execution_authority(
        target_official_date,
        project_directory=project_directory,
    )
    inspection = shadow.inspect_shadow_prediction_slot(
        target_official_date,
        shadow_protocol_sha256=authority.shadow_protocol_sha256,
        execution_manifest_sha256=authority.execution_manifest_sha256,
        model_artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
        project_directory=project_directory,
    )
    if inspection.state is not shadow.ShadowPredictionSlotState.COMPLETED_EXACT:
        raise shadow.ShadowPredictionError(
            "Le slot local doit etre un lot COMPLETED exact avant certification."
        )

    prefix = (
        shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH / target_official_date
    ).as_posix()
    expected_paths = tuple(f"{prefix}/{name}" for name in EXPECTED_RESULT_FILENAMES)
    actual_paths = _git_lines(
        project_directory,
        ("ls-tree", "-r", "--name-only", results_commit, "--", prefix),
        field="arbre Git du lot de resultats",
    )
    if actual_paths != expected_paths:
        raise shadow.ShadowPredictionError(
            "Le commit de resultats ne contient pas exactement les huit "
            "fichiers attendus."
        )
    history = _git_lines(
        project_directory,
        ("log", "--format=%H", "--", prefix),
        field="historique Git du lot de resultats",
    )
    if history != (results_commit,):
        raise shadow.ShadowPredictionError(
            "Le lot doit etre introduit une seule fois par le commit de "
            "resultats et ne jamais avoir ete modifie."
        )
    blobs = {
        filename: shadow._git_blob_at_commit(
            project_directory,
            results_commit,
            f"{prefix}/{filename}",
        )
        for filename in EXPECTED_RESULT_FILENAMES
    }
    return _validate_committed_batch_blobs(
        project_directory=project_directory,
        target_official_date=target_official_date,
        results_commit=results_commit,
        authority=authority,
        blobs=blobs,
    )


def _remote_publication_lead(
    earliest_predicted_start_utc: str,
    remote_http_date_utc: str,
) -> tuple[int, str]:
    earliest = shadow._require_utc_timestamp(
        earliest_predicted_start_utc,
        field="earliest_predicted_start_utc",
    )
    remote = shadow._require_utc_timestamp(
        remote_http_date_utc,
        field="remote_http_date_utc",
    )
    earliest_datetime = datetime.fromisoformat(earliest.replace("Z", "+00:00"))
    remote_datetime = datetime.fromisoformat(remote.replace("Z", "+00:00"))
    lead_seconds_value = (earliest_datetime - remote_datetime).total_seconds()
    if not lead_seconds_value.is_integer():
        raise shadow.ShadowPredictionError(
            "L'avance distante doit etre un nombre entier de secondes."
        )
    lead_seconds = int(lead_seconds_value)
    if lead_seconds < MINIMUM_REMOTE_PUBLICATION_LEAD_SECONDS:
        raise shadow.ShadowPredictionError(
            "La preuve GitHub arrive moins de 60 minutes avant le premier "
            "match : ce lot reste LOCAL_SHADOW_ONLY_NOT_PROSPECTIVELY_CERTIFIED."
        )
    lead_minutes = (
        Decimal(lead_seconds) / Decimal(60)
    ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
    return lead_seconds, format(lead_minutes, ".6f")


def _validate_prepared_certification(
    certification: Mapping[str, Any],
    *,
    batch: CompletedShadowBatchCommit,
    evidence: shadow.ShadowActivationReverificationEvidence,
    raw_evidence_relative_path: str,
) -> bytes:
    if not shadow._has_exact_keys(certification, _CERTIFICATION_KEYS):
        raise shadow.ShadowPredictionError(
            "La certification ne respecte pas le schema exact v2."
        )
    expected_values: dict[str, object] = {
        "certification_schema_version": 1,
        "target_official_date": batch.target_official_date,
        "batch_id": batch.batch_id,
        "results_commit": batch.results_commit,
        "results_remote_ref": shadow.GITHUB_REMOTE_REF,
        "remote_query_url": evidence.raw_evidence["request_url"],
        "remote_effective_url": evidence.raw_evidence["effective_url"],
        "remote_response_status_code": 200,
        "remote_response_redirect_count": 0,
        "remote_http_date_utc": evidence.activation_remote_reverified_at_utc,
        "remote_response_received_at_utc": evidence.response_received_at_utc,
        "remote_response_body_sha256": evidence.response_body_sha256,
        "raw_remote_evidence_path": raw_evidence_relative_path,
        "raw_remote_evidence_sha256": evidence.canonical_gzip_sha256,
        "earliest_predicted_start_utc": batch.earliest_predicted_start_utc,
        "status": CERTIFICATION_STATUS,
        "claim_level": CERTIFICATION_CLAIM_LEVEL,
    }
    for key, expected in expected_values.items():
        actual = certification.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise shadow.ShadowPredictionError(
                f"Valeur de certification invalide pour {key}."
            )

    entries = certification.get("results_tree_file_hashes")
    if type(entries) is not list or tuple(entries) != batch.results_tree_file_hashes:
        raise shadow.ShadowPredictionError(
            "Les empreintes de l'arbre Git ne sont pas exactes."
        )
    paths: list[str] = []
    for entry in entries:
        if not shadow._has_exact_keys(entry, _TREE_HASH_KEYS):
            raise shadow.ShadowPredictionError(
                "Une entree d'empreinte Git est invalide."
            )
        assert isinstance(entry, dict)
        paths.append(entry["path"])
        shadow._require_sha256(entry["sha256"], field="tree.sha256")
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0:
            raise shadow.ShadowPredictionError(
                "Une taille de blob Git est invalide."
            )
    if tuple(paths) != EXPECTED_RESULT_FILENAMES:
        raise shadow.ShadowPredictionError(
            "L'ordre des huit blobs Git est invalide."
        )

    _lead_seconds, expected_minutes = _remote_publication_lead(
        batch.earliest_predicted_start_utc,
        evidence.activation_remote_reverified_at_utc,
    )
    if certification.get("remote_publication_lead_minutes") != expected_minutes:
        raise shadow.ShadowPredictionError(
            "L'avance distante formatee est incoherente."
        )
    shadow._validate_activation_reverification_evidence(evidence)
    if evidence.activation_introduction_commit != batch.results_commit:
        raise shadow.ShadowPredictionError(
            "La preuve GitHub ne nomme pas le commit de resultats exact."
        )
    return shadow._canonical_json_file_bytes(dict(certification))


def _publish_prepared_certification(
    *,
    batch: CompletedShadowBatchCommit,
    evidence: shadow.ShadowActivationReverificationEvidence,
    raw_evidence_path: Path,
    certification_path: Path,
    raw_evidence_relative_path: str,
    certification_relative_path: str,
) -> ShadowCertificationPublication:
    """Publie raw puis JSON; tout orphelin reste definitif et visible."""
    _lead_seconds, lead_minutes = _remote_publication_lead(
        batch.earliest_predicted_start_utc,
        evidence.activation_remote_reverified_at_utc,
    )
    certification: dict[str, Any] = {
        "certification_schema_version": 1,
        "target_official_date": batch.target_official_date,
        "batch_id": batch.batch_id,
        "results_commit": batch.results_commit,
        "results_remote_ref": shadow.GITHUB_REMOTE_REF,
        "results_tree_file_hashes": [
            dict(entry) for entry in batch.results_tree_file_hashes
        ],
        "remote_query_url": evidence.raw_evidence["request_url"],
        "remote_effective_url": evidence.raw_evidence["effective_url"],
        "remote_response_status_code": 200,
        "remote_response_redirect_count": 0,
        "remote_http_date_utc": evidence.activation_remote_reverified_at_utc,
        "remote_response_received_at_utc": evidence.response_received_at_utc,
        "remote_response_body_sha256": evidence.response_body_sha256,
        "raw_remote_evidence_path": raw_evidence_relative_path,
        "raw_remote_evidence_sha256": evidence.canonical_gzip_sha256,
        "earliest_predicted_start_utc": batch.earliest_predicted_start_utc,
        "remote_publication_lead_minutes": lead_minutes,
        "status": CERTIFICATION_STATUS,
        "claim_level": CERTIFICATION_CLAIM_LEVEL,
    }
    certification_bytes = _validate_prepared_certification(
        certification,
        batch=batch,
        evidence=evidence,
        raw_evidence_relative_path=raw_evidence_relative_path,
    )

    raw_sha256 = shadow._publish_exclusive_verified(
        raw_evidence_path,
        evidence.canonical_gzip_bytes,
    )
    if raw_sha256 != evidence.canonical_gzip_sha256:
        raise shadow.ShadowPredictionError(
            "La preuve GitHub publiee diverge de ses octets prepares."
        )
    certification_sha256 = shadow._publish_exclusive_verified(
        certification_path,
        certification_bytes,
    )
    return ShadowCertificationPublication(
        certification_path=certification_path,
        certification_relative_path=certification_relative_path,
        certification_sha256=certification_sha256,
        certification_size_bytes=len(certification_bytes),
        raw_evidence_path=raw_evidence_path,
        raw_evidence_relative_path=raw_evidence_relative_path,
        raw_evidence_sha256=raw_sha256,
        raw_evidence_size_bytes=len(evidence.canonical_gzip_bytes),
        target_official_date=batch.target_official_date,
        batch_id=batch.batch_id,
        results_commit=batch.results_commit,
        remote_http_date_utc=evidence.activation_remote_reverified_at_utc,
        remote_publication_lead_minutes=lead_minutes,
        certification=certification,
    )


def certify_shadow_prediction(
    target_official_date: str,
    expected_results_commit: str,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowCertificationPublication:
    """Certifie une fois un lot deja commite et visible sur GitHub main."""
    target = shadow._parse_target_official_date(target_official_date)
    if target.year != shadow.EXPECTED_TARGET_SEASON:
        raise shadow.ShadowPredictionError(
            "La certification exige exactement une date de la saison 2026."
        )
    target_text = target.isoformat()
    project = shadow._preflight_project_directory(project_directory)
    (
        raw_path,
        certification_path,
        raw_relative,
        certification_relative,
    ) = _require_absent_certification_paths(project, target_text)
    results_commit = _require_exact_results_commit_head(
        project,
        expected_results_commit,
    )
    batch = _read_completed_batch_commit(
        target_text,
        results_commit,
        project_directory=project,
    )

    # L'unique acces reseau intervient apres tous les controles locaux et
    # avant le premier octet de certification.
    evidence = shadow.fetch_activation_reverification_evidence(results_commit)
    _remote_publication_lead(
        batch.earliest_predicted_start_utc,
        evidence.activation_remote_reverified_at_utc,
    )
    return _publish_prepared_certification(
        batch=batch,
        evidence=evidence,
        raw_evidence_path=raw_path,
        certification_path=certification_path,
        raw_evidence_relative_path=raw_relative,
        certification_relative_path=certification_relative,
    )


def _canonical_certification_output_bytes(
    publication: ShadowCertificationPublication,
) -> bytes:
    result = shadow._read_canonical_json_object(publication.certification_path)
    if result is None:
        raise shadow.ShadowPredictionError(
            "La certification publiee ne peut pas etre relue."
        )
    certification, content = result
    if (
        certification != publication.certification
        or hashlib.sha256(content).hexdigest()
        != publication.certification_sha256
        or len(content) != publication.certification_size_bytes
    ):
        raise shadow.ShadowPredictionError(
            "La certification publiee diverge de sa preuve de sortie."
        )
    return content


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.shadow_certification",
        description=(
            "Certifie sur GitHub un lot fantome v2 deja termine, commite et "
            "pousse, sans lire MLB, SQLite ou le modele."
        ),
    )
    parser.add_argument(
        "--certify-shadow",
        action="store_true",
        help="Demande explicitement la publication append-only.",
    )
    parser.add_argument(
        "--target-official-date",
        metavar="YYYY-MM-DD",
        required=True,
        help="Date officielle MLB 2026 du lot termine.",
    )
    parser.add_argument(
        "--expected-results-commit",
        metavar="SHA40",
        required=True,
        help="Commit HEAD exact contenant les huit fichiers du lot.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)
    if not arguments.certify_shadow:
        parser.error("--certify-shadow est obligatoire pour certifier un lot.")
    try:
        publication = certify_shadow_prediction(
            arguments.target_official_date,
            arguments.expected_results_commit,
        )
        output = _canonical_certification_output_bytes(publication)
    except shadow.ShadowPredictionError as error:
        parser.error(str(error))
    sys.stdout.buffer.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
