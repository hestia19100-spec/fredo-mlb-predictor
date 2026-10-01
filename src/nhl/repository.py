"""Journal transactionnel et auditable des observations synthétiques NHL."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from src.nhl.database import (
    NHLDatabaseError,
    connect_nhl_database,
    connect_nhl_database_read_only,
    initialize_nhl_database,
    verify_nhl_database,
)
from src.nhl.offline_sources import (
    NormalizedObservation,
    OfflineFixtureBundle,
    ValueState,
)


class NHLRepositoryError(RuntimeError):
    """Signale un conflit avec le journal NHL immuable."""


@dataclass(frozen=True, slots=True)
class ArchiveResult:
    run_id: int
    created: bool
    observation_count: int
    bundle_sha256: str


@dataclass(frozen=True, slots=True)
class StorageAudit:
    schema_valid: bool
    run_count: int
    observation_count: int
    unknown_count: int
    providers: tuple[str, ...]
    bundle_sha256s: tuple[str, ...]


def _utc_text(value: datetime, *, field_name: str) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise NHLRepositoryError(f"{field_name} doit être exprimé en UTC.")
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise NHLRepositoryError("Le document NHL n'est pas un JSON canonique.") from error


def _sha256_document(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _observation_document(observation: NormalizedObservation) -> dict[str, Any]:
    evidence = observation.evidence
    return {
        "code_commit": evidence.code_commit,
        "effective_available_at_utc": _utc_text(
            observation.effective_available_at_utc,
            field_name="effective_available_at_utc",
        ),
        "entity_id": observation.entity_id,
        "fixture_id": observation.fixture_id,
        "kind": observation.kind.value,
        "observation_id": observation.observation_id,
        "observed_at_utc": _utc_text(
            evidence.observed_at_utc,
            field_name="observed_at_utc",
        ),
        "provider": evidence.provider,
        "request_sha256": observation.request_sha256,
        "response_sha256": evidence.response_sha256,
        "source_game_id": observation.source_game_id,
        "source_updated_at_utc": (
            None
            if evidence.source_updated_at_utc is None
            else _utc_text(
                evidence.source_updated_at_utc,
                field_name="source_updated_at_utc",
            )
        ),
        "target_game_id": observation.target_game_id,
        "temporal_status": observation.temporal_status.value,
        "value": observation.value,
        "value_state": observation.value_state.value,
    }


def _proof_document(bundle: OfflineFixtureBundle) -> dict[str, Any]:
    proof = bundle.proof
    return {
        "capability": proof.capability.value,
        "code_commit": proof.code_commit,
        "effective_available_at_utc": _utc_text(
            proof.effective_available_at_utc,
            field_name="effective_available_at_utc",
        ),
        "fixture_id": proof.fixture_id,
        "observed_at_utc": _utc_text(
            proof.observed_at_utc,
            field_name="observed_at_utc",
        ),
        "provider": proof.provider.value,
        "request_sha256": proof.request_sha256,
        "response_sha256": proof.response_sha256,
        "source_updated_at_utc": (
            None
            if proof.source_updated_at_utc is None
            else _utc_text(
                proof.source_updated_at_utc,
                field_name="source_updated_at_utc",
            )
        ),
    }


def _bundle_document(bundle: OfflineFixtureBundle) -> dict[str, Any]:
    return {
        "observations": [
            _observation_document(observation)
            for observation in bundle.observations
        ],
        "proof": _proof_document(bundle),
    }


def _validate_bundle_provenance(bundle: OfflineFixtureBundle) -> None:
    proof = bundle.proof
    for observation in bundle.observations:
        evidence = observation.evidence
        valid = (
            evidence.observation_id == observation.observation_id
            and evidence.provider == proof.provider.value
            and evidence.response_sha256 == proof.response_sha256
            and evidence.observed_at_utc == proof.observed_at_utc
            and evidence.code_commit == proof.code_commit
            and observation.fixture_id == proof.fixture_id
            and observation.request_sha256 == proof.request_sha256
        )
        if not valid:
            raise NHLRepositoryError("La preuve normalisée diverge du lot NHL.")


def _stored_observation_document(row: sqlite3.Row) -> dict[str, Any]:
    raw_value = row["canonical_value_json"]
    try:
        value = None if raw_value is None else json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise NHLRepositoryError("Une valeur archivée n'est plus un JSON valide.") from error
    return {
        "code_commit": row["code_commit"],
        "effective_available_at_utc": row["effective_available_at_utc"],
        "entity_id": row["entity_id"],
        "fixture_id": row["fixture_id"],
        "kind": row["kind"],
        "observation_id": row["observation_id"],
        "observed_at_utc": row["observed_at_utc"],
        "provider": row["provider"],
        "request_sha256": row["request_sha256"],
        "response_sha256": row["response_sha256"],
        "source_game_id": row["source_game_id"],
        "source_updated_at_utc": row["source_updated_at_utc"],
        "target_game_id": row["target_game_id"],
        "temporal_status": row["temporal_status"],
        "value": value,
        "value_state": row["value_state"],
    }


def archive_offline_fixture(
    bundle: OfflineFixtureBundle,
    *,
    database_path: Path,
    allowed_root: Path,
    archived_at_utc: datetime,
) -> ArchiveResult:
    """Archive une preuve synthétique une seule fois, sans donnée partielle."""

    if not isinstance(bundle, OfflineFixtureBundle):
        raise NHLRepositoryError("bundle doit être une OfflineFixtureBundle.")
    _validate_bundle_provenance(bundle)
    archived_text = _utc_text(archived_at_utc, field_name="archived_at_utc")
    if archived_at_utc < bundle.proof.effective_available_at_utc:
        raise NHLRepositoryError("L'archivage ne peut pas précéder la preuve source.")
    bundle_sha256 = _sha256_document(_bundle_document(bundle))
    try:
        initialize_nhl_database(database_path, allowed_root=allowed_root)
        with closing(
            connect_nhl_database(
                database_path,
                allowed_root=allowed_root,
            )
        ) as connection, connection:
            previous = connection.execute(
                """
                SELECT run_id, bundle_sha256
                FROM nhl_ingestion_runs
                WHERE fixture_id = ?
                """,
                (bundle.proof.fixture_id,),
            ).fetchone()
            if previous is not None:
                if previous["bundle_sha256"] != bundle_sha256:
                    raise NHLRepositoryError(
                        "Une fixture déjà archivée présente un autre contenu."
                    )
                count = connection.execute(
                    """
                    SELECT COUNT(*) FROM nhl_normalized_observations
                    WHERE run_id = ?
                    """,
                    (previous["run_id"],),
                ).fetchone()[0]
                if count != len(bundle.observations):
                    raise NHLRepositoryError("Le lot NHL archivé est incomplet.")
                return ArchiveResult(
                    run_id=previous["run_id"],
                    created=False,
                    observation_count=count,
                    bundle_sha256=bundle_sha256,
                )

            proof = bundle.proof
            cursor = connection.execute(
                """
                INSERT INTO nhl_ingestion_runs(
                    fixture_id, provider, capability, request_sha256,
                    response_sha256, bundle_sha256, observed_at_utc,
                    source_updated_at_utc, effective_available_at_utc,
                    code_commit, archived_at_utc, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'COMPLETED')
                """,
                (
                    proof.fixture_id,
                    proof.provider.value,
                    proof.capability.value,
                    proof.request_sha256,
                    proof.response_sha256,
                    bundle_sha256,
                    _utc_text(proof.observed_at_utc, field_name="observed_at_utc"),
                    (
                        None
                        if proof.source_updated_at_utc is None
                        else _utc_text(
                            proof.source_updated_at_utc,
                            field_name="source_updated_at_utc",
                        )
                    ),
                    _utc_text(
                        proof.effective_available_at_utc,
                        field_name="effective_available_at_utc",
                    ),
                    proof.code_commit,
                    archived_text,
                ),
            )
            run_id = cursor.lastrowid
            if run_id is None:
                raise NHLRepositoryError("SQLite n'a pas retourné de run_id.")

            for observation in bundle.observations:
                document = _observation_document(observation)
                value_json = (
                    None
                    if observation.value_state is ValueState.UNKNOWN
                    else _canonical_json(observation.value)
                )
                connection.execute(
                    """
                    INSERT INTO nhl_normalized_observations(
                        observation_id, observation_sha256, run_id, kind,
                        target_game_id, source_game_id, entity_id, value_state,
                        canonical_value_json, provider, fixture_id,
                        request_sha256, response_sha256, observed_at_utc,
                        source_updated_at_utc, effective_available_at_utc,
                        code_commit, temporal_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation.observation_id,
                        _sha256_document(document),
                        run_id,
                        observation.kind.value,
                        observation.target_game_id,
                        observation.source_game_id,
                        observation.entity_id,
                        observation.value_state.value,
                        value_json,
                        observation.evidence.provider,
                        observation.fixture_id,
                        observation.request_sha256,
                        observation.evidence.response_sha256,
                        document["observed_at_utc"],
                        document["source_updated_at_utc"],
                        document["effective_available_at_utc"],
                        observation.evidence.code_commit,
                        observation.temporal_status.value,
                    ),
                )
    except NHLRepositoryError:
        raise
    except (NHLDatabaseError, sqlite3.Error, ValueError) as error:
        raise NHLRepositoryError(
            "Le lot NHL entre en conflit avec le journal immuable."
        ) from error

    return ArchiveResult(
        run_id=run_id,
        created=True,
        observation_count=len(bundle.observations),
        bundle_sha256=bundle_sha256,
    )


def audit_nhl_storage(
    database_path: Path,
    *,
    allowed_root: Path,
) -> StorageAudit:
    """Relit et recalcule toutes les empreintes sans modifier la base."""

    try:
        verify_nhl_database(database_path, allowed_root=allowed_root)
        with closing(
            connect_nhl_database_read_only(
                database_path,
                allowed_root=allowed_root,
            )
        ) as connection:
            runs = connection.execute(
                "SELECT * FROM nhl_ingestion_runs ORDER BY run_id"
            ).fetchall()
            observation_count = 0
            unknown_count = 0
            providers: set[str] = set()
            bundle_sha256s: list[str] = []
            for run in runs:
                rows = connection.execute(
                    """
                    SELECT * FROM nhl_normalized_observations
                    WHERE run_id = ? ORDER BY observation_id
                    """,
                    (run["run_id"],),
                ).fetchall()
                if not rows:
                    raise NHLRepositoryError("Un lot NHL archivé est vide.")
                documents: list[dict[str, Any]] = []
                for row in rows:
                    document = _stored_observation_document(row)
                    if _sha256_document(document) != row["observation_sha256"]:
                        raise NHLRepositoryError(
                            "L'empreinte d'une observation NHL diverge."
                        )
                    if (
                        row["fixture_id"] != run["fixture_id"]
                        or row["provider"] != run["provider"]
                        or row["request_sha256"] != run["request_sha256"]
                        or row["response_sha256"] != run["response_sha256"]
                        or row["code_commit"] != run["code_commit"]
                    ):
                        raise NHLRepositoryError("La provenance d'un lot NHL diverge.")
                    documents.append(document)
                    unknown_count += row["value_state"] == "UNKNOWN"
                proof = {
                    "capability": run["capability"],
                    "code_commit": run["code_commit"],
                    "effective_available_at_utc": run["effective_available_at_utc"],
                    "fixture_id": run["fixture_id"],
                    "observed_at_utc": run["observed_at_utc"],
                    "provider": run["provider"],
                    "request_sha256": run["request_sha256"],
                    "response_sha256": run["response_sha256"],
                    "source_updated_at_utc": run["source_updated_at_utc"],
                }
                calculated = _sha256_document(
                    {"observations": documents, "proof": proof}
                )
                if calculated != run["bundle_sha256"]:
                    raise NHLRepositoryError("L'empreinte d'un lot NHL diverge.")
                providers.add(run["provider"])
                bundle_sha256s.append(calculated)
                observation_count += len(rows)
            return StorageAudit(
                schema_valid=True,
                run_count=len(runs),
                observation_count=observation_count,
                unknown_count=unknown_count,
                providers=tuple(sorted(providers)),
                bundle_sha256s=tuple(bundle_sha256s),
            )
    except NHLRepositoryError:
        raise
    except (NHLDatabaseError, sqlite3.Error, ValueError) as error:
        raise NHLRepositoryError("L'audit du stockage NHL a échoué.") from error


__all__ = [
    "ArchiveResult",
    "NHLRepositoryError",
    "StorageAudit",
    "archive_offline_fixture",
    "audit_nhl_storage",
]
