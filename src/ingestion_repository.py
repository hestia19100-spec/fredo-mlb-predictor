"""Gestion du journal des collectes de données MLB."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
import json
from pathlib import Path

from src.database import (
    DATABASE_PATH,
    get_connection,
    initialize_database,
)


class IngestionRunError(RuntimeError):
    """Signale une opération invalide sur un journal de collecte."""


def _utc_now() -> str:
    """Retourne un horodatage UTC explicite."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_game_types(game_types: Iterable[str]) -> str:
    """Normalise les types de matchs pour un stockage stable."""
    normalized_types = sorted(
        {
            str(game_type).strip().upper()
            for game_type in game_types
            if str(game_type).strip()
        }
    )

    if not normalized_types:
        raise ValueError("Au moins un type de match est obligatoire.")

    return ",".join(normalized_types)


def _serialize_parameters(
    request_parameters: Mapping[str, object],
) -> str:
    """Sérialise les paramètres dans un ordre reproductible."""
    return json.dumps(
        dict(request_parameters),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_period(
    start_date: date,
    end_date: date,
) -> None:
    """Refuse une période inversée."""
    if end_date < start_date:
        raise ValueError(
            "La date de fin ne peut pas précéder la date de début."
        )


def _validate_counts(
    records_received: int,
    records_saved: int,
) -> None:
    """Contrôle les compteurs d’une collecte terminée."""
    if records_received < 0 or records_saved < 0:
        raise ValueError("Les compteurs ne peuvent pas être négatifs.")

    if records_saved > records_received:
        raise ValueError(
            "Le nombre enregistré ne peut pas dépasser "
            "le nombre reçu."
        )


def _validate_sha256(response_sha256: str) -> str:
    """Contrôle une empreinte SHA-256."""
    normalized_hash = response_sha256.strip().lower()

    if len(normalized_hash) != 64:
        raise ValueError(
            "Une empreinte SHA-256 doit contenir 64 caractères."
        )

    try:
        int(normalized_hash, 16)
    except ValueError as error:
        raise ValueError(
            "L’empreinte SHA-256 doit être hexadécimale."
        ) from error

    return normalized_hash


def start_ingestion_run(
    *,
    source: str,
    start_date: date,
    end_date: date,
    game_types: Iterable[str],
    request_parameters: Mapping[str, object],
    code_version: str | None = None,
    database_path: Path = DATABASE_PATH,
) -> int:
    """Crée un journal avec le statut started."""
    _validate_period(start_date, end_date)

    normalized_source = source.strip()
    if not normalized_source:
        raise ValueError("La source de données est obligatoire.")

    normalized_game_types = _normalize_game_types(game_types)
    parameters_json = _serialize_parameters(request_parameters)
    normalized_code_version = (
        code_version.strip()
        if code_version is not None and code_version.strip()
        else None
    )

    initialize_database(database_path)

    with get_connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO ingestion_runs (
                source,
                requested_start_date,
                requested_end_date,
                game_types,
                request_parameters_json,
                started_at_utc,
                status,
                code_version
            )
            VALUES (?, ?, ?, ?, ?, ?, 'started', ?)
            """,
            (
                normalized_source,
                start_date.isoformat(),
                end_date.isoformat(),
                normalized_game_types,
                parameters_json,
                _utc_now(),
                normalized_code_version,
            ),
        )

        run_id = cursor.lastrowid

    if run_id is None:
        raise IngestionRunError(
            "SQLite n’a pas retourné l’identifiant de la collecte."
        )

    return int(run_id)


def mark_ingestion_success(
    *,
    run_id: int,
    records_received: int,
    records_saved: int,
    raw_response_path: str,
    response_sha256: str,
    database_path: Path = DATABASE_PATH,
) -> None:
    """Clôture une collecte réussie."""
    _validate_counts(records_received, records_saved)

    normalized_path = raw_response_path.strip()
    if not normalized_path:
        raise ValueError(
            "Le chemin de la réponse brute est obligatoire."
        )

    normalized_hash = _validate_sha256(response_sha256)

    initialize_database(database_path)

    with get_connection(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE ingestion_runs
            SET
                completed_at_utc = ?,
                status = 'success',
                records_received = ?,
                records_saved = ?,
                raw_response_path = ?,
                response_sha256 = ?,
                error_message = NULL
            WHERE run_id = ?
              AND status = 'started'
            """,
            (
                _utc_now(),
                records_received,
                records_saved,
                normalized_path,
                normalized_hash,
                run_id,
            ),
        )

        if cursor.rowcount != 1:
            raise IngestionRunError(
                "La collecte est absente ou déjà terminée."
            )


def mark_ingestion_error(
    *,
    run_id: int,
    error_message: str,
    database_path: Path = DATABASE_PATH,
) -> None:
    """Clôture une collecte en erreur."""
    normalized_error = error_message.strip()
    if not normalized_error:
        raise ValueError("Le message d’erreur est obligatoire.")

    initialize_database(database_path)

    with get_connection(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE ingestion_runs
            SET
                completed_at_utc = ?,
                status = 'error',
                error_message = ?
            WHERE run_id = ?
              AND status = 'started'
            """,
            (
                _utc_now(),
                normalized_error,
                run_id,
            ),
        )

        if cursor.rowcount != 1:
            raise IngestionRunError(
                "La collecte est absente ou déjà terminée."
            )


def get_ingestion_run(
    run_id: int,
    database_path: Path = DATABASE_PATH,
) -> dict[str, object]:
    """Relit un journal précis."""
    initialize_database(database_path)

    with get_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM ingestion_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

    if row is None:
        raise IngestionRunError(
            f"Collecte introuvable : {run_id}."
        )

    return dict(row)