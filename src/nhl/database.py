"""Base SQLite NHL isolée, créée uniquement sur demande explicite.

Ce module ne se connecte jamais à la base MLB et ne crée aucun fichier lors de
son import. NHL-04 utilise exclusivement des chemins placés sous une racine
nommée ``nhl``. Les trois tables du journal sont append-only.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NHL_DATA_ROOT = PROJECT_ROOT / "data" / "nhl"
NHL_DATABASE_PATH = NHL_DATA_ROOT / "fredo_nhl.db"
MLB_DATABASE_PATH = PROJECT_ROOT / "data" / "fredo_mlb.db"
STORAGE_PROTOCOL_PATH = (
    PROJECT_ROOT / "nhl_protocols" / "data" / "nhl_storage_protocol_v1.json"
)
SCHEMA_VERSION = 1
MIGRATION_RELEASED_AT_UTC = "2026-09-19T00:00:00Z"


class NHLDatabaseError(RuntimeError):
    """Signale une base NHL non conforme ou un chemin dangereux."""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    statements: tuple[str, ...]

    @property
    def sha256(self) -> str:
        payload = "\n-- statement --\n".join(self.statements).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


MIGRATION_1 = Migration(
    version=1,
    statements=(
        """
        CREATE TABLE nhl_schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at_utc TEXT NOT NULL,
            migration_sha256 TEXT NOT NULL
                CHECK(length(migration_sha256) = 64)
        ) STRICT
        """.strip(),
        """
        CREATE TABLE nhl_ingestion_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id TEXT NOT NULL UNIQUE,
            provider TEXT NOT NULL,
            capability TEXT NOT NULL,
            request_sha256 TEXT NOT NULL CHECK(length(request_sha256) = 64),
            response_sha256 TEXT NOT NULL CHECK(length(response_sha256) = 64),
            bundle_sha256 TEXT NOT NULL UNIQUE CHECK(length(bundle_sha256) = 64),
            observed_at_utc TEXT NOT NULL,
            source_updated_at_utc TEXT,
            effective_available_at_utc TEXT NOT NULL,
            code_commit TEXT NOT NULL CHECK(length(code_commit) = 40),
            archived_at_utc TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status = 'COMPLETED')
        ) STRICT
        """.strip(),
        """
        CREATE TABLE nhl_normalized_observations (
            observation_id TEXT PRIMARY KEY,
            observation_sha256 TEXT NOT NULL UNIQUE
                CHECK(length(observation_sha256) = 64),
            run_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            target_game_id INTEGER NOT NULL CHECK(target_game_id > 0),
            source_game_id INTEGER CHECK(source_game_id > 0),
            entity_id TEXT NOT NULL,
            value_state TEXT NOT NULL CHECK(value_state IN ('KNOWN', 'UNKNOWN')),
            canonical_value_json TEXT,
            provider TEXT NOT NULL,
            fixture_id TEXT NOT NULL,
            request_sha256 TEXT NOT NULL CHECK(length(request_sha256) = 64),
            response_sha256 TEXT NOT NULL CHECK(length(response_sha256) = 64),
            observed_at_utc TEXT NOT NULL,
            source_updated_at_utc TEXT,
            effective_available_at_utc TEXT NOT NULL,
            code_commit TEXT NOT NULL CHECK(length(code_commit) = 40),
            temporal_status TEXT NOT NULL CHECK(temporal_status = 'UNASSESSED'),
            FOREIGN KEY(run_id) REFERENCES nhl_ingestion_runs(run_id),
            CHECK(source_game_id IS NULL OR source_game_id <> target_game_id),
            CHECK(
                (value_state = 'UNKNOWN' AND canonical_value_json IS NULL)
                OR
                (value_state = 'KNOWN' AND canonical_value_json IS NOT NULL)
            )
        ) STRICT
        """.strip(),
        """
        CREATE TRIGGER nhl_schema_migrations_no_update
        BEFORE UPDATE ON nhl_schema_migrations
        BEGIN
            SELECT RAISE(ABORT, 'nhl_schema_migrations is append-only');
        END
        """.strip(),
        """
        CREATE TRIGGER nhl_schema_migrations_no_delete
        BEFORE DELETE ON nhl_schema_migrations
        BEGIN
            SELECT RAISE(ABORT, 'nhl_schema_migrations is append-only');
        END
        """.strip(),
        """
        CREATE TRIGGER nhl_ingestion_runs_no_update
        BEFORE UPDATE ON nhl_ingestion_runs
        BEGIN
            SELECT RAISE(ABORT, 'nhl_ingestion_runs is append-only');
        END
        """.strip(),
        """
        CREATE TRIGGER nhl_ingestion_runs_no_delete
        BEFORE DELETE ON nhl_ingestion_runs
        BEGIN
            SELECT RAISE(ABORT, 'nhl_ingestion_runs is append-only');
        END
        """.strip(),
        """
        CREATE TRIGGER nhl_normalized_observations_no_update
        BEFORE UPDATE ON nhl_normalized_observations
        BEGIN
            SELECT RAISE(ABORT, 'nhl_normalized_observations is append-only');
        END
        """.strip(),
        """
        CREATE TRIGGER nhl_normalized_observations_no_delete
        BEFORE DELETE ON nhl_normalized_observations
        BEGIN
            SELECT RAISE(ABORT, 'nhl_normalized_observations is append-only');
        END
        """.strip(),
    ),
)

MIGRATIONS: tuple[Migration, ...] = (MIGRATION_1,)
EXPECTED_SCHEMA_OBJECTS = frozenset(
    {
        "nhl_schema_migrations",
        "nhl_ingestion_runs",
        "nhl_normalized_observations",
        "nhl_schema_migrations_no_update",
        "nhl_schema_migrations_no_delete",
        "nhl_ingestion_runs_no_update",
        "nhl_ingestion_runs_no_delete",
        "nhl_normalized_observations_no_update",
        "nhl_normalized_observations_no_delete",
    }
)


def validate_nhl_database_path(
    database_path: Path,
    *,
    allowed_root: Path,
) -> Path:
    """Garantit qu'une base reste sous une racine dédiée nommée ``nhl``."""

    if not isinstance(database_path, Path) or not isinstance(allowed_root, Path):
        raise NHLDatabaseError("Les chemins doivent être des objets Path.")
    resolved_path = database_path.resolve(strict=False)
    resolved_root = allowed_root.resolve(strict=False)
    if resolved_root.name.lower() != "nhl":
        raise NHLDatabaseError("La racine autorisée doit être dédiée à NHL.")
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise NHLDatabaseError("La base doit rester sous la racine NHL.") from error
    if not relative.parts or len(relative.parts) != 1:
        raise NHLDatabaseError("La base doit être un fichier direct de la racine NHL.")
    if resolved_path == MLB_DATABASE_PATH.resolve(strict=False):
        raise NHLDatabaseError("La base MLB ne peut jamais servir à NHL.")
    if resolved_path.suffix.lower() != ".db" or "mlb" in resolved_path.name.lower():
        raise NHLDatabaseError("Le nom de la base NHL est invalide.")
    return resolved_path


def connect_nhl_database(
    database_path: Path,
    *,
    allowed_root: Path,
) -> sqlite3.Connection:
    """Ouvre explicitement une base validée et active les clés étrangères."""

    validated = validate_nhl_database_path(database_path, allowed_root=allowed_root)
    validated.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(validated)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def connect_nhl_database_read_only(
    database_path: Path,
    *,
    allowed_root: Path,
) -> sqlite3.Connection:
    """Relit une base existante sans pouvoir la créer ni la modifier."""

    validated = validate_nhl_database_path(database_path, allowed_root=allowed_root)
    if not validated.is_file() or validated.is_symlink():
        raise NHLDatabaseError("La base NHL à auditer n'est pas un fichier régulier.")
    connection = sqlite3.connect(validated.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA query_only = ON")
    return connection


def _verify_schema(connection: sqlite3.Connection) -> None:
    objects = {
        row[0]
        for row in connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE (type = 'table' OR type = 'trigger') AND name LIKE 'nhl_%'
            """
        )
    }
    if objects != EXPECTED_SCHEMA_OBJECTS:
        raise NHLDatabaseError("Le schéma NHL contient des objets inattendus.")
    rows = connection.execute(
        """
        SELECT version, migration_sha256
        FROM nhl_schema_migrations ORDER BY version
        """
    ).fetchall()
    expected = [(migration.version, migration.sha256) for migration in MIGRATIONS]
    actual = [(row["version"], row["migration_sha256"]) for row in rows]
    if actual != expected:
        raise NHLDatabaseError("L'historique des migrations NHL diverge.")
    if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        raise NHLDatabaseError("La version SQLite NHL diverge.")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise NHLDatabaseError("La base NHL contient une clé étrangère invalide.")


def initialize_nhl_database(
    database_path: Path,
    *,
    allowed_root: Path,
) -> None:
    """Applique les migrations une seule fois, dans une transaction atomique."""

    try:
        with closing(
            connect_nhl_database(database_path, allowed_root=allowed_root)
        ) as connection, connection:
            existing = {
                row[0]
                for row in connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE (type = 'table' OR type = 'trigger')
                      AND name LIKE 'nhl_%'
                    """
                )
            }
            if not existing:
                for statement in MIGRATION_1.statements:
                    connection.execute(statement)
                connection.execute(
                    """
                    INSERT INTO nhl_schema_migrations(
                        version, applied_at_utc, migration_sha256
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        MIGRATION_1.version,
                        MIGRATION_RELEASED_AT_UTC,
                        MIGRATION_1.sha256,
                    ),
                )
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            _verify_schema(connection)
    except NHLDatabaseError:
        raise
    except sqlite3.Error as error:
        raise NHLDatabaseError("La migration de la base NHL a échoué.") from error


def verify_nhl_database(
    database_path: Path,
    *,
    allowed_root: Path,
) -> None:
    """Vérifie le schéma existant au moyen d'une connexion en lecture seule."""

    try:
        with closing(
            connect_nhl_database_read_only(
                database_path,
                allowed_root=allowed_root,
            )
        ) as connection:
            _verify_schema(connection)
    except NHLDatabaseError:
        raise
    except sqlite3.Error as error:
        raise NHLDatabaseError("La base NHL ne peut pas être auditée.") from error


def validate_storage_protocol(document: Mapping[str, Any]) -> None:
    """Vérifie que le protocole NHL-04 n'active aucune collecte réelle."""

    if not isinstance(document, Mapping):
        raise NHLDatabaseError("Le protocole NHL-04 doit être un objet JSON.")
    expected_safety = {
        "interface_enabled": False,
        "mlb_database_access_allowed": False,
        "model_enabled": False,
        "odds_enabled": False,
        "provider_calls_allowed": False,
        "synthetic_offline_fixtures_only": True,
    }
    if document.get("protocol_id") != "lpf_edge_nhl_storage_protocol_v1":
        raise NHLDatabaseError("L'identifiant du protocole NHL-04 est invalide.")
    if document.get("protocol_schema_version") != 1:
        raise NHLDatabaseError("La version du protocole NHL-04 est invalide.")
    if document.get("status") != "OFFLINE_SYNTHETIC_STORAGE_ONLY":
        raise NHLDatabaseError("Le statut du protocole NHL-04 est invalide.")
    if document.get("safety") != expected_safety:
        raise NHLDatabaseError("Les limites de sécurité NHL-04 divergent.")
    storage = document.get("storage")
    if not isinstance(storage, Mapping):
        raise NHLDatabaseError("La section storage NHL-04 est absente.")
    if storage.get("schema_version") != SCHEMA_VERSION:
        raise NHLDatabaseError("La version de stockage NHL-04 diverge.")
    if storage.get("default_database") != "data/nhl/fredo_nhl.db":
        raise NHLDatabaseError("Le chemin de stockage NHL-04 diverge.")
    if storage.get("forbidden_database") != "data/fredo_mlb.db":
        raise NHLDatabaseError("L'interdiction de la base MLB est absente.")


def load_and_validate_storage_protocol(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NHLDatabaseError("Le protocole NHL-04 n'est pas lisible.") from error
    validate_storage_protocol(document)
    return document


__all__ = [
    "EXPECTED_SCHEMA_OBJECTS",
    "MIGRATION_1",
    "MIGRATIONS",
    "MLB_DATABASE_PATH",
    "NHL_DATA_ROOT",
    "NHL_DATABASE_PATH",
    "NHLDatabaseError",
    "SCHEMA_VERSION",
    "STORAGE_PROTOCOL_PATH",
    "connect_nhl_database",
    "connect_nhl_database_read_only",
    "initialize_nhl_database",
    "load_and_validate_storage_protocol",
    "validate_nhl_database_path",
    "validate_storage_protocol",
    "verify_nhl_database",
]
