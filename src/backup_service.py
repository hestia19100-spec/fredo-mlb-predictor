"""Crée et vérifie les sauvegardes locales des données MLB."""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import stat
import subprocess
import tarfile
import tempfile
from typing import Any, BinaryIO
import uuid

from src.database import DATABASE_PATH


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"
BACKUP_DIRECTORY = PROJECT_DIRECTORY / "backups"

BACKUP_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
BUFFER_SIZE = 1024 * 1024
MAX_MANIFEST_SIZE = 8 * 1024 * 1024


class BackupError(RuntimeError):
    """Erreur compréhensible liée à une sauvegarde."""


@dataclass(frozen=True, slots=True)
class BackupResult:
    """Résumé d'une sauvegarde créée."""

    absolute_path: Path
    relative_path: str
    archive_sha256: str
    archive_size_bytes: int
    file_count: int
    raw_archive_count: int
    code_version: str


@dataclass(frozen=True, slots=True)
class BackupVerification:
    """Résumé d'une sauvegarde vérifiée."""

    absolute_path: Path
    archive_sha256: str
    archive_size_bytes: int
    file_count: int
    raw_archive_count: int
    code_version: str
    created_at_utc: str


@dataclass(frozen=True, slots=True)
class _SourceFile:
    """Fichier raw contrôlé avant sa copie."""

    path: Path
    archive_path: str
    signature: tuple[int, int, int, int, int]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise BackupError(
            "La date de sauvegarde doit inclure un fuseau horaire."
        )

    return value.astimezone(timezone.utc).isoformat(
        timespec="seconds"
    )


def _file_signature(
    file_stat: os.stat_result,
) -> tuple[int, int, int, int, int]:
    return (
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
        int(file_stat.st_ctime_ns),
    )


def _require_regular_file(path: Path) -> os.stat_result:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as error:
        raise BackupError(
            f"Fichier absent : {path}"
        ) from error

    if stat.S_ISLNK(file_stat.st_mode):
        raise BackupError(
            f"Lien symbolique interdit : {path}"
        )

    if not stat.S_ISREG(file_stat.st_mode):
        raise BackupError(
            f"Fichier non régulier interdit : {path}"
        )

    return file_stat


def _validate_archive_path(path: str) -> None:
    if not path or "\\" in path or "\x00" in path:
        raise BackupError(
            f"Chemin interne interdit : {path!r}"
        )

    pure_path = PurePosixPath(path)

    if (
        pure_path.is_absolute()
        or pure_path.as_posix() != path
        or any(
            part in {"", ".", ".."}
            for part in pure_path.parts
        )
    ):
        raise BackupError(
            f"Chemin interne interdit : {path!r}"
        )


def _hash_stream(
    source: BinaryIO,
    destination: BinaryIO | None = None,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0

    while True:
        block = source.read(BUFFER_SIZE)

        if not block:
            break

        digest.update(block)
        size += len(block)

        if destination is not None:
            destination.write(block)

    return digest.hexdigest(), size


def _hash_file(path: Path) -> tuple[str, int]:
    _require_regular_file(path)

    with path.open("rb") as source:
        return _hash_stream(source)


def _table_exists(
    connection: sqlite3.Connection,
    name: str,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (name,),
    ).fetchone()

    return row is not None


def _assert_no_active_ingestion(
    database_path: Path,
) -> None:
    _require_regular_file(database_path)

    try:
        with closing(
            sqlite3.connect(
                database_path,
                timeout=30,
            )
        ) as connection:
            if not _table_exists(
                connection,
                "ingestion_runs",
            ):
                return

            active_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM ingestion_runs
                    WHERE status = 'started'
                    """
                ).fetchone()[0]
            )

    except sqlite3.Error as error:
        raise BackupError(
            f"Lecture du journal impossible : {error}"
        ) from error

    if active_count:
        raise BackupError(
            "Une collecte MLB est encore en cours. "
            "Attends sa fin avant de créer la sauvegarde."
        )


def _read_migrations(
    connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    if not _table_exists(
        connection,
        "schema_migrations",
    ):
        return []

    cursor = connection.execute(
        "SELECT * FROM schema_migrations"
    )

    columns = [
        description[0]
        for description in cursor.description or ()
    ]

    migrations = [
        {
            column: row[index]
            for index, column in enumerate(columns)
        }
        for row in cursor.fetchall()
    ]

    return sorted(
        migrations,
        key=lambda item: json.dumps(
            item,
            sort_keys=True,
            default=str,
        ),
    )


def _inspect_database(
    database_path: Path,
) -> dict[str, Any]:
    try:
        with closing(
            sqlite3.connect(
                database_path,
                timeout=30,
            )
        ) as connection:
            integrity = [
                str(row[0])
                for row in connection.execute(
                    "PRAGMA integrity_check"
                ).fetchall()
            ]

            if integrity != ["ok"]:
                raise BackupError(
                    "Le contrôle d'intégrité SQLite "
                    "a échoué : "
                    + "; ".join(integrity)
                )

            foreign_key_errors = connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()

            if foreign_key_errors:
                raise BackupError(
                    "La base SQLite contient "
                    f"{len(foreign_key_errors)} "
                    "violation(s) de clé étrangère."
                )

            return {
                "integrity_check": "ok",
                "foreign_key_violations": 0,
                "user_version": int(
                    connection.execute(
                        "PRAGMA user_version"
                    ).fetchone()[0]
                ),
                "schema_migrations": _read_migrations(
                    connection
                ),
            }

    except BackupError:
        raise

    except sqlite3.Error as error:
        raise BackupError(
            f"Base SQLite invalide : {error}"
        ) from error


def _snapshot_database(
    source_path: Path,
    destination_path: Path,
) -> dict[str, Any]:
    _require_regular_file(source_path)
    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        source_uri = (
            source_path.resolve().as_uri()
            + "?mode=ro"
        )

        with closing(
            sqlite3.connect(
                source_uri,
                uri=True,
                timeout=30,
            )
        ) as source_connection:
            source_connection.execute(
                "PRAGMA busy_timeout = 30000"
            )

            with closing(
                sqlite3.connect(destination_path)
            ) as destination_connection:
                source_connection.backup(
                    destination_connection,
                    pages=1024,
                    sleep=0.05,
                )

                destination_connection.commit()

    except sqlite3.Error as error:
        raise BackupError(
            "Création du snapshot SQLite impossible : "
            f"{error}"
        ) from error

    return _inspect_database(destination_path)


def _scan_raw_files(
    raw_directory: Path,
) -> list[_SourceFile]:
    if not raw_directory.exists():
        return []

    root_stat = raw_directory.lstat()

    if (
        stat.S_ISLNK(root_stat.st_mode)
        or not stat.S_ISDIR(root_stat.st_mode)
    ):
        raise BackupError(
            "Le chemin raw n'est pas un dossier sûr : "
            f"{raw_directory}"
        )

    sources: list[_SourceFile] = []

    def visit(
        directory: Path,
        relative_parts: tuple[str, ...],
    ) -> None:
        with os.scandir(directory) as entries:
            ordered_entries = sorted(
                entries,
                key=lambda entry: entry.name,
            )

        for entry in ordered_entries:
            entry_path = Path(entry.path)
            entry_stat = entry_path.lstat()

            if entry.is_symlink():
                raise BackupError(
                    "Lien symbolique raw interdit : "
                    f"{entry_path}"
                )

            parts = (
                *relative_parts,
                entry.name,
            )

            if stat.S_ISDIR(entry_stat.st_mode):
                visit(
                    entry_path,
                    parts,
                )
                continue

            if not stat.S_ISREG(entry_stat.st_mode):
                raise BackupError(
                    "Fichier raw spécial interdit : "
                    f"{entry_path}"
                )

            archive_path = PurePosixPath(
                "data",
                "raw",
                *parts,
            ).as_posix()

            _validate_archive_path(archive_path)

            sources.append(
                _SourceFile(
                    path=entry_path,
                    archive_path=archive_path,
                    signature=_file_signature(
                        entry_stat
                    ),
                )
            )

    visit(
        raw_directory,
        (),
    )

    return sorted(
        sources,
        key=lambda source: source.archive_path,
    )


def _validate_raw_gzip(path: Path) -> None:
    if not path.name.endswith(".json.gz"):
        raise BackupError(
            f"Format raw inattendu : {path}"
        )

    try:
        with gzip.open(path, "rb") as compressed:
            json.load(compressed)

    except (
        OSError,
        EOFError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise BackupError(
            f"Archive raw gzip invalide : {path}"
        ) from error


def _copy_stable_raw(
    source: _SourceFile,
    destination: Path,
) -> dict[str, Any]:
    current_signature = _file_signature(
        _require_regular_file(source.path)
    )

    if current_signature != source.signature:
        raise BackupError(
            "Fichier raw modifié avant sa copie : "
            f"{source.path}"
        )

    _validate_raw_gzip(source.path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        source.path.open("rb") as input_file,
        destination.open("xb") as output_file,
    ):
        digest, size = _hash_stream(
            input_file,
            output_file,
        )

        output_file.flush()
        os.fsync(output_file.fileno())

    final_signature = _file_signature(
        _require_regular_file(source.path)
    )

    if final_signature != source.signature:
        raise BackupError(
            "Fichier raw modifié pendant sa copie : "
            f"{source.path}"
        )

    second_digest, second_size = _hash_file(
        source.path
    )

    if (
        second_digest != digest
        or second_size != size
    ):
        raise BackupError(
            "Fichier raw modifié après sa copie : "
            f"{source.path}"
        )

    return {
        "path": source.archive_path,
        "kind": "raw",
        "size_bytes": size,
        "sha256": digest,
    }


def _current_code_version(
    project_directory: Path,
) -> str:
    try:
        completed = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=project_directory,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )

    except (
        OSError,
        subprocess.SubprocessError,
    ):
        return "unknown"

    return completed.stdout.strip() or "unknown"


def _manifest_bytes(
    manifest: dict[str, Any],
) -> bytes:
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _add_tar_member(
    archive: tarfile.TarFile,
    source_path: Path,
    archive_path: str,
    modification_time: int,
) -> None:
    _validate_archive_path(archive_path)
    _require_regular_file(source_path)

    member = archive.gettarinfo(
        str(source_path),
        arcname=archive_path,
    )

    if not member.isfile():
        raise BackupError(
            "Membre non régulier interdit : "
            f"{archive_path}"
        )

    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mode = 0o600
    member.mtime = modification_time
    member.pax_headers = {}

    with source_path.open("rb") as source:
        archive.addfile(
            member,
            source,
        )


def _write_tar_archive(
    archive_path: Path,
    staged_files: dict[str, Path],
    modification_time: int,
) -> None:
    try:
        with tarfile.open(
            archive_path,
            "x:gz",
            format=tarfile.PAX_FORMAT,
        ) as archive:
            for internal_path in sorted(
                staged_files
            ):
                _add_tar_member(
                    archive,
                    staged_files[internal_path],
                    internal_path,
                    modification_time,
                )

    except (
        OSError,
        tarfile.TarError,
    ) as error:
        raise BackupError(
            "Création de l'archive impossible : "
            f"{error}"
        ) from error


def _load_manifest(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> dict[str, Any]:
    if member.size > MAX_MANIFEST_SIZE:
        raise BackupError(
            "Le manifeste est anormalement volumineux."
        )

    stream = archive.extractfile(member)

    if stream is None:
        raise BackupError(
            "Le manifeste est illisible."
        )

    try:
        with stream:
            content = stream.read(
                MAX_MANIFEST_SIZE + 1
            )

        if len(content) > MAX_MANIFEST_SIZE:
            raise BackupError(
                "Le manifeste est anormalement "
                "volumineux."
            )

        manifest = json.loads(
            content.decode("utf-8")
        )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise BackupError(
            f"Manifeste JSON invalide : {error}"
        ) from error

    if not isinstance(manifest, dict):
        raise BackupError(
            "Le manifeste doit être un objet JSON."
        )

    return manifest


def _validate_manifest(
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    if (
        manifest.get("format_version")
        != BACKUP_FORMAT_VERSION
    ):
        raise BackupError(
            "Version de sauvegarde non prise "
            "en charge."
        )

    files = manifest.get("files")

    if not isinstance(files, list) or not files:
        raise BackupError(
            "Liste de fichiers absente du manifeste."
        )

    normalized: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for item in files:
        if not isinstance(item, dict):
            raise BackupError(
                "Entrée de fichier invalide "
                "dans le manifeste."
            )

        path = item.get("path")
        kind = item.get("kind")
        size = item.get("size_bytes")
        digest = item.get("sha256")

        if not isinstance(path, str):
            raise BackupError(
                "Chemin de fichier absent "
                "du manifeste."
            )

        _validate_archive_path(path)

        if path in seen_paths:
            raise BackupError(
                "Chemin dupliqué dans le manifeste : "
                f"{path}"
            )

        seen_paths.add(path)

        if kind not in {
            "database",
            "raw",
        }:
            raise BackupError(
                f"Type de fichier invalide : {path}"
            )

        if (
            kind == "database"
            and not path.startswith("data/")
        ):
            raise BackupError(
                f"Chemin SQLite invalide : {path}"
            )

        if (
            kind == "raw"
            and not path.startswith("data/raw/")
        ):
            raise BackupError(
                f"Chemin raw invalide : {path}"
            )

        if type(size) is not int or size < 0:
            raise BackupError(
                f"Taille invalide : {path}"
            )

        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character
                not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise BackupError(
                f"SHA-256 invalide : {path}"
            )

        normalized.append(item)

    database_entries = [
        item
        for item in normalized
        if item["kind"] == "database"
    ]

    if len(database_entries) != 1:
        raise BackupError(
            "La sauvegarde doit contenir "
            "un seul snapshot SQLite."
        )

    return normalized


def verify_backup_bundle(
    archive_path: Path,
) -> BackupVerification:
    """
    Vérifie tous les membres et le snapshot,
    sans extraction non contrôlée.
    """
    archive_path = (
        Path(archive_path)
        .expanduser()
        .absolute()
    )

    _require_regular_file(archive_path)

    archive_digest, archive_size = _hash_file(
        archive_path
    )

    try:
        with tempfile.TemporaryDirectory(
            prefix="fredo-backup-verify-"
        ) as temporary_name:
            temporary_directory = Path(
                temporary_name
            )

            with tarfile.open(
                archive_path,
                "r:gz",
            ) as archive:
                members = archive.getmembers()
                names: list[str] = []

                for member in members:
                    _validate_archive_path(
                        member.name
                    )

                    if not member.isfile():
                        raise BackupError(
                            "Membre non régulier interdit : "
                            f"{member.name}"
                        )

                    names.append(member.name)

                if len(names) != len(set(names)):
                    raise BackupError(
                        "L'archive contient des "
                        "membres dupliqués."
                    )

                members_by_name = {
                    member.name: member
                    for member in members
                }

                if MANIFEST_NAME not in members_by_name:
                    raise BackupError(
                        "Le manifeste est absent "
                        "de la sauvegarde."
                    )

                manifest = _load_manifest(
                    archive,
                    members_by_name[MANIFEST_NAME],
                )

                files = _validate_manifest(
                    manifest
                )

                expected_names = {
                    MANIFEST_NAME
                } | {
                    str(item["path"])
                    for item in files
                }

                if set(names) != expected_names:
                    raise BackupError(
                        "La liste des membres ne "
                        "correspond pas au manifeste."
                    )

                snapshot_path = (
                    temporary_directory
                    / "snapshot.db"
                )

                for item in files:
                    internal_path = str(
                        item["path"]
                    )

                    member_stream = archive.extractfile(
                        members_by_name[
                            internal_path
                        ]
                    )

                    if member_stream is None:
                        raise BackupError(
                            "Membre illisible : "
                            f"{internal_path}"
                        )

                    destination: BinaryIO | None = None

                    if item["kind"] == "database":
                        destination = (
                            snapshot_path.open("xb")
                        )

                    try:
                        with member_stream:
                            digest, size = _hash_stream(
                                member_stream,
                                destination,
                            )

                    finally:
                        if destination is not None:
                            destination.close()

                    if size != item["size_bytes"]:
                        raise BackupError(
                            "Taille incorrecte : "
                            f"{internal_path}"
                        )

                    if digest != item["sha256"]:
                        raise BackupError(
                            "SHA-256 incorrect : "
                            f"{internal_path}"
                        )

            database_metadata = _inspect_database(
                snapshot_path
            )

            expected_database = manifest.get(
                "database"
            )

            if not isinstance(
                expected_database,
                dict,
            ):
                raise BackupError(
                    "Métadonnées SQLite absentes "
                    "du manifeste."
                )

            if (
                database_metadata
                != expected_database
            ):
                raise BackupError(
                    "Métadonnées SQLite incohérentes."
                )

    except BackupError:
        raise

    except (
        OSError,
        EOFError,
        tarfile.TarError,
    ) as error:
        raise BackupError(
            "Vérification de l'archive "
            f"impossible : {error}"
        ) from error

    raw_count = sum(
        item["kind"] == "raw"
        for item in files
    )

    code_version = manifest.get(
        "code_version"
    )

    created_at_utc = manifest.get(
        "created_at_utc"
    )

    if (
        not isinstance(code_version, str)
        or not isinstance(created_at_utc, str)
    ):
        raise BackupError(
            "Métadonnées générales invalides."
        )

    return BackupVerification(
        absolute_path=archive_path,
        archive_sha256=archive_digest,
        archive_size_bytes=archive_size,
        file_count=len(files),
        raw_archive_count=raw_count,
        code_version=code_version,
        created_at_utc=created_at_utc,
    )


def _publish_without_overwrite(
    partial_path: Path,
    final_path: Path,
) -> None:
    try:
        os.link(
            partial_path,
            final_path,
        )

    except FileExistsError as error:
        raise BackupError(
            "La sauvegarde existe déjà : "
            f"{final_path}"
        ) from error

    except OSError as error:
        raise BackupError(
            "Publication de la sauvegarde "
            f"impossible : {error}"
        ) from error

    partial_path.unlink()


def create_backup_bundle(
    *,
    database_path: Path = DATABASE_PATH,
    data_directory: Path = DATA_DIRECTORY,
    backup_directory: Path = BACKUP_DIRECTORY,
    code_version: str | None = None,
    created_at_utc: datetime | None = None,
) -> BackupResult:
    """
    Crée, vérifie puis publie
    une sauvegarde complète.
    """
    database_path = Path(
        database_path
    ).absolute()

    data_directory = Path(
        data_directory
    ).absolute()

    backup_directory = Path(
        backup_directory
    ).absolute()

    try:
        backup_directory.relative_to(
            data_directory
        )
    except ValueError:
        pass
    else:
        raise BackupError(
            "Le dossier backups doit rester "
            "en dehors de data."
        )

    try:
        data_directory.relative_to(
            backup_directory
        )
    except ValueError:
        pass
    else:
        raise BackupError(
            "Le dossier data doit rester "
            "en dehors de backups."
        )

    _assert_no_active_ingestion(
        database_path
    )

    raw_directory = (
        data_directory
        / "raw"
    )

    raw_sources = _scan_raw_files(
        raw_directory
    )

    backup_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    directory_stat = (
        backup_directory.lstat()
    )

    if (
        stat.S_ISLNK(directory_stat.st_mode)
        or not stat.S_ISDIR(
            directory_stat.st_mode
        )
    ):
        raise BackupError(
            "Le chemin backups n'est pas "
            "un dossier sûr."
        )

    created_at = (
        created_at_utc
        or _utc_now()
    )

    created_text = _utc_text(
        created_at
    )

    effective_code_version = (
        code_version
        or _current_code_version(
            data_directory.parent
        )
    )

    timestamp = (
        created_at
        .astimezone(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )

    final_path = backup_directory / (
        f"fredo-mlb-backup-"
        f"{timestamp}.tar.gz"
    )

    partial_path = backup_directory / (
        f".{final_path.name}."
        f"{uuid.uuid4().hex}.partial"
    )

    if final_path.exists():
        raise BackupError(
            "La sauvegarde existe déjà : "
            f"{final_path}"
        )

    try:
        with tempfile.TemporaryDirectory(
            prefix=".fredo-backup-stage-",
            dir=backup_directory,
        ) as stage_name:
            stage = Path(stage_name)

            staged_files: dict[
                str,
                Path,
            ] = {}

            manifest_files: list[
                dict[str, Any]
            ] = []

            database_internal_path = (
                PurePosixPath(
                    "data",
                    database_path.name,
                ).as_posix()
            )

            staged_database = (
                stage
                / database_internal_path
            )

            database_metadata = (
                _snapshot_database(
                    database_path,
                    staged_database,
                )
            )

            (
                database_digest,
                database_size,
            ) = _hash_file(
                staged_database
            )

            staged_files[
                database_internal_path
            ] = staged_database

            manifest_files.append(
                {
                    "path": (
                        database_internal_path
                    ),
                    "kind": "database",
                    "size_bytes": (
                        database_size
                    ),
                    "sha256": (
                        database_digest
                    ),
                }
            )

            _assert_no_active_ingestion(
                database_path
            )

            for source in raw_sources:
                staged_path = (
                    stage
                    / PurePosixPath(
                        source.archive_path
                    )
                )

                metadata = _copy_stable_raw(
                    source,
                    staged_path,
                )

                staged_files[
                    source.archive_path
                ] = staged_path

                manifest_files.append(
                    metadata
                )

            final_raw_sources = (
                _scan_raw_files(
                    raw_directory
                )
            )

            if (
                final_raw_sources
                != raw_sources
            ):
                raise BackupError(
                    "Le contenu de data/raw "
                    "a changé pendant "
                    "la sauvegarde."
                )

            _assert_no_active_ingestion(
                database_path
            )

            manifest = {
                "format_version": (
                    BACKUP_FORMAT_VERSION
                ),
                "created_at_utc": (
                    created_text
                ),
                "code_version": (
                    effective_code_version
                ),
                "database": (
                    database_metadata
                ),
                "files": sorted(
                    manifest_files,
                    key=lambda item: str(
                        item["path"]
                    ),
                ),
            }

            staged_manifest = (
                stage
                / MANIFEST_NAME
            )

            staged_manifest.write_bytes(
                _manifest_bytes(
                    manifest
                )
            )

            staged_files[
                MANIFEST_NAME
            ] = staged_manifest

            _write_tar_archive(
                partial_path,
                staged_files,
                int(
                    created_at.timestamp()
                ),
            )

            verification = (
                verify_backup_bundle(
                    partial_path
                )
            )

            _publish_without_overwrite(
                partial_path,
                final_path,
            )

    finally:
        try:
            partial_path.unlink()
        except FileNotFoundError:
            pass

    (
        archive_digest,
        archive_size,
    ) = _hash_file(
        final_path
    )

    try:
        relative_path = (
            final_path
            .relative_to(
                data_directory.parent
            )
            .as_posix()
        )
    except ValueError:
        relative_path = (
            final_path.as_posix()
        )

    return BackupResult(
        absolute_path=final_path,
        relative_path=relative_path,
        archive_sha256=archive_digest,
        archive_size_bytes=archive_size,
        file_count=verification.file_count,
        raw_archive_count=(
            verification.raw_archive_count
        ),
        code_version=(
            verification.code_version
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crée ou vérifie une sauvegarde "
            "locale des données MLB."
        )
    )

    parser.add_argument(
        "--verify",
        type=Path,
        metavar="ARCHIVE",
        help=(
            "Vérifie une sauvegarde existante "
            "sans la restaurer."
        ),
    )

    arguments = parser.parse_args()

    try:
        if arguments.verify is not None:
            result = verify_backup_bundle(
                arguments.verify
            )

            print(
                "Sauvegarde valide : "
                f"{result.absolute_path}"
            )
            print(
                "Fichiers de données : "
                f"{result.file_count}"
            )
            print(
                "Archives raw : "
                f"{result.raw_archive_count}"
            )
            print(
                "Version du code : "
                f"{result.code_version}"
            )
            print(
                "SHA-256 : "
                f"{result.archive_sha256}"
            )
            return

        result = create_backup_bundle()

        print(
            "Sauvegarde créée et vérifiée : "
            f"{result.relative_path}"
        )
        print(
            "Fichiers de données : "
            f"{result.file_count}"
        )
        print(
            "Archives raw : "
            f"{result.raw_archive_count}"
        )
        print(
            "Taille : "
            f"{result.archive_size_bytes} octets"
        )
        print(
            "Version du code : "
            f"{result.code_version}"
        )
        print(
            "SHA-256 : "
            f"{result.archive_sha256}"
        )

    except BackupError as error:
        raise SystemExit(
            f"Échec de la sauvegarde : {error}"
        ) from error


if __name__ == "__main__":
    main()