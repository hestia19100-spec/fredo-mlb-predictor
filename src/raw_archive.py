"""Archivage immuable des réponses brutes reçues depuis MLB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import gzip
from hashlib import sha256
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from src.database import DATA_DIR


SOURCE_PATTERN = re.compile(r"^[a-z0-9_-]+$")


class RawArchiveError(RuntimeError):
    """Signale une archive absente, corrompue ou incohérente."""


@dataclass(frozen=True, slots=True)
class RawArchive:
    """Informations permettant de retrouver une réponse brute."""

    absolute_path: Path
    relative_path: str
    sha256: str
    uncompressed_size_bytes: int
    compressed_size_bytes: int


def _validate_source_name(source_name: str) -> str:
    """Contrôle le nom utilisé dans le chemin de l’archive."""
    normalized_name = source_name.strip().lower()

    if not SOURCE_PATTERN.fullmatch(normalized_name):
        raise ValueError(
            "Le nom de source doit contenir uniquement "
            "des lettres minuscules, chiffres, tirets ou underscores."
        )

    return normalized_name


def _validate_period(
    start_date: date,
    end_date: date,
) -> None:
    """Refuse une période inversée."""
    if end_date < start_date:
        raise ValueError(
            "La date de fin ne peut pas précéder la date de début."
        )


def load_raw_archive(archive_path: Path) -> bytes:
    """Décompresse et relit une archive existante."""
    try:
        with gzip.open(archive_path, "rb") as archive_file:
            return archive_file.read()
    except (OSError, EOFError) as error:
        raise RawArchiveError(
            f"Archive illisible : {archive_path}"
        ) from error


def _verify_existing_archive(
    archive_path: Path,
    raw_content: bytes,
    expected_sha256: str,
) -> None:
    """Empêche le remplacement silencieux d’une archive différente."""
    existing_content = load_raw_archive(archive_path)
    existing_sha256 = sha256(existing_content).hexdigest()

    if (
        existing_sha256 != expected_sha256
        or existing_content != raw_content
    ):
        raise RawArchiveError(
            f"Conflit avec une archive existante : {archive_path}"
        )


def _write_compressed_archive(
    archive_path: Path,
    raw_content: bytes,
    expected_sha256: str,
) -> None:
    """Écrit l’archive via un fichier temporaire."""
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=archive_path.parent,
            prefix=".archive-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=temporary_file,
                mtime=0,
            ) as gzip_file:
                gzip_file.write(raw_content)

        if archive_path.exists():
            _verify_existing_archive(
                archive_path,
                raw_content,
                expected_sha256,
            )
        else:
            temporary_path.replace(archive_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def archive_raw_response(
    *,
    raw_content: bytes,
    source_name: str,
    start_date: date,
    end_date: date,
    data_directory: Path = DATA_DIR,
) -> RawArchive:
    """Archive une réponse brute et retourne ses informations."""
    if not raw_content:
        raise ValueError("La réponse brute ne peut pas être vide.")

    _validate_period(start_date, end_date)
    normalized_source = _validate_source_name(source_name)

    response_sha256 = sha256(raw_content).hexdigest()
    archive_directory = (
        data_directory
        / "raw"
        / normalized_source
        / str(start_date.year)
    )
    archive_directory.mkdir(parents=True, exist_ok=True)

    archive_name = (
        f"{start_date.isoformat()}_to_{end_date.isoformat()}_"
        f"{response_sha256[:16]}.json.gz"
    )
    archive_path = archive_directory / archive_name

    if archive_path.exists():
        _verify_existing_archive(
            archive_path,
            raw_content,
            response_sha256,
        )
    else:
        _write_compressed_archive(
            archive_path,
            raw_content,
            response_sha256,
        )

    relative_path = archive_path.relative_to(
        data_directory.parent
    ).as_posix()

    return RawArchive(
        absolute_path=archive_path,
        relative_path=relative_path,
        sha256=response_sha256,
        uncompressed_size_bytes=len(raw_content),
        compressed_size_bytes=archive_path.stat().st_size,
    )