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


def _validate_sha256(expected_sha256: str) -> str:
    """Contrôle une empreinte attendue avant comparaison."""
    normalized_hash = expected_sha256.strip().lower()

    if len(normalized_hash) != 64:
        raise RawArchiveError(
            "Une empreinte SHA-256 doit contenir 64 caractères."
        )

    try:
        int(normalized_hash, 16)
    except ValueError as error:
        raise RawArchiveError(
            "L’empreinte SHA-256 attendue doit être hexadécimale."
        ) from error

    return normalized_hash


def _resolve_archive_path(
    *,
    relative_path: str,
    data_directory: Path,
) -> tuple[str, Path]:
    """Résout un chemin relatif sans autoriser de sortie de data/raw."""
    normalized_relative_path = relative_path.strip()

    if not normalized_relative_path:
        raise RawArchiveError(
            "Le chemin relatif de l’archive est obligatoire."
        )

    if "\\" in normalized_relative_path:
        raise RawArchiveError(
            "Le chemin d’archive doit utiliser des barres obliques."
        )

    relative_path_object = Path(normalized_relative_path)

    if (
        relative_path_object.is_absolute()
        or ".." in relative_path_object.parts
    ):
        raise RawArchiveError(
            "Le chemin d’archive doit rester relatif à data/raw."
        )

    raw_directory = (data_directory / "raw").resolve()
    absolute_path = (
        data_directory.parent / relative_path_object
    ).resolve()

    if not absolute_path.is_relative_to(raw_directory):
        raise RawArchiveError(
            "Le chemin d’archive sort du dossier data/raw."
        )

    return relative_path_object.as_posix(), absolute_path


def load_raw_archive(archive_path: Path) -> bytes:
    """Décompresse et relit une archive existante."""
    try:
        with gzip.open(archive_path, "rb") as archive_file:
            return archive_file.read()
    except (OSError, EOFError) as error:
        raise RawArchiveError(
            f"Archive illisible : {archive_path}"
        ) from error


def verify_raw_archive(
    *,
    relative_path: str,
    expected_sha256: str,
    data_directory: Path = DATA_DIR,
) -> RawArchive:
    """Vérifie l’existence, la lisibilité et le SHA-256 d’une archive."""
    normalized_hash = _validate_sha256(expected_sha256)
    normalized_relative_path, absolute_path = (
        _resolve_archive_path(
            relative_path=relative_path,
            data_directory=data_directory,
        )
    )

    if not absolute_path.is_file():
        raise RawArchiveError(
            f"Archive absente : {normalized_relative_path}"
        )

    raw_content = load_raw_archive(absolute_path)
    actual_sha256 = sha256(raw_content).hexdigest()

    if actual_sha256 != normalized_hash:
        raise RawArchiveError(
            "L’empreinte de l’archive ne correspond pas au journal : "
            f"{normalized_relative_path}"
        )

    return RawArchive(
        absolute_path=absolute_path,
        relative_path=normalized_relative_path,
        sha256=actual_sha256,
        uncompressed_size_bytes=len(raw_content),
        compressed_size_bytes=absolute_path.stat().st_size,
    )


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
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
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