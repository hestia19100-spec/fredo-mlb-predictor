"""Archive locale des candidats MLB 2027, sans publication ni certification.

Un créneau est à usage unique. Le marqueur COMPLETED n'est créé qu'après
écriture de tous les octets; une interruption laisse un créneau incomplet
qui ne peut être réutilisé ou présenté comme une preuve de publication.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

from src.mlb_2027_observed_source import (
    ObservedSourceBundle,
    verify_2027_observed_source,
)
from src.mlb_2027_reference_predictor import (
    ReferenceCalculation,
    calculate_reference_probabilities,
)


class MLB2027ArchiveError(ValueError):
    """Le candidat ne peut pas être archivé ou vérifié sans ambiguïté."""


@dataclass(frozen=True, slots=True)
class CandidateArchive:
    path: Path
    manifest_sha256: str


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode_date(item: object) -> str:
    if hasattr(item, "isoformat"):
        return item.isoformat()
    raise TypeError(f"Type JSON non pris en charge : {type(item)!r}")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
            default=_encode_date,
        ).encode("utf-8") + b"\n"
    )


def _utc(value: datetime) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise MLB2027ArchiveError("L'horodatage doit être UTC.")
    return value.isoformat().replace("+00:00", "Z")


def archive_reference_candidate(
    root: Path,
    source: ObservedSourceBundle,
    calculation: ReferenceCalculation,
    *,
    issued_at_utc: datetime,
) -> CandidateArchive:
    """Scelle les octets sources et le calcul en local, une seule fois.

    Cette fonction ne valide aucune heure de publication distante, ne pousse
    aucun fichier et n'autorise aucun pronostic public.
    """
    if type(source) is not ObservedSourceBundle or type(calculation) is not ReferenceCalculation:
        raise MLB2027ArchiveError("Source ou calcul absent.")
    if source != verify_2027_observed_source(source.target_date, source.observations):
        raise MLB2027ArchiveError("Source MLB altérée.")
    expected = calculate_reference_probabilities(
        source.target_date, source.upcoming_games, source.prior_final_games,
        source_receipt_sha256=source.receipt_sha256,
    )
    if calculation != expected:
        raise MLB2027ArchiveError("Calcul non conforme au modèle figé.")
    issued = _utc(issued_at_utc)
    if (
        source.latest_received_at_utc > issued_at_utc
        or issued_at_utc > calculation.readiness.lead_window.latest_completion_utc
    ):
        raise MLB2027ArchiveError("Candidat émis après la limite T-60 ou avant la source.")
    if not isinstance(root, Path) or root.is_symlink():
        raise MLB2027ArchiveError("Racine d'archive invalide.")
    day_dir = root / source.target_date.isoformat()
    slot = day_dir / source.receipt_sha256
    if slot.exists() or slot.is_symlink():
        raise MLB2027ArchiveError("Créneau déjà utilisé : aucun écrasement autorisé.")

    files: dict[str, bytes] = {"source-receipt.json": source.receipt_bytes}
    for index, observation in enumerate(source.observations, start=1):
        files[f"mlb-response-{index:03d}.json"] = observation.raw_content
    files["candidate.json"] = _json_bytes({
        "schema_version": 1,
        "status": "LOCAL_ONLY_UNPUBLISHED",
        "target_date": source.target_date,
        "issued_at_utc": issued,
        "source_receipt_sha256": source.receipt_sha256,
        "lead_window": asdict(calculation.readiness.lead_window),
        "readiness": [asdict(game) for game in calculation.readiness.games],
        "probabilities": [asdict(row) for row in calculation.probabilities],
        "abstained_count": calculation.abstained_count,
    })
    manifest = _json_bytes({
        "schema_version": 1,
        "status": "LOCAL_ONLY_UNPUBLISHED",
        "target_date": source.target_date,
        "source_receipt_sha256": source.receipt_sha256,
        "files": {name: _sha(data) for name, data in sorted(files.items())},
    })
    day_dir.mkdir(parents=True, exist_ok=True)
    try:
        slot.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise MLB2027ArchiveError("Créneau déjà utilisé.") from error
    for name, data in files.items():
        with (slot / name).open("xb") as output:
            output.write(data)
    with (slot / "manifest.json").open("xb") as output:
        output.write(manifest)
    with (slot / "COMPLETED").open("xb") as output:
        output.write(_sha(manifest).encode("ascii") + b"\n")
    return CandidateArchive(path=slot, manifest_sha256=_sha(manifest))


def verify_candidate_archive(
    path: Path, *, expected_manifest_sha256: str,
) -> dict[str, object]:
    """Contrôle l'intégrité locale; n'atteste jamais une visibilité GitHub."""
    if not isinstance(path, Path) or path.is_symlink() or not path.is_dir():
        raise MLB2027ArchiveError("Créneau local absent ou non fiable.")
    if type(expected_manifest_sha256) is not str or len(expected_manifest_sha256) != 64:
        raise MLB2027ArchiveError("Empreinte du manifeste invalide.")
    if any(item.is_symlink() for item in path.iterdir()):
        raise MLB2027ArchiveError("Lien symbolique interdit dans l'archive.")
    try:
        manifest_bytes = (path / "manifest.json").read_bytes()
        marker = (path / "COMPLETED").read_bytes()
        if _sha(manifest_bytes) != expected_manifest_sha256:
            raise MLB2027ArchiveError("Manifeste modifié.")
        if marker != (expected_manifest_sha256 + "\n").encode("ascii"):
            raise MLB2027ArchiveError("Créneau incomplet.")
        manifest = json.loads(manifest_bytes)
        if (
            type(manifest) is not dict
            or manifest.get("schema_version") != 1
            or manifest.get("status") != "LOCAL_ONLY_UNPUBLISHED"
            or manifest.get("source_receipt_sha256") != path.name
            or manifest.get("target_date") != path.parent.name
            or type(manifest.get("files")) is not dict
        ):
            raise MLB2027ArchiveError("Manifeste invalide.")
        files = manifest["files"]
        if set(item.name for item in path.iterdir()) != set(files) | {"manifest.json", "COMPLETED"}:
            raise MLB2027ArchiveError("Fichiers manquants ou inattendus.")
        for name, digest in files.items():
            if (
                type(name) is not str or "/" in name or "\\" in name
                or name in {"", ".", "..", "manifest.json", "COMPLETED"}
                or type(digest) is not str or len(digest) != 64
                or _sha((path / name).read_bytes()) != digest
            ):
                raise MLB2027ArchiveError("Octets du candidat modifiés.")
        if files.get("source-receipt.json") != manifest["source_receipt_sha256"]:
            raise MLB2027ArchiveError("Reçu source incohérent.")
        candidate = json.loads((path / "candidate.json").read_bytes())
        if (
            candidate.get("status") != "LOCAL_ONLY_UNPUBLISHED"
            or candidate.get("source_receipt_sha256") != manifest["source_receipt_sha256"]
            or candidate.get("target_date") != manifest["target_date"]
        ):
            raise MLB2027ArchiveError("Candidat incohérent.")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        if isinstance(error, MLB2027ArchiveError):
            raise
        raise MLB2027ArchiveError("Archive illisible ou incomplète.") from error
    return manifest
