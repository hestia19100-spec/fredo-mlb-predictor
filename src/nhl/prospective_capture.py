"""Capture locale NHL horodatée, sans réseau ni activation des pronostics."""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from src.nhl.database import NHL_DATA_ROOT, PROJECT_ROOT

SCHEMA_VERSION = "nhl_pregame_local_capture_v1"
DEFAULT_ROOT = NHL_DATA_ROOT / "pregame_captures"
REGISTRY_PATH = PROJECT_ROOT / "nhl_protocols" / "data" / "nhl_source_registry_v1.json"
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class NHLPregameCaptureError(ValueError):
    """La capture ou sa preuve est invalide."""


@dataclass(frozen=True, slots=True)
class CaptureReceipt:
    path: Path
    observed_at_utc: datetime
    response_sha256: str
    status: str


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise NHLPregameCaptureError(f"{field} doit être en UTC.")
    return value.astimezone(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _source_url(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("https://") or len(value) > 2000:
        raise NHLPregameCaptureError("URL source invalide.")
    address = value[len("https://"):]
    host, separator, path = address.partition("/")
    valid = (
        separator == "/" and "." in host and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*[A-Za-z0-9]", host)
        and path and not any(c.isspace() or c in "?#@" for c in path)
    )
    if not valid:
        raise NHLPregameCaptureError("URL HTTPS sans identifiant, port, requête ni fragment requise.")
    return value


def _registry_evidence(provider_id: str, registry_path: Path) -> tuple[str, bool]:
    try:
        raw = registry_path.read_bytes()
        registry = json.loads(raw)
        matches = [item for item in registry["providers"] if item["provider_id"] == provider_id]
        if len(matches) != 1 or type(matches[0]["enabled"]) is not bool:
            raise ValueError("fournisseur absent ou ambigu")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise NHLPregameCaptureError("Registre NHL non vérifiable.") from error
    return hashlib.sha256(raw).hexdigest(), matches[0]["enabled"]


def capture_local_response(
    *, raw_response: bytes, provider_id: str, declared_source_url: str,
    target_game_id: int, scheduled_start_utc: datetime, information_cutoff_utc: datetime,
    root: Path = DEFAULT_ROOT, registry_path: Path = REGISTRY_PATH,
) -> CaptureReceipt:
    """Fige une copie fournie localement. L'URL déclarée n'authentifie pas la source."""
    if not isinstance(raw_response, bytes) or not 0 < len(raw_response) <= MAX_RESPONSE_BYTES:
        raise NHLPregameCaptureError("Réponse vide ou trop volumineuse.")
    try:
        parsed = json.loads(raw_response)
    except (ValueError, UnicodeDecodeError) as error:
        raise NHLPregameCaptureError("La réponse doit être un JSON valide.") from error
    if not isinstance(parsed, (dict, list)):
        raise NHLPregameCaptureError("La réponse doit être un objet ou une liste JSON.")
    if type(target_game_id) is not int or target_game_id <= 0:
        raise NHLPregameCaptureError("Identifiant de match invalide.")
    start = _utc(scheduled_start_utc, "scheduled_start_utc")
    cutoff = _utc(information_cutoff_utc, "information_cutoff_utc")
    if cutoff >= start:
        raise NHLPregameCaptureError("La limite d'information doit précéder le match.")
    url = _source_url(declared_source_url)
    registry_sha, provider_enabled = _registry_evidence(provider_id, registry_path)
    observed = _utc(_now_utc(), "observed_at_utc")
    digest = hashlib.sha256(raw_response).hexdigest()
    before_cutoff = observed < cutoff
    status = "PRE_CUTOFF_LOCAL_COPY_UNVERIFIED" if before_cutoff else "LATE_LOCAL_COPY"
    document = {
        "schema_version": SCHEMA_VERSION,
        "capture_method": "LOCAL_FILE_IMPORT_NO_NETWORK",
        "provider_id": provider_id,
        "provider_enabled_at_capture": provider_enabled,
        "source_registry_sha256": registry_sha,
        "declared_source_url": url,
        "source_origin_authenticated": False,
        "target_game_id_asserted": target_game_id,
        "target_game_in_response_verified": False,
        "scheduled_start_utc_asserted": _stamp(start),
        "information_cutoff_utc_asserted": _stamp(cutoff),
        "observed_at_utc": _stamp(observed),
        "captured_before_cutoff": before_cutoff,
        "status": status,
        "response_sha256": digest,
        "response_base64": base64.b64encode(raw_response).decode("ascii"),
        "historical_pregame_availability_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }
    directory = Path(root) / str(target_game_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (observed.strftime("%Y%m%dT%H%M%S%fZ") + "_" + digest[:16] + ".json")
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise NHLPregameCaptureError("Capture déjà présente : aucun écrasement autorisé.") from error
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return CaptureReceipt(path, observed, digest, status)


def verify_local_capture(path: Path) -> dict[str, object]:
    """Recalcule l'empreinte et vérifie que les interdictions restent fermées."""
    try:
        document = json.loads(Path(path).read_bytes())
        raw = base64.b64decode(document["response_base64"], validate=True)
        observed = _utc(datetime.fromisoformat(document["observed_at_utc"].replace("Z", "+00:00")), "observed_at_utc")
        cutoff = _utc(datetime.fromisoformat(document["information_cutoff_utc_asserted"].replace("Z", "+00:00")), "information_cutoff_utc")
        start = _utc(datetime.fromisoformat(document["scheduled_start_utc_asserted"].replace("Z", "+00:00")), "scheduled_start_utc")
        before = observed < cutoff
        expected = "PRE_CUTOFF_LOCAL_COPY_UNVERIFIED" if before else "LATE_LOCAL_COPY"
        if (document["schema_version"] != SCHEMA_VERSION or document["capture_method"] != "LOCAL_FILE_IMPORT_NO_NETWORK"
            or hashlib.sha256(raw).hexdigest() != document["response_sha256"] or cutoff >= start
            or document["captured_before_cutoff"] is not before or document["status"] != expected):
            raise ValueError("capture incohérente")
        for gate in ("source_origin_authenticated", "target_game_in_response_verified",
                     "historical_pregame_availability_proven", "training_permitted", "prediction_publication_permitted"):
            if document[gate] is not False:
                raise ValueError("garde-fou ouvert")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise NHLPregameCaptureError("Capture NHL corrompue ou statut non vérifiable.") from error
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description="Archiver une réponse NHL locale sans appel réseau.")
    parser.add_argument("response_file", type=Path)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--game-id", required=True, type=int)
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--cutoff-utc", required=True)
    args = parser.parse_args()
    try:
        receipt = capture_local_response(
            raw_response=args.response_file.read_bytes(), provider_id=args.provider,
            declared_source_url=args.source_url, target_game_id=args.game_id,
            scheduled_start_utc=datetime.fromisoformat(args.start_utc.replace("Z", "+00:00")),
            information_cutoff_utc=datetime.fromisoformat(args.cutoff_utc.replace("Z", "+00:00")),
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"{receipt.status}: {receipt.path} SHA-256={receipt.response_sha256}")


if __name__ == "__main__":
    main()
