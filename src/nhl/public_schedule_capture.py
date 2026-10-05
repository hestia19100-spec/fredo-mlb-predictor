"""Capture manuelle et vérifiable du calendrier public NHL, sans usage modèle."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable

from src.nhl.database import NHL_DATA_ROOT, PROJECT_ROOT
from src.nhl.public_schedule_candidates import (
    MAX_SCHEDULE_BYTES,
    PublicScheduleError,
    ScheduledGame,
    parse_public_schedule,
)

POLICY_PATH = PROJECT_ROOT / "nhl_protocols" / "data" / "nhl20_schedule_capture_protocol_v1.json"
DEFAULT_ROOT = NHL_DATA_ROOT / "public_schedule_captures"
SCHEMA_VERSION = "nhl_public_schedule_capture_v1"
USER_COPY_SCHEMA_VERSION = "nhl_public_schedule_user_copy_v1"
USER_COPY_POLICY_PATH = PROJECT_ROOT / "nhl_protocols" / "data" / "nhl27_user_schedule_copy_v1.json"
BASE_URL = "https://api-web.nhle.com/v1/schedule/"


class PublicScheduleCaptureError(ValueError):
    """Une capture manuelle ne peut pas être acceptée ou vérifiée."""


@dataclass(frozen=True, slots=True)
class ScheduleCaptureReceipt:
    path: Path
    response_sha256: str
    observed_at_utc: datetime
    game_count: int


def _clock_utc(now: Callable[[], datetime]) -> datetime:
    value = now()
    if (not isinstance(value, datetime) or value.tzinfo is None
            or value.utcoffset() != timedelta(0)):
        raise PublicScheduleCaptureError("Horloge UTC requise.")
    return value.astimezone(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _policy_digest(path: Path) -> str:
    try:
        raw = path.read_bytes()
        policy = json.loads(raw)
    except (OSError, ValueError) as error:
        raise PublicScheduleCaptureError("Protocole de capture illisible.") from error
    required = {
        "protocol_id": "lpf_edge_nhl_schedule_capture_v1",
        "protocol_schema_version": 1,
        "authorization_basis": "USER_ATTESTED_ORAL_PERSONAL_AUTOMATED_COLLECTION_AND_MODEL_TRAINING",
        "authorization_independently_verified": False,
        "status": "MANUAL_SCHEDULE_CAPTURE_ONLY",
        "provider_id": "nhl_public_web_api",
        "endpoint_template": BASE_URL + "{date}",
        "manual_schedule_capture_allowed": True,
        "automatic_daily_collection_allowed": False,
        "historical_as_of_availability_proven": False,
        "training_permitted_from_this_capture": False,
        "prediction_publication_permitted": False,
        "source_registry_v1_provider_enabled": False,
    }
    if not isinstance(policy, dict) or any(policy.get(key) != value for key, value in required.items()):
        raise PublicScheduleCaptureError("Protocole de capture non autorisé.")
    return hashlib.sha256(raw).hexdigest()


def _game_record(game: ScheduledGame) -> dict[str, object]:
    return {
        "game_id": game.game_id,
        "season": game.season,
        "start_utc": _stamp(game.start_utc),
        "away_abbr": game.away_abbr,
        "home_abbr": game.home_abbr,
    }


def capture_public_schedule(
    target_date: date,
    *,
    explicit_manual_run: bool = False,
    root: Path = DEFAULT_ROOT,
    policy_path: Path = POLICY_PATH,
    transport: Callable | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> ScheduleCaptureReceipt:
    """Un seul GET HTTPS, sans redirection ni relance; archive brute privée.

    Le résultat n'autorise ni entraînement ni publication. L'appelant doit
    explicitement choisir une capture manuelle.
    """
    if explicit_manual_run is not True:
        raise PublicScheduleCaptureError("Capture manuelle explicite requise.")
    if transport is None:
        raise PublicScheduleCaptureError("Transport explicite requis.")
    if not isinstance(target_date, date) or isinstance(target_date, datetime):
        raise PublicScheduleCaptureError("Date cible invalide.")
    policy_sha = _policy_digest(Path(policy_path))
    url = BASE_URL + target_date.isoformat()
    started = _clock_utc(now)
    try:
        with transport(url, timeout=15) as response:
            if response.geturl() != url or response.status != 200:
                raise PublicScheduleCaptureError("Réponse redirigée ou non réussie.")
            if response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise PublicScheduleCaptureError("Type de réponse inattendu.")
            raw = response.read(MAX_SCHEDULE_BYTES + 1)
    except PublicScheduleCaptureError:
        raise
    except (OSError, ValueError) as error:
        raise PublicScheduleCaptureError("Collecte NHL échouée.") from error
    observed = _clock_utc(now)
    if observed < started:
        raise PublicScheduleCaptureError("Horloge incohérente.")
    try:
        games = parse_public_schedule(raw, target_date, observed)
    except PublicScheduleError as error:
        raise PublicScheduleCaptureError("Calendrier NHL rejeté.") from error
    digest = hashlib.sha256(raw).hexdigest()
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "CAPTURE_ONLY_NOT_MODEL_ELIGIBLE",
        "provider_id": "nhl_public_web_api",
        "source_url": url,
        "target_date": target_date.isoformat(),
        "request_started_at_utc": _stamp(started),
        "observed_at_utc": _stamp(observed),
        "response_sha256": digest,
        "policy_sha256": policy_sha,
        "future_regular_games": [_game_record(game) for game in games],
        "historical_as_of_availability_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }
    key = observed.strftime("%Y%m%dT%H%M%S%fZ") + "-" + digest[:16]
    slot = Path(root) / target_date.isoformat() / key
    try:
        slot.mkdir(parents=True, exist_ok=False)
        with (slot / "response.json").open("xb") as output:
            output.write(raw)
        with (slot / "receipt.json").open("xb") as output:
            output.write((json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
        with (slot / "COMPLETED").open("xb"):
            pass
    except OSError as error:
        raise PublicScheduleCaptureError("Archivage incomplet; créneau non utilisable.") from error
    return ScheduleCaptureReceipt(slot, digest, observed, len(games))


def ingest_user_supplied_schedule(
    target_date: date,
    source_path: Path,
    *,
    explicit_manual_run: bool = False,
    root: Path = DEFAULT_ROOT,
    policy_path: Path = USER_COPY_POLICY_PATH,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> ScheduleCaptureReceipt:
    """Archive a user's browser copy without claiming a direct NHL API capture.

    The intake time is the only proven availability time. The supplied content
    cannot establish its own origin, response headers, or historical timestamp.
    """
    if explicit_manual_run is not True:
        raise PublicScheduleCaptureError("Import manuel explicite requis.")
    if not isinstance(target_date, date) or isinstance(target_date, datetime):
        raise PublicScheduleCaptureError("Date cible invalide.")
    try:
        policy_raw = Path(policy_path).read_bytes()
        policy = json.loads(policy_raw)
    except (OSError, ValueError) as error:
        raise PublicScheduleCaptureError("Protocole de copie illisible.") from error
    required = {
        "schema_version": USER_COPY_SCHEMA_VERSION,
        "acquisition_mode": "user_supplied_browser_copy",
        "manual_intake_allowed": True,
        "source_origin_independently_verified": False,
        "automatic_collection_allowed": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }
    if not isinstance(policy, dict) or any(policy.get(key) != value for key, value in required.items()):
        raise PublicScheduleCaptureError("Protocole de copie non autorisé.")
    started = _clock_utc(now)
    try:
        with Path(source_path).open("rb") as source:
            raw = source.read(MAX_SCHEDULE_BYTES + 1)
    except OSError as error:
        raise PublicScheduleCaptureError("Copie de calendrier illisible.") from error
    observed = _clock_utc(now)
    if observed < started:
        raise PublicScheduleCaptureError("Horloge incohérente.")
    try:
        games = parse_public_schedule(raw, target_date, observed)
    except PublicScheduleError as error:
        raise PublicScheduleCaptureError("Copie de calendrier NHL rejetée.") from error
    digest = hashlib.sha256(raw).hexdigest()
    receipt = {
        "schema_version": USER_COPY_SCHEMA_VERSION,
        "status": "CAPTURE_ONLY_NOT_MODEL_ELIGIBLE",
        "provider_id": "user_supplied_browser_copy",
        "claimed_source_url": BASE_URL + target_date.isoformat(),
        "acquisition_mode": "user_supplied_browser_copy",
        "source_origin_independently_verified": False,
        "target_date": target_date.isoformat(),
        "intake_started_at_utc": _stamp(started),
        "observed_at_utc": _stamp(observed),
        "response_sha256": digest,
        "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
        "future_regular_games": [_game_record(game) for game in games],
        "historical_as_of_availability_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }
    key = observed.strftime("%Y%m%dT%H%M%S%fZ") + "-" + digest[:16]
    slot = Path(root) / target_date.isoformat() / key
    try:
        slot.mkdir(parents=True, exist_ok=False)
        with (slot / "response.json").open("xb") as output:
            output.write(raw)
        with (slot / "receipt.json").open("xb") as output:
            output.write((json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
        with (slot / "COMPLETED").open("xb"):
            pass
    except OSError as error:
        raise PublicScheduleCaptureError("Archivage incomplet; créneau non utilisable.") from error
    return ScheduleCaptureReceipt(slot, digest, observed, len(games))

def verify_public_schedule_capture(slot: Path) -> dict[str, object]:
    """Relit le brut et le reçu; ne consulte jamais le réseau."""
    slot = Path(slot)
    try:
        if not (slot / "COMPLETED").is_file():
            raise PublicScheduleCaptureError("Capture non terminée.")
        raw = (slot / "response.json").read_bytes()
        receipt_bytes = (slot / "receipt.json").read_bytes()
        if len(receipt_bytes) > 128 * 1024:
            raise PublicScheduleCaptureError("Reçu trop volumineux.")
        receipt = json.loads(receipt_bytes)
        target = date.fromisoformat(receipt["target_date"])
        observed = datetime.fromisoformat(receipt["observed_at_utc"].replace("Z", "+00:00"))
        schema = receipt["schema_version"]
        if schema == SCHEMA_VERSION:
            started = datetime.fromisoformat(receipt["request_started_at_utc"].replace("Z", "+00:00"))
            provenance_valid = (receipt["provider_id"] == "nhl_public_web_api"
                                and receipt["source_url"] == BASE_URL + target.isoformat())
        elif schema == USER_COPY_SCHEMA_VERSION:
            started = datetime.fromisoformat(receipt["intake_started_at_utc"].replace("Z", "+00:00"))
            provenance_valid = (
                receipt["provider_id"] == "user_supplied_browser_copy"
                and receipt["claimed_source_url"] == BASE_URL + target.isoformat()
                and receipt["acquisition_mode"] == "user_supplied_browser_copy"
                and receipt["source_origin_independently_verified"] is False
            )
        else:
            raise PublicScheduleCaptureError("Schéma de capture inconnu.")
        if (not provenance_valid
                or receipt["status"] != "CAPTURE_ONLY_NOT_MODEL_ELIGIBLE"
                or receipt["response_sha256"] != hashlib.sha256(raw).hexdigest()
                or started.utcoffset() != timedelta(0)
                or observed.utcoffset() != timedelta(0)
                or observed < started
                or receipt["historical_as_of_availability_proven"] is not False
                or receipt["training_permitted"] is not False
                or receipt["prediction_publication_permitted"] is not False):
            raise PublicScheduleCaptureError("Reçu de capture incohérent.")
        games = parse_public_schedule(raw, target, observed)
        if receipt["future_regular_games"] != [_game_record(game) for game in games]:
            raise PublicScheduleCaptureError("Projection des matchs altérée.")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, PublicScheduleError) as error:
        raise PublicScheduleCaptureError("Archive NHL invalide.") from error
    return receipt
