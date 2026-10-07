"""Append-only NHL pregame evidence, never a prediction or a model output."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Callable

from .database import PROJECT_ROOT
from .prospective_real_readiness import audit_real_pregame_readiness

SCHEMA = "nhl31_pregame_checkpoint_v1"
ROOT = PROJECT_ROOT / "nhl_prospective_checkpoints"
POLICY = PROJECT_ROOT / "nhl_protocols/data/nhl31_pregame_checkpoint_v1.json"
GAME_FIELDS = ("game_id", "away_abbr", "home_abbr", "scheduled_start_utc",
               "information_cutoff_utc", "status", "feature_sha256",
               "current_season_import_id", "history_capture_id",
               "history_response_sha256", "form_effective_available_at_utc",
               "away_current_season_games", "home_current_season_games")


class NHLCheckpointError(ValueError):
    """The pregame checkpoint is not safely verifiable."""


def _bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _time(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise NHLCheckpointError("Heure UTC non canonique.")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise NHLCheckpointError("Heure UTC invalide.") from error


def _stamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset().total_seconds() != 0:
        raise NHLCheckpointError("Horloge UTC requise.")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _policy_hash() -> str:
    try:
        raw = POLICY.read_bytes()
        policy = json.loads(raw)
    except (OSError, ValueError) as error:
        raise NHLCheckpointError("Protocole NHL-31 illisible.") from error
    required = {"schema_version": SCHEMA, "manual_only": True,
                "direct_schedule_required": True, "pre_cutoff_required": True, "lead_minutes": 60,
                "historical_backtest_asof_proven": False,
                "training_permitted": False, "prediction_publication_permitted": False,
                "odds_ingestion": False, "mlb_database_mutation": False,
                "nhl_historical_database_mutation": False}
    if not isinstance(policy, dict) or any(policy.get(k) != v for k, v in required.items()):
        raise NHLCheckpointError("Protocole NHL-31 incompatible.")
    return sha256(raw).hexdigest()


def _validate(proof: dict[str, object]) -> None:
    sealed = _time(proof["sealed_at_utc"])
    if (proof.get("schema_version") != SCHEMA or proof.get("status") != "DESCRIPTIVE_ONLY"
            or proof.get("schedule_acquisition_mode") != "direct_https"
            or proof.get("lead_minutes") != 60
            or proof.get("historical_backtest_asof_proven") is not False
            or proof.get("training_permitted") is not False
            or proof.get("prediction_publication_permitted") is not False
            or proof.get("odds_ingestion") is not False
            or not _digest(proof.get("schedule_response_sha256"))
            or proof.get("policy_sha256") != _policy_hash()
            or _time(proof["schedule_observed_at_utc"]) > sealed):
        raise NHLCheckpointError("Preuve prospective invalide.")
    games = proof.get("games")
    if not isinstance(games, list) or not games or len({g["game_id"] for g in games}) != len(games):
        raise NHLCheckpointError("Matchs prospectifs absents ou dupliques.")
    for game in games:
        if (not isinstance(game, dict) or set(game) != set(GAME_FIELDS)
                or game["status"] != "CURRENT_SEASON_FORM_DESCRIPTIVE_ONLY"
                or not _digest(game["feature_sha256"])
                or not _digest(game["history_response_sha256"])
                or not game["current_season_import_id"] or not game["history_capture_id"]
                or _time(game["information_cutoff_utc"]) <= sealed
                or _time(game["form_effective_available_at_utc"]) > _time(game["information_cutoff_utc"])
                or _time(game["scheduled_start_utc"]) - _time(game["information_cutoff_utc"]) != timedelta(minutes=60)):
            raise NHLCheckpointError("Match non admissible avant sa limite.")


def build_checkpoint(report: dict[str, object], sealed_at: datetime) -> dict[str, object]:
    """Reduce verified readiness to source hashes and descriptive form hashes."""
    if (report.get("schema_version") != "nhl26_prospective_readiness_v1"
            or report.get("schedule_acquisition_mode") != "direct_https"
            or report.get("lead_minutes") != 60
            or report.get("training_permitted") is not False
            or report.get("prediction_publication_permitted") is not False
            or report.get("current_season_import_verified") is not True
            or report.get("game_count") != len(report.get("games", ()))):
        raise NHLCheckpointError("Audit prospectif non admissible.")
    games = []
    for row in report["games"]:
        if (row.get("schedule_before_cutoff") is not True
                or row.get("current_season_import_selection") != "LATEST_BEFORE_CUTOFF"
                or row.get("current_season_import_before_cutoff") is not True):
            raise NHLCheckpointError("Source posterieure a la limite.")
        try:
            projected = {key: row[key] for key in GAME_FIELDS}
        except KeyError as error:
            raise NHLCheckpointError("Preuve descriptive incomplete.") from error
        try:
            form_time = datetime.fromisoformat(projected["form_effective_available_at_utc"].replace("Z", "+00:00"))
            projected["form_effective_available_at_utc"] = _stamp(form_time)
        except (AttributeError, TypeError, ValueError) as error:
            raise NHLCheckpointError("Heure de forme invalide.") from error
        games.append(projected)
    proof = {"schema_version": SCHEMA, "status": "DESCRIPTIVE_ONLY",
             "sealed_at_utc": _stamp(sealed_at), "target_date": report["target_date"],
             "schedule_acquisition_mode": report["schedule_acquisition_mode"],
             "schedule_observed_at_utc": report["schedule_observed_at_utc"],
             "schedule_response_sha256": report["schedule_response_sha256"],
             "policy_sha256": _policy_hash(), "lead_minutes": 60, "games": games,
             "historical_backtest_asof_proven": False,
             "training_permitted": False, "prediction_publication_permitted": False,
             "odds_ingestion": False}
    _validate(proof)
    return {**proof, "proof_sha256": sha256(_bytes(proof)).hexdigest()}


def seal_checkpoint(schedule_slot: Path, import_slots: tuple[Path, ...], *,
                    explicit_manual_run: bool = False, root: Path = ROOT,
                    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> Path:
    """Audit first, then seal locally before every match cutoff."""
    if explicit_manual_run is not True:
        raise NHLCheckpointError("Scellement manuel explicite requis.")
    report = audit_real_pregame_readiness(
        schedule_slot, current_season_import_slots=import_slots, lead_minutes=60,
    )
    proof = build_checkpoint(report, now())
    directory = Path(root) / proof["target_date"]
    if Path(root).is_symlink() or directory.is_symlink():
        raise NHLCheckpointError("Repertoire de preuve non autorise.")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (proof["sealed_at_utc"].replace(":", "").replace("-", "")
                        + "-" + proof["proof_sha256"][:16] + ".json")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(_bytes(proof) + b"\n")
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise NHLCheckpointError("Preuve deja presente ou ecriture impossible.") from error
    return path


def verify_checkpoint(path: Path) -> dict[str, object]:
    """Check file integrity; publication of its hash supplies external timing."""
    try:
        raw = Path(path).read_bytes()
        if len(raw) > 128 * 1024:
            raise NHLCheckpointError("Preuve trop volumineuse.")
        document = json.loads(raw)
        digest = document.pop("proof_sha256")
        if digest != sha256(_bytes(document)).hexdigest():
            raise NHLCheckpointError("Empreinte de preuve incorrecte.")
        _validate(document)
        return {**document, "proof_sha256": digest}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise NHLCheckpointError("Preuve NHL-31 non verifiable.") from error
