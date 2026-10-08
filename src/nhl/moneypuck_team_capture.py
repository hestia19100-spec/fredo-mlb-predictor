"""Capture horodatée du CSV d'équipes MoneyPuck, sans éligibilité modèle.

Une capture d'aujourd'hui ne prouve jamais la disponibilité avant un match passé.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any, Callable

from .database import NHL_DATA_ROOT, PROJECT_ROOT
from .moneypuck_five_season import REFERENCE_SEASONS, FiveSeasonHistory, select_five_season_history
from .moneypuck_team_import import (
    ATTRIBUTION, MAX_FILE_BYTES, SOURCE_PAGE, TEAM_GAME_DOWNLOAD,
    import_team_games,
)
from .public_schedule_candidates import ScheduledGame
from .temporal_policy import build_information_cutoff

POLICY_PATH = PROJECT_ROOT / "nhl_protocols/data/nhl23_team_capture_v1.json"
DEFAULT_ROOT = NHL_DATA_ROOT / "moneypuck_team_captures"
SCHEMA = "nhl_moneypuck_team_capture_v1"
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")

class NHLTeamCaptureError(ValueError):
    """Capture ou preuve NHL rejetée."""

@dataclass(frozen=True, slots=True)
class VerifiedTeamCapture:
    path: Path
    observed_at_utc: datetime
    response_sha256: str
    history: FiveSeasonHistory

def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise NHLTeamCaptureError("Horodatage UTC requis.")
    if value.utcoffset().total_seconds() != 0:
        raise NHLTeamCaptureError("Horodatage UTC requis.")
    return value.astimezone(timezone.utc)

def _policy_hash() -> str:
    raw = POLICY_PATH.read_bytes()
    policy = json.loads(raw)
    expected = {"schema_version": SCHEMA, "source_url": TEAM_GAME_DOWNLOAD,
                "historical_asof_availability_proven": False,
                "training_permitted": False, "prediction_publication_permitted": False}
    if not isinstance(policy, dict) or any(policy.get(k) != v for k, v in expected.items()):
        raise NHLTeamCaptureError("Protocole de capture incompatible.")
    return sha256(raw).hexdigest()

def _parse(raw: bytes) -> FiveSeasonHistory:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_FILE_BYTES:
        raise NHLTeamCaptureError("Taille du CSV MoneyPuck invalide.")
    try:
        with TemporaryDirectory(prefix="nhl-team-capture-") as directory:
            path = Path(directory) / "all_teams.csv"
            path.write_bytes(raw)
            return select_five_season_history(import_team_games(path, seasons=REFERENCE_SEASONS))
    except (OSError, ValueError, UnicodeError) as error:
        raise NHLTeamCaptureError("CSV MoneyPuck rejeté.") from error

def _coverage(history: FiveSeasonHistory) -> list[dict[str, int]]:
    return [asdict(item) for item in history.coverage]

def capture_team_history(*, explicit_manual_run: bool = False,
                         transport: Callable[..., Any] | None = None,
                         source_file: Path | None = None,
                         code_commit: str, root: Path = DEFAULT_ROOT,
                         now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                         ) -> VerifiedTeamCapture:
    """GET explicite ou reprise locale, sans accès à la base ni au modèle."""
    if explicit_manual_run is not True or (transport is None) == (source_file is None):
        raise NHLTeamCaptureError("Choisir exactement un transport ou un fichier local.")
    if not isinstance(code_commit, str) or not _COMMIT.fullmatch(code_commit):
        raise NHLTeamCaptureError("Commit Git complet requis.")
    policy_hash = _policy_hash()
    started = _utc(now())
    try:
        if source_file is not None:
            candidate = Path(source_file)
            if candidate.stat().st_size > MAX_FILE_BYTES:
                raise NHLTeamCaptureError("Fichier local trop volumineux.")
            raw = candidate.read_bytes()
        else:
            assert transport is not None
            with transport(TEAM_GAME_DOWNLOAD, timeout=45) as response:
                if response.status != 200 or response.geturl() != TEAM_GAME_DOWNLOAD:
                    raise NHLTeamCaptureError("Réponse redirigée ou non réussie.")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                if content_type not in {"text/csv", "text/plain", "application/octet-stream"}:
                    raise NHLTeamCaptureError("Type de réponse inattendu.")
                raw = response.read(MAX_FILE_BYTES + 1)
    except NHLTeamCaptureError:
        raise
    except (OSError, ValueError) as error:
        raise NHLTeamCaptureError("Collecte MoneyPuck échouée.") from error
    observed = _utc(now())
    if observed < started:
        raise NHLTeamCaptureError("Horloge de capture incohérente.")
    history = replace(_parse(raw), observed_at_utc=observed)
    digest = sha256(raw).hexdigest()
    receipt = {"schema_version": SCHEMA, "status": "CAPTURE_ONLY_NOT_MODEL_ELIGIBLE",
               "source_page": SOURCE_PAGE, "source_url": TEAM_GAME_DOWNLOAD,
               "attribution": ATTRIBUTION, "request_started_at_utc": started.isoformat(),
               "acquisition_mode": "local_file_intake" if source_file is not None else "https_get",
               "network_source_independently_verified": False,
               "observed_at_utc": observed.isoformat(),
               "effective_available_at_utc": observed.isoformat(),
               "response_sha256": digest, "response_bytes": len(raw),
               "policy_sha256": policy_hash, "code_commit": code_commit,
               "five_season_coverage": _coverage(history),
               "regular_coverage_complete": history.regular_coverage_complete,
               "historical_asof_availability_proven": False,
               "training_permitted": False, "prediction_publication_permitted": False}
    key = observed.strftime("%Y%m%dT%H%M%S%fZ") + "-" + digest[:16]
    slot = Path(root) / observed.date().isoformat() / key
    try:
        slot.mkdir(parents=True, exist_ok=False)
        (slot / "response.csv").write_bytes(raw)
        (slot / "receipt.json").write_text(json.dumps(receipt, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
        if sha256((slot / "response.csv").read_bytes()).hexdigest() != digest:
            raise NHLTeamCaptureError("Archive écrite avec une empreinte différente.")
        (slot / "COMPLETED").touch(exist_ok=False)
    except OSError as error:
        raise NHLTeamCaptureError("Archivage incomplet; créneau inutilisable.") from error
    return VerifiedTeamCapture(slot, observed, digest, history)

def verify_team_capture(slot: Path, *, allowed_root: Path = DEFAULT_ROOT) -> VerifiedTeamCapture:
    """Vérifie à nouveau les octets, le protocole et la couverture des cinq saisons."""
    path = Path(slot).resolve()
    if not path.is_relative_to(Path(allowed_root).resolve()) or not (path / "COMPLETED").is_file():
        raise NHLTeamCaptureError("Créneau hors périmètre ou incomplet.")
    try:
        receipt = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
        raw = (path / "response.csv").read_bytes()
        observed = _utc(datetime.fromisoformat(receipt["observed_at_utc"]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise NHLTeamCaptureError("Preuve de capture illisible.") from error
    digest = sha256(raw).hexdigest()
    history = replace(_parse(raw), observed_at_utc=observed)
    expected = {"schema_version": SCHEMA, "status": "CAPTURE_ONLY_NOT_MODEL_ELIGIBLE",
                "source_page": SOURCE_PAGE, "source_url": TEAM_GAME_DOWNLOAD,
                "attribution": ATTRIBUTION, "response_sha256": digest,
                "network_source_independently_verified": False,
                "response_bytes": len(raw), "policy_sha256": _policy_hash(),
                "effective_available_at_utc": observed.isoformat(),
                "five_season_coverage": _coverage(history),
                "regular_coverage_complete": history.regular_coverage_complete,
                "historical_asof_availability_proven": False,
                "training_permitted": False, "prediction_publication_permitted": False}
    if not isinstance(receipt, dict) or receipt.get("acquisition_mode") not in {"https_get", "local_file_intake"} or any(receipt.get(k) != v for k, v in expected.items()):
        raise NHLTeamCaptureError("Preuve de capture incohérente ou altérée.")
    if not isinstance(receipt.get("code_commit"), str) or not _COMMIT.fullmatch(receipt["code_commit"]):
        raise NHLTeamCaptureError("Commit de capture invalide.")
    return VerifiedTeamCapture(path, observed, digest, history)

def assess_history_before_game(capture: VerifiedTeamCapture, game: ScheduledGame,
                               *, lead_minutes: int) -> dict[str, object]:
    """Les lignes héritent de l'heure de réception, jamais de leur date de match."""
    cutoff = build_information_cutoff(game.start_utc, lead_minutes=lead_minutes)
    before = capture.observed_at_utc <= cutoff
    counts = {game.away_abbr: 0, game.home_abbr: 0}
    if before:
        for row in capture.history.regular_rows:
            if row.situation == "all" and row.team in counts and row.game_date < game.start_utc.date():
                counts[row.team] += 1
    return {"game_id": game.game_id, "cutoff_utc": cutoff.isoformat(),
            "capture_sha256": capture.response_sha256,
            "captured_before_cutoff": before,
            "prior_regular_games_by_team": counts,
            "historical_backtest_asof_proven": False,
            "training_permitted": False, "prediction_publication_permitted": False}
