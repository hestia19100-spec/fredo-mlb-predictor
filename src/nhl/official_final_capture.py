"""Manual NHL final-result capture, linked to immutable pregame evidence.

This postgame archive cannot be used to change the sealed pregame checkpoint.
It does not train a model, collect odds, or publish predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

from .database import NHL_DATA_ROOT, PROJECT_ROOT
from .prospective_checkpoint import verify_checkpoint

BASE_URL = "https://api-web.nhle.com/v1/gamecenter/"
SCHEMA = "nhl32_official_final_capture_v1"
POLICY = PROJECT_ROOT / "nhl_protocols/data/nhl32_official_final_capture_v1.json"
ROOT = NHL_DATA_ROOT / "official_final_captures"
MAX_BYTES = 1_000_000


class OfficialFinalCaptureError(ValueError):
    """An official final cannot be safely bound to the pregame checkpoint."""


@dataclass(frozen=True, slots=True)
class OfficialFinalReceipt:
    path: Path
    game_id: int
    observed_at_utc: datetime
    response_sha256: str
    away_score: int
    home_score: int
    winner_abbr: str


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise OfficialFinalCaptureError("Champ JSON duplique.")
        result[key] = value
    return result


def _utc(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise OfficialFinalCaptureError("Heure UTC canonique requise.")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise OfficialFinalCaptureError("Heure UTC invalide.") from error


def _stamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset().total_seconds() != 0:
        raise OfficialFinalCaptureError("Horloge UTC requise.")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _policy_sha256() -> str:
    try:
        raw = POLICY.read_bytes()
        policy = json.loads(raw, object_pairs_hook=_unique)
    except (OSError, ValueError) as error:
        raise OfficialFinalCaptureError("Protocole NHL-32 illisible.") from error
    required = {
        "schema_version": SCHEMA, "manual_only": True,
        "endpoint_template": BASE_URL + "{game_id}/landing",
        "direct_https_required": True, "regular_season_only": True,
        "final_state_required": "OFF", "same_game_identity_required": True,
        "changed_start_requires_review": True,
        "automatic_daily_collection_allowed": False,
        "training_permitted": False, "prediction_publication_permitted": False,
        "odds_ingestion": False, "mlb_database_mutation": False,
        "nhl_historical_database_mutation": False,
    }
    if not isinstance(policy, dict) or any(policy.get(key) != value for key, value in required.items()):
        raise OfficialFinalCaptureError("Protocole NHL-32 incompatible.")
    return sha256(raw).hexdigest()


def _game(proof: dict[str, object], game_id: int) -> dict[str, object]:
    if type(game_id) is not int:
        raise OfficialFinalCaptureError("Identifiant de match invalide.")
    matches = [game for game in proof["games"] if game["game_id"] == game_id]
    if len(matches) != 1:
        raise OfficialFinalCaptureError("Match absent de la preuve avant match.")
    return matches[0]


def _final(raw: bytes, game: dict[str, object], observed: datetime) -> tuple[int, int, str]:
    if not 0 < len(raw) <= MAX_BYTES:
        raise OfficialFinalCaptureError("Taille de reponse invalide.")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OfficialFinalCaptureError("Reponse NHL illisible.") from error
    if not isinstance(document, dict):
        raise OfficialFinalCaptureError("Objet NHL invalide.")
    game_id = game["game_id"]
    season = int(str(game_id)[:4]) * 10_000 + int(str(game_id)[:4]) + 1
    if (document.get("id") != game_id or document.get("season") != season
            or document.get("gameType") != 2
            or document.get("gameState") != "OFF"
            or document.get("gameScheduleState") != "OK"):
        raise OfficialFinalCaptureError("Match non termine ou identite incompatible.")
    start = _utc(document.get("startTimeUTC"))
    if start != _utc(game["scheduled_start_utc"]):
        raise OfficialFinalCaptureError("Horaire modifie : examen manuel requis.")
    if observed <= start:
        raise OfficialFinalCaptureError("Resultat observe avant le match.")
    away, home = document.get("awayTeam"), document.get("homeTeam")
    if (not isinstance(away, dict) or not isinstance(home, dict)
            or away.get("abbrev") != game["away_abbr"]
            or home.get("abbrev") != game["home_abbr"]):
        raise OfficialFinalCaptureError("Equipes incompatibles avec la preuve.")
    away_score, home_score = away.get("score"), home.get("score")
    if (type(away_score) is not int or type(home_score) is not int
            or min(away_score, home_score) < 0 or away_score == home_score):
        raise OfficialFinalCaptureError("Score final decisif absent.")
    winner = game["away_abbr"] if away_score > home_score else game["home_abbr"]
    return away_score, home_score, winner


def _receipt(proof: dict[str, object], game_id: int, observed: datetime,
             raw: bytes, final: tuple[int, int, str]) -> dict[str, object]:
    away_score, home_score, winner = final
    return {
        "schema_version": SCHEMA, "status": "OFFICIAL_FINAL_OBSERVED",
        "source_url": BASE_URL + str(game_id) + "/landing",
        "game_id": game_id, "observed_at_utc": _stamp(observed),
        "checkpoint_sha256": proof["proof_sha256"],
        "response_sha256": sha256(raw).hexdigest(),
        "policy_sha256": _policy_sha256(),
        "away_score": away_score, "home_score": home_score,
        "winner_abbr": winner, "game_state": "OFF",
        "training_permitted": False, "prediction_publication_permitted": False,
        "odds_ingestion": False,
    }


def capture_official_final(checkpoint_path: Path, game_id: int, *,
                           explicit_manual_run: bool = False,
                           transport: Callable | None = None,
                           root: Path = ROOT,
                           now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                           ) -> OfficialFinalReceipt:
    """One explicit direct GET; keep full response private and append-only."""
    if explicit_manual_run is not True or transport is None:
        raise OfficialFinalCaptureError("Capture manuelle et transport explicite requis.")
    proof = verify_checkpoint(checkpoint_path)
    game = _game(proof, game_id)
    _policy_sha256()
    started = now()
    _stamp(started)
    if started <= _utc(game["scheduled_start_utc"]):
        raise OfficialFinalCaptureError("Match non commence : collecte refusee.")
    url = BASE_URL + str(game_id) + "/landing"
    try:
        with transport(url, timeout=15) as response:
            if (response.geturl() != url or response.status != 200
                    or response.headers.get("Content-Type", "").split(";", 1)[0].lower() != "application/json"):
                raise OfficialFinalCaptureError("Reponse NHL redirigee ou invalide.")
            raw = response.read(MAX_BYTES + 1)
    except OfficialFinalCaptureError:
        raise
    except (OSError, ValueError) as error:
        raise OfficialFinalCaptureError("Collecte officielle echouee.") from error
    observed = now()
    if observed < started:
        raise OfficialFinalCaptureError("Horloge de collecte incoherente.")
    _stamp(observed)
    final = _final(raw, game, observed)
    receipt = _receipt(proof, game_id, observed, raw, final)
    key = observed.strftime("%Y%m%dT%H%M%S%fZ") + "-" + receipt["response_sha256"][:16]
    slot = Path(root) / proof["target_date"] / str(game_id) / key
    try:
        slot.mkdir(parents=True, exist_ok=False)
        with (slot / "response.json").open("xb") as output:
            output.write(raw)
        with (slot / "receipt.json").open("xb") as output:
            output.write((json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
        with (slot / "COMPLETED").open("xb"):
            pass
    except OSError as error:
        raise OfficialFinalCaptureError("Archivage NHL incomplet.") from error
    return OfficialFinalReceipt(slot, game_id, observed, receipt["response_sha256"], *final)


def verify_official_final_capture(slot: Path, checkpoint_path: Path) -> OfficialFinalReceipt:
    """Verify raw bytes, identity, timing and the unchanged pregame checkpoint."""
    slot = Path(slot)
    if not (slot / "COMPLETED").is_file():
        raise OfficialFinalCaptureError("Capture non terminee.")
    try:
        raw = (slot / "response.json").read_bytes()
        receipt = json.loads((slot / "receipt.json").read_bytes(), object_pairs_hook=_unique)
    except (OSError, ValueError) as error:
        raise OfficialFinalCaptureError("Archive NHL illisible.") from error
    if not isinstance(receipt, dict) or type(receipt.get("game_id")) is not int:
        raise OfficialFinalCaptureError("Recu final invalide.")
    proof = verify_checkpoint(checkpoint_path)
    game = _game(proof, receipt["game_id"])
    observed = _utc(receipt.get("observed_at_utc"))
    final = _final(raw, game, observed)
    if receipt != _receipt(proof, receipt["game_id"], observed, raw, final):
        raise OfficialFinalCaptureError("Capture et recu divergent.")
    return OfficialFinalReceipt(slot, receipt["game_id"], observed,
                                receipt["response_sha256"], *final)
