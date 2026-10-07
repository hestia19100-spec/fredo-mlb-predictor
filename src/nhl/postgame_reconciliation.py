"""Read-only NHL postgame status linked to an immutable pregame checkpoint.

No outcome becomes a pregame input, model label, prediction, or bet here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

from .database import PROJECT_ROOT
from .official_final_capture import verify_official_final_capture
from .prospective_checkpoint import verify_checkpoint

SCHEMA = "nhl33_postgame_reconciliation_v1"
POLICY = PROJECT_ROOT / "nhl_protocols/data/nhl33_postgame_reconciliation_v1.json"
ROOT = PROJECT_ROOT / "nhl_postgame_reconciliations"
MAX_MANIFEST_BYTES = 128 * 1024


class NHLPostgameReconciliationError(ValueError):
    """A postgame manifest is incomplete, contradictory, or unverifiable."""


def _bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _stamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset().total_seconds() != 0:
        raise NHLPostgameReconciliationError("Horloge UTC requise.")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _time(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise NHLPostgameReconciliationError("Heure UTC non canonique.")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise NHLPostgameReconciliationError("Heure UTC invalide.") from error


def _policy_sha256() -> str:
    try:
        raw = POLICY.read_bytes()
        policy = json.loads(raw)
    except (OSError, ValueError) as error:
        raise NHLPostgameReconciliationError("Protocole NHL-33 illisible.") from error
    required = {
        "schema_version": SCHEMA, "manual_seal_only": True,
        "network_calls_allowed": False, "incomplete_games_remain_pending": True,
        "conflicting_final_captures_rejected": True,
        "pregame_checkpoint_mutation_allowed": False,
        "training_permitted": False, "prediction_publication_permitted": False,
        "odds_ingestion": False, "mlb_database_mutation": False,
        "nhl_historical_database_mutation": False,
    }
    if not isinstance(policy, dict) or any(policy.get(k) != v for k, v in required.items()):
        raise NHLPostgameReconciliationError("Protocole NHL-33 incompatible.")
    return sha256(raw).hexdigest()


def build_postgame_reconciliation(checkpoint_path: Path,
                                  capture_slots: tuple[Path, ...], *,
                                  checked_at_utc: datetime) -> dict[str, object]:
    """Recompute coverage solely from verified postgame captures."""
    checked_stamp = _stamp(checked_at_utc)
    proof = verify_checkpoint(checkpoint_path)
    if checked_at_utc < _time(proof["sealed_at_utc"]):
        raise NHLPostgameReconciliationError("Bilan anterieur a la preuve avant match.")
    if len(capture_slots) > 1000:
        raise NHLPostgameReconciliationError("Trop de captures finales.")
    seen_slots: set[Path] = set()
    by_game: dict[int, list] = {}
    for slot in capture_slots:
        path = Path(slot).resolve()
        if path in seen_slots:
            raise NHLPostgameReconciliationError("Capture finale dupliquee.")
        seen_slots.add(path)
        receipt = verify_official_final_capture(path, checkpoint_path)
        if receipt.observed_at_utc > checked_at_utc:
            raise NHLPostgameReconciliationError("Resultat posterieur a la date du bilan.")
        by_game.setdefault(receipt.game_id, []).append(receipt)
    games = []
    for game in sorted(proof["games"], key=lambda value: value["game_id"]):
        receipts = by_game.get(game["game_id"], [])
        row = {
            "game_id": game["game_id"], "away_abbr": game["away_abbr"],
            "home_abbr": game["home_abbr"],
            "scheduled_start_utc": game["scheduled_start_utc"],
            "feature_sha256": game["feature_sha256"],
        }
        if receipts:
            outcomes = {(item.away_score, item.home_score, item.winner_abbr)
                        for item in receipts}
            if len(outcomes) != 1:
                raise NHLPostgameReconciliationError("Captures officielles contradictoires.")
            first = min(receipts, key=lambda item: (item.observed_at_utc, item.response_sha256))
            row.update({
                "status": "FINAL_VERIFIED", "away_score": first.away_score,
                "home_score": first.home_score, "winner_abbr": first.winner_abbr,
                "first_observed_at_utc": _stamp(first.observed_at_utc),
                "response_sha256": first.response_sha256,
                "capture_count": len(receipts),
            })
        else:
            row.update({
                "status": "PENDING_OFFICIAL_FINAL", "away_score": None,
                "home_score": None, "winner_abbr": None,
                "first_observed_at_utc": None, "response_sha256": None,
                "capture_count": 0,
            })
        games.append(row)
    final_count = sum(game["status"] == "FINAL_VERIFIED" for game in games)
    report = {
        "schema_version": SCHEMA, "checked_at_utc": checked_stamp,
        "target_date": proof["target_date"],
        "checkpoint_sha256": proof["proof_sha256"],
        "policy_sha256": _policy_sha256(),
        "game_count": len(games), "verified_final_count": final_count,
        "pending_count": len(games) - final_count, "games": games,
        "training_permitted": False, "prediction_publication_permitted": False,
        "odds_ingestion": False,
    }
    return {**report, "report_sha256": sha256(_bytes(report)).hexdigest()}


def seal_postgame_reconciliation(checkpoint_path: Path,
                                 capture_slots: tuple[Path, ...], *,
                                 explicit_manual_run: bool = False,
                                 root: Path = ROOT,
                                 now=lambda: datetime.now(timezone.utc)) -> Path:
    """Write an append-only manifest; never alter captures or checkpoint."""
    if explicit_manual_run is not True:
        raise NHLPostgameReconciliationError("Scellement manuel explicite requis.")
    report = build_postgame_reconciliation(checkpoint_path, capture_slots,
                                           checked_at_utc=now())
    directory = Path(root) / report["target_date"]
    if Path(root).is_symlink() or directory.is_symlink():
        raise NHLPostgameReconciliationError("Repertoire de preuve non autorise.")
    directory.mkdir(parents=True, exist_ok=True)
    name = report["checked_at_utc"].replace(":", "").replace("-", "")
    path = directory / (name + "-" + report["report_sha256"][:16] + ".json")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(_bytes(report) + b"\n")
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise NHLPostgameReconciliationError("Bilan deja present ou ecriture impossible.") from error
    return path


def verify_postgame_reconciliation(path: Path, checkpoint_path: Path,
                                   capture_slots: tuple[Path, ...]) -> dict[str, object]:
    """Rebuild the manifest from the unchanged source proofs and compare."""
    try:
        raw = Path(path).read_bytes()
        if len(raw) > MAX_MANIFEST_BYTES:
            raise NHLPostgameReconciliationError("Bilan trop volumineux.")
        document = json.loads(raw)
        expected = build_postgame_reconciliation(
            checkpoint_path, capture_slots,
            checked_at_utc=_time(document["checked_at_utc"]),
        )
        if document != expected:
            raise NHLPostgameReconciliationError("Bilan et preuves divergent.")
        return expected
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise NHLPostgameReconciliationError("Bilan NHL-33 non verifiable.") from error
