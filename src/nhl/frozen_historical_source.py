"""Select the immutable NHL reference capture used by historical final receipts.

The newest MoneyPuck download can contain current-season updates and therefore
has a different whole-file digest even when 2021-2025 rows are unchanged.
Never rebind an official result to a different source capture. This module
only audits candidate coverage; it does not fit or authorize a model.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from .historical_backtest_readiness import audit_historical_backtest_readiness
from .historical_final_backlog import _slots, inspect_historical_final_backlog
from .historical_official_final import ROOT as FINAL_ROOT
from .moneypuck_team_capture import DEFAULT_ROOT as TEAM_ROOT, verify_team_capture

MAX_RECEIPT_BYTES = 64 * 1024


class FrozenHistoricalSourceError(ValueError):
    """The archived results do not identify one unambiguous reference source."""


@dataclass(frozen=True, slots=True)
class FrozenHistoricalSourceAudit:
    team_slot: Path
    source_capture_sha256: str
    verified_official_tie_finals: int
    unresolved_tie_finals: int
    candidate_train_games: int
    candidate_validation_games: int
    candidate_holdout_games: int
    audit_sha256: str
    historical_pregame_availability_proven: bool = False
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def _receipt(path: Path) -> dict:
    if path.is_symlink():
        raise FrozenHistoricalSourceError("Lien symbolique de reçu non admis.")
    try:
        raw = path.read_bytes()
        if not 0 < len(raw) <= MAX_RECEIPT_BYTES:
            raise FrozenHistoricalSourceError("Reçu absent ou trop volumineux.")
        value = json.loads(raw)
    except (OSError, ValueError, UnicodeError) as error:
        raise FrozenHistoricalSourceError("Reçu historique illisible.") from error
    if not isinstance(value, dict):
        raise FrozenHistoricalSourceError("Reçu historique invalide.")
    return value


def _digest(value: object) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise FrozenHistoricalSourceError("Empreinte historique invalide.")
    return value


def audit_frozen_historical_source(
    *, team_root: Path = TEAM_ROOT, final_root: Path = FINAL_ROOT,
) -> FrozenHistoricalSourceAudit:
    """Verify source identity and all tied finals without network or writes."""
    team_root, final_root = Path(team_root), Path(final_root)
    final_slots = _slots(final_root)
    if not final_slots:
        raise FrozenHistoricalSourceError("Aucun résultat officiel historique.")
    source_hashes = {
        _digest(_receipt(slot / "receipt.json").get("source_capture_sha256"))
        for slot in final_slots
    }
    if len(source_hashes) != 1:
        raise FrozenHistoricalSourceError("Résultats liés à plusieurs captures statistiques.")
    source_hash = source_hashes.pop()
    if not team_root.is_dir() or team_root.is_symlink():
        raise FrozenHistoricalSourceError("Racine des captures statistiques invalide.")
    matching: list[Path] = []
    for marker in team_root.rglob("COMPLETED"):
        slot = marker.parent
        if slot.is_symlink() or marker.is_symlink():
            raise FrozenHistoricalSourceError("Capture statistique symbolique.")
        receipt = _receipt(slot / "receipt.json")
        if receipt.get("response_sha256") == source_hash:
            matching.append(slot)
    if len(matching) != 1:
        raise FrozenHistoricalSourceError("Capture statistique de référence absente ou ambiguë.")
    source = verify_team_capture(matching[0], allowed_root=team_root)
    if source.response_sha256 != source_hash:
        raise FrozenHistoricalSourceError("Capture statistique divergente.")
    backlog = inspect_historical_final_backlog(
        source.path, root=final_root, team_root=team_root)
    readiness = audit_historical_backtest_readiness(source.history)
    if (backlog.source_capture_sha256 != source_hash
            or not readiness.regular_coverage_complete
            or readiness.inconsistent_game_ids
            or backlog.pending_tied_game_ids
            or set(backlog.verified_game_ids) != set(readiness.unresolved_tied_game_ids)):
        raise FrozenHistoricalSourceError("Couverture historique ou résultats incomplets.")
    counts = {role: sum(
        season.both_teams_have_prior_form for season in readiness.seasons
        if season.role == role
    ) for role in ("TRAIN", "VALIDATION", "HOLDOUT")}
    proof = {
        "schema": "nhl_frozen_historical_source_audit_v1",
        "source_capture_sha256": source_hash,
        "source_observed_at_utc": source.observed_at_utc.isoformat(),
        "backlog_sha256": backlog.backlog_sha256,
        "readiness_sha256": readiness.audit_sha256,
        "verified_tie_finals": len(backlog.verified_game_ids),
        "candidate_games": counts,
        "historical_pregame_availability_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }
    digest = sha256(json.dumps(
        proof, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return FrozenHistoricalSourceAudit(
        source.path, source_hash, len(backlog.verified_game_ids),
        len(backlog.pending_tied_game_ids),
        counts["TRAIN"], counts["VALIDATION"], counts["HOLDOUT"], digest,
    )


def main() -> int:
    result = audit_frozen_historical_source()
    print(json.dumps({
        "source_capture_sha256": result.source_capture_sha256,
        "team_slot": str(result.team_slot),
        "verified_official_tie_finals": result.verified_official_tie_finals,
        "unresolved_tie_finals": result.unresolved_tie_finals,
        "candidate_train_games": result.candidate_train_games,
        "candidate_validation_games": result.candidate_validation_games,
        "candidate_holdout_games": result.candidate_holdout_games,
        "audit_sha256": result.audit_sha256,
        "training_permitted": result.training_permitted,
        "prediction_publication_permitted": result.prediction_publication_permitted,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
