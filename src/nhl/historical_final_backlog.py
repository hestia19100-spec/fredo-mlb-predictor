"""Read-only NHL tied-final backlog and one-match manual resumption.

No network access occurs while inspecting the backlog. A resume call makes at
most one request through an explicitly supplied transport.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

from .database import PROJECT_ROOT
from .historical_official_final import (
    ROOT, HistoricalFinalReceipt, HistoricalOfficialFinalError,
    _verify, capture_historical_final,
)
from .moneypuck_five_season import REFERENCE_SEASONS
from .moneypuck_team_capture import DEFAULT_ROOT as TEAM_ROOT, verify_team_capture
from .real_archive_audit import audit_reference_history

POLICY = PROJECT_ROOT / "nhl_protocols/data/nhl38_historical_final_backlog_v1.json"
SCHEMA = "nhl38_historical_final_backlog_v1"


def _policy_hash() -> str:
    try:
        raw = POLICY.read_bytes()
        policy = json.loads(raw)
    except (OSError, ValueError) as error:
        raise HistoricalOfficialFinalError("Protocole de reprise illisible.") from error
    required = {
        "schema_version": SCHEMA, "inventory_network_allowed": False,
        "manual_single_game_only": True,
        "incomplete_or_duplicate_archive_rejected": True,
        "historical_pregame_availability_proven": False,
        "origin_independently_verified": False,
        "training_permitted": False, "prediction_publication_permitted": False,
        "database_mutation_allowed": False,
    }
    if not isinstance(policy, dict) or any(policy.get(key) != value for key, value in required.items()):
        raise HistoricalOfficialFinalError("Protocole de reprise incompatible.")
    return sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class HistoricalFinalBacklog:
    source_capture_sha256: str
    verified_game_ids: tuple[int, ...]
    pending_tied_game_ids: tuple[int, ...]
    backlog_sha256: str
    training_permitted: bool = False
    prediction_publication_permitted: bool = False

    @property
    def next_game_id(self) -> int | None:
        return self.pending_tied_game_ids[0] if self.pending_tied_game_ids else None


def _slots(root: Path) -> tuple[Path, ...]:
    """Reject malformed or incomplete evidence rather than silently skipping it."""
    if root.is_symlink():
        raise HistoricalOfficialFinalError("Racine des resultats historiques invalide.")
    if not root.exists():
        return ()
    if not root.is_dir():
        raise HistoricalOfficialFinalError("Racine des resultats historiques invalide.")
    found = []
    for season in sorted(root.iterdir()):
        if (season.is_symlink() or not season.is_dir()
                or not season.name.isdecimal() or int(season.name) not in REFERENCE_SEASONS
                or season.name != str(int(season.name))):
            raise HistoricalOfficialFinalError("Dossier de saison historique invalide.")
        games = tuple(sorted(season.iterdir()))
        if not games:
            raise HistoricalOfficialFinalError("Dossier de saison vide.")
        for game in games:
            if (game.is_symlink() or not game.is_dir()
                    or len(game.name) != 10 or not game.name.isdecimal()
                    or int(game.name) // 1_000_000 != int(season.name)):
                raise HistoricalOfficialFinalError("Dossier de match historique invalide.")
            slots = tuple(sorted(game.iterdir()))
            if not slots:
                raise HistoricalOfficialFinalError("Dossier de match vide.")
            for slot in slots:
                if slot.is_symlink() or not slot.is_dir() or not (slot / "COMPLETED").is_file():
                    raise HistoricalOfficialFinalError("Capture historique incomplete ou invalide.")
                found.append(slot)
    return tuple(found)


def inspect_historical_final_backlog(
    team_slot: Path, *, root: Path = ROOT, team_root: Path = TEAM_ROOT,
) -> HistoricalFinalBacklog:
    """Verify every local final, then enumerate only unresolved source ties."""
    policy_sha256 = _policy_hash()
    source = verify_team_capture(team_slot, allowed_root=team_root)
    audit = audit_reference_history(source.history)
    receipts = tuple(_verify(slot, source, root) for slot in _slots(Path(root)))
    game_ids = tuple(item.game_id for item in receipts)
    verified_ids = set(game_ids)
    if len(game_ids) != len(verified_ids):
        raise HistoricalOfficialFinalError("Plusieurs captures pour un meme match.")
    pending = tuple(game_id for game_id in audit.tied_score_game_ids
                    if game_id not in verified_ids)
    proof = {
        "schema_version": SCHEMA,
        "policy_sha256": policy_sha256,
        "source_capture_sha256": source.response_sha256,
        "verified_finals": sorted((item.game_id, item.response_sha256,
                                   item.observed_at_utc.isoformat()) for item in receipts),
        "pending_tied_game_ids": pending,
    }
    digest = sha256(json.dumps(proof, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return HistoricalFinalBacklog(source.response_sha256, tuple(sorted(game_ids)), pending, digest)


def capture_next_tied_final(
    team_slot: Path, *, explicit_manual_run: bool = False,
    transport: Callable | None = None, root: Path = ROOT, team_root: Path = TEAM_ROOT,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> HistoricalFinalReceipt | None:
    """One explicit request for the first missing tie; never a background loop."""
    if explicit_manual_run is not True or transport is None:
        raise HistoricalOfficialFinalError("Reprise manuelle explicite requise.")
    backlog = inspect_historical_final_backlog(team_slot, root=root, team_root=team_root)
    if backlog.next_game_id is None:
        return None
    return capture_historical_final(
        team_slot, backlog.next_game_id, explicit_manual_run=True,
        transport=transport, root=root, team_root=team_root, now=now,
    )
