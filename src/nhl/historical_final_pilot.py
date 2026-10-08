"""Manual, bounded pilot for unresolved historical NHL overtime/shootout finals.

The plan is read-only. Execution requires its exact backlog hash and never
retries a failed request. Completed single-game captures remain append-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

from .database import PROJECT_ROOT
from .historical_final_backlog import inspect_historical_final_backlog
from .historical_official_final import (
    ROOT, HistoricalFinalReceipt, HistoricalOfficialFinalError,
    capture_historical_final,
)
from .moneypuck_team_capture import DEFAULT_ROOT as TEAM_ROOT

SCHEMA = "nhl39_historical_final_pilot_v1"
POLICY = PROJECT_ROOT / "nhl_protocols/data/nhl39_historical_final_pilot_v1.json"
MAX_REQUESTS = 10


def _policy_hash() -> str:
    try:
        raw = POLICY.read_bytes()
        policy = json.loads(raw)
    except (OSError, ValueError) as error:
        raise HistoricalOfficialFinalError("Protocole NHL-39 illisible.") from error
    required = {
        "schema_version": SCHEMA,
        "manual_only": True,
        "maximum_requests_per_run": MAX_REQUESTS,
        "backlog_hash_required_before_network": True,
        "single_game_capture_contract_required": True,
        "automatic_retry_allowed": False,
        "origin_independently_verified": False,
        "historical_pregame_availability_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
        "database_mutation_allowed": False,
    }
    if not isinstance(policy, dict) or any(policy.get(k) != v for k, v in required.items()):
        raise HistoricalOfficialFinalError("Protocole NHL-39 incompatible.")
    return sha256(raw).hexdigest()


def _limit(value: int) -> int:
    if type(value) is not int or not 1 <= value <= MAX_REQUESTS:
        raise HistoricalOfficialFinalError("Lot historique limite a 1-10 matchs.")
    return value


@dataclass(frozen=True, slots=True)
class HistoricalFinalPilotPlan:
    source_capture_sha256: str
    backlog_sha256: str
    game_ids: tuple[int, ...]
    request_limit: int
    policy_sha256: str
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def plan_historical_final_pilot(
    team_slot: Path, *, request_limit: int = MAX_REQUESTS,
    root: Path = ROOT, team_root: Path = TEAM_ROOT,
) -> HistoricalFinalPilotPlan:
    """Choose at most ten missing tied games, without any network call."""
    limit = _limit(request_limit)
    policy_sha256 = _policy_hash()
    backlog = inspect_historical_final_backlog(
        team_slot, root=root, team_root=team_root,
    )
    return HistoricalFinalPilotPlan(
        backlog.source_capture_sha256, backlog.backlog_sha256,
        backlog.pending_tied_game_ids[:limit], limit, policy_sha256,
    )


def capture_historical_final_pilot(
    team_slot: Path, *, expected_backlog_sha256: str,
    explicit_manual_run: bool = False, request_limit: int = MAX_REQUESTS,
    transport: Callable | None = None, root: Path = ROOT,
    team_root: Path = TEAM_ROOT,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> tuple[HistoricalFinalReceipt, ...]:
    """Run the reviewed plan once; stop on first error and preserve successes.

    Each request uses NHL-36's independently checked single-game capture. A
    restart must obtain a new plan/hash, so verified games are not repeated.
    """
    if explicit_manual_run is not True or transport is None:
        raise HistoricalOfficialFinalError("Execution manuelle explicite requise.")
    if (not isinstance(expected_backlog_sha256, str)
            or len(expected_backlog_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_backlog_sha256)):
        raise HistoricalOfficialFinalError("Empreinte du retard invalide.")
    plan = plan_historical_final_pilot(
        team_slot, request_limit=request_limit, root=root, team_root=team_root,
    )
    if plan.backlog_sha256 != expected_backlog_sha256:
        raise HistoricalOfficialFinalError("Retard historique modifie : refaire le plan.")
    receipts = []
    for game_id in plan.game_ids:
        game_dir = Path(root) / str(game_id // 1_000_000) / str(game_id)
        if game_dir.exists() or game_dir.is_symlink():
            raise HistoricalOfficialFinalError("Match deja archive : interrompre le lot.")
        receipts.append(capture_historical_final(
            team_slot, game_id, explicit_manual_run=True, transport=transport,
            root=root, team_root=team_root, now=now,
        ))
    return tuple(receipts)
