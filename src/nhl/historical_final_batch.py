"""Manual 50-game historical-final batch built from audited ten-game pilots."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import sleep as system_sleep
from typing import Callable

from .database import PROJECT_ROOT
from .historical_final_backlog import inspect_historical_final_backlog
from .historical_final_pilot import (
    capture_historical_final_pilot, plan_historical_final_pilot,
)
from .historical_official_final import (
    ROOT, HistoricalFinalReceipt, HistoricalOfficialFinalError,
)
from .moneypuck_team_capture import DEFAULT_ROOT as TEAM_ROOT

SCHEMA = "nhl40_historical_final_batch_v1"
POLICY = PROJECT_ROOT / "nhl_protocols/data/nhl40_historical_final_batch_v1.json"
MAX_REQUESTS = 50
CHUNK_SIZE = 10
INTER_REQUEST_SECONDS = 1.0


def _policy_hash() -> str:
    try:
        raw = POLICY.read_bytes()
        policy = json.loads(raw)
    except (OSError, ValueError) as error:
        raise HistoricalOfficialFinalError("Protocole NHL-40 illisible.") from error
    required = {
        "schema_version": SCHEMA,
        "manual_only": True,
        "maximum_requests_per_run": MAX_REQUESTS,
        "pilot_chunk_size": CHUNK_SIZE,
        "minimum_seconds_between_requests": INTER_REQUEST_SECONDS,
        "backlog_hash_required_before_network": True,
        "stop_on_first_error": True,
        "automatic_retry_allowed": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
        "database_mutation_allowed": False,
    }
    if not isinstance(policy, dict) or any(policy.get(k) != v for k, v in required.items()):
        raise HistoricalOfficialFinalError("Protocole NHL-40 incompatible.")
    return sha256(raw).hexdigest()


def _limit(value: int) -> int:
    if type(value) is not int or not 1 <= value <= MAX_REQUESTS:
        raise HistoricalOfficialFinalError("Lot historique limite a 1-50 matchs.")
    return value


@dataclass(frozen=True, slots=True)
class HistoricalFinalBatchPlan:
    source_capture_sha256: str
    backlog_sha256: str
    game_ids: tuple[int, ...]
    request_limit: int
    policy_sha256: str
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def plan_historical_final_batch(
    team_slot: Path, *, request_limit: int = MAX_REQUESTS,
    root: Path = ROOT, team_root: Path = TEAM_ROOT,
) -> HistoricalFinalBatchPlan:
    """Select up to 50 missing tied games without contacting the NHL."""
    limit = _limit(request_limit)
    policy_sha256 = _policy_hash()
    backlog = inspect_historical_final_backlog(
        team_slot, root=root, team_root=team_root,
    )
    return HistoricalFinalBatchPlan(
        backlog.source_capture_sha256, backlog.backlog_sha256,
        backlog.pending_tied_game_ids[:limit], limit, policy_sha256,
    )


def capture_historical_final_batch(
    team_slot: Path, *, expected_backlog_sha256: str,
    explicit_manual_run: bool = False, request_limit: int = MAX_REQUESTS,
    transport: Callable | None = None, root: Path = ROOT,
    team_root: Path = TEAM_ROOT,
    sleep: Callable[[float], None] = system_sleep,
    on_chunk: Callable[[tuple[HistoricalFinalReceipt, ...]], None] | None = None,
) -> tuple[HistoricalFinalReceipt, ...]:
    """Capture a reviewed plan through ten-game pilots; preserve partial success."""
    if explicit_manual_run is not True or transport is None:
        raise HistoricalOfficialFinalError("Execution manuelle explicite requise.")
    if (not isinstance(expected_backlog_sha256, str)
            or len(expected_backlog_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_backlog_sha256)):
        raise HistoricalOfficialFinalError("Empreinte du retard invalide.")
    plan = plan_historical_final_batch(
        team_slot, request_limit=request_limit, root=root, team_root=team_root,
    )
    if plan.backlog_sha256 != expected_backlog_sha256:
        raise HistoricalOfficialFinalError("Retard historique modifie : refaire le plan.")

    calls = 0

    def paced_transport(url: str, timeout: int):
        nonlocal calls
        if calls:
            sleep(INTER_REQUEST_SECONDS)
        calls += 1
        return transport(url, timeout=timeout)

    completed: list[HistoricalFinalReceipt] = []
    for offset in range(0, len(plan.game_ids), CHUNK_SIZE):
        expected_ids = plan.game_ids[offset:offset + CHUNK_SIZE]
        pilot = plan_historical_final_pilot(
            team_slot, request_limit=len(expected_ids), root=root,
            team_root=team_root,
        )
        if pilot.game_ids != expected_ids:
            raise HistoricalOfficialFinalError("Lot historique modifie : interrompre.")
        chunk = capture_historical_final_pilot(
            team_slot, expected_backlog_sha256=pilot.backlog_sha256,
            explicit_manual_run=True, request_limit=len(expected_ids),
            transport=paced_transport, root=root, team_root=team_root,
        )
        if tuple(item.game_id for item in chunk) != expected_ids:
            raise HistoricalOfficialFinalError("Captures historiques inattendues.")
        completed.extend(chunk)
        if on_chunk is not None:
            on_chunk(chunk)
    return tuple(completed)
