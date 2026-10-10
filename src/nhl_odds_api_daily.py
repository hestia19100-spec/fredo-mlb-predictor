"""Budgeted, capture-only NHL Odds API entry points; no scheduler or model use."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

import requests

from src.nhl.database import NHL_DATA_ROOT
from src.nhl.odds_api_scores import NHLScoreEvidenceError, parse_nhl_final_scores
from src.nhl_odds_api_capture import (
    NHLOddsCaptureReceipt,
    _get,
    _stamp,
    _utc,
    capture_nhl_odds_candidates,
)
from src.nhl_odds_api_quota import NHLDailyQuotaGate, SCORES_URL
from src.odds_api import _read_api_key


DAILY_LIMIT = 3
DEFAULT_LEDGER = NHL_DATA_ROOT / "odds_api_capture_only" / "daily_quota.sqlite"
DEFAULT_SCORES_ROOT = NHL_DATA_ROOT / "odds_api_scores_capture_only"
SCORES_SCHEMA = "nhl_odds_api_scores_capture_only_v1"


class NHLDailyScoresError(RuntimeError):
    """The provider scores could not be captured as safe evidence."""


@dataclass(frozen=True, slots=True)
class NHLDailyScoresReceipt:
    path: Path
    observed_at_utc: datetime
    final_count: int
    quota_cost: int


def capture_nhl_daily_pregame(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    root: Path = NHL_DATA_ROOT / "odds_api_capture_only",
    transport: Callable = requests.get,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    on_reserved: Callable[[], None] | None = None,
) -> NHLOddsCaptureReceipt:
    """Capture bookmaker-listed events and odds, once per Paris day."""
    gate = NHLDailyQuotaGate(ledger_path, DAILY_LIMIT, transport, now, on_reserved)
    return capture_nhl_odds_candidates(root=root, transport=gate.get, now=now)


def capture_nhl_daily_scores(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    root: Path = DEFAULT_SCORES_ROOT,
    transport: Callable = requests.get,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    on_reserved: Callable[[], None] | None = None,
) -> NHLDailyScoresReceipt:
    """Archive recent provider scores; they are not official NHL labels."""
    gate = NHLDailyQuotaGate(ledger_path, DAILY_LIMIT, transport, now, on_reserved)
    key = _read_api_key()
    started = _utc(now)
    raw, quota = _get(gate.get, SCORES_URL, key, {"daysFrom": "3", "dateFormat": "iso"})
    observed = _utc(now)
    if observed < started:
        raise NHLDailyScoresError("Horloge NHL incohérente.")
    try:
        finals = parse_nhl_final_scores(raw, observed)
    except NHLScoreEvidenceError as error:
        raise NHLDailyScoresError("Résultats fournisseur NHL invalides.") from error
    receipt = {
        "schema_version": SCORES_SCHEMA,
        "status": "CAPTURE_ONLY_NOT_OFFICIAL_LABELS",
        "provider": "the_odds_api",
        "source_url": SCORES_URL,
        "started_at_utc": _stamp(started),
        "observed_at_utc": _stamp(observed),
        "scores_sha256": sha256(raw).hexdigest(),
        "final_count": len(finals),
        "quota": quota,
        "nhl_game_ids_verified": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }
    slot = Path(root) / (observed.strftime("%Y%m%dT%H%M%S%fZ") + "-" + receipt["scores_sha256"][:16])
    try:
        slot.mkdir(parents=True, exist_ok=False)
        with (slot / "scores.json").open("xb") as output:
            output.write(raw)
        with (slot / "receipt.json").open("xb") as output:
            output.write((json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
        with (slot / "COMPLETED").open("xb"):
            pass
    except OSError as error:
        raise NHLDailyScoresError("Archivage NHL incomplet; créneau non utilisable.") from error
    return NHLDailyScoresReceipt(slot, observed, len(finals), quota["x-requests-last"])
