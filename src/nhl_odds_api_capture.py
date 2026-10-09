"""Audited, capture-only NHL event and two-way odds feed.

This feed is bookmaker-listed, not the complete official NHL schedule. Its
provider IDs must never be used as NHL game IDs or as training labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

import requests

from src.nhl.database import NHL_DATA_ROOT
from src.nhl.odds_api_candidates import (
    NHLOddsCandidateError,
    parse_nhl_events,
    parse_nhl_h2h_odds,
)
from src.odds_api import _read_api_key


EVENTS_URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/events"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
DEFAULT_ROOT = NHL_DATA_ROOT / "odds_api_capture_only"
SCHEMA = "nhl_odds_api_capture_only_v1"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class NHLOddsCaptureError(RuntimeError):
    """The provider feed could not be captured safely."""


@dataclass(frozen=True, slots=True)
class NHLOddsCaptureReceipt:
    path: Path
    observed_at_utc: datetime
    event_count: int
    two_way_quote_count: int
    odds_quota_cost: int


def _utc(now: Callable[[], datetime]) -> datetime:
    value = now()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise NHLOddsCaptureError("Horloge UTC requise.")
    return value.astimezone(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _quota(headers: object, name: str) -> int:
    try:
        value = headers.get(name)
        if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
            raise ValueError
        return int(value)
    except (AttributeError, ValueError, TypeError) as error:
        raise NHLOddsCaptureError("En-têtes de quota absents ou invalides.") from error


def _get(transport: Callable, url: str, key: str, params: dict[str, str]) -> tuple[bytes, dict[str, int]]:
    # No exception text is propagated: requests may include the secret query key.
    try:
        response = transport(url, params={"apiKey": key, **params}, timeout=30, allow_redirects=False)
        if response.status_code != 200 or response.url.split("?", 1)[0] != url:
            raise NHLOddsCaptureError("Réponse NHL non réussie ou redirigée.")
        if response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise NHLOddsCaptureError("Type de réponse NHL inattendu.")
        raw = response.content
        if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_RESPONSE_BYTES:
            raise NHLOddsCaptureError("Réponse NHL vide ou trop volumineuse.")
        quota = {name: _quota(response.headers, name) for name in (
            "x-requests-last", "x-requests-remaining", "x-requests-used",
        )}
        return raw, quota
    except NHLOddsCaptureError:
        raise
    except Exception as error:
        raise NHLOddsCaptureError("Collecte NHL échouée, sans archivage.") from None


def capture_nhl_odds_candidates(
    *,
    root: Path = DEFAULT_ROOT,
    transport: Callable = requests.get,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> NHLOddsCaptureReceipt:
    """One no-cost events query and one one-region odds query; never scores.

    This is an explicit capture operation, not a scheduler. Only already-listed
    events can be observed; an empty response is not proof of zero NHL games.
    """
    key = _read_api_key()
    started = _utc(now)
    events_raw, events_quota = _get(transport, EVENTS_URL, key, {})
    odds_raw, odds_quota = _get(transport, ODDS_URL, key, {
        "bookmakers": "netbet_fr", "markets": "h2h", "oddsFormat": "decimal",
    })
    observed = _utc(now)
    if observed < started:
        raise NHLOddsCaptureError("Horloge incohérente.")
    try:
        events = parse_nhl_events(events_raw)
        odds = parse_nhl_h2h_odds(odds_raw)
    except NHLOddsCandidateError as error:
        raise NHLOddsCaptureError("Réponses NHL incompatibles.") from error
    by_id = {event.provider_event_id: event for event in events}
    for event in odds.events:
        if by_id.get(event.provider_event_id) != event:
            raise NHLOddsCaptureError("Événement coté absent ou divergent de la liste.")
    for quote in odds.two_way_quotes:
        if quote.market_observed_at_utc > observed:
            raise NHLOddsCaptureError("Cote datée dans le futur.")
    if events_quota["x-requests-last"] != 0 or odds_quota["x-requests-last"] > 1:
        raise NHLOddsCaptureError("Coût de requête NHL inattendu.")
    receipt = {
        "schema_version": SCHEMA,
        "status": "CAPTURE_ONLY_NOT_MODEL_ELIGIBLE",
        "provider": "the_odds_api",
        "source_events_url": EVENTS_URL,
        "source_odds_url": ODDS_URL,
        "bookmaker": "netbet_fr",
        "market": "h2h_two_outcomes_only",
        "started_at_utc": _stamp(started),
        "observed_at_utc": _stamp(observed),
        "events_sha256": sha256(events_raw).hexdigest(),
        "odds_sha256": sha256(odds_raw).hexdigest(),
        "event_count": len(events),
        "two_way_quote_count": len(odds.two_way_quotes),
        "rejected_market_count": len(odds.rejected_market_keys),
        "events_quota": events_quota,
        "odds_quota": odds_quota,
        "schedule_complete": False,
        "nhl_game_ids_verified": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }
    key_name = observed.strftime("%Y%m%dT%H%M%S%fZ") + "-" + receipt["odds_sha256"][:16]
    slot = Path(root) / key_name
    try:
        slot.mkdir(parents=True, exist_ok=False)
        with (slot / "events.json").open("xb") as output:
            output.write(events_raw)
        with (slot / "odds.json").open("xb") as output:
            output.write(odds_raw)
        with (slot / "receipt.json").open("xb") as output:
            output.write((json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
        with (slot / "COMPLETED").open("xb"):
            pass
    except OSError as error:
        raise NHLOddsCaptureError("Archivage incomplet; créneau non utilisable.") from error
    return NHLOddsCaptureReceipt(slot, observed, len(events), len(odds.two_way_quotes), odds_quota["x-requests-last"])
