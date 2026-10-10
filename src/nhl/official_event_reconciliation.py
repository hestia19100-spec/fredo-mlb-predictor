"""Offline, fail-closed link between NHL schedule IDs and captured Odds API events.

A matched fixture is evidence about identity only: not a prediction, eligible
training row, or proof of a bookmaker's settlement rules. No network I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
import unicodedata

from .odds_api_candidates import (
    MAX_RESPONSE_BYTES,
    NHLEventCandidate,
    parse_nhl_events,
    parse_nhl_h2h_odds,
)
from .public_schedule_candidates import parse_public_schedule
from .public_schedule_capture import SCHEMA_VERSION, verify_public_schedule_capture

ODDS_RECEIPT_SCHEMA = "nhl_odds_api_capture_only_v2"
MAX_RECEIPT_BYTES = 64 * 1024


class NHLEventReconciliationError(ValueError):
    """Captured evidence cannot be reconciled safely."""


@dataclass(frozen=True, slots=True)
class MatchedNHLEvent:
    nhl_game_id: int
    provider_event_id: str
    season: int
    start_utc: datetime
    away_abbr: str
    home_abbr: str


@dataclass(frozen=True, slots=True)
class NHLEventReconciliation:
    target_date: date
    schedule_sha256: str
    provider_events_sha256: str
    matched: tuple[MatchedNHLEvent, ...]
    unmatched_nhl_game_ids: tuple[int, ...]
    unmatched_provider_event_ids: tuple[str, ...]
    ambiguous_nhl_game_ids: tuple[int, ...]
    late_nhl_game_ids: tuple[int, ...]
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def _utc(value: datetime) -> bool:
    return (isinstance(value, datetime) and value.tzinfo is not None
            and value.utcoffset() == timedelta(0))


def _name(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    text = value.get("default")
    return text.strip() if isinstance(text, str) and text.strip() else None


def _official_name(team: object) -> str | None:
    if not isinstance(team, dict):
        return None
    place, common = _name(team.get("placeName")), _name(team.get("commonName"))
    return f"{place} {common}" if place and common else None


def _key(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).casefold()
    words = []
    for char in folded:
        if unicodedata.category(char) == "Mn":
            continue
        words.append(char if char.isalnum() else " ")
    return " ".join("".join(words).split())


def reconcile_nhl_schedule_events(
    schedule_raw: bytes,
    target_date: date,
    schedule_observed_at_utc: datetime,
    provider_events_raw: bytes,
    provider_observed_at_utc: datetime,
) -> NHLEventReconciliation:
    """Match only exact home/away names and start; no fuzzy or ID assumptions.

    Both capture times must precede each match by at least one hour. A duplicate
    provider fixture is ambiguous and never linked. Missing provider rows do
    not imply that the official schedule has no games.
    """
    if not _utc(schedule_observed_at_utc) or not _utc(provider_observed_at_utc):
        raise NHLEventReconciliationError("Horodatages de capture UTC requis.")
    games = parse_public_schedule(schedule_raw, target_date, schedule_observed_at_utc)
    events = parse_nhl_events(provider_events_raw)
    if len({event.provider_event_id for event in events}) != len(events):
        raise NHLEventReconciliationError("Identifiant fournisseur dupliqué.")
    payload = json.loads(schedule_raw)
    day = next(item for item in payload["gameWeek"] if item["date"] == target_date.isoformat())
    by_game_id: dict[int, dict] = {}
    for game in games:
        rows = [
            row for row in day["games"]
            if (isinstance(row, dict) and row.get("id") == game.game_id
                and row.get("gameType") == 2 and row.get("gameState") == "FUT"
                and row.get("gameScheduleState") == "OK"
                and row.get("startTimeUTC") == game.start_utc.isoformat().replace("+00:00", "Z"))
        ]
        if len(rows) != 1:
            raise NHLEventReconciliationError("Match officiel ambigu.")
        by_game_id[game.game_id] = rows[0]
    by_key: dict[tuple[str, str, datetime], list[NHLEventCandidate]] = {}
    for event in events:
        key = (_key(event.away_team_name), _key(event.home_team_name), event.start_utc)
        by_key.setdefault(key, []).append(event)

    official_keys: dict[tuple[str, str, datetime], list[int]] = {}
    for game in games:
        row = by_game_id[game.game_id]
        away, home = _official_name(row.get("awayTeam")), _official_name(row.get("homeTeam"))
        if away and home:
            official_keys.setdefault((_key(away), _key(home), game.start_utc), []).append(game.game_id)

    matched: list[MatchedNHLEvent] = []
    unmatched: list[int] = []
    ambiguous: list[int] = []
    late: list[int] = []
    consumed: set[str] = set()
    for game in games:
        row = by_game_id[game.game_id]
        away = _official_name(row.get("awayTeam"))
        home = _official_name(row.get("homeTeam"))
        if not away or not home or _key(away) == _key(home):
            unmatched.append(game.game_id)
            continue
        fixture_key = (_key(away), _key(home), game.start_utc)
        if len(official_keys.get(fixture_key, [])) > 1:
            ambiguous.append(game.game_id)
            unmatched.append(game.game_id)
            continue
        candidates = by_key.get(fixture_key, [])
        if len(candidates) > 1:
            ambiguous.append(game.game_id)
            unmatched.append(game.game_id)
            continue
        if not candidates:
            unmatched.append(game.game_id)
            continue
        if (schedule_observed_at_utc > game.start_utc - timedelta(hours=1)
                or provider_observed_at_utc > game.start_utc - timedelta(hours=1)):
            late.append(game.game_id)
            unmatched.append(game.game_id)
            continue
        event = candidates[0]
        if event.provider_event_id in consumed:
            ambiguous.append(game.game_id)
            unmatched.append(game.game_id)
            continue
        consumed.add(event.provider_event_id)
        matched.append(MatchedNHLEvent(
            game.game_id, event.provider_event_id, game.season, game.start_utc,
            game.away_abbr, game.home_abbr,
        ))
    return NHLEventReconciliation(
        target_date,
        sha256(schedule_raw).hexdigest(),
        sha256(provider_events_raw).hexdigest(),
        tuple(matched),
        tuple(sorted(unmatched)),
        tuple(sorted(event.provider_event_id for event in events
                     if event.provider_event_id not in consumed)),
        tuple(sorted(ambiguous)),
        tuple(sorted(late)),
    )


def _read(slot: Path, filename: str, limit: int) -> bytes:
    path = slot / filename
    if path.is_symlink():
        raise NHLEventReconciliationError("Lien symbolique non admis.")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise NHLEventReconciliationError("Archive NHL illisible.") from error
    if not 0 < len(raw) <= limit:
        raise NHLEventReconciliationError("Archive NHL vide ou trop volumineuse.")
    return raw


def reconcile_archived_nhl_events(
    schedule_slot: Path, odds_slot: Path,
) -> NHLEventReconciliation:
    """Read two completed, hash-verified captures; never call either API."""
    schedule_slot, odds_slot = Path(schedule_slot), Path(odds_slot)
    if (schedule_slot.is_symlink() or odds_slot.is_symlink()
            or not (schedule_slot / "COMPLETED").is_file()
            or not (odds_slot / "COMPLETED").is_file()):
        raise NHLEventReconciliationError("Captures NHL non clôturées.")
    try:
        schedule_receipt = verify_public_schedule_capture(schedule_slot)
        if schedule_receipt["schema_version"] != SCHEMA_VERSION:
            raise NHLEventReconciliationError("Calendrier direct NHL requis.")
        odds_receipt = json.loads(_read(odds_slot, "receipt.json", MAX_RECEIPT_BYTES))
        events_raw = _read(odds_slot, "events.json", MAX_RESPONSE_BYTES)
        odds_raw = _read(odds_slot, "odds.json", MAX_RESPONSE_BYTES)
        if (odds_receipt.get("schema_version") != ODDS_RECEIPT_SCHEMA
                or odds_receipt.get("status") != "CAPTURE_ONLY_NOT_MODEL_ELIGIBLE"
                or odds_receipt.get("provider") != "the_odds_api"
                or odds_receipt.get("training_permitted") is not False
                or odds_receipt.get("prediction_publication_permitted") is not False
                or odds_receipt.get("events_sha256") != sha256(events_raw).hexdigest()
                or odds_receipt.get("odds_sha256") != sha256(odds_raw).hexdigest()):
            raise NHLEventReconciliationError("Reçu des cotes NHL incohérent.")
        events = parse_nhl_events(events_raw)
        odds = parse_nhl_h2h_odds(odds_raw)
        by_id = {event.provider_event_id: event for event in events}
        if (odds_receipt.get("event_count") != len(events)
                or odds_receipt.get("two_way_quote_count") != len(odds.two_way_quotes)
                or odds_receipt.get("rejected_market_count") != len(odds.rejected_market_keys)
                or any(by_id.get(event.provider_event_id) != event for event in odds.events)):
            raise NHLEventReconciliationError("Événements et cotes divergents.")
        target = date.fromisoformat(schedule_receipt["target_date"])
        schedule_at = datetime.fromisoformat(
            schedule_receipt["observed_at_utc"].replace("Z", "+00:00"))
        odds_at = datetime.fromisoformat(
            odds_receipt["observed_at_utc"].replace("Z", "+00:00"))
        if not _utc(odds_at):
            raise NHLEventReconciliationError("Horodatage des cotes invalide.")
        return reconcile_nhl_schedule_events(
            _read(schedule_slot, "response.json", MAX_RESPONSE_BYTES),
            target, schedule_at, events_raw, odds_at,
        )
    except NHLEventReconciliationError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        raise NHLEventReconciliationError("Captures NHL incompatibles.") from error
