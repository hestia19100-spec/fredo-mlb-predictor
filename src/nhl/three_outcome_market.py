"""Offline inspection of archived NHL three-outcome market shapes.

This does not establish that the third outcome settles as a regulation draw.
It never calls a provider, trains a model, or publishes a prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

from .odds_api_candidates import (
    FRENCH_BOOKMAKER_KEYS,
    MAX_RESPONSE_BYTES,
    NHLOddsCandidateError,
    _payload,
    _price,
    _text,
    _utc,
    parse_nhl_events,
    parse_nhl_h2h_odds,
)

RECEIPT_SCHEMA = "nhl_odds_api_capture_only_v2"
MAX_RECEIPT_BYTES = 64 * 1024


class NHLThreeOutcomeInspectionError(ValueError):
    """An archived market cannot be inspected without sufficient evidence."""


@dataclass(frozen=True, slots=True)
class NHLThreeOutcomeCandidate:
    provider_event_id: str
    bookmaker_key: str
    market_observed_at_utc: datetime
    away_decimal_odds: Decimal
    third_outcome_label: str
    third_decimal_odds: Decimal
    home_decimal_odds: Decimal
    # Three prices alone cannot prove the bookmaker's settlement rules.
    regulation_draw_verified: bool = False


@dataclass(frozen=True, slots=True)
class NHLThreeOutcomeInspection:
    capture_observed_at_utc: datetime
    events_sha256: str
    odds_sha256: str
    candidates: tuple[NHLThreeOutcomeCandidate, ...]
    late_three_outcome_shapes: int
    market_shape_counts: tuple[tuple[str, int, int, int], ...]
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def _read(path: Path, limit: int) -> bytes:
    try:
        if path.is_symlink():
            raise NHLThreeOutcomeInspectionError("Lien symbolique non admis.")
        raw = path.read_bytes()
    except OSError as error:
        raise NHLThreeOutcomeInspectionError("Archive NHL illisible.") from error
    if not 0 < len(raw) <= limit:
        raise NHLThreeOutcomeInspectionError("Archive NHL vide ou trop volumineuse.")
    return raw


def inspect_archived_three_outcome_market(slot: Path) -> NHLThreeOutcomeInspection:
    """Inspect only a completed, hash-verified pregame capture.

    Candidates are shape-only evidence, not approved 1/N/2 betting quotes.
    A capture after the one-hour cutoff is never returned as a candidate.
    """
    slot = Path(slot)
    if slot.is_symlink() or not (slot / "COMPLETED").is_file():
        raise NHLThreeOutcomeInspectionError("Capture NHL non clôturée.")
    receipt_raw = _read(slot / "receipt.json", MAX_RECEIPT_BYTES)
    events_raw = _read(slot / "events.json", MAX_RESPONSE_BYTES)
    odds_raw = _read(slot / "odds.json", MAX_RESPONSE_BYTES)
    try:
        receipt = json.loads(receipt_raw)
        if not isinstance(receipt, dict) or any(
            receipt.get(key) != expected for key, expected in (
                ("schema_version", RECEIPT_SCHEMA),
                ("status", "CAPTURE_ONLY_NOT_MODEL_ELIGIBLE"),
                ("market_requested", "h2h"),
                ("bookmakers_requested", list(FRENCH_BOOKMAKER_KEYS)),
                ("training_permitted", False),
                ("prediction_publication_permitted", False),
            )
        ):
            raise NHLThreeOutcomeInspectionError("Reçu NHL incompatible.")
        events_digest = sha256(events_raw).hexdigest()
        odds_digest = sha256(odds_raw).hexdigest()
        if (receipt.get("events_sha256") != events_digest
                or receipt.get("odds_sha256") != odds_digest):
            raise NHLThreeOutcomeInspectionError("Empreinte d'archive NHL divergente.")
        observed = _utc(receipt.get("observed_at_utc"), "observed_at_utc")
        events = {item.provider_event_id: item for item in parse_nhl_events(events_raw)}
        parsed = parse_nhl_h2h_odds(odds_raw)
        if (len(events) != receipt.get("event_count")
                or len(parsed.two_way_quotes) != receipt.get("two_way_quote_count")
                or len(parsed.rejected_market_keys) != receipt.get("rejected_market_count")
                or any(events.get(item.provider_event_id) != item for item in parsed.events)):
            raise NHLThreeOutcomeInspectionError("Reçu et événements NHL divergents.")
        counts = {key: {"two_team_outcomes": 0, "three_way": 0, "other": 0}
                  for key in FRENCH_BOOKMAKER_KEYS}
        shapes = {}
        for event_id, book_key, shape in parsed.observed_market_shapes:
            counts[book_key][shape] += 1
            shapes[(event_id, book_key)] = shape
        if receipt.get("market_shape_counts") != counts:
            raise NHLThreeOutcomeInspectionError("Comptage des marchés divergent.")

        candidates = []
        late = 0
        for row in _payload(odds_raw):
            event = events[row["id"]]
            for bookmaker in row["bookmakers"]:
                key = bookmaker["key"]
                if shapes.get((event.provider_event_id, key)) != "three_way":
                    continue
                market = bookmaker["markets"][0]
                prices = {}
                for outcome in market["outcomes"]:
                    name = _text(outcome.get("name"), "outcome.name")
                    prices[name] = _price(outcome.get("price"))
                third = (set(prices) - {event.away_team_name, event.home_team_name}).pop()
                market_at = _utc(market.get("last_update"), "market.last_update")
                if market_at > observed:
                    raise NHLThreeOutcomeInspectionError("Marché daté dans le futur.")
                if observed > event.start_utc - timedelta(hours=1):
                    late += 1
                    continue
                candidates.append(NHLThreeOutcomeCandidate(
                    event.provider_event_id, key, market_at,
                    prices[event.away_team_name], third, prices[third],
                    prices[event.home_team_name],
                ))
        return NHLThreeOutcomeInspection(
            observed, events_digest, odds_digest,
            tuple(sorted(candidates, key=lambda c: (c.provider_event_id, c.bookmaker_key))),
            late,
            tuple((key, counts[key]["two_team_outcomes"], counts[key]["three_way"],
                   counts[key]["other"]) for key in FRENCH_BOOKMAKER_KEYS),
        )
    except NHLThreeOutcomeInspectionError:
        raise
    except (KeyError, IndexError, AttributeError, TypeError, ValueError,
            NHLOddsCandidateError) as error:
        raise NHLThreeOutcomeInspectionError("Archive NHL incompatible.") from error
