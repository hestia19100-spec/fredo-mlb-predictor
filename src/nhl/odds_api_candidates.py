"""Read-only NHL candidates from The Odds API; never a prediction.

The provider event identifier is deliberately not treated as an NHL game ID.
Likewise, two h2h outcomes are necessary but not sufficient proof of the
bookmaker's settlement rules. The original payload must be archived separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json


SPORT_KEY = "icehockey_nhl"
# Only this bookmaker's two-way NHL h2h settlement has been checked against
# its published ice-hockey rules. Other books currently expose three-way h2h.
FRENCH_BOOKMAKERS = frozenset({"netbet_fr"})
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class NHLOddsCandidateError(ValueError):
    """A provider response cannot safely be interpreted as NHL candidates."""


@dataclass(frozen=True, slots=True)
class NHLEventCandidate:
    provider_event_id: str
    start_utc: datetime
    away_team_name: str
    home_team_name: str


@dataclass(frozen=True, slots=True)
class NHLTwoWayQuote:
    provider_event_id: str
    bookmaker_key: str
    market_observed_at_utc: datetime
    away_decimal_odds: Decimal
    home_decimal_odds: Decimal


@dataclass(frozen=True, slots=True)
class NHLOddsCandidates:
    events: tuple[NHLEventCandidate, ...]
    two_way_quotes: tuple[NHLTwoWayQuote, ...]
    rejected_market_keys: tuple[tuple[str, str], ...]


def _payload(raw: bytes) -> list[object]:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_RESPONSE_BYTES:
        raise NHLOddsCandidateError("Réponse NHL absente ou trop volumineuse.")
    try:
        document = json.loads(raw, parse_float=Decimal)
    except (ValueError, UnicodeDecodeError) as error:
        raise NHLOddsCandidateError("Réponse NHL non JSON.") from error
    if not isinstance(document, list):
        raise NHLOddsCandidateError("Liste d'événements NHL attendue.")
    return document


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise NHLOddsCandidateError(f"Champ {field} invalide.")
    return value


def _utc(value: object, field: str) -> datetime:
    text = _text(value, field)
    if not text.endswith("Z"):
        raise NHLOddsCandidateError(f"Champ {field} non UTC canonique.")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise NHLOddsCandidateError(f"Champ {field} invalide.") from error
    if parsed.utcoffset() != timedelta(0):
        raise NHLOddsCandidateError(f"Champ {field} non UTC.")
    return parsed.astimezone(timezone.utc)


def _event(value: object) -> NHLEventCandidate:
    if not isinstance(value, dict) or value.get("sport_key") != SPORT_KEY:
        raise NHLOddsCandidateError("Événement étranger à la NHL.")
    event_id = _text(value.get("id"), "id")
    if len(event_id) != 32 or any(c not in "0123456789abcdef" for c in event_id):
        raise NHLOddsCandidateError("Identifiant du fournisseur invalide.")
    away = _text(value.get("away_team"), "away_team")
    home = _text(value.get("home_team"), "home_team")
    if away == home:
        raise NHLOddsCandidateError("Équipes identiques.")
    return NHLEventCandidate(
        event_id, _utc(value.get("commence_time"), "commence_time"), away, home,
    )


def parse_nhl_events(raw: bytes) -> tuple[NHLEventCandidate, ...]:
    """Parse upcoming-event metadata without claiming schedule completeness."""
    events = tuple(_event(row) for row in _payload(raw))
    ids = [event.provider_event_id for event in events]
    if len(ids) != len(set(ids)):
        raise NHLOddsCandidateError("Événement fournisseur dupliqué.")
    return tuple(sorted(events, key=lambda item: (item.start_utc, item.provider_event_id)))


def _price(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise NHLOddsCandidateError("Cote décimale invalide.")
    try:
        price = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise NHLOddsCandidateError("Cote décimale invalide.") from error
    if not price.is_finite() or price <= 1:
        raise NHLOddsCandidateError("Cote décimale hors limites.")
    return price


def parse_nhl_h2h_odds(raw: bytes) -> NHLOddsCandidates:
    """Accept only two named outcomes from listed French bookmakers.

    A three-way or otherwise ambiguous bookmaker market is rejected in place,
    while its event remains visible as a fixture candidate. No quote here is
    eligible for publication until settlement semantics are independently checked.
    """
    document = _payload(raw)
    events: list[NHLEventCandidate] = []
    quotes: list[NHLTwoWayQuote] = []
    rejected: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for row in document:
        event = _event(row)
        if event.provider_event_id in seen_ids:
            raise NHLOddsCandidateError("Événement fournisseur dupliqué.")
        seen_ids.add(event.provider_event_id)
        events.append(event)
        assert isinstance(row, dict)
        bookmakers = row.get("bookmakers")
        if not isinstance(bookmakers, list):
            raise NHLOddsCandidateError("Bookmakers absents.")
        seen_books: set[str] = set()
        for bookmaker in bookmakers:
            if not isinstance(bookmaker, dict):
                raise NHLOddsCandidateError("Bookmaker invalide.")
            book_key = _text(bookmaker.get("key"), "bookmaker.key")
            if book_key in seen_books:
                raise NHLOddsCandidateError("Bookmaker dupliqué.")
            seen_books.add(book_key)
            if book_key not in FRENCH_BOOKMAKERS:
                rejected.append((event.provider_event_id, book_key))
                continue
            markets = bookmaker.get("markets")
            if not isinstance(markets, list) or len(markets) != 1:
                rejected.append((event.provider_event_id, book_key))
                continue
            market = markets[0]
            if not isinstance(market, dict) or market.get("key") != "h2h":
                rejected.append((event.provider_event_id, book_key))
                continue
            outcomes = market.get("outcomes")
            if not isinstance(outcomes, list) or len(outcomes) != 2:
                rejected.append((event.provider_event_id, book_key))
                continue
            names: dict[str, Decimal] = {}
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    raise NHLOddsCandidateError("Issue de marché invalide.")
                name = _text(outcome.get("name"), "outcome.name")
                if name in names:
                    raise NHLOddsCandidateError("Issue de marché dupliquée.")
                names[name] = _price(outcome.get("price"))
            if set(names) != {event.away_team_name, event.home_team_name}:
                rejected.append((event.provider_event_id, book_key))
                continue
            observed = _utc(market.get("last_update"), "market.last_update")
            quotes.append(NHLTwoWayQuote(
                event.provider_event_id, book_key, observed,
                names[event.away_team_name], names[event.home_team_name],
            ))
    return NHLOddsCandidates(
        tuple(sorted(events, key=lambda item: (item.start_utc, item.provider_event_id))),
        tuple(sorted(quotes, key=lambda item: (item.provider_event_id, item.bookmaker_key))),
        tuple(sorted(rejected)),
    )
