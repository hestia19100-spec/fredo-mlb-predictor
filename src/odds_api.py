"""Client strict de The Odds API pour les cotes Moneyline MLB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from typing import Any

import requests


ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
ODDS_API_KEY_ENVIRONMENT_VARIABLE = "THE_ODDS_API_KEY"
ODDS_API_REGION = "eu"
ODDS_API_MARKET = "h2h"
REQUEST_TIMEOUT_SECONDS = 30


class OddsAPIError(RuntimeError):
    """Erreur contrôlée liée à la récupération des cotes."""


class OddsAPIRetryableError(OddsAPIError):
    """Erreur temporaire autorisant une nouvelle tentative ultérieure."""


class OddsAPIConfigurationError(OddsAPIError):
    """Configuration locale absente ou invalide."""


@dataclass(frozen=True, slots=True)
class MoneylineBookmaker:
    """Cotes décimales d’un bookmaker pour les deux équipes."""

    key: str
    title: str
    last_update_utc: datetime
    away_decimal_odds: Decimal
    home_decimal_odds: Decimal


@dataclass(frozen=True, slots=True)
class MoneylineEvent:
    """Événement MLB et ses cotes Moneyline disponibles."""

    provider_event_id: str
    commence_time_utc: datetime
    away_team_name: str
    home_team_name: str
    bookmakers: tuple[MoneylineBookmaker, ...]


@dataclass(frozen=True, slots=True)
class OddsFetchResult:
    """Réponse contrôlée sans jamais exposer la clé secrète."""

    provider: str
    sport_key: str
    region: str
    market: str
    odds_format: str
    response_received_at_utc: datetime
    response_body_sha256: str
    quota_remaining: int
    quota_used: int
    quota_last_cost: int
    raw_content: bytes
    events: tuple[MoneylineEvent, ...]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_api_key() -> str:
    value = os.environ.get(ODDS_API_KEY_ENVIRONMENT_VARIABLE)
    if value is None or not value:
        raise OddsAPIConfigurationError(
            "La clé THE_ODDS_API_KEY n’est pas configurée."
        )
    if value != value.strip() or any(character.isspace() for character in value):
        raise OddsAPIConfigurationError(
            "La clé THE_ODDS_API_KEY contient un caractère interdit."
        )
    if len(value) < 16 or len(value) > 256:
        raise OddsAPIConfigurationError(
            "La clé THE_ODDS_API_KEY a une longueur invalide."
        )
    return value


def odds_api_key_is_configured() -> bool:
    """Indique seulement si la clé locale est valide, sans jamais la renvoyer."""
    try:
        _read_api_key()
    except OddsAPIConfigurationError:
        return False
    return True


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OddsAPIError(f"Le champ {field_name} est absent ou invalide.")
    return value


def _utc_timestamp(value: object, field_name: str) -> datetime:
    text = _required_text(value, field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise OddsAPIError(f"Le champ {field_name} n’est pas un instant UTC.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OddsAPIError(f"Le champ {field_name} n’est pas un instant UTC.")
    if parsed.utcoffset().total_seconds() != 0:
        raise OddsAPIError(f"Le champ {field_name} n’est pas exprimé en UTC.")
    return parsed.astimezone(timezone.utc)


def _decimal_odds(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise OddsAPIError("Une cote Moneyline décimale est invalide.")
    try:
        decimal_value = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise OddsAPIError("Une cote Moneyline décimale est invalide.") from error
    if not decimal_value.is_finite() or decimal_value <= Decimal("1"):
        raise OddsAPIError("Une cote Moneyline décimale doit dépasser 1.")
    return decimal_value


def _parse_bookmaker(
    value: object,
    *,
    away_team_name: str,
    home_team_name: str,
) -> MoneylineBookmaker:
    if not isinstance(value, dict):
        raise OddsAPIError("Un bookmaker reçu est invalide.")
    key = _required_text(value.get("key"), "bookmaker.key")
    title = _required_text(value.get("title"), "bookmaker.title")
    last_update = _utc_timestamp(
        value.get("last_update"),
        "bookmaker.last_update",
    )
    markets = value.get("markets")
    if not isinstance(markets, list):
        raise OddsAPIError(
            f"La liste des marchés du bookmaker {key} est invalide."
        )
    markets_by_key: dict[str, dict[str, object]] = {}
    for market in markets:
        if not isinstance(market, dict):
            raise OddsAPIError(f"Un marché du bookmaker {key} est invalide.")
        market_key = _required_text(market.get("key"), "market.key")
        if market_key not in {ODDS_API_MARKET, "h2h_lay"}:
            raise OddsAPIError(
                f"Le bookmaker {key} contient le marché inattendu {market_key}."
            )
        if market_key in markets_by_key:
            raise OddsAPIError(
                f"Le bookmaker {key} répète le marché {market_key}."
            )
        markets_by_key[market_key] = market
    if ODDS_API_MARKET not in markets_by_key:
        raise OddsAPIError(
            f"Le bookmaker {key} ne contient pas le marché Moneyline h2h."
        )
    market = markets_by_key[ODDS_API_MARKET]
    outcomes = market.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 2:
        raise OddsAPIError(
            f"Le bookmaker {key} ne contient pas exactement deux cotes."
        )
    prices: dict[str, Decimal] = {}
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise OddsAPIError(f"Une cote du bookmaker {key} est invalide.")
        name = _required_text(outcome.get("name"), "outcome.name")
        if name in prices:
            raise OddsAPIError(f"Une équipe est répétée chez le bookmaker {key}.")
        prices[name] = _decimal_odds(outcome.get("price"))
    if set(prices) != {away_team_name, home_team_name}:
        raise OddsAPIError(
            f"Les équipes du bookmaker {key} ne correspondent pas au match."
        )
    return MoneylineBookmaker(
        key=key,
        title=title,
        last_update_utc=last_update,
        away_decimal_odds=prices[away_team_name],
        home_decimal_odds=prices[home_team_name],
    )


def _parse_event(value: object) -> MoneylineEvent:
    if not isinstance(value, dict):
        raise OddsAPIError("Un événement reçu est invalide.")
    if value.get("sport_key") != "baseball_mlb":
        raise OddsAPIError("Une réponse étrangère à la MLB a été reçue.")
    event_id = _required_text(value.get("id"), "event.id")
    away_team = _required_text(value.get("away_team"), "event.away_team")
    home_team = _required_text(value.get("home_team"), "event.home_team")
    if away_team == home_team:
        raise OddsAPIError("Les deux équipes d’un événement sont identiques.")
    commence_time = _utc_timestamp(
        value.get("commence_time"),
        "event.commence_time",
    )
    raw_bookmakers = value.get("bookmakers")
    if not isinstance(raw_bookmakers, list):
        raise OddsAPIError("La liste des bookmakers est invalide.")
    bookmakers = tuple(
        _parse_bookmaker(
            bookmaker,
            away_team_name=away_team,
            home_team_name=home_team,
        )
        for bookmaker in raw_bookmakers
    )
    bookmaker_keys = [bookmaker.key for bookmaker in bookmakers]
    if len(bookmaker_keys) != len(set(bookmaker_keys)):
        raise OddsAPIError("Un bookmaker est répété pour le même événement.")
    return MoneylineEvent(
        provider_event_id=event_id,
        commence_time_utc=commence_time,
        away_team_name=away_team,
        home_team_name=home_team,
        bookmakers=bookmakers,
    )


def _quota_header(headers: object, name: str) -> int:
    if not hasattr(headers, "get"):
        raise OddsAPIError("Les informations de quota sont absentes.")
    value = headers.get(name)
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise OddsAPIError(f"L’en-tête {name} est absent ou invalide.")
    return int(value)


def _parse_payload(raw_content: bytes) -> tuple[MoneylineEvent, ...]:
    try:
        payload = json.loads(raw_content, parse_float=Decimal)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise OddsAPIError("La réponse de cotes n’est pas un JSON valide.") from error
    if not isinstance(payload, list):
        raise OddsAPIError("La réponse de cotes n’est pas une liste d’événements.")
    events = tuple(_parse_event(event) for event in payload)
    event_ids = [event.provider_event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise OddsAPIError("Un événement The Odds API est répété.")
    return events


def fetch_mlb_moneyline_odds() -> OddsFetchResult:
    """Récupère une fois les cotes MLB européennes au format décimal."""
    api_key = _read_api_key()
    public_parameters = {
        "regions": ODDS_API_REGION,
        "markets": ODDS_API_MARKET,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    request_parameters = {"apiKey": api_key, **public_parameters}
    try:
        response = requests.get(
            ODDS_API_URL,
            params=request_parameters,
            headers={"User-Agent": "fredo-mlb-predictor/0.1"},
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
        received_at = _utc_now().astimezone(timezone.utc)
        response.raise_for_status()
    except (requests.Timeout, requests.ConnectionError) as error:
        raise OddsAPIRetryableError(
            "Impossible de joindre The Odds API."
        ) from None
    except requests.HTTPError as error:
        status_code = (
            error.response.status_code
            if error.response is not None
            else None
        )
        message = (
            "The Odds API a répondu avec une erreur HTTP."
            if status_code is None
            else f"The Odds API a répondu avec une erreur HTTP {status_code}."
        )
        if status_code in {408, 429} or (
            status_code is not None and 500 <= status_code <= 599
        ):
            raise OddsAPIRetryableError(message) from None
        raise OddsAPIError(message) from None
    except requests.RequestException as error:
        raise OddsAPIError("La requête de cotes est invalide.") from None

    if getattr(response, "status_code", None) != 200:
        raise OddsAPIError("The Odds API n’a pas confirmé une réponse HTTP 200.")
    if getattr(response, "history", None):
        raise OddsAPIError("Une redirection inattendue a été refusée.")
    raw_content = response.content
    if not isinstance(raw_content, bytes) or not raw_content:
        raise OddsAPIError("The Odds API a renvoyé une réponse vide.")
    events = _parse_payload(raw_content)
    quota_remaining = _quota_header(
        response.headers,
        "x-requests-remaining",
    )
    quota_used = _quota_header(response.headers, "x-requests-used")
    quota_last_cost = _quota_header(response.headers, "x-requests-last")
    if quota_last_cost not in {0, 1}:
        raise OddsAPIError(
            "La collecte a consommé un nombre inattendu de crédits API."
        )
    return OddsFetchResult(
        provider="the_odds_api_v4",
        sport_key="baseball_mlb",
        region=public_parameters["regions"],
        market=public_parameters["markets"],
        odds_format=public_parameters["oddsFormat"],
        response_received_at_utc=received_at,
        response_body_sha256=hashlib.sha256(raw_content).hexdigest(),
        quota_remaining=quota_remaining,
        quota_used=quota_used,
        quota_last_cost=quota_last_cost,
        raw_content=raw_content,
        events=events,
    )


__all__ = [
    "MoneylineBookmaker",
    "MoneylineEvent",
    "OddsAPIConfigurationError",
    "OddsAPIError",
    "OddsAPIRetryableError",
    "OddsFetchResult",
    "fetch_mlb_moneyline_odds",
    "odds_api_key_is_configured",
]
