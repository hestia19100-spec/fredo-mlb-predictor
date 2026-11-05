"""Rapproche hors réseau les cotes MLB françaises et les matchs officiels 2027.

Ce module ne collecte rien, ne sélectionne aucun pari et ne publie rien. Une
correspondance incertaine est signalée, jamais utilisée comme une cote valide.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib

from src.mlb_2027_observed_source import ObservedSourceBundle
from src.odds_api import OddsFetchResult, OddsAPIError, _parse_payload


MAX_START_DIFFERENCE = timedelta(minutes=20)
MAX_QUOTE_AGE = timedelta(hours=2)


class MLB2027MarketMatchError(ValueError):
    """La réponse de marché ne peut pas être reliée sans risque à la source MLB."""


@dataclass(frozen=True, slots=True)
class MatchedMarketGame:
    game_id: int
    status: str
    provider_event_id: str | None
    home_best_odds: Decimal | None
    home_best_bookmakers: tuple[str, ...]
    away_best_odds: Decimal | None
    away_best_bookmakers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MatchedMarketDay:
    target_date: str
    source_receipt_sha256: str
    odds_response_sha256: str
    odds_received_at_utc: datetime
    quota_remaining: int
    quota_used: int
    quota_last_cost: int
    games: tuple[MatchedMarketGame, ...]


def _utc(value: datetime) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() == timedelta(0)


def _empty(game_id: int, status: str) -> MatchedMarketGame:
    return MatchedMarketGame(game_id, status, None, None, (), None, ())


def match_mlb_2027_market(
    source: ObservedSourceBundle,
    odds: OddsFetchResult,
) -> MatchedMarketDay:
    """Associe une seule réponse h2h/fr aux matchs avant coup d'envoi.

    Les noms d'équipes et heures viennent des observations MLB déjà scellées.
    Une double affiche ou un événement ambigu reste sans cote exploitable.
    """
    if type(source) is not ObservedSourceBundle or type(odds) is not OddsFetchResult:
        raise MLB2027MarketMatchError("Source MLB ou réponse de cotes absente.")
    if (odds.provider, odds.sport_key, odds.region, odds.market, odds.odds_format) != (
        "the_odds_api_v4", "baseball_mlb", "fr", "h2h", "decimal",
    ):
        raise MLB2027MarketMatchError("Marché MLB français à deux issues non confirmé.")
    if not _utc(odds.response_received_at_utc):
        raise MLB2027MarketMatchError("Heure de réception des cotes invalide.")
    if hashlib.sha256(odds.raw_content).hexdigest() != odds.response_body_sha256:
        raise MLB2027MarketMatchError("Réponse brute de cotes altérée.")
    try:
        if _parse_payload(odds.raw_content) != odds.events:
            raise MLB2027MarketMatchError("Cotes décodées différentes des octets reçus.")
    except OddsAPIError as error:
        raise MLB2027MarketMatchError("Réponse brute de cotes invalide.") from error
    if any(type(value) is not int or value < 0 for value in (
        odds.quota_remaining, odds.quota_used, odds.quota_last_cost,
    )) or odds.quota_last_cost > 1:
        raise MLB2027MarketMatchError("Coût ou quota de la collecte inattendu.")

    scheduled = {}
    for observation in source.observations:
        for game in observation.games:
            if game.official_date == source.target_date.isoformat():
                if game.game_id in scheduled:
                    raise MLB2027MarketMatchError("Match MLB dupliqué dans la source.")
                scheduled[game.game_id] = game
    upcoming = {game.game_id: game for game in source.upcoming_games}
    if not upcoming or set(scheduled) != set(upcoming):
        raise MLB2027MarketMatchError("Noms MLB et calendrier du jour incohérents.")
    if odds.response_received_at_utc >= min(game.scheduled_start_utc for game in upcoming.values()):
        raise MLB2027MarketMatchError("Cotes reçues après le premier coup d'envoi.")

    candidates: dict[int, list] = {game_id: [] for game_id in upcoming}
    event_hits: dict[str, list[int]] = {}
    for event in odds.events:
        if not _utc(event.commence_time_utc):
            raise MLB2027MarketMatchError("Heure d'événement du bookmaker invalide.")
        hits = []
        for game_id, game in upcoming.items():
            named = scheduled[game_id]
            if (
                event.away_team_name == named.away_team_name
                and event.home_team_name == named.home_team_name
                and abs(event.commence_time_utc - game.scheduled_start_utc) <= MAX_START_DIFFERENCE
            ):
                hits.append(game_id)
                candidates[game_id].append(event)
        event_hits[event.provider_event_id] = hits

    rows = []
    for game_id in sorted(upcoming):
        events = candidates[game_id]
        if not events:
            rows.append(_empty(game_id, "NO_MATCHING_EVENT"))
            continue
        if len(events) != 1 or len(event_hits[events[0].provider_event_id]) != 1:
            rows.append(_empty(game_id, "AMBIGUOUS_EVENT"))
            continue
        event = events[0]
        valid_quotes = []
        for quote in event.bookmakers:
            if not _utc(quote.last_update_utc) or quote.last_update_utc > odds.response_received_at_utc:
                raise MLB2027MarketMatchError("Horodatage de bookmaker futur ou invalide.")
            if odds.response_received_at_utc - quote.last_update_utc <= MAX_QUOTE_AGE:
                valid_quotes.append(quote)
        if not valid_quotes:
            rows.append(_empty(game_id, "NO_FRESH_TWO_WAY_QUOTES"))
            continue
        home = max(quote.home_decimal_odds for quote in valid_quotes)
        away = max(quote.away_decimal_odds for quote in valid_quotes)
        rows.append(MatchedMarketGame(
            game_id=game_id, status="MATCHED", provider_event_id=event.provider_event_id,
            home_best_odds=home,
            home_best_bookmakers=tuple(sorted(quote.title for quote in valid_quotes if quote.home_decimal_odds == home)),
            away_best_odds=away,
            away_best_bookmakers=tuple(sorted(quote.title for quote in valid_quotes if quote.away_decimal_odds == away)),
        ))
    return MatchedMarketDay(
        target_date=source.target_date.isoformat(),
        source_receipt_sha256=source.receipt_sha256,
        odds_response_sha256=odds.response_body_sha256,
        odds_received_at_utc=odds.response_received_at_utc,
        quota_remaining=odds.quota_remaining,
        quota_used=odds.quota_used,
        quota_last_cost=odds.quota_last_cost,
        games=tuple(rows),
    )
