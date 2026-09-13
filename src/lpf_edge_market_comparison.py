"""Comparaison descriptive entre prédictions certifiées et marché français."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Mapping

from src.lpf_edge_dashboard import CertifiedPredictionDay
from src.lpf_edge_odds_display import (
    MoneylineBookmakerDisplayQuote,
    MoneylineOddsDisplay,
)


HUNDRED = Decimal("100")


class LPFEdgeMarketComparisonError(RuntimeError):
    """Les prédictions et les cotes ne peuvent pas être comparées sûrement."""


@dataclass(frozen=True, slots=True)
class MarketComparisonRow:
    """Comparaison du côté donné devant par LPF Edge pour un match."""

    game_id: int
    scheduled_start_utc: datetime
    home_team_name: str
    away_team_name: str
    predicted_side: str
    predicted_team_name: str
    model_probability: Decimal
    french_market_probability: Decimal | None
    gap_percentage_points: Decimal | None
    best_decimal_odds: Decimal | None
    best_bookmakers: tuple[str, ...]
    bookmaker_count: int

    @property
    def has_market_comparison(self) -> bool:
        return self.french_market_probability is not None


@dataclass(frozen=True, slots=True)
class MarketComparison:
    """Tableau complet fondé sur une collecte française et un lot certifié."""

    target_date: date
    odds_run_id: int | None
    odds_completed_at_utc: datetime | None
    rows: tuple[MarketComparisonRow, ...]

    @property
    def comparable_count(self) -> int:
        return sum(row.has_market_comparison for row in self.rows)

    @property
    def missing_count(self) -> int:
        return len(self.rows) - self.comparable_count


def _team_name(team_id: int, team_names: Mapping[int, str]) -> str:
    value = team_names.get(team_id)
    if not isinstance(value, str) or not value.strip():
        raise LPFEdgeMarketComparisonError(
            f"Le nom de l’équipe MLB {team_id} est absent."
        )
    return value.strip()


def _fair_probability(
    quote: MoneylineBookmakerDisplayQuote,
    predicted_side: str,
) -> Decimal:
    """Retire proportionnellement la marge d’une paire du même bookmaker."""
    if any(
        not price.is_finite() or price <= Decimal("1")
        for price in (quote.home_decimal_odds, quote.away_decimal_odds)
    ):
        raise LPFEdgeMarketComparisonError(
            "Une paire de cotes françaises est invalide."
        )
    implied_home = Decimal("1") / quote.home_decimal_odds
    implied_away = Decimal("1") / quote.away_decimal_odds
    implied_total = implied_home + implied_away
    if not implied_total.is_finite() or implied_total <= Decimal("0"):
        raise LPFEdgeMarketComparisonError(
            "Une paire de probabilités implicites est invalide."
        )
    if predicted_side == "HOME":
        return implied_home / implied_total
    if predicted_side == "AWAY":
        return implied_away / implied_total
    raise LPFEdgeMarketComparisonError(
        "Le côté prédit par LPF Edge est invalide."
    )


def _average_fair_probability(
    quotes: tuple[MoneylineBookmakerDisplayQuote, ...],
    predicted_side: str,
) -> Decimal:
    if not quotes:
        raise LPFEdgeMarketComparisonError(
            "Une moyenne de marché exige au moins un bookmaker."
        )
    values = tuple(_fair_probability(quote, predicted_side) for quote in quotes)
    return sum(values, Decimal("0")) / Decimal(len(values))


def _validated_best_price(
    quotes: tuple[MoneylineBookmakerDisplayQuote, ...],
    predicted_side: str,
) -> tuple[Decimal, tuple[str, ...]]:
    bookmaker_keys: set[str] = set()
    prices: list[tuple[Decimal, str]] = []
    for quote in quotes:
        if not quote.key.strip() or not quote.title.strip():
            raise LPFEdgeMarketComparisonError(
                "Un bookmaker français est incomplet."
            )
        if quote.key in bookmaker_keys:
            raise LPFEdgeMarketComparisonError(
                "Un bookmaker français est répété pour le même match."
            )
        bookmaker_keys.add(quote.key)
        price = (
            quote.home_decimal_odds
            if predicted_side == "HOME"
            else quote.away_decimal_odds
        )
        if not price.is_finite() or price <= Decimal("1"):
            raise LPFEdgeMarketComparisonError(
                "Une cote française est invalide."
            )
        prices.append((price, quote.title.strip()))

    best_price = max(price for price, _ in prices)
    best_bookmakers = tuple(
        sorted({title for price, title in prices if price == best_price})
    )
    return best_price, best_bookmakers


def build_french_market_comparison(
    prediction_day: CertifiedPredictionDay,
    odds_display: MoneylineOddsDisplay,
    *,
    team_names: Mapping[int, str],
) -> MarketComparison:
    """Compare sans recalcul les prédictions aux cotes françaises archivées."""
    if prediction_day.target_date != odds_display.target_date:
        raise LPFEdgeMarketComparisonError(
            "Les prédictions et les cotes visent deux journées différentes."
        )
    if odds_display.region not in {None, "fr"}:
        raise LPFEdgeMarketComparisonError(
            "Seule une collecte française peut alimenter cette comparaison."
        )
    if odds_display.region is None:
        if (
            odds_display.run_id is not None
            or odds_display.completed_at_utc is not None
            or odds_display.quote_count != 0
            or any(game.bookmaker_quotes for game in odds_display.games)
        ):
            raise LPFEdgeMarketComparisonError(
                "Des cotes existent sans collecte française certifiée."
            )
    elif odds_display.run_id is None or odds_display.completed_at_utc is None:
        raise LPFEdgeMarketComparisonError(
            "La collecte française est incomplète."
        )
    elif odds_display.quote_count != sum(
        len(game.bookmaker_quotes) for game in odds_display.games
    ):
        raise LPFEdgeMarketComparisonError(
            "Le nombre total de cotes françaises est incohérent."
        )

    odds_games_by_id = {game.game_id: game for game in odds_display.games}
    if len(odds_games_by_id) != len(odds_display.games):
        raise LPFEdgeMarketComparisonError(
            "Un match est répété dans les cotes affichées."
        )

    rows: list[MarketComparisonRow] = []
    prediction_game_ids: set[int] = set()
    for prediction in prediction_day.predictions:
        if prediction.game_id in prediction_game_ids:
            raise LPFEdgeMarketComparisonError(
                "Un match est répété dans les prédictions certifiées."
            )
        prediction_game_ids.add(prediction.game_id)
        home_name = _team_name(prediction.home_team_id, team_names)
        away_name = _team_name(prediction.away_team_id, team_names)
        predicted_side = prediction.predicted_side
        if predicted_side not in {"HOME", "AWAY"}:
            raise LPFEdgeMarketComparisonError(
                "Le côté prédit par LPF Edge est invalide."
            )
        predicted_name = home_name if predicted_side == "HOME" else away_name
        odds_game = odds_games_by_id.get(prediction.game_id)

        if odds_game is None or not odds_game.bookmaker_quotes:
            market_probability = None
            gap = None
            best_odds = None
            best_bookmakers: tuple[str, ...] = ()
            bookmaker_count = 0
        else:
            if (
                odds_game.home_team_name != home_name
                or odds_game.away_team_name != away_name
            ):
                raise LPFEdgeMarketComparisonError(
                    "Les équipes d’un match divergent entre prédictions et cotes."
                )
            if odds_game.bookmaker_count != len(odds_game.bookmaker_quotes):
                raise LPFEdgeMarketComparisonError(
                    "Le nombre de bookmakers du match est incohérent."
                )
            market_probability = _average_fair_probability(
                odds_game.bookmaker_quotes,
                predicted_side,
            )
            gap = (
                prediction.predicted_probability - market_probability
            ) * HUNDRED
            best_odds, best_bookmakers = _validated_best_price(
                odds_game.bookmaker_quotes,
                predicted_side,
            )
            cached_best_odds = (
                odds_game.home_best_decimal_odds
                if predicted_side == "HOME"
                else odds_game.away_best_decimal_odds
            )
            cached_best_bookmakers = (
                odds_game.home_best_bookmakers
                if predicted_side == "HOME"
                else odds_game.away_best_bookmakers
            )
            if (
                cached_best_odds != best_odds
                or cached_best_bookmakers != best_bookmakers
            ):
                raise LPFEdgeMarketComparisonError(
                    "La meilleure cote française affichée est incohérente."
                )
            bookmaker_count = len(odds_game.bookmaker_quotes)

        rows.append(
            MarketComparisonRow(
                game_id=prediction.game_id,
                scheduled_start_utc=prediction.scheduled_start_utc,
                home_team_name=home_name,
                away_team_name=away_name,
                predicted_side=predicted_side,
                predicted_team_name=predicted_name,
                model_probability=prediction.predicted_probability,
                french_market_probability=market_probability,
                gap_percentage_points=gap,
                best_decimal_odds=best_odds,
                best_bookmakers=best_bookmakers,
                bookmaker_count=bookmaker_count,
            )
        )

    return MarketComparison(
        target_date=prediction_day.target_date,
        odds_run_id=odds_display.run_id,
        odds_completed_at_utc=odds_display.completed_at_utc,
        rows=tuple(rows),
    )


__all__ = [
    "LPFEdgeMarketComparisonError",
    "MarketComparison",
    "MarketComparisonRow",
    "build_french_market_comparison",
]
