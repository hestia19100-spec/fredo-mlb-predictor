"""Tests du tableau LPF Edge comparant modèle et marché français."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
import unittest

from src.lpf_edge_dashboard import CertifiedPrediction, CertifiedPredictionDay
from src.lpf_edge_market_comparison import (
    LPFEdgeMarketComparisonError,
    build_french_market_comparison,
)
from src.lpf_edge_odds_display import (
    MoneylineBookmakerDisplayQuote,
    MoneylineOddsDisplay,
    MoneylineOddsDisplayGame,
)


UTC = timezone.utc
TARGET = date(2026, 9, 13)
START = datetime(2026, 9, 13, 18, 10, tzinfo=UTC)
TEAM_NAMES = {10: "Équipe extérieure", 20: "Équipe domicile"}


def prediction_day(
    *,
    p_home: Decimal = Decimal("0.60"),
    p_away: Decimal = Decimal("0.40"),
    target: date = TARGET,
) -> CertifiedPredictionDay:
    prediction = CertifiedPrediction(
        prediction_id="prediction-1",
        batch_id="a" * 64,
        game_id=100,
        official_date=target,
        away_team_id=10,
        home_team_id=20,
        scheduled_start_utc=START,
        issued_at_utc=datetime(2026, 9, 13, 12, 0, tzinfo=UTC),
        p_home_win=p_home,
        p_away_win=p_away,
    )
    return CertifiedPredictionDay(
        target_date=target,
        batch_id="a" * 64,
        results_commit="b" * 40,
        certified_at_utc=datetime(2026, 9, 13, 12, 1, tzinfo=UTC),
        remote_lead_minutes=Decimal("300"),
        predictions_sha256="c" * 64,
        receipt_sha256="d" * 64,
        predictions=(prediction,),
    )


def quote(
    key: str,
    title: str,
    *,
    home: str,
    away: str,
) -> MoneylineBookmakerDisplayQuote:
    return MoneylineBookmakerDisplayQuote(
        key=key,
        title=title,
        last_update_utc=datetime(2026, 9, 13, 11, 59, tzinfo=UTC),
        home_decimal_odds=Decimal(home),
        away_decimal_odds=Decimal(away),
    )


def odds_display(
    *quotes: MoneylineBookmakerDisplayQuote,
    region: str | None = "fr",
    target: date = TARGET,
    completed_at: datetime = datetime(2026, 9, 13, 12, 0, tzinfo=UTC),
) -> MoneylineOddsDisplay:
    home_best = max(
        (item.home_decimal_odds for item in quotes),
        default=None,
    )
    away_best = max(
        (item.away_decimal_odds for item in quotes),
        default=None,
    )
    game = MoneylineOddsDisplayGame(
        game_id=100,
        scheduled_start_utc=START,
        home_team_name="Équipe domicile",
        away_team_name="Équipe extérieure",
        bookmaker_count=len(quotes),
        home_best_decimal_odds=home_best,
        home_best_bookmakers=tuple(
            item.title
            for item in quotes
            if item.home_decimal_odds == home_best
        ),
        away_best_decimal_odds=away_best,
        away_best_bookmakers=tuple(
            item.title
            for item in quotes
            if item.away_decimal_odds == away_best
        ),
        bookmaker_titles=tuple(item.title for item in quotes),
        latest_bookmaker_update_utc=(
            max((item.last_update_utc for item in quotes), default=None)
        ),
        bookmaker_quotes=tuple(quotes),
    )
    return MoneylineOddsDisplay(
        target_date=target,
        run_id=4 if region is not None else None,
        region=region,
        completed_at_utc=(
            completed_at
            if region is not None
            else None
        ),
        quote_count=len(quotes),
        games=(game,),
    )


class LPFEdgeMarketComparisonTests(unittest.TestCase):
    def test_home_favorite_uses_average_devigged_french_market(self) -> None:
        first = quote("betclic_fr", "Betclic", home="1.80", away="2.10")
        second = quote("pmu_fr", "PMU", home="1.90", away="2.00")

        comparison = build_french_market_comparison(
            prediction_day(),
            odds_display(first, second),
            team_names=TEAM_NAMES,
        )

        row = comparison.rows[0]
        expected_market = (
            (Decimal("1") / Decimal("1.80"))
            / (
                Decimal("1") / Decimal("1.80")
                + Decimal("1") / Decimal("2.10")
            )
            + (Decimal("1") / Decimal("1.90"))
            / (
                Decimal("1") / Decimal("1.90")
                + Decimal("1") / Decimal("2.00")
            )
        ) / Decimal("2")
        self.assertEqual(row.home_team_name, "Équipe domicile")
        self.assertEqual(row.away_team_name, "Équipe extérieure")
        self.assertEqual(row.predicted_side, "HOME")
        self.assertEqual(row.predicted_team_name, "Équipe domicile")
        self.assertEqual(row.model_probability, Decimal("0.60"))
        self.assertEqual(row.french_market_probability, expected_market)
        self.assertEqual(
            row.gap_percentage_points,
            (Decimal("0.60") - expected_market) * Decimal("100"),
        )
        self.assertEqual(row.best_decimal_odds, Decimal("1.90"))
        self.assertEqual(row.best_bookmakers, ("PMU",))
        self.assertEqual(row.bookmaker_count, 2)
        self.assertEqual(comparison.comparable_count, 1)
        self.assertEqual(comparison.missing_count, 0)

    def test_away_favorite_uses_away_side_and_tied_best_books(self) -> None:
        first = quote("netbet_fr", "NetBet", home="2.20", away="1.75")
        second = quote("winamax_fr", "Winamax", home="2.15", away="1.75")

        comparison = build_french_market_comparison(
            prediction_day(p_home=Decimal("0.45"), p_away=Decimal("0.55")),
            odds_display(first, second),
            team_names=TEAM_NAMES,
        )

        row = comparison.rows[0]
        self.assertEqual(row.predicted_side, "AWAY")
        self.assertEqual(row.predicted_team_name, "Équipe extérieure")
        self.assertEqual(row.best_decimal_odds, Decimal("1.75"))
        self.assertEqual(row.best_bookmakers, ("NetBet", "Winamax"))

    def test_missing_french_quotes_keeps_prediction_without_comparison(self) -> None:
        comparison = build_french_market_comparison(
            prediction_day(),
            odds_display(),
            team_names=TEAM_NAMES,
        )

        row = comparison.rows[0]
        self.assertFalse(row.has_market_comparison)
        self.assertIsNone(row.french_market_probability)
        self.assertIsNone(row.gap_percentage_points)
        self.assertIsNone(row.best_decimal_odds)
        self.assertEqual(comparison.comparable_count, 0)
        self.assertEqual(comparison.missing_count, 1)

    def test_european_collection_is_never_presented_as_french(self) -> None:
        with self.assertRaisesRegex(
            LPFEdgeMarketComparisonError,
            "Seule une collecte française",
        ):
            build_french_market_comparison(
                prediction_day(),
                odds_display(
                    quote("legacy", "Ancien UE", home="1.80", away="2.10"),
                    region="eu",
                ),
                team_names=TEAM_NAMES,
            )

    def test_quotes_without_a_french_run_are_rejected(self) -> None:
        display = odds_display(region=None)
        quote_without_run = quote(
            "pmu_fr", "PMU", home="1.80", away="2.10"
        )
        game = display.games[0]
        display = MoneylineOddsDisplay(
            target_date=display.target_date,
            run_id=None,
            region=None,
            completed_at_utc=None,
            quote_count=1,
            games=(
                MoneylineOddsDisplayGame(
                    game_id=game.game_id,
                    scheduled_start_utc=game.scheduled_start_utc,
                    home_team_name=game.home_team_name,
                    away_team_name=game.away_team_name,
                    bookmaker_count=1,
                    home_best_decimal_odds=Decimal("1.80"),
                    home_best_bookmakers=("PMU",),
                    away_best_decimal_odds=Decimal("2.10"),
                    away_best_bookmakers=("PMU",),
                    bookmaker_titles=("PMU",),
                    latest_bookmaker_update_utc=quote_without_run.last_update_utc,
                    bookmaker_quotes=(quote_without_run,),
                ),
            ),
        )

        with self.assertRaisesRegex(
            LPFEdgeMarketComparisonError,
            "sans collecte française certifiée",
        ):
            build_french_market_comparison(
                prediction_day(),
                display,
                team_names=TEAM_NAMES,
            )

    def test_inconsistent_cached_best_price_is_rejected(self) -> None:
        display = odds_display(
            quote("pmu_fr", "PMU", home="1.80", away="2.10")
        )
        game = display.games[0]
        display = MoneylineOddsDisplay(
            target_date=display.target_date,
            run_id=display.run_id,
            region=display.region,
            completed_at_utc=display.completed_at_utc,
            quote_count=display.quote_count,
            games=(
                MoneylineOddsDisplayGame(
                    game_id=game.game_id,
                    scheduled_start_utc=game.scheduled_start_utc,
                    home_team_name=game.home_team_name,
                    away_team_name=game.away_team_name,
                    bookmaker_count=game.bookmaker_count,
                    home_best_decimal_odds=Decimal("1.90"),
                    home_best_bookmakers=game.home_best_bookmakers,
                    away_best_decimal_odds=game.away_best_decimal_odds,
                    away_best_bookmakers=game.away_best_bookmakers,
                    bookmaker_titles=game.bookmaker_titles,
                    latest_bookmaker_update_utc=game.latest_bookmaker_update_utc,
                    bookmaker_quotes=game.bookmaker_quotes,
                ),
            ),
        )

        with self.assertRaisesRegex(
            LPFEdgeMarketComparisonError,
            "meilleure cote française affichée est incohérente",
        ):
            build_french_market_comparison(
                prediction_day(),
                display,
                team_names=TEAM_NAMES,
            )

    def test_different_dates_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            LPFEdgeMarketComparisonError,
            "deux journées différentes",
        ):
            build_french_market_comparison(
                prediction_day(),
                odds_display(target=date(2026, 9, 14)),
                team_names=TEAM_NAMES,
            )

    def test_collection_completed_after_certification_is_rejected(self) -> None:
        display = odds_display(
            quote("pmu_fr", "PMU", home="1.80", away="2.10"),
            completed_at=datetime(2026, 9, 13, 13, 1, tzinfo=UTC),
        )

        with self.assertRaisesRegex(
            LPFEdgeMarketComparisonError,
            "postérieure à la certification",
        ):
            build_french_market_comparison(
                prediction_day(),
                display,
                team_names=TEAM_NAMES,
            )

    def test_collection_without_timezone_is_rejected(self) -> None:
        display = odds_display(
            quote("pmu_fr", "PMU", home="1.80", away="2.10"),
            completed_at=datetime(2026, 9, 13, 12, 0),
        )

        with self.assertRaisesRegex(
            LPFEdgeMarketComparisonError,
            "collecte française n’est pas horodatée",
        ):
            build_french_market_comparison(
                prediction_day(),
                display,
                team_names=TEAM_NAMES,
            )

    def test_team_name_disagreement_is_rejected(self) -> None:
        display = odds_display(
            quote("pmu_fr", "PMU", home="1.80", away="2.10")
        )
        different_names = {10: "Autre extérieur", 20: "Équipe domicile"}
        with self.assertRaisesRegex(
            LPFEdgeMarketComparisonError,
            "divergent",
        ):
            build_french_market_comparison(
                prediction_day(),
                display,
                team_names=different_names,
            )

    def test_comparison_has_no_network_model_or_write_dependency(self) -> None:
        source = Path(__file__).parents[1].joinpath(
            "src", "lpf_edge_market_comparison.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("requests", source)
        self.assertNotIn("predict_proba", source)
        self.assertNotIn("sqlite3", source)
        self.assertNotIn("INSERT ", source)
        self.assertNotIn("UPDATE ", source)


if __name__ == "__main__":
    unittest.main()
