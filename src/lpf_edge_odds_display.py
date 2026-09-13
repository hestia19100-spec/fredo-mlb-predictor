"""Lecture locale des dernières cotes Moneyline affichées dans LPF Edge."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sqlite3

from src.database import DATABASE_PATH


class LPFEdgeOddsDisplayError(RuntimeError):
    """Les cotes locales ne peuvent pas être affichées sans ambiguïté."""


@dataclass(frozen=True, slots=True)
class MoneylineBookmakerDisplayQuote:
    """Paire de cotes cohérente provenant d’un même bookmaker."""

    key: str
    title: str
    last_update_utc: datetime
    home_decimal_odds: Decimal
    away_decimal_odds: Decimal


@dataclass(frozen=True, slots=True)
class MoneylineOddsDisplayGame:
    """Meilleures cotes descriptives d’un match, domicile en premier."""

    game_id: int
    scheduled_start_utc: datetime
    home_team_name: str
    away_team_name: str
    bookmaker_count: int
    home_best_decimal_odds: Decimal | None
    home_best_bookmakers: tuple[str, ...]
    away_best_decimal_odds: Decimal | None
    away_best_bookmakers: tuple[str, ...]
    bookmaker_titles: tuple[str, ...]
    latest_bookmaker_update_utc: datetime | None
    bookmaker_quotes: tuple[MoneylineBookmakerDisplayQuote, ...]

    @property
    def has_odds(self) -> bool:
        return self.bookmaker_count > 0


@dataclass(frozen=True, slots=True)
class MoneylineOddsDisplay:
    """Dernière collecte réussie et ses matchs présentés localement."""

    target_date: date
    run_id: int | None
    region: str | None
    completed_at_utc: datetime | None
    quote_count: int
    games: tuple[MoneylineOddsDisplayGame, ...]

    @property
    def game_count(self) -> int:
        return len(self.games)

    @property
    def quoted_game_count(self) -> int:
        return sum(game.has_odds for game in self.games)

    @property
    def missing_game_count(self) -> int:
        return self.game_count - self.quoted_game_count

    @property
    def bookmaker_titles(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    title
                    for game in self.games
                    for title in game.bookmaker_titles
                },
                key=str.casefold,
            )
        )


def _utc_datetime(value: object, description: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise LPFEdgeOddsDisplayError(f"{description} est absent.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise LPFEdgeOddsDisplayError(f"{description} est invalide.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LPFEdgeOddsDisplayError(f"{description} n’est pas horodaté en UTC.")
    return parsed.astimezone(timezone.utc)


def _decimal_odds(value: object, description: str) -> Decimal:
    if not isinstance(value, str) or not value:
        raise LPFEdgeOddsDisplayError(f"{description} est absente.")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise LPFEdgeOddsDisplayError(f"{description} est invalide.") from error
    if not result.is_finite() or result <= Decimal("1"):
        raise LPFEdgeOddsDisplayError(f"{description} doit dépasser 1.")
    return result


def _empty_game(row: tuple[object, ...]) -> MoneylineOddsDisplayGame:
    home_name = str(row[2] or "").strip()
    away_name = str(row[3] or "").strip()
    if not home_name or not away_name:
        raise LPFEdgeOddsDisplayError(
            "Une équipe est absente du tableau local des cotes."
        )
    return MoneylineOddsDisplayGame(
        game_id=int(row[0]),
        scheduled_start_utc=_utc_datetime(row[1], "L’heure du match"),
        home_team_name=home_name,
        away_team_name=away_name,
        bookmaker_count=0,
        home_best_decimal_odds=None,
        home_best_bookmakers=(),
        away_best_decimal_odds=None,
        away_best_bookmakers=(),
        bookmaker_titles=(),
        latest_bookmaker_update_utc=None,
        bookmaker_quotes=(),
    )


def load_latest_moneyline_odds_display(
    target_date: date,
    *,
    database_path: Path = DATABASE_PATH,
    required_region: str | None = None,
    completed_at_or_before_utc: datetime | None = None,
) -> MoneylineOddsDisplay:
    """Relit la dernière collecte réussie sans réseau ni écriture SQLite."""
    if not isinstance(target_date, date) or isinstance(target_date, datetime):
        raise TypeError("target_date doit être une date exacte.")
    if required_region not in {None, "eu", "fr"}:
        raise ValueError("La région de cotes demandée est invalide.")
    if completed_at_or_before_utc is not None:
        if not isinstance(completed_at_or_before_utc, datetime):
            raise TypeError(
                "completed_at_or_before_utc doit être un instant exact."
            )
        if (
            completed_at_or_before_utc.tzinfo is None
            or completed_at_or_before_utc.utcoffset() is None
        ):
            raise ValueError(
                "La limite temporelle des cotes doit être horodatée."
            )
        completed_at_or_before_utc = completed_at_or_before_utc.astimezone(
            timezone.utc
        )
    if not database_path.is_file() or database_path.is_symlink():
        return MoneylineOddsDisplay(target_date, None, None, None, 0, ())

    uri = f"{database_path.resolve().as_uri()}?mode=ro"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True)
        game_rows = connection.execute(
            """
            SELECT
                games.game_id,
                games.game_datetime_utc,
                home_team.name,
                away_team.name
            FROM games
            JOIN teams AS home_team
                ON home_team.team_id = games.home_team_id
            JOIN teams AS away_team
                ON away_team.team_id = games.away_team_id
            WHERE games.official_date = ?
            ORDER BY games.game_datetime_utc, games.game_id
            """,
            (target_date.isoformat(),),
        ).fetchall()
        base_games = tuple(_empty_game(tuple(row)) for row in game_rows)

        available_tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        required_tables = {
            "moneyline_odds",
            "odds_ingestion_runs",
            "odds_events",
        }
        if not required_tables.issubset(available_tables):
            return MoneylineOddsDisplay(
                target_date, None, None, None, 0, base_games
            )

        query = """
            SELECT run_id, region, completed_at_utc
            FROM odds_ingestion_runs
            WHERE target_official_date = ? AND status = 'success'
        """
        parameters: list[object] = [target_date.isoformat()]
        if required_region is not None:
            query += " AND region = ?"
            parameters.append(required_region)
        query += " ORDER BY run_id DESC"
        candidate_runs = connection.execute(query, parameters).fetchall()
        run: tuple[object, ...] | None = None
        completed_at: datetime | None = None
        for candidate in candidate_runs:
            candidate_completed_at = _utc_datetime(
                candidate[2],
                "La fin de la collecte de cotes",
            )
            if (
                completed_at_or_before_utc is None
                or candidate_completed_at <= completed_at_or_before_utc
            ):
                run = tuple(candidate)
                completed_at = candidate_completed_at
                break
        if run is None:
            return MoneylineOddsDisplay(
                target_date, None, None, None, 0, base_games
            )

        run_id = int(run[0])
        region = str(run[1] or "").strip()
        if region not in {"eu", "fr"}:
            raise LPFEdgeOddsDisplayError(
                "La région de la collecte de cotes est invalide."
            )
        if completed_at is None:
            raise LPFEdgeOddsDisplayError(
                "La fin de la collecte de cotes est absente."
            )
        quote_rows = connection.execute(
            """
            SELECT
                moneyline_odds.game_id,
                moneyline_odds.bookmaker_key,
                moneyline_odds.bookmaker_title,
                moneyline_odds.bookmaker_last_update_utc,
                moneyline_odds.away_decimal_odds,
                moneyline_odds.home_decimal_odds
            FROM moneyline_odds
            JOIN games ON games.game_id = moneyline_odds.game_id
            WHERE
                moneyline_odds.run_id = ?
                AND games.official_date = ?
            ORDER BY moneyline_odds.game_id, moneyline_odds.bookmaker_key
            """,
            (run_id, target_date.isoformat()),
        ).fetchall()
    except sqlite3.Error as error:
        raise LPFEdgeOddsDisplayError(
            "Les cotes locales ne peuvent pas être relues."
        ) from error
    finally:
        if connection is not None:
            connection.close()

    quotes_by_game: dict[int, list[tuple[object, ...]]] = {}
    for row in quote_rows:
        quotes_by_game.setdefault(int(row[0]), []).append(tuple(row))

    games: list[MoneylineOddsDisplayGame] = []
    known_game_ids = {game.game_id for game in base_games}
    if set(quotes_by_game) - known_game_ids:
        raise LPFEdgeOddsDisplayError(
            "Une cote de la collecte ne correspond pas à un match du jour."
        )
    for base in base_games:
        rows = quotes_by_game.get(base.game_id, [])
        if not rows:
            games.append(base)
            continue

        bookmaker_keys: set[str] = set()
        bookmaker_titles: list[str] = []
        home_prices: list[tuple[Decimal, str]] = []
        away_prices: list[tuple[Decimal, str]] = []
        updates: list[datetime] = []
        bookmaker_quotes: list[MoneylineBookmakerDisplayQuote] = []
        for row in rows:
            bookmaker_key = str(row[1] or "").strip()
            bookmaker_title = str(row[2] or "").strip()
            if not bookmaker_key or not bookmaker_title:
                raise LPFEdgeOddsDisplayError(
                    "Un bookmaker de la collecte est invalide."
                )
            if bookmaker_key in bookmaker_keys:
                raise LPFEdgeOddsDisplayError(
                    "Un bookmaker est répété pour le même match."
                )
            bookmaker_keys.add(bookmaker_key)
            bookmaker_titles.append(bookmaker_title)
            last_update = _utc_datetime(
                row[3], "L’actualisation du bookmaker"
            )
            away_odds = _decimal_odds(row[4], "La cote extérieure")
            home_odds = _decimal_odds(row[5], "La cote domicile")
            updates.append(last_update)
            away_prices.append(
                (away_odds, bookmaker_title)
            )
            home_prices.append(
                (home_odds, bookmaker_title)
            )
            bookmaker_quotes.append(
                MoneylineBookmakerDisplayQuote(
                    key=bookmaker_key,
                    title=bookmaker_title,
                    last_update_utc=last_update,
                    home_decimal_odds=home_odds,
                    away_decimal_odds=away_odds,
                )
            )

        best_home = max(price for price, _ in home_prices)
        best_away = max(price for price, _ in away_prices)
        games.append(
            MoneylineOddsDisplayGame(
                game_id=base.game_id,
                scheduled_start_utc=base.scheduled_start_utc,
                home_team_name=base.home_team_name,
                away_team_name=base.away_team_name,
                bookmaker_count=len(rows),
                home_best_decimal_odds=best_home,
                home_best_bookmakers=tuple(
                    sorted(
                        {
                            title
                            for price, title in home_prices
                            if price == best_home
                        },
                        key=str.casefold,
                    )
                ),
                away_best_decimal_odds=best_away,
                away_best_bookmakers=tuple(
                    sorted(
                        {
                            title
                            for price, title in away_prices
                            if price == best_away
                        },
                        key=str.casefold,
                    )
                ),
                bookmaker_titles=tuple(
                    sorted(set(bookmaker_titles), key=str.casefold)
                ),
                latest_bookmaker_update_utc=max(updates),
                bookmaker_quotes=tuple(bookmaker_quotes),
            )
        )

    return MoneylineOddsDisplay(
        target_date=target_date,
        run_id=run_id,
        region=region,
        completed_at_utc=completed_at,
        quote_count=len(quote_rows),
        games=tuple(games),
    )


__all__ = [
    "LPFEdgeOddsDisplayError",
    "MoneylineBookmakerDisplayQuote",
    "MoneylineOddsDisplay",
    "MoneylineOddsDisplayGame",
    "load_latest_moneyline_odds_display",
]
