"""Fail-closed daily quota gate for the three approved NHL Odds API calls.

The ledger contains no API key or response body. An uncertain request retains
its full reservation, so a network error cannot trigger a free retry.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from typing import Callable
from uuid import uuid4
from zoneinfo import ZoneInfo


EVENTS_URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/events"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
SCORES_URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/scores"
PARIS = ZoneInfo("Europe/Paris")


class NHLQuotaError(RuntimeError):
    """An NHL request is not authorized by the local daily budget."""


def _kind_and_cost(url: str, params: dict[str, str]) -> tuple[str, int]:
    if not isinstance(params, dict) or not isinstance(params.get("apiKey"), str) or not params["apiKey"]:
        raise NHLQuotaError("Clé API absente de la requête NHL.")
    fields = {key: value for key, value in params.items() if key != "apiKey"}
    if url == EVENTS_URL and fields == {}:
        return "events", 0
    if url == ODDS_URL and fields == {
        "bookmakers": "netbet_fr", "markets": "h2h", "oddsFormat": "decimal",
    }:
        return "odds", 1
    if url == SCORES_URL and fields == {"daysFrom": "3", "dateFormat": "iso"}:
        return "scores", 2
    raise NHLQuotaError("Requête NHL hors du périmètre budgété.")


def _paris_day(now: Callable[[], datetime]) -> str:
    observed = now()
    if (not isinstance(observed, datetime) or observed.tzinfo is None
            or observed.utcoffset() != timedelta(0)):
        raise NHLQuotaError("Horloge UTC requise pour le budget NHL.")
    return observed.astimezone(PARIS).date().isoformat()


class NHLDailyQuotaGate:
    """Reserve a maximum cost atomically, then settle from provider headers.

    A kind may be called at most once per Paris calendar day. Reservations
    remain charged at their maximum cost after transport or header failure.
    """

    def __init__(
        self,
        ledger_path: Path,
        daily_limit: int,
        transport: Callable,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if type(daily_limit) is not int or daily_limit < 0:
            raise NHLQuotaError("Plafond quotidien NHL invalide.")
        self.ledger_path = Path(ledger_path)
        self.daily_limit = daily_limit
        self.transport = transport
        self.now = now

    def _connect(self) -> sqlite3.Connection:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.ledger_path, timeout=10, isolation_level=None)
        connection.execute("""CREATE TABLE IF NOT EXISTS requests (
            token TEXT PRIMARY KEY,
            paris_day TEXT NOT NULL,
            kind TEXT NOT NULL,
            reserved_cost INTEGER NOT NULL,
            actual_cost INTEGER,
            UNIQUE (paris_day, kind)
        )""")
        return connection

    def _reserve(self, day: str, kind: str, cost: int) -> str:
        token = uuid4().hex
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT 1 FROM requests WHERE paris_day = ? AND kind = ?", (day, kind),
                ).fetchone()
                if existing is not None:
                    raise NHLQuotaError("Appel NHL déjà tenté aujourd'hui.")
                used = connection.execute(
                    "SELECT COALESCE(SUM(COALESCE(actual_cost, reserved_cost)), 0) "
                    "FROM requests WHERE paris_day = ?", (day,),
                ).fetchone()[0]
                if used + cost > self.daily_limit:
                    raise NHLQuotaError("Plafond quotidien NHL atteint.")
                try:
                    connection.execute(
                        "INSERT INTO requests (token, paris_day, kind, reserved_cost) "
                        "VALUES (?, ?, ?, ?)", (token, day, kind, cost),
                    )
                except sqlite3.IntegrityError as error:
                    raise NHLQuotaError("Appel NHL déjà tenté aujourd'hui.") from error
        except sqlite3.Error as error:
            raise NHLQuotaError("Registre de quota NHL indisponible.") from error
        return token

    def _settle(self, token: str, cost: int, actual: int) -> None:
        if actual > cost:
            raise NHLQuotaError("Coût NHL supérieur à la réservation; appels suivants bloqués.")
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                updated = connection.execute(
                    "UPDATE requests SET actual_cost = ? "
                    "WHERE token = ? AND actual_cost IS NULL", (actual, token),
                ).rowcount
                if updated != 1:
                    raise NHLQuotaError("Réservation NHL introuvable ou déjà soldée.")
        except sqlite3.Error as error:
            raise NHLQuotaError("Registre de quota NHL indisponible.") from error

    def get(self, url: str, *, params: dict[str, str], timeout: int, allow_redirects: bool):
        """A requests.get-compatible, fixed-parameter transport adapter."""
        kind, cost = _kind_and_cost(url, params)
        if timeout != 30 or allow_redirects is not False:
            raise NHLQuotaError("Paramètres de transport NHL inattendus.")
        token = self._reserve(_paris_day(self.now), kind, cost)
        try:
            response = self.transport(
                url, params=params, timeout=timeout, allow_redirects=allow_redirects,
            )
            value = response.headers.get("x-requests-last")
            if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
                raise NHLQuotaError("Coût fournisseur inconnu; réservation conservée.")
            self._settle(token, cost, int(value))
            return response
        except NHLQuotaError:
            raise
        except Exception:
            # Never propagate request URLs: they can contain apiKey.
            raise NHLQuotaError("Requête NHL incertaine; réservation conservée.") from None
