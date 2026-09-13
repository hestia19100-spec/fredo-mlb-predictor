"""Journal et stockage immuable des collectes de cotes MLB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from src.database import DATABASE_PATH, get_connection, initialize_database
from src.odds_api import MoneylineEvent, OddsFetchResult


class OddsRepositoryError(RuntimeError):
    """État SQLite incompatible avec une collecte auditée."""


MATCH_STATUSES = frozenset(
    {
        "MATCHED",
        "NO_LOCAL_GAME",
        "START_TIME_MISMATCH",
        "AMBIGUOUS_LOCAL_GAME",
    }
)

ODDS_STORAGE_SCHEMA_VERSION = 1
ODDS_STORAGE_SCHEMA_NAME = "audited_moneyline_odds"
ODDS_STORAGE_STATEMENTS = (
    """
    CREATE TABLE odds_ingestion_runs (
        run_id INTEGER PRIMARY KEY,
        provider TEXT NOT NULL,
        target_official_date TEXT NOT NULL,
        sport_key TEXT NOT NULL,
        region TEXT NOT NULL,
        market TEXT NOT NULL,
        odds_format TEXT NOT NULL,
        started_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        completed_at_utc TEXT,
        status TEXT NOT NULL DEFAULT 'started'
            CHECK (status IN ('started', 'success', 'error')),
        events_received INTEGER NOT NULL DEFAULT 0
            CHECK (events_received >= 0),
        events_matched INTEGER NOT NULL DEFAULT 0
            CHECK (
                events_matched >= 0
                AND events_matched <= events_received
            ),
        bookmaker_quotes_saved INTEGER NOT NULL DEFAULT 0
            CHECK (bookmaker_quotes_saved >= 0),
        raw_response_path TEXT,
        response_sha256 TEXT,
        quota_remaining INTEGER CHECK (quota_remaining >= 0),
        quota_used INTEGER CHECK (quota_used >= 0),
        quota_last_cost INTEGER CHECK (quota_last_cost IN (0, 1)),
        code_version TEXT,
        error_message TEXT,

        CHECK (provider = 'the_odds_api_v4'),
        CHECK (sport_key = 'baseball_mlb'),
        CHECK (region = 'eu'),
        CHECK (market = 'h2h'),
        CHECK (odds_format = 'decimal'),
        CHECK (
            response_sha256 IS NULL
            OR length(response_sha256) = 64
        ),
        CHECK (
            (status = 'started' AND completed_at_utc IS NULL)
            OR
            (
                status IN ('success', 'error')
                AND completed_at_utc IS NOT NULL
            )
        ),
        CHECK (status <> 'error' OR error_message IS NOT NULL)
    )
    """,
    """
    CREATE TABLE odds_events (
        odds_event_id INTEGER PRIMARY KEY,
        run_id INTEGER NOT NULL,
        provider_event_id TEXT NOT NULL,
        commence_time_utc TEXT NOT NULL,
        away_team_name TEXT NOT NULL,
        home_team_name TEXT NOT NULL,
        matched_game_id INTEGER,
        match_status TEXT NOT NULL
            CHECK (
                match_status IN (
                    'MATCHED',
                    'NO_LOCAL_GAME',
                    'START_TIME_MISMATCH',
                    'AMBIGUOUS_LOCAL_GAME'
                )
            ),

        FOREIGN KEY (run_id)
            REFERENCES odds_ingestion_runs (run_id),
        FOREIGN KEY (matched_game_id)
            REFERENCES games (game_id),
        UNIQUE (run_id, provider_event_id),
        CHECK (
            (match_status = 'MATCHED' AND matched_game_id IS NOT NULL)
            OR
            (match_status <> 'MATCHED' AND matched_game_id IS NULL)
        )
    )
    """,
    """
    CREATE TABLE moneyline_odds (
        odds_quote_id INTEGER PRIMARY KEY,
        odds_event_id INTEGER NOT NULL,
        run_id INTEGER NOT NULL,
        game_id INTEGER NOT NULL,
        bookmaker_key TEXT NOT NULL,
        bookmaker_title TEXT NOT NULL,
        bookmaker_last_update_utc TEXT NOT NULL,
        observed_at_utc TEXT NOT NULL,
        away_decimal_odds TEXT NOT NULL,
        home_decimal_odds TEXT NOT NULL,

        FOREIGN KEY (odds_event_id)
            REFERENCES odds_events (odds_event_id),
        FOREIGN KEY (run_id)
            REFERENCES odds_ingestion_runs (run_id),
        FOREIGN KEY (game_id)
            REFERENCES games (game_id),
        UNIQUE (odds_event_id, bookmaker_key),
        CHECK (CAST(away_decimal_odds AS REAL) > 1.0),
        CHECK (CAST(home_decimal_odds AS REAL) > 1.0)
    )
    """,
    """
    CREATE INDEX idx_odds_ingestion_runs_target
    ON odds_ingestion_runs (target_official_date, run_id)
    """,
    """
    CREATE INDEX idx_odds_events_game
    ON odds_events (matched_game_id, run_id)
    """,
    """
    CREATE INDEX idx_moneyline_odds_game
    ON moneyline_odds (game_id, observed_at_utc)
    """,
)
ODDS_STORAGE_SCHEMA_CHECKSUM = sha256(
    "\n".join(
        (
            str(ODDS_STORAGE_SCHEMA_VERSION),
            ODDS_STORAGE_SCHEMA_NAME,
            *ODDS_STORAGE_STATEMENTS,
        )
    ).encode("utf-8")
).hexdigest()
ODDS_STORAGE_TABLES = frozenset(
    {
        "moneyline_odds",
        "odds_events",
        "odds_ingestion_runs",
        "odds_schema_migrations",
    }
)


@dataclass(frozen=True, slots=True)
class MatchedOddsEvent:
    """Décision de rapprochement d’un événement de cotes."""

    event: MoneylineEvent
    matched_game_id: int | None
    match_status: str

    def __post_init__(self) -> None:
        if self.match_status not in MATCH_STATUSES:
            raise ValueError("Le statut de rapprochement est invalide.")
        if (self.match_status == "MATCHED") != (
            self.matched_game_id is not None
        ):
            raise ValueError("Le match local et son statut sont incohérents.")


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Un horodatage UTC avec fuseau est obligatoire.")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _decimal_text(value: Decimal) -> str:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 1:
        raise ValueError("Une cote décimale enregistrée est invalide.")
    return format(value, "f")


def initialize_odds_storage(
    database_path: Path = DATABASE_PATH,
) -> Path:
    """Crée ou contrôle le stockage des cotes sans modifier Shadow v2."""
    initialize_database(database_path)
    with get_connection(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS odds_schema_migrations (
                version INTEGER PRIMARY KEY CHECK (version > 0),
                name TEXT NOT NULL UNIQUE,
                checksum TEXT NOT NULL,
                applied_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        row = connection.execute(
            """
            SELECT name, checksum
            FROM odds_schema_migrations
            WHERE version = ?
            """,
            (ODDS_STORAGE_SCHEMA_VERSION,),
        ).fetchone()
        if row is None:
            connection.execute("SAVEPOINT odds_storage_schema_1")
            try:
                for statement in ODDS_STORAGE_STATEMENTS:
                    connection.execute(statement)
                connection.execute(
                    """
                    INSERT INTO odds_schema_migrations (
                        version,
                        name,
                        checksum
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        ODDS_STORAGE_SCHEMA_VERSION,
                        ODDS_STORAGE_SCHEMA_NAME,
                        ODDS_STORAGE_SCHEMA_CHECKSUM,
                    ),
                )
            except Exception:
                connection.execute(
                    "ROLLBACK TO SAVEPOINT odds_storage_schema_1"
                )
                connection.execute("RELEASE SAVEPOINT odds_storage_schema_1")
                raise
            else:
                connection.execute("RELEASE SAVEPOINT odds_storage_schema_1")
        elif (
            str(row["name"]) != ODDS_STORAGE_SCHEMA_NAME
            or str(row["checksum"]) != ODDS_STORAGE_SCHEMA_CHECKSUM
        ):
            raise OddsRepositoryError(
                "La migration du stockage des cotes a été modifiée."
            )

        actual_tables = {
            str(table["name"])
            for table in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            ).fetchall()
        }
        if not ODDS_STORAGE_TABLES.issubset(actual_tables):
            raise OddsRepositoryError(
                "Le stockage SQLite des cotes est incomplet."
            )
    return database_path


def start_odds_ingestion_run(
    *,
    target_date: date,
    code_version: str | None,
    database_path: Path = DATABASE_PATH,
) -> int:
    """Réserve un nouveau journal avant tout appel à l’API de cotes."""
    if not isinstance(target_date, date) or isinstance(target_date, datetime):
        raise TypeError("target_date doit être une date exacte.")
    normalized_version = (
        code_version.strip()
        if code_version is not None and code_version.strip()
        else None
    )
    initialize_odds_storage(database_path)
    with get_connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO odds_ingestion_runs (
                provider,
                target_official_date,
                sport_key,
                region,
                market,
                odds_format,
                status,
                code_version
            )
            VALUES (
                'the_odds_api_v4', ?, 'baseball_mlb', 'eu',
                'h2h', 'decimal', 'started', ?
            )
            """,
            (target_date.isoformat(), normalized_version),
        )
        run_id = cursor.lastrowid
    if run_id is None:
        raise OddsRepositoryError(
            "SQLite n’a pas retourné l’identifiant de la collecte de cotes."
        )
    return int(run_id)


def _validate_completion_inputs(
    *,
    fetch_result: OddsFetchResult,
    matched_events: tuple[MatchedOddsEvent, ...],
    response_sha256: str,
    raw_response_path: str,
) -> None:
    if type(fetch_result) is not OddsFetchResult:
        raise TypeError("Le reçu de The Odds API est invalide.")
    if response_sha256 != fetch_result.response_body_sha256:
        raise OddsRepositoryError(
            "L’empreinte archivée diverge du reçu de cotes."
        )
    if len(response_sha256) != 64:
        raise ValueError("L’empreinte de la réponse doit contenir 64 caractères.")
    try:
        int(response_sha256, 16)
    except ValueError as error:
        raise ValueError("L’empreinte de la réponse doit être hexadécimale.") from error
    if not raw_response_path or raw_response_path != raw_response_path.strip():
        raise ValueError("Le chemin de l’archive de cotes est invalide.")
    if tuple(item.event for item in matched_events) != fetch_result.events:
        raise OddsRepositoryError(
            "Les événements rapprochés divergent de la réponse contrôlée."
        )


def complete_odds_ingestion_run(
    *,
    run_id: int,
    target_date: date,
    fetch_result: OddsFetchResult,
    matched_events: tuple[MatchedOddsEvent, ...],
    raw_response_path: str,
    response_sha256: str,
    database_path: Path = DATABASE_PATH,
) -> int:
    """Enregistre atomiquement événements, cotes et clôture du journal."""
    _validate_completion_inputs(
        fetch_result=fetch_result,
        matched_events=matched_events,
        response_sha256=response_sha256,
        raw_response_path=raw_response_path,
    )
    observed_at = _utc_text(fetch_result.response_received_at_utc)
    quotes_saved = 0
    initialize_odds_storage(database_path)
    with get_connection(database_path) as connection:
        run = connection.execute(
            """
            SELECT *
            FROM odds_ingestion_runs
            WHERE run_id = ? AND status = 'started'
            """,
            (run_id,),
        ).fetchone()
        if run is None:
            raise OddsRepositoryError(
                "La collecte de cotes est absente ou déjà terminée."
            )
        expected_run = (
            str(run["target_official_date"]) == target_date.isoformat()
            and str(run["provider"]) == fetch_result.provider
            and str(run["sport_key"]) == fetch_result.sport_key
            and str(run["region"]) == fetch_result.region
            and str(run["market"]) == fetch_result.market
            and str(run["odds_format"]) == fetch_result.odds_format
        )
        if not expected_run:
            raise OddsRepositoryError(
                "Le reçu de cotes ne correspond pas au journal réservé."
            )

        for matched in matched_events:
            event = matched.event
            if matched.matched_game_id is not None:
                local_game = connection.execute(
                    """
                    SELECT
                        games.official_date,
                        away_team.name AS away_team_name,
                        home_team.name AS home_team_name
                    FROM games
                    JOIN teams AS away_team
                        ON away_team.team_id = games.away_team_id
                    JOIN teams AS home_team
                        ON home_team.team_id = games.home_team_id
                    WHERE games.game_id = ?
                    """,
                    (matched.matched_game_id,),
                ).fetchone()
                if (
                    local_game is None
                    or str(local_game["official_date"]) != target_date.isoformat()
                    or str(local_game["away_team_name"]) != event.away_team_name
                    or str(local_game["home_team_name"]) != event.home_team_name
                ):
                    raise OddsRepositoryError(
                        "Un rapprochement ne correspond pas au match MLB local."
                    )
            cursor = connection.execute(
                """
                INSERT INTO odds_events (
                    run_id,
                    provider_event_id,
                    commence_time_utc,
                    away_team_name,
                    home_team_name,
                    matched_game_id,
                    match_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    event.provider_event_id,
                    _utc_text(event.commence_time_utc),
                    event.away_team_name,
                    event.home_team_name,
                    matched.matched_game_id,
                    matched.match_status,
                ),
            )
            odds_event_id = cursor.lastrowid
            if odds_event_id is None:
                raise OddsRepositoryError(
                    "SQLite n’a pas identifié l’événement de cotes."
                )
            if matched.match_status != "MATCHED":
                continue
            for bookmaker in event.bookmakers:
                connection.execute(
                    """
                    INSERT INTO moneyline_odds (
                        odds_event_id,
                        run_id,
                        game_id,
                        bookmaker_key,
                        bookmaker_title,
                        bookmaker_last_update_utc,
                        observed_at_utc,
                        away_decimal_odds,
                        home_decimal_odds
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        odds_event_id,
                        run_id,
                        matched.matched_game_id,
                        bookmaker.key,
                        bookmaker.title,
                        _utc_text(bookmaker.last_update_utc),
                        observed_at,
                        _decimal_text(bookmaker.away_decimal_odds),
                        _decimal_text(bookmaker.home_decimal_odds),
                    ),
                )
                quotes_saved += 1

        matched_count = sum(
            item.match_status == "MATCHED" for item in matched_events
        )
        cursor = connection.execute(
            """
            UPDATE odds_ingestion_runs
            SET
                completed_at_utc = ?,
                status = 'success',
                events_received = ?,
                events_matched = ?,
                bookmaker_quotes_saved = ?,
                raw_response_path = ?,
                response_sha256 = ?,
                quota_remaining = ?,
                quota_used = ?,
                quota_last_cost = ?,
                error_message = NULL
            WHERE run_id = ? AND status = 'started'
            """,
            (
                observed_at,
                len(matched_events),
                matched_count,
                quotes_saved,
                raw_response_path,
                response_sha256,
                fetch_result.quota_remaining,
                fetch_result.quota_used,
                fetch_result.quota_last_cost,
                run_id,
            ),
        )
        if cursor.rowcount != 1:
            raise OddsRepositoryError(
                "Le journal de cotes n’a pas pu être clôturé."
            )
    return quotes_saved


def mark_odds_ingestion_error(
    *,
    run_id: int,
    error_message: str,
    database_path: Path = DATABASE_PATH,
) -> None:
    """Ferme une collecte en erreur sans enregistrer de cotes partielles."""
    normalized = error_message.strip()
    if not normalized:
        raise ValueError("Le message d’erreur est obligatoire.")
    initialize_odds_storage(database_path)
    with get_connection(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE odds_ingestion_runs
            SET
                completed_at_utc = ?,
                status = 'error',
                error_message = ?
            WHERE run_id = ? AND status = 'started'
            """,
            (_utc_text(datetime.now(timezone.utc)), normalized, run_id),
        )
        if cursor.rowcount != 1:
            raise OddsRepositoryError(
                "La collecte de cotes est absente ou déjà terminée."
            )


def get_odds_ingestion_run(
    run_id: int,
    database_path: Path = DATABASE_PATH,
) -> dict[str, object]:
    """Relit un journal de cotes précis."""
    initialize_odds_storage(database_path)
    with get_connection(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM odds_ingestion_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    if row is None:
        raise OddsRepositoryError(f"Collecte de cotes introuvable : {run_id}.")
    return dict(row)


def list_moneyline_odds_for_date(
    target_date: date,
    database_path: Path = DATABASE_PATH,
) -> list[dict[str, object]]:
    """Relit les instantanés Moneyline d’une journée MLB."""
    initialize_odds_storage(database_path)
    with get_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT moneyline_odds.*
            FROM moneyline_odds
            JOIN games ON games.game_id = moneyline_odds.game_id
            WHERE games.official_date = ?
            ORDER BY
                moneyline_odds.observed_at_utc,
                moneyline_odds.game_id,
                moneyline_odds.bookmaker_key
            """,
            (target_date.isoformat(),),
        ).fetchall()
    return [dict(row) for row in rows]


__all__ = [
    "MatchedOddsEvent",
    "OddsRepositoryError",
    "complete_odds_ingestion_run",
    "get_odds_ingestion_run",
    "initialize_odds_storage",
    "list_moneyline_odds_for_date",
    "mark_odds_ingestion_error",
    "start_odds_ingestion_run",
]
