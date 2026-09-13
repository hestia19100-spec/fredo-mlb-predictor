"""Collecte auditable et rapprochement local des cotes Moneyline MLB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
from pathlib import Path

from src.database import DATA_DIR, DATABASE_PATH, get_connection
from src.ingestion_service import detect_code_version
from src.odds_api import OddsAPIError, OddsFetchResult, fetch_mlb_moneyline_odds
from src.odds_repository import (
    MatchedOddsEvent,
    complete_odds_ingestion_run,
    mark_odds_ingestion_error,
    start_odds_ingestion_run,
)
from src.raw_archive import archive_raw_response


ARCHIVE_SOURCE = "the_odds_api_mlb_moneyline"
MAX_START_TIME_DIFFERENCE_MINUTES = 20


class OddsIngestionError(RuntimeError):
    """Collecte de cotes impossible à terminer de façon auditée."""


@dataclass(frozen=True, slots=True)
class OddsIngestionResult:
    """Résumé d’une collecte de cotes enregistrée."""

    run_id: int
    target_date: date
    events_received: int
    events_matched: int
    unmatched_events: int
    bookmaker_quotes_saved: int
    raw_response_path: str
    response_sha256: str
    quota_remaining: int
    quota_used: int
    quota_last_cost: int
    code_version: str | None


@dataclass(frozen=True, slots=True)
class _LocalGame:
    game_id: int
    start_utc: datetime
    away_team_name: str
    home_team_name: str


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise OddsIngestionError("Une heure de match MLB locale est invalide.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise OddsIngestionError(
            "Une heure de match MLB locale est invalide."
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OddsIngestionError("Une heure de match MLB locale est invalide.")
    return parsed.astimezone(timezone.utc)


def _load_local_games(
    target_date: date,
    database_path: Path,
) -> tuple[_LocalGame, ...]:
    with get_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                games.game_id,
                games.game_datetime_utc,
                away_team.name AS away_team_name,
                home_team.name AS home_team_name
            FROM games
            JOIN teams AS away_team
                ON away_team.team_id = games.away_team_id
            JOIN teams AS home_team
                ON home_team.team_id = games.home_team_id
            WHERE games.official_date = ?
            ORDER BY games.game_datetime_utc, games.game_id
            """,
            (target_date.isoformat(),),
        ).fetchall()
    return tuple(
        _LocalGame(
            game_id=int(row["game_id"]),
            start_utc=_parse_utc(row["game_datetime_utc"]),
            away_team_name=str(row["away_team_name"]),
            home_team_name=str(row["home_team_name"]),
        )
        for row in rows
    )


def _match_events(
    fetch_result: OddsFetchResult,
    local_games: tuple[_LocalGame, ...],
) -> tuple[MatchedOddsEvent, ...]:
    preliminary: list[MatchedOddsEvent] = []
    for event in fetch_result.events:
        same_teams = [
            game
            for game in local_games
            if game.away_team_name == event.away_team_name
            and game.home_team_name == event.home_team_name
        ]
        if not same_teams:
            preliminary.append(MatchedOddsEvent(event, None, "NO_LOCAL_GAME"))
            continue
        ranked = sorted(
            [
                (
                    abs(
                        (
                            game.start_utc - event.commence_time_utc
                        ).total_seconds()
                    ),
                    game,
                )
                for game in same_teams
            ],
            key=lambda item: item[0],
        )
        best_seconds, best_game = ranked[0]
        if best_seconds > MAX_START_TIME_DIFFERENCE_MINUTES * 60:
            preliminary.append(
                MatchedOddsEvent(event, None, "START_TIME_MISMATCH")
            )
            continue
        if len(ranked) > 1 and ranked[1][0] == best_seconds:
            preliminary.append(
                MatchedOddsEvent(event, None, "AMBIGUOUS_LOCAL_GAME")
            )
            continue
        preliminary.append(MatchedOddsEvent(event, best_game.game_id, "MATCHED"))

    proposed_counts: dict[int, int] = {}
    for item in preliminary:
        if item.matched_game_id is not None:
            proposed_counts[item.matched_game_id] = (
                proposed_counts.get(item.matched_game_id, 0) + 1
            )
    return tuple(
        (
            MatchedOddsEvent(item.event, None, "AMBIGUOUS_LOCAL_GAME")
            if item.matched_game_id is not None
            and proposed_counts[item.matched_game_id] > 1
            else item
        )
        for item in preliminary
    )


def _safe_error_message(error: Exception) -> str:
    if isinstance(error, (OddsAPIError, OddsIngestionError)):
        return f"{type(error).__name__}: {error}"
    return f"{type(error).__name__}: erreur interne sans détail exposé"


def run_odds_ingestion(
    *,
    target_date: date,
    database_path: Path = DATABASE_PATH,
    data_directory: Path = DATA_DIR,
    code_version: str | None = None,
) -> OddsIngestionResult:
    """Récupère, archive, rapproche et enregistre une observation de cotes."""
    if not isinstance(target_date, date) or isinstance(target_date, datetime):
        raise TypeError("target_date doit être une date exacte.")
    effective_code_version = (
        code_version.strip()
        if code_version is not None and code_version.strip()
        else detect_code_version()
    )
    run_id = start_odds_ingestion_run(
        target_date=target_date,
        code_version=effective_code_version,
        database_path=database_path,
    )
    try:
        fetch_result = fetch_mlb_moneyline_odds()
        if type(fetch_result) is not OddsFetchResult:
            raise OddsIngestionError(
                "The Odds API a retourné un reçu de type inattendu."
            )
        actual_sha256 = hashlib.sha256(fetch_result.raw_content).hexdigest()
        if fetch_result.response_body_sha256 != actual_sha256:
            raise OddsIngestionError(
                "L’empreinte du reçu de cotes diffère de la réponse brute."
            )
        archive = archive_raw_response(
            raw_content=fetch_result.raw_content,
            source_name=ARCHIVE_SOURCE,
            start_date=target_date,
            end_date=target_date,
            data_directory=data_directory,
        )
        if archive.sha256 != fetch_result.response_body_sha256:
            raise OddsIngestionError(
                "L’archive brute des cotes diffère du reçu API."
            )
        local_games = _load_local_games(target_date, database_path)
        matched_events = _match_events(fetch_result, local_games)
        quotes_saved = complete_odds_ingestion_run(
            run_id=run_id,
            target_date=target_date,
            fetch_result=fetch_result,
            matched_events=matched_events,
            raw_response_path=archive.relative_path,
            response_sha256=archive.sha256,
            database_path=database_path,
        )
    except Exception as error:
        try:
            mark_odds_ingestion_error(
                run_id=run_id,
                error_message=_safe_error_message(error),
                database_path=database_path,
            )
        except Exception as journal_error:
            raise OddsIngestionError(
                "La collecte de cotes a échoué et son journal n’a pas pu être fermé."
            ) from journal_error
        raise

    events_matched = sum(
        event.match_status == "MATCHED" for event in matched_events
    )
    return OddsIngestionResult(
        run_id=run_id,
        target_date=target_date,
        events_received=len(fetch_result.events),
        events_matched=events_matched,
        unmatched_events=len(fetch_result.events) - events_matched,
        bookmaker_quotes_saved=quotes_saved,
        raw_response_path=archive.relative_path,
        response_sha256=archive.sha256,
        quota_remaining=fetch_result.quota_remaining,
        quota_used=fetch_result.quota_used,
        quota_last_cost=fetch_result.quota_last_cost,
        code_version=effective_code_version,
    )


__all__ = [
    "OddsIngestionError",
    "OddsIngestionResult",
    "run_odds_ingestion",
]
