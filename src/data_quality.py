"""Audit reproductible et strictement en lecture seule des données MLB."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
from pathlib import Path
import sqlite3
from typing import Iterable

from src.database import DATABASE_PATH


AUDIT_RULES_VERSION = "1"
SAMPLE_LIMIT = 20

FINAL_STATUS_CODES = frozenset({"F"})
FINAL_STATUS_DETAILS = frozenset(
    {
        "FINAL",
        "GAME OVER",
        "COMPLETED EARLY",
    }
)
POSTPONED_STATUS_DETAILS = frozenset({"POSTPONED"})
CANCELLED_STATUS_DETAILS = frozenset({"CANCELLED", "CANCELED"})

REQUIRED_TABLES = frozenset(
    {
        "games",
        "teams",
        "pitchers",
        "ingestion_runs",
    }
)

REQUIRED_GAME_COLUMNS = frozenset(
    {
        "game_id",
        "season",
        "official_date",
        "game_datetime_utc",
        "game_type",
        "status_code",
        "status_detail",
        "away_team_id",
        "home_team_id",
        "away_score",
        "home_score",
        "away_probable_pitcher_id",
        "home_probable_pitcher_id",
    }
)

TEMPORAL_UNVERIFIABLE_REASON = (
    "La table games conserve le dernier état connu de chaque match, "
    "sans historique reliant chaque observation à son run_id et à son "
    "instant de disponibilité. Les scores peuvent servir de cibles, mais "
    "aucune variable historique ne doit encore être déclarée disponible "
    "avant match à partir de cette table seule."
)


class DataQualityError(RuntimeError):
    """Erreur empêchant la production fiable du rapport."""


@dataclass(frozen=True, slots=True)
class SeasonSummary:
    """Mesures principales pour une saison."""

    season: int
    total_games: int
    final_games: int
    non_final_games: int
    target_games: int
    final_games_with_two_scores: int
    final_games_with_two_pitchers: int
    first_official_date: str | None
    last_official_date: str | None


@dataclass(frozen=True, slots=True)
class StatusSummary:
    """Effectif d’un couple code/détail de statut."""

    status_code: str
    status_detail: str
    count: int


@dataclass(frozen=True, slots=True)
class Finding:
    """Anomalie, avertissement ou information issue de l’audit."""

    severity: str
    code: str
    message: str
    count: int
    sample_game_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Rapport structuré produit pour une base et une coupure précises."""

    rules_version: str
    database_path: Path
    database_sha256: str
    cutoff_date: date
    total_games: int
    unique_game_ids: int
    target_games: int
    successful_ingestions: int
    failed_ingestions: int
    active_ingestions: int
    latest_successful_run_id: int | None
    season_summaries: tuple[SeasonSummary, ...]
    status_summaries: tuple[StatusSummary, ...]
    findings: tuple[Finding, ...]
    integrity_verdict: str
    temporal_verdict: str
    temporal_reason: str


def _normalize_text(value: object) -> str:
    """Normalise une valeur textuelle pour les comparaisons."""
    if value is None:
        return ""

    return str(value).strip().upper()


def _is_integer(value: object) -> bool:
    """Refuse notamment les booléens, sous-classe d’int en Python."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_final(row: sqlite3.Row) -> bool:
    """Applique la définition versionnée d’un match terminé."""
    return (
        _normalize_text(row["status_code"]) in FINAL_STATUS_CODES
        or _normalize_text(row["status_detail"])
        in FINAL_STATUS_DETAILS
    )


def _is_postponed(row: sqlite3.Row) -> bool:
    """Reconnaît un match reporté."""
    return (
        _normalize_text(row["status_detail"])
        in POSTPONED_STATUS_DETAILS
    )


def _is_cancelled(row: sqlite3.Row) -> bool:
    """Reconnaît les deux orthographes possibles d’une annulation."""
    return (
        _normalize_text(row["status_detail"])
        in CANCELLED_STATUS_DETAILS
    )


def _parse_iso_date(value: object) -> date | None:
    """Retourne None lorsque la date officielle est invalide."""
    if not isinstance(value, str):
        return None

    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_iso_datetime(value: object) -> datetime | None:
    """Valide un véritable horodatage UTC avec fuseau explicite."""
    if not isinstance(value, str):
        return None

    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        return None

    return parsed


def _has_two_scores(row: sqlite3.Row) -> bool:
    """Indique si les deux scores entiers sont présents."""
    return _is_integer(row["away_score"]) and _is_integer(
        row["home_score"]
    )


def _has_two_pitchers(row: sqlite3.Row) -> bool:
    """Indique si les deux identifiants de lanceur sont positifs."""
    away_id = row["away_probable_pitcher_id"]
    home_id = row["home_probable_pitcher_id"]

    return (
        _is_integer(away_id)
        and away_id > 0
        and _is_integer(home_id)
        and home_id > 0
    )


def _is_valid_target(
    row: sqlite3.Row,
    official_date: date | None,
    cutoff_date: date,
) -> bool:
    """Définit la population utilisable comme cible de victoire."""
    if (
        not _is_final(row)
        or row["game_type"] != "R"
        or official_date is None
        or official_date > cutoff_date
        or not _has_two_scores(row)
    ):
        return False

    away_score = int(row["away_score"])
    home_score = int(row["home_score"])

    return away_score >= 0 and home_score >= 0 and away_score != home_score


def _readonly_uri(path: Path) -> str:
    """Construit une URI SQLite qui interdit toute écriture."""
    return path.resolve().as_uri() + "?mode=ro"


def _table_names(connection: sqlite3.Connection) -> set[str]:
    """Retourne les tables visibles dans le snapshot SQLite."""
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """
    ).fetchall()

    return {str(row[0]) for row in rows}


def _table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    """Retourne les colonnes déclarées pour une table connue."""
    rows = connection.execute(
        f'PRAGMA table_info("{table_name}")'
    ).fetchall()

    return {str(row[1]) for row in rows}


def _game_id(row: sqlite3.Row) -> int | None:
    """Retourne un identifiant exploitable pour les exemples."""
    value = row["game_id"]
    return int(value) if _is_integer(value) else None


def _finding(
    severity: str,
    code: str,
    message: str,
    rows: Iterable[sqlite3.Row],
) -> Finding | None:
    """Construit un constat stable à partir de lignes concernées."""
    materialized = list(rows)
    if not materialized:
        return None

    identifiers = sorted(
        {
            identifier
            for row in materialized
            if (identifier := _game_id(row)) is not None
        }
    )

    return Finding(
        severity=severity,
        code=code,
        message=message,
        count=len(materialized),
        sample_game_ids=tuple(identifiers[:SAMPLE_LIMIT]),
    )


def _season_summaries(
    rows: list[sqlite3.Row],
    parsed_dates: dict[int, date | None],
    cutoff_date: date,
) -> tuple[SeasonSummary, ...]:
    """Agrège les mesures sans supposer 2 430 matchs par saison."""
    grouped: dict[int, list[sqlite3.Row]] = defaultdict(list)

    for row in rows:
        season = row["season"]
        if _is_integer(season):
            grouped[int(season)].append(row)

    summaries: list[SeasonSummary] = []

    for season in sorted(grouped):
        season_rows = grouped[season]
        final_rows = [row for row in season_rows if _is_final(row)]
        valid_dates = [
            parsed_dates[id(row)]
            for row in season_rows
            if parsed_dates[id(row)] is not None
        ]

        summaries.append(
            SeasonSummary(
                season=season,
                total_games=len(season_rows),
                final_games=len(final_rows),
                non_final_games=len(season_rows) - len(final_rows),
                target_games=sum(
                    _is_valid_target(
                        row,
                        parsed_dates[id(row)],
                        cutoff_date,
                    )
                    for row in season_rows
                ),
                final_games_with_two_scores=sum(
                    _has_two_scores(row) for row in final_rows
                ),
                final_games_with_two_pitchers=sum(
                    _has_two_pitchers(row) for row in final_rows
                ),
                first_official_date=(
                    min(valid_dates).isoformat() if valid_dates else None
                ),
                last_official_date=(
                    max(valid_dates).isoformat() if valid_dates else None
                ),
            )
        )

    return tuple(summaries)


def _status_summaries(
    rows: list[sqlite3.Row],
) -> tuple[StatusSummary, ...]:
    """Compte les couples de statut avec un ordre reproductible."""
    counts = Counter(
        (
            str(row["status_code"] or ""),
            str(row["status_detail"] or ""),
        )
        for row in rows
    )

    summaries = [
        StatusSummary(code, detail, count)
        for (code, detail), count in counts.items()
    ]

    return tuple(
        sorted(
            summaries,
            key=lambda item: (
                -item.count,
                item.status_code,
                item.status_detail,
            ),
        )
    )


def run_data_quality_audit(
    *,
    cutoff_date: date,
    database_path: Path = DATABASE_PATH,
) -> DataQualityReport:
    """Audite un snapshot SQLite sans jamais l’ouvrir en écriture."""
    database_path = Path(database_path)

    if not database_path.is_file():
        raise DataQualityError(f"Base SQLite absente : {database_path}")

    try:
        with closing(
            sqlite3.connect(
                _readonly_uri(database_path),
                uri=True,
                timeout=30,
            )
        ) as source_connection:
            with closing(sqlite3.connect(":memory:")) as connection:
                source_connection.backup(connection)
                snapshot_sha256 = hashlib.sha256(
                    connection.serialize()
                ).hexdigest()
                connection.row_factory = sqlite3.Row

                missing_tables = REQUIRED_TABLES - _table_names(
                    connection
                )
                if missing_tables:
                    raise DataQualityError(
                        "Tables obligatoires absentes : "
                        + ", ".join(sorted(missing_tables))
                    )

                missing_columns = (
                    REQUIRED_GAME_COLUMNS
                    - _table_columns(connection, "games")
                )
                if missing_columns:
                    raise DataQualityError(
                        "Colonnes games obligatoires absentes : "
                        + ", ".join(sorted(missing_columns))
                    )

                rows = connection.execute(
                    """
                    SELECT
                        game_id,
                        season,
                        official_date,
                        game_datetime_utc,
                        game_type,
                        status_code,
                        status_detail,
                        away_team_id,
                        home_team_id,
                        away_score,
                        home_score,
                        away_probable_pitcher_id,
                        home_probable_pitcher_id
                    FROM games
                    ORDER BY season, official_date, game_id
                    """
                ).fetchall()

                foreign_key_violations = connection.execute(
                    "PRAGMA foreign_key_check"
                ).fetchall()

                integrity_check_messages = tuple(
                    str(row[0])
                    for row in connection.execute(
                        "PRAGMA integrity_check"
                    ).fetchall()
                    if str(row[0]).lower() != "ok"
                )

                ingestion_counts = {
                    str(row[0]): int(row[1])
                    for row in connection.execute(
                        """
                        SELECT status, COUNT(*)
                        FROM ingestion_runs
                        GROUP BY status
                        """
                    ).fetchall()
                }

                latest_success_row = connection.execute(
                    """
                    SELECT MAX(run_id)
                    FROM ingestion_runs
                    WHERE status = 'success'
                    """
                ).fetchone()
                latest_successful_run_id = (
                    int(latest_success_row[0])
                    if latest_success_row is not None
                    and latest_success_row[0] is not None
                    else None
                )

    except sqlite3.Error as error:
        raise DataQualityError(
            f"Audit SQLite impossible : {error}"
        ) from error

    parsed_dates = {
        id(row): _parse_iso_date(row["official_date"]) for row in rows
    }
    parsed_datetimes = {
        id(row): _parse_iso_datetime(row["game_datetime_utc"])
        for row in rows
    }

    findings: list[Finding] = []

    def add(finding: Finding | None) -> None:
        if finding is not None:
            findings.append(finding)

    id_counts = Counter(row["game_id"] for row in rows)
    duplicate_ids = {
        value for value, count in id_counts.items() if count > 1
    }

    add(
        _finding(
            "ERROR",
            "duplicate_game_id",
            "Des identifiants de match sont dupliqués.",
            [row for row in rows if row["game_id"] in duplicate_ids],
        )
    )
    add(
        _finding(
            "ERROR",
            "invalid_game_id",
            "Des identifiants de match sont absents ou non positifs.",
            [
                row
                for row in rows
                if not _is_integer(row["game_id"]) or row["game_id"] <= 0
            ],
        )
    )
    add(
        _finding(
            "ERROR",
            "invalid_season",
            "Des saisons sont absentes ou invalides.",
            [
                row
                for row in rows
                if not _is_integer(row["season"]) or row["season"] <= 0
            ],
        )
    )
    add(
        _finding(
            "ERROR",
            "invalid_official_date",
            "Des dates officielles ne sont pas au format ISO valide.",
            [row for row in rows if parsed_dates[id(row)] is None],
        )
    )
    add(
        _finding(
            "ERROR",
            "invalid_game_datetime",
            "Des horaires UTC de match sont invalides.",
            [row for row in rows if parsed_datetimes[id(row)] is None],
        )
    )
    add(
        _finding(
            "WARNING",
            "season_date_mismatch",
            "L’année de la date officielle diffère de la saison déclarée.",
            [
                row
                for row in rows
                if parsed_dates[id(row)] is not None
                and _is_integer(row["season"])
                and parsed_dates[id(row)].year != row["season"]
            ],
        )
    )
    add(
        _finding(
            "ERROR",
            "unexpected_game_type",
            "Des matchs sortent du périmètre de saison régulière R.",
            [row for row in rows if row["game_type"] != "R"],
        )
    )
    add(
        _finding(
            "ERROR",
            "empty_status",
            "Des matchs possèdent un code ou un détail de statut vide.",
            [
                row
                for row in rows
                if not _normalize_text(row["status_code"])
                or not _normalize_text(row["status_detail"])
            ],
        )
    )
    add(
        _finding(
            "ERROR",
            "invalid_team_ids",
            "Des équipes sont absentes, invalides ou identiques.",
            [
                row
                for row in rows
                if not _is_integer(row["away_team_id"])
                or row["away_team_id"] <= 0
                or not _is_integer(row["home_team_id"])
                or row["home_team_id"] <= 0
                or row["away_team_id"] == row["home_team_id"]
            ],
        )
    )
    add(
        _finding(
            "ERROR",
            "partial_score",
            "Un seul des deux scores est renseigné.",
            [
                row
                for row in rows
                if (row["away_score"] is None)
                != (row["home_score"] is None)
            ],
        )
    )
    add(
        _finding(
            "ERROR",
            "final_without_valid_scores",
            "Des matchs terminés n’ont pas deux scores entiers valides.",
            [row for row in rows if _is_final(row) and not _has_two_scores(row)],
        )
    )
    add(
        _finding(
            "ERROR",
            "negative_score",
            "Des scores négatifs sont présents.",
            [
                row
                for row in rows
                if (
                    _is_integer(row["away_score"])
                    and row["away_score"] < 0
                )
                or (
                    _is_integer(row["home_score"])
                    and row["home_score"] < 0
                )
            ],
        )
    )
    add(
        _finding(
            "WARNING",
            "final_tie",
            "Des matchs terminés sont à égalité et sont exclus de la cible.",
            [
                row
                for row in rows
                if _is_final(row)
                and _has_two_scores(row)
                and row["away_score"] == row["home_score"]
            ],
        )
    )
    add(
        _finding(
            "WARNING",
            "extreme_score",
            "Au moins un score dépasse 35 points et mérite vérification.",
            [
                row
                for row in rows
                if (
                    _is_integer(row["away_score"])
                    and row["away_score"] > 35
                )
                or (
                    _is_integer(row["home_score"])
                    and row["home_score"] > 35
                )
            ],
        )
    )
    add(
        _finding(
            "WARNING",
            "score_on_non_final",
            "Des matchs non terminés possèdent déjà au moins un score.",
            [
                row
                for row in rows
                if not _is_final(row)
                and (row["away_score"] is not None or row["home_score"] is not None)
            ],
        )
    )
    add(
        _finding(
            "ERROR",
            "invalid_pitcher_id",
            "Des identifiants de lanceur présents ne sont pas positifs.",
            [
                row
                for row in rows
                if (
                    row["away_probable_pitcher_id"] is not None
                    and (
                        not _is_integer(row["away_probable_pitcher_id"])
                        or row["away_probable_pitcher_id"] <= 0
                    )
                )
                or (
                    row["home_probable_pitcher_id"] is not None
                    and (
                        not _is_integer(row["home_probable_pitcher_id"])
                        or row["home_probable_pitcher_id"] <= 0
                    )
                )
            ],
        )
    )
    add(
        _finding(
            "ERROR",
            "same_pitcher",
            "Le même lanceur est affecté aux deux équipes d’un match.",
            [
                row
                for row in rows
                if row["away_probable_pitcher_id"] is not None
                and row["away_probable_pitcher_id"]
                == row["home_probable_pitcher_id"]
            ],
        )
    )
    add(
        _finding(
            "INFO",
            "missing_probable_pitcher",
            "Des matchs terminés ont au moins un lanceur probable manquant.",
            [row for row in rows if _is_final(row) and not _has_two_pitchers(row)],
        )
    )
    add(
        _finding(
            "ERROR",
            "final_after_cutoff",
            "Des résultats finaux apparaissent après la date de coupure.",
            [
                row
                for row in rows
                if _is_final(row)
                and parsed_dates[id(row)] is not None
                and parsed_dates[id(row)] > cutoff_date
            ],
        )
    )
    add(
        _finding(
            "ERROR",
            "score_after_cutoff",
            "Des scores apparaissent après la date de coupure.",
            [
                row
                for row in rows
                if parsed_dates[id(row)] is not None
                and parsed_dates[id(row)] > cutoff_date
                and (row["away_score"] is not None or row["home_score"] is not None)
            ],
        )
    )
    add(
        _finding(
            "ERROR",
            "contradictory_status_category",
            "Des matchs appartiennent à plusieurs catégories de statut.",
            [
                row
                for row in rows
                if sum(
                    (
                        _is_final(row),
                        _is_postponed(row),
                        _is_cancelled(row),
                    )
                )
                > 1
            ],
        )
    )
    unclassified_rows = [
        row
        for row in rows
        if not _is_final(row)
        and not _is_postponed(row)
        and not _is_cancelled(row)
    ]
    add(
        _finding(
            "ERROR",
            "stale_non_final_before_cutoff",
            "Des matchs antérieurs à la coupure ont un statut non final non classé.",
            [
                row
                for row in unclassified_rows
                if parsed_dates[id(row)] is not None
                and parsed_dates[id(row)] <= cutoff_date
            ],
        )
    )
    add(
        _finding(
            "INFO",
            "future_non_final_status",
            "Des matchs futurs possèdent un statut non final non classé.",
            [
                row
                for row in unclassified_rows
                if parsed_dates[id(row)] is not None
                and parsed_dates[id(row)] > cutoff_date
            ],
        )
    )
    add(
        _finding(
            "INFO",
            "postponed_games",
            "Des matchs reportés sont conservés sans être utilisés comme cibles.",
            [row for row in rows if _is_postponed(row)],
        )
    )
    add(
        _finding(
            "INFO",
            "cancelled_games",
            "Des matchs annulés sont conservés sans être utilisés comme cibles.",
            [row for row in rows if _is_cancelled(row)],
        )
    )

    if foreign_key_violations:
        findings.append(
            Finding(
                severity="ERROR",
                code="foreign_key_violation",
                message="SQLite signale des références étrangères invalides.",
                count=len(foreign_key_violations),
                sample_game_ids=(),
            )
        )

    if integrity_check_messages:
        findings.append(
            Finding(
                severity="ERROR",
                code="sqlite_integrity_error",
                message=(
                    "Le contrôle d’intégrité interne de SQLite a échoué."
                ),
                count=len(integrity_check_messages),
                sample_game_ids=(),
            )
        )

    unique_ids = {
        row["game_id"] for row in rows if _is_integer(row["game_id"])
    }
    target_games = sum(
        _is_valid_target(row, parsed_dates[id(row)], cutoff_date)
        for row in rows
    )

    if not rows:
        findings.append(
            Finding(
                severity="ERROR",
                code="empty_games",
                message="La table games ne contient aucun match.",
                count=1,
                sample_game_ids=(),
            )
        )
    elif target_games == 0:
        findings.append(
            Finding(
                severity="ERROR",
                code="no_usable_targets",
                message=(
                    "Aucun résultat ne peut servir de cible avant la coupure."
                ),
                count=1,
                sample_game_ids=(),
            )
        )

    active_ingestions = ingestion_counts.get("started", 0)
    if active_ingestions:
        findings.append(
            Finding(
                severity="ERROR",
                code="active_ingestion",
                message=(
                    "Une collecte est encore marquée comme active dans le journal."
                ),
                count=active_ingestions,
                sample_game_ids=(),
            )
        )

    severity_order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    findings.sort(key=lambda item: (severity_order[item.severity], item.code))

    integrity_verdict = (
        "FAIL" if any(item.severity == "ERROR" for item in findings) else "PASS"
    )

    return DataQualityReport(
        rules_version=AUDIT_RULES_VERSION,
        database_path=database_path.resolve(),
        database_sha256=snapshot_sha256,
        cutoff_date=cutoff_date,
        total_games=len(rows),
        unique_game_ids=len(unique_ids),
        target_games=target_games,
        successful_ingestions=ingestion_counts.get("success", 0),
        failed_ingestions=ingestion_counts.get("error", 0),
        active_ingestions=active_ingestions,
        latest_successful_run_id=latest_successful_run_id,
        season_summaries=_season_summaries(rows, parsed_dates, cutoff_date),
        status_summaries=_status_summaries(rows),
        findings=tuple(findings),
        integrity_verdict=integrity_verdict,
        temporal_verdict="UNVERIFIABLE",
        temporal_reason=TEMPORAL_UNVERIFIABLE_REASON,
    )


def _percentage(numerator: int, denominator: int) -> str:
    """Formate un taux tout en gérant une population vide."""
    if denominator == 0:
        return "n/a"

    return f"{100 * numerator / denominator:.2f}%"


def display_report(report: DataQualityReport) -> None:
    """Affiche le rapport dans un format lisible et stable."""
    print("Audit qualité des données MLB")
    print(f"Version des règles : {report.rules_version}")
    print(f"Base : {report.database_path}")
    print(f"SHA-256 du snapshot SQLite : {report.database_sha256}")
    print(f"Date de coupure : {report.cutoff_date.isoformat()}")
    print(f"Intégrité structurelle : {report.integrity_verdict}")
    print(f"Sécurité temporelle : {report.temporal_verdict}")
    print(f"Matchs en base : {report.total_games}")
    print(f"Identifiants uniques : {report.unique_game_ids}")
    print(f"Cibles de victoire utilisables : {report.target_games}")
    print(f"Collectes réussies : {report.successful_ingestions}")
    print(f"Collectes en erreur : {report.failed_ingestions}")
    print(f"Collectes encore actives : {report.active_ingestions}")
    print(f"Dernière collecte réussie : {report.latest_successful_run_id}")

    print("\nPar saison")
    print(
        "Saison | Total | Finaux | Non finaux | Cibles | "
        "Scores finaux | Deux lanceurs | Première date | Dernière date"
    )
    for item in report.season_summaries:
        print(
            f"{item.season} | {item.total_games} | {item.final_games} | "
            f"{item.non_final_games} | {item.target_games} | "
            f"{item.final_games_with_two_scores} "
            f"({_percentage(item.final_games_with_two_scores, item.final_games)}) | "
            f"{item.final_games_with_two_pitchers} "
            f"({_percentage(item.final_games_with_two_pitchers, item.final_games)}) | "
            f"{item.first_official_date} | {item.last_official_date}"
        )

    print("\nStatuts")
    print("Code | Détail | Nombre")
    for item in report.status_summaries:
        print(f"{item.status_code} | {item.status_detail} | {item.count}")

    print("\nConstats")
    if not report.findings:
        print("Aucun constat.")
    else:
        labels = {"ERROR": "ERREUR", "WARNING": "ATTENTION", "INFO": "INFO"}
        for item in report.findings:
            examples = (
                ", exemples : "
                + ", ".join(str(value) for value in item.sample_game_ids)
                if item.sample_game_ids
                else ""
            )
            print(
                f"- [{labels[item.severity]}] {item.code} : "
                f"{item.message} ({item.count}{examples})"
            )

    print("\nLimite temporelle")
    print(report.temporal_reason)
    print(
        "Conséquence : les scores finaux peuvent servir de cibles. "
        "Les variables du premier modèle seront calculées à J-1, et les "
        "lanceurs probables resteront exclus tant que leur disponibilité "
        "pré-match n’est pas historisée."
    )


def _argument_date(value: str) -> date:
    """Valide une date de coupure explicite."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "La date doit respecter le format AAAA-MM-JJ."
        ) from error


def main() -> None:
    """Exécute l’audit depuis le terminal."""
    parser = argparse.ArgumentParser(
        description="Audite le snapshot SQLite MLB en lecture seule."
    )
    parser.add_argument(
        "--cutoff-date",
        required=True,
        type=_argument_date,
        help="Dernière date officielle autorisée, au format AAAA-MM-JJ.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help="Chemin de la base SQLite à auditer.",
    )
    arguments = parser.parse_args()

    try:
        report = run_data_quality_audit(
            cutoff_date=arguments.cutoff_date,
            database_path=arguments.database,
        )
    except DataQualityError as error:
        parser.exit(status=1, message=f"Échec de l’audit : {error}\n")

    display_report(report)

    if report.integrity_verdict != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
