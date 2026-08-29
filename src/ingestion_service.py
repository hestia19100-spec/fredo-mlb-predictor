"""Orchestration auditable d’une collecte de calendrier MLB."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path
import subprocess

from src.database import (
    DATA_DIR,
    DATABASE_PATH,
    PROJECT_ROOT,
)
from src.game_repository import save_schedule
from src.ingestion_repository import (
    mark_ingestion_error,
    mark_ingestion_success,
    start_ingestion_run,
)
from src.mlb_api import fetch_schedule_range
from src.raw_archive import archive_raw_response


INGESTION_SOURCE = "mlb_stats_api_schedule"
ARCHIVE_SOURCE = "mlb_schedule"


class ScheduleIngestionError(RuntimeError):
    """Signale une collecte qui ne peut pas être auditée."""


@dataclass(frozen=True, slots=True)
class ScheduleIngestionResult:
    """Résumé d’une collecte terminée avec succès."""

    run_id: int
    start_date: date
    end_date: date
    games_received: int
    games_saved: int
    archive_relative_path: str
    response_sha256: str
    code_version: str | None


def _normalize_game_types(
    game_types: Iterable[str],
) -> tuple[str, ...]:
    """Normalise les types de matchs avant la requête."""
    normalized_types = tuple(
        sorted(
            {
                str(game_type).strip().upper()
                for game_type in game_types
                if str(game_type).strip()
            }
        )
    )

    if not normalized_types:
        raise ValueError("Au moins un type de match est obligatoire.")

    return normalized_types


def detect_code_version(
    project_root: Path = PROJECT_ROOT,
) -> str | None:
    """Détecte le commit Git utilisé pour la collecte."""
    github_sha = os.environ.get("GITHUB_SHA", "").strip()
    if github_sha:
        return github_sha

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (
        FileNotFoundError,
        subprocess.SubprocessError,
    ):
        return None

    detected_version = result.stdout.strip()
    return detected_version or None


def _build_request_parameters(
    *,
    start_date: date,
    end_date: date,
    game_types: tuple[str, ...],
) -> dict[str, object]:
    """Construit les paramètres attendus par le service."""
    return {
        "sportId": 1,
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "gameTypes": ",".join(game_types),
        "hydrate": "probablePitcher",
    }


def run_schedule_ingestion(
    *,
    start_date: date,
    end_date: date,
    game_types: Iterable[str] = ("R",),
    database_path: Path = DATABASE_PATH,
    data_directory: Path = DATA_DIR,
    code_version: str | None = None,
) -> ScheduleIngestionResult:
    """Récupère, archive, enregistre et journalise une période."""
    normalized_game_types = _normalize_game_types(game_types)
    request_parameters = _build_request_parameters(
        start_date=start_date,
        end_date=end_date,
        game_types=normalized_game_types,
    )

    effective_code_version = (
        code_version.strip()
        if code_version is not None and code_version.strip()
        else detect_code_version()
    )

    run_id = start_ingestion_run(
        source=INGESTION_SOURCE,
        start_date=start_date,
        end_date=end_date,
        game_types=normalized_game_types,
        request_parameters=request_parameters,
        code_version=effective_code_version,
        database_path=database_path,
    )

    try:
        fetch_result = fetch_schedule_range(
            start_date,
            end_date,
            game_types=normalized_game_types,
        )

        if fetch_result.request_parameters != request_parameters:
            raise ScheduleIngestionError(
                "Les paramètres réellement envoyés à MLB "
                "diffèrent des paramètres journalisés."
            )

        archive = archive_raw_response(
            raw_content=fetch_result.raw_content,
            source_name=ARCHIVE_SOURCE,
            start_date=start_date,
            end_date=end_date,
            data_directory=data_directory,
        )

        games_saved = save_schedule(
            fetch_result.games,
            database_path,
        )

        mark_ingestion_success(
            run_id=run_id,
            records_received=len(fetch_result.games),
            records_saved=games_saved,
            raw_response_path=archive.relative_path,
            response_sha256=archive.sha256,
            database_path=database_path,
        )
    except Exception as error:
        error_description = (
            f"{type(error).__name__}: {error}"
        )

        try:
            mark_ingestion_error(
                run_id=run_id,
                error_message=error_description,
                database_path=database_path,
            )
        except Exception as journal_error:
            raise ScheduleIngestionError(
                "La collecte a échoué et son journal n’a pas pu "
                f"être clôturé : {journal_error}"
            ) from error

        raise

    return ScheduleIngestionResult(
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
        games_received=len(fetch_result.games),
        games_saved=games_saved,
        archive_relative_path=archive.relative_path,
        response_sha256=archive.sha256,
        code_version=effective_code_version,
    )


def _parse_date(value: str) -> date:
    """Valide une date au format AAAA-MM-JJ."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "La date doit respecter le format AAAA-MM-JJ."
        ) from error


def main() -> None:
    """Lance une collecte auditable depuis le terminal."""
    parser = argparse.ArgumentParser(
        description="Collecte et archive une période MLB."
    )
    parser.add_argument(
        "--start-date",
        required=True,
        type=_parse_date,
        help="Début de période au format AAAA-MM-JJ.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        type=_parse_date,
        help="Fin de période au format AAAA-MM-JJ.",
    )
    arguments = parser.parse_args()

    try:
        result = run_schedule_ingestion(
            start_date=arguments.start_date,
            end_date=arguments.end_date,
        )
    except Exception as error:
        raise SystemExit(
            f"Échec de la collecte MLB : {error}"
        ) from error

    print(f"Collecte terminée : {result.run_id}")
    print(
        "Période : "
        f"{result.start_date.isoformat()} au "
        f"{result.end_date.isoformat()}"
    )
    print(f"Matchs reçus : {result.games_received}")
    print(f"Matchs enregistrés : {result.games_saved}")
    print(f"Archive : {result.archive_relative_path}")
    print(f"SHA-256 : {result.response_sha256}")
    print(
        "Version du code : "
        f"{result.code_version or 'inconnue'}"
    )


if __name__ == "__main__":
    main()