"""Préparation et reprise des collectes historiques MLB."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
import time

from src.backfill_planner import (
    BackfillPlanError,
    DateChunk,
    build_date_chunks,
    parse_iso_date,
)
from src.database import DATA_DIR, DATABASE_PATH
from src.ingestion_repository import (
    find_successful_ingestion_run,
)
from src.ingestion_service import (
    INGESTION_SOURCE,
    ScheduleIngestionResult,
    build_schedule_request_parameters,
    run_schedule_ingestion,
)
from src.mlb_api import MLBAPIRetryableError
from src.raw_archive import (
    RawArchiveError,
    verify_raw_archive,
)
from src.retry_policy import (
    RetryPolicy,
    run_with_retries,
)


ACTION_COLLECT = "collect"
ACTION_SKIP = "skip"

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 1.0
DEFAULT_CHUNK_DELAY_SECONDS = 1.0
RETRY_BACKOFF_MULTIPLIER = 2.0


@dataclass(frozen=True, slots=True)
class BackfillChunkPlan:
    """Décision prise pour un lot historique."""

    chunk: DateChunk
    action: str
    successful_run_id: int | None
    reason: str

    @property
    def should_collect(self) -> bool:
        """Indique si le lot doit contacter MLB ultérieurement."""
        return self.action == ACTION_COLLECT


@dataclass(frozen=True, slots=True)
class BackfillPreview:
    """Résumé complet d’un aperçu sans téléchargement."""

    start_date: date
    end_date: date
    chunk_plans: tuple[BackfillChunkPlan, ...]

    @property
    def total_chunks(self) -> int:
        """Retourne le nombre total de lots."""
        return len(self.chunk_plans)

    @property
    def skipped_chunks(self) -> int:
        """Compte les lots déjà réussis et vérifiés."""
        return sum(
            plan.action == ACTION_SKIP
            for plan in self.chunk_plans
        )

    @property
    def pending_chunks(self) -> int:
        """Compte les lots qui resteraient à collecter."""
        return sum(
            plan.action == ACTION_COLLECT
            for plan in self.chunk_plans
        )


@dataclass(frozen=True, slots=True)
class BackfillExecutionResult:
    """Résumé d’une exécution historique limitée."""

    preview: BackfillPreview
    ingestion_results: tuple[ScheduleIngestionResult, ...]
    attempt_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.ingestion_results) != len(self.attempt_counts):
            raise ValueError(
                "Chaque collecte doit posséder un compteur "
                "de tentatives."
            )

        if any(
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 1
            for attempts in self.attempt_counts
        ):
            raise ValueError(
                "Les compteurs de tentatives doivent être positifs."
            )

    @property
    def executed_chunks(self) -> int:
        """Compte les lots réellement collectés."""
        return len(self.ingestion_results)

    @property
    def games_received(self) -> int:
        """Additionne les matchs reçus."""
        return sum(
            result.games_received
            for result in self.ingestion_results
        )

    @property
    def games_saved(self) -> int:
        """Additionne les matchs enregistrés."""
        return sum(
            result.games_saved
            for result in self.ingestion_results
        )

    @property
    def total_attempts(self) -> int:
        """Additionne toutes les tentatives MLB."""
        return sum(self.attempt_counts)

    @property
    def retried_chunks(self) -> int:
        """Compte les lots ayant nécessité une nouvelle tentative."""
        return sum(
            attempts > 1
            for attempts in self.attempt_counts
        )

    @property
    def remaining_pending_chunks(self) -> int:
        """Compte les lots restant après cette exécution."""
        return max(
            0,
            self.preview.pending_chunks - self.executed_chunks,
        )


def _validate_max_chunks(max_chunks: int) -> int:
    """Refuse une limite absente, booléenne ou non positive."""
    if (
        isinstance(max_chunks, bool)
        or not isinstance(max_chunks, int)
        or max_chunks < 1
    ):
        raise ValueError(
            "Le nombre maximal de lots doit être un entier positif."
        )

    return max_chunks


def _validate_delay_seconds(
    delay_seconds: float,
    *,
    field_name: str,
) -> float:
    """Refuse une temporisation négative ou non finie."""
    if (
        isinstance(delay_seconds, bool)
        or not isinstance(delay_seconds, (int, float))
    ):
        raise ValueError(
            f"{field_name} doit être un nombre positif ou nul."
        )

    normalized_delay = float(delay_seconds)

    if (
        not math.isfinite(normalized_delay)
        or normalized_delay < 0
    ):
        raise ValueError(
            f"{field_name} doit être un nombre positif ou nul."
        )

    return normalized_delay


def _resolve_data_directory(
    *,
    database_path: Path,
    data_directory: Path | None,
) -> Path:
    """Retrouve le dossier data associé à la base SQLite."""
    if data_directory is not None:
        return data_directory

    return database_path.parent


def build_backfill_preview(
    *,
    start_date: date,
    end_date: date,
    game_types: Iterable[str] = ("R",),
    database_path: Path = DATABASE_PATH,
    data_directory: Path | None = None,
) -> BackfillPreview:
    """Prépare la reprise et vérifie chaque archive réutilisée."""
    selected_game_types = tuple(game_types)
    effective_data_directory = _resolve_data_directory(
        database_path=database_path,
        data_directory=data_directory,
    )
    chunks = build_date_chunks(start_date, end_date)
    chunk_plans: list[BackfillChunkPlan] = []

    for chunk in chunks:
        request_parameters = build_schedule_request_parameters(
            start_date=chunk.start_date,
            end_date=chunk.end_date,
            game_types=selected_game_types,
        )

        successful_run = find_successful_ingestion_run(
            source=INGESTION_SOURCE,
            start_date=chunk.start_date,
            end_date=chunk.end_date,
            game_types=selected_game_types,
            request_parameters=request_parameters,
            database_path=database_path,
        )

        if successful_run is None:
            chunk_plans.append(
                BackfillChunkPlan(
                    chunk=chunk,
                    action=ACTION_COLLECT,
                    successful_run_id=None,
                    reason=(
                        "aucune réussite strictement identique"
                    ),
                )
            )
            continue

        successful_run_id = int(successful_run["run_id"])
        raw_response_path = successful_run[
            "raw_response_path"
        ]
        response_sha256 = successful_run["response_sha256"]

        if (
            not isinstance(raw_response_path, str)
            or not isinstance(response_sha256, str)
        ):
            chunk_plans.append(
                BackfillChunkPlan(
                    chunk=chunk,
                    action=ACTION_COLLECT,
                    successful_run_id=successful_run_id,
                    reason=(
                        f"collecte n° {successful_run_id} "
                        "non réutilisable : journal incomplet"
                    ),
                )
            )
            continue

        try:
            verify_raw_archive(
                relative_path=raw_response_path,
                expected_sha256=response_sha256,
                data_directory=effective_data_directory,
            )
        except RawArchiveError as error:
            chunk_plans.append(
                BackfillChunkPlan(
                    chunk=chunk,
                    action=ACTION_COLLECT,
                    successful_run_id=successful_run_id,
                    reason=(
                        f"collecte n° {successful_run_id} "
                        f"non réutilisable : {error}"
                    ),
                )
            )
            continue

        chunk_plans.append(
            BackfillChunkPlan(
                chunk=chunk,
                action=ACTION_SKIP,
                successful_run_id=successful_run_id,
                reason=(
                    f"collecte réussie n° {successful_run_id}, "
                    "archive vérifiée"
                ),
            )
        )

    return BackfillPreview(
        start_date=start_date,
        end_date=end_date,
        chunk_plans=tuple(chunk_plans),
    )


def execute_backfill(
    *,
    start_date: date,
    end_date: date,
    game_types: Iterable[str] = ("R",),
    max_chunks: int = 1,
    database_path: Path = DATABASE_PATH,
    data_directory: Path = DATA_DIR,
    code_version: str | None = None,
    retry_policy: RetryPolicy = RetryPolicy(),
    chunk_delay_seconds: float = DEFAULT_CHUNK_DELAY_SECONDS,
    sleep_function: Callable[[float], None] = time.sleep,
) -> BackfillExecutionResult:
    """Collecte les lots nécessaires avec reprise et temporisation."""
    validated_max_chunks = _validate_max_chunks(max_chunks)
    validated_chunk_delay = _validate_delay_seconds(
        chunk_delay_seconds,
        field_name="Le délai entre les lots",
    )
    selected_game_types = tuple(game_types)

    preview = build_backfill_preview(
        start_date=start_date,
        end_date=end_date,
        game_types=selected_game_types,
        database_path=database_path,
        data_directory=data_directory,
    )

    pending_plans = tuple(
        plan
        for plan in preview.chunk_plans
        if plan.should_collect
    )
    selected_plans = pending_plans[:validated_max_chunks]

    ingestion_results: list[ScheduleIngestionResult] = []
    attempt_counts: list[int] = []

    for plan_index, plan in enumerate(selected_plans):
        if plan_index > 0 and validated_chunk_delay > 0:
            sleep_function(validated_chunk_delay)

        retry_outcome = run_with_retries(
            lambda current_plan=plan: run_schedule_ingestion(
                start_date=current_plan.chunk.start_date,
                end_date=current_plan.chunk.end_date,
                game_types=selected_game_types,
                database_path=database_path,
                data_directory=data_directory,
                code_version=code_version,
            ),
            retry_exceptions=(MLBAPIRetryableError,),
            policy=retry_policy,
            sleep_function=sleep_function,
        )

        ingestion_results.append(retry_outcome.value)
        attempt_counts.append(retry_outcome.attempts)

    return BackfillExecutionResult(
        preview=preview,
        ingestion_results=tuple(ingestion_results),
        attempt_counts=tuple(attempt_counts),
    )


def parse_positive_integer(value: str) -> int:
    """Valide un entier positif fourni dans le terminal."""
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "La valeur doit être un entier positif."
        ) from error

    try:
        return _validate_max_chunks(parsed_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_non_negative_float(value: str) -> float:
    """Valide une durée positive ou nulle."""
    try:
        parsed_value = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "La durée doit être un nombre positif ou nul."
        ) from error

    try:
        return _validate_delay_seconds(
            parsed_value,
            field_name="La durée",
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_argument_parser() -> argparse.ArgumentParser:
    """Construit les options du mode historique."""
    parser = argparse.ArgumentParser(
        description=(
            "Prévisualise ou exécute une collecte historique MLB "
            "avec reprise."
        )
    )
    parser.add_argument(
        "--start-date",
        type=parse_iso_date,
        required=True,
        help="Première date incluse au format AAAA-MM-JJ.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_iso_date,
        required=True,
        help="Dernière date incluse au format AAAA-MM-JJ.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Autorise explicitement les appels réseau et "
            "l’enregistrement des données."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=parse_positive_integer,
        default=1,
        help=(
            "Nombre maximal de lots à collecter pendant cette "
            "exécution. Valeur par défaut : 1."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=parse_positive_integer,
        default=DEFAULT_MAX_ATTEMPTS,
        help=(
            "Nombre maximal de tentatives MLB par lot. "
            "Valeur par défaut : 3."
        ),
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=parse_non_negative_float,
        default=DEFAULT_RETRY_DELAY_SECONDS,
        help=(
            "Première attente après une erreur MLB, avant "
            "multiplication par deux. Valeur par défaut : 1 seconde."
        ),
    )
    parser.add_argument(
        "--chunk-delay-seconds",
        type=parse_non_negative_float,
        default=DEFAULT_CHUNK_DELAY_SECONDS,
        help=(
            "Pause entre deux lots réussis. "
            "Valeur par défaut : 1 seconde."
        ),
    )
    return parser


def print_preview(preview: BackfillPreview) -> None:
    """Affiche le plan initial."""
    print(
        "Période : "
        f"{preview.start_date.isoformat()} au "
        f"{preview.end_date.isoformat()}"
    )
    print(f"Lots planifiés : {preview.total_chunks}")
    print(f"Lots déjà réussis : {preview.skipped_chunks}")
    print(f"Lots restant à collecter : {preview.pending_chunks}")
    print()

    for plan in preview.chunk_plans:
        period = (
            f"{plan.chunk.start_date.isoformat()} au "
            f"{plan.chunk.end_date.isoformat()}"
        )

        if plan.action == ACTION_SKIP:
            print(
                f"- IGNORER : {period} | {plan.reason}"
            )
        else:
            print(
                f"- COLLECTER : {period} | {plan.reason}"
            )


def print_execution(
    execution: BackfillExecutionResult,
) -> None:
    """Affiche le résultat des lots réellement collectés."""
    print()
    print("Résultat de l’exécution contrôlée :")
    print(
        f"Lots réellement collectés : "
        f"{execution.executed_chunks}"
    )
    print(
        f"Tentatives MLB utilisées : "
        f"{execution.total_attempts}"
    )
    print(
        f"Lots ayant nécessité une nouvelle tentative : "
        f"{execution.retried_chunks}"
    )
    print(
        f"Matchs reçus : {execution.games_received}"
    )
    print(
        f"Matchs enregistrés : {execution.games_saved}"
    )
    print(
        f"Lots restant à collecter : "
        f"{execution.remaining_pending_chunks}"
    )

    for result, attempts in zip(
        execution.ingestion_results,
        execution.attempt_counts,
        strict=True,
    ):
        print(
            "- COLLECTÉ : "
            f"{result.start_date.isoformat()} au "
            f"{result.end_date.isoformat()} | "
            f"collecte n° {result.run_id} | "
            f"{result.games_received} matchs | "
            f"{attempts} tentative(s) | "
            f"archive {result.archive_relative_path}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Prévisualise ou exécute une collecte historique."""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    if arguments.execute:
        print(
            "Mode exécution contrôlée : "
            f"{arguments.max_chunks} lot(s) maximum."
        )
        print(
            "Politique réseau : "
            f"{arguments.max_attempts} tentative(s) maximum "
            "par lot."
        )
        print(
            "Temporisation : "
            f"{arguments.chunk_delay_seconds:g} seconde(s) "
            "entre les lots."
        )

        retry_policy = RetryPolicy(
            max_attempts=arguments.max_attempts,
            initial_delay_seconds=(
                arguments.retry_delay_seconds
            ),
            backoff_multiplier=RETRY_BACKOFF_MULTIPLIER,
        )

        try:
            execution = execute_backfill(
                start_date=arguments.start_date,
                end_date=arguments.end_date,
                max_chunks=arguments.max_chunks,
                retry_policy=retry_policy,
                chunk_delay_seconds=(
                    arguments.chunk_delay_seconds
                ),
            )
        except Exception as error:
            raise SystemExit(
                f"Échec de la collecte historique MLB : {error}"
            ) from error

        print_preview(execution.preview)
        print_execution(execution)
        return 0

    print("Mode aperçu : aucun appel réseau ne sera effectué.")

    try:
        preview = build_backfill_preview(
            start_date=arguments.start_date,
            end_date=arguments.end_date,
        )
    except (BackfillPlanError, ValueError) as error:
        parser.error(str(error))

    print_preview(preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())