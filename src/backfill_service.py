"""Préparation et reprise des collectes historiques MLB."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.backfill_planner import (
    BackfillPlanError,
    DateChunk,
    build_date_chunks,
    parse_iso_date,
)
from src.database import DATABASE_PATH
from src.ingestion_repository import (
    find_successful_ingestion_run,
)
from src.ingestion_service import (
    INGESTION_SOURCE,
    build_schedule_request_parameters,
)


ACTION_COLLECT = "collect"
ACTION_SKIP = "skip"


@dataclass(frozen=True, slots=True)
class BackfillChunkPlan:
    """Décision prise pour un lot historique."""

    chunk: DateChunk
    action: str
    successful_run_id: int | None

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
        """Compte les lots déjà réussis."""
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


def build_backfill_preview(
    *,
    start_date: date,
    end_date: date,
    game_types: Iterable[str] = ("R",),
    database_path: Path = DATABASE_PATH,
) -> BackfillPreview:
    """Prépare les décisions de reprise sans appel réseau."""
    selected_game_types = tuple(game_types)
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
                )
            )
        else:
            chunk_plans.append(
                BackfillChunkPlan(
                    chunk=chunk,
                    action=ACTION_SKIP,
                    successful_run_id=int(
                        successful_run["run_id"]
                    ),
                )
            )

    return BackfillPreview(
        start_date=start_date,
        end_date=end_date,
        chunk_plans=tuple(chunk_plans),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Construit les options du mode aperçu."""
    parser = argparse.ArgumentParser(
        description=(
            "Prévisualise une collecte historique MLB avec reprise, "
            "sans effectuer d’appel réseau."
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Affiche les lots à ignorer ou à collecter."""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        preview = build_backfill_preview(
            start_date=arguments.start_date,
            end_date=arguments.end_date,
        )
    except (BackfillPlanError, ValueError) as error:
        parser.error(str(error))

    print("Mode aperçu : aucun appel réseau ne sera effectué.")
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
                f"- IGNORER : {period} | "
                f"collecte réussie n° {plan.successful_run_id}"
            )
        else:
            print(
                f"- COLLECTER : {period} | "
                "aucune réussite strictement identique"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())