from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta


MAX_CHUNK_DAYS = 31


class BackfillPlanError(ValueError):
    """Signale une période historique impossible à planifier."""


@dataclass(frozen=True, slots=True)
class DateChunk:
    """Représente un lot inclusif de dates à collecter."""

    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if self.end_date < self.start_date:
            raise BackfillPlanError(
                "La date de fin d’un lot précède sa date de début."
            )

        if self.day_count > MAX_CHUNK_DAYS:
            raise BackfillPlanError(
                f"Un lot ne peut pas dépasser {MAX_CHUNK_DAYS} jours."
            )

    @property
    def day_count(self) -> int:
        """Retourne le nombre de jours inclus dans le lot."""
        return (self.end_date - self.start_date).days + 1


def build_date_chunks(
    start_date: date,
    end_date: date,
    *,
    max_days: int = MAX_CHUNK_DAYS,
) -> tuple[DateChunk, ...]:
    """Découpe une période en lots continus et sans chevauchement."""
    if end_date < start_date:
        raise BackfillPlanError(
            "La date de fin doit être égale ou postérieure "
            "à la date de début."
        )

    if (
        isinstance(max_days, bool)
        or not isinstance(max_days, int)
        or not 1 <= max_days <= MAX_CHUNK_DAYS
    ):
        raise BackfillPlanError(
            f"La taille d’un lot doit être comprise entre 1 et "
            f"{MAX_CHUNK_DAYS} jours."
        )

    chunks: list[DateChunk] = []
    current_start = start_date

    while current_start <= end_date:
        current_end = min(
            end_date,
            current_start + timedelta(days=max_days - 1),
        )
        chunks.append(
            DateChunk(
                start_date=current_start,
                end_date=current_end,
            )
        )
        current_start = current_end + timedelta(days=1)

    return tuple(chunks)


def parse_iso_date(value: str) -> date:
    """Convertit une date ISO fournie dans le terminal."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Date invalide : {value}. Format attendu : AAAA-MM-JJ."
        ) from error


def build_argument_parser() -> argparse.ArgumentParser:
    """Construit les options du mode aperçu."""
    parser = argparse.ArgumentParser(
        description=(
            "Préparer les lots d’une collecte historique MLB "
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
    """Affiche un résumé du découpage prévu."""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        chunks = build_date_chunks(
            arguments.start_date,
            arguments.end_date,
        )
    except BackfillPlanError as error:
        parser.error(str(error))

    first_chunk = chunks[0]
    last_chunk = chunks[-1]

    print(
        "Période planifiée : "
        f"{arguments.start_date.isoformat()} au "
        f"{arguments.end_date.isoformat()}"
    )
    print(
        f"Taille maximale d’un lot : {MAX_CHUNK_DAYS} jours"
    )
    print(f"Nombre de lots : {len(chunks)}")
    print(
        "Premier lot : "
        f"{first_chunk.start_date.isoformat()} au "
        f"{first_chunk.end_date.isoformat()} "
        f"({first_chunk.day_count} jours)"
    )
    print(
        "Dernier lot : "
        f"{last_chunk.start_date.isoformat()} au "
        f"{last_chunk.end_date.isoformat()} "
        f"({last_chunk.day_count} jours)"
    )
    print("Aucun appel réseau n’a été effectué.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())