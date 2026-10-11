"""Audit whether the frozen NHL history itself predates historical games.

A retrospective CSV can supply descriptive lagged rows and labels, but its
download date is not evidence that those rows were available at an old puck
drop. This report deliberately grants no training or publication permission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path

from .frozen_historical_source import audit_frozen_historical_source
from .historical_official_final import ROOT as FINAL_ROOT
from .moneypuck_five_season import FiveSeasonHistory, REFERENCE_SEASONS
from .moneypuck_team_capture import DEFAULT_ROOT as TEAM_ROOT, verify_team_capture


class NHLHistoricalAsOfCoverageError(ValueError):
    """The historical source timing cannot be audited safely."""


@dataclass(frozen=True, slots=True)
class SeasonHistoricalAsOfCoverage:
    season: int
    regular_games: int
    source_received_after_game: int
    source_received_on_game_date: int


@dataclass(frozen=True, slots=True)
class HistoricalAsOfCoverage:
    source_capture_sha256: str
    source_observed_at_utc: datetime
    historical_games: int
    source_received_after_game: int
    source_received_on_game_date: int
    seasons: tuple[SeasonHistoricalAsOfCoverage, ...]
    audit_sha256: str
    historical_pregame_availability_proven: bool = False
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def classify_frozen_source_timing(
    history: FiveSeasonHistory, *,
    source_observed_at_utc: datetime,
    frozen_audit_sha256: str,
) -> HistoricalAsOfCoverage:
    """Classify all regular games by the archived CSV reception date.

    A date-only game field cannot resolve a same-day cutoff. A CSV containing
    a target game's final statistics before that game date is contradictory.
    Neither case is promoted to pregame evidence.
    """
    if not isinstance(history, FiveSeasonHistory) or not history.regular_coverage_complete:
        raise NHLHistoricalAsOfCoverageError("Historique régulier incomplet.")
    if (not isinstance(source_observed_at_utc, datetime)
            or source_observed_at_utc.tzinfo is None
            or source_observed_at_utc.utcoffset() != timedelta(0)):
        raise NHLHistoricalAsOfCoverageError("Réception UTC requise.")
    if (not isinstance(frozen_audit_sha256, str)
            or len(frozen_audit_sha256) != 64
            or any(char not in "0123456789abcdef" for char in frozen_audit_sha256)):
        raise NHLHistoricalAsOfCoverageError("Empreinte d'audit invalide.")
    if history.observed_at_utc != source_observed_at_utc:
        raise NHLHistoricalAsOfCoverageError("Horodatages de source divergents.")

    games: dict[int, tuple[int, date]] = {}
    for row in history.regular_rows:
        if row.season not in REFERENCE_SEASONS or type(row.game_date) is not date:
            raise NHLHistoricalAsOfCoverageError("Match historique hors périmètre.")
        identity = (row.season, row.game_date)
        previous = games.setdefault(row.game_id, identity)
        if previous != identity:
            raise NHLHistoricalAsOfCoverageError("Dates d'un match divergentes.")
    if not games:
        raise NHLHistoricalAsOfCoverageError("Aucun match historique.")
    observed_date = source_observed_at_utc.date()
    buckets = {season: {"regular": 0, "after": 0, "same": 0}
               for season in REFERENCE_SEASONS}
    for season, game_date in games.values():
        bucket = buckets[season]
        bucket["regular"] += 1
        if observed_date > game_date:
            bucket["after"] += 1
        elif observed_date == game_date:
            bucket["same"] += 1
        else:
            raise NHLHistoricalAsOfCoverageError(
                "Une capture contient un match non encore joué."
            )
    expected = {item.season: item.regular_games for item in history.coverage}
    if set(expected) != set(REFERENCE_SEASONS) or any(
            buckets[season]["regular"] != expected[season]
            for season in REFERENCE_SEASONS):
        raise NHLHistoricalAsOfCoverageError("Couverture et dates divergentes.")
    seasons = tuple(SeasonHistoricalAsOfCoverage(
        season, buckets[season]["regular"], buckets[season]["after"],
        buckets[season]["same"],
    ) for season in REFERENCE_SEASONS)
    proof = {
        "schema": "nhl_historical_asof_coverage_v1",
        "source_file_sha256": history.file_sha256,
        "source_observed_at_utc": source_observed_at_utc.isoformat(),
        "frozen_audit_sha256": frozen_audit_sha256,
        "seasons": [asdict(item) for item in seasons],
        "historical_pregame_availability_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }
    digest = sha256(json.dumps(
        proof, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return HistoricalAsOfCoverage(
        history.file_sha256, source_observed_at_utc,
        sum(item.regular_games for item in seasons),
        sum(item.source_received_after_game for item in seasons),
        sum(item.source_received_on_game_date for item in seasons),
        seasons, digest,
    )


def audit_historical_asof_coverage(
    *, team_root: Path = TEAM_ROOT, final_root: Path = FINAL_ROOT,
) -> HistoricalAsOfCoverage:
    """Reverify the exact receipt-bound source; never fetch or mutate data."""
    frozen = audit_frozen_historical_source(
        team_root=team_root, final_root=final_root
    )
    source = verify_team_capture(frozen.team_slot, allowed_root=team_root)
    if source.response_sha256 != frozen.source_capture_sha256:
        raise NHLHistoricalAsOfCoverageError("Capture historique divergente.")
    return classify_frozen_source_timing(
        source.history, source_observed_at_utc=source.observed_at_utc,
        frozen_audit_sha256=frozen.audit_sha256,
    )


def main() -> int:
    result = audit_historical_asof_coverage()
    print(json.dumps({
        "source_capture_sha256": result.source_capture_sha256,
        "source_observed_at_utc": result.source_observed_at_utc.isoformat(),
        "historical_games": result.historical_games,
        "source_received_after_game": result.source_received_after_game,
        "source_received_on_game_date": result.source_received_on_game_date,
        "seasons": [asdict(item) for item in result.seasons],
        "audit_sha256": result.audit_sha256,
        "historical_pregame_availability_proven":
            result.historical_pregame_availability_proven,
        "training_permitted": result.training_permitted,
        "prediction_publication_permitted":
            result.prediction_publication_permitted,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
