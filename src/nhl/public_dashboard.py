"""Read-only, non-predictive NHL historical coverage for the public dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPORT_PATH = (
    Path(__file__).resolve().parents[2]
    / "nhl_protocols"
    / "data"
    / "nhl15_real_archive_report_2026-10-03.json"
)


@dataclass(frozen=True)
class SeasonCoverage:
    season: int
    regular_games: int
    full_lookback_games: int


@dataclass(frozen=True)
class NHLCoverage:
    regular_games: int
    full_lookback_games: int
    seasons: tuple[SeasonCoverage, ...]
    attribution: str


def load_nhl_coverage(path: Path = REPORT_PATH) -> NHLCoverage:
    """Expose aggregate coverage only; never treat it as model performance."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Rapport NHL invalide.")
    required_false = (
        "historical_pregame_availability_proven",
        "independent_final_labels_proven",
        "training_permitted",
        "prediction_publication_permitted",
    )
    if any(payload.get(flag) is not False for flag in required_false):
        raise ValueError("Le statut du rapport NHL a changé : revue requise.")
    rows = payload.get("seasons")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Saisons NHL absentes.")
    seasons = tuple(
        SeasonCoverage(
            season=int(row["season"]),
            regular_games=int(row["regular_games"]),
            full_lookback_games=int(row["full_lookback_games"]),
        )
        for row in rows
    )
    if len({row.season for row in seasons}) != len(seasons):
        raise ValueError("Saisons NHL dupliquées.")
    total = sum(row.regular_games for row in seasons)
    lookback = sum(row.full_lookback_games for row in seasons)
    if total != payload.get("regular_games") or lookback != payload.get("full_lookback_games"):
        raise ValueError("Totaux NHL incohérents.")
    if any(row.regular_games < 0 or not 0 <= row.full_lookback_games <= row.regular_games for row in seasons):
        raise ValueError("Couverture NHL incohérente.")
    attribution = payload.get("attribution")
    if not isinstance(attribution, str) or not attribution:
        raise ValueError("Source NHL absente.")
    return NHLCoverage(total, lookback, seasons, attribution)
