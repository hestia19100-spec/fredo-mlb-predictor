"""Deterministic, read-only candidate team-history features for NHL games.

Only observations admitted by the NHL-05 as-of view are considered. These
summaries do not authorize a model, a wager, or a production data source.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_EVEN
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .asof_dataset import AsOfGameView, AsOfObservation, build_asof_game_view


_METRICS = (
    "goals_for", "goals_against", "shots_for", "shots_against",
    "power_play_goals", "power_play_opportunities",
    "penalty_kills", "penalty_kill_opportunities",
)
_PRECISION = Decimal("0.000001")


class TeamHistoryFeatureError(ValueError):
    """A historical statistic cannot be used safely."""


@dataclass(frozen=True)
class TeamHistorySummary:
    team_id: int
    source_game_ids: tuple[int, ...]
    observation_ids: tuple[str, ...]
    complete_game_count: int
    unknown_observation_ids: tuple[str, ...]
    incomplete_observation_ids: tuple[str, ...]
    average_goals_for: Decimal | None
    average_goals_against: Decimal | None
    average_shots_for: Decimal | None
    average_shots_against: Decimal | None
    power_play_conversion_rate: Decimal | None
    penalty_kill_rate: Decimal | None


@dataclass(frozen=True)
class TeamHistoryFeatures:
    target_game_id: int
    information_cutoff_utc: datetime
    away: TeamHistorySummary
    home: TeamHistorySummary
    min_games_per_team: int
    minimum_sample_reached: bool
    source_snapshot_sha256: str
    feature_sha256: str


def _average(rows: list[dict[str, int]], name: str) -> Decimal | None:
    if not rows:
        return None
    return (Decimal(sum(row[name] for row in rows)) / len(rows)).quantize(
        _PRECISION, rounding=ROUND_HALF_EVEN
    )


def _ratio(rows: list[dict[str, int]], numerator: str, denominator: str) -> Decimal | None:
    total = sum(row[denominator] for row in rows)
    if total == 0:
        return None
    return (Decimal(sum(row[numerator] for row in rows)) / total).quantize(
        _PRECISION, rounding=ROUND_HALF_EVEN
    )


def _numeric_id(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TeamHistoryFeatureError(f"Invalid {label}: {value!r}")
    return value


def _stat_row(observation: AsOfObservation) -> dict[str, int] | None:
    value = observation.value
    if not isinstance(value, dict):
        raise TeamHistoryFeatureError("A known team statistic must have an object value")
    if any(key not in value for key in _METRICS):
        return None
    row = {key: _numeric_id(value[key], key) for key in _METRICS}
    if row["power_play_goals"] > row["power_play_opportunities"]:
        raise TeamHistoryFeatureError("Power-play goals exceed opportunities")
    if row["penalty_kills"] > row["penalty_kill_opportunities"]:
        raise TeamHistoryFeatureError("Penalty kills exceed opportunities")
    return row


def _summarize_team(view: AsOfGameView, team_id: int) -> TeamHistorySummary:
    included: list[tuple[int, AsOfObservation, dict[str, int]]] = []
    unknown: list[int] = []
    incomplete: list[int] = []
    seen: set[int] = set()
    for observation in view.observations:
        if observation.kind != "TEAM_STATISTICS":
            continue
        if observation.entity_id not in (f"team:{view.away_team_id}", f"team:{view.home_team_id}"):
            raise TeamHistoryFeatureError("Team statistics refer to an unrelated team")
        if observation.entity_id != f"team:{team_id}":
            continue
        source = observation.source_game_id
        if source is None:
            raise TeamHistoryFeatureError("Team statistics require a source game")
        source = _numeric_id(source, "source_game_id")
        if source == view.target_game_id or source in seen:
            raise TeamHistoryFeatureError("Target-game or duplicate source statistics")
        seen.add(source)
        if observation.effective_available_at_utc > view.information_cutoff_utc:
            raise TeamHistoryFeatureError("After-cutoff statistics in as-of view")
        if observation.value_state == "UNKNOWN":
            unknown.append(observation.observation_id)
            continue
        if observation.value_state != "KNOWN":
            raise TeamHistoryFeatureError("Unexpected team statistic value state")
        row = _stat_row(observation)
        if row is None:
            incomplete.append(observation.observation_id)
            continue
        included.append((source, observation, row))
    included.sort(key=lambda item: (item[0], item[1].observation_id))
    rows = [item[2] for item in included]
    return TeamHistorySummary(
        team_id=team_id,
        source_game_ids=tuple(item[0] for item in included),
        observation_ids=tuple(item[1].observation_id for item in included),
        complete_game_count=len(rows),
        unknown_observation_ids=tuple(sorted(unknown)),
        incomplete_observation_ids=tuple(sorted(incomplete)),
        average_goals_for=_average(rows, "goals_for"),
        average_goals_against=_average(rows, "goals_against"),
        average_shots_for=_average(rows, "shots_for"),
        average_shots_against=_average(rows, "shots_against"),
        power_play_conversion_rate=_ratio(rows, "power_play_goals", "power_play_opportunities"),
        penalty_kill_rate=_ratio(rows, "penalty_kills", "penalty_kill_opportunities"),
    )


def summarize_asof_team_history(
    view: AsOfGameView, *, min_games_per_team: int = 5
) -> TeamHistoryFeatures:
    """Aggregate permitted historical observations; never infer missing games."""
    if isinstance(min_games_per_team, bool) or not isinstance(min_games_per_team, int) or min_games_per_team < 1:
        raise TeamHistoryFeatureError("min_games_per_team must be a positive integer")
    away = _summarize_team(view, view.away_team_id)
    home = _summarize_team(view, view.home_team_id)
    payload = {
        "schema": "nhl_team_history_features_v1",
        "target_game_id": view.target_game_id,
        "information_cutoff_utc": view.information_cutoff_utc.isoformat(),
        "source_snapshot_sha256": view.snapshot_sha256,
        "min_games_per_team": min_games_per_team,
        "away": {
            **{name: str(getattr(away, name)) if isinstance(getattr(away, name), Decimal) else getattr(away, name)
               for name in away.__dataclass_fields__},
            "source_observation_sha256": [
                observation.observation_sha256 for observation in view.observations
                if observation.observation_id in away.observation_ids
            ],
        },
        "home": {
            **{name: str(getattr(home, name)) if isinstance(getattr(home, name), Decimal) else getattr(home, name)
               for name in home.__dataclass_fields__},
            "source_observation_sha256": [
                observation.observation_sha256 for observation in view.observations
                if observation.observation_id in home.observation_ids
            ],
        },
    }
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    return TeamHistoryFeatures(
        target_game_id=view.target_game_id,
        information_cutoff_utc=view.information_cutoff_utc,
        away=away,
        home=home,
        min_games_per_team=min_games_per_team,
        minimum_sample_reached=(away.complete_game_count >= min_games_per_team and home.complete_game_count >= min_games_per_team),
        source_snapshot_sha256=view.snapshot_sha256,
        feature_sha256=digest,
    )


def build_team_history_features(
    database_path: Path, *, allowed_root: Path, target_game_id: int,
    information_cutoff_utc: datetime, min_games_per_team: int = 5,
) -> TeamHistoryFeatures:
    """Read the audited as-of dataset and derive candidate historical features."""
    view = build_asof_game_view(
        database_path, allowed_root=allowed_root, target_game_id=target_game_id,
        information_cutoff_utc=information_cutoff_utc,
    )
    return summarize_asof_team_history(view, min_games_per_team=min_games_per_team)
