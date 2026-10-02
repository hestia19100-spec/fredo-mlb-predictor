"""NHL-08: one read-only, point-in-time bundle of candidate features.

This is an auditable input record, not a probability, model, or betting signal.
Only the NHL-05 as-of view may supply observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from .asof_dataset import AsOfGameView, build_asof_game_view
from .contracts import require_utc
from .pregame_context_features import (
    PregameContextFeatures,
    summarize_asof_pregame_context,
)
from .team_history_features import (
    TeamHistoryFeatures,
    summarize_asof_team_history,
)


_ALLOWED_KINDS = frozenset({
    "TEAM_STATISTICS", "PREGAME_GOALIE", "PLAYER_AVAILABILITY",
})


class NHLFeatureBundleError(ValueError):
    """The as-of input cannot be assembled without ambiguity or leakage."""


@dataclass(frozen=True, slots=True)
class NHLFeatureBundle:
    target_game_id: int
    information_cutoff_utc: datetime
    away_team_id: int
    home_team_id: int
    history: TeamHistoryFeatures
    pregame: PregameContextFeatures
    source_snapshot_sha256: str
    feature_sha256: str


def _validate_view(view: AsOfGameView) -> None:
    require_utc(view.information_cutoff_utc, field_name="information_cutoff_utc")
    require_utc(view.scheduled_start_utc, field_name="scheduled_start_utc")
    if type(view.target_game_id) is not int or view.target_game_id <= 0:
        raise NHLFeatureBundleError("Invalid target game")
    if any(type(team_id) is not int or team_id <= 0
           for team_id in (view.away_team_id, view.home_team_id)):
        raise NHLFeatureBundleError("Invalid team ID")
    if view.away_team_id == view.home_team_id:
        raise NHLFeatureBundleError("A team cannot play itself")
    if view.information_cutoff_utc >= view.scheduled_start_utc:
        raise NHLFeatureBundleError("Cutoff must precede the scheduled start")
    if len(view.snapshot_sha256) != 64 or any(char not in "0123456789abcdef"
                                               for char in view.snapshot_sha256):
        raise NHLFeatureBundleError("Invalid as-of snapshot digest")
    seen: set[str] = set()
    for observation in view.observations:
        if observation.kind not in _ALLOWED_KINDS:
            raise NHLFeatureBundleError("Target outcome, game state, or unsupported input")
        if observation.observation_id in seen:
            raise NHLFeatureBundleError("Duplicate observation ID")
        seen.add(observation.observation_id)
        require_utc(observation.effective_available_at_utc,
                    field_name="effective_available_at_utc")
        if observation.effective_available_at_utc > view.information_cutoff_utc:
            raise NHLFeatureBundleError("After-cutoff observation")


def summarize_asof_feature_bundle(
    view: AsOfGameView, *, min_games_per_team: int = 5,
) -> NHLFeatureBundle:
    """Assemble both NHL feature families from the exact same audited view."""
    _validate_view(view)
    history = summarize_asof_team_history(view, min_games_per_team=min_games_per_team)
    pregame = summarize_asof_pregame_context(view)
    for component in (history, pregame):
        if (component.target_game_id != view.target_game_id
                or component.information_cutoff_utc != view.information_cutoff_utc
                or component.source_snapshot_sha256 != view.snapshot_sha256
                or component.away.team_id != view.away_team_id
                or component.home.team_id != view.home_team_id):
            raise NHLFeatureBundleError("Feature components do not share one as-of view")
    proof = {
        "schema": "nhl_feature_bundle_v1",
        "target_game_id": view.target_game_id,
        "cutoff_utc": view.information_cutoff_utc.isoformat(),
        "away_team_id": view.away_team_id,
        "home_team_id": view.home_team_id,
        "source_snapshot_sha256": view.snapshot_sha256,
        "team_history_sha256": history.feature_sha256,
        "pregame_context_sha256": pregame.feature_sha256,
    }
    digest = sha256(json.dumps(proof, sort_keys=True, separators=(",", ":"))
                    .encode("utf-8")).hexdigest()
    return NHLFeatureBundle(
        target_game_id=view.target_game_id,
        information_cutoff_utc=view.information_cutoff_utc,
        away_team_id=view.away_team_id,
        home_team_id=view.home_team_id,
        history=history,
        pregame=pregame,
        source_snapshot_sha256=view.snapshot_sha256,
        feature_sha256=digest,
    )


def build_asof_feature_bundle(
    database_path: Path, *, allowed_root: Path, target_game_id: int,
    information_cutoff_utc: datetime, min_games_per_team: int = 5,
) -> NHLFeatureBundle:
    """Audit and read NHL storage once; do not write or call any provider."""
    view = build_asof_game_view(
        database_path, allowed_root=allowed_root,
        target_game_id=target_game_id,
        information_cutoff_utc=information_cutoff_utc,
    )
    return summarize_asof_feature_bundle(view, min_games_per_team=min_games_per_team)
