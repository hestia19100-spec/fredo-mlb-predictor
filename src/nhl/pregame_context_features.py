"""Read-only, point-in-time NHL pregame context from the audited as-of view.

Observed players are not a complete roster. An absent record is never interpreted
as evidence that a player is available or that a goalie is confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from .asof_dataset import AsOfGameView, AsOfObservation, build_asof_game_view
from .contracts import GoalieStatus, PlayerAvailabilityStatus, require_utc


class PregameContextFeatureError(ValueError):
    """A pregame observation cannot be interpreted without ambiguity."""


@dataclass(frozen=True)
class GoalieContext:
    team_id: int
    status: GoalieStatus
    goalie_id: int | None
    evidence_state: str
    observation_id: str | None
    effective_available_at_utc: datetime | None


@dataclass(frozen=True)
class PlayerContext:
    player_id: int
    team_id: int
    status: PlayerAvailabilityStatus
    observation_id: str
    effective_available_at_utc: datetime


@dataclass(frozen=True)
class TeamPregameContext:
    team_id: int
    goalie: GoalieContext
    observed_players: tuple[PlayerContext, ...]
    observed_status_counts: tuple[tuple[str, int], ...]
    roster_completeness_proven: bool


@dataclass(frozen=True)
class PregameContextFeatures:
    target_game_id: int
    information_cutoff_utc: datetime
    away: TeamPregameContext
    home: TeamPregameContext
    unattributed_unknown_player_observation_ids: tuple[str, ...]
    source_snapshot_sha256: str
    feature_sha256: str


def _positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise PregameContextFeatureError(f"{field} must be a positive integer")
    return value


def _known_object(observation: AsOfObservation) -> dict:
    if observation.value_state != "KNOWN" or not isinstance(observation.value, dict):
        raise PregameContextFeatureError("A known pregame observation requires an object")
    return observation.value


def _check_pregame_observation(view: AsOfGameView, observation: AsOfObservation) -> None:
    require_utc(observation.effective_available_at_utc, field_name="effective_available_at_utc")
    if observation.effective_available_at_utc > view.information_cutoff_utc:
        raise PregameContextFeatureError("After-cutoff pregame observation")
    if observation.source_game_id is not None:
        raise PregameContextFeatureError("Pregame observation must not reference a source game")


def _goalie(view: AsOfGameView, observation: AsOfObservation) -> GoalieContext:
    if not observation.entity_id.endswith(":goalie"):
        raise PregameContextFeatureError("Invalid goalie entity")
    team_text = observation.entity_id.removeprefix("team:").removesuffix(":goalie")
    if not team_text.isdecimal():
        raise PregameContextFeatureError("Invalid goalie team")
    team_id = _positive_int(int(team_text), "goalie team")
    if observation.entity_id != f"team:{team_id}:goalie":
        raise PregameContextFeatureError("Noncanonical goalie entity")
    if team_id not in (view.away_team_id, view.home_team_id):
        raise PregameContextFeatureError("Goalie belongs to an unrelated team")
    if observation.value_state == "UNKNOWN":
        if observation.value is not None:
            raise PregameContextFeatureError("Unknown goalie has a non-null value")
        return GoalieContext(team_id, GoalieStatus.UNKNOWN, None, "SOURCE_UNKNOWN",
                             observation.observation_id, observation.effective_available_at_utc)
    value = _known_object(observation)
    if _positive_int(value.get("team_id"), "goalie team") != team_id:
        raise PregameContextFeatureError("Goalie team contradicts entity")
    goalie_id = _positive_int(value.get("goalie_id"), "goalie_id")
    try:
        status = GoalieStatus(value.get("status"))
    except ValueError as error:
        raise PregameContextFeatureError("Invalid goalie status") from error
    if status not in (GoalieStatus.PROBABLE, GoalieStatus.CONFIRMED):
        raise PregameContextFeatureError("Known goalie must be probable or confirmed")
    return GoalieContext(team_id, status, goalie_id, "KNOWN",
                         observation.observation_id, observation.effective_available_at_utc)


def _player(view: AsOfGameView, observation: AsOfObservation) -> PlayerContext | None:
    player_text = observation.entity_id.removeprefix("player:")
    if not player_text.isdecimal():
        raise PregameContextFeatureError("Invalid player entity")
    player_id = _positive_int(int(player_text), "player_id")
    if observation.entity_id != f"player:{player_id}":
        raise PregameContextFeatureError("Noncanonical player entity")
    if observation.value_state == "UNKNOWN":
        if observation.value is not None:
            raise PregameContextFeatureError("Unknown player has a non-null value")
        return None
    value = _known_object(observation)
    if _positive_int(value.get("player_id"), "player_id") != player_id:
        raise PregameContextFeatureError("Player ID contradicts entity")
    team_id = _positive_int(value.get("team_id"), "player team")
    if team_id not in (view.away_team_id, view.home_team_id):
        raise PregameContextFeatureError("Player belongs to an unrelated team")
    try:
        status = PlayerAvailabilityStatus(value.get("status"))
    except ValueError as error:
        raise PregameContextFeatureError("Invalid player status") from error
    if status is PlayerAvailabilityStatus.UNKNOWN:
        raise PregameContextFeatureError("Unknown status requires UNKNOWN value state")
    return PlayerContext(player_id, team_id, status, observation.observation_id,
                         observation.effective_available_at_utc)


def _team(view: AsOfGameView, team_id: int, goalies: dict[int, GoalieContext],
          players: dict[int, PlayerContext]) -> TeamPregameContext:
    goalie = goalies.get(team_id, GoalieContext(team_id, GoalieStatus.UNKNOWN, None,
                                                "NOT_OBSERVED", None, None))
    observed = tuple(sorted((item for item in players.values() if item.team_id == team_id),
                            key=lambda item: item.player_id))
    counts = tuple(sorted((status.value, sum(item.status is status for item in observed))
                          for status in PlayerAvailabilityStatus
                          if status is not PlayerAvailabilityStatus.UNKNOWN
                          and any(item.status is status for item in observed)))
    return TeamPregameContext(team_id, goalie, observed, counts, False)


def summarize_asof_pregame_context(view: AsOfGameView) -> PregameContextFeatures:
    """Summarize pregame observations already admitted by NHL-05."""
    require_utc(view.information_cutoff_utc, field_name="information_cutoff_utc")
    goalies: dict[int, GoalieContext] = {}
    players: dict[int, PlayerContext] = {}
    unknown_players: list[str] = []
    seen_players: set[int] = set()
    evidence: list[tuple[str, str, str]] = []
    seen_observations: set[str] = set()
    for observation in view.observations:
        if observation.kind not in ("PREGAME_GOALIE", "PLAYER_AVAILABILITY"):
            continue
        _check_pregame_observation(view, observation)
        if observation.observation_id in seen_observations:
            raise PregameContextFeatureError("Duplicate pregame observation ID")
        seen_observations.add(observation.observation_id)
        evidence.append((observation.observation_id, observation.observation_sha256,
                         observation.effective_available_at_utc.isoformat()))
        if observation.kind == "PREGAME_GOALIE":
            item = _goalie(view, observation)
            if item.team_id in goalies:
                raise PregameContextFeatureError("Duplicate goalie observation for team")
            goalies[item.team_id] = item
        else:
            item = _player(view, observation)
            player_id = int(observation.entity_id.removeprefix("player:"))
            if player_id in seen_players:
                raise PregameContextFeatureError("Duplicate player observation")
            seen_players.add(player_id)
            if item is None:
                unknown_players.append(observation.observation_id)
            else:
                players[item.player_id] = item
    away = _team(view, view.away_team_id, goalies, players)
    home = _team(view, view.home_team_id, goalies, players)
    unknown_ids = tuple(sorted(unknown_players))
    proof = {
        "schema": "nhl_pregame_context_features_v1",
        "target_game_id": view.target_game_id,
        "cutoff": view.information_cutoff_utc.isoformat(),
        "source_snapshot_sha256": view.snapshot_sha256,
        "evidence": sorted(evidence),
        "away": _canonical_team(away),
        "home": _canonical_team(home),
        "unattributed_unknown_player_observation_ids": unknown_ids,
    }
    digest = sha256(json.dumps(proof, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return PregameContextFeatures(view.target_game_id, view.information_cutoff_utc,
                                  away, home, unknown_ids, view.snapshot_sha256, digest)


def _canonical_team(team: TeamPregameContext) -> dict:
    return {
        "team_id": team.team_id,
        "goalie": {
            "status": team.goalie.status.value,
            "goalie_id": team.goalie.goalie_id,
            "evidence_state": team.goalie.evidence_state,
            "observation_id": team.goalie.observation_id,
            "available_at": None if team.goalie.effective_available_at_utc is None
                            else team.goalie.effective_available_at_utc.isoformat(),
        },
        "players": [(item.player_id, item.status.value, item.observation_id,
                     item.effective_available_at_utc.isoformat()) for item in team.observed_players],
        "roster_completeness_proven": team.roster_completeness_proven,
    }


def build_pregame_context_features(
    database_path: Path, *, allowed_root: Path, target_game_id: int,
    information_cutoff_utc: datetime,
) -> PregameContextFeatures:
    """Read audited NHL storage without writing, requesting data, or predicting."""
    view = build_asof_game_view(database_path, allowed_root=allowed_root,
                               target_game_id=target_game_id,
                               information_cutoff_utc=information_cutoff_utc)
    return summarize_asof_pregame_context(view)
