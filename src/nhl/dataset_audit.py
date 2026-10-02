"""NHL-10: read-only leakage and label-coverage audit of offline fixtures.

This checks evidence structure, not data rights or predictive quality. It never
trains a model, estimates probabilities, or authorizes real-source collection.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from .asof_dataset import AsOfGameView, build_asof_game_view
from .contracts import require_utc
from .feature_bundle import summarize_asof_feature_bundle
from .outcome_labels import NHLLabeledGame, assemble_labeled_game, load_audited_final_label

ALLOWED_PREGAME_KINDS = frozenset({
    "TEAM_STATISTICS", "PREGAME_GOALIE", "PLAYER_AVAILABILITY",
})


class NHLDatasetAuditError(ValueError):
    """A dataset row is ambiguous, inconsistent, or leaks future data."""


@dataclass(frozen=True, slots=True)
class NHLDatasetAudit:
    candidate_count: int
    labeled_count: int
    pending_count: int
    pending_game_ids: tuple[int, ...]
    labeled_game_ids: tuple[int, ...]
    coverage_percent: float
    audit_sha256: str
    training_permitted: bool = False


def _digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _hash(document: dict) -> str:
    return sha256(json.dumps(document, sort_keys=True, separators=(",", ":"))
                  .encode("utf-8")).hexdigest()


def _validate_view(view: AsOfGameView) -> None:
    require_utc(view.information_cutoff_utc, field_name="information_cutoff_utc")
    require_utc(view.scheduled_start_utc, field_name="scheduled_start_utc")
    if (type(view.target_game_id) is not int or view.target_game_id <= 0
            or view.information_cutoff_utc >= view.scheduled_start_utc
            or not _digest(view.snapshot_sha256)):
        raise NHLDatasetAuditError("Invalid pregame view or cutoff")
    if (type(view.away_team_id) is not int or type(view.home_team_id) is not int
            or min(view.away_team_id, view.home_team_id) <= 0
            or view.away_team_id == view.home_team_id):
        raise NHLDatasetAuditError("Invalid pregame teams")
    seen: set[str] = set()
    for item in view.observations:
        require_utc(item.effective_available_at_utc,
                    field_name="effective_available_at_utc")
        if (item.kind not in ALLOWED_PREGAME_KINDS
                or item.effective_available_at_utc > view.information_cutoff_utc
                or item.source_game_id == view.target_game_id
                or item.observation_id in seen
                or not _digest(item.observation_sha256)):
            raise NHLDatasetAuditError("Leaking or ambiguous pregame observation")
        seen.add(item.observation_id)


def _validate_pair(view: AsOfGameView, pair: NHLLabeledGame) -> None:
    features, label = pair.features, pair.label
    if (features.target_game_id != view.target_game_id
            or features.information_cutoff_utc != view.information_cutoff_utc
            or features.away_team_id != view.away_team_id
            or features.home_team_id != view.home_team_id
            or features.source_snapshot_sha256 != view.snapshot_sha256
            or not _digest(features.feature_sha256)):
        raise NHLDatasetAuditError("Feature bundle differs from its pregame view")
    for component in (features.history, features.pregame):
        if (component.target_game_id != view.target_game_id
                or component.information_cutoff_utc != view.information_cutoff_utc
                or component.source_snapshot_sha256 != view.snapshot_sha256
                or component.away.team_id != view.away_team_id
                or component.home.team_id != view.home_team_id
                or not _digest(component.feature_sha256)):
            raise NHLDatasetAuditError("Feature component differs from its pregame view")
    expected_feature = _hash({
        "schema": "nhl_feature_bundle_v1",
        "target_game_id": view.target_game_id,
        "cutoff_utc": view.information_cutoff_utc.isoformat(),
        "away_team_id": view.away_team_id,
        "home_team_id": view.home_team_id,
        "source_snapshot_sha256": view.snapshot_sha256,
        "team_history_sha256": features.history.feature_sha256,
        "pregame_context_sha256": features.pregame.feature_sha256,
    })
    if features.feature_sha256 != expected_feature:
        raise NHLDatasetAuditError("Feature bundle digest is inconsistent")
    require_utc(label.final_observed_at_utc, field_name="final_observed_at_utc")
    require_utc(label.evidence_available_at_utc, field_name="evidence_available_at_utc")
    require_utc(label.label_checkpoint_utc, field_name="label_checkpoint_utc")
    if (label.target_game_id != view.target_game_id
            or label.final_observed_at_utc <= view.scheduled_start_utc
            or label.final_observed_at_utc > label.evidence_available_at_utc
            or label.evidence_available_at_utc <= view.information_cutoff_utc
            or label.evidence_available_at_utc > label.label_checkpoint_utc
            or type(label.away_score) is not int or type(label.home_score) is not int
            or min(label.away_score, label.home_score) < 0
            or label.away_score == label.home_score
            or not _digest(label.source_observation_sha256)):
        raise NHLDatasetAuditError("Invalid or prematurely available final label")
    winner = view.away_team_id if label.away_score > label.home_score else view.home_team_id
    if label.winner_team_id != winner:
        raise NHLDatasetAuditError("Final winner conflicts with score or teams")
    expected_pair = _hash({
        "schema": "nhl_labeled_game_v1",
        "target_game_id": view.target_game_id,
        "feature_sha256": features.feature_sha256,
        "source_snapshot_sha256": view.snapshot_sha256,
        "label_checkpoint_utc": label.label_checkpoint_utc.isoformat(),
        "final_observation_sha256": label.source_observation_sha256,
        "away_score": label.away_score,
        "home_score": label.home_score,
        "winner_team_id": label.winner_team_id,
    })
    if pair.pair_sha256 != expected_pair:
        raise NHLDatasetAuditError("Pregame/postgame pair digest is inconsistent")


def audit_labeled_dataset(
    views: Iterable[AsOfGameView], pairs: Iterable[NHLLabeledGame],
) -> NHLDatasetAudit:
    """Fail closed on leakage; report missing labels without imputing them."""
    view_by_id: dict[int, AsOfGameView] = {}
    for view in views:
        _validate_view(view)
        if view.target_game_id in view_by_id:
            raise NHLDatasetAuditError("Duplicate candidate game")
        view_by_id[view.target_game_id] = view
    if not view_by_id:
        raise NHLDatasetAuditError("No candidate games to audit")
    pair_by_id: dict[int, NHLLabeledGame] = {}
    for pair in pairs:
        game_id = pair.label.target_game_id
        if game_id not in view_by_id or game_id in pair_by_id:
            raise NHLDatasetAuditError("Unexpected or duplicate final label")
        _validate_pair(view_by_id[game_id], pair)
        pair_by_id[game_id] = pair
    labeled = tuple(sorted(pair_by_id))
    pending = tuple(sorted(set(view_by_id) - set(pair_by_id)))
    proof = {
        "schema": "nhl_dataset_audit_v1",
        "candidates": [
            {"game_id": game_id,
             "cutoff_utc": view_by_id[game_id].information_cutoff_utc.isoformat(),
             "snapshot_sha256": view_by_id[game_id].snapshot_sha256,
             "pair_sha256": pair_by_id[game_id].pair_sha256 if game_id in pair_by_id else None}
            for game_id in sorted(view_by_id)
        ],
    }
    return NHLDatasetAudit(
        candidate_count=len(view_by_id), labeled_count=len(labeled),
        pending_count=len(pending), pending_game_ids=pending,
        labeled_game_ids=labeled,
        coverage_percent=round(100 * len(labeled) / len(view_by_id), 1),
        audit_sha256=_hash(proof), training_permitted=False,
    )


def audit_offline_labeled_dataset(
    database_path: Path, *, allowed_root: Path,
    targets: Iterable[tuple[int, datetime]], label_checkpoint_utc: datetime,
    min_games_per_team: int = 5,
) -> NHLDatasetAudit:
    """Read audited offline storage; do not modify it or contact a provider."""
    require_utc(label_checkpoint_utc, field_name="label_checkpoint_utc")
    views: list[AsOfGameView] = []
    pairs: list[NHLLabeledGame] = []
    for game_id, cutoff in targets:
        view = build_asof_game_view(
            database_path, allowed_root=allowed_root,
            target_game_id=game_id, information_cutoff_utc=cutoff,
        )
        views.append(view)
        features = summarize_asof_feature_bundle(
            view, min_games_per_team=min_games_per_team)
        label = load_audited_final_label(
            database_path, allowed_root=allowed_root, view=view,
            label_checkpoint_utc=label_checkpoint_utc,
        )
        if label is not None:
            pairs.append(assemble_labeled_game(view, features, label))
    return audit_labeled_dataset(views, pairs)
