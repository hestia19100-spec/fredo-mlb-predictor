"""NHL-09: audited postgame labels kept outside the pregame feature view.

Only synthetic archived observations are supported. No provider, model, odds,
or prediction is invoked here.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from .asof_dataset import AsOfGameView, build_asof_game_view
from .contracts import require_utc
from .database import connect_nhl_database_read_only
from .feature_bundle import NHLFeatureBundle, summarize_asof_feature_bundle
from .repository import audit_nhl_storage


class NHLOutcomeLabelError(ValueError):
    """A final result cannot be linked safely to a pregame snapshot."""


@dataclass(frozen=True, slots=True)
class NHLFinalLabel:
    target_game_id: int
    label_checkpoint_utc: datetime
    final_observed_at_utc: datetime
    evidence_available_at_utc: datetime
    away_score: int
    home_score: int
    winner_team_id: int
    source_observation_id: str
    source_observation_sha256: str


@dataclass(frozen=True, slots=True)
class NHLLabeledGame:
    features: NHLFeatureBundle
    label: NHLFinalLabel
    pair_sha256: str


def _utc(value: str, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise NHLOutcomeLabelError(f"{field_name} must be UTC with Z")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
        require_utc(result, field_name=field_name)
        return result
    except ValueError as error:
        raise NHLOutcomeLabelError(f"Invalid {field_name}") from error


def _candidate(row: sqlite3.Row, *, view: AsOfGameView,
               checkpoint: datetime) -> NHLFinalLabel | None:
    available = _utc(row["effective_available_at_utc"],
                     field_name="effective_available_at_utc")
    if available > checkpoint or row["value_state"] == "UNKNOWN":
        return None
    if row["kind"] != "FINAL_RESULT" or row["source_game_id"] != view.target_game_id:
        raise NHLOutcomeLabelError("Result does not identify the target game")
    if row["target_game_id"] == view.target_game_id:
        raise NHLOutcomeLabelError("A result cannot be a pregame input for itself")
    try:
        value = json.loads(row["canonical_value_json"])
    except (TypeError, json.JSONDecodeError) as error:
        raise NHLOutcomeLabelError("Invalid final-result JSON") from error
    if not isinstance(value, dict) or value.get("game_state") != "FINAL":
        raise NHLOutcomeLabelError("Only an observed final result can be a label")
    final_at = _utc(value.get("final_observed_at_utc"),
                    field_name="final_observed_at_utc")
    if final_at <= view.scheduled_start_utc or final_at > available:
        raise NHLOutcomeLabelError("Final timestamp is inconsistent with its proof")
    away_score, home_score = value.get("away_score"), value.get("home_score")
    if (type(away_score) is not int or type(home_score) is not int
            or min(away_score, home_score) < 0 or away_score == home_score):
        raise NHLOutcomeLabelError("A completed NHL game needs unequal nonnegative scores")
    expected_winner = (view.away_team_id if away_score > home_score
                       else view.home_team_id)
    if type(value.get("winner_team_id")) is not int or value["winner_team_id"] != expected_winner:
        raise NHLOutcomeLabelError("Winner conflicts with the final score or teams")
    digest = row["observation_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest):
        raise NHLOutcomeLabelError("Invalid observation digest")
    return NHLFinalLabel(
        target_game_id=view.target_game_id,
        label_checkpoint_utc=checkpoint,
        final_observed_at_utc=final_at,
        evidence_available_at_utc=available,
        away_score=away_score,
        home_score=home_score,
        winner_team_id=expected_winner,
        source_observation_id=row["observation_id"],
        source_observation_sha256=digest,
    )

def load_audited_final_label(
    database_path: Path, *, allowed_root: Path, view: AsOfGameView,
    label_checkpoint_utc: datetime,
) -> NHLFinalLabel | None:
    """Read a postgame label separately; never inject it into the as-of view."""
    require_utc(label_checkpoint_utc, field_name="label_checkpoint_utc")
    if label_checkpoint_utc <= view.scheduled_start_utc:
        raise NHLOutcomeLabelError("Label checkpoint must follow scheduled start")
    audit_nhl_storage(database_path, allowed_root=allowed_root)
    with closing(connect_nhl_database_read_only(
            database_path, allowed_root=allowed_root)) as connection:
        rows = connection.execute(
            "SELECT * FROM nhl_normalized_observations "
            "WHERE kind = 'FINAL_RESULT' AND source_game_id = ? "
            "ORDER BY effective_available_at_utc, observation_id",
            (view.target_game_id,),
        ).fetchall()
    candidates = [label for row in rows if (label := _candidate(
        row, view=view, checkpoint=label_checkpoint_utc)) is not None]
    if not candidates:
        return None
    newest = max(label.evidence_available_at_utc for label in candidates)
    peers = [label for label in candidates if label.evidence_available_at_utc == newest]
    outcomes = {(label.away_score, label.home_score, label.winner_team_id)
                for label in peers}
    if len(outcomes) != 1:
        raise NHLOutcomeLabelError("Simultaneous final-result proofs disagree")
    return min(peers, key=lambda label: label.source_observation_id)


def assemble_labeled_game(
    view: AsOfGameView, features: NHLFeatureBundle, label: NHLFinalLabel,
) -> NHLLabeledGame:
    """Bind a sealed pregame feature digest to one later outcome proof."""
    if (features.target_game_id != view.target_game_id
            or features.information_cutoff_utc != view.information_cutoff_utc
            or features.source_snapshot_sha256 != view.snapshot_sha256
            or features.away_team_id != view.away_team_id
            or features.home_team_id != view.home_team_id):
        raise NHLOutcomeLabelError("Features do not match the pregame view")
    if (label.target_game_id != view.target_game_id
            or label.final_observed_at_utc <= view.scheduled_start_utc
            or label.evidence_available_at_utc <= view.information_cutoff_utc
            or label.evidence_available_at_utc > label.label_checkpoint_utc):
        raise NHLOutcomeLabelError("Label is not a later result for this game")
    proof = {
        "schema": "nhl_labeled_game_v1",
        "target_game_id": view.target_game_id,
        "feature_sha256": features.feature_sha256,
        "source_snapshot_sha256": view.snapshot_sha256,
        "label_checkpoint_utc": label.label_checkpoint_utc.isoformat(),
        "final_observation_sha256": label.source_observation_sha256,
        "away_score": label.away_score,
        "home_score": label.home_score,
        "winner_team_id": label.winner_team_id,
    }
    digest = sha256(json.dumps(proof, sort_keys=True, separators=(",", ":"))
                    .encode("utf-8")).hexdigest()
    return NHLLabeledGame(features=features, label=label, pair_sha256=digest)


def build_audited_labeled_game(
    database_path: Path, *, allowed_root: Path, target_game_id: int,
    information_cutoff_utc: datetime, label_checkpoint_utc: datetime,
    min_games_per_team: int = 5,
) -> NHLLabeledGame | None:
    """Build an offline training/evaluation record only when a final is proven."""
    view = build_asof_game_view(
        database_path, allowed_root=allowed_root, target_game_id=target_game_id,
        information_cutoff_utc=information_cutoff_utc,
    )
    features = summarize_asof_feature_bundle(
        view, min_games_per_team=min_games_per_team)
    label = load_audited_final_label(
        database_path, allowed_root=allowed_root, view=view,
        label_checkpoint_utc=label_checkpoint_utc,
    )
    if label is None:
        return None
    return assemble_labeled_game(view, features, label)
