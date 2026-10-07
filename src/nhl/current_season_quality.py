"""NHL-30: read-only quality comparison of two verified current-season imports.

A later source may add or revise old game rows. The earlier sidecar is never
rewritten and the later one is never backdated for retrospective use.
"""
from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

from .current_season_import import (
    CAPTURE_ROOT, DEFAULT_ROOT, CurrentSeasonImport, verify_current_season_import,
)
from .database import PROJECT_ROOT
from .moneypuck_team_import import TeamGameRow

POLICY_PATH = PROJECT_ROOT / "nhl_protocols/data/nhl30_current_season_quality_v1.json"


class NHLCurrentSeasonQualityError(ValueError):
    """Two imports cannot be compared without preserving their source order."""


def _verify_policy() -> None:
    try:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise NHLCurrentSeasonQualityError("Protocole qualité NHL-30 illisible.") from error
    expected = {
        "schema_version": "nhl30_current_season_quality_v1",
        "purpose": "read_only_capture_comparison",
        "prior_import_immutable": True,
        "later_import_not_backdated": True,
        "current_season_coverage_complete_proven": False,
        "historical_backtest_asof_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
        "mlb_database_mutation": False,
        "nhl_historical_database_mutation": False,
    }
    if not isinstance(policy, dict) or any(policy.get(k) != value for k, value in expected.items()):
        raise NHLCurrentSeasonQualityError("Protocole qualité NHL-30 incompatible.")


def _row_map(imported: CurrentSeasonImport) -> dict[tuple[int, str, str], TeamGameRow]:
    result = {(row.game_id, row.team, row.situation): row for row in imported.regular_rows}
    if len(result) != len(imported.regular_rows):
        raise NHLCurrentSeasonQualityError("Lignes de source dupliquées.")
    if any(row.season != imported.season for row in result.values()):
        raise NHLCurrentSeasonQualityError("Saison de ligne incohérente.")
    return result


def _team_counts(rows: dict[tuple[int, str, str], TeamGameRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows.values():
        if row.situation == "all":
            counts[row.team] = counts.get(row.team, 0) + 1
    return counts


def compare_verified_current_season_imports(
    previous: CurrentSeasonImport, latest: CurrentSeasonImport,
) -> dict[str, object]:
    """Compare verified evidence, reporting additions, removals and revisions."""
    _verify_policy()
    if (not isinstance(previous, CurrentSeasonImport)
            or not isinstance(latest, CurrentSeasonImport)
            or previous.season != latest.season
            or previous.path == latest.path
            or previous.source_observed_at_utc >= latest.source_observed_at_utc
            or previous.imported_at_utc >= latest.imported_at_utc
            or previous.effective_available_at_utc >= latest.effective_available_at_utc
            or previous.training_permitted or latest.training_permitted
            or previous.prediction_publication_permitted or latest.prediction_publication_permitted):
        raise NHLCurrentSeasonQualityError("Ordre ou périmètre des imports invalide.")
    before = _row_map(previous)
    after = _row_map(latest)
    added = set(after) - set(before)
    removed = set(before) - set(after)
    common = set(before) & set(after)
    changed = sorted(key for key in common if before[key] != after[key])
    row_fields = tuple(field.name for field in fields(TeamGameRow))
    revisions = [{
        "game_id": key[0], "team": key[1], "situation": key[2],
        "changed_fields": [name for name in row_fields
                           if getattr(before[key], name) != getattr(after[key], name)],
    } for key in changed]
    older_counts, newer_counts = _team_counts(before), _team_counts(after)
    teams = sorted(set(older_counts) | set(newer_counts))
    prior_dated = sorted({after[key].game_id for key in added
                          if after[key].game_date < previous.source_observed_at_utc.date()})
    status = ("SOURCE_REGRESSION_REVIEW" if removed else
              "SOURCE_REVISIONS_REVIEW" if changed else
              "COVERAGE_EXPANDED" if added else "NO_CHANGE")
    return {
        "schema_version": "nhl30_current_season_quality_v1",
        "status": status,
        "season": previous.season,
        "previous_import_id": previous.path.name,
        "latest_import_id": latest.path.name,
        "previous_response_sha256": previous.source_response_sha256,
        "latest_response_sha256": latest.source_response_sha256,
        "previous_effective_available_at_utc": previous.effective_available_at_utc.isoformat(),
        "latest_effective_available_at_utc": latest.effective_available_at_utc.isoformat(),
        "previous_regular_games": previous.regular_game_count,
        "latest_regular_games": latest.regular_game_count,
        "added_game_ids": sorted({after[key].game_id for key in added}),
        "removed_game_ids": sorted({before[key].game_id for key in removed}),
        "prior_dated_new_game_ids": prior_dated,
        "added_row_count": len(added),
        "removed_row_count": len(removed),
        "revised_row_count": len(changed),
        "revisions": revisions,
        "team_regular_games": [{
            "team": team, "previous": older_counts.get(team, 0),
            "latest": newer_counts.get(team, 0),
        } for team in teams],
        "source_revision_review_required": bool(removed or changed),
        "current_season_coverage_complete_proven": False,
        "historical_backtest_asof_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }


def audit_current_season_import_quality(
    previous_slot: Path, latest_slot: Path, *,
    root: Path = DEFAULT_ROOT, capture_root: Path = CAPTURE_ROOT,
) -> dict[str, object]:
    """Public entry verifies both immutable slots before comparison; no writes."""
    previous = verify_current_season_import(previous_slot, root=root, capture_root=capture_root)
    latest = verify_current_season_import(latest_slot, root=root, capture_root=capture_root)
    return compare_verified_current_season_imports(previous, latest)
