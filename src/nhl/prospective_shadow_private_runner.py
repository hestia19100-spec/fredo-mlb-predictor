"""Run the private NHL research shadow only from already sealed public evidence.

This module never calls a network source or The Odds API. It does not publish
subscriber predictions. The private Git publication is a separate workflow step.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .current_season_import import (
    DEFAULT_ROOT as IMPORT_ROOT, verify_current_season_import,
)
from .database import PROJECT_ROOT
from .moneypuck_team_capture import DEFAULT_ROOT as TEAM_ROOT
from .prospective_shadow_observation import (
    NHLProspectiveShadowError, build_prospective_shadow_observation,
)
from .prospective_shadow_seal import (
    NHLPrivateShadowSealError, PRIVATE_REPOSITORY_NAME,
    seal_private_shadow_observation,
)
from .public_schedule_capture import DEFAULT_ROOT as SCHEDULE_ROOT
from .schedule_capture_readiness import audit_schedule_capture

HORIZON = timedelta(hours=36)
LEAD = timedelta(hours=1)


class NHLPrivateShadowRunnerError(ValueError):
    """The private shadow cannot be run safely."""


@dataclass(frozen=True, slots=True)
class ShadowRunSummary:
    future_games: int
    eligible_games: int
    sealed_games: int
    already_sealed_games: int
    missing_evidence_games: int


def _utc(now: Callable[[], datetime]) -> datetime:
    value = now()
    if (not isinstance(value, datetime) or value.tzinfo is None
            or value.utcoffset() != timedelta(0)):
        raise NHLPrivateShadowRunnerError("Horloge UTC requise.")
    return value.astimezone(timezone.utc)


def _time(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise NHLPrivateShadowRunnerError("Horodatage scellé invalide.")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def run_private_shadow(
    *, private_repository_root: Path,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    schedule_root: Path = SCHEDULE_ROOT,
    import_root: Path = IMPORT_ROOT,
    capture_root: Path = TEAM_ROOT,
) -> ShadowRunSummary:
    """Evaluate only future games proven by completed, verified captures."""
    observed = _utc(now)
    private_root = Path(private_repository_root)
    if (private_root.is_symlink() or not private_root.is_dir()
            or private_root.name != PRIVATE_REPOSITORY_NAME
            or not (private_root / ".git").exists()
            or private_root.resolve().is_relative_to(PROJECT_ROOT.resolve())):
        raise NHLPrivateShadowRunnerError("Dépôt privé distinct non vérifié.")

    # Verify every completed schedule before accepting any game identity. A
    # later verified capture supersedes an earlier schedule for the same ID.
    games: dict[int, tuple[datetime, Path, dict[str, object]]] = {}
    for marker in sorted(Path(schedule_root).rglob("COMPLETED")):
        slot = marker.parent
        audit = audit_schedule_capture(slot)
        captured = _time(audit["observed_at_utc"])
        if captured > observed:
            raise NHLPrivateShadowRunnerError("Calendrier daté du futur.")
        for game in audit["games"]:
            start = _time(game["scheduled_start_utc"])
            if not observed + LEAD < start <= observed + HORIZON:
                continue
            previous = games.get(game["game_id"])
            if previous is None or captured > previous[0]:
                games[game["game_id"]] = (captured, slot, game)
            elif captured == previous[0] and game != previous[2]:
                raise NHLPrivateShadowRunnerError("Calendriers contradictoires.")

    # An absent schedule is a normal no-op. It is never replaced with bookmaker
    # events or a guessed game list.
    if not games:
        return ShadowRunSummary(0, 0, 0, 0, 0)

    imports: dict[int, tuple[datetime, Path]] = {}
    for marker in sorted(Path(import_root).rglob("COMPLETED")):
        imported = verify_current_season_import(
            marker.parent, root=import_root, capture_root=capture_root)
        received = imported.effective_available_at_utc
        if received > observed:
            raise NHLPrivateShadowRunnerError("Import daté du futur.")
        previous = imports.get(imported.season)
        if previous is None or received > previous[0]:
            imports[imported.season] = (received, marker.parent)

    eligible = sealed = already = missing = 0
    for game_id, (_, schedule_slot, game) in sorted(games.items()):
        season = game["season"] // 10_000
        selected = imports.get(season)
        if selected is None:
            missing += 1
            continue
        try:
            observation = build_prospective_shadow_observation(
                schedule_slot, selected[1], target_game_id=game_id,
                explicit_shadow_run=True, now=lambda: observed,
                import_root=import_root, capture_root=capture_root)
        except NHLProspectiveShadowError as error:
            if "Cinq matchs antérieurs par équipe requis" in str(error):
                missing += 1
                continue
            raise
        eligible += 1
        destination = (private_root / "captures" / "nhl" / "shadow"
                       / observation.scheduled_start_utc[:10] / str(game_id))
        if destination.exists() or destination.is_symlink():
            if not (destination / "COMPLETED").is_file():
                raise NHLPrivateShadowRunnerError("Observation privée incomplète.")
            already += 1
            continue
        try:
            seal_private_shadow_observation(
                observation, private_repository_root=private_root,
                explicit_private_seal=True)
        except NHLPrivateShadowSealError as error:
            raise NHLPrivateShadowRunnerError("Scellement privé refusé.") from error
        sealed += 1
    return ShadowRunSummary(len(games), eligible, sealed, already, missing)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Observation NHL privée sans réseau")
    parser.add_argument("--private-repo", type=Path, required=True)
    args = parser.parse_args()
    result = run_private_shadow(private_repository_root=args.private_repo)
    print(
        "NHL_SHADOW_PRIVATE_ONLY "
        f"future={result.future_games} eligible={result.eligible_games} "
        f"sealed={result.sealed_games} already={result.already_sealed_games} "
        f"missing_evidence={result.missing_evidence_games}"
    )


if __name__ == "__main__":
    main()
