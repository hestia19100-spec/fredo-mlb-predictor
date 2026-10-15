"""Collect fresh NHL evidence for a private, non-publishable research shadow.

The source captures and current-season import are append-only. This module does
not fit a model, choose a bet, write to the public repository, or use an odds API.
A separate private workflow may run the prospective shadow after these checks.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import re
from typing import Callable
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from src.nhl.current_season_import import (
    DEFAULT_ROOT as IMPORT_ROOT, import_current_season_capture,
)
from src.nhl.moneypuck_team_capture import (
    DEFAULT_ROOT as TEAM_ROOT, capture_team_history,
)
from src.nhl.public_schedule_capture import (
    DEFAULT_ROOT as SCHEDULE_ROOT, capture_public_schedule,
)

NEW_YORK = ZoneInfo("America/New_York")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class NHLAutomaticResearchError(ValueError):
    """Automatic research evidence cannot be collected safely."""


@dataclass(frozen=True, slots=True)
class AutomaticResearchCapture:
    nhl_date: date
    future_games: int
    current_regular_games: int
    schedule_slot: Path
    team_slot: Path | None
    import_slot: Path | None
    prediction_publication_permitted: bool = False


def prepare_automatic_research(
    *, code_commit: str, explicit_automated_run: bool = False,
    schedule_root: Path = SCHEDULE_ROOT,
    team_root: Path = TEAM_ROOT,
    import_root: Path = IMPORT_ROOT,
    transport: Callable = urlopen,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> AutomaticResearchCapture:
    """One official-date schedule GET, then one CSV GET only if games are future."""
    observed = now()
    if (explicit_automated_run is not True
            or not isinstance(observed, datetime)
            or observed.tzinfo is None
            or observed.utcoffset() != timezone.utc.utcoffset(observed)
            or not isinstance(code_commit, str)
            or not COMMIT.fullmatch(code_commit)):
        raise NHLAutomaticResearchError("Mode, heure ou commit de recherche invalide.")
    target = observed.astimezone(NEW_YORK).date()
    schedule = capture_public_schedule(
        target, explicit_automated_run=True,
        root=schedule_root, transport=transport, now=now,
    )
    if not schedule.game_count:
        return AutomaticResearchCapture(target, 0, 0, schedule.path, None, None)
    team = capture_team_history(
        explicit_automated_run=True, transport=transport,
        code_commit=code_commit, root=team_root, now=now,
    )
    season = target.year if target.month >= 7 else target.year - 1
    imported = import_current_season_capture(
        team.path, season=season, explicit_automated_run=True,
        root=import_root, capture_root=team_root, now=now,
    )
    return AutomaticResearchCapture(
        target, schedule.game_count, imported.regular_game_count,
        schedule.path, team.path, imported.path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture NHL automatique pour recherche privée uniquement"
    )
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    result = prepare_automatic_research(
        code_commit=args.code_commit, explicit_automated_run=True,
    )
    print(
        f"NHL_RESEARCH_CAPTURE nhl_date={result.nhl_date} "
        f"future_games={result.future_games} "
        f"prior_regular_games={result.current_regular_games} "
        "publication_permitted=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
