"""Collect complete team evidence once, or verify a recent immutable cached copy."""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from src.nhl.contracts import require_utc
from src.nhl.moneypuck_team_capture import (
    DEFAULT_ROOT, VerifiedTeamCapture, capture_team_history, verify_team_capture,
)

MAX_AGE = timedelta(hours=6)
PARIS = ZoneInfo('Europe/Paris')


class TeamDataCacheError(ValueError):
    """No fresh, complete, verified research data is available."""


class NoFreshTeamData(TeamDataCacheError):
    """The cache is empty or its valid contents are too old."""


def require_fresh_team_capture(root: Path, *, observed: datetime) -> VerifiedTeamCapture:
    observed = require_utc(observed, field_name='cache observation')
    root = Path(root)
    if root.is_symlink():
        raise TeamDataCacheError('Cache sous forme de lien symbolique refusé.')
    candidates = []
    # A previous UTC day can still belong to the same Paris day.
    for day in {observed.date(), (observed - timedelta(days=1)).date()}:
        for marker in sorted((root / day.isoformat()).glob('*/COMPLETED')):
            slot = marker.parent
            if (slot.is_symlink() or any((slot / name).is_symlink()
                    for name in ('COMPLETED', 'receipt.json', 'response.csv'))):
                raise TeamDataCacheError('Capture en cache non autonome.')
            try:
                capture = verify_team_capture(slot, allowed_root=root)
            except (OSError, ValueError) as error:
                raise TeamDataCacheError('Capture en cache incomplète ou altérée.') from error
            if capture.observed_at_utc > observed:
                raise TeamDataCacheError('Capture en cache datée du futur.')
            if (capture.acquisition_mode == 'automated_https_get'
                    and observed - capture.observed_at_utc <= MAX_AGE
                    and capture.observed_at_utc.astimezone(PARIS).date()
                        == observed.astimezone(PARIS).date()):
                candidates.append(capture)
    if not candidates:
        raise NoFreshTeamData(
            'Aucune capture automatique complète du jour de moins de 6 heures. '
            'Exécuter NHL verified team data refresh; aucun remplacement par une archive ancienne.')
    return max(candidates, key=lambda c: c.observed_at_utc)


def collect_or_reuse(*, root: Path = DEFAULT_ROOT, code_commit: str,
                     transport: Callable, force_refresh: bool = False,
                     now: Callable = lambda: datetime.now(timezone.utc)
                     ) -> tuple[VerifiedTeamCapture, bool]:
    try:
        existing = require_fresh_team_capture(root, observed=now())
    except NoFreshTeamData:
        pass
    else:
        if not force_refresh:
            return existing, True
    capture = capture_team_history(
        explicit_automated_run=True, transport=transport, code_commit=code_commit,
        root=root, now=now, download_attempt_seconds=900, download_max_attempts=2,
    )
    # Verify written bytes again before the workflow is permitted to save them.
    return require_fresh_team_capture(root, observed=now()), False


def materialize_capture(root: Path, destination: Path, *, observed: datetime) -> VerifiedTeamCapture:
    """Copy only the verified fresh slot; never replace frozen historical evidence."""
    capture = require_fresh_team_capture(root, observed=observed)
    destination = Path(destination)
    if destination.is_symlink():
        raise TeamDataCacheError('Destination sous forme de lien symbolique refusée.')
    target = destination / capture.path.parent.name / capture.path.name
    if target.parent.is_symlink() or target.is_symlink():
        raise TeamDataCacheError('Destination de capture non autonome.')
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(capture.path, target)
    verified = verify_team_capture(target, allowed_root=destination)
    if (verified.response_sha256 != capture.response_sha256
            or verified.observed_at_utc != capture.observed_at_utc):
        raise TeamDataCacheError('Capture matérialisée divergente; aucun écrasement autorisé.')
    return verified


def main():
    from nhl_operations.automatic_research_capture import _default_transport
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect', 'check', 'materialize'))
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--force-refresh', action='store_true')
    args = parser.parse_args()
    if args.mode == 'collect':
        capture, reused = collect_or_reuse(root=args.root, code_commit=args.code_commit,
                                         transport=_default_transport, force_refresh=args.force_refresh)
    elif args.mode == 'materialize':
        if args.destination is None:
            parser.error('--destination requis pour materialize')
        capture = materialize_capture(args.root, args.destination, observed=datetime.now(timezone.utc))
        reused = True
    else:
        capture = require_fresh_team_capture(args.root, observed=datetime.now(timezone.utc))
        reused = True
    # Keep the original source observation timestamp: cache restoration is not a new receipt.
    if output := os.environ.get('GITHUB_OUTPUT'):
        with open(output, 'a', encoding='utf-8') as stream:
            stream.write(f'reused={str(reused).lower()}\nsha256={capture.response_sha256}\n'
                         f'day={capture.observed_at_utc.astimezone(PARIS).date().isoformat()}\n')
    print(f'NHL_TEAM_DATA_VERIFIED reused={str(reused).lower()} '
          f'observed_at_utc={capture.observed_at_utc.isoformat()} '
          f'sha256={capture.response_sha256} max_age_hours=6')


if __name__ == '__main__':
    main()
