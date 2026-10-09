"""Explicit NHL-41 command-line network adapter; never scheduled automatically."""
from __future__ import annotations

import argparse
from datetime import date
from http.client import HTTPException
import json
import subprocess
from urllib.request import urlopen

from src.nhl.database import PROJECT_ROOT
from src.nhl.manual_prospective_day import run_manual_prospective_day


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture NHL manuelle, sans modèle ni pari.")
    parser.add_argument("--date", required=True, type=date.fromisoformat)
    parser.add_argument("--season", required=True, type=int, help="année de début, ex. 2026")
    parser.add_argument("--manual", required=True, action="store_true")
    args = parser.parse_args(argv)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip()
        report = run_manual_prospective_day(
            args.date, season=args.season, code_commit=commit,
            explicit_manual_run=args.manual, team_transport=urlopen,
            schedule_transport=urlopen,
        )
    except (OSError, subprocess.CalledProcessError, ValueError, HTTPException) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0 if report["status"] == "CAPTURES_VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
