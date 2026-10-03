"""Retrospective, score-only NHL form candidates from pinned public releases.

Only results from earlier calendar dates may enter a feature. These rows are
not historical pregame snapshots and never authorize model training.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping

from .public_schedule_reconciliation import (
    MAX_ARCHIVE_BYTES, RELEASE_SEASONS, REQUIRED_COLUMNS, TEAM_CODE,
    PublicScheduleFinal, reconcile_public_schedules,
)


class PublicScheduleFormError(ValueError):
    """Pinned release or historical game rows are inconsistent."""


@dataclass(frozen=True, slots=True)
class PriorScoreForm:
    game_id: int
    source_season: int
    game_date: date
    home_team: str
    away_team: str
    home_prior_games: int
    away_prior_games: int
    home_prior_wins: int
    away_prior_wins: int
    home_prior_goal_diff: int
    away_prior_goal_diff: int


@dataclass(frozen=True, slots=True)
class PublicScheduleFormAudit:
    observed_at_utc: datetime
    nhl15_report_sha256: str
    nhl17_manifest_sha256: str
    source_archive_sha256: tuple[tuple[int, str], ...]
    forms: tuple[PriorScoreForm, ...]
    lookback_games: int

    def summary(self) -> dict[str, object]:
        rows = [(r.game_id, r.home_prior_games, r.away_prior_games,
                 r.home_prior_wins, r.away_prior_wins,
                 r.home_prior_goal_diff, r.away_prior_goal_diff)
                for r in self.forms]
        return {
            "schema_version": "nhl18_prior_score_form_v1",
            "observed_at_utc": self.observed_at_utc.isoformat(),
            "source": "SportsDataverse processed NHL schedule release",
            "nhl15_report_sha256": self.nhl15_report_sha256,
            "nhl17_manifest_sha256": self.nhl17_manifest_sha256,
            "source_is_historical_pregame_snapshot": False,
            "release_archive_sha256": {
                str(season): digest for season, digest in self.source_archive_sha256
            },
            "regular_final_games": len(self.forms),
            "regular_final_games_by_season": {
                str(season): sum(r.source_season == season for r in self.forms)
                for season in sorted(RELEASE_SEASONS)
            },
            "lookback_games": self.lookback_games,
            "form_rows_sha256": hashlib.sha256(
                json.dumps(rows, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "cold_start_rows": sum(
                r.home_prior_games == 0 or r.away_prior_games == 0
                for r in self.forms
            ),
            "features_contain_target_score": False,
            "same_date_results_used_as_priors": False,
            "postseason_included": False,
            "historical_pregame_availability_proven": False,
            "training_permitted": False,
            "prediction_publication_permitted": False,
        }


def _uint(value: object, field: str) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise PublicScheduleFormError(f"Invalid {field}")
    return int(value)


def _parse_regular_finals(season: int, raw: bytes) -> tuple[PublicScheduleFinal, ...]:
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), strict=True)
        columns = reader.fieldnames
        if (columns is None or len(columns) != len(set(columns))
                or not REQUIRED_COLUMNS.issubset(columns)):
            raise PublicScheduleFormError(f"Invalid release columns: {season}")
        games: dict[int, PublicScheduleFinal] = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise PublicScheduleFormError(f"Malformed row: {season}")
            if row["game_type"] != "R":
                continue
            game_id = _uint(row["game_id"], "game_id")
            if game_id in games:
                raise PublicScheduleFormError(f"Duplicate regular game: {game_id}")
            if (game_id // 1_000_000 != season - 1
                    or _uint(row["season"], "season") not in (season - 1, season)
                    or row["game_state"] != "OFF"):
                raise PublicScheduleFormError(f"Non-final or wrong-season game: {game_id}")
            try:
                game_date = date.fromisoformat(row["game_date"])
            except ValueError as error:
                raise PublicScheduleFormError(f"Invalid game date: {game_id}") from error
            if game_date.year not in (season - 1, season):
                raise PublicScheduleFormError(f"Wrong game date season: {game_id}")
            home, away = row["home_team_abbr"], row["away_team_abbr"]
            if not TEAM_CODE.fullmatch(home) or not TEAM_CODE.fullmatch(away) or home == away:
                raise PublicScheduleFormError(f"Invalid teams: {game_id}")
            home_score = _uint(row["home_score"], "home_score")
            away_score = _uint(row["away_score"], "away_score")
            if home_score == away_score:
                raise PublicScheduleFormError(f"No decisive final score: {game_id}")
            games[game_id] = PublicScheduleFinal(
                game_id, season, game_date, home, away, home_score,
                away_score, home if home_score > away_score else away)
    except (UnicodeDecodeError, csv.Error) as error:
        raise PublicScheduleFormError(f"Invalid CSV: {season}") from error
    if len(games) != 1312:
        raise PublicScheduleFormError(f"Incomplete regular season {season}: {len(games)}")
    return tuple(games[game_id] for game_id in sorted(games))


def build_public_schedule_form(
    nhl15_report_bytes: bytes,
    nhl17_manifest_bytes: bytes,
    archives_by_season: Mapping[int, bytes],
    *,
    observed_at_utc: datetime,
    lookback_games: int = 5,
) -> PublicScheduleFormAudit:
    """Build date-lagged candidates from five hash-pinned release assets."""
    if type(lookback_games) is not int or not 1 <= lookback_games <= 30:
        raise PublicScheduleFormError("Lookback must be between 1 and 30 games")
    if set(archives_by_season) != RELEASE_SEASONS:
        raise PublicScheduleFormError("Exactly five release seasons are required")
    try:
        manifest = json.loads(nhl17_manifest_bytes)
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        raise PublicScheduleFormError("Invalid NHL-17 manifest") from error
    if (not isinstance(manifest, dict)
            or manifest.get("schema_version") != "nhl17_public_schedule_reconciliation_v1"
            or manifest.get("training_permitted") is not False
            or manifest.get("prediction_publication_permitted") is not False
            or manifest.get("nhl15_report_sha256")
            != hashlib.sha256(nhl15_report_bytes).hexdigest()):
        raise PublicScheduleFormError("NHL-17 manifest or gates differ")
    assets = manifest.get("archives")
    if (not isinstance(assets, list) or len(assets) != 5
            or {a.get("season") for a in assets if isinstance(a, dict)}
            != RELEASE_SEASONS):
        raise PublicScheduleFormError("Invalid NHL-17 archive manifest")
    expected = {asset["season"]: asset["sha256"] for asset in assets}
    digests: list[tuple[int, str]] = []
    for season in sorted(RELEASE_SEASONS):
        raw = archives_by_season[season]
        if type(raw) is not bytes or not 0 < len(raw) <= MAX_ARCHIVE_BYTES:
            raise PublicScheduleFormError(f"Invalid archive bytes: {season}")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected[season]:
            raise PublicScheduleFormError(f"Release digest changed: {season}")
        digests.append((season, digest))
    tied = reconcile_public_schedules(
        nhl15_report_bytes, archives_by_season, observed_at_utc=observed_at_utc)
    tied_summary = tied.summary()
    if (tied_summary["resolved_count"] != manifest.get("resolved_count")
            or tied_summary["resolved_ids_sha256"] != manifest.get("resolved_ids_sha256")
            or tied_summary["outcomes_sha256"] != manifest.get("outcomes_sha256")):
        raise PublicScheduleFormError("NHL-17 reconciliation digest differs")
    games = tuple(game for season in sorted(RELEASE_SEASONS)
                  for game in _parse_regular_finals(season, archives_by_season[season]))
    by_day: dict[tuple[int, date], list[PublicScheduleFinal]] = defaultdict(list)
    for game in games:
        by_day[game.source_season, game.game_date].append(game)
    prior: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    forms: list[PriorScoreForm] = []
    for season, game_date in sorted(by_day):
        day = sorted(by_day[season, game_date], key=lambda game: game.game_id)
        for game in day:
            home = prior[season, game.home_team][-lookback_games:]
            away = prior[season, game.away_team][-lookback_games:]
            forms.append(PriorScoreForm(
                game.game_id, season, game_date, game.home_team, game.away_team,
                len(home), len(away), sum(win for win, _ in home),
                sum(win for win, _ in away), sum(diff for _, diff in home),
                sum(diff for _, diff in away)))
        for game in day:
            diff = game.home_score - game.away_score
            prior[season, game.home_team].append((int(diff > 0), diff))
            prior[season, game.away_team].append((int(diff < 0), -diff))
    return PublicScheduleFormAudit(
        tied.observed_at_utc,
        hashlib.sha256(nhl15_report_bytes).hexdigest(),
        hashlib.sha256(nhl17_manifest_bytes).hexdigest(),
        tuple(digests),
        tuple(sorted(forms, key=lambda row: row.game_id)), lookback_games)
