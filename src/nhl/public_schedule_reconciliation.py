"""Offline reconciliation of public SportsDataverse NHL final scores.

This processed release is not independently verified NHL gamecenter evidence.
The module never requests the network or opens training/publication gates.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Mapping

RELEASE_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/"
               "releases/download/nhl_schedules/nhl_schedule_{season}.csv")
RELEASE_SEASONS = frozenset(range(2022, 2027))
REQUIRED_COLUMNS = frozenset({"game_id", "season", "game_type", "game_date",
                              "game_state", "home_team_abbr", "away_team_abbr",
                              "home_score", "away_score"})
TEAM_CODE = re.compile(r"[A-Z]{2,3}\Z")
MAX_ARCHIVE_BYTES = 8_000_000


class PublicScheduleReconciliationError(ValueError):
    """Source evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class PublicScheduleFinal:
    game_id: int
    source_season: int
    game_date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    winner: str


@dataclass(frozen=True, slots=True)
class PublicScheduleReconciliation:
    observed_at_utc: datetime
    nhl15_report_sha256: str
    moneypuck_archive_sha256: str
    archive_sha256_by_season: tuple[tuple[int, str], ...]
    finals: tuple[PublicScheduleFinal, ...]

    @property
    def resolved_count(self) -> int:
        return len(self.finals)

    def summary(self) -> dict[str, object]:
        """Reproducible manifest without republishing bulk game rows."""
        return {
            "schema_version": "nhl17_public_schedule_reconciliation_v1",
            "source": "SportsDataverse processed NHL schedule release",
            "source_origin": "NHL public data, republished by SportsDataverse",
            "source_is_direct_nhl_gamecenter_evidence": False,
            "observed_at_utc": self.observed_at_utc.isoformat(),
            "nhl15_report_sha256": self.nhl15_report_sha256,
            "moneypuck_archive_sha256": self.moneypuck_archive_sha256,
            "archives": [
                {"season": season, "url": RELEASE_URL.format(season=season),
                 "sha256": digest}
                for season, digest in self.archive_sha256_by_season
            ],
            "resolved_count": self.resolved_count,
            "resolved_count_by_season": {
                str(season): sum(game.source_season == season for game in self.finals)
                for season in sorted(RELEASE_SEASONS)
            },
            "release_assets_may_change": True,
            "resolved_ids_sha256": hashlib.sha256(json.dumps(
                [game.game_id for game in self.finals],
                separators=(",", ":")).encode("ascii")).hexdigest(),
            "outcomes_sha256": hashlib.sha256(json.dumps(
                [(game.game_id, game.home_score, game.away_score)
                 for game in self.finals],
                separators=(",", ":")).encode("ascii")).hexdigest(),
            "historical_pregame_availability_proven": False,
            "independent_final_labels_proven": False,
            "training_permitted": False,
            "prediction_publication_permitted": False,
        }


def _integer(value: object, field: str) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise PublicScheduleReconciliationError(f"Invalid {field}")
    return int(value)


def reconcile_public_schedules(
    nhl15_report_bytes: bytes,
    archives_by_season: Mapping[int, bytes],
    *,
    observed_at_utc: datetime,
) -> PublicScheduleReconciliation:
    """Match every NHL-15 tied-source ID to a decisive final release row."""
    if (not isinstance(observed_at_utc, datetime)
            or observed_at_utc.tzinfo is None
            or observed_at_utc.utcoffset() is None):
        raise PublicScheduleReconciliationError("Observation time must be timezone-aware")
    if type(nhl15_report_bytes) is not bytes or len(nhl15_report_bytes) > MAX_ARCHIVE_BYTES:
        raise PublicScheduleReconciliationError("Invalid NHL-15 report bytes")
    try:
        report = json.loads(nhl15_report_bytes)
    except (ValueError, UnicodeDecodeError) as error:
        raise PublicScheduleReconciliationError("Invalid NHL-15 report JSON") from error
    if not isinstance(report, dict):
        raise PublicScheduleReconciliationError("Invalid NHL-15 report")
    ids = report.get("tied_score_game_ids")
    if (not isinstance(ids, list) or len(ids) != 475
            or any(type(game_id) is not int for game_id in ids)
            or len(set(ids)) != len(ids)
            or any(game_id // 1_000_000 not in range(2021, 2026)
                   for game_id in ids)):
        raise PublicScheduleReconciliationError("Invalid NHL-15 tied game IDs")
    source_sha = report.get("source_file_sha256")
    if not isinstance(source_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", source_sha):
        raise PublicScheduleReconciliationError("Invalid MoneyPuck source digest")
    if (report.get("training_permitted") is not False
            or report.get("prediction_publication_permitted") is not False):
        raise PublicScheduleReconciliationError("NHL-15 gates must remain closed")
    if set(archives_by_season) != RELEASE_SEASONS:
        raise PublicScheduleReconciliationError("Exactly five release seasons are required")

    targets = set(ids)
    found: dict[int, PublicScheduleFinal] = {}
    digests: list[tuple[int, str]] = []
    for season in sorted(RELEASE_SEASONS):
        raw = archives_by_season[season]
        if type(raw) is not bytes or not 0 < len(raw) <= MAX_ARCHIVE_BYTES:
            raise PublicScheduleReconciliationError(f"Invalid archive bytes: {season}")
        digests.append((season, hashlib.sha256(raw).hexdigest()))
        try:
            reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), strict=True)
            columns = reader.fieldnames
            if (columns is None or len(columns) != len(set(columns))
                    or not REQUIRED_COLUMNS.issubset(columns)):
                raise PublicScheduleReconciliationError(f"Invalid release columns: {season}")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise PublicScheduleReconciliationError(f"Malformed release row: {season}")
                raw_id = row["game_id"]
                if not raw_id.isascii() or not raw_id.isdecimal():
                    continue
                game_id = int(raw_id)
                if game_id not in targets:
                    continue
                if game_id in found:
                    raise PublicScheduleReconciliationError(f"Duplicate game ID: {game_id}")
                if (game_id // 1_000_000 != season - 1
                        or _integer(row["season"], "season") not in (season - 1, season)
                        or row["game_type"] != "R"
                        or row["game_state"] != "OFF"):
                    raise PublicScheduleReconciliationError(
                        f"Non-final or wrong-season game: {game_id}")
                try:
                    game_date = date.fromisoformat(row["game_date"])
                except ValueError as error:
                    raise PublicScheduleReconciliationError(
                        f"Invalid game date: {game_id}") from error
                if game_date.year not in (season - 1, season):
                    raise PublicScheduleReconciliationError(
                        f"Wrong game date season: {game_id}")
                home, away = row["home_team_abbr"], row["away_team_abbr"]
                if not TEAM_CODE.fullmatch(home) or not TEAM_CODE.fullmatch(away) or home == away:
                    raise PublicScheduleReconciliationError(
                        f"Invalid team codes: {game_id}")
                home_score = _integer(row["home_score"], "home_score")
                away_score = _integer(row["away_score"], "away_score")
                if home_score == away_score:
                    raise PublicScheduleReconciliationError(
                        f"No decisive final score: {game_id}")
                found[game_id] = PublicScheduleFinal(
                    game_id, season, game_date, home, away, home_score,
                    away_score, home if home_score > away_score else away)
        except (UnicodeDecodeError, csv.Error) as error:
            raise PublicScheduleReconciliationError(
                f"Invalid CSV archive: {season}") from error
    missing = targets - found.keys()
    if missing:
        raise PublicScheduleReconciliationError(
            f"Missing {len(missing)} target games; first ID: {min(missing)}")
    return PublicScheduleReconciliation(
        observed_at_utc.astimezone(timezone.utc),
        hashlib.sha256(nhl15_report_bytes).hexdigest(), source_sha,
        tuple(digests), tuple(found[game_id] for game_id in sorted(found)))
