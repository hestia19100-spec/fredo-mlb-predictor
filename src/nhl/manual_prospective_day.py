"""One explicit NHL capture attempt; unavailable schedule is never no games.

The command archives listed MoneyPuck team data and then attempts the official
schedule once. It neither schedules itself nor trains/publishes a model.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

from .current_season_import import (
    DEFAULT_ROOT as IMPORT_ROOT, NHLCurrentSeasonImportError,
    import_current_season_capture, verify_current_season_import,
)
from .database import PROJECT_ROOT
from .moneypuck_team_capture import (
    DEFAULT_ROOT as TEAM_ROOT, capture_team_history, verify_team_capture,
)
from .public_schedule_capture import (
    DEFAULT_ROOT as SCHEDULE_ROOT, PublicScheduleCaptureError,
    capture_public_schedule, verify_public_schedule_capture,
)

SCHEMA = "nhl41_manual_prospective_day_v1"
POLICY = PROJECT_ROOT / "nhl_protocols/data/nhl41_manual_prospective_day_v1.json"


class NHLManualProspectiveDayError(ValueError):
    """The manual run was not explicitly requested or its protocol is invalid."""


def _policy_sha256() -> str:
    try:
        raw = POLICY.read_bytes()
        policy = json.loads(raw)
    except (OSError, ValueError) as error:
        raise NHLManualProspectiveDayError("Protocole NHL-41 illisible.") from error
    expected = {
        "schema_version": SCHEMA,
        "manual_only": True,
        "automatic_daily_collection_allowed": False,
        "schedule_failure_is_not_no_games": True,
        "no_future_games_is_not_no_games_today": True,
        "training_permitted": False,
        "prediction_publication_permitted": False,
        "odds_ingestion": False,
        "mlb_database_mutation": False,
        "nhl_historical_database_mutation": False,
    }
    if not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected.items()):
        raise NHLManualProspectiveDayError("Protocole NHL-41 incompatible.")
    return sha256(raw).hexdigest()


def _season_matches_date(target_date: date, season: int) -> bool:
    if type(season) is not int or not 2026 <= season <= 2099:
        return False
    return (target_date.year == season and target_date.month >= 7) or (
        target_date.year == season + 1 and target_date.month <= 6
    )


def _http_status(error: BaseException) -> int | None:
    current: BaseException | None = error
    while current is not None:
        if type(current).__name__ == "HTTPError":
            code = getattr(current, "code", None)
            if type(code) is int and 100 <= code <= 599:
                return code
        current = current.__cause__
    return None


def run_manual_prospective_day(
    target_date: date, *, season: int, code_commit: str,
    explicit_manual_run: bool = False,
    team_transport: Callable[..., Any] | None = None,
    schedule_transport: Callable[..., Any] | None = None,
    team_root: Path = TEAM_ROOT, import_root: Path = IMPORT_ROOT,
    schedule_root: Path = SCHEDULE_ROOT,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, object]:
    """One team GET and one schedule attempt; all durable proofs stay separate.

    Import failure does not erase the team capture. Schedule failure never
    creates a zero-games receipt. The output is an operational status only.
    """
    if (explicit_manual_run is not True or not isinstance(target_date, date)
            or isinstance(target_date, datetime) or not _season_matches_date(target_date, season)
            or team_transport is None or schedule_transport is None):
        raise NHLManualProspectiveDayError("Date, saison, transports et exécution manuelle requis.")
    policy_sha = _policy_sha256()
    captured = capture_team_history(
        explicit_manual_run=True, transport=team_transport,
        code_commit=code_commit, root=team_root, now=now,
    )
    verified = verify_team_capture(captured.path, allowed_root=team_root)
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "policy_sha256": policy_sha,
        "target_date": target_date.isoformat(),
        "season": season,
        "team_capture_status": "VERIFIED",
        "team_capture_path": str(verified.path),
        "team_capture_sha256": verified.response_sha256,
        "team_capture_observed_at_utc": verified.observed_at_utc.isoformat(),
        "current_season_import_status": "NOT_ATTEMPTED",
        "current_season_import_path": None,
        "current_season_regular_games": None,
        "schedule_status": "NOT_VERIFIED",
        "schedule_path": None,
        "future_regular_games": None,
        "schedule_http_status": None,
        "historical_backtest_asof_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }
    try:
        imported = import_current_season_capture(
            verified.path, season=season, explicit_manual_run=True,
            root=import_root, capture_root=team_root, now=now,
        )
    except NHLCurrentSeasonImportError as error:
        result["current_season_import_status"] = (
            "NO_COMPLETED_REGULAR_GAMES_YET"
            if str(error) == "Aucun match régulier de saison courante."
            else "IMPORT_REJECTED"
        )
    else:
        checked = verify_current_season_import(
            imported.path, root=import_root, capture_root=team_root,
        )
        result.update({
            "current_season_import_status": "VERIFIED_DESCRIPTIVE_ONLY",
            "current_season_import_path": str(checked.path),
            "current_season_regular_games": checked.regular_game_count,
        })
    try:
        schedule = capture_public_schedule(
            target_date, explicit_manual_run=True, root=schedule_root,
            transport=schedule_transport, now=now,
        )
    except PublicScheduleCaptureError as error:
        result["schedule_http_status"] = _http_status(error)
    else:
        receipt = verify_public_schedule_capture(schedule.path)
        if (receipt["target_date"] != target_date.isoformat()
                or receipt["response_sha256"] != schedule.response_sha256
                or len(receipt["future_regular_games"]) != schedule.game_count):
            raise NHLManualProspectiveDayError("Calendrier vérifié incohérent.")
        result.update({
            "schedule_status": (
                "FUTURE_REGULAR_GAMES_VERIFIED" if schedule.game_count
                else "NO_FUTURE_REGULAR_GAMES_VERIFIED"
            ),
            "schedule_path": str(schedule.path),
            "future_regular_games": schedule.game_count,
        })
    result["status"] = (
        "CAPTURES_VERIFIED"
        if result["current_season_import_status"] == "VERIFIED_DESCRIPTIVE_ONLY"
        and result["schedule_status"] != "NOT_VERIFIED"
        else "PARTIAL_CAPTURE"
    )
    return result
