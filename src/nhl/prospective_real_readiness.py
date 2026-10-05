"""NHL-26: read-only prospective audit of verified real-source evidence.

Never redate a capture or turn a descriptive team form into a prediction.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from .contracts import require_utc
from .database import NHL_DATA_ROOT, PROJECT_ROOT
from .pregame_team_form import summarize_pregame_team_form
from .public_schedule_candidates import ScheduledGame
from .schedule_capture_readiness import audit_schedule_capture
from .team_history_store import (
    DATABASE_PATH, NHLHistoryStoreError, load_pregame_team_history,
    verify_history_database,
)
from .temporal_policy import build_information_cutoff

POLICY_PATH = PROJECT_ROOT / "nhl_protocols/data/nhl26_prospective_readiness_v1.json"


class NHLProspectiveReadinessError(ValueError):
    """Prospective evidence cannot safely be audited."""


def _verify_protocol() -> None:
    try:
        document = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise NHLProspectiveReadinessError("Protocole prospectif NHL illisible.") from error
    expected = {
        "schema_version": "nhl26_prospective_readiness_v1",
        "source_database": "data/nhl/team_history.db",
        "capture_mode": "manual_only",
        "report_mode": "read_only",
        "historical_backtest_asof_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
        "odds_ingestion": False,
        "mlb_database_mutation": False,
    }
    if not isinstance(document, dict) or any(document.get(key) != value for key, value in expected.items()):
        raise NHLProspectiveReadinessError("Protocole prospectif NHL incompatible.")


def _utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise NHLProspectiveReadinessError("Horodatage de capture non canonique.")
    try:
        return require_utc(datetime.fromisoformat(value.replace("Z", "+00:00")),
                           field_name="capture NHL")
    except ValueError as error:
        raise NHLProspectiveReadinessError("Horodatage de capture invalide.") from error


def audit_real_pregame_readiness(
    schedule_slot: Path, *, lead_minutes: int = 120,
    min_games_per_team: int = 5, window_games: int = 10,
    database_path: Path = DATABASE_PATH, allowed_root: Path = NHL_DATA_ROOT,
) -> dict[str, object]:
    """Audit one sealed schedule and the as-of real-source history, without I/O writes.

    A historical sample, even with current-season rows, remains descriptive;
    it never proves retrospective availability or authorizes training.
    """
    _verify_protocol()
    if (type(min_games_per_team) is not int or min_games_per_team < 1
            or type(window_games) is not int or window_games < min_games_per_team):
        raise NHLProspectiveReadinessError("Fenêtre ou échantillon minimal invalide.")
    schedule = audit_schedule_capture(Path(schedule_slot))
    observed = _utc(schedule["observed_at_utc"])
    database = Path(database_path)
    has_database = database.is_file()
    if has_database:
        verify_history_database(database, allowed_root=allowed_root)

    games: list[dict[str, object]] = []
    for item in schedule["games"]:
        start = _utc(item["scheduled_start_utc"])
        cutoff = build_information_cutoff(start, lead_minutes=lead_minutes)
        season = item["season"] // 10_000
        if (type(item["season"]) is not int
                or item["season"] % 10_000 != season + 1):
            raise NHLProspectiveReadinessError("Saison cible NHL incohérente.")
        row: dict[str, object] = {
            "game_id": item["game_id"],
            "away_abbr": item["away_abbr"],
            "home_abbr": item["home_abbr"],
            "scheduled_start_utc": item["scheduled_start_utc"],
            "information_cutoff_utc": cutoff.isoformat().replace("+00:00", "Z"),
            "schedule_before_cutoff": observed <= cutoff,
            "current_season": season,
        }
        if observed > cutoff:
            row["status"] = "SCHEDULE_AFTER_CUTOFF"
        elif not has_database:
            row["status"] = "NO_REAL_HISTORY_DATABASE"
        else:
            game = ScheduledGame(item["game_id"], item["season"], start,
                                 item["away_abbr"], item["home_abbr"])
            try:
                history = load_pregame_team_history(
                    game, schedule_observed_at_utc=observed,
                    lead_minutes=lead_minutes, database_path=database,
                    allowed_root=allowed_root,
                )
            except NHLHistoryStoreError as error:
                if str(error) != "Aucune capture historique avant le cutoff.":
                    raise
                row["status"] = "NO_PRE_CUTOFF_HISTORY_CAPTURE"
            else:
                form = summarize_pregame_team_form(
                    history, away_abbr=game.away_abbr, home_abbr=game.home_abbr,
                    target_date=game.start_utc.date(), window_games=window_games,
                    min_games_per_team=min_games_per_team,
                )
                current_away = sum(game_id // 1_000_000 == season
                                   for game_id in form.away.source_game_ids)
                current_home = sum(game_id // 1_000_000 == season
                                   for game_id in form.home.source_game_ids)
                row.update({
                    "history_capture_id": form.capture_id,
                    "history_response_sha256": form.response_sha256,
                    "history_effective_available_at_utc": form.effective_available_at_utc.isoformat(),
                    "feature_sha256": form.feature_sha256,
                    "away_complete_games": form.away.complete_games,
                    "home_complete_games": form.home.complete_games,
                    "away_current_season_games": current_away,
                    "home_current_season_games": current_home,
                    "minimum_sample_reached": form.minimum_sample_reached,
                    "reference_coverage_complete": form.regular_coverage_complete,
                })
                row["status"] = (
                    "INSUFFICIENT_HISTORICAL_SAMPLE" if not form.minimum_sample_reached
                    else "CURRENT_SEASON_HISTORY_NOT_IMPORTED"
                    if not current_away or not current_home
                    else "DESCRIPTIVE_FORM_ONLY"
                )
        games.append(row)
    return {
        "schema_version": "nhl26_prospective_readiness_v1",
        "status": "AUDIT_ONLY_NOT_MODEL_ELIGIBLE",
        "target_date": schedule["target_date"],
        "schedule_response_sha256": schedule["response_sha256"],
        "schedule_observed_at_utc": schedule["observed_at_utc"],
        "lead_minutes": lead_minutes,
        "window_games": window_games,
        "min_games_per_team": min_games_per_team,
        "game_count": len(games),
        "descriptive_form_count": sum(row["status"] == "DESCRIPTIVE_FORM_ONLY" for row in games),
        "games": games,
        "historical_backtest_asof_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }
