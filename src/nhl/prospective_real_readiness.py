"""NHL-26: read-only prospective audit of verified real-source evidence.

Never redate a capture or turn a descriptive team form into a prediction.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from .contracts import require_utc
from .current_season_import import (
    CurrentSeasonImport, assess_current_season_before_game,
    verify_current_season_import,
)
from .current_season_form import summarize_current_season_pregame_form
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
        "multi_import_rule": "latest_verified_import_effective_at_or_before_each_game_cutoff",
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


def _select_asof_import(
    imports: tuple[CurrentSeasonImport, ...], *, season: int, cutoff: datetime,
) -> CurrentSeasonImport | None:
    """Use the newest import that genuinely existed before this game's cutoff."""
    if any(imported.season != season for imported in imports):
        raise NHLProspectiveReadinessError("Import saison courante incompatible.")
    eligible = [
        imported for imported in imports
        if (imported.source_observed_at_utc <= cutoff
            and imported.imported_at_utc <= cutoff
            and imported.effective_available_at_utc <= cutoff)
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda imported: (
        imported.effective_available_at_utc,
        imported.imported_at_utc,
        imported.source_observed_at_utc,
        imported.path.name,
    ))


def audit_real_pregame_readiness(
    schedule_slot: Path, *, lead_minutes: int = 120,
    min_games_per_team: int = 5, window_games: int = 10,
    database_path: Path = DATABASE_PATH, allowed_root: Path = NHL_DATA_ROOT,
    current_season_import_slot: Path | None = None,
    current_season_import_slots: tuple[Path, ...] | None = None,
) -> dict[str, object]:
    """Audit one sealed schedule and the as-of real-source history, without I/O writes.

    A historical sample, even with current-season rows, remains descriptive;
    it never proves retrospective availability or authorizes training.
    """
    _verify_protocol()
    if (type(min_games_per_team) is not int or min_games_per_team < 1
            or type(window_games) is not int or window_games < min_games_per_team):
        raise NHLProspectiveReadinessError("Fenêtre ou échantillon minimal invalide.")
    if current_season_import_slot is not None and current_season_import_slots is not None:
        raise NHLProspectiveReadinessError("Choisir un seul mode d'import saison courante.")
    if current_season_import_slots is not None and not isinstance(current_season_import_slots, tuple):
        raise NHLProspectiveReadinessError("Liste d'imports invalide.")
    schedule = audit_schedule_capture(Path(schedule_slot))
    current_import = (verify_current_season_import(Path(current_season_import_slot))
                      if current_season_import_slot is not None else None)
    verified_imports = tuple(
        verify_current_season_import(Path(slot)) for slot in (current_season_import_slots or ())
    )
    if len({imported.path.resolve() for imported in verified_imports}) != len(verified_imports):
        raise NHLProspectiveReadinessError("Import saison courante dupliqué.")
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
        selected_import = (_select_asof_import(verified_imports, season=season, cutoff=cutoff)
                           if current_season_import_slots is not None else current_import)
        row: dict[str, object] = {
            "game_id": item["game_id"],
            "away_abbr": item["away_abbr"],
            "home_abbr": item["home_abbr"],
            "scheduled_start_utc": item["scheduled_start_utc"],
            "information_cutoff_utc": cutoff.isoformat().replace("+00:00", "Z"),
            "schedule_before_cutoff": observed <= cutoff,
            "current_season": season,
        }
        if current_season_import_slots is not None:
            row["current_season_import_selection"] = (
                "LATEST_BEFORE_CUTOFF" if selected_import is not None else "NO_IMPORT_BEFORE_CUTOFF"
            )
        current_assessment = None
        if selected_import is not None:
            if selected_import.season != season:
                raise NHLProspectiveReadinessError("Import saison courante incompatible.")
            scheduled = ScheduledGame(item["game_id"], item["season"], start,
                                      item["away_abbr"], item["home_abbr"])
            current_assessment = assess_current_season_before_game(
                selected_import, scheduled, lead_minutes=lead_minutes,
            )
            row.update({
                "current_season_import_id": current_assessment["current_season_import_id"],
                "current_season_import_effective_at_utc": current_assessment["current_season_import_effective_at_utc"],
                "current_season_import_before_cutoff": current_assessment["current_season_import_before_cutoff"],
                "away_current_season_imported_games": current_assessment["prior_regular_games_by_team"][item["away_abbr"]],
                "home_current_season_imported_games": current_assessment["prior_regular_games_by_team"][item["home_abbr"]],
            })
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
                joined = None
                if (selected_import is not None
                        and current_assessment["current_season_import_before_cutoff"]):
                    joined = summarize_current_season_pregame_form(
                        history, selected_import, game, window_games=window_games,
                        min_games_per_team=min_games_per_team,
                    )
                    form = joined.form
                current_away = sum(game_id // 1_000_000 == season
                                   for game_id in form.away.source_game_ids)
                current_home = sum(game_id // 1_000_000 == season
                                   for game_id in form.home.source_game_ids)
                row.update({
                    "history_capture_id": form.capture_id,
                    "history_response_sha256": form.response_sha256,
                    "history_effective_available_at_utc": history.effective_available_at_utc.isoformat(),
                    "form_effective_available_at_utc": form.effective_available_at_utc.isoformat(),
                    "feature_sha256": joined.feature_sha256 if joined else form.feature_sha256,
                    "current_season_form_joined": joined is not None,
                    "away_complete_games": form.away.complete_games,
                    "home_complete_games": form.home.complete_games,
                    "away_current_season_games": current_away,
                    "home_current_season_games": current_home,
                    "minimum_sample_reached": form.minimum_sample_reached,
                    "reference_coverage_complete": form.regular_coverage_complete,
                })
                row["status"] = (
                    ("INSUFFICIENT_DESCRIPTIVE_SAMPLE" if joined is not None
                     else "INSUFFICIENT_HISTORICAL_SAMPLE") if not form.minimum_sample_reached
                    else "CURRENT_SEASON_TEAM_COVERAGE_MISSING"
                    if joined is not None and (not current_away or not current_home)
                    else "CURRENT_SEASON_HISTORY_NOT_IMPORTED"
                    if not current_away or not current_home
                    else "CURRENT_SEASON_FORM_DESCRIPTIVE_ONLY" if joined is not None
                    else "DESCRIPTIVE_FORM_ONLY"
                )
        if current_assessment is not None and row["status"] == "CURRENT_SEASON_HISTORY_NOT_IMPORTED":
            if not current_assessment["current_season_import_before_cutoff"]:
                row["status"] = "CURRENT_SEASON_IMPORT_AFTER_CUTOFF"
        elif (current_season_import_slots is not None and verified_imports
              and selected_import is None and row["status"] == "CURRENT_SEASON_HISTORY_NOT_IMPORTED"):
            row["status"] = "CURRENT_SEASON_IMPORT_AFTER_CUTOFF"
        games.append(row)
    return {
        "schema_version": "nhl26_prospective_readiness_v1",
        "status": ("AUDIT_ONLY_USER_SUPPLIED_COPY"
                   if schedule.get("acquisition_mode") == "user_supplied_browser_copy"
                   else "AUDIT_ONLY_NOT_MODEL_ELIGIBLE"),
        "target_date": schedule["target_date"],
        "schedule_response_sha256": schedule["response_sha256"],
        "schedule_acquisition_mode": schedule.get("acquisition_mode", "direct_https"),
        "schedule_observed_at_utc": schedule["observed_at_utc"],
        "current_season_import_verified": current_import is not None or bool(verified_imports),
        "current_season_import_id": current_import.path.name if current_import is not None else None,
        "current_season_import_count": len(verified_imports) if current_season_import_slots is not None
                                       else int(current_import is not None),
        "lead_minutes": lead_minutes,
        "window_games": window_games,
        "min_games_per_team": min_games_per_team,
        "game_count": len(games),
        "descriptive_form_count": sum(row["status"] in {"DESCRIPTIVE_FORM_ONLY",
                                                  "CURRENT_SEASON_FORM_DESCRIPTIVE_ONLY"} for row in games),
        "games": games,
        "historical_backtest_asof_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }
