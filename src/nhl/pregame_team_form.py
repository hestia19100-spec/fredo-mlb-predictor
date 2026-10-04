"""NHL-25: descriptive pre-game team form from verified real-source captures.

The window is a read-only observation aid. It never authorizes backtesting,
training, probability estimation, or publication of a prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_EVEN
from hashlib import sha256
import json
from pathlib import Path

from .contracts import require_utc
from .database import NHL_DATA_ROOT, PROJECT_ROOT
from .moneypuck_team_import import TeamGameRow
from .schedule_capture_readiness import audit_schedule_capture
from .team_history_store import (
    DATABASE_PATH, PregameTeamHistory, load_verified_schedule_team_history,
)

_PRECISION = Decimal("0.0001")
POLICY_PATH = PROJECT_ROOT / "nhl_protocols/data/nhl25_pregame_team_form_v1.json"


class NHLTeamFormError(ValueError):
    """Historical rows cannot safely describe this scheduled match."""


def _verify_protocol() -> None:
    try:
        document = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise NHLTeamFormError("Protocole NHL-25 illisible.") from error
    expected = {
        "schema_version": "nhl25_pregame_team_form_v1",
        "purpose": "descriptive_pregame_team_form_only",
        "source_database": "data/nhl/team_history.db",
        "availability_rule": "max(capture_observed_at_utc,imported_at_utc)<=information_cutoff_utc",
        "schedule_rule": "verified_schedule_observed_at_utc<=information_cutoff_utc",
        "historical_backtest_asof_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
        "odds_ingestion": False,
        "mlb_database_mutation": False,
        "synthetic_nhl_journal_mutation": False,
    }
    if not isinstance(document, dict) or any(document.get(key) != value for key, value in expected.items()):
        raise NHLTeamFormError("Protocole NHL-25 incompatible.")


@dataclass(frozen=True, slots=True)
class TeamForm:
    team: str
    source_game_ids: tuple[int, ...]
    complete_games: int
    incomplete_game_ids: tuple[int, ...]
    goals_for_mean: Decimal | None
    goals_against_mean: Decimal | None
    x_goals_for_mean: Decimal | None
    x_goals_against_mean: Decimal | None
    five_on_five_x_goals_for_mean: Decimal | None
    five_on_five_x_goals_against_mean: Decimal | None


@dataclass(frozen=True, slots=True)
class PregameTeamForm:
    target_game_id: int
    information_cutoff_utc: datetime
    capture_id: str
    response_sha256: str
    effective_available_at_utc: datetime
    away: TeamForm
    home: TeamForm
    window_games: int
    min_games_per_team: int
    minimum_sample_reached: bool
    regular_coverage_complete: bool
    feature_sha256: str
    historical_backtest_asof_proven: bool = False
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def _mean(rows: list[TeamGameRow], field: str) -> Decimal | None:
    if not rows:
        return None
    values = [Decimal(getattr(row, field)) for row in rows]
    return (sum(values) / len(values)).quantize(_PRECISION, rounding=ROUND_HALF_EVEN)


def _team_form(rows: tuple[TeamGameRow, ...], *, team: str,
               target_game_id: int, target_date: date, window_games: int) -> TeamForm:
    by_game: dict[int, dict[str, TeamGameRow]] = {}
    for row in rows:
        if not isinstance(row, TeamGameRow) or row.team != team:
            raise NHLTeamFormError("Ligne historique d'une autre équipe.")
        if row.game_id == target_game_id or row.game_date >= target_date:
            raise NHLTeamFormError("Statistique du match cible ou postérieure.")
        if row.situation not in {"all", "5on5"}:
            raise NHLTeamFormError("Situation historique inattendue.")
        situations = by_game.setdefault(row.game_id, {})
        if row.situation in situations:
            raise NHLTeamFormError("Ligne historique dupliquée.")
        situations[row.situation] = row
    complete: list[tuple[date, int, TeamGameRow, TeamGameRow]] = []
    incomplete: list[int] = []
    for game_id, situations in by_game.items():
        if set(situations) != {"all", "5on5"}:
            incomplete.append(game_id)
            continue
        all_row, five_row = situations["all"], situations["5on5"]
        if (all_row.game_date != five_row.game_date or all_row.opponent != five_row.opponent
                or all_row.home_or_away != five_row.home_or_away):
            raise NHLTeamFormError("Situations contradictoires pour un match source.")
        complete.append((all_row.game_date, game_id, all_row, five_row))
    complete.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = complete[:window_games]
    all_rows = [item[2] for item in chosen]
    five_rows = [item[3] for item in chosen]
    return TeamForm(
        team=team, source_game_ids=tuple(item[1] for item in chosen),
        complete_games=len(chosen), incomplete_game_ids=tuple(sorted(incomplete)),
        goals_for_mean=_mean(all_rows, "goals_for"),
        goals_against_mean=_mean(all_rows, "goals_against"),
        x_goals_for_mean=_mean(all_rows, "x_goals_for"),
        x_goals_against_mean=_mean(all_rows, "x_goals_against"),
        five_on_five_x_goals_for_mean=_mean(five_rows, "x_goals_for"),
        five_on_five_x_goals_against_mean=_mean(five_rows, "x_goals_against"),
    )


def summarize_pregame_team_form(history: PregameTeamHistory, *,
                                away_abbr: str, home_abbr: str,
                                target_date: date, window_games: int = 10,
                                min_games_per_team: int = 5) -> PregameTeamForm:
    """Summarize an already verified as-of history without creating a pick."""
    _verify_protocol()
    if (type(window_games) is not int or window_games < 1
            or type(min_games_per_team) is not int or not 1 <= min_games_per_team <= window_games):
        raise NHLTeamFormError("Fenêtre ou seuil de matchs invalides.")
    if not isinstance(history, PregameTeamHistory):
        raise NHLTeamFormError("Historique pré-match vérifié requis.")
    require_utc(history.information_cutoff_utc, field_name="information_cutoff_utc")
    require_utc(history.effective_available_at_utc, field_name="effective_available_at_utc")
    if (history.effective_available_at_utc > history.information_cutoff_utc
            or history.training_permitted or history.prediction_publication_permitted
            or history.historical_backtest_asof_proven):
        raise NHLTeamFormError("Historique non admissible à l'observation NHL-25.")
    if (not away_abbr or not home_abbr or away_abbr == home_abbr
            or type(target_date) is not date):
        raise NHLTeamFormError("Équipes ou date cible invalides.")
    away = _team_form(history.away_rows, team=away_abbr,
                      target_game_id=history.target_game_id,
                      target_date=target_date, window_games=window_games)
    home = _team_form(history.home_rows, team=home_abbr,
                      target_game_id=history.target_game_id,
                      target_date=target_date, window_games=window_games)
    def serial(team_form: TeamForm) -> dict[str, object]:
        return {name: str(value) if isinstance(value, Decimal) else value
                for field in fields(team_form) for name, value in [(field.name, getattr(team_form, field.name))]}
    proof = {
        "schema": "nhl25_pregame_team_form_v1",
        "target_game_id": history.target_game_id,
        "information_cutoff_utc": history.information_cutoff_utc.isoformat(),
        "capture_id": history.capture_id,
        "response_sha256": history.response_sha256,
        "effective_available_at_utc": history.effective_available_at_utc.isoformat(),
        "target_date": target_date.isoformat(),
        "window_games": window_games,
        "min_games_per_team": min_games_per_team,
        "away": serial(away), "home": serial(home),
    }
    digest = sha256(json.dumps(proof, sort_keys=True, separators=(",", ":"))
                    .encode("utf-8")).hexdigest()
    return PregameTeamForm(
        target_game_id=history.target_game_id,
        information_cutoff_utc=history.information_cutoff_utc,
        capture_id=history.capture_id, response_sha256=history.response_sha256,
        effective_available_at_utc=history.effective_available_at_utc,
        away=away, home=home, window_games=window_games,
        min_games_per_team=min_games_per_team,
        minimum_sample_reached=(away.complete_games >= min_games_per_team
                                and home.complete_games >= min_games_per_team),
        regular_coverage_complete=history.regular_coverage_complete,
        feature_sha256=digest,
    )


def load_verified_schedule_team_form(schedule_slot: Path, *, target_game_id: int,
                                     lead_minutes: int, window_games: int = 10,
                                     min_games_per_team: int = 5,
                                     database_path: Path = DATABASE_PATH,
                                     allowed_root: Path = NHL_DATA_ROOT) -> PregameTeamForm:
    """Public entry point: verify the schedule and historical capture at cutoff."""
    schedule = audit_schedule_capture(schedule_slot)
    matches = [game for game in schedule["games"] if game["game_id"] == target_game_id]
    if len(matches) != 1:
        raise NHLTeamFormError("Match cible absent du calendrier scellé.")
    game = matches[0]
    history = load_verified_schedule_team_history(
        schedule_slot, target_game_id=target_game_id, lead_minutes=lead_minutes,
        database_path=database_path, allowed_root=allowed_root,
    )
    start = datetime.fromisoformat(game["scheduled_start_utc"].replace("Z", "+00:00"))
    return summarize_pregame_team_form(
        history, away_abbr=game["away_abbr"], home_abbr=game["home_abbr"],
        target_date=start.date(), window_games=window_games,
        min_games_per_team=min_games_per_team,
    )
