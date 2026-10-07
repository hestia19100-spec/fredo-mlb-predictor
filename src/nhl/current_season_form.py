"""NHL-29: read-only descriptive join of verified historical and current rows.

This view is not a training dataset or a prediction. The source import must
have existed before the target game's information cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json

from .current_season_import import CurrentSeasonImport
from .database import PROJECT_ROOT
from .moneypuck_team_import import TeamGameRow
from .pregame_team_form import PregameTeamForm, summarize_pregame_team_form
from .public_schedule_candidates import ScheduledGame
from .team_history_store import PregameTeamHistory

POLICY_PATH = PROJECT_ROOT / "nhl_protocols/data/nhl29_current_season_form_v1.json"


class NHLCurrentSeasonFormError(ValueError):
    """A current-season row cannot safely enter this descriptive form."""


@dataclass(frozen=True, slots=True)
class CurrentSeasonPregameForm:
    form: PregameTeamForm
    import_id: str
    import_response_sha256: str
    import_effective_available_at_utc: str
    away_current_season_games: int
    home_current_season_games: int
    feature_sha256: str
    historical_backtest_asof_proven: bool = False
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def _verify_policy() -> None:
    try:
        document = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise NHLCurrentSeasonFormError("Protocole de forme NHL-29 illisible.") from error
    expected = {
        "schema_version": "nhl29_current_season_form_v1",
        "purpose": "descriptive_pregame_team_form_only",
        "current_source": "verified_current_season_import_sidecar",
        "availability_rule": "import_effective_at_utc<=information_cutoff_utc",
        "source_game_rule": "regular_game_date<target_utc_date",
        "historical_backtest_asof_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
        "odds_ingestion": False,
        "mlb_database_mutation": False,
        "nhl_historical_database_mutation": False,
    }
    if not isinstance(document, dict) or any(document.get(k) != v for k, v in expected.items()):
        raise NHLCurrentSeasonFormError("Protocole de forme NHL-29 incompatible.")


def _merge_team_rows(
    historical: tuple[TeamGameRow, ...], imported: CurrentSeasonImport,
    game: ScheduledGame, team: str,
) -> tuple[TeamGameRow, ...]:
    rows = {(row.game_id, row.situation): row for row in historical}
    if len(rows) != len(historical):
        raise NHLCurrentSeasonFormError("Historique avec lignes dupliquees.")
    for row in imported.regular_rows:
        if row.team != team:
            continue
        if (row.season != imported.season or row.game_id // 1_000_000 != imported.season
                or row.game_id == game.game_id or row.game_date >= game.start_utc.date()):
            raise NHLCurrentSeasonFormError("Ligne de saison courante future ou incoherente.")
        key = (row.game_id, row.situation)
        previous = rows.get(key)
        if previous is not None and previous != row:
            raise NHLCurrentSeasonFormError("Sources historiques et courantes contradictoires.")
        rows[key] = row
    return tuple(sorted(rows.values(), key=lambda row: (row.game_date, row.game_id, row.situation)))


def summarize_current_season_pregame_form(
    history: PregameTeamHistory, imported: CurrentSeasonImport, game: ScheduledGame,
    *, window_games: int = 10, min_games_per_team: int = 5,
) -> CurrentSeasonPregameForm:
    """Join evidence available before the cutoff, never changing stored rows."""
    _verify_policy()
    if (not isinstance(history, PregameTeamHistory)
            or not isinstance(imported, CurrentSeasonImport)
            or not isinstance(game, ScheduledGame)
            or history.target_game_id != game.game_id
            or imported.season != game.season // 10_000
            or history.information_cutoff_utc >= game.start_utc
            or history.effective_available_at_utc > history.information_cutoff_utc
            or imported.source_observed_at_utc > history.information_cutoff_utc
            or imported.imported_at_utc > history.information_cutoff_utc
            or imported.effective_available_at_utc > history.information_cutoff_utc
            or imported.effective_available_at_utc < imported.source_observed_at_utc
            or imported.training_permitted or imported.prediction_publication_permitted
            or history.training_permitted or history.prediction_publication_permitted
            or history.historical_backtest_asof_proven):
        raise NHLCurrentSeasonFormError("Jointure NHL-29 non admissible au cutoff.")
    combined = replace(
        history,
        away_rows=_merge_team_rows(history.away_rows, imported, game, game.away_abbr),
        home_rows=_merge_team_rows(history.home_rows, imported, game, game.home_abbr),
        effective_available_at_utc=max(history.effective_available_at_utc,
                                       imported.effective_available_at_utc),
    )
    form = summarize_pregame_team_form(
        combined, away_abbr=game.away_abbr, home_abbr=game.home_abbr,
        target_date=game.start_utc.date(), window_games=window_games,
        min_games_per_team=min_games_per_team,
    )
    away_count = sum(game_id // 1_000_000 == imported.season for game_id in form.away.source_game_ids)
    home_count = sum(game_id // 1_000_000 == imported.season for game_id in form.home.source_game_ids)
    proof = {
        "schema_version": "nhl29_current_season_form_v1",
        "base_feature_sha256": form.feature_sha256,
        "import_id": imported.path.name,
        "import_response_sha256": imported.source_response_sha256,
        "import_effective_available_at_utc": imported.effective_available_at_utc.isoformat(),
        "away_current_season_games": away_count,
        "home_current_season_games": home_count,
    }
    digest = sha256(json.dumps(proof, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return CurrentSeasonPregameForm(
        form, imported.path.name, imported.source_response_sha256,
        imported.effective_available_at_utc.isoformat(), away_count, home_count, digest,
    )
