"""Research-only NHL forecast on evidence captured before a real game's cutoff.

The result is returned in memory; this module neither archives it nor exposes
subscriber picks. A separate private sealing workflow is required for a genuine
prospective performance record.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .current_season_import import (
    DEFAULT_ROOT as IMPORT_ROOT, CurrentSeasonImport, verify_current_season_import,
)
from .database import PROJECT_ROOT
from .exploratory_chronological_backtest import (
    FEATURES, SPLITS, _policy_hash as research_policy_hash,
    build_exploratory_examples,
)
from .frozen_historical_source import audit_frozen_historical_source
from .historical_final_backlog import _slots
from .historical_official_final import ROOT as FINAL_ROOT, _verify
from .moneypuck_team_capture import DEFAULT_ROOT as TEAM_ROOT, verify_team_capture
from .public_schedule_candidates import ScheduledGame
from .schedule_capture_readiness import audit_schedule_capture
from .temporal_policy import build_information_cutoff

POLICY_PATH = PROJECT_ROOT / "nhl_protocols/data/nhl_prospective_shadow_observation_v1.json"
MIN_GAMES = 5
LEAD_MINUTES = 60


class NHLProspectiveShadowError(ValueError):
    """A research forecast cannot be certified as a pregame observation."""


@dataclass(frozen=True, slots=True)
class ProspectiveShadowObservation:
    game_id: int
    scheduled_start_utc: str
    observed_at_utc: str
    information_cutoff_utc: str
    away_abbr: str
    home_abbr: str
    home_win_probability: float
    away_win_probability: float
    schedule_sha256: str
    current_import_sha256: str
    training_source_sha256: str
    training_examples_sha256: str
    feature_sha256: str
    policy_sha256: str
    observation_sha256: str
    official_tie_labels_in_training: int
    status: str = "PRIVATE_RESEARCH_SHADOW_ONLY"
    production_training_permitted: bool = False
    prediction_publication_permitted: bool = False
    betting_recommendation_permitted: bool = False


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise NHLProspectiveShadowError("Horodatage UTC requis.")
    if value.utcoffset().total_seconds() != 0:
        raise NHLProspectiveShadowError("Horodatage UTC requis.")
    return value.astimezone(timezone.utc)


def _stamp(value: datetime) -> str:
    return _utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _policy_hash() -> str:
    try:
        raw = POLICY_PATH.read_bytes()
        policy = json.loads(raw)
    except (OSError, ValueError) as error:
        raise NHLProspectiveShadowError("Protocole d'observation NHL illisible.") from error
    expected = {
        "schema_version": "nhl_prospective_shadow_observation_v1",
        "purpose": "private_prospective_observation_only",
        "lead_minutes": LEAD_MINUTES,
        "market": "two_way_including_overtime_and_shootout",
        "source": "verified_frozen_research_fit_and_verified_current_season_import",
        "training_seasons": [2021, 2022, 2023],
        "current_season_min_prior_games_per_team": MIN_GAMES,
        "prior_utc_dates_only": True,
        "source_capture_and_import_before_cutoff": True,
        "official_tie_labels_before_cutoff": True,
        "retrospective_asof_proven": False,
        "production_training_permitted": False,
        "subscriber_publication_permitted": False,
        "betting_recommendation_permitted": False,
        "network_calls_allowed": False,
        "odds_api_calls_allowed": False,
        "public_repo_observation_storage_allowed": False,
    }
    if not isinstance(policy, dict) or policy != expected:
        raise NHLProspectiveShadowError("Protocole d'observation NHL incompatible.")
    return sha256(raw).hexdigest()


def _game_from_capture(slot: Path, game_id: int) -> tuple[ScheduledGame, dict[str, object]]:
    audit = audit_schedule_capture(Path(slot))
    matches = [item for item in audit["games"] if item["game_id"] == game_id]
    if type(game_id) is not int or len(matches) != 1:
        raise NHLProspectiveShadowError("Match absent du calendrier NHL scellé.")
    item = matches[0]
    game = ScheduledGame(
        game_id=item["game_id"], season=item["season"],
        start_utc=datetime.fromisoformat(item["scheduled_start_utc"].replace("Z", "+00:00")),
        away_abbr=item["away_abbr"], home_abbr=item["home_abbr"],
    )
    return game, audit


def _current_features(imported: CurrentSeasonImport, game: ScheduledGame
                      ) -> tuple[tuple[float, ...], tuple[int, ...], tuple[int, ...]]:
    """Match the six unrounded five-game features of the retrospective fit."""
    if imported.season != game.season // 10_000:
        raise NHLProspectiveShadowError("Saison de l'import différente du calendrier.")
    by_team: dict[str, dict[int, dict[str, object]]] = {
        game.away_abbr: {}, game.home_abbr: {},
    }
    for row in imported.regular_rows:
        if row.team not in by_team:
            continue
        if row.game_id == game.game_id:
            raise NHLProspectiveShadowError("Le match cible apparaît dans les statistiques.")
        if row.game_date >= game.start_utc.date():
            continue
        bucket = by_team[row.team].setdefault(row.game_id, {})
        if row.situation in bucket:
            raise NHLProspectiveShadowError("Statistique de match dupliquée.")
        bucket[row.situation] = row

    forms: dict[str, dict[str, Decimal]] = {}
    source_ids: dict[str, tuple[int, ...]] = {}
    for team in (game.away_abbr, game.home_abbr):
        pairs = []
        for game_id, situations in by_team[team].items():
            if set(situations) != {"all", "5on5"}:
                raise NHLProspectiveShadowError("Statistiques antérieures incomplètes.")
            all_row, five = situations["all"], situations["5on5"]
            if all_row.game_date != five.game_date:
                raise NHLProspectiveShadowError("Dates de situations divergentes.")
            pairs.append((all_row.game_date, game_id, all_row, five))
        pairs.sort(key=lambda item: (item[0], item[1]), reverse=True)
        chosen = pairs[:MIN_GAMES]
        if len(chosen) != MIN_GAMES:
            raise NHLProspectiveShadowError("Cinq matchs antérieurs par équipe requis.")
        source_ids[team] = tuple(item[1] for item in chosen)
        all_rows = [item[2] for item in chosen]
        five_rows = [item[3] for item in chosen]
        forms[team] = {
            "goals_for_mean": sum((Decimal(row.goals_for) for row in all_rows), Decimal(0)) / MIN_GAMES,
            "goals_against_mean": sum((Decimal(row.goals_against) for row in all_rows), Decimal(0)) / MIN_GAMES,
            "x_goals_for_mean": sum((row.x_goals_for for row in all_rows), Decimal(0)) / MIN_GAMES,
            "x_goals_against_mean": sum((row.x_goals_against for row in all_rows), Decimal(0)) / MIN_GAMES,
            "five_on_five_x_goals_for_mean": sum((row.x_goals_for for row in five_rows), Decimal(0)) / MIN_GAMES,
            "five_on_five_x_goals_against_mean": sum((row.x_goals_against for row in five_rows), Decimal(0)) / MIN_GAMES,
        }
    vector = tuple(float(forms[game.home_abbr][name] - forms[game.away_abbr][name])
                   for name in FEATURES)
    if not all(isfinite(value) for value in vector):
        raise NHLProspectiveShadowError("Caractéristique non finie.")
    return vector, source_ids[game.away_abbr], source_ids[game.home_abbr]


def _fit_verified_research_model(available_by: datetime):
    """Fit only 2021-23 in memory, after checking every source existed by cutoff."""
    research_policy_hash()
    frozen = audit_frozen_historical_source()
    source = verify_team_capture(frozen.team_slot, allowed_root=TEAM_ROOT)
    if source.observed_at_utc > available_by:
        raise NHLProspectiveShadowError("Source d'entraînement reçue après le cutoff.")
    finals = tuple(_verify(slot, source, FINAL_ROOT) for slot in _slots(FINAL_ROOT))
    if (len(finals) != frozen.verified_official_tie_finals
            or any(item.observed_at_utc > available_by for item in finals)):
        raise NHLProspectiveShadowError("Résultats d'entraînement reçus après le cutoff.")
    examples = build_exploratory_examples(
        source.history, {item.game_id: item.winner_abbr for item in finals})
    train = sorted((item for item in examples if SPLITS[item.season] == "TRAIN"),
                   key=lambda item: item.game_id)
    if not train or {item.home_won for item in train} != {0, 1}:
        raise NHLProspectiveShadowError("Entraînement exploratoire incomplet.")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )
    model.fit(np.asarray([item.features for item in train], dtype=float),
              np.asarray([item.home_won for item in train], dtype=int))
    train_hash = _digest([(item.game_id, item.features, item.home_won) for item in train])
    return model, frozen.source_capture_sha256, train_hash, sum(
        item.label_source == "OFFICIAL_TIE" for item in train)


def build_prospective_shadow_observation(
    schedule_slot: Path, current_import_slot: Path, *, target_game_id: int,
    explicit_shadow_run: bool = False,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    import_root: Path = IMPORT_ROOT,
    capture_root: Path = TEAM_ROOT,
) -> ProspectiveShadowObservation:
    """No network or writes; reject late, incomplete, or fabricated pregame evidence."""
    if explicit_shadow_run is not True:
        raise NHLProspectiveShadowError("Exécution d'observation explicite requise.")
    policy_sha = _policy_hash()
    observed = _utc(now())
    game, schedule = _game_from_capture(schedule_slot, target_game_id)
    cutoff = build_information_cutoff(_utc(game.start_utc), lead_minutes=LEAD_MINUTES)
    schedule_observed = _utc(datetime.fromisoformat(
        schedule["observed_at_utc"].replace("Z", "+00:00")))
    if (observed > cutoff or schedule_observed > observed
            or schedule_observed > cutoff or game.start_utc <= observed):
        raise NHLProspectiveShadowError("Calendrier ou exécution après la limite pré-match.")
    imported = verify_current_season_import(
        current_import_slot, root=import_root, capture_root=capture_root)
    if (imported.source_observed_at_utc > observed
            or imported.imported_at_utc > observed
            or imported.effective_available_at_utc > cutoff):
        raise NHLProspectiveShadowError("Statistiques reçues après la limite pré-match.")
    vector, away_ids, home_ids = _current_features(imported, game)
    model, source_sha, train_sha, official_count = _fit_verified_research_model(observed)
    probability = float(model.predict_proba(np.asarray([vector], dtype=float))[0, 1])
    if not isfinite(probability) or not 0 <= probability <= 1:
        raise NHLProspectiveShadowError("Probabilité exploratoire invalide.")
    feature_hash = _digest({
        "features": vector, "away_source_games": away_ids, "home_source_games": home_ids,
        "current_import_sha256": imported.source_response_sha256,
    })
    record = {
        "game_id": game.game_id, "scheduled_start_utc": _stamp(game.start_utc),
        "observed_at_utc": _stamp(observed), "information_cutoff_utc": _stamp(cutoff),
        "away_abbr": game.away_abbr, "home_abbr": game.home_abbr,
        "home_win_probability": probability, "away_win_probability": 1.0 - probability,
        "schedule_sha256": schedule["response_sha256"],
        "current_import_sha256": imported.source_response_sha256,
        "training_source_sha256": source_sha, "training_examples_sha256": train_sha,
        "feature_sha256": feature_hash, "policy_sha256": policy_sha,
        "official_tie_labels_in_training": official_count,
        "status": "PRIVATE_RESEARCH_SHADOW_ONLY",
        "production_training_permitted": False,
        "prediction_publication_permitted": False,
        "betting_recommendation_permitted": False,
    }
    return ProspectiveShadowObservation(
        **record, observation_sha256=_digest(record))
