"""Research-only NHL chronology test; never a subscriber prediction.

All features use prior game dates, but the MoneyPuck CSV was received after
the historical seasons. This is retrospective exploration, not a certified
as-of backtest. The fitted estimator lives only in memory and is not saved.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .database import PROJECT_ROOT
from .frozen_historical_source import audit_frozen_historical_source
from .historical_asof_coverage import classify_frozen_source_timing
from .historical_final_backlog import _slots
from .historical_official_final import ROOT as FINAL_ROOT, _verify
from .lagged_historical_features import (
    LaggedTeamForm, build_lagged_regular_features,
)
from .moneypuck_five_season import FiveSeasonHistory
from .moneypuck_team_capture import DEFAULT_ROOT as TEAM_ROOT, verify_team_capture

FEATURES = (
    "goals_for_mean", "goals_against_mean",
    "x_goals_for_mean", "x_goals_against_mean",
    "five_on_five_x_goals_for_mean", "five_on_five_x_goals_against_mean",
)
SPLITS = {2021: "TRAIN", 2022: "TRAIN", 2023: "TRAIN",
          2024: "VALIDATION", 2025: "HOLDOUT"}
POLICY_PATH = PROJECT_ROOT / "nhl_protocols/data/nhl_exploratory_chronological_backtest_v1.json"


class NHLExploratoryBacktestError(ValueError):
    """Retrospective examples or a chronological evaluation are inconsistent."""


def _policy_hash() -> str:
    try:
        raw = POLICY_PATH.read_bytes()
        policy = json.loads(raw)
    except (OSError, ValueError) as error:
        raise NHLExploratoryBacktestError("Protocole exploratoire illisible.") from error
    expected = {
        "schema_version": "nhl_exploratory_chronological_backtest_v1",
        "purpose": "aggregate_research_metrics_only",
        "source": "verified_frozen_retrospective_moneypuck_capture",
        "train_seasons": [2021, 2022, 2023],
        "validation_seasons": [2024], "holdout_seasons": [2025],
        "lagged_prior_game_dates_only": True,
        "same_day_source_games_excluded": True,
        "official_final_required_for_tied_source_scores": True,
        "source_decisive_labels_not_independently_verified": True,
        "historical_asof_proven": False,
        "research_fit_permitted": True,
        "production_training_permitted": False,
        "model_artifact_allowed": False,
        "per_game_predictions_allowed": False,
        "prediction_publication_permitted": False,
        "network_calls_allowed": False,
        "odds_api_calls_allowed": False,
        "database_mutation_allowed": False,
    }
    if not isinstance(policy, dict) or any(policy.get(k) != v for k, v in expected.items()):
        raise NHLExploratoryBacktestError("Protocole exploratoire incompatible.")
    return sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class ExploratoryExample:
    game_id: int
    season: int
    features: tuple[float, ...]
    home_won: int
    label_source: str


@dataclass(frozen=True, slots=True)
class ExploratorySplitMetrics:
    role: str
    games: int
    home_win_rate: float
    model_brier: float
    baseline_brier: float
    model_log_loss: float
    baseline_log_loss: float
    model_accuracy: float
    baseline_accuracy: float
    prediction_sha256: str


@dataclass(frozen=True, slots=True)
class NHLExploratoryBacktest:
    source_capture_sha256: str
    examples: int
    excluded_without_prior_form: int
    official_tie_labels: int
    source_decisive_labels: int
    splits: tuple[ExploratorySplitMetrics, ...]
    audit_sha256: str
    retrospective_only: bool = True
    historical_asof_proven: bool = False
    all_labels_independently_verified: bool = False
    production_training_permitted: bool = False
    prediction_publication_permitted: bool = False


def _feature_vector(home: LaggedTeamForm, away: LaggedTeamForm
                    ) -> tuple[float, ...]:
    values = []
    for name in FEATURES:
        home_value, away_value = getattr(home, name), getattr(away, name)
        if home_value is None or away_value is None:
            raise NHLExploratoryBacktestError("Forme antérieure incomplète.")
        difference = float(home_value - away_value)
        if not isfinite(difference):
            raise NHLExploratoryBacktestError("Caractéristique non finie.")
        values.append(difference)
    return tuple(values)


def build_exploratory_examples(
    history: FiveSeasonHistory, official_tie_winners: dict[int, str],
    *, min_prior_games: int = 5,
) -> tuple[ExploratoryExample, ...]:
    """Pair lagged forms and retrospective winner labels; no target score feature."""
    if (not isinstance(history, FiveSeasonHistory)
            or history.training_permitted
            or type(min_prior_games) is not int
            or not 1 <= min_prior_games <= 5
            or not isinstance(official_tie_winners, dict)):
        raise NHLExploratoryBacktestError("Source exploratoire invalide.")
    forms = build_lagged_regular_features(history, lookback_games=5)
    by_form: dict[int, dict[str, LaggedTeamForm]] = defaultdict(dict)
    by_game: dict[int, dict[str, object]] = defaultdict(dict)
    for form in forms.rows:
        side = form.home_or_away
        if side in by_form[form.game_id]:
            raise NHLExploratoryBacktestError("Forme d'équipe dupliquée.")
        by_form[form.game_id][side] = form
    for row in history.regular_rows:
        if row.situation != "all":
            continue
        side = row.home_or_away
        if side in by_game[row.game_id]:
            raise NHLExploratoryBacktestError("Score d'équipe dupliqué.")
        by_game[row.game_id][side] = row
    if set(by_form) != set(by_game):
        raise NHLExploratoryBacktestError("Formes et scores divergents.")
    observed_ties: set[int] = set()
    examples = []
    for game_id in sorted(by_game):
        rows, pair = by_game[game_id], by_form[game_id]
        if set(rows) != {"HOME", "AWAY"} or set(pair) != {"HOME", "AWAY"}:
            raise NHLExploratoryBacktestError("Match incomplet.")
        home_row, away_row = rows["HOME"], rows["AWAY"]
        home, away = pair["HOME"], pair["AWAY"]
        if (home_row.team != home.team or away_row.team != away.team
                or home_row.season != away_row.season
                or home_row.season != home.season or home.season != away.season
                or home_row.game_date != away_row.game_date
                or home_row.game_date != home.game_date
                or home.game_date != away.game_date
                or home_row.goals_for != away_row.goals_against
                or away_row.goals_for != home_row.goals_against
                or home_row.team != away_row.opponent
                or away_row.team != home_row.opponent
                or home.last_prior_date is not None
                and home.last_prior_date >= home.game_date
                or away.last_prior_date is not None
                and away.last_prior_date >= away.game_date):
            raise NHLExploratoryBacktestError("Identité ou chronologie divergente.")
        if home_row.goals_for == away_row.goals_for:
            observed_ties.add(game_id)
            winner = official_tie_winners.get(game_id)
            if winner not in {home.team, away.team}:
                raise NHLExploratoryBacktestError("Match nul sans vainqueur officiel.")
            label_source = "OFFICIAL_TIE"
        else:
            if game_id in official_tie_winners:
                raise NHLExploratoryBacktestError("Résultat officiel non lié à un nul.")
            winner = home.team if home_row.goals_for > away_row.goals_for else away.team
            label_source = "SOURCE_DECISIVE"
        if home.sample_games >= min_prior_games and away.sample_games >= min_prior_games:
            examples.append(ExploratoryExample(
                game_id, home.season, _feature_vector(home, away),
                int(winner == home.team), label_source,
            ))
    if observed_ties != set(official_tie_winners):
        raise NHLExploratoryBacktestError("Couverture des nuls officiels incomplète.")
    return tuple(examples)


def _metrics(role: str, examples: tuple[ExploratoryExample, ...],
             probabilities: np.ndarray, baseline: float
             ) -> ExploratorySplitMetrics:
    labels = np.asarray([item.home_won for item in examples], dtype=int)
    if len(probabilities) != len(labels) or not len(labels):
        raise NHLExploratoryBacktestError("Prédictions de validation incomplètes.")
    if not np.all(np.isfinite(probabilities)) or not np.all(
            (probabilities >= 0) & (probabilities <= 1)):
        raise NHLExploratoryBacktestError("Probabilités invalides.")
    prior = np.full(len(labels), baseline)
    predictions = [float(value) for value in probabilities]
    prediction_hash = sha256(json.dumps(
        [(item.game_id, round(float(p), 12))
         for item, p in zip(examples, probabilities)],
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return ExploratorySplitMetrics(
        role=role, games=len(labels), home_win_rate=float(labels.mean()),
        model_brier=float(brier_score_loss(labels, probabilities)),
        baseline_brier=float(brier_score_loss(labels, prior)),
        model_log_loss=float(log_loss(labels, probabilities, labels=[0, 1])),
        baseline_log_loss=float(log_loss(labels, prior, labels=[0, 1])),
        model_accuracy=float(accuracy_score(labels, probabilities >= 0.5)),
        baseline_accuracy=float(accuracy_score(labels, prior >= 0.5)),
        prediction_sha256=prediction_hash,
    )


def evaluate_exploratory_examples(
    examples: tuple[ExploratoryExample, ...], *, source_capture_sha256: str,
    historical_games: int, official_tie_labels: int,
) -> NHLExploratoryBacktest:
    """One fixed logistic fit on 2021-23; 2024/25 never fit or tune."""
    policy_sha256 = _policy_hash()
    if (not isinstance(examples, tuple)
            or len(examples) >= historical_games
            or len(examples) < 1
            or type(historical_games) is not int
            or type(official_tie_labels) is not int
            or official_tie_labels < 0
            or official_tie_labels > historical_games
            or not isinstance(source_capture_sha256, str)
            or len(source_capture_sha256) != 64
            or any(char not in "0123456789abcdef"
                   for char in source_capture_sha256)):
        raise NHLExploratoryBacktestError("Périmètre d'évaluation invalide.")
    if len({item.game_id for item in examples}) != len(examples):
        raise NHLExploratoryBacktestError("Match dupliqué.")
    for item in examples:
        if (item.season not in SPLITS
                or type(item.home_won) is not int or item.home_won not in (0, 1)
                or item.label_source not in {"OFFICIAL_TIE", "SOURCE_DECISIVE"}
                or len(item.features) != len(FEATURES)
                or not all(isfinite(value) for value in item.features)):
            raise NHLExploratoryBacktestError("Exemple invalide.")
    if official_tie_labels < sum(item.label_source == "OFFICIAL_TIE" for item in examples):
        raise NHLExploratoryBacktestError("Nombre de labels officiels insuffisant.")
    by_role = {role: tuple(sorted(
        (item for item in examples if SPLITS[item.season] == role),
        key=lambda item: item.game_id
    )) for role in ("TRAIN", "VALIDATION", "HOLDOUT")}
    if any(not group for group in by_role.values()):
        raise NHLExploratoryBacktestError("Saison de test absente.")
    train = by_role["TRAIN"]
    if {item.home_won for item in train} != {0, 1}:
        raise NHLExploratoryBacktestError("Deux classes d'entraînement requises.")
    X_train = np.asarray([item.features for item in train], dtype=float)
    y_train = np.asarray([item.home_won for item in train], dtype=int)
    baseline = float(y_train.mean())
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )
    model.fit(X_train, y_train)
    metrics = tuple(_metrics(
        role, group,
        model.predict_proba(np.asarray(
            [item.features for item in group], dtype=float
        ))[:, 1],
        baseline,
    ) for role, group in by_role.items())
    proof = {
        "schema": "nhl_exploratory_chronological_backtest_v1",
        "source_capture_sha256": source_capture_sha256,
        "policy_sha256": policy_sha256,
        "historical_games": historical_games,
        "official_tie_labels": official_tie_labels,
        "features": FEATURES,
        "split": SPLITS,
        "examples": [asdict(item) for item in sorted(
            examples, key=lambda item: item.game_id)],
        "metrics": [asdict(item) for item in metrics],
        "retrospective_only": True,
        "historical_asof_proven": False,
        "production_training_permitted": False,
        "prediction_publication_permitted": False,
    }
    digest = sha256(json.dumps(
        proof, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return NHLExploratoryBacktest(
        source_capture_sha256=source_capture_sha256,
        examples=len(examples),
        excluded_without_prior_form=historical_games - len(examples),
        official_tie_labels=sum(item.label_source == "OFFICIAL_TIE"
                                for item in examples),
        source_decisive_labels=sum(item.label_source == "SOURCE_DECISIVE"
                                   for item in examples),
        splits=metrics, audit_sha256=digest,
    )


def run_exploratory_backtest(
    *, team_root: Path = TEAM_ROOT, final_root: Path = FINAL_ROOT,
) -> NHLExploratoryBacktest:
    """Read-only local experiment; reject incomplete source or final receipts."""
    frozen = audit_frozen_historical_source(
        team_root=team_root, final_root=final_root
    )
    source = verify_team_capture(frozen.team_slot, allowed_root=team_root)
    availability = classify_frozen_source_timing(
        source.history, source_observed_at_utc=source.observed_at_utc,
        frozen_audit_sha256=frozen.audit_sha256,
    )
    if (source.response_sha256 != frozen.source_capture_sha256
            or availability.historical_pregame_availability_proven
            or availability.source_received_after_game != availability.historical_games):
        raise NHLExploratoryBacktestError("Périmètre rétrospectif non attendu.")
    receipts = tuple(_verify(slot, source, final_root)
                     for slot in _slots(Path(final_root)))
    tie_winners = {item.game_id: item.winner_abbr for item in receipts}
    if (len(tie_winners) != frozen.verified_official_tie_finals
            or any(not item.source_was_tied for item in receipts)):
        raise NHLExploratoryBacktestError("Résultats officiels incomplets.")
    examples = build_exploratory_examples(source.history, tie_winners)
    return evaluate_exploratory_examples(
        examples, source_capture_sha256=source.response_sha256,
        historical_games=availability.historical_games,
        official_tie_labels=len(tie_winners),
    )


def main() -> int:
    result = run_exploratory_backtest()
    print(json.dumps({
        "source_capture_sha256": result.source_capture_sha256,
        "examples": result.examples,
        "excluded_without_prior_form": result.excluded_without_prior_form,
        "official_tie_labels_in_examples": result.official_tie_labels,
        "source_decisive_labels_in_examples": result.source_decisive_labels,
        "splits": [asdict(item) for item in result.splits],
        "audit_sha256": result.audit_sha256,
        "retrospective_only": result.retrospective_only,
        "historical_asof_proven": result.historical_asof_proven,
        "production_training_permitted": result.production_training_permitted,
        "prediction_publication_permitted": result.prediction_publication_permitted,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
