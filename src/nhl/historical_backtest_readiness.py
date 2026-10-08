"""NHL-35: inspect historical split candidates without fitting a model.

A current download of old games cannot establish what was known before each
historical puck drop. MoneyPuck ties are not two-outcome match winners.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path

from .database import PROJECT_ROOT
from .lagged_historical_features import build_lagged_regular_features
from .moneypuck_five_season import FiveSeasonHistory, REFERENCE_SEASONS
from .real_archive_audit import audit_reference_history

SCHEMA = "nhl35_historical_backtest_readiness_v1"
POLICY = PROJECT_ROOT / "nhl_protocols/data/nhl35_historical_backtest_readiness_v1.json"
SPLIT = {2021: "TRAIN", 2022: "TRAIN", 2023: "TRAIN",
         2024: "VALIDATION", 2025: "HOLDOUT"}


class NHLHistoricalBacktestReadinessError(ValueError):
    """Historical candidate ordering or source identity is inconsistent."""


@dataclass(frozen=True, slots=True)
class SeasonSplitReadiness:
    season: int
    role: str
    first_game_date: date | None
    last_game_date: date | None
    regular_games: int
    both_teams_have_prior_form: int
    decisive_source_scores: int
    tied_source_scores: int
    inconsistent_source_scores: int
    provisional_binary_candidates: int


@dataclass(frozen=True, slots=True)
class HistoricalBacktestReadiness:
    source_file_sha256: str
    source_observed_at_utc: datetime
    regular_coverage_complete: bool
    archive_audit_sha256: str
    lookback_games: int
    min_prior_games: int
    seasons: tuple[SeasonSplitReadiness, ...]
    provisional_train_games: int
    provisional_validation_games: int
    provisional_holdout_games: int
    unresolved_tied_game_ids: tuple[int, ...]
    inconsistent_game_ids: tuple[int, ...]
    audit_sha256: str
    historical_asof_proven: bool = False
    independent_final_labels_proven: bool = False
    backtest_authorized: bool = False
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def _policy_sha256() -> str:
    try:
        raw = POLICY.read_bytes()
        document = json.loads(raw)
    except (OSError, ValueError) as error:
        raise NHLHistoricalBacktestReadinessError("Protocole NHL-35 illisible.") from error
    required = {
        "schema_version": SCHEMA,
        "source_seasons": list(REFERENCE_SEASONS),
        "train_seasons": [2021, 2022, 2023],
        "validation_seasons": [2024], "holdout_seasons": [2025],
        "same_day_history_excluded": True,
        "tied_source_score_is_not_binary_winner": True,
        "historical_asof_proven": False,
        "independent_final_labels_proven": False,
        "backtest_authorized": False, "training_permitted": False,
        "prediction_publication_permitted": False,
        "network_calls_allowed": False, "database_mutation_allowed": False,
    }
    if not isinstance(document, dict) or any(document.get(k) != v for k, v in required.items()):
        raise NHLHistoricalBacktestReadinessError("Protocole NHL-35 incompatible.")
    return sha256(raw).hexdigest()


def audit_historical_backtest_readiness(
    history: FiveSeasonHistory, *, lookback_games: int = 5,
    min_prior_games: int = 5,
) -> HistoricalBacktestReadiness:
    """Count candidate rows per chronological role; never create training rows."""
    policy_sha = _policy_sha256()
    if (type(min_prior_games) is not int or min_prior_games < 1
            or type(lookback_games) is not int
            or not min_prior_games <= lookback_games <= 30):
        raise NHLHistoricalBacktestReadinessError("Fenetre historique invalide.")
    archive = audit_reference_history(history, lookback_games=lookback_games)
    forms = build_lagged_regular_features(history, lookback_games=lookback_games)
    by_game: dict[int, dict[str, object]] = defaultdict(dict)
    by_form: dict[int, dict[str, object]] = defaultdict(dict)
    for row in history.regular_rows:
        if row.situation == "all":
            by_game[row.game_id][row.home_or_away] = row
    for form in forms.rows:
        by_form[form.game_id][form.home_or_away] = form
    if set(by_game) != set(by_form) or len(by_game) != archive.regular_games:
        raise NHLHistoricalBacktestReadinessError("Formes et matchs divergents.")

    counts = {season: {"regular": 0, "warm": 0, "decisive": 0,
                       "tied": 0, "inconsistent": 0, "candidate": 0}
              for season in REFERENCE_SEASONS}
    dates: dict[int, list[date]] = defaultdict(list)
    proof_rows = []
    for game_id in sorted(by_game):
        rows, paired = by_game[game_id], by_form[game_id]
        if set(rows) != {"HOME", "AWAY"} or set(paired) != {"HOME", "AWAY"}:
            raise NHLHistoricalBacktestReadinessError("Match ou forme incomplet.")
        home, away = rows["HOME"], rows["AWAY"]
        season = home.season
        if (season not in counts or away.season != season
                or home.game_date != away.game_date):
            raise NHLHistoricalBacktestReadinessError("Saison ou date incoherente.")
        for side, row in (("HOME", home), ("AWAY", away)):
            form = paired[side]
            if (form.game_id != game_id or form.season != season
                    or form.game_date != row.game_date or form.team != row.team
                    or form.sample_games > lookback_games
                    or form.sample_games > form.prior_games
                    or form.last_prior_date is not None
                    and form.last_prior_date >= row.game_date):
                raise NHLHistoricalBacktestReadinessError("Forme posterieure au match.")
        mirrored = (home.goals_for == away.goals_against
                    and away.goals_for == home.goals_against)
        status = ("INCONSISTENT" if not mirrored else
                  "TIED_SOURCE_SCORE" if home.goals_for == away.goals_for else
                  "DECISIVE_SOURCE_SCORE")
        warm = all(paired[side].sample_games >= min_prior_games
                   for side in ("HOME", "AWAY"))
        bucket = counts[season]
        bucket["regular"] += 1
        bucket["warm"] += int(warm)
        bucket[{"INCONSISTENT": "inconsistent", "TIED_SOURCE_SCORE": "tied",
                "DECISIVE_SOURCE_SCORE": "decisive"}[status]] += 1
        bucket["candidate"] += int(warm and status == "DECISIVE_SOURCE_SCORE")
        dates[season].append(home.game_date)
        proof_rows.append((game_id, season, home.game_date.isoformat(), warm, status))

    seasons = []
    previous_last = None
    for observed in archive.seasons:
        season = observed.season
        date_list = dates[season]
        first, last = (min(date_list), max(date_list)) if date_list else (None, None)
        if first is not None and previous_last is not None and first <= previous_last:
            raise NHLHistoricalBacktestReadinessError("Saisons chronologiques chevauchees.")
        if last is not None:
            previous_last = last
        bucket = counts[season]
        if (bucket["regular"] != observed.regular_games
                or bucket["decisive"] != observed.mirrored_decisive_scores
                or bucket["tied"] != observed.tied_score_games
                or bucket["inconsistent"] != observed.inconsistent_score_games):
            raise NHLHistoricalBacktestReadinessError("Audit de source divergent.")
        seasons.append(SeasonSplitReadiness(
            season, SPLIT[season], first, last, bucket["regular"],
            bucket["warm"], bucket["decisive"], bucket["tied"],
            bucket["inconsistent"], bucket["candidate"],
        ))
    proof = {
        "schema_version": SCHEMA, "source_file_sha256": history.file_sha256,
        "source_observed_at_utc": history.observed_at_utc.isoformat(),
        "regular_coverage_complete": archive.regular_coverage_complete,
        "archive_audit_sha256": archive.audit_sha256,
        "policy_sha256": policy_sha, "lookback_games": lookback_games,
        "min_prior_games": min_prior_games,
        "seasons": [asdict(item) for item in seasons],
        "games": proof_rows,
    }
    digest = sha256(json.dumps(
        proof, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()
    return HistoricalBacktestReadiness(
        source_file_sha256=history.file_sha256,
        source_observed_at_utc=history.observed_at_utc,
        regular_coverage_complete=archive.regular_coverage_complete,
        archive_audit_sha256=archive.audit_sha256,
        lookback_games=lookback_games, min_prior_games=min_prior_games,
        seasons=tuple(seasons),
        provisional_train_games=sum(row.provisional_binary_candidates
                                    for row in seasons if row.role == "TRAIN"),
        provisional_validation_games=sum(row.provisional_binary_candidates
                                         for row in seasons if row.role == "VALIDATION"),
        provisional_holdout_games=sum(row.provisional_binary_candidates
                                      for row in seasons if row.role == "HOLDOUT"),
        unresolved_tied_game_ids=archive.tied_score_game_ids,
        inconsistent_game_ids=archive.inconsistent_score_game_ids,
        audit_sha256=digest,
    )
