"""Audit descriptif de cinq saisons NHL réelles, sans autoriser un modèle.

Une archive téléchargée après les matchs prouve leur couverture, pas la
 disponibilité historique des variables au moment d'un pronostic.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from .contracts import require_utc
from .lagged_historical_features import (
    DEFAULT_LOOKBACK_GAMES, build_lagged_regular_features,
)
from .moneypuck_five_season import (
    EXPECTED_REGULAR_GAMES_PER_SEASON, FiveSeasonHistory,
    REFERENCE_SEASONS, select_five_season_history,
)
from .moneypuck_team_import import import_team_games


class NHLRealArchiveAuditError(ValueError):
    """Provenance, forme passée ou classement des scores incohérent."""


@dataclass(frozen=True, slots=True)
class SeasonArchiveAudit:
    season: int
    regular_games: int
    expected_regular_games: int
    playoff_games_excluded: int
    full_lookback_games: int
    mirrored_decisive_scores: int
    tied_score_games: int
    inconsistent_score_games: int


@dataclass(frozen=True, slots=True)
class NHLRealArchiveAudit:
    source_file_sha256: str
    source_observed_at_utc: datetime
    source_page: str
    attribution: str
    lookback_games: int
    seasons: tuple[SeasonArchiveAudit, ...]
    regular_games: int
    feature_candidate_rows: int
    full_lookback_games: int
    mirrored_decisive_scores: int
    tied_score_game_ids: tuple[int, ...]
    inconsistent_score_game_ids: tuple[int, ...]
    audit_sha256: str
    regular_coverage_complete: bool
    prior_date_order_verified: bool = True
    historical_pregame_availability_proven: bool = False
    independent_final_labels_proven: bool = False
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def _valid_sha256(value: str) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def audit_reference_history(
    history: FiveSeasonHistory, *, lookback_games: int = DEFAULT_LOOKBACK_GAMES,
) -> NHLRealArchiveAudit:
    """Classe les matchs sans confondre score source et label final indépendant."""
    if not isinstance(history, FiveSeasonHistory):
        raise TypeError("Un historique NHL-12 vérifié est requis.")
    require_utc(history.observed_at_utc, field_name="source_observed_at_utc")
    if (not _valid_sha256(history.file_sha256) or history.training_permitted
            or not history.regular_rows
            or tuple(item.season for item in history.coverage) != REFERENCE_SEASONS):
        raise NHLRealArchiveAuditError("Provenance ou couverture non admissible.")
    candidates = build_lagged_regular_features(
        history, lookback_games=lookback_games)
    games: dict[int, list] = defaultdict(list)
    forms: dict[int, list] = defaultdict(list)
    for row in history.regular_rows:
        if row.situation == "all":
            games[row.game_id].append(row)
    for form in candidates.rows:
        forms[form.game_id].append(form)
    if set(games) != set(forms) or len(candidates.rows) != 2 * len(games):
        raise NHLRealArchiveAuditError("Formes et matchs non raccordables.")

    counts = {season: {"games": 0, "full": 0, "decisive": 0,
                       "tied": 0, "inconsistent": 0}
              for season in REFERENCE_SEASONS}
    tied: list[int] = []
    inconsistent: list[int] = []
    proof_rows: list[tuple] = []
    for game_id in sorted(games):
        rows = games[game_id]
        by_side = {row.home_or_away: row for row in rows}
        by_form = {form.home_or_away: form for form in forms[game_id]}
        if (len(rows) != 2 or len(forms[game_id]) != 2
                or set(by_side) != {"HOME", "AWAY"}
                or set(by_form) != {"HOME", "AWAY"}):
            raise NHLRealArchiveAuditError("Match ou forme dupliqué/incomplet.")
        home, away = by_side["HOME"], by_side["AWAY"]
        if (home.season not in counts or home.season != away.season
                or home.game_date != away.game_date
                or home.team != away.opponent or away.team != home.opponent
                or game_id // 1_000_000 != home.season):
            raise NHLRealArchiveAuditError("Identité de match incohérente.")
        if history.observed_at_utc.date() <= home.game_date:
            raise NHLRealArchiveAuditError("Archive non postérieure au match.")
        for side, row in (("HOME", home), ("AWAY", away)):
            form = by_form[side]
            if (form.game_id != game_id or form.season != row.season
                    or form.game_date != row.game_date or form.team != row.team
                    or form.opponent != row.opponent
                    or form.last_prior_date is not None
                    and form.last_prior_date >= row.game_date
                    or form.sample_games > form.prior_games
                    or form.sample_games > lookback_games):
                raise NHLRealArchiveAuditError("Forme passée ou chronologie invalide.")
        season = counts[home.season]
        season["games"] += 1
        full = all(by_form[side].sample_games == lookback_games
                   for side in ("HOME", "AWAY"))
        season["full"] += int(full)
        if (home.goals_for != away.goals_against
                or away.goals_for != home.goals_against):
            status = "INCONSISTENT"
            inconsistent.append(game_id)
            season["inconsistent"] += 1
        elif home.goals_for == away.goals_for:
            status = "TIED_SOURCE_SCORE"
            tied.append(game_id)
            season["tied"] += 1
        else:
            status = "MIRRORED_DECISIVE"
            season["decisive"] += 1
        proof_rows.append((game_id, home.game_date.isoformat(),
                           home.goals_for, away.goals_for, status,
                           by_form["HOME"].sample_games,
                           by_form["AWAY"].sample_games))

    seasons = tuple(SeasonArchiveAudit(
        season=coverage.season,
        regular_games=counts[coverage.season]["games"],
        expected_regular_games=EXPECTED_REGULAR_GAMES_PER_SEASON,
        playoff_games_excluded=coverage.playoff_games,
        full_lookback_games=counts[coverage.season]["full"],
        mirrored_decisive_scores=counts[coverage.season]["decisive"],
        tied_score_games=counts[coverage.season]["tied"],
        inconsistent_score_games=counts[coverage.season]["inconsistent"],
    ) for coverage in history.coverage)
    if any(item.regular_games != coverage.regular_games
           for item, coverage in zip(seasons, history.coverage)):
        raise NHLRealArchiveAuditError("Couverture déclarée incohérente.")
    proof = {"schema": "nhl_real_archive_audit_v1",
             "source_file_sha256": history.file_sha256,
             "lookback_games": lookback_games,
             "seasons": [asdict(item) for item in seasons],
             "games": proof_rows}
    digest = sha256(json.dumps(
        proof, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return NHLRealArchiveAudit(
        source_file_sha256=history.file_sha256,
        source_observed_at_utc=history.observed_at_utc,
        source_page=history.source_page,
        attribution=history.attribution,
        lookback_games=lookback_games, seasons=seasons,
        regular_games=len(games), feature_candidate_rows=len(candidates.rows),
        full_lookback_games=sum(item.full_lookback_games for item in seasons),
        mirrored_decisive_scores=sum(item.mirrored_decisive_scores for item in seasons),
        tied_score_game_ids=tuple(tied),
        inconsistent_score_game_ids=tuple(inconsistent),
        audit_sha256=digest,
        regular_coverage_complete=history.regular_coverage_complete,
    )


def audit_local_reference_archive(path: Path) -> NHLRealArchiveAudit:
    """Lit uniquement le CSV local; les autres saisons n'entrent pas dans l'audit."""
    snapshots = [import_team_games(path, season=season)
                 for season in REFERENCE_SEASONS]
    if len({item.file_sha256 for item in snapshots}) != 1:
        raise NHLRealArchiveAuditError("Le fichier source a changé pendant l'audit.")
    combined = replace(snapshots[-1], rows=tuple(
        row for snapshot in snapshots for row in snapshot.rows))
    return audit_reference_history(select_five_season_history(combined))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(asdict(audit_local_reference_archive(args.csv_path)),
                     default=str, ensure_ascii=False, sort_keys=True, indent=2))
