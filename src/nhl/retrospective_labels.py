"""Rapprochement rétrospectif NHL : forme passée et résultat final séparés.

Les scores proviennent d'un téléchargement observé après les matchs. Cela ne
prouve pas la disponibilité pré-match des statistiques historiques. Aucun
entraînement ni pronostic publiable n'est autorisé par ce module.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from hashlib import sha256
import json

from .contracts import require_utc
from .lagged_historical_features import (
    LaggedHistoricalCandidates, LaggedTeamForm, NHLLaggedHistoryError,
    build_lagged_regular_features,
)
from .moneypuck_five_season import FiveSeasonHistory, REFERENCE_SEASONS
from .moneypuck_team_import import TeamGameRow


class NHLRetrospectiveLabelError(ValueError):
    """Résultat ou rapprochement non démontré par les lignes d'équipe."""


@dataclass(frozen=True, slots=True)
class HistoricalFinalResult:
    game_id: int
    season: int
    game_date: date
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    winner: str
    source_observed_at_utc: datetime


@dataclass(frozen=True, slots=True)
class HistoricalLabeledGame:
    home_form: LaggedTeamForm
    away_form: LaggedTeamForm
    result: HistoricalFinalResult


@dataclass(frozen=True, slots=True)
class RetrospectiveLabelAudit:
    source_file_sha256: str
    source_observed_at_utc: datetime
    lookback_games: int
    games_by_season: tuple[tuple[int, int], ...]
    rows: tuple[HistoricalLabeledGame, ...]
    audit_sha256: str
    regular_coverage_complete: bool
    historical_pregame_availability_proven: bool = False
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def _final_result(rows: list[TeamGameRow], observed_at: datetime) -> HistoricalFinalResult:
    if len(rows) != 2 or {row.home_or_away for row in rows} != {"HOME", "AWAY"}:
        raise NHLRetrospectiveLabelError("Deux lignes all domicile/extérieur sont requises.")
    home = next(row for row in rows if row.home_or_away == "HOME")
    away = next(row for row in rows if row.home_or_away == "AWAY")
    if (home.game_date != away.game_date or home.season != away.season
            or home.team != away.opponent or away.team != home.opponent
            or type(home.goals_for) is not int or type(away.goals_for) is not int
            or min(home.goals_for, away.goals_for) < 0
            or home.goals_for != away.goals_against
            or away.goals_for != home.goals_against
            or home.goals_for == away.goals_for):
        raise NHLRetrospectiveLabelError("Score final ou équipes incohérents.")
    if observed_at.date() <= home.game_date:
        raise NHLRetrospectiveLabelError("Le téléchargement n'est pas postérieur au match.")
    return HistoricalFinalResult(
        game_id=home.game_id, season=home.season, game_date=home.game_date,
        home_team=home.team, away_team=away.team,
        home_goals=home.goals_for, away_goals=away.goals_for,
        winner=home.team if home.goals_for > away.goals_for else away.team,
        source_observed_at_utc=observed_at,
    )


def build_retrospective_regular_labels(
    history: FiveSeasonHistory, candidates: LaggedHistoricalCandidates,
) -> RetrospectiveLabelAudit:
    """Audite les résultats réguliers sans injecter le score dans la forme passée."""
    if not isinstance(history, FiveSeasonHistory) or not isinstance(
            candidates, LaggedHistoricalCandidates):
        raise TypeError("Historique NHL-12 et candidats NHL-13 vérifiés requis.")
    require_utc(history.observed_at_utc, field_name="observed_at_utc")
    digest = history.file_sha256
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or history.training_permitted or candidates.training_permitted
            or candidates.labels_included or not history.regular_rows):
        raise NHLRetrospectiveLabelError("Provenance ou périmètre non admissible.")
    try:
        expected = build_lagged_regular_features(
            history, lookback_games=candidates.lookback_games)
    except NHLLaggedHistoryError as error:
        raise NHLRetrospectiveLabelError("Historique des formes invalide.") from error
    if candidates != expected:
        raise NHLRetrospectiveLabelError("Les formes ne correspondent pas à la source.")

    groups: dict[int, list[TeamGameRow]] = defaultdict(list)
    for row in history.regular_rows:
        if row.situation == "all":
            groups[row.game_id].append(row)
    forms = {(row.game_id, row.home_or_away): row for row in candidates.rows}
    if len(forms) != len(candidates.rows) or len(forms) != 2 * len(groups):
        raise NHLRetrospectiveLabelError("Formes dupliquées ou matchs incomplets.")

    output: list[HistoricalLabeledGame] = []
    counts = {season: 0 for season in REFERENCE_SEASONS}
    for game_id, game_rows in sorted(groups.items()):
        result = _final_result(game_rows, history.observed_at_utc)
        home_form = forms.get((game_id, "HOME"))
        away_form = forms.get((game_id, "AWAY"))
        if (home_form is None or away_form is None
                or home_form.team != result.home_team
                or away_form.team != result.away_team
                or home_form.game_date != result.game_date
                or away_form.game_date != result.game_date
                or home_form.season != result.season
                or away_form.season != result.season):
            raise NHLRetrospectiveLabelError("Match et formes non raccordables.")
        output.append(HistoricalLabeledGame(home_form, away_form, result))
        counts[result.season] += 1
    if tuple(counts[season] for season in REFERENCE_SEASONS) != tuple(
            item.regular_games for item in history.coverage):
        raise NHLRetrospectiveLabelError("Couverture des saisons incohérente.")
    proof = {
        "schema": "nhl_retrospective_regular_labels_v1",
        "source_file_sha256": digest,
        "source_observed_at_utc": history.observed_at_utc.isoformat(),
        "lookback_games": candidates.lookback_games,
        "rows": [asdict(row) for row in output],
    }
    audit_sha256 = sha256(json.dumps(
        proof, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()
    return RetrospectiveLabelAudit(
        source_file_sha256=digest,
        source_observed_at_utc=history.observed_at_utc,
        lookback_games=candidates.lookback_games,
        games_by_season=tuple((season, counts[season]) for season in REFERENCE_SEASONS),
        rows=tuple(output), audit_sha256=audit_sha256,
        regular_coverage_complete=history.regular_coverage_complete,
    )
