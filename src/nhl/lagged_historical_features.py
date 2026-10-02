"""Forme NHL rétrospective calculée uniquement sur des dates antérieures.

Ce module ne prouve pas une disponibilité historique avant match et ne forme
aucun modèle. Les résultats du match cible ne sont jamais exposés.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from .moneypuck_five_season import (
    FiveSeasonHistory,
    REFERENCE_SEASONS,
    REGULAR_GAME_TYPE,
    REQUIRED_SITUATIONS,
    SeasonCoverage,
)
from .moneypuck_team_import import TeamGameRow

DEFAULT_LOOKBACK_GAMES = 5


class NHLLaggedHistoryError(ValueError):
    """Historique incomplet ou incohérent : calcul refusé."""


@dataclass(frozen=True, slots=True)
class LaggedTeamForm:
    game_id: int
    season: int
    game_date: date
    team: str
    opponent: str
    home_or_away: str
    prior_games: int
    sample_games: int
    last_prior_date: date | None
    goals_for_mean: Decimal | None
    goals_against_mean: Decimal | None
    x_goals_for_mean: Decimal | None
    x_goals_against_mean: Decimal | None
    five_on_five_x_goals_for_mean: Decimal | None
    five_on_five_x_goals_against_mean: Decimal | None


@dataclass(frozen=True, slots=True)
class LaggedHistoricalCandidates:
    observed_at_utc: datetime
    file_sha256: str
    attribution: str
    lookback_games: int
    coverage: tuple[SeasonCoverage, ...]
    rows: tuple[LaggedTeamForm, ...]
    labels_included: bool = False
    training_permitted: bool = False


def _mean(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    return sum(values, Decimal(0)) / Decimal(len(values))


def _validate_group(game_id: int, teams: dict[str, dict[str, TeamGameRow]]) -> None:
    if len(teams) != 2:
        raise NHLLaggedHistoryError(f"Match {game_id} : deux équipes requises.")
    for team, situations in teams.items():
        if set(situations) != REQUIRED_SITUATIONS:
            raise NHLLaggedHistoryError(f"Match {game_id}, {team} : situations incomplètes.")
        all_row, five_row = situations["all"], situations["5on5"]
        if (all_row.game_date, all_row.opponent, all_row.home_or_away) != (
            five_row.game_date, five_row.opponent, five_row.home_or_away
        ):
            raise NHLLaggedHistoryError(f"Match {game_id}, {team} : lignes incompatibles.")
    first, second = (situations["all"] for situations in teams.values())
    if first.opponent != second.team or second.opponent != first.team:
        raise NHLLaggedHistoryError(f"Match {game_id} : adversaires incohérents.")
    if {first.home_or_away, second.home_or_away} != {"HOME", "AWAY"}:
        raise NHLLaggedHistoryError(f"Match {game_id} : domicile/extérieur incohérent.")


def build_lagged_regular_features(
    history: FiveSeasonHistory, *, lookback_games: int = DEFAULT_LOOKBACK_GAMES
) -> LaggedHistoricalCandidates:
    """Construit des indicateurs descriptifs sans utiliser le jour cible.

    Tous les matchs d'une même date sont calculés avant de mettre à jour
    l'historique : un double match du jour ne fuit pas dans l'autre.
    Les séries ne sont ni des cibles ni des antécédents.
    """
    if not isinstance(history, FiveSeasonHistory):
        raise TypeError("Un historique NHL-12 vérifié est requis.")
    if type(lookback_games) is not int or not 1 <= lookback_games <= 30:
        raise NHLLaggedHistoryError("La fenêtre doit compter entre 1 et 30 matchs.")

    by_day: dict[tuple[int, date], dict[int, dict[str, dict[str, TeamGameRow]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for row in history.regular_rows:
        if row.season not in REFERENCE_SEASONS or row.game_id // 1_000_000 != row.season:
            raise NHLLaggedHistoryError("Saison de match hors périmètre ou incohérente.")
        if (row.game_id // 10_000) % 100 != REGULAR_GAME_TYPE:
            raise NHLLaggedHistoryError("Un match non régulier apparaît dans la saison régulière.")
        if row.situation not in REQUIRED_SITUATIONS:
            raise NHLLaggedHistoryError("Situation non autorisée.")
        situations = by_day[(row.season, row.game_date)][row.game_id][row.team]
        if row.situation in situations:
            raise NHLLaggedHistoryError("Ligne équipe/situation dupliquée.")
        situations[row.situation] = row

    prior: dict[tuple[int, str], list[tuple[TeamGameRow, TeamGameRow]]] = defaultdict(list)
    output: list[LaggedTeamForm] = []
    for (season, target_date), games in sorted(by_day.items()):
        for game_id, teams in sorted(games.items()):
            _validate_group(game_id, teams)
            for team, situations in sorted(teams.items()):
                all_row = situations["all"]
                recent = prior[(season, team)][-lookback_games:]
                if recent and recent[-1][0].game_date >= target_date:
                    raise NHLLaggedHistoryError("Une date non antérieure entre dans la forme.")
                output.append(LaggedTeamForm(
                    game_id=game_id, season=season, game_date=target_date,
                    team=team, opponent=all_row.opponent,
                    home_or_away=all_row.home_or_away,
                    prior_games=len(prior[(season, team)]), sample_games=len(recent),
                    last_prior_date=recent[-1][0].game_date if recent else None,
                    goals_for_mean=_mean([Decimal(pair[0].goals_for) for pair in recent]),
                    goals_against_mean=_mean([Decimal(pair[0].goals_against) for pair in recent]),
                    x_goals_for_mean=_mean([pair[0].x_goals_for for pair in recent]),
                    x_goals_against_mean=_mean([pair[0].x_goals_against for pair in recent]),
                    five_on_five_x_goals_for_mean=_mean([pair[1].x_goals_for for pair in recent]),
                    five_on_five_x_goals_against_mean=_mean([pair[1].x_goals_against for pair in recent]),
                ))
        for game_id, teams in sorted(games.items()):
            for team, situations in sorted(teams.items()):
                prior[(season, team)].append((situations["all"], situations["5on5"]))

    return LaggedHistoricalCandidates(
        observed_at_utc=history.observed_at_utc,
        file_sha256=history.file_sha256,
        attribution=history.attribution,
        lookback_games=lookback_games,
        coverage=history.coverage,
        rows=tuple(output),
    )
