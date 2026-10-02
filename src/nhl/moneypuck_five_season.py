"""Sélection et contrôle descriptif de cinq saisons MoneyPuck, sans entraînement."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from .moneypuck_team_import import MoneyPuckTeamSnapshot, TeamGameRow

REFERENCE_SEASONS = (2021, 2022, 2023, 2024, 2025)
REGULAR_GAME_TYPE = 2
PLAYOFF_GAME_TYPE = 3
EXPECTED_REGULAR_GAMES_PER_SEASON = 1312
REQUIRED_SITUATIONS = frozenset({"all", "5on5"})


class NHLFiveSeasonAuditError(ValueError):
    """Les lignes ne permettent pas un contrôle non ambigu des rencontres."""


@dataclass(frozen=True, slots=True)
class SeasonCoverage:
    season: int
    regular_games: int
    playoff_games: int
    expected_regular_games: int = EXPECTED_REGULAR_GAMES_PER_SEASON

    @property
    def regular_complete(self) -> bool:
        return self.regular_games == self.expected_regular_games


@dataclass(frozen=True, slots=True)
class FiveSeasonHistory:
    """Vue descriptive; aucune disponibilité historique pré-match n'est inférée."""

    observed_at_utc: datetime
    file_sha256: str
    source_page: str
    attribution: str
    regular_rows: tuple[TeamGameRow, ...]
    playoff_rows: tuple[TeamGameRow, ...]
    coverage: tuple[SeasonCoverage, ...]
    training_permitted: bool = False

    @property
    def regular_coverage_complete(self) -> bool:
        return all(item.regular_complete for item in self.coverage)


def _check_pair(
    key: tuple[int, int, str], rows: list[TeamGameRow]
) -> None:
    if len(rows) != 2:
        raise NHLFiveSeasonAuditError(f"Match/situation {key}: deux équipes requises.")
    by_side = {row.home_or_away: row for row in rows}
    if set(by_side) != {"HOME", "AWAY"}:
        raise NHLFiveSeasonAuditError(f"Match/situation {key}: domicile/extérieur invalides.")
    home, away = by_side["HOME"], by_side["AWAY"]
    if (
        home.team == away.team
        or home.team != away.opponent
        or away.team != home.opponent
        or home.game_date != away.game_date
    ):
        raise NHLFiveSeasonAuditError(f"Match/situation {key}: équipes ou date incohérentes.")


def select_five_season_history(snapshot: MoneyPuckTeamSnapshot) -> FiveSeasonHistory:
    """Isole 2021-2025, avec saison régulière et séries dans deux vues distinctes.

    La complétude sportive est mesurée; elle n'autorise jamais l'entraînement.
    """
    if not isinstance(snapshot, MoneyPuckTeamSnapshot):
        raise TypeError("Un instantané MoneyPuck vérifié est requis.")
    groups: dict[tuple[int, int, str], list[TeamGameRow]] = defaultdict(list)
    rows_by_type: dict[int, list[TeamGameRow]] = {
        REGULAR_GAME_TYPE: [], PLAYOFF_GAME_TYPE: [],
    }
    game_keys_by_type: dict[int, set[tuple[int, int]]] = {
        REGULAR_GAME_TYPE: set(), PLAYOFF_GAME_TYPE: set(),
    }
    for row in snapshot.rows:
        if row.season not in REFERENCE_SEASONS:
            continue
        if row.game_id // 1_000_000 != row.season:
            raise NHLFiveSeasonAuditError("Identifiant de match/saison incohérent.")
        game_type = (row.game_id // 10_000) % 100
        if game_type not in rows_by_type:
            continue  # Pré-saison ou autre compétition: hors périmètre.
        if row.situation not in REQUIRED_SITUATIONS:
            continue
        groups[(row.season, row.game_id, row.situation)].append(row)
        rows_by_type[game_type].append(row)
        game_keys_by_type[game_type].add((row.season, row.game_id))

    situations_by_game: dict[tuple[int, int], set[str]] = defaultdict(set)
    for key, pair in groups.items():
        _check_pair(key, pair)
        situations_by_game[(key[0], key[1])].add(key[2])
    for game_key, situations in situations_by_game.items():
        if situations != REQUIRED_SITUATIONS:
            raise NHLFiveSeasonAuditError(
                f"Match {game_key}: situations 'all' et '5on5' requises."
            )

    coverage = tuple(
        SeasonCoverage(
            season=season,
            regular_games=sum(key[0] == season for key in game_keys_by_type[REGULAR_GAME_TYPE]),
            playoff_games=sum(key[0] == season for key in game_keys_by_type[PLAYOFF_GAME_TYPE]),
        )
        for season in REFERENCE_SEASONS
    )
    sort_key = lambda row: (row.season, row.game_date, row.game_id, row.team, row.situation)
    return FiveSeasonHistory(
        observed_at_utc=snapshot.observed_at_utc,
        file_sha256=snapshot.file_sha256,
        source_page=snapshot.source_page,
        attribution=snapshot.attribution,
        regular_rows=tuple(sorted(rows_by_type[REGULAR_GAME_TYPE], key=sort_key)),
        playoff_rows=tuple(sorted(rows_by_type[PLAYOFF_GAME_TYPE], key=sort_key)),
        coverage=coverage,
    )
