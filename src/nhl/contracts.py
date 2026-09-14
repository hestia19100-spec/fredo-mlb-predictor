"""Contrats de données purs pour la future chaîne LPF Edge NHL.

Les objets de ce module décrivent ce qui devra être prouvé avant qu'une
information puisse contribuer à une variable prédictive. Ils sont sans I/O et
ne connaissent ni client HTTP, ni base de données, ni modèle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
import re


_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class NHLContractError(ValueError):
    """Signale une donnée NHL qui ne respecte pas son contrat."""


class FeatureKind(str, Enum):
    """Familles de variables envisagées, sans définir encore un modèle."""

    TEAM_FORM = "TEAM_FORM"
    GOALIE_FORM = "GOALIE_FORM"
    GOALIE_STATUS = "GOALIE_STATUS"
    SHOT_FORM = "SHOT_FORM"
    EXPECTED_GOALS_FORM = "EXPECTED_GOALS_FORM"
    GOAL_FORM = "GOAL_FORM"
    POWER_PLAY = "POWER_PLAY"
    PENALTY_KILL = "PENALTY_KILL"
    REST = "REST"
    BACK_TO_BACK = "BACK_TO_BACK"
    TRAVEL = "TRAVEL"
    INJURY_STATUS = "INJURY_STATUS"
    HOME_ICE = "HOME_ICE"


class GoalieStatus(str, Enum):
    """Niveau de connaissance pré-match du gardien annoncé."""

    UNKNOWN = "UNKNOWN"
    PROBABLE = "PROBABLE"
    CONFIRMED = "CONFIRMED"


class PlayerAvailabilityStatus(str, Enum):
    """Statut pré-match observé pour un joueur potentiellement absent."""

    UNKNOWN = "UNKNOWN"
    AVAILABLE = "AVAILABLE"
    PROBABLE = "PROBABLE"
    QUESTIONABLE = "QUESTIONABLE"
    DOUBTFUL = "DOUBTFUL"
    OUT = "OUT"


def require_utc(value: datetime, *, field_name: str) -> datetime:
    """Refuse les horodatages naïfs ou exprimés hors UTC."""
    if not isinstance(value, datetime):
        raise NHLContractError(f"{field_name} doit être un datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise NHLContractError(f"{field_name} doit avoir un fuseau UTC.")
    if value.utcoffset() != timedelta(0):
        raise NHLContractError(f"{field_name} doit être exprimé en UTC.")
    return value


def _required_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise NHLContractError(
            f"{field_name} doit être un identifiant canonique en minuscules."
        )
    return value


def _positive_integer(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NHLContractError(f"{field_name} doit être un entier positif.")
    return value


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    """Preuve immuable associée à une observation d'un fournisseur."""

    provider: str
    observation_id: str
    observed_at_utc: datetime
    response_sha256: str
    code_commit: str
    source_updated_at_utc: datetime | None = None

    def __post_init__(self) -> None:
        _required_identifier(self.provider, field_name="provider")
        _required_identifier(self.observation_id, field_name="observation_id")
        require_utc(self.observed_at_utc, field_name="observed_at_utc")
        if (
            not isinstance(self.response_sha256, str)
            or not _SHA256_PATTERN.fullmatch(self.response_sha256)
        ):
            raise NHLContractError(
                "response_sha256 doit être une empreinte SHA-256 minuscule."
            )
        if (
            not isinstance(self.code_commit, str)
            or not _GIT_COMMIT_PATTERN.fullmatch(self.code_commit)
        ):
            raise NHLContractError(
                "code_commit doit être un identifiant Git SHA-1 minuscule."
            )
        if self.source_updated_at_utc is not None:
            require_utc(
                self.source_updated_at_utc,
                field_name="source_updated_at_utc",
            )

    @property
    def effective_available_at_utc(self) -> datetime:
        """Retourne l'instant le plus tardif prouvant la disponibilité."""
        if self.source_updated_at_utc is None:
            return self.observed_at_utc
        return max(self.observed_at_utc, self.source_updated_at_utc)


@dataclass(frozen=True, slots=True)
class ScheduledGameObservation:
    """Version pré-match observée d'une rencontre NHL."""

    game_id: int
    season_id: int
    official_date: date
    away_team_id: int
    home_team_id: int
    scheduled_start_utc: datetime
    evidence: SourceEvidence

    def __post_init__(self) -> None:
        _positive_integer(self.game_id, field_name="game_id")
        _positive_integer(self.season_id, field_name="season_id")
        if type(self.official_date) is not date:
            raise NHLContractError("official_date doit être une date.")
        _positive_integer(self.away_team_id, field_name="away_team_id")
        _positive_integer(self.home_team_id, field_name="home_team_id")
        if self.away_team_id == self.home_team_id:
            raise NHLContractError(
                "Les équipes à domicile et à l'extérieur doivent différer."
            )
        require_utc(
            self.scheduled_start_utc,
            field_name="scheduled_start_utc",
        )
        if not isinstance(self.evidence, SourceEvidence):
            raise NHLContractError("evidence doit être une SourceEvidence.")


@dataclass(frozen=True, slots=True)
class SourceGameResult:
    """Résultat antérieur et instant auquel il a été observé final."""

    game_id: int
    scheduled_start_utc: datetime
    final_observed_at_utc: datetime

    def __post_init__(self) -> None:
        _positive_integer(self.game_id, field_name="source game_id")
        require_utc(
            self.scheduled_start_utc,
            field_name="source scheduled_start_utc",
        )
        require_utc(
            self.final_observed_at_utc,
            field_name="source final_observed_at_utc",
        )
        if self.final_observed_at_utc <= self.scheduled_start_utc:
            raise NHLContractError(
                "Un résultat ne peut pas être final avant l'heure prévue."
            )


@dataclass(frozen=True, slots=True)
class FeatureProvenance:
    """Lignée temporelle minimale d'une future variable NHL."""

    target_game_id: int
    feature_kind: FeatureKind
    evidence: tuple[SourceEvidence, ...]
    source_games: tuple[SourceGameResult, ...] = ()

    def __post_init__(self) -> None:
        _positive_integer(self.target_game_id, field_name="target_game_id")
        if not isinstance(self.feature_kind, FeatureKind):
            raise NHLContractError("feature_kind doit être un FeatureKind.")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise NHLContractError(
                "Une variable doit référencer un tuple de preuves non vide."
            )
        if any(not isinstance(item, SourceEvidence) for item in self.evidence):
            raise NHLContractError(
                "Toutes les preuves doivent être des SourceEvidence."
            )
        observation_ids = [item.observation_id for item in self.evidence]
        if len(observation_ids) != len(set(observation_ids)):
            raise NHLContractError(
                "Une variable ne peut pas répéter une même preuve."
            )
        if not isinstance(self.source_games, tuple) or any(
            not isinstance(item, SourceGameResult)
            for item in self.source_games
        ):
            raise NHLContractError(
                "Tous les matchs sources doivent être des SourceGameResult."
            )
        source_game_ids = [item.game_id for item in self.source_games]
        if len(source_game_ids) != len(set(source_game_ids)):
            raise NHLContractError(
                "Une variable ne peut pas répéter un même match source."
            )


@dataclass(frozen=True, slots=True)
class GoalieObservation:
    """Gardien connu ou inconnu pour une équipe et un match cible."""

    game_id: int
    team_id: int
    status: GoalieStatus
    evidence: SourceEvidence
    goalie_id: int | None = None

    def __post_init__(self) -> None:
        _positive_integer(self.game_id, field_name="goalie game_id")
        _positive_integer(self.team_id, field_name="goalie team_id")
        if not isinstance(self.status, GoalieStatus):
            raise NHLContractError("status doit être un GoalieStatus.")
        if not isinstance(self.evidence, SourceEvidence):
            raise NHLContractError("evidence doit être une SourceEvidence.")
        if self.status is GoalieStatus.UNKNOWN:
            if self.goalie_id is not None:
                raise NHLContractError(
                    "Un gardien UNKNOWN ne doit pas avoir d'identifiant."
                )
        elif self.goalie_id is None:
            raise NHLContractError(
                "Un gardien probable ou confirmé doit être identifié."
            )
        else:
            _positive_integer(self.goalie_id, field_name="goalie_id")


@dataclass(frozen=True, slots=True)
class PlayerAvailabilityObservation:
    """Disponibilité pré-match observée d'un joueur identifié."""

    game_id: int
    team_id: int
    player_id: int
    status: PlayerAvailabilityStatus
    evidence: SourceEvidence

    def __post_init__(self) -> None:
        _positive_integer(self.game_id, field_name="availability game_id")
        _positive_integer(self.team_id, field_name="availability team_id")
        _positive_integer(self.player_id, field_name="player_id")
        if not isinstance(self.status, PlayerAvailabilityStatus):
            raise NHLContractError(
                "status doit être un PlayerAvailabilityStatus."
            )
        if not isinstance(self.evidence, SourceEvidence):
            raise NHLContractError("evidence doit être une SourceEvidence.")
