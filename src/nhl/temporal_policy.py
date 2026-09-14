"""Politique stricte contre les fuites temporelles de LPF Edge NHL."""

from __future__ import annotations

from datetime import datetime, timedelta

from src.nhl.contracts import (
    FeatureProvenance,
    GoalieObservation,
    PlayerAvailabilityObservation,
    ScheduledGameObservation,
    require_utc,
)


class NHLTemporalLeakageError(RuntimeError):
    """Signale une information indisponible au moment de la prédiction."""


def build_information_cutoff(
    scheduled_start_utc: datetime,
    *,
    lead_minutes: int,
) -> datetime:
    """Calcule un cutoff déterministe strictement antérieur au match."""
    require_utc(scheduled_start_utc, field_name="scheduled_start_utc")
    if (
        isinstance(lead_minutes, bool)
        or not isinstance(lead_minutes, int)
        or lead_minutes <= 0
    ):
        raise ValueError("lead_minutes doit être un entier strictement positif.")
    return scheduled_start_utc - timedelta(minutes=lead_minutes)


def validate_information_cutoff(
    game: ScheduledGameObservation,
    information_cutoff_utc: datetime,
) -> None:
    """Vérifie que le match et son horaire étaient connus avant le cutoff."""
    require_utc(
        information_cutoff_utc,
        field_name="information_cutoff_utc",
    )
    if information_cutoff_utc >= game.scheduled_start_utc:
        raise NHLTemporalLeakageError(
            "Le cutoff doit être strictement antérieur au début prévu."
        )
    if game.evidence.effective_available_at_utc > information_cutoff_utc:
        raise NHLTemporalLeakageError(
            "La rencontre n'était pas encore prouvée au cutoff."
        )


def _validate_evidence_before_cutoff(
    *,
    effective_available_at_utc: datetime,
    information_cutoff_utc: datetime,
    description: str,
) -> None:
    if effective_available_at_utc > information_cutoff_utc:
        raise NHLTemporalLeakageError(
            f"{description} n'était pas disponible au cutoff."
        )


def validate_feature_provenance(
    *,
    game: ScheduledGameObservation,
    provenance: FeatureProvenance,
    information_cutoff_utc: datetime,
) -> None:
    """Refuse toute preuve ou tout résultat disponible trop tard."""
    validate_information_cutoff(game, information_cutoff_utc)
    if provenance.target_game_id != game.game_id:
        raise NHLTemporalLeakageError(
            "La provenance ne vise pas le match contrôlé."
        )

    for evidence in provenance.evidence:
        _validate_evidence_before_cutoff(
            effective_available_at_utc=evidence.effective_available_at_utc,
            information_cutoff_utc=information_cutoff_utc,
            description=f"La preuve {evidence.observation_id}",
        )

    for source_game in provenance.source_games:
        if source_game.game_id == game.game_id:
            raise NHLTemporalLeakageError(
                "Le match cible ne peut jamais alimenter ses propres variables."
            )
        if source_game.final_observed_at_utc > information_cutoff_utc:
            raise NHLTemporalLeakageError(
                "Un résultat source n'était pas final au cutoff."
            )


def validate_goalie_observation(
    *,
    game: ScheduledGameObservation,
    goalie: GoalieObservation,
    information_cutoff_utc: datetime,
) -> None:
    """Valide uniquement un statut de gardien réellement observé à temps."""
    validate_information_cutoff(game, information_cutoff_utc)
    if goalie.game_id != game.game_id:
        raise NHLTemporalLeakageError(
            "Le gardien ne correspond pas au match contrôlé."
        )
    if goalie.team_id not in {game.away_team_id, game.home_team_id}:
        raise NHLTemporalLeakageError(
            "Le gardien n'appartient à aucune équipe du match."
        )
    _validate_evidence_before_cutoff(
        effective_available_at_utc=goalie.evidence.effective_available_at_utc,
        information_cutoff_utc=information_cutoff_utc,
        description="Le statut du gardien",
    )


def validate_player_availability_observation(
    *,
    game: ScheduledGameObservation,
    availability: PlayerAvailabilityObservation,
    information_cutoff_utc: datetime,
) -> None:
    """Valide une absence ou disponibilité connue avant le match."""
    validate_information_cutoff(game, information_cutoff_utc)
    if availability.game_id != game.game_id:
        raise NHLTemporalLeakageError(
            "La disponibilité du joueur vise un autre match."
        )
    if availability.team_id not in {game.away_team_id, game.home_team_id}:
        raise NHLTemporalLeakageError(
            "Le joueur n'appartient à aucune équipe du match."
        )
    _validate_evidence_before_cutoff(
        effective_available_at_utc=(
            availability.evidence.effective_available_at_utc
        ),
        information_cutoff_utc=information_cutoff_utc,
        description="La disponibilité du joueur",
    )
