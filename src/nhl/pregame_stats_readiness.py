"""Contrôle conservateur des statistiques NHL disponibles avant les matchs.

Aucune collecte, écriture, prédiction ou autorisation d'entraînement n'est faite ici.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .asof_dataset import NHLAsOfError
from .contracts import require_utc
from .database import validate_nhl_database_path, verify_nhl_database
from .feature_bundle import NHLFeatureBundleError, build_asof_feature_bundle
from .pregame_context_features import PregameContextFeatureError
from .schedule_capture_readiness import audit_schedule_capture
from .team_history_features import TeamHistoryFeatureError
from .temporal_policy import build_information_cutoff


class PregameStatsReadinessError(ValueError):
    """La demande d'audit ne respecte pas les limites de sécurité."""


def _utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PregameStatsReadinessError("Horodatage NHL non canonique.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
        return require_utc(parsed, field_name="capture NHL")
    except ValueError as error:
        raise PregameStatsReadinessError("Horodatage NHL invalide.") from error


def audit_pregame_stats_availability(
    slot: Path,
    *,
    database_path: Path,
    allowed_root: Path,
    lead_minutes: int,
    min_games_per_team: int = 5,
) -> dict[str, object]:
    """Audite chaque match à un cutoff explicite, en lecture seule.

    Une vue as-of réussie ne prouve pas à elle seule la disponibilité historique
    réelle des archives. Le rapport reste donc impropre à entraîner un modèle.
    """
    if type(min_games_per_team) is not int or min_games_per_team <= 0:
        raise PregameStatsReadinessError("Échantillon minimal invalide.")
    database = validate_nhl_database_path(Path(database_path), allowed_root=Path(allowed_root))
    capture = audit_schedule_capture(Path(slot))
    observed = _utc(capture["observed_at_utc"])
    has_database = database.is_file()
    if has_database:
        verify_nhl_database(database, allowed_root=Path(allowed_root))

    games: list[dict[str, object]] = []
    for game in capture["games"]:
        start = _utc(game["scheduled_start_utc"])
        cutoff = build_information_cutoff(start, lead_minutes=lead_minutes)
        row: dict[str, object] = {
            "game_id": game["game_id"],
            "away_team_id": game["away_team_id"],
            "home_team_id": game["home_team_id"],
            "scheduled_start_utc": game["scheduled_start_utc"],
            "information_cutoff_utc": cutoff.isoformat().replace("+00:00", "Z"),
            "capture_before_cutoff": observed <= cutoff,
            "history_sample_present": False,
        }
        if observed > cutoff:
            row["status"] = "CAPTURE_AFTER_CUTOFF"
        elif not has_database:
            row["status"] = "NO_LOCAL_NHL_DATABASE"
        else:
            try:
                bundle = build_asof_feature_bundle(
                    database,
                    allowed_root=Path(allowed_root),
                    target_game_id=game["game_id"],
                    information_cutoff_utc=cutoff,
                    min_games_per_team=min_games_per_team,
                )
            except NHLAsOfError:
                row["status"] = "NO_VALID_ASOF_VIEW"
            except (NHLFeatureBundleError, TeamHistoryFeatureError, PregameContextFeatureError):
                row["status"] = "INVALID_ASOF_FEATURES"
            else:
                row["away_complete_games"] = bundle.history.away.complete_game_count
                row["home_complete_games"] = bundle.history.home.complete_game_count
                row["feature_sha256"] = bundle.feature_sha256
                row["history_sample_present"] = bundle.history.minimum_sample_reached
                row["status"] = (
                    "HISTORY_SAMPLE_PRESENT_UNCERTIFIED"
                    if bundle.history.minimum_sample_reached
                    else "INSUFFICIENT_TEAM_HISTORY"
                )
        games.append(row)
    return {
        "schema_version": "nhl22_pregame_stats_readiness_v1",
        "status": "AUDIT_ONLY_NOT_MODEL_ELIGIBLE",
        "target_date": capture["target_date"],
        "capture_sha256": capture["response_sha256"],
        "capture_observed_at_utc": capture["observed_at_utc"],
        "lead_minutes": lead_minutes,
        "min_games_per_team": min_games_per_team,
        "game_count": len(games),
        "history_sample_present_count": sum(row["history_sample_present"] for row in games),
        "games": games,
        "historical_asof_availability_independently_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }
