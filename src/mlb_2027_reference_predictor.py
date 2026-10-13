"""Calcule des probabilités MLB 2027 en mémoire, sans publication.

Entrées fournies par un orchestrateur qui doit vérifier la provenance et
l'instant de disponibilité des résultats. Le SHA du reçu ne suffit pas à les
certifier. Les sorties ne sont ni des conseils de pari ni des pronostics publiés.
Le moteur et les preuves Shadow V2 2026 ne sont jamais modifiés.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import sys
from threading import Lock
from pathlib import Path

import numpy as np
from sklearn import __version__ as SKLEARN_VERSION

from src.calibrated_model import (
    CalibratedModelArtifact,
    DEFAULT_ARTIFACT_PATH,
    load_calibrated_artifact,
)
from src.mlb_2027_reference_readiness import (
    DayReadiness,
    PriorFinalGame,
    REFERENCE_ARTIFACT_SHA256,
    REFERENCE_MODEL_VERSION,
    UpcomingGame,
    plan_reference_readiness,
)
from src.training_dataset import FEATURE_COLUMNS


_MODEL_LOADING_LOCK = Lock()


def load_pinned_reference_artifact(path: Path) -> CalibratedModelArtifact:
    """Vérifie le SHA puis charge l'ancien pickle avec son alias historique.

    Le modèle 2026 a été sérialisé avec une classe située dans __main__.
    L'alias temporaire est retiré même en cas d'erreur. On ne modifie pas
    l'artefact, le moteur 2026 ni le module de chargement partagé.
    """
    main = sys.modules.get("__main__")
    if main is None:
        raise MLB2027PredictionError("Module principal absent.")
    missing = object()
    with _MODEL_LOADING_LOCK:
        previous = vars(main).get("CalibratedModelArtifact", missing)
        vars(main)["CalibratedModelArtifact"] = CalibratedModelArtifact
        try:
            artifact = load_calibrated_artifact(
                path, expected_sha256=REFERENCE_ARTIFACT_SHA256,
            )
        finally:
            if previous is missing:
                vars(main).pop("CalibratedModelArtifact", None)
            else:
                vars(main)["CalibratedModelArtifact"] = previous
    if (
        artifact.numpy_version != np.__version__
        or artifact.sklearn_version != SKLEARN_VERSION
    ):
        raise MLB2027PredictionError("Versions du runtime incompatibles.")
    return artifact


class MLB2027PredictionError(ValueError):
    """Le calcul du candidat de référence n'est pas sûr."""


@dataclass(frozen=True, slots=True)
class ReferenceProbability:
    game_id: int
    away_team_id: int
    home_team_id: int
    feature_values: tuple[float, ...]
    p_away_win: float
    p_home_win: float
    model_version: str
    artifact_sha256: str


@dataclass(frozen=True, slots=True)
class ReferenceCalculation:
    readiness: DayReadiness
    probabilities: tuple[ReferenceProbability, ...]

    @property
    def abstained_count(self) -> int:
        return len(self.readiness.games) - len(self.probabilities)


def _six_places(numerator: int, denominator: int) -> float:
    # Identique à la quantification .6f du calcul Shadow V2.
    return float(format(numerator / denominator, ".6f"))


def _team_features(
    team_id: int, history: tuple[PriorFinalGame, ...],
) -> tuple[float, float, float, float]:
    games = wins = scored = allowed = 0
    for game in history:
        if game.away_team_id == team_id:
            games += 1
            wins += int(game.away_score > game.home_score)
            scored += game.away_score
            allowed += game.home_score
        elif game.home_team_id == team_id:
            games += 1
            wins += int(game.home_score > game.away_score)
            scored += game.home_score
            allowed += game.away_score
    if games < 10:
        raise MLB2027PredictionError("Historique admissible insuffisant.")
    return (
        float(games),
        _six_places(wins, games),
        _six_places(scored, games),
        _six_places(allowed, games),
    )


def calculate_reference_probabilities(
    target_date: date,
    upcoming_games: tuple[UpcomingGame, ...],
    prior_final_games: tuple[PriorFinalGame, ...],
    *,
    source_receipt_sha256: str,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
) -> ReferenceCalculation:
    """Prédit seulement les candidats ayant 10 finales 2027 par équipe.

    La préparation et les variables partagent exactement les mêmes entrées.
    Aucun réentraînement, cote, sélection, écriture ou publication n'a lieu.
    L'appelant doit encore contrôler le reçu source et le délai T−60.
    """
    readiness = plan_reference_readiness(
        target_date, upcoming_games, prior_final_games,
        source_receipt_sha256=source_receipt_sha256,
    )
    eligible_ids = {
        game.game_id for game in readiness.games
        if game.status == "REFERENCE_ELIGIBLE"
    }
    if not eligible_ids:
        return ReferenceCalculation(readiness=readiness, probabilities=())

    candidates = sorted(
        (game for game in upcoming_games if game.game_id in eligible_ids),
        key=lambda game: (game.scheduled_start_utc, game.game_id),
    )
    by_team: dict[int, tuple[float, float, float, float]] = {}
    for game in candidates:
        for team_id in (game.away_team_id, game.home_team_id):
            if team_id not in by_team:
                by_team[team_id] = _team_features(team_id, prior_final_games)
    rows = tuple(
        by_team[game.away_team_id] + by_team[game.home_team_id]
        for game in candidates
    )
    if tuple(FEATURE_COLUMNS) != (
        "away_games_before", "away_win_pct_before",
        "away_runs_scored_per_game_before",
        "away_runs_allowed_per_game_before", "home_games_before",
        "home_win_pct_before", "home_runs_scored_per_game_before",
        "home_runs_allowed_per_game_before",
    ):
        raise MLB2027PredictionError("Ordre des variables inattendu.")

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(candidates), 8) or not np.isfinite(matrix).all():
        raise MLB2027PredictionError("Matrice de variables invalide.")

    # Le chargeur vérifie SHA-256 avant la désérialisation joblib.
    artifact = load_pinned_reference_artifact(artifact_path)
    if (
        artifact.calibrated_model_version != REFERENCE_MODEL_VERSION
        or tuple(artifact.feature_columns) != tuple(FEATURE_COLUMNS)
    ):
        raise MLB2027PredictionError("Artefact incompatible avec les variables.")
    classifier = artifact.calibrated_classifier
    if tuple(classifier.classes_) != (0, 1):
        raise MLB2027PredictionError("Étiquettes de classe inattendues.")
    values = np.asarray(classifier.predict_proba(matrix), dtype=np.float64)
    if (
        values.shape != (len(candidates), 2)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-12)
    ):
        raise MLB2027PredictionError("Probabilités invalides.")

    probabilities = tuple(
        ReferenceProbability(
            game_id=game.game_id,
            away_team_id=game.away_team_id,
            home_team_id=game.home_team_id,
            feature_values=rows[index],
            p_away_win=float(values[index, 0]),
            p_home_win=float(values[index, 1]),
            model_version=REFERENCE_MODEL_VERSION,
            artifact_sha256=REFERENCE_ARTIFACT_SHA256,
        )
        for index, game in enumerate(candidates)
    )
    return ReferenceCalculation(readiness=readiness, probabilities=probabilities)
