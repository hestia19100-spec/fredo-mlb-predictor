"""Lecture seule des prédictions certifiées pour l'interface LPF Edge."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTION_ROOT = Path(
    "shadow_results/logistic_team_form_v1_platt_shadow_v2"
)
CERTIFICATION_ROOT = Path(
    "shadow_certifications/logistic_team_form_v1_platt_shadow_v2"
)
SCORING_ROOT = Path(
    "shadow_scores/"
    "logistic_team_form_v1_platt_shadow_v2_2026_v1"
)
DATABASE_RELATIVE_PATH = Path("data/fredo_mlb.db")

CERTIFIED_STATUS = "PROSPECTIVELY_CERTIFIED_REMOTE_MAIN_BEFORE_GAMES"
COMPLETED_BATCH_STATUS = "COMPLETED_WITH_PREDICTIONS"
PARIS_TIMEZONE = ZoneInfo("Europe/Paris")

PREDICTION_COLUMNS = (
    "prediction_id",
    "batch_id",
    "game_id",
    "occurrence_key",
    "season",
    "official_date_at_prediction",
    "away_team_id",
    "home_team_id",
    "scheduled_start_utc_at_prediction",
    "information_cutoff_utc",
    "issued_at_utc",
    "feature_as_of_date",
    "away_max_source_date",
    "home_max_source_date",
    "away_games_before",
    "away_win_pct_before",
    "away_runs_scored_per_game_before",
    "away_runs_allowed_per_game_before",
    "home_games_before",
    "home_win_pct_before",
    "home_runs_scored_per_game_before",
    "home_runs_allowed_per_game_before",
    "p_home_win",
    "p_away_win",
    "model_version",
    "artifact_sha256",
    "protocol_sha256",
    "code_commit",
)


class LPFEdgeDashboardError(RuntimeError):
    """Signale qu'une preuve affichable est absente ou incohérente."""


@dataclass(frozen=True, slots=True)
class CertifiedPrediction:
    """Une probabilité certifiée, relue sans recalcul du modèle."""

    prediction_id: str
    batch_id: str
    game_id: int
    official_date: date
    away_team_id: int
    home_team_id: int
    scheduled_start_utc: datetime
    issued_at_utc: datetime
    p_home_win: Decimal
    p_away_win: Decimal

    @property
    def predicted_side(self) -> str:
        """Applique uniquement le seuil descriptif déjà préenregistré."""
        return "HOME" if self.p_home_win >= Decimal("0.5") else "AWAY"

    @property
    def predicted_probability(self) -> Decimal:
        """Retourne la probabilité du côté descriptif affiché."""
        if self.predicted_side == "HOME":
            return self.p_home_win
        return self.p_away_win


@dataclass(frozen=True, slots=True)
class CertifiedPredictionDay:
    """Lot journalier dont les octets ont été certifiés sur GitHub."""

    target_date: date
    batch_id: str
    results_commit: str
    certified_at_utc: datetime
    remote_lead_minutes: Decimal
    predictions_sha256: str
    receipt_sha256: str
    predictions: tuple[CertifiedPrediction, ...]


@dataclass(frozen=True, slots=True)
class DailyScoreSummary:
    """Dernier rapport de résultats officiellement fermé pour une date."""

    checkpoint_date: date
    scored_count: int
    void_count: int
    pending_count: int
    correct_count: int
    incorrect_count: int
    accuracy: float | None
    mean_log_loss: float | None
    mean_brier_score: float | None


def _regular_file_bytes(path: Path, *, description: str) -> bytes:
    """Lit un fichier régulier sans accepter de lien symbolique."""
    if path.is_symlink() or not path.is_file():
        raise LPFEdgeDashboardError(f"{description} est absent ou invalide.")
    return path.read_bytes()


def _canonical_json(raw: bytes, *, description: str) -> dict[str, Any]:
    """Relit un objet JSON canonique publié par les moteurs figés."""
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LPFEdgeDashboardError(f"{description} n'est pas un JSON valide.") from error
    if type(value) is not dict:
        raise LPFEdgeDashboardError(f"{description} doit être un objet JSON.")
    canonical = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    if canonical != raw:
        raise LPFEdgeDashboardError(f"{description} n'est pas canonique.")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, *, description: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LPFEdgeDashboardError(f"{description} n'est pas un SHA-256 valide.")
    return value


def _parse_datetime(value: object, *, description: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise LPFEdgeDashboardError(f"{description} n'est pas une date UTC valide.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise LPFEdgeDashboardError(
            f"{description} n'est pas une date UTC valide."
        ) from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise LPFEdgeDashboardError(f"{description} n'est pas en UTC.")
    return parsed


def _parse_probability(value: str, *, description: str) -> Decimal:
    try:
        probability = Decimal(value)
    except InvalidOperation as error:
        raise LPFEdgeDashboardError(f"{description} est invalide.") from error
    if not probability.is_finite() or not Decimal("0") <= probability <= Decimal("1"):
        raise LPFEdgeDashboardError(f"{description} est hors de [0, 1].")
    return probability


def _tree_file_map(certification: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_files = certification.get("results_tree_file_hashes")
    if type(raw_files) is not list:
        raise LPFEdgeDashboardError("La certification ne décrit pas ses fichiers.")
    result: dict[str, dict[str, Any]] = {}
    for item in raw_files:
        if type(item) is not dict or type(item.get("path")) is not str:
            raise LPFEdgeDashboardError("La liste des fichiers certifiés est invalide.")
        name = item["path"]
        if name in result:
            raise LPFEdgeDashboardError("Un fichier est certifié deux fois.")
        result[name] = item
    return result


def list_certified_prediction_dates(
    *, project_directory: Path = PROJECT_ROOT
) -> list[date]:
    """Liste les dates qui possèdent certification et lot local complet."""
    certification_root = project_directory / CERTIFICATION_ROOT
    if certification_root.is_symlink() or not certification_root.is_dir():
        return []

    dates: list[date] = []
    for path in certification_root.iterdir():
        if path.suffix != ".json" or path.is_symlink() or not path.is_file():
            continue
        try:
            target = date.fromisoformat(path.stem)
        except ValueError:
            continue
        prediction_path = project_directory / PREDICTION_ROOT / path.stem / "predictions.csv"
        completed_path = project_directory / PREDICTION_ROOT / path.stem / "COMPLETED"
        if prediction_path.is_file() and not prediction_path.is_symlink() and completed_path.is_file() and not completed_path.is_symlink():
            dates.append(target)
    return sorted(dates)


def load_certified_prediction_day(
    target_date: date,
    *,
    project_directory: Path = PROJECT_ROOT,
) -> CertifiedPredictionDay:
    """Charge un lot certifié et contrôle ses empreintes avant affichage."""
    target_text = target_date.isoformat()
    certification_path = project_directory / CERTIFICATION_ROOT / f"{target_text}.json"
    result_root = project_directory / PREDICTION_ROOT / target_text
    prediction_path = result_root / "predictions.csv"
    receipt_path = result_root / "receipt.json"

    certification = _canonical_json(
        _regular_file_bytes(certification_path, description="La certification"),
        description="La certification",
    )
    if certification.get("status") != CERTIFIED_STATUS:
        raise LPFEdgeDashboardError("Le lot ne possède pas une certification prospective valide.")
    if certification.get("target_official_date") != target_text:
        raise LPFEdgeDashboardError("La certification vise une autre date.")
    if certification.get("remote_response_status_code") != 200:
        raise LPFEdgeDashboardError("La preuve distante n'a pas reçu le statut HTTP 200.")

    try:
        lead_minutes = Decimal(str(certification["remote_publication_lead_minutes"]))
    except (KeyError, InvalidOperation) as error:
        raise LPFEdgeDashboardError("L'avance de certification est invalide.") from error
    if not lead_minutes.is_finite() or lead_minutes < Decimal("60"):
        raise LPFEdgeDashboardError("La certification n'a pas une heure d'avance.")

    tree_files = _tree_file_map(certification)
    if "predictions.csv" not in tree_files or "receipt.json" not in tree_files:
        raise LPFEdgeDashboardError("La certification ne couvre pas les sorties requises.")

    prediction_bytes = _regular_file_bytes(
        prediction_path, description="Le fichier de prédictions"
    )
    receipt_bytes = _regular_file_bytes(receipt_path, description="Le reçu du lot")

    prediction_sha = _require_sha256(
        tree_files["predictions.csv"].get("sha256"),
        description="L'empreinte certifiée des prédictions",
    )
    receipt_sha = _require_sha256(
        tree_files["receipt.json"].get("sha256"),
        description="L'empreinte certifiée du reçu",
    )
    if _sha256(prediction_bytes) != prediction_sha:
        raise LPFEdgeDashboardError("Le fichier de prédictions diffère de sa certification.")
    if _sha256(receipt_bytes) != receipt_sha:
        raise LPFEdgeDashboardError("Le reçu diffère de sa certification.")
    if tree_files["predictions.csv"].get("size_bytes") != len(prediction_bytes):
        raise LPFEdgeDashboardError("La taille certifiée des prédictions est invalide.")
    if tree_files["receipt.json"].get("size_bytes") != len(receipt_bytes):
        raise LPFEdgeDashboardError("La taille certifiée du reçu est invalide.")

    receipt = _canonical_json(receipt_bytes, description="Le reçu du lot")
    if receipt.get("batch", {}).get("status") != COMPLETED_BATCH_STATUS:
        raise LPFEdgeDashboardError("Le reçu ne décrit pas un lot terminé avec prédictions.")
    if receipt.get("batch", {}).get("target_official_date") != target_text:
        raise LPFEdgeDashboardError("Le reçu vise une autre date.")
    if receipt.get("output_hashes", {}).get("predictions_sha256") != prediction_sha:
        raise LPFEdgeDashboardError("Le reçu ne couvre pas les prédictions certifiées.")

    try:
        text = prediction_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LPFEdgeDashboardError("Le CSV des prédictions n'est pas en UTF-8.") from error
    if not text.endswith("\n") or "\r" in text:
        raise LPFEdgeDashboardError("Le CSV des prédictions n'est pas canonique.")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
        raise LPFEdgeDashboardError("Les colonnes des prédictions ont changé.")

    batch_id = certification.get("batch_id")
    if type(batch_id) is not str or len(batch_id) != 64:
        raise LPFEdgeDashboardError("L'identifiant du lot est invalide.")
    predictions: list[CertifiedPrediction] = []
    prediction_ids: set[str] = set()
    game_ids: set[int] = set()
    for line_number, row in enumerate(reader, start=2):
        try:
            prediction_id = row["prediction_id"]
            game_id = int(row["game_id"])
            away_team_id = int(row["away_team_id"])
            home_team_id = int(row["home_team_id"])
            row_date = date.fromisoformat(row["official_date_at_prediction"])
        except (KeyError, TypeError, ValueError) as error:
            raise LPFEdgeDashboardError(
                f"La ligne {line_number} des prédictions est invalide."
            ) from error
        if prediction_id in prediction_ids or game_id in game_ids:
            raise LPFEdgeDashboardError("Une prédiction ou un match apparaît deux fois.")
        if row.get("batch_id") != batch_id or row_date != target_date:
            raise LPFEdgeDashboardError("Une prédiction appartient à un autre lot.")
        if away_team_id <= 0 or home_team_id <= 0 or away_team_id == home_team_id:
            raise LPFEdgeDashboardError("Une identité d'équipe est invalide.")
        p_home = _parse_probability(
            row["p_home_win"], description=f"p_home_win ligne {line_number}"
        )
        p_away = _parse_probability(
            row["p_away_win"], description=f"p_away_win ligne {line_number}"
        )
        p_home_float = float(p_home)
        p_away_float = float(p_away)
        if (
            p_away_float != 1.0 - p_home_float
            or abs(p_home_float + p_away_float - 1.0) > 1e-12
        ):
            raise LPFEdgeDashboardError(
                "Les probabilités d'un match ne totalisent pas 100 %."
            )
        predictions.append(
            CertifiedPrediction(
                prediction_id=prediction_id,
                batch_id=batch_id,
                game_id=game_id,
                official_date=row_date,
                away_team_id=away_team_id,
                home_team_id=home_team_id,
                scheduled_start_utc=_parse_datetime(
                    row["scheduled_start_utc_at_prediction"],
                    description=f"L'heure du match ligne {line_number}",
                ),
                issued_at_utc=_parse_datetime(
                    row["issued_at_utc"],
                    description=f"L'heure d'émission ligne {line_number}",
                ),
                p_home_win=p_home,
                p_away_win=p_away,
            )
        )
        prediction_ids.add(prediction_id)
        game_ids.add(game_id)

    if not predictions:
        raise LPFEdgeDashboardError("Le lot certifié ne contient aucune prédiction.")
    if receipt.get("counts", {}).get("predicted_games") != len(predictions):
        raise LPFEdgeDashboardError("Le compteur du reçu ne correspond pas au CSV.")
    if receipt.get("batch", {}).get("batch_id") != batch_id:
        raise LPFEdgeDashboardError("Le reçu et la certification visent deux lots différents.")

    results_commit = certification.get("results_commit")
    if type(results_commit) is not str or len(results_commit) != 40:
        raise LPFEdgeDashboardError("Le commit certifié est invalide.")

    predictions.sort(key=lambda item: (item.scheduled_start_utc, item.game_id))
    return CertifiedPredictionDay(
        target_date=target_date,
        batch_id=batch_id,
        results_commit=results_commit,
        certified_at_utc=_parse_datetime(
            certification.get("remote_http_date_utc"),
            description="La date de certification",
        ),
        remote_lead_minutes=lead_minutes,
        predictions_sha256=prediction_sha,
        receipt_sha256=receipt_sha,
        predictions=tuple(predictions),
    )


def load_team_names(
    team_ids: Iterable[int],
    *,
    project_directory: Path = PROJECT_ROOT,
) -> dict[int, str]:
    """Relit les noms dans SQLite en mode strictement non modifiable."""
    requested = sorted(set(team_ids))
    if not requested:
        return {}
    database_path = project_directory / DATABASE_RELATIVE_PATH
    if database_path.is_symlink() or not database_path.is_file():
        return {}
    placeholders = ",".join("?" for _ in requested)
    try:
        connection = sqlite3.connect(
            f"file:{database_path.as_posix()}?mode=ro", uri=True
        )
        rows = connection.execute(
            f"SELECT team_id, name FROM teams WHERE team_id IN ({placeholders})",
            requested,
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        if "connection" in locals():
            connection.close()
    return {int(team_id): str(name) for team_id, name in rows}


def team_label(team_id: int, names: dict[int, str]) -> str:
    """Fournit un libellé honnête même sans la base locale."""
    return names.get(team_id, f"Équipe MLB #{team_id}")


def format_paris_datetime(value: datetime) -> str:
    """Affiche une date UTC certifiée dans le fuseau de Paris."""
    return value.astimezone(PARIS_TIMEZONE).strftime("%d/%m/%Y à %H:%M")


def format_paris_time(value: datetime) -> str:
    """Affiche une heure UTC certifiée dans le fuseau de Paris."""
    return value.astimezone(PARIS_TIMEZONE).strftime("%H:%M")


def probability_percent(value: Decimal) -> str:
    """Formate une probabilité sans la présenter comme une cote ou un pari."""
    return f"{float(value) * 100:.1f} %"


def load_latest_score_summary(
    target_date: date,
    *,
    project_directory: Path = PROJECT_ROOT,
) -> DailyScoreSummary | None:
    """Relit le dernier rapport de scoring fermé, s'il existe déjà."""
    observations_root = (
        project_directory
        / SCORING_ROOT
        / target_date.isoformat()
        / "observations"
    )
    if observations_root.is_symlink() or not observations_root.is_dir():
        return None
    candidates: list[tuple[date, Path]] = []
    for path in observations_root.iterdir():
        if path.is_symlink() or not path.is_dir():
            continue
        try:
            checkpoint = date.fromisoformat(path.name)
        except ValueError:
            continue
        if (path / "COMPLETED").is_file() and not (path / "COMPLETED").is_symlink():
            candidates.append((checkpoint, path))
    if not candidates:
        return None
    checkpoint, slot = max(candidates, key=lambda item: item[0])
    report = _canonical_json(
        _regular_file_bytes(slot / "daily_report.json", description="Le rapport quotidien"),
        description="Le rapport quotidien",
    )
    if report.get("target_official_date") != target_date.isoformat():
        raise LPFEdgeDashboardError("Le rapport de résultats vise une autre date.")
    if report.get("checkpoint_utc_date") != checkpoint.isoformat():
        raise LPFEdgeDashboardError("Le rapport et son créneau ne concordent pas.")

    def count(name: str) -> int:
        value = report.get(name)
        if type(value) is not int or value < 0:
            raise LPFEdgeDashboardError(f"Le compteur {name} est invalide.")
        return value

    def metric(name: str) -> float | None:
        value = report.get(name)
        if value is None:
            return None
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise LPFEdgeDashboardError(f"La mesure {name} est invalide.")
        return float(value)

    return DailyScoreSummary(
        checkpoint_date=checkpoint,
        scored_count=count("scored_count"),
        void_count=count("void_count"),
        pending_count=count("pending_count"),
        correct_count=count("correct_count"),
        incorrect_count=count("incorrect_count"),
        accuracy=metric("accuracy"),
        mean_log_loss=metric("mean_log_loss"),
        mean_brier_score=metric("mean_brier_score"),
    )
