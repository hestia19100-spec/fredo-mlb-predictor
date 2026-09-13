"""Scellement prospectif du journal LPF Edge et marché français."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Mapping

from src.lpf_edge_dashboard import CertifiedPredictionDay
from src.lpf_edge_market_comparison import (
    LPFEdgeMarketComparisonError,
    build_french_market_comparison,
)
from src.lpf_edge_odds_display import MoneylineOddsDisplay


MARKET_SNAPSHOT_ROOT = Path(
    "market_snapshots/lpf_edge_mlb_market_v1"
)
SNAPSHOT_FILENAME = "market_snapshot.json"
COMPLETED_FILENAME = "COMPLETED"
SNAPSHOT_SCHEMA = "lpf_edge_mlb_market_snapshot_v1"
COMPLETED_SCHEMA = "lpf_edge_mlb_market_snapshot_completed_v1"


class LPFEdgeMarketSnapshotError(RuntimeError):
    """Le journal prospectif ne peut pas être scellé sans ambiguïté."""


@dataclass(frozen=True, slots=True)
class MarketSnapshotPublication:
    """Preuve locale prête à être publiée avec la certification."""

    target_date: date
    slot_path: Path
    snapshot_path: Path
    completed_path: Path
    snapshot_sha256: str
    odds_run_id: int | None
    row_count: int
    comparable_count: int

    @property
    def paths(self) -> tuple[Path, Path]:
        return (self.completed_path, self.snapshot_path)


def _canonical_json_file_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise LPFEdgeMarketSnapshotError(
            "Le journal de marché ne peut pas être sérialisé."
        ) from error
    return text.encode("utf-8") + b"\n"


def _utc_text(value: datetime, description: str) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise LPFEdgeMarketSnapshotError(
            f"{description} n’est pas horodaté."
        )
    return value.astimezone(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _decimal_text(value: Decimal, description: str) -> str:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise LPFEdgeMarketSnapshotError(f"{description} est invalide.")
    return format(value, "f")


def _validate_sha256(value: str, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LPFEdgeMarketSnapshotError(f"{description} est invalide.")
    return value


def _validate_commit(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LPFEdgeMarketSnapshotError(
            "Le commit des prédictions est invalide."
        )
    return value


def _require_safe_directory(path: Path, project: Path) -> None:
    try:
        relative = path.relative_to(project)
    except ValueError as error:
        raise LPFEdgeMarketSnapshotError(
            "Le dossier du journal sort du projet."
        ) from error
    cursor = project
    for part in relative.parts:
        cursor /= part
        if not cursor.exists():
            continue
        try:
            mode = cursor.lstat().st_mode
        except OSError as error:
            raise LPFEdgeMarketSnapshotError(
                "Le dossier du journal ne peut pas être contrôlé."
            ) from error
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise LPFEdgeMarketSnapshotError(
                "Le chemin du journal contient un élément interdit."
            )


def _write_new_file(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError as error:
        raise LPFEdgeMarketSnapshotError(
            "Un fichier du journal existe déjà."
        ) from error
    except OSError as error:
        raise LPFEdgeMarketSnapshotError(
            "Un fichier du journal ne peut pas être écrit."
        ) from error
    try:
        persisted = path.read_bytes()
    except OSError as error:
        raise LPFEdgeMarketSnapshotError(
            "Un fichier du journal ne peut pas être relu."
        ) from error
    if persisted != content:
        raise LPFEdgeMarketSnapshotError(
            "Les octets relus du journal ont changé."
        )


def create_market_snapshot_publication(
    prediction_day: CertifiedPredictionDay,
    odds_display: MoneylineOddsDisplay,
    *,
    team_names: Mapping[int, str],
    certification_sha256: str,
    certification_evidence_sha256: str,
    project_directory: Path,
) -> MarketSnapshotPublication:
    """Crée deux fichiers immuables sans réseau, modèle ni mutation SQLite."""
    batch_id = _validate_sha256(
        prediction_day.batch_id,
        "L’identifiant du lot de prédictions",
    )
    try:
        comparison = build_french_market_comparison(
            prediction_day,
            odds_display,
            team_names=team_names,
        )
    except LPFEdgeMarketComparisonError as error:
        raise LPFEdgeMarketSnapshotError(
            "La comparaison LPF/marché du journal est incohérente."
        ) from error
    if len(comparison.rows) != len(prediction_day.predictions):
        raise LPFEdgeMarketSnapshotError(
            "Le journal ne couvre pas toutes les prédictions certifiées."
        )

    predictions_by_game = {
        prediction.game_id: prediction
        for prediction in prediction_day.predictions
    }
    if len(predictions_by_game) != len(prediction_day.predictions):
        raise LPFEdgeMarketSnapshotError(
            "Une prédiction certifiée est répétée dans le journal."
        )
    odds_by_game = {game.game_id: game for game in odds_display.games}
    if len(odds_by_game) != len(odds_display.games):
        raise LPFEdgeMarketSnapshotError(
            "Un match de cotes est répété dans le journal."
        )

    rows: list[dict[str, object]] = []
    for comparison_row in comparison.rows:
        prediction = predictions_by_game[comparison_row.game_id]
        odds_game = odds_by_game.get(comparison_row.game_id)
        quotes = []
        if odds_game is not None:
            for quote in sorted(
                odds_game.bookmaker_quotes,
                key=lambda item: (item.key.casefold(), item.title.casefold()),
            ):
                quotes.append(
                    {
                        "away_decimal_odds": _decimal_text(
                            quote.away_decimal_odds,
                            "La cote extérieure",
                        ),
                        "bookmaker_key": quote.key,
                        "bookmaker_title": quote.title,
                        "home_decimal_odds": _decimal_text(
                            quote.home_decimal_odds,
                            "La cote domicile",
                        ),
                        "last_update_utc": _utc_text(
                            quote.last_update_utc,
                            "L’actualisation du bookmaker",
                        ),
                    }
                )

        rows.append(
            {
                "away_team_id": prediction.away_team_id,
                "away_team_name": comparison_row.away_team_name,
                "best_bookmakers": list(comparison_row.best_bookmakers),
                "best_decimal_odds": (
                    None
                    if comparison_row.best_decimal_odds is None
                    else _decimal_text(
                        comparison_row.best_decimal_odds,
                        "La meilleure cote",
                    )
                ),
                "bookmaker_count": comparison_row.bookmaker_count,
                "bookmaker_quotes": quotes,
                "french_market_probability": (
                    None
                    if comparison_row.french_market_probability is None
                    else _decimal_text(
                        comparison_row.french_market_probability,
                        "La probabilité du marché",
                    )
                ),
                "game_id": prediction.game_id,
                "gap_percentage_points": (
                    None
                    if comparison_row.gap_percentage_points is None
                    else _decimal_text(
                        comparison_row.gap_percentage_points,
                        "L’écart LPF–marché",
                    )
                ),
                "home_team_id": prediction.home_team_id,
                "home_team_name": comparison_row.home_team_name,
                "issued_at_utc": _utc_text(
                    prediction.issued_at_utc,
                    "L’émission de la prédiction",
                ),
                "model_probability": _decimal_text(
                    comparison_row.model_probability,
                    "La probabilité LPF",
                ),
                "p_away_win": _decimal_text(
                    prediction.p_away_win,
                    "La probabilité extérieure",
                ),
                "p_home_win": _decimal_text(
                    prediction.p_home_win,
                    "La probabilité domicile",
                ),
                "predicted_side": comparison_row.predicted_side,
                "predicted_team_name": comparison_row.predicted_team_name,
                "prediction_id": prediction.prediction_id,
                "scheduled_start_utc": _utc_text(
                    prediction.scheduled_start_utc,
                    "L’heure du match",
                ),
            }
        )

    payload = {
        "batch_id": batch_id,
        "certification_evidence_sha256": _validate_sha256(
            certification_evidence_sha256,
            "L’empreinte de la preuve distante de certification",
        ),
        "certification_sha256": _validate_sha256(
            certification_sha256,
            "L’empreinte de la certification",
        ),
        "certified_at_utc": _utc_text(
            prediction_day.certified_at_utc,
            "La certification",
        ),
        "comparable_count": comparison.comparable_count,
        "odds_completed_at_utc": (
            None
            if comparison.odds_completed_at_utc is None
            else _utc_text(
                comparison.odds_completed_at_utc,
                "La collecte française",
            )
        ),
        "odds_region": odds_display.region,
        "odds_run_id": comparison.odds_run_id,
        "prediction_receipt_sha256": _validate_sha256(
            prediction_day.receipt_sha256,
            "L’empreinte du reçu de prédiction",
        ),
        "prediction_results_commit": _validate_commit(
            prediction_day.results_commit
        ),
        "predictions": rows,
        "predictions_sha256": _validate_sha256(
            prediction_day.predictions_sha256,
            "L’empreinte des prédictions",
        ),
        "schema": SNAPSHOT_SCHEMA,
        "target_official_date": prediction_day.target_date.isoformat(),
    }
    snapshot_bytes = _canonical_json_file_bytes(payload)
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    completed = {
        "batch_id": batch_id,
        "comparable_count": comparison.comparable_count,
        "odds_run_id": comparison.odds_run_id,
        "prediction_results_commit": prediction_day.results_commit,
        "row_count": len(rows),
        "schema": COMPLETED_SCHEMA,
        "snapshot_filename": SNAPSHOT_FILENAME,
        "snapshot_sha256": snapshot_sha256,
        "status": "COMPLETED",
        "target_official_date": prediction_day.target_date.isoformat(),
    }
    completed_bytes = _canonical_json_file_bytes(completed)

    try:
        project = Path(project_directory).resolve(strict=True)
    except OSError as error:
        raise LPFEdgeMarketSnapshotError(
            "Le projet du journal est inaccessible."
        ) from error
    root = project / MARKET_SNAPSHOT_ROOT
    slot = root / prediction_day.target_date.isoformat()
    _require_safe_directory(root, project)
    if slot.exists() or slot.is_symlink():
        raise LPFEdgeMarketSnapshotError(
            "Le créneau immuable du journal existe déjà."
        )
    try:
        slot.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise LPFEdgeMarketSnapshotError(
            "Le créneau du journal ne peut pas être réservé."
        ) from error

    snapshot_path = slot / SNAPSHOT_FILENAME
    completed_path = slot / COMPLETED_FILENAME
    _write_new_file(snapshot_path, snapshot_bytes)
    _write_new_file(completed_path, completed_bytes)
    return MarketSnapshotPublication(
        target_date=prediction_day.target_date,
        slot_path=slot,
        snapshot_path=snapshot_path,
        completed_path=completed_path,
        snapshot_sha256=snapshot_sha256,
        odds_run_id=comparison.odds_run_id,
        row_count=len(rows),
        comparable_count=comparison.comparable_count,
    )


__all__ = [
    "COMPLETED_FILENAME",
    "LPFEdgeMarketSnapshotError",
    "MARKET_SNAPSHOT_ROOT",
    "MarketSnapshotPublication",
    "SNAPSHOT_FILENAME",
    "create_market_snapshot_publication",
]
