"""Verdict immuable d'un journal prospectif LPF Edge et marché français."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from src.lpf_edge_dashboard import (
    DailyScoreSummary,
    PROJECT_ROOT,
    SCORING_ROOT,
)
from src.lpf_edge_market_snapshot import (
    COMPLETED_FILENAME as SNAPSHOT_COMPLETED_FILENAME,
    COMPLETED_SCHEMA as SNAPSHOT_COMPLETED_SCHEMA,
    MARKET_SNAPSHOT_ROOT,
    SNAPSHOT_FILENAME,
    SNAPSHOT_SCHEMA,
)


MARKET_SETTLEMENT_ROOT = Path(
    "market_evaluations/lpf_edge_mlb_market_v1"
)
SETTLEMENT_FILENAME = "market_evaluation.json"
COMPLETED_FILENAME = "COMPLETED"
SETTLEMENT_SCHEMA = "lpf_edge_mlb_market_evaluation_v1"
COMPLETED_SCHEMA = "lpf_edge_mlb_market_evaluation_completed_v1"
SCORING_EVIDENCE_FILENAMES = (
    "COMPLETED",
    "RESERVED",
    "adjudications.csv",
    "daily_report.json",
    "observation_receipt.json",
    "outcome_observation.remote.json.gz",
)
ONE = Decimal("1")
HUNDRED = Decimal("100")


class LPFEdgeMarketSettlementError(RuntimeError):
    """Le verdict LPF/marché ne peut pas être établi sans ambiguïté."""


@dataclass(frozen=True, slots=True)
class MarketSettlementPublication:
    """Deux preuves locales prêtes à être publiées avec le scoring."""

    target_date: date
    slot_path: Path
    settlement_path: Path
    completed_path: Path
    settlement_sha256: str
    snapshot_sha256: str
    evaluated_count: int
    correct_count: int
    missing_market_count: int
    void_count: int
    theoretical_net_units: Decimal
    theoretical_roi_percent: Decimal | None

    @property
    def paths(self) -> tuple[Path, Path]:
        return (self.completed_path, self.settlement_path)


@dataclass(frozen=True, slots=True)
class SealedMarketSettlement:
    """Résumé relu depuis un verdict prospectif déjà publié."""

    target_date: date
    checkpoint_date: date
    observation_id: str
    settlement_sha256: str
    snapshot_sha256: str
    evaluated_count: int
    correct_count: int
    missing_market_count: int
    void_count: int
    accuracy_percent: Decimal | None
    theoretical_net_units: Decimal
    theoretical_roi_percent: Decimal | None


def _canonical_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise LPFEdgeMarketSettlementError(
            "Le verdict LPF/marché ne peut pas être sérialisé."
        ) from error
    return text.encode("utf-8") + b"\n"


def _regular_bytes(path: Path, description: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise LPFEdgeMarketSettlementError(
            f"{description} est absent ou invalide."
        )
    try:
        return path.read_bytes()
    except OSError as error:
        raise LPFEdgeMarketSettlementError(
            f"{description} ne peut pas être relu."
        ) from error


def _canonical_object(raw: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LPFEdgeMarketSettlementError(
            f"{description} n'est pas un JSON valide."
        ) from error
    if type(value) is not dict or raw != _canonical_bytes(value):
        raise LPFEdgeMarketSettlementError(
            f"{description} n'est pas canonique."
        )
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(value: object, description: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LPFEdgeMarketSettlementError(f"{description} est invalide.")
    return value


def _commit_text(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LPFEdgeMarketSettlementError(
            "Le commit parent du scoring est invalide."
        )
    return value


def _decimal(value: object, description: str) -> Decimal:
    if type(value) is not str:
        raise LPFEdgeMarketSettlementError(f"{description} est invalide.")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise LPFEdgeMarketSettlementError(
            f"{description} est invalide."
        ) from error
    if not parsed.is_finite():
        raise LPFEdgeMarketSettlementError(f"{description} est invalide.")
    return parsed


def _decimal_text(value: Decimal, description: str) -> str:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise LPFEdgeMarketSettlementError(f"{description} est invalide.")
    return format(value, "f")


def _optional_decimal(value: object, description: str) -> Decimal | None:
    return None if value is None else _decimal(value, description)


def _nonnegative_int(value: object, description: str) -> int:
    if type(value) is not int or value < 0:
        raise LPFEdgeMarketSettlementError(f"{description} est invalide.")
    return value


def _require_safe_directory(path: Path, project: Path) -> None:
    try:
        relative = path.relative_to(project)
    except ValueError as error:
        raise LPFEdgeMarketSettlementError(
            "Le dossier du verdict sort du projet."
        ) from error
    cursor = project
    for part in relative.parts:
        cursor /= part
        if not cursor.exists():
            continue
        try:
            mode = cursor.lstat().st_mode
        except OSError as error:
            raise LPFEdgeMarketSettlementError(
                "Le dossier du verdict ne peut pas être contrôlé."
            ) from error
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise LPFEdgeMarketSettlementError(
                "Le chemin du verdict contient un élément interdit."
            )


def _write_new_file(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError as error:
        raise LPFEdgeMarketSettlementError(
            "Un fichier du verdict existe déjà."
        ) from error
    except OSError as error:
        raise LPFEdgeMarketSettlementError(
            "Un fichier du verdict ne peut pas être écrit."
        ) from error
    if _regular_bytes(path, "Le fichier du verdict") != content:
        raise LPFEdgeMarketSettlementError(
            "Les octets relus du verdict ont changé."
        )


def _load_snapshot(
    target_date: date,
    project: Path,
) -> tuple[dict[str, Any], str]:
    slot = project / MARKET_SNAPSHOT_ROOT / target_date.isoformat()
    if slot.is_symlink() or not slot.is_dir():
        raise LPFEdgeMarketSettlementError(
            "Le journal prospectif immuable est absent."
        )
    entries = sorted(path.name for path in slot.iterdir())
    if entries != sorted((SNAPSHOT_COMPLETED_FILENAME, SNAPSHOT_FILENAME)):
        raise LPFEdgeMarketSettlementError(
            "Le journal prospectif contient des fichiers inattendus."
        )
    snapshot_raw = _regular_bytes(
        slot / SNAPSHOT_FILENAME,
        "Le journal prospectif",
    )
    completed_raw = _regular_bytes(
        slot / SNAPSHOT_COMPLETED_FILENAME,
        "Le marqueur du journal prospectif",
    )
    snapshot = _canonical_object(snapshot_raw, "Le journal prospectif")
    completed = _canonical_object(
        completed_raw,
        "Le marqueur du journal prospectif",
    )
    snapshot_sha256 = _sha256(snapshot_raw)
    target_text = target_date.isoformat()
    if (
        snapshot.get("schema") != SNAPSHOT_SCHEMA
        or snapshot.get("target_official_date") != target_text
        or completed.get("schema") != SNAPSHOT_COMPLETED_SCHEMA
        or completed.get("status") != "COMPLETED"
        or completed.get("target_official_date") != target_text
        or completed.get("snapshot_filename") != SNAPSHOT_FILENAME
        or completed.get("snapshot_sha256") != snapshot_sha256
    ):
        raise LPFEdgeMarketSettlementError(
            "Les preuves du journal prospectif ne concordent pas."
        )
    return snapshot, snapshot_sha256


def _scoring_evidence(
    target_date: date,
    checkpoint_date: date,
    project: Path,
) -> tuple[dict[str, str], Path]:
    slot = (
        project
        / SCORING_ROOT
        / target_date.isoformat()
        / "observations"
        / checkpoint_date.isoformat()
    )
    if slot.is_symlink() or not slot.is_dir():
        raise LPFEdgeMarketSettlementError(
            "Le dossier du scoring vérifié est absent."
        )
    entries = sorted(path.name for path in slot.iterdir())
    if entries != sorted(SCORING_EVIDENCE_FILENAMES):
        raise LPFEdgeMarketSettlementError(
            "Le dossier du scoring contient des fichiers inattendus."
        )
    hashes = {
        filename: _sha256(
            _regular_bytes(slot / filename, f"La preuve {filename}")
        )
        for filename in SCORING_EVIDENCE_FILENAMES
    }
    return hashes, slot


def _validate_score_counts(score: DailyScoreSummary) -> None:
    if score.pending_count != 0:
        raise LPFEdgeMarketSettlementError(
            "Le verdict attend que tous les matchs soient tranchés ou annulés."
        )
    counts = {
        "scored_count": sum(
            result.classification_correct is not None for result in score.results
        ),
        "void_count": sum(
            result.classification_correct is None for result in score.results
        ),
        "correct_count": sum(
            result.classification_correct is True for result in score.results
        ),
        "incorrect_count": sum(
            result.classification_correct is False for result in score.results
        ),
    }
    if (
        score.scored_count != counts["scored_count"]
        or score.void_count != counts["void_count"]
        or score.correct_count != counts["correct_count"]
        or score.incorrect_count != counts["incorrect_count"]
        or score.scored_count != score.correct_count + score.incorrect_count
        or len(score.results) != score.scored_count + score.void_count
    ):
        raise LPFEdgeMarketSettlementError(
            "Les compteurs du scoring vérifié sont incohérents."
        )


def create_market_settlement_publication(
    target_date: date,
    score: DailyScoreSummary,
    *,
    observation_id: str,
    publication_parent_commit: str,
    project_directory: Path,
) -> MarketSettlementPublication:
    """Scelle le résultat réel sans modifier le journal prospectif."""
    if not isinstance(target_date, date):
        raise TypeError("target_date doit être une date exacte.")
    if not isinstance(score, DailyScoreSummary):
        raise TypeError("score doit être un rapport quotidien vérifié.")
    observation = _sha256_text(
        observation_id,
        "L'identifiant de l'observation",
    )
    parent_commit = _commit_text(publication_parent_commit)
    _validate_score_counts(score)
    try:
        project = Path(project_directory).resolve(strict=True)
    except OSError as error:
        raise LPFEdgeMarketSettlementError(
            "Le projet du verdict est inaccessible."
        ) from error

    snapshot, snapshot_sha256 = _load_snapshot(target_date, project)
    evidence_hashes, evidence_slot = _scoring_evidence(
        target_date,
        score.checkpoint_date,
        project,
    )
    snapshot_rows = snapshot.get("predictions")
    if type(snapshot_rows) is not list or not snapshot_rows:
        raise LPFEdgeMarketSettlementError(
            "Le journal prospectif ne contient aucune prédiction."
        )
    by_game: dict[int, dict[str, Any]] = {}
    for row in snapshot_rows:
        if type(row) is not dict:
            raise LPFEdgeMarketSettlementError(
                "Une ligne du journal prospectif est invalide."
            )
        game_id = row.get("game_id")
        if type(game_id) is not int or game_id <= 0 or game_id in by_game:
            raise LPFEdgeMarketSettlementError(
                "Un match du journal prospectif est invalide ou répété."
            )
        by_game[game_id] = row
    results_by_game = {result.game_id: result for result in score.results}
    if (
        len(results_by_game) != len(score.results)
        or set(results_by_game) != set(by_game)
    ):
        raise LPFEdgeMarketSettlementError(
            "Le journal et les résultats ne couvrent pas les mêmes matchs."
        )

    rows: list[dict[str, object]] = []
    evaluated_count = 0
    correct_count = 0
    missing_market_count = 0
    void_count = 0
    theoretical_net = Decimal("0")
    for game_id in sorted(by_game):
        journal = by_game[game_id]
        result = results_by_game[game_id]
        model_probability = _decimal(
            journal.get("model_probability"),
            "La probabilité LPF figée",
        )
        if (
            journal.get("prediction_id") != result.prediction_id
            or journal.get("away_team_id") != result.away_team_id
            or journal.get("home_team_id") != result.home_team_id
            or journal.get("predicted_side") != result.predicted_side
            or model_probability != result.predicted_probability
        ):
            raise LPFEdgeMarketSettlementError(
                "Un résultat diverge de son pronostic prospectif."
            )

        gap = _optional_decimal(
            journal.get("gap_percentage_points"),
            "L'écart LPF–marché figé",
        )
        odds = _optional_decimal(
            journal.get("best_decimal_odds"),
            "La meilleure cote figée",
        )
        market_probability = _optional_decimal(
            journal.get("french_market_probability"),
            "La probabilité du marché figée",
        )
        if result.classification_correct is None:
            status = "VOID"
            net_units = None
            void_count += 1
        elif gap is None or odds is None or market_probability is None:
            status = "MISSING_MARKET"
            net_units = None
            missing_market_count += 1
        else:
            if odds <= ONE:
                raise LPFEdgeMarketSettlementError(
                    "La meilleure cote figée est invalide."
                )
            status = "EVALUATED"
            net_units = odds - ONE if result.classification_correct else -ONE
            evaluated_count += 1
            correct_count += result.classification_correct
            theoretical_net += net_units

        rows.append(
            {
                "actual_winner": result.actual_winner,
                "away_score": result.away_score,
                "away_team_id": result.away_team_id,
                "away_team_name": journal.get("away_team_name"),
                "best_bookmakers": journal.get("best_bookmakers"),
                "best_decimal_odds": (
                    None if odds is None else _decimal_text(odds, "La cote")
                ),
                "classification_correct": result.classification_correct,
                "evaluation_status": status,
                "french_market_probability": (
                    None
                    if market_probability is None
                    else _decimal_text(
                        market_probability,
                        "La probabilité du marché",
                    )
                ),
                "game_id": game_id,
                "gap_percentage_points": (
                    None
                    if gap is None
                    else _decimal_text(gap, "L'écart LPF–marché")
                ),
                "home_score": result.home_score,
                "home_team_id": result.home_team_id,
                "home_team_name": journal.get("home_team_name"),
                "model_probability": _decimal_text(
                    model_probability,
                    "La probabilité LPF",
                ),
                "outcome_status": result.outcome_status,
                "predicted_side": result.predicted_side,
                "predicted_team_name": journal.get("predicted_team_name"),
                "prediction_id": result.prediction_id,
                "theoretical_net_units": (
                    None
                    if net_units is None
                    else _decimal_text(net_units, "Le résultat théorique")
                ),
            }
        )

    accuracy = (
        None
        if evaluated_count == 0
        else Decimal(correct_count) / Decimal(evaluated_count) * HUNDRED
    )
    roi = (
        None
        if evaluated_count == 0
        else theoretical_net / Decimal(evaluated_count) * HUNDRED
    )
    payload = {
        "accuracy_percent": (
            None
            if accuracy is None
            else _decimal_text(accuracy, "La réussite")
        ),
        "checkpoint_utc_date": score.checkpoint_date.isoformat(),
        "correct_count": correct_count,
        "evaluated_count": evaluated_count,
        "missing_market_count": missing_market_count,
        "observation_id": observation,
        "prediction_batch_id": _sha256_text(
            snapshot.get("batch_id"),
            "L'identifiant du lot prospectif",
        ),
        "publication_parent_commit": parent_commit,
        "results": rows,
        "schema": SETTLEMENT_SCHEMA,
        "scoring_evidence_relative_path": evidence_slot.relative_to(
            project
        ).as_posix(),
        "scoring_evidence_sha256": evidence_hashes,
        "source_market_snapshot_sha256": snapshot_sha256,
        "target_official_date": target_date.isoformat(),
        "theoretical_net_units": _decimal_text(
            theoretical_net,
            "Le résultat théorique total",
        ),
        "theoretical_roi_percent": (
            None if roi is None else _decimal_text(roi, "Le rendement théorique")
        ),
        "void_count": void_count,
    }
    settlement_bytes = _canonical_bytes(payload)
    settlement_sha256 = _sha256(settlement_bytes)
    completed = {
        "correct_count": correct_count,
        "evaluated_count": evaluated_count,
        "missing_market_count": missing_market_count,
        "observation_id": observation,
        "schema": COMPLETED_SCHEMA,
        "settlement_filename": SETTLEMENT_FILENAME,
        "settlement_sha256": settlement_sha256,
        "source_market_snapshot_sha256": snapshot_sha256,
        "status": "COMPLETED",
        "target_official_date": target_date.isoformat(),
        "void_count": void_count,
    }

    root = project / MARKET_SETTLEMENT_ROOT
    slot = root / target_date.isoformat()
    _require_safe_directory(root, project)
    if slot.exists() or slot.is_symlink():
        raise LPFEdgeMarketSettlementError(
            "Le créneau immuable du verdict existe déjà."
        )
    try:
        slot.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise LPFEdgeMarketSettlementError(
            "Le créneau du verdict ne peut pas être réservé."
        ) from error
    settlement_path = slot / SETTLEMENT_FILENAME
    completed_path = slot / COMPLETED_FILENAME
    _write_new_file(settlement_path, settlement_bytes)
    _write_new_file(completed_path, _canonical_bytes(completed))
    return MarketSettlementPublication(
        target_date=target_date,
        slot_path=slot,
        settlement_path=settlement_path,
        completed_path=completed_path,
        settlement_sha256=settlement_sha256,
        snapshot_sha256=snapshot_sha256,
        evaluated_count=evaluated_count,
        correct_count=correct_count,
        missing_market_count=missing_market_count,
        void_count=void_count,
        theoretical_net_units=theoretical_net,
        theoretical_roi_percent=roi,
    )


def load_market_settlement(
    target_date: date,
    *,
    project_directory: Path = PROJECT_ROOT,
) -> SealedMarketSettlement | None:
    """Relit et vérifie un verdict publié, sans réseau ni écriture."""
    if not isinstance(target_date, date):
        raise TypeError("target_date doit être une date exacte.")
    project = Path(project_directory).resolve(strict=True)
    slot = project / MARKET_SETTLEMENT_ROOT / target_date.isoformat()
    if not slot.exists():
        return None
    if slot.is_symlink() or not slot.is_dir():
        raise LPFEdgeMarketSettlementError(
            "Le créneau du verdict est invalide."
        )
    entries = sorted(path.name for path in slot.iterdir())
    if entries != sorted((COMPLETED_FILENAME, SETTLEMENT_FILENAME)):
        raise LPFEdgeMarketSettlementError(
            "Le verdict publié contient des fichiers inattendus."
        )
    settlement_raw = _regular_bytes(
        slot / SETTLEMENT_FILENAME,
        "Le verdict LPF/marché",
    )
    completed_raw = _regular_bytes(
        slot / COMPLETED_FILENAME,
        "Le marqueur du verdict LPF/marché",
    )
    settlement = _canonical_object(settlement_raw, "Le verdict LPF/marché")
    completed = _canonical_object(
        completed_raw,
        "Le marqueur du verdict LPF/marché",
    )
    digest = _sha256(settlement_raw)
    target_text = target_date.isoformat()
    if (
        settlement.get("schema") != SETTLEMENT_SCHEMA
        or settlement.get("target_official_date") != target_text
        or completed.get("schema") != COMPLETED_SCHEMA
        or completed.get("status") != "COMPLETED"
        or completed.get("target_official_date") != target_text
        or completed.get("settlement_filename") != SETTLEMENT_FILENAME
        or completed.get("settlement_sha256") != digest
        or completed.get("source_market_snapshot_sha256")
        != settlement.get("source_market_snapshot_sha256")
    ):
        raise LPFEdgeMarketSettlementError(
            "Les preuves du verdict LPF/marché ne concordent pas."
        )
    evaluated = _nonnegative_int(
        settlement.get("evaluated_count"),
        "Le nombre de pronostics évalués",
    )
    correct = _nonnegative_int(
        settlement.get("correct_count"),
        "Le nombre de pronostics réussis",
    )
    missing = _nonnegative_int(
        settlement.get("missing_market_count"),
        "Le nombre de cotes manquantes",
    )
    void = _nonnegative_int(
        settlement.get("void_count"),
        "Le nombre de matchs annulés",
    )
    if correct > evaluated:
        raise LPFEdgeMarketSettlementError(
            "Le nombre de réussites du verdict est incohérent."
        )
    if any(
        completed.get(name) != settlement.get(name)
        for name in (
            "correct_count",
            "evaluated_count",
            "missing_market_count",
            "observation_id",
            "void_count",
        )
    ):
        raise LPFEdgeMarketSettlementError(
            "Les compteurs du marqueur et du verdict divergent."
        )
    try:
        checkpoint = date.fromisoformat(
            settlement.get("checkpoint_utc_date", "")
        )
    except (TypeError, ValueError) as error:
        raise LPFEdgeMarketSettlementError(
            "La date de contrôle du verdict est invalide."
        ) from error
    _, source_snapshot_sha256 = _load_snapshot(target_date, project)
    if source_snapshot_sha256 != settlement.get(
        "source_market_snapshot_sha256"
    ):
        raise LPFEdgeMarketSettlementError(
            "Le journal prospectif lié au verdict a changé."
        )
    scoring_hashes, scoring_slot = _scoring_evidence(
        target_date,
        checkpoint,
        project,
    )
    if (
        settlement.get("scoring_evidence_relative_path")
        != scoring_slot.relative_to(project).as_posix()
        or settlement.get("scoring_evidence_sha256") != scoring_hashes
    ):
        raise LPFEdgeMarketSettlementError(
            "Les preuves du scoring liées au verdict ont changé."
        )
    _commit_text(settlement.get("publication_parent_commit"))
    rows = settlement.get("results")
    if type(rows) is not list:
        raise LPFEdgeMarketSettlementError(
            "Les résultats détaillés du verdict sont invalides."
        )
    statuses = [
        row.get("evaluation_status") if type(row) is dict else None
        for row in rows
    ]
    if (
        len(rows) != evaluated + missing + void
        or statuses.count("EVALUATED") != evaluated
        or statuses.count("MISSING_MARKET") != missing
        or statuses.count("VOID") != void
        or any(
            status not in {"EVALUATED", "MISSING_MARKET", "VOID"}
            for status in statuses
        )
    ):
        raise LPFEdgeMarketSettlementError(
            "Les catégories détaillées du verdict sont incohérentes."
        )
    evaluated_rows = [
        row for row in rows if row["evaluation_status"] == "EVALUATED"
    ]
    recomputed_correct = sum(
        row.get("classification_correct") is True
        for row in evaluated_rows
    )
    if (
        any(type(row.get("classification_correct")) is not bool for row in evaluated_rows)
        or recomputed_correct != correct
    ):
        raise LPFEdgeMarketSettlementError(
            "Les réussites détaillées du verdict sont incohérentes."
        )
    recomputed_net = sum(
        (
            _decimal(
                row.get("theoretical_net_units"),
                "Un résultat théorique détaillé",
            )
            for row in evaluated_rows
        ),
        Decimal("0"),
    )
    net = _decimal(
        settlement.get("theoretical_net_units"),
        "Le résultat théorique du verdict",
    )
    accuracy = _optional_decimal(
        settlement.get("accuracy_percent"),
        "La réussite du verdict",
    )
    roi = _optional_decimal(
        settlement.get("theoretical_roi_percent"),
        "Le rendement théorique du verdict",
    )
    expected_accuracy = (
        None
        if evaluated == 0
        else Decimal(correct) / Decimal(evaluated) * HUNDRED
    )
    expected_roi = (
        None if evaluated == 0 else net / Decimal(evaluated) * HUNDRED
    )
    if (
        recomputed_net != net
        or accuracy != expected_accuracy
        or roi != expected_roi
    ):
        raise LPFEdgeMarketSettlementError(
            "Les totaux du verdict ne correspondent pas au détail."
        )
    return SealedMarketSettlement(
        target_date=target_date,
        checkpoint_date=checkpoint,
        observation_id=_sha256_text(
            settlement.get("observation_id"),
            "L'identifiant de l'observation",
        ),
        settlement_sha256=digest,
        snapshot_sha256=_sha256_text(
            settlement.get("source_market_snapshot_sha256"),
            "L'empreinte du journal prospectif",
        ),
        evaluated_count=evaluated,
        correct_count=correct,
        missing_market_count=missing,
        void_count=void,
        accuracy_percent=accuracy,
        theoretical_net_units=net,
        theoretical_roi_percent=roi,
    )


__all__ = [
    "LPFEdgeMarketSettlementError",
    "MARKET_SETTLEMENT_ROOT",
    "MarketSettlementPublication",
    "SealedMarketSettlement",
    "create_market_settlement_publication",
    "load_market_settlement",
]
