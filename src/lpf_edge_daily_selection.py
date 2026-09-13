"""Sélection prospective et explicable des pronostics Moneyline LPF Edge."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from src.lpf_edge_dashboard import PROJECT_ROOT
from src.lpf_edge_market_snapshot import (
    COMPLETED_FILENAME as SNAPSHOT_COMPLETED_FILENAME,
    COMPLETED_SCHEMA as SNAPSHOT_COMPLETED_SCHEMA,
    MARKET_SNAPSHOT_ROOT,
    SNAPSHOT_FILENAME,
    SNAPSHOT_SCHEMA,
)


DAILY_SELECTION_ROOT = Path(
    "daily_selections/lpf_edge_mlb_moneyline_v1"
)
SELECTION_FILENAME = "daily_selection.json"
COMPLETED_FILENAME = "COMPLETED"
SELECTION_SCHEMA = "lpf_edge_mlb_moneyline_selection_v1"
COMPLETED_SCHEMA = "lpf_edge_mlb_moneyline_selection_completed_v1"
POLICY_VERSION = "lpf_edge_mlb_moneyline_policy_v1"
MAXIMUM_SELECTIONS = 2
MINIMUM_BOOKMAKERS = 2
MINIMUM_MODEL_PROBABILITY = Decimal("0.52")
MINIMUM_GAP_PERCENTAGE_POINTS = Decimal("2.0")
MINIMUM_EXPECTED_VALUE_PERCENT = Decimal("3.0")
MINIMUM_DECIMAL_ODDS = Decimal("1.35")
MAXIMUM_DECIMAL_ODDS = Decimal("3.00")
HUNDRED = Decimal("100")
ONE = Decimal("1")
ZERO = Decimal("0")


class LPFEdgeDailySelectionError(RuntimeError):
    """La sélection prospective ne peut pas être établie sûrement."""


@dataclass(frozen=True, slots=True)
class DailySelectionPick:
    """Un match retenu par la politique prospective figée."""

    rank: int
    role: str
    game_id: int
    scheduled_start_utc: str
    home_team_name: str
    away_team_name: str
    predicted_side: str
    predicted_team_name: str
    model_probability: Decimal
    french_market_probability: Decimal
    gap_percentage_points: Decimal
    best_decimal_odds: Decimal
    best_bookmakers: tuple[str, ...]
    expected_value_percent: Decimal
    home_probable_pitcher_name: str
    away_probable_pitcher_name: str


@dataclass(frozen=True, slots=True)
class DailySelectionPublication:
    """Preuve locale prête à être publiée avec la certification."""

    target_date: date
    slot_path: Path
    selection_path: Path
    completed_path: Path
    selection_sha256: str
    market_snapshot_sha256: str
    selection_count: int
    eligible_count: int

    @property
    def paths(self) -> tuple[Path, Path]:
        return (self.completed_path, self.selection_path)


@dataclass(frozen=True, slots=True)
class SealedDailySelection:
    """Sélection prospective relue et vérifiée pour l'interface."""

    target_date: date
    selection_sha256: str
    market_snapshot_sha256: str
    status: str
    eligible_count: int
    picks: tuple[DailySelectionPick, ...]
    rejection_reasons: tuple[tuple[str, int], ...]

    @property
    def selection_count(self) -> int:
        return len(self.picks)


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
        raise LPFEdgeDailySelectionError(
            "La sélection ne peut pas être sérialisée."
        ) from error
    return text.encode("utf-8") + b"\n"


def _regular_bytes(path: Path, description: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise LPFEdgeDailySelectionError(f"{description} est absent ou invalide.")
    try:
        return path.read_bytes()
    except OSError as error:
        raise LPFEdgeDailySelectionError(
            f"{description} ne peut pas être relu."
        ) from error


def _canonical_object(raw: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LPFEdgeDailySelectionError(
            f"{description} n'est pas un JSON valide."
        ) from error
    if type(value) is not dict or raw != _canonical_bytes(value):
        raise LPFEdgeDailySelectionError(f"{description} n'est pas canonique.")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(value: object, description: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LPFEdgeDailySelectionError(f"{description} est invalide.")
    return value


def _decimal(value: object, description: str) -> Decimal:
    if type(value) is not str:
        raise LPFEdgeDailySelectionError(f"{description} est invalide.")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise LPFEdgeDailySelectionError(f"{description} est invalide.") from error
    if not parsed.is_finite():
        raise LPFEdgeDailySelectionError(f"{description} est invalide.")
    return parsed


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise LPFEdgeDailySelectionError("Une valeur décimale est invalide.")
    return format(value, "f")


def _bookmakers(value: object) -> tuple[str, ...]:
    if (
        type(value) is not list
        or any(type(item) is not str or not item.strip() for item in value)
    ):
        raise LPFEdgeDailySelectionError(
            "La liste des meilleurs bookmakers est invalide."
        )
    cleaned = tuple(item.strip() for item in value)
    if len(set(cleaned)) != len(cleaned):
        raise LPFEdgeDailySelectionError(
            "Un meilleur bookmaker est répété."
        )
    return cleaned


def _pitcher_name(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value.strip():
        raise LPFEdgeDailySelectionError("Un nom de lanceur est invalide.")
    return value.strip()


def _require_safe_directory(path: Path, project: Path) -> None:
    try:
        relative = path.relative_to(project)
    except ValueError as error:
        raise LPFEdgeDailySelectionError(
            "Le dossier de sélection sort du projet."
        ) from error
    cursor = project
    for part in relative.parts:
        cursor /= part
        if not cursor.exists():
            continue
        mode = cursor.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise LPFEdgeDailySelectionError(
                "Le chemin de sélection contient un élément interdit."
            )


def _write_new_file(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError as error:
        raise LPFEdgeDailySelectionError(
            "Un fichier de sélection existe déjà."
        ) from error
    except OSError as error:
        raise LPFEdgeDailySelectionError(
            "Un fichier de sélection ne peut pas être écrit."
        ) from error
    if _regular_bytes(path, "Le fichier de sélection") != content:
        raise LPFEdgeDailySelectionError(
            "Les octets relus de la sélection ont changé."
        )


def _load_market_snapshot(
    target_date: date,
    project: Path,
) -> tuple[dict[str, Any], str]:
    slot = project / MARKET_SNAPSHOT_ROOT / target_date.isoformat()
    if slot.is_symlink() or not slot.is_dir():
        raise LPFEdgeDailySelectionError(
            "Le journal prospectif LPF/marché est absent."
        )
    if sorted(path.name for path in slot.iterdir()) != sorted(
        (SNAPSHOT_COMPLETED_FILENAME, SNAPSHOT_FILENAME)
    ):
        raise LPFEdgeDailySelectionError(
            "Le journal prospectif contient des fichiers inattendus."
        )
    snapshot_raw = _regular_bytes(slot / SNAPSHOT_FILENAME, "Le journal prospectif")
    completed_raw = _regular_bytes(
        slot / SNAPSHOT_COMPLETED_FILENAME,
        "Le marqueur du journal prospectif",
    )
    snapshot = _canonical_object(snapshot_raw, "Le journal prospectif")
    completed = _canonical_object(completed_raw, "Le marqueur du journal prospectif")
    digest = _sha256(snapshot_raw)
    target_text = target_date.isoformat()
    if (
        snapshot.get("schema") != SNAPSHOT_SCHEMA
        or snapshot.get("target_official_date") != target_text
        or completed.get("schema") != SNAPSHOT_COMPLETED_SCHEMA
        or completed.get("status") != "COMPLETED"
        or completed.get("target_official_date") != target_text
        or completed.get("snapshot_filename") != SNAPSHOT_FILENAME
        or completed.get("snapshot_sha256") != digest
    ):
        raise LPFEdgeDailySelectionError(
            "Les preuves du journal prospectif ne concordent pas."
        )
    return snapshot, digest


def _policy_payload() -> dict[str, object]:
    return {
        "maximum_selections": MAXIMUM_SELECTIONS,
        "maximum_decimal_odds": _decimal_text(MAXIMUM_DECIMAL_ODDS),
        "minimum_bookmakers": MINIMUM_BOOKMAKERS,
        "minimum_decimal_odds": _decimal_text(MINIMUM_DECIMAL_ODDS),
        "minimum_expected_value_percent": _decimal_text(
            MINIMUM_EXPECTED_VALUE_PERCENT
        ),
        "minimum_gap_percentage_points": _decimal_text(
            MINIMUM_GAP_PERCENTAGE_POINTS
        ),
        "minimum_model_probability": _decimal_text(
            MINIMUM_MODEL_PROBABILITY
        ),
        "probable_pitchers_required": True,
        "ranking": "EXPECTED_VALUE_THEN_GAP_THEN_PROBABILITY",
        "version": POLICY_VERSION,
    }


def _decision_for_row(
    row: dict[str, Any],
    pitchers: tuple[str | None, str | None],
) -> dict[str, object]:
    game_id = row.get("game_id")
    if type(game_id) is not int or game_id <= 0:
        raise LPFEdgeDailySelectionError("Un identifiant de match est invalide.")
    model_probability = _decimal(
        row.get("model_probability"),
        "La probabilité LPF",
    )
    gap_value = row.get("gap_percentage_points")
    odds_value = row.get("best_decimal_odds")
    market_value = row.get("french_market_probability")
    gap = None if gap_value is None else _decimal(gap_value, "L'écart LPF–marché")
    odds = None if odds_value is None else _decimal(odds_value, "La meilleure cote")
    market_probability = (
        None
        if market_value is None
        else _decimal(market_value, "La probabilité du marché")
    )
    bookmaker_count = row.get("bookmaker_count")
    if type(bookmaker_count) is not int or bookmaker_count < 0:
        raise LPFEdgeDailySelectionError(
            "Le nombre de bookmakers est invalide."
        )
    bookmakers = _bookmakers(row.get("best_bookmakers"))
    away_pitcher = _pitcher_name(pitchers[0])
    home_pitcher = _pitcher_name(pitchers[1])
    reasons: list[str] = []
    if away_pitcher is None or home_pitcher is None:
        reasons.append("LANCEURS_INCOMPLETS")
    if gap is None or odds is None or market_probability is None:
        reasons.append("COTES_ABSENTES")
        expected_value = None
    else:
        if odds <= ONE:
            raise LPFEdgeDailySelectionError("La meilleure cote est invalide.")
        expected_value = (model_probability * odds - ONE) * HUNDRED
        if bookmaker_count < MINIMUM_BOOKMAKERS:
            reasons.append("BOOKMAKERS_INSUFFISANTS")
        if model_probability < MINIMUM_MODEL_PROBABILITY:
            reasons.append("PROBABILITE_LPF_TROP_FAIBLE")
        if gap < MINIMUM_GAP_PERCENTAGE_POINTS:
            reasons.append("ECART_LPF_MARCHE_INSUFFISANT")
        if expected_value < MINIMUM_EXPECTED_VALUE_PERCENT:
            reasons.append("VALEUR_THEORIQUE_INSUFFISANTE")
        if odds < MINIMUM_DECIMAL_ODDS or odds > MAXIMUM_DECIMAL_ODDS:
            reasons.append("COTE_HORS_PLAGE")
    if bookmaker_count == 0 and "COTES_ABSENTES" not in reasons:
        raise LPFEdgeDailySelectionError(
            "Une cote existe sans bookmaker observé."
        )
    return {
        "away_probable_pitcher_name": away_pitcher,
        "away_team_name": row.get("away_team_name"),
        "best_bookmakers": list(bookmakers),
        "best_decimal_odds": None if odds is None else _decimal_text(odds),
        "bookmaker_count": bookmaker_count,
        "decision_reasons": reasons,
        "expected_value_percent": (
            None if expected_value is None else _decimal_text(expected_value)
        ),
        "french_market_probability": (
            None
            if market_probability is None
            else _decimal_text(market_probability)
        ),
        "game_id": game_id,
        "gap_percentage_points": None if gap is None else _decimal_text(gap),
        "home_probable_pitcher_name": home_pitcher,
        "home_team_name": row.get("home_team_name"),
        "model_probability": _decimal_text(model_probability),
        "predicted_side": row.get("predicted_side"),
        "predicted_team_name": row.get("predicted_team_name"),
        "prediction_id": row.get("prediction_id"),
        "scheduled_start_utc": row.get("scheduled_start_utc"),
        "selection_rank": None,
        "selection_role": None,
        "selection_status": "REJECTED",
    }


def create_daily_selection_publication(
    target_date: date,
    *,
    probable_pitchers_by_game: Mapping[
        int,
        tuple[str | None, str | None],
    ],
    project_directory: Path,
) -> DailySelectionPublication:
    """Crée une sélection prospective figée, sans pari ni appel externe."""
    if not isinstance(target_date, date):
        raise TypeError("target_date doit être une date exacte.")
    project = Path(project_directory).resolve(strict=True)
    snapshot, snapshot_sha256 = _load_market_snapshot(target_date, project)
    snapshot_rows = snapshot.get("predictions")
    if type(snapshot_rows) is not list or not snapshot_rows:
        raise LPFEdgeDailySelectionError(
            "Le journal prospectif ne contient aucune prédiction."
        )
    snapshot_game_ids = [row.get("game_id") for row in snapshot_rows if type(row) is dict]
    if (
        len(snapshot_game_ids) != len(snapshot_rows)
        or any(type(game_id) is not int for game_id in snapshot_game_ids)
        or len(set(snapshot_game_ids)) != len(snapshot_game_ids)
        or set(probable_pitchers_by_game) != set(snapshot_game_ids)
    ):
        raise LPFEdgeDailySelectionError(
            "Les lanceurs et le journal ne couvrent pas les mêmes matchs."
        )
    decisions: list[dict[str, object]] = []
    for row in snapshot_rows:
        game_id = row["game_id"]
        pitchers = probable_pitchers_by_game[game_id]
        if type(pitchers) is not tuple or len(pitchers) != 2:
            raise LPFEdgeDailySelectionError(
                "La paire de lanceurs d'un match est invalide."
            )
        decisions.append(_decision_for_row(row, pitchers))

    eligible = [item for item in decisions if not item["decision_reasons"]]
    eligible.sort(
        key=lambda item: (
            -_decimal(item["expected_value_percent"], "La valeur théorique"),
            -_decimal(item["gap_percentage_points"], "L'écart LPF–marché"),
            -_decimal(item["model_probability"], "La probabilité LPF"),
            item["game_id"],
        )
    )
    selected_ids = {
        item["game_id"] for item in eligible[:MAXIMUM_SELECTIONS]
    }
    selected_rank = {
        item["game_id"]: index
        for index, item in enumerate(eligible[:MAXIMUM_SELECTIONS], start=1)
    }
    for item in decisions:
        game_id = item["game_id"]
        if game_id in selected_ids:
            rank = selected_rank[game_id]
            item["selection_rank"] = rank
            item["selection_role"] = "PRINCIPAL" if rank == 1 else "SECONDAIRE"
            item["selection_status"] = "SELECTED"
        elif not item["decision_reasons"]:
            item["decision_reasons"] = ["LIMITE_DE_DEUX_SELECTIONS"]
    decisions.sort(key=lambda item: item["game_id"])
    selected = sorted(
        (item for item in decisions if item["selection_status"] == "SELECTED"),
        key=lambda item: item["selection_rank"],
    )
    payload = {
        "certified_at_utc": snapshot.get("certified_at_utc"),
        "decision_count": len(decisions),
        "decisions": decisions,
        "eligible_count": len(eligible),
        "mode": "OBSERVATION",
        "odds_run_id": snapshot.get("odds_run_id"),
        "policy": _policy_payload(),
        "prediction_batch_id": _sha256_text(
            snapshot.get("batch_id"),
            "L'identifiant du lot prospectif",
        ),
        "schema": SELECTION_SCHEMA,
        "selected_game_ids": [item["game_id"] for item in selected],
        "selection_count": len(selected),
        "status": "PICKS_AVAILABLE" if selected else "NO_PICK",
        "target_official_date": target_date.isoformat(),
        "warning": (
            "Sélection expérimentale en observation : aucune mise réelle "
            "ni garantie de gain."
        ),
        "source_market_snapshot_sha256": snapshot_sha256,
    }
    selection_bytes = _canonical_bytes(payload)
    selection_sha256 = _sha256(selection_bytes)
    completed = {
        "eligible_count": len(eligible),
        "mode": "OBSERVATION",
        "schema": COMPLETED_SCHEMA,
        "selection_count": len(selected),
        "selection_filename": SELECTION_FILENAME,
        "selection_sha256": selection_sha256,
        "source_market_snapshot_sha256": snapshot_sha256,
        "status": "COMPLETED",
        "target_official_date": target_date.isoformat(),
    }
    root = project / DAILY_SELECTION_ROOT
    slot = root / target_date.isoformat()
    _require_safe_directory(root, project)
    if slot.exists() or slot.is_symlink():
        raise LPFEdgeDailySelectionError(
            "Le créneau immuable de sélection existe déjà."
        )
    slot.mkdir(parents=True, exist_ok=False)
    selection_path = slot / SELECTION_FILENAME
    completed_path = slot / COMPLETED_FILENAME
    _write_new_file(selection_path, selection_bytes)
    _write_new_file(completed_path, _canonical_bytes(completed))
    return DailySelectionPublication(
        target_date=target_date,
        slot_path=slot,
        selection_path=selection_path,
        completed_path=completed_path,
        selection_sha256=selection_sha256,
        market_snapshot_sha256=snapshot_sha256,
        selection_count=len(selected),
        eligible_count=len(eligible),
    )


def _pick_from_decision(value: object) -> DailySelectionPick:
    if type(value) is not dict or value.get("selection_status") != "SELECTED":
        raise LPFEdgeDailySelectionError("Une sélection détaillée est invalide.")
    rank = value.get("selection_rank")
    game_id = value.get("game_id")
    if type(rank) is not int or rank not in {1, 2} or type(game_id) is not int:
        raise LPFEdgeDailySelectionError("Le rang d'une sélection est invalide.")
    expected_role = "PRINCIPAL" if rank == 1 else "SECONDAIRE"
    if value.get("selection_role") != expected_role:
        raise LPFEdgeDailySelectionError("Le rôle d'une sélection est invalide.")
    text_fields = (
        "scheduled_start_utc",
        "home_team_name",
        "away_team_name",
        "predicted_side",
        "predicted_team_name",
        "home_probable_pitcher_name",
        "away_probable_pitcher_name",
    )
    if any(type(value.get(name)) is not str or not value[name].strip() for name in text_fields):
        raise LPFEdgeDailySelectionError("Une sélection est incomplète.")
    model_probability = _decimal(
        value.get("model_probability"),
        "La probabilité LPF",
    )
    market_probability = _decimal(
        value.get("french_market_probability"),
        "La probabilité du marché",
    )
    if not ZERO <= model_probability <= ONE or not ZERO <= market_probability <= ONE:
        raise LPFEdgeDailySelectionError("Une probabilité de sélection est invalide.")
    predicted_side = value["predicted_side"]
    expected_team = (
        value["home_team_name"]
        if predicted_side == "HOME"
        else value["away_team_name"]
        if predicted_side == "AWAY"
        else None
    )
    if expected_team is None or value["predicted_team_name"] != expected_team:
        raise LPFEdgeDailySelectionError("L'équipe retenue est incohérente.")
    try:
        scheduled_start = datetime.fromisoformat(
            value["scheduled_start_utc"].replace("Z", "+00:00")
        )
    except ValueError as error:
        raise LPFEdgeDailySelectionError("L'heure d'une sélection est invalide.") from error
    if scheduled_start.tzinfo is None or scheduled_start.utcoffset() != timedelta(0):
        raise LPFEdgeDailySelectionError("L'heure d'une sélection n'est pas en UTC.")
    return DailySelectionPick(
        rank=rank,
        role=expected_role,
        game_id=game_id,
        scheduled_start_utc=value["scheduled_start_utc"],
        home_team_name=value["home_team_name"],
        away_team_name=value["away_team_name"],
        predicted_side=predicted_side,
        predicted_team_name=value["predicted_team_name"],
        model_probability=model_probability,
        french_market_probability=market_probability,
        gap_percentage_points=_decimal(
            value.get("gap_percentage_points"),
            "L'écart LPF–marché",
        ),
        best_decimal_odds=_decimal(value.get("best_decimal_odds"), "La meilleure cote"),
        best_bookmakers=_bookmakers(value.get("best_bookmakers")),
        expected_value_percent=_decimal(
            value.get("expected_value_percent"),
            "La valeur théorique",
        ),
        home_probable_pitcher_name=value["home_probable_pitcher_name"],
        away_probable_pitcher_name=value["away_probable_pitcher_name"],
    )


def load_daily_selection(
    target_date: date,
    *,
    project_directory: Path = PROJECT_ROOT,
) -> SealedDailySelection | None:
    """Relit une sélection prospective sans recalcul ni effet de bord."""
    project = Path(project_directory).resolve(strict=True)
    slot = project / DAILY_SELECTION_ROOT / target_date.isoformat()
    if not slot.exists():
        return None
    if slot.is_symlink() or not slot.is_dir():
        raise LPFEdgeDailySelectionError("Le créneau de sélection est invalide.")
    if sorted(path.name for path in slot.iterdir()) != sorted(
        (COMPLETED_FILENAME, SELECTION_FILENAME)
    ):
        raise LPFEdgeDailySelectionError(
            "La sélection publiée contient des fichiers inattendus."
        )
    raw = _regular_bytes(slot / SELECTION_FILENAME, "La sélection prospective")
    marker_raw = _regular_bytes(slot / COMPLETED_FILENAME, "Le marqueur de sélection")
    payload = _canonical_object(raw, "La sélection prospective")
    marker = _canonical_object(marker_raw, "Le marqueur de sélection")
    digest = _sha256(raw)
    target_text = target_date.isoformat()
    snapshot, snapshot_sha256 = _load_market_snapshot(target_date, project)
    if (
        payload.get("schema") != SELECTION_SCHEMA
        or payload.get("mode") != "OBSERVATION"
        or payload.get("target_official_date") != target_text
        or payload.get("source_market_snapshot_sha256") != snapshot_sha256
        or payload.get("prediction_batch_id") != snapshot.get("batch_id")
        or payload.get("policy") != _policy_payload()
        or marker.get("schema") != COMPLETED_SCHEMA
        or marker.get("status") != "COMPLETED"
        or marker.get("target_official_date") != target_text
        or marker.get("selection_filename") != SELECTION_FILENAME
        or marker.get("selection_sha256") != digest
        or marker.get("source_market_snapshot_sha256") != snapshot_sha256
    ):
        raise LPFEdgeDailySelectionError(
            "Les preuves de la sélection prospective ne concordent pas."
        )
    decisions = payload.get("decisions")
    selection_count = payload.get("selection_count")
    eligible_count = payload.get("eligible_count")
    if (
        type(decisions) is not list
        or type(selection_count) is not int
        or selection_count < 0
        or selection_count > MAXIMUM_SELECTIONS
        or type(eligible_count) is not int
        or eligible_count < selection_count
        or marker.get("selection_count") != selection_count
        or marker.get("eligible_count") != eligible_count
    ):
        raise LPFEdgeDailySelectionError(
            "Les compteurs de la sélection prospective sont incohérents."
        )
    selected_values = [
        value
        for value in decisions
        if type(value) is dict and value.get("selection_status") == "SELECTED"
    ]
    picks = tuple(
        sorted((_pick_from_decision(value) for value in selected_values), key=lambda item: item.rank)
    )
    if len(decisions) != len(snapshot.get("predictions", [])):
        raise LPFEdgeDailySelectionError(
            "Les décisions ne couvrent pas exactement le journal prospectif."
        )
    decision_game_ids: list[int] = []
    rejection_counts: dict[str, int] = {}
    eligible_decision_count = 0
    for value in decisions:
        if type(value) is not dict or type(value.get("game_id")) is not int:
            raise LPFEdgeDailySelectionError("Une décision prospective est invalide.")
        decision_game_ids.append(value["game_id"])
        reasons = value.get("decision_reasons")
        if (
            type(reasons) is not list
            or any(type(reason) is not str or not reason for reason in reasons)
            or len(reasons) != len(set(reasons))
        ):
            raise LPFEdgeDailySelectionError("Les motifs d'une décision sont invalides.")
        if value.get("selection_status") == "SELECTED":
            if reasons:
                raise LPFEdgeDailySelectionError(
                    "Un match retenu contient un motif de rejet."
                )
            eligible_decision_count += 1
        elif value.get("selection_status") == "REJECTED" and reasons:
            if reasons == ["LIMITE_DE_DEUX_SELECTIONS"]:
                eligible_decision_count += 1
            for reason in reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        else:
            raise LPFEdgeDailySelectionError("Le statut d'une décision est invalide.")
    snapshot_game_ids = [row.get("game_id") for row in snapshot["predictions"]]
    if (
        len(picks) != selection_count
        or tuple(pick.rank for pick in picks) != tuple(range(1, selection_count + 1))
        or payload.get("selected_game_ids") != [pick.game_id for pick in picks]
        or payload.get("status") != ("PICKS_AVAILABLE" if picks else "NO_PICK")
        or len(set(decision_game_ids)) != len(decision_game_ids)
        or set(decision_game_ids) != set(snapshot_game_ids)
        or eligible_decision_count != eligible_count
    ):
        raise LPFEdgeDailySelectionError(
            "Les matchs retenus par la sélection sont incohérents."
        )
    return SealedDailySelection(
        target_date=target_date,
        selection_sha256=digest,
        market_snapshot_sha256=snapshot_sha256,
        status=payload["status"],
        eligible_count=eligible_count,
        picks=picks,
        rejection_reasons=tuple(sorted(rejection_counts.items())),
    )


__all__ = [
    "DAILY_SELECTION_ROOT",
    "DailySelectionPick",
    "DailySelectionPublication",
    "LPFEdgeDailySelectionError",
    "MAXIMUM_SELECTIONS",
    "POLICY_VERSION",
    "SealedDailySelection",
    "create_daily_selection_publication",
    "load_daily_selection",
]
