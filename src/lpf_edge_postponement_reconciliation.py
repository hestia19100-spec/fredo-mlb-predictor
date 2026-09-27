"""Complément d'audit des reports MLB, distinct du scoring Shadow V2."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import stat
from typing import Any, Mapping
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.lpf_edge_dashboard import (
    PROJECT_ROOT,
    SCORING_ROOT,
    CertifiedPredictionDay,
    DailyScoreSummary,
)


ROOT = Path("outcome_reconciliations/lpf_edge_mlb_postponements_v1")
SCHEMA = "lpf_edge_mlb_postponements_v1"
API_URL = "https://statsapi.mlb.com/api/v1/schedule"
LAST_CHECKPOINT = date(2026, 10, 12)
RAW_NAME = "schedule_response.json.gz"
RECEIPT_NAME = "reconciliation.json"
MARKER_NAME = "COMPLETED"


class PostponementReconciliationError(RuntimeError):
    """Une preuve ne permet pas de résoudre le report sans ambiguïté."""


@dataclass(frozen=True, slots=True)
class RescheduledGame:
    prediction_id: str
    game_id: int
    original_official_date: date
    final_official_date: date
    away_score: int
    home_score: int
    receipt_sha256: str
    observed_at_utc: datetime


@dataclass(frozen=True, slots=True)
class ReconciliationPublication:
    slot_path: Path
    paths: tuple[Path, ...]
    resolved: tuple[RescheduledGame, ...]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PostponementReconciliationError("La réponse MLB contient une clé répétée.")
        result[key] = value
    return result


def _read_regular(path: Path) -> bytes:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise PostponementReconciliationError("Une preuve est absente.") from error
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise PostponementReconciliationError("Une preuve n'est pas un fichier régulier.")
    return path.read_bytes()


def _safe_slot(project: Path, slot: Path) -> None:
    if not slot.is_relative_to(project):
        raise PostponementReconciliationError("Le complément sort du projet.")
    parent = slot.parent
    while parent != project:
        if parent.is_symlink():
            raise PostponementReconciliationError("Un dossier de preuve est un lien symbolique.")
        parent = parent.parent


def _parse_schedule(raw: bytes) -> dict[int, tuple[date, int, int, int, int]]:
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise PostponementReconciliationError("La réponse MLB est invalide.") from error
    if type(payload) is not dict or type(payload.get("dates")) is not list:
        raise PostponementReconciliationError("Le calendrier MLB est invalide.")
    finals: dict[int, tuple[date, int, int, int, int]] = {}
    for block in payload["dates"]:
        if type(block) is not dict or type(block.get("games")) is not list:
            raise PostponementReconciliationError("Un bloc du calendrier est invalide.")
        for game in block["games"]:
            if type(game) is not dict:
                raise PostponementReconciliationError("Une rencontre MLB est invalide.")
            game_id = game.get("gamePk")
            if type(game_id) is not int or game_id <= 0:
                raise PostponementReconciliationError("L'identifiant MLB est invalide.")
            status = game.get("status")
            if type(status) is not dict:
                raise PostponementReconciliationError("Le statut MLB est invalide.")
            code = status.get("statusCode")
            detail = status.get("detailedState")
            if type(code) is not str or type(detail) is not str:
                raise PostponementReconciliationError("Le statut MLB est incomplet.")
            normalized_code = code.strip().upper()
            normalized_detail = detail.strip().upper()
            code_final = normalized_code in {"F", "FG", "FO", "FR"}
            detail_final = normalized_detail in {"FINAL", "GAME OVER", "COMPLETED EARLY"}
            if ((code_final and normalized_detail in {"POSTPONED", "CANCELLED"})
                    or (detail_final and normalized_code in {"D", "DI", "DR", "C", "CI", "CR"})):
                raise PostponementReconciliationError("La famille du statut MLB est contradictoire.")
            final = code_final or detail_final
            if not final:
                continue
            try:
                official_date = date.fromisoformat(game["officialDate"])
                away = game["teams"]["away"]
                home = game["teams"]["home"]
                away_team = away["team"]["id"]
                home_team = home["team"]["id"]
                away_score = away["score"]
                home_score = home["score"]
            except (KeyError, TypeError, ValueError) as error:
                raise PostponementReconciliationError("Une finale MLB est incomplète.") from error
            if (game.get("gameType") != "R" or str(game.get("season")) != "2026"
                    or any(type(x) is not int or x < 0 for x in
                           (away_team, home_team, away_score, home_score))
                    or away_team == home_team or away_score == home_score):
                raise PostponementReconciliationError("Une finale MLB est incohérente.")
            item = (official_date, away_team, home_team, away_score, home_score)
            if game_id in finals and finals[game_id] != item:
                raise PostponementReconciliationError("Finales contradictoires pour le même match.")
            finals[game_id] = item
    return finals


def _pending_predictions(day: CertifiedPredictionDay, score: DailyScoreSummary):
    by_id = {p.game_id: p for p in day.predictions}
    if len(by_id) != len(day.predictions) or len(score.results) != len(by_id):
        raise PostponementReconciliationError("Les prédictions et résultats divergent.")
    pending = []
    for result in score.results:
        prediction = by_id.get(result.game_id)
        if prediction is None or result.prediction_id != prediction.prediction_id:
            raise PostponementReconciliationError("Le lien avec une prédiction est invalide.")
        if result.outcome_status == "PENDING_POSTPONED":
            pending.append(prediction)
    return pending


def _resolve(day: CertifiedPredictionDay, score: DailyScoreSummary, raw: bytes,
             observed_at: datetime, receipt_sha: str) -> tuple[RescheduledGame, ...]:
    finals = _parse_schedule(raw)
    resolved = []
    for prediction in _pending_predictions(day, score):
        item = finals.get(prediction.game_id)
        if item is None:
            continue
        final_date, away_id, home_id, away_score, home_score = item
        if (away_id != prediction.away_team_id or home_id != prediction.home_team_id):
            raise PostponementReconciliationError("Les équipes de la finale ne correspondent pas.")
        if final_date == prediction.official_date:
            raise PostponementReconciliationError("Finale à la date initiale : examen nécessaire.")
        if final_date < prediction.official_date or final_date > observed_at.date():
            raise PostponementReconciliationError("La date officielle finale est incohérente.")
        resolved.append(RescheduledGame(
            prediction.prediction_id, prediction.game_id,
            prediction.official_date, final_date, away_score, home_score,
            receipt_sha, observed_at,
        ))
    return tuple(sorted(resolved, key=lambda x: x.game_id))


def _slot(project: Path, target: date, checkpoint: date) -> Path:
    return project / ROOT / target.isoformat() / checkpoint.isoformat()


def _write_new(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def create_reconciliation(
    day: CertifiedPredictionDay, score: DailyScoreSummary,
    *, project_directory: Path = PROJECT_ROOT,
    checkpoint_date: date | None = None,
    response: object | None = None,
) -> ReconciliationPublication | None:
    """Une requête MLB au plus, puis un journal local immuable pour le jour UTC."""
    project = Path(project_directory).resolve(strict=True)
    checkpoint = checkpoint_date or datetime.now(timezone.utc).date()
    if (day.target_date < date(2026, 9, 10) or day.target_date > date(2026, 9, 27)
            or checkpoint <= day.target_date or checkpoint > LAST_CHECKPOINT):
        raise PostponementReconciliationError("Date hors de l'horizon enregistré.")
    pending = _pending_predictions(day, score)
    if not pending:
        return None
    slot = _slot(project, day.target_date, checkpoint)
    _safe_slot(project, slot)
    if slot.exists() or slot.is_symlink():
        raise PostponementReconciliationError("Le créneau complémentaire est déjà consommé.")
    report_path = (project / SCORING_ROOT / day.target_date.isoformat()
                   / "observations" / score.checkpoint_date.isoformat() / "daily_report.json")
    report_sha = _sha(_read_regular(report_path))
    requested = API_URL + "?" + urlencode({
        "sportId": 1, "startDate": day.target_date.isoformat(),
        "endDate": checkpoint.isoformat(), "gameTypes": "R",
    })
    # La réservation atomique précède le réseau : deux clics concurrents ne
    # peuvent jamais lancer deux requêtes pour le même jour UTC.
    try:
        slot.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise PostponementReconciliationError(
            "Le créneau complémentaire est déjà consommé."
        ) from error
    try:
        if response is None:
            request = Request(requested, headers={
                "User-Agent": "fredo-mlb-predictor-postponement-v1/1.0",
            })
            with urlopen(request, timeout=30) as stream:
                actual = SimpleNamespace(
                    status_code=stream.status, url=stream.geturl(),
                    headers=stream.headers, content=stream.read(),
                )
        else:
            actual = response
    except (HTTPError, URLError, OSError) as error:
        raise PostponementReconciliationError("La collecte MLB a échoué.") from error
    if actual.status_code != 200 or actual.url != requested:
        raise PostponementReconciliationError("La réponse MLB ne correspond pas à la requête.")
    date_header = actual.headers.get("Date")
    try:
        observed_at = parsedate_to_datetime(date_header).astimezone(timezone.utc)
    except (TypeError, ValueError, AttributeError) as error:
        raise PostponementReconciliationError("La date HTTP MLB est invalide.") from error
    if observed_at.date() != checkpoint:
        raise PostponementReconciliationError("La réponse MLB est datée d'un autre jour UTC.")
    raw = actual.content
    if type(raw) is not bytes or not raw:
        raise PostponementReconciliationError("La réponse MLB est vide.")
    # Une réponse invalide laisse un créneau réservé, mais aucune preuve publiée.
    preliminary = _resolve(day, score, raw, observed_at, "0" * 64)
    evidence = {
        "schema": SCHEMA, "target_official_date": day.target_date.isoformat(),
        "checkpoint_utc_date": checkpoint.isoformat(),
        "observed_at_utc": observed_at.isoformat().replace("+00:00", "Z"),
        "requested_url": requested, "effective_url": actual.url,
        "http_status": actual.status_code, "http_date": date_header,
        "response_sha256": _sha(raw),
        "score_report_sha256": report_sha,
        "source_score_checkpoint": score.checkpoint_date.isoformat(),
        "predictions_sha256": day.predictions_sha256,
        "batch_id": day.batch_id,
        "resolved_game_ids": [x.game_id for x in preliminary],
    }
    receipt_raw = _canonical(evidence)
    receipt_sha = _sha(receipt_raw)
    resolved = _resolve(day, score, raw, observed_at, receipt_sha)
    raw_gzip = gzip.compress(raw, mtime=0)
    raw_path, receipt_path, marker_path = (
        slot / RAW_NAME, slot / RECEIPT_NAME, slot / MARKER_NAME,
    )
    _write_new(raw_path, raw_gzip)
    _write_new(receipt_path, receipt_raw)
    _write_new(marker_path, _canonical({
        "schema": SCHEMA + "_completed", "receipt_sha256": receipt_sha,
        "raw_gzip_sha256": _sha(raw_gzip),
    }))
    return ReconciliationPublication(slot, (raw_path, receipt_path, marker_path), resolved)


def load_resolutions(
    day: CertifiedPredictionDay, score: DailyScoreSummary | None,
    *, project_directory: Path = PROJECT_ROOT,
) -> Mapping[int, RescheduledGame]:
    """Relit et vérifie les compléments sans appel réseau ni modification du score."""
    if score is None:
        return {}
    if not any(x.outcome_status == "PENDING_POSTPONED" for x in score.results):
        return {}
    project = Path(project_directory).resolve(strict=True)
    directory = project / ROOT / day.target_date.isoformat()
    _safe_slot(project, directory)
    if not directory.exists():
        return {}
    if directory.is_symlink() or not directory.is_dir():
        raise PostponementReconciliationError("Le dossier des compléments est invalide.")
    selected: dict[int, RescheduledGame] = {}
    for slot in sorted(directory.iterdir()):
        if slot.is_symlink() or not slot.is_dir():
            raise PostponementReconciliationError("Un créneau complémentaire est invalide.")
        try:
            checkpoint = date.fromisoformat(slot.name)
        except ValueError as error:
            raise PostponementReconciliationError("Date de créneau invalide.") from error
        if checkpoint > LAST_CHECKPOINT:
            raise PostponementReconciliationError("Créneau hors horizon.")
        _safe_slot(project, slot)
        if not (slot / MARKER_NAME).is_file():
            # Une tentative interrompue reste visible mais non consommable.
            continue
        if {x.name for x in slot.iterdir()} != {RAW_NAME, RECEIPT_NAME, MARKER_NAME}:
            raise PostponementReconciliationError("Le créneau contient des fichiers inattendus.")
        marker = json.loads(_read_regular(slot / MARKER_NAME))
        receipt_raw = _read_regular(slot / RECEIPT_NAME)
        receipt = json.loads(receipt_raw)
        gz = _read_regular(slot / RAW_NAME)
        if (type(marker) is not dict or type(receipt) is not dict
                or _canonical(marker) != _read_regular(slot / MARKER_NAME)
                or _canonical(receipt) != receipt_raw
                or marker.get("schema") != SCHEMA + "_completed"
                or marker.get("receipt_sha256") != _sha(receipt_raw)
                or marker.get("raw_gzip_sha256") != _sha(gz)
                or receipt.get("schema") != SCHEMA
                or receipt.get("target_official_date") != day.target_date.isoformat()
                or receipt.get("checkpoint_utc_date") != checkpoint.isoformat()
                or receipt.get("predictions_sha256") != day.predictions_sha256
                or receipt.get("batch_id") != day.batch_id):
            raise PostponementReconciliationError("Les empreintes du complément divergent.")
        raw = gzip.decompress(gz)
        if receipt.get("response_sha256") != _sha(raw):
            raise PostponementReconciliationError("La réponse MLB du complément a changé.")
        try:
            observed = datetime.fromisoformat(receipt["observed_at_utc"].replace("Z", "+00:00"))
        except (KeyError, ValueError, TypeError) as error:
            raise PostponementReconciliationError("L'instant du complément est invalide.") from error
        if observed.tzinfo is None:
            raise PostponementReconciliationError("L'instant du complément manque son fuseau.")
        if observed.astimezone(timezone.utc).date() != checkpoint:
            raise PostponementReconciliationError("L'instant du complément diverge du créneau.")
        try:
            source_checkpoint = date.fromisoformat(receipt["source_score_checkpoint"])
        except (KeyError, TypeError, ValueError) as error:
            raise PostponementReconciliationError("Le checkpoint source est invalide.") from error
        source_slot = (project / SCORING_ROOT / day.target_date.isoformat()
                       / "observations" / source_checkpoint.isoformat())
        report_path = source_slot / "daily_report.json"
        if receipt.get("score_report_sha256") != _sha(_read_regular(report_path)):
            raise PostponementReconciliationError("Le rapport référencé a changé.")
        adjudications = csv.DictReader(io.StringIO(
            _read_regular(source_slot / "adjudications.csv").decode("utf-8")
        ))
        source_pending = {
            int(row["game_id"]): row["prediction_id"]
            for row in adjudications
            if row["outcome_status"] == "PENDING_POSTPONED"
        }
        if not source_pending:
            raise PostponementReconciliationError("La preuve source ne comporte aucun report.")
        expected_url = API_URL + "?" + urlencode({
            "sportId": 1, "startDate": day.target_date.isoformat(),
            "endDate": checkpoint.isoformat(), "gameTypes": "R",
        })
        if (receipt.get("requested_url") != expected_url
                or receipt.get("effective_url") != expected_url
                or receipt.get("http_status") != 200):
            raise PostponementReconciliationError("La requête archivée est invalide.")
        predictions = {p.game_id: p for p in day.predictions}
        finals = _parse_schedule(raw)
        slot_resolved: list[RescheduledGame] = []
        expected_ids = sorted(
            game_id for game_id in source_pending
            if game_id in finals
        )
        if receipt.get("resolved_game_ids") != expected_ids:
            raise PostponementReconciliationError("La liste des finales est incomplète.")
        for game_id in expected_ids:
            prediction = predictions.get(game_id)
            item = finals.get(game_id)
            if (prediction is None or item is None
                    or source_pending.get(game_id) != prediction.prediction_id):
                raise PostponementReconciliationError("Le report résolu n'est pas prouvé.")
            final_date, away_id, home_id, away_score, home_score = item
            if (away_id != prediction.away_team_id or home_id != prediction.home_team_id
                    or final_date <= day.target_date or final_date > observed.date()):
                raise PostponementReconciliationError("La finale complémentaire diverge.")
            slot_resolved.append(RescheduledGame(
                prediction.prediction_id, game_id, day.target_date,
                final_date, away_score, home_score, _sha(receipt_raw), observed,
            ))
        if len(set(receipt.get("resolved_game_ids", []))) != len(slot_resolved):
            raise PostponementReconciliationError("Un match résolu est répété.")
        if receipt.get("resolved_game_ids") != [x.game_id for x in slot_resolved]:
            raise PostponementReconciliationError("Le reçu omet un match résolu.")
        for result in slot_resolved:
            previous = selected.get(result.game_id)
            if previous is not None and (
                previous.final_official_date != result.final_official_date
                or previous.away_score != result.away_score
                or previous.home_score != result.home_score
                or previous.prediction_id != result.prediction_id
            ):
                raise PostponementReconciliationError("Deux compléments divergent pour le même match.")
            selected.setdefault(result.game_id, result)
    # Un résultat terminal officiel a priorité : le complément reste historique.
    pending_ids = {x.game_id for x in score.results if x.outcome_status == "PENDING_POSTPONED"}
    return {game_id: value for game_id, value in selected.items() if game_id in pending_ids}
