"""Vue NHL au cutoff."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from src.nhl.contracts import ScheduledGameObservation, SourceEvidence, require_utc
from src.nhl.database import connect_nhl_database_read_only
from src.nhl.repository import audit_nhl_storage
from src.nhl.temporal_policy import validate_information_cutoff


class NHLAsOfError(RuntimeError):
    """La vue pré-match ne peut pas être établie sans ambiguïté."""


@dataclass(frozen=True, slots=True)
class AsOfObservation:
    observation_id: str
    kind: str
    entity_id: str
    source_game_id: int | None
    value_state: str
    value: Any
    effective_available_at_utc: datetime
    observation_sha256: str


@dataclass(frozen=True, slots=True)
class AsOfGameView:
    target_game_id: int
    information_cutoff_utc: datetime
    scheduled_start_utc: datetime
    away_team_id: int
    home_team_id: int
    schedule_observation_id: str
    observations: tuple[AsOfObservation, ...]
    source_final_observation_ids: tuple[str, ...]
    after_cutoff_observation_ids: tuple[str, ...]
    unfinalized_source_observation_ids: tuple[str, ...]
    ignored_target_state_observation_ids: tuple[str, ...]
    snapshot_sha256: str


def _utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise NHLAsOfError("Un horodatage du journal NHL n'est pas UTC.")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
        require_utc(result, field_name="horodatage NHL")
        return result
    except ValueError as error:
        raise NHLAsOfError("Un horodatage du journal NHL est invalide.") from error


def _value(row: sqlite3.Row) -> Any:
    if row["value_state"] == "UNKNOWN":
        return None
    try:
        return json.loads(row["canonical_value_json"])
    except (TypeError, json.JSONDecodeError) as error:
        raise NHLAsOfError("Une valeur NHL archivée est invalide.") from error


def _evidence(row: sqlite3.Row) -> SourceEvidence:
    updated = row["source_updated_at_utc"]
    return SourceEvidence(
        provider=row["provider"],
        observation_id=row["observation_id"],
        observed_at_utc=_utc(row["observed_at_utc"]),
        response_sha256=row["response_sha256"],
        code_commit=row["code_commit"],
        source_updated_at_utc=None if updated is None else _utc(updated),
    )


def _latest(rows: list[sqlite3.Row]) -> sqlite3.Row:
    """Retient la dernière preuve, refuse deux versions simultanées divergentes."""
    latest_time = max(_utc(row["effective_available_at_utc"]) for row in rows)
    peers = [row for row in rows if _utc(row["effective_available_at_utc"]) == latest_time]
    fingerprints = {(row["value_state"], row["canonical_value_json"]) for row in peers}
    if len(fingerprints) > 1:
        raise NHLAsOfError("Deux observations NHL simultanées divergent.")
    return min(peers, key=lambda row: row["observation_id"])


def _item(row: sqlite3.Row) -> AsOfObservation:
    return AsOfObservation(
        observation_id=row["observation_id"],
        kind=row["kind"],
        entity_id=row["entity_id"],
        source_game_id=row["source_game_id"],
        value_state=row["value_state"],
        value=_value(row),
        effective_available_at_utc=_utc(row["effective_available_at_utc"]),
        observation_sha256=row["observation_sha256"],
    )


def build_asof_game_view(
    database_path: Path,
    *,
    allowed_root: Path,
    target_game_id: int,
    information_cutoff_utc: datetime,
) -> AsOfGameView:
    """Reconstruit un match au cutoff sans modifier le journal NHL.

    Les statistiques d'un ancien match exigent une preuve de résultat final
    elle-même disponible avant le cutoff. Le résultat source et l'état du
    match cible ne sont jamais fournis comme variables prédictives.
    """
    if type(target_game_id) is not int or target_game_id <= 0:
        raise NHLAsOfError("target_game_id doit être positif.")
    require_utc(information_cutoff_utc, field_name="information_cutoff_utc")
    audit_nhl_storage(database_path, allowed_root=allowed_root)
    with closing(connect_nhl_database_read_only(database_path, allowed_root=allowed_root)) as connection:
        rows = connection.execute(
            "SELECT * FROM nhl_normalized_observations WHERE target_game_id = ? "
            "ORDER BY observation_id", (target_game_id,)
        ).fetchall()
    schedule_rows = [
        row for row in rows
        if row["kind"] == "SCHEDULED_GAME"
        and row["value_state"] == "KNOWN"
        and _utc(row["effective_available_at_utc"]) <= information_cutoff_utc
    ]
    if not schedule_rows:
        raise NHLAsOfError("Aucun calendrier NHL prouvé avant le cutoff.")
    schedule = _latest(schedule_rows)
    scheduled = _value(schedule)
    if not isinstance(scheduled, dict):
        raise NHLAsOfError("Le calendrier NHL est invalide.")
    try:
        game = ScheduledGameObservation(
            game_id=target_game_id,
            season_id=scheduled["season_id"],
            official_date=date.fromisoformat(scheduled["official_date"]),
            away_team_id=scheduled["away_team_id"],
            home_team_id=scheduled["home_team_id"],
            scheduled_start_utc=_utc(scheduled["scheduled_start_utc"]),
            evidence=_evidence(schedule),
        )
        validate_information_cutoff(game, information_cutoff_utc)
    except (KeyError, TypeError, ValueError) as error:
        raise NHLAsOfError("Le calendrier NHL est incomplet ou tardif.") from error

    other = [row for row in rows if row["kind"] != "SCHEDULED_GAME"]
    early = [row for row in other if _utc(row["effective_available_at_utc"]) <= information_cutoff_utc]
    late = [row["observation_id"] for row in other if _utc(row["effective_available_at_utc"]) > information_cutoff_utc]
    latest_by_key: dict[tuple[str, str, int | None], list[sqlite3.Row]] = {}
    for row in early:
        key = (row["kind"], row["entity_id"], row["source_game_id"])
        latest_by_key.setdefault(key, []).append(row)
    selected = [_latest(group) for group in latest_by_key.values()]

    finals: dict[int, str] = {}
    for row in selected:
        if row["kind"] != "FINAL_RESULT" or row["value_state"] != "KNOWN":
            continue
        value = _value(row)
        if not isinstance(value, dict) or value.get("game_state") != "FINAL":
            continue
        if _utc(value["final_observed_at_utc"]) <= information_cutoff_utc:
            finals[row["source_game_id"]] = row["observation_id"]

    accepted: list[AsOfObservation] = []
    no_final: list[str] = []
    target_state: list[str] = []
    for row in selected:
        kind = row["kind"]
        source_id = row["source_game_id"]
        if kind == "GAME_STATE":
            target_state.append(row["observation_id"])
        elif kind == "FINAL_RESULT":
            continue
        elif source_id is not None and source_id not in finals:
            no_final.append(row["observation_id"])
        else:
            accepted.append(_item(row))
    accepted.sort(key=lambda item: (item.kind, item.entity_id, item.source_game_id or 0))
    proof = {
        "target_game_id": target_game_id,
        "cutoff": information_cutoff_utc.isoformat().replace("+00:00", "Z"),
        "schedule": schedule["observation_sha256"],
        "observations": [item.observation_sha256 for item in accepted],
        "source_finals": sorted(finals.values()),
    }
    digest = hashlib.sha256(
        json.dumps(proof, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return AsOfGameView(
        target_game_id=target_game_id,
        information_cutoff_utc=information_cutoff_utc,
        scheduled_start_utc=game.scheduled_start_utc,
        away_team_id=game.away_team_id,
        home_team_id=game.home_team_id,
        schedule_observation_id=schedule["observation_id"],
        observations=tuple(accepted),
        source_final_observation_ids=tuple(sorted(finals.values())),
        after_cutoff_observation_ids=tuple(sorted(late)),
        unfinalized_source_observation_ids=tuple(sorted(no_final)),
        ignored_target_state_observation_ids=tuple(sorted(target_state)),
        snapshot_sha256=digest,
    )
