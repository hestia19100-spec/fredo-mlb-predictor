"""Offline NHL final-score evidence from The Odds API, without model activation.

Provider event IDs remain separate from official NHL game IDs. A completed
score is reconciled only with a matching provider event captured before play.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json

from src.nhl.odds_api_candidates import (
    MAX_RESPONSE_BYTES,
    NHLEventCandidate,
    NHLOddsCandidateError,
    parse_nhl_events,
)


class NHLScoreEvidenceError(ValueError):
    """The scores response cannot safely establish a result."""


@dataclass(frozen=True, slots=True)
class NHLProviderFinal:
    event: NHLEventCandidate
    away_score: int
    home_score: int
    last_update_utc: datetime

    @property
    def winner_name(self) -> str:
        return (self.event.home_team_name if self.home_score > self.away_score
                else self.event.away_team_name)


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise NHLScoreEvidenceError(f"{field} non UTC canonique.")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise NHLScoreEvidenceError(f"{field} invalide.") from error
    if result.utcoffset() != timedelta(0):
        raise NHLScoreEvidenceError(f"{field} non UTC.")
    return result.astimezone(timezone.utc)


def _score(value: object) -> int:
    if (not isinstance(value, str) or not value or len(value) > 2
            or any(char not in "0123456789" for char in value)
            or (len(value) > 1 and value.startswith("0"))):
        raise NHLScoreEvidenceError("Score NHL invalide.")
    return int(value)


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise NHLScoreEvidenceError("Clé JSON dupliquée dans le résultat.")
        result[key] = value
    return result


def parse_nhl_final_scores(raw: bytes, observed_at_utc: datetime) -> tuple[NHLProviderFinal, ...]:
    """Return completed, non-tied NHL scores; ignore live/upcoming rows."""
    if (not isinstance(observed_at_utc, datetime) or observed_at_utc.tzinfo is None
            or observed_at_utc.utcoffset() != timedelta(0)):
        raise NHLScoreEvidenceError("Horloge d'observation UTC requise.")
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_RESPONSE_BYTES:
        raise NHLScoreEvidenceError("Réponse de résultats absente ou trop volumineuse.")
    try:
        rows = json.loads(raw, object_pairs_hook=_unique_pairs)
        if not isinstance(rows, list):
            raise NHLScoreEvidenceError("Liste de résultats NHL attendue.")
        metadata = {event.provider_event_id: event for event in parse_nhl_events(raw)}
    except (ValueError, UnicodeDecodeError, NHLOddsCandidateError) as error:
        raise NHLScoreEvidenceError("Réponse de résultats NHL invalide.") from error
    finals: list[NHLProviderFinal] = []
    for row in rows:
        assert isinstance(row, dict)
        completed = row.get("completed")
        if type(completed) is not bool:
            raise NHLScoreEvidenceError("État completed absent ou invalide.")
        if not completed:
            continue
        event = metadata[row["id"]]
        if event.start_utc > observed_at_utc:
            raise NHLScoreEvidenceError("Match futur prétendument terminé.")
        scores = row.get("scores")
        if not isinstance(scores, list) or len(scores) != 2:
            raise NHLScoreEvidenceError("Score final NHL absent ou incomplet.")
        names: dict[str, int] = {}
        for item in scores:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise NHLScoreEvidenceError("Équipe du score invalide.")
            name = item["name"]
            if name in names:
                raise NHLScoreEvidenceError("Équipe du score dupliquée.")
            names[name] = _score(item.get("score"))
        if set(names) != {event.away_team_name, event.home_team_name}:
            raise NHLScoreEvidenceError("Équipes du score différentes du match.")
        updated = _utc(row.get("last_update"), "last_update")
        if updated < event.start_utc or updated > observed_at_utc:
            raise NHLScoreEvidenceError("Horodatage du score final incohérent.")
        away_score = names[event.away_team_name]
        home_score = names[event.home_team_name]
        if away_score == home_score:
            raise NHLScoreEvidenceError("Vainqueur NHL indéterminé.")
        finals.append(NHLProviderFinal(event, away_score, home_score, updated))
    return tuple(sorted(finals, key=lambda final: final.event.provider_event_id))


def reconcile_pregame_events(
    pregame_events: tuple[NHLEventCandidate, ...],
    captured_at_utc: datetime,
    finals: tuple[NHLProviderFinal, ...],
) -> tuple[NHLProviderFinal, ...]:
    """Match by provider ID plus identical teams/start, with one-hour cutoff.

    A postponed game whose start has changed is intentionally not reconciled
    here; it needs separate, evidenced rescheduling logic.
    """
    if (not isinstance(captured_at_utc, datetime) or captured_at_utc.tzinfo is None
            or captured_at_utc.utcoffset() != timedelta(0)):
        raise NHLScoreEvidenceError("Horloge de capture UTC requise.")
    by_id = {event.provider_event_id: event for event in pregame_events}
    if len(by_id) != len(pregame_events):
        raise NHLScoreEvidenceError("Événement pré-match dupliqué.")
    accepted: list[NHLProviderFinal] = []
    for final in finals:
        candidate = by_id.get(final.event.provider_event_id)
        if candidate is None:
            continue
        if candidate != final.event:
            raise NHLScoreEvidenceError("Match fournisseur divergent entre avant et après match.")
        if captured_at_utc > candidate.start_utc - timedelta(hours=1):
            raise NHLScoreEvidenceError("Capture après la limite d'une heure.")
        accepted.append(final)
    return tuple(accepted)
