"""NHL-16: compare user-supplied final evidence with the historical audit.

This module never calls a provider. A URL written in an offline file is a
claim of origin, not proof of authenticity or permission to train a model.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
import re

from .contracts import require_utc
from .moneypuck_five_season import FiveSeasonHistory, REFERENCE_SEASONS
from .real_archive_audit import NHLRealArchiveAudit, audit_reference_history

MAX_EVIDENCE_BYTES = 8 * 1024 * 1024
SCHEMA = "nhl_offline_final_evidence_v1"
MARKET = "MATCH_WINNER_INCLUDING_OT_AND_SHOOTOUT"
TEAM_CODE = re.compile(r"^(?:[A-Z]{3}|[A-Z]\.[A-Z])$")
FINAL_TYPES = frozenset({"REGULATION", "OVERTIME", "SHOOTOUT"})


class NHLOfflineFinalError(ValueError):
    """The offline evidence cannot be matched without ambiguity."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise NHLOfflineFinalError("Duplicate JSON field")
        result[key] = value
    return result


def _utc(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise NHLOfflineFinalError("observed_at_utc must end in Z")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
        require_utc(result, field_name="observed_at_utc")
    except ValueError as error:
        raise NHLOfflineFinalError("Invalid observed_at_utc") from error
    return result


@dataclass(frozen=True, slots=True)
class OfflineFinalGame:
    game_id: int
    game_date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    final_type: str
    source_url: str

    def __post_init__(self) -> None:
        if (type(self.game_id) is not int
                or self.game_id // 1_000_000 not in REFERENCE_SEASONS
                or (self.game_id // 10_000) % 100 != 2):
            raise NHLOfflineFinalError("Game ID is not a reference regular game")
        if type(self.game_date) is not date:
            raise NHLOfflineFinalError("game_date must be a date")
        if not isinstance(self.home_team, str) or not isinstance(self.away_team, str):
            raise NHLOfflineFinalError("Team codes must be strings")
        if (not TEAM_CODE.fullmatch(self.home_team)
                or not TEAM_CODE.fullmatch(self.away_team)
                or self.home_team == self.away_team):
            raise NHLOfflineFinalError("Invalid home/away team codes")
        if (type(self.home_score) is not int or type(self.away_score) is not int
                or min(self.home_score, self.away_score) < 0
                or self.home_score == self.away_score):
            raise NHLOfflineFinalError("A final winner needs unequal scores")
        if self.final_type not in FINAL_TYPES:
            raise NHLOfflineFinalError("Unknown final type")
        if (not isinstance(self.source_url, str)
                or not self.source_url.startswith("https://www.nhl.com/gamecenter/")
                or str(self.game_id) not in self.source_url):
            raise NHLOfflineFinalError("Gamecenter URL does not identify this game")

    @property
    def winner(self) -> str:
        return self.home_team if self.home_score > self.away_score else self.away_team


@dataclass(frozen=True, slots=True)
class OfflineFinalArchive:
    observed_at_utc: datetime
    file_sha256: str
    games: tuple[OfflineFinalGame, ...]
    origin_verified: bool = False

    def __post_init__(self) -> None:
        require_utc(self.observed_at_utc, field_name="observed_at_utc")
        if (not isinstance(self.file_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", self.file_sha256)):
            raise NHLOfflineFinalError("Invalid evidence file SHA-256")
        if (not isinstance(self.games, tuple)
                or any(not isinstance(game, OfflineFinalGame) for game in self.games)):
            raise NHLOfflineFinalError("Final games must be an immutable tuple")
        ids = [game.game_id for game in self.games]
        if len(ids) != len(set(ids)):
            raise NHLOfflineFinalError("Duplicate final game ID")
        if any(self.observed_at_utc.date() < game.game_date for game in self.games):
            raise NHLOfflineFinalError("Evidence observed before a game date")
        if self.origin_verified is not False:
            raise NHLOfflineFinalError("An offline file cannot certify its own origin")


def load_offline_final_archive(path: Path) -> OfflineFinalArchive:
    """Read a bounded local JSON file; no network or database side effects."""
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_EVIDENCE_BYTES:
            raise NHLOfflineFinalError("Evidence file size is not allowed")
        raw = path.read_bytes()
        if len(raw) != size or path.stat().st_size != size:
            raise NHLOfflineFinalError("Evidence file changed during read")
        document = json.loads(raw.decode("utf-8-sig"),
                              object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NHLOfflineFinalError("Unreadable offline final file") from error
    if (not isinstance(document, dict)
            or set(document) != {"schema", "market", "observed_at_utc", "games"}
            or document["schema"] != SCHEMA or document["market"] != MARKET
            or not isinstance(document["games"], list)):
        raise NHLOfflineFinalError("Wrong final evidence schema or market")
    required = {"game_id", "game_date", "home_team", "away_team",
                "home_score", "away_score", "final_type", "source_url"}
    games = []
    for item in document["games"]:
        if not isinstance(item, dict) or set(item) != required:
            raise NHLOfflineFinalError("Invalid final game fields")
        try:
            if not isinstance(item["game_date"], str):
                raise ValueError("Invalid date")
            game_date = date.fromisoformat(item["game_date"])
        except ValueError as error:
            raise NHLOfflineFinalError("Invalid final game date") from error
        games.append(OfflineFinalGame(
            game_id=item["game_id"], game_date=game_date,
            home_team=item["home_team"], away_team=item["away_team"],
            home_score=item["home_score"], away_score=item["away_score"],
            final_type=item["final_type"], source_url=item["source_url"],
        ))
    return OfflineFinalArchive(
        observed_at_utc=_utc(document["observed_at_utc"]),
        file_sha256=sha256(raw).hexdigest(), games=tuple(games),
    )


@dataclass(frozen=True, slots=True)
class OfflineFinalReconciliation:
    source_audit_sha256: str
    evidence_file_sha256: str
    matched_final_games: int
    matched_decisive_games: int
    resolved_tied_game_ids: tuple[int, ...]
    unresolved_tied_game_ids: tuple[int, ...]
    reconciliation_sha256: str
    independent_final_origin_verified: bool = False
    historical_pregame_availability_proven: bool = False
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def reconcile_offline_finals(
    history: FiveSeasonHistory, audit: NHLRealArchiveAudit,
    evidence: OfflineFinalArchive,
) -> OfflineFinalReconciliation:
    """Match ID, teams, date and scores; reject any conflict as a whole."""
    if (audit.source_file_sha256 != history.file_sha256
            or audit.audit_sha256 != audit_reference_history(
                history, lookback_games=audit.lookback_games).audit_sha256):
        raise NHLOfflineFinalError("Audit and historical archive differ")
    by_game: dict[int, dict[str, object]] = defaultdict(dict)
    for row in history.regular_rows:
        if row.situation == "all":
            if row.home_or_away in by_game[row.game_id]:
                raise NHLOfflineFinalError("Duplicate source side")
            by_game[row.game_id][row.home_or_away] = row
    tied = set(audit.tied_score_game_ids)
    resolved: list[int] = []
    decisive = 0
    for game in evidence.games:
        sides = by_game.get(game.game_id)
        if sides is None or set(sides) != {"HOME", "AWAY"}:
            raise NHLOfflineFinalError("Foreign or incomplete game")
        home, away = sides["HOME"], sides["AWAY"]
        if (game.game_date != home.game_date or game.game_date != away.game_date
                or game.home_team != home.team or game.away_team != away.team):
            raise NHLOfflineFinalError("Final game identity differs from archive")
        if game.game_id in tied:
            score = home.goals_for
            if (game.final_type == "REGULATION"
                    or away.goals_for != score
                    or min(game.home_score, game.away_score) != score
                    or max(game.home_score, game.away_score) != score + 1):
                raise NHLOfflineFinalError("Tied source score not reconciled")
            resolved.append(game.game_id)
        elif (game.home_score, game.away_score) != (
                home.goals_for, away.goals_for):
            raise NHLOfflineFinalError("Decisive source and final scores disagree")
        else:
            decisive += 1
    unresolved = tuple(sorted(tied - set(resolved)))
    resolved_ids = tuple(sorted(resolved))
    proof = {
        "schema": "nhl_offline_final_reconciliation_v1",
        "source_audit_sha256": audit.audit_sha256,
        "evidence_file_sha256": evidence.file_sha256,
        "matched_final_games": len(evidence.games),
        "matched_decisive_games": decisive,
        "resolved_tied_game_ids": resolved_ids,
        "unresolved_tied_game_ids": unresolved,
    }
    digest = sha256(json.dumps(
        proof, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return OfflineFinalReconciliation(
        source_audit_sha256=audit.audit_sha256,
        evidence_file_sha256=evidence.file_sha256,
        matched_final_games=len(evidence.games),
        matched_decisive_games=decisive,
        resolved_tied_game_ids=resolved_ids,
        unresolved_tied_game_ids=unresolved,
        reconciliation_sha256=digest,
    )
