"""NHL-36: manually capture independent historical final scores, without fitting."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Sequence

from .database import NHL_DATA_ROOT, PROJECT_ROOT
from .moneypuck_five_season import REFERENCE_SEASONS
from .moneypuck_team_capture import DEFAULT_ROOT as TEAM_ROOT, VerifiedTeamCapture, verify_team_capture
from .official_final_capture import BASE_URL, MAX_BYTES
from .real_archive_audit import audit_reference_history

SCHEMA = "nhl36_historical_official_final_v1"
POLICY = PROJECT_ROOT / "nhl_protocols/data/nhl36_historical_official_final_v1.json"
ROOT = NHL_DATA_ROOT / "historical_official_finals"


class HistoricalOfficialFinalError(ValueError):
    """The official response and the captured statistical game cannot be reconciled."""


@dataclass(frozen=True, slots=True)
class HistoricalFinalReceipt:
    path: Path
    game_id: int
    observed_at_utc: datetime
    response_sha256: str
    winner_abbr: str
    home_score: int
    away_score: int
    final_type: str
    source_was_tied: bool


@dataclass(frozen=True, slots=True)
class HistoricalFinalReconciliation:
    source_capture_sha256: str
    verified_final_count: int
    resolved_tied_game_ids: tuple[int, ...]
    unresolved_tied_game_ids: tuple[int, ...]
    matched_decisive_game_ids: tuple[int, ...]
    reconciliation_sha256: str
    final_capture_coverage_complete: bool = False
    origin_independently_verified: bool = False
    historical_pregame_availability_proven: bool = False
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise HistoricalOfficialFinalError("Champ JSON duplique.")
        result[key] = value
    return result


def _utc(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise HistoricalOfficialFinalError("Horodatage UTC invalide.")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise HistoricalOfficialFinalError("Horodatage UTC invalide.") from error


def _stamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset().total_seconds() != 0:
        raise HistoricalOfficialFinalError("Horloge UTC requise.")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _policy_hash() -> str:
    try:
        raw = POLICY.read_bytes()
        policy = json.loads(raw, object_pairs_hook=_unique)
    except (OSError, ValueError) as error:
        raise HistoricalOfficialFinalError("Protocole NHL-36 illisible.") from error
    required = {
        "schema_version": SCHEMA, "manual_single_game_only": True,
        "endpoint_template": BASE_URL + "{game_id}/landing",
        "regular_season_only": True, "final_state_required": "OFF",
        "same_game_identity_required": True, "source_tie_requires_ot_or_so": True,
        "historical_pregame_availability_proven": False,
        "origin_independently_verified": False,
        "training_permitted": False, "prediction_publication_permitted": False,
        "database_mutation_allowed": False,
    }
    if not isinstance(policy, dict) or any(policy.get(k) != v for k, v in required.items()):
        raise HistoricalOfficialFinalError("Protocole NHL-36 incompatible.")
    return sha256(raw).hexdigest()


def _source_game(source: VerifiedTeamCapture, game_id: int):
    if type(game_id) is not int or game_id // 1_000_000 not in REFERENCE_SEASONS or (game_id // 10_000) % 100 != 2:
        raise HistoricalOfficialFinalError("Identifiant de saison reguliere invalide.")
    rows = [row for row in source.history.regular_rows
            if row.game_id == game_id and row.situation == "all"]
    by_side = {row.home_or_away: row for row in rows}
    if len(rows) != 2 or set(by_side) != {"HOME", "AWAY"}:
        raise HistoricalOfficialFinalError("Match historique absent ou ambigu.")
    home, away = by_side["HOME"], by_side["AWAY"]
    if (home.season != away.season or home.game_date != away.game_date
            or home.team != away.opponent or away.team != home.opponent
            or home.goals_for != away.goals_against
            or away.goals_for != home.goals_against):
        raise HistoricalOfficialFinalError("Identite ou score source incoherent.")
    return home, away


def _final(raw: bytes, source: VerifiedTeamCapture, game_id: int,
           observed: datetime) -> dict[str, object]:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BYTES:
        raise HistoricalOfficialFinalError("Taille du resultat invalide.")
    home, away = _source_game(source, game_id)
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise HistoricalOfficialFinalError("Resultat NHL illisible.") from error
    if not isinstance(document, dict):
        raise HistoricalOfficialFinalError("Objet NHL invalide.")
    if (document.get("id") != game_id
            or document.get("season") != home.season * 10_000 + home.season + 1
            or document.get("gameType") != 2
            or document.get("gameState") != "OFF"
            or document.get("gameScheduleState") != "OK"
            or document.get("gameDate") != home.game_date.isoformat()):
        raise HistoricalOfficialFinalError("Resultat non final ou identite incompatible.")
    if observed <= _utc(document.get("startTimeUTC")):
        raise HistoricalOfficialFinalError("Resultat observe avant le match.")
    away_team, home_team = document.get("awayTeam"), document.get("homeTeam")
    period = document.get("periodDescriptor")
    if (not isinstance(away_team, dict) or not isinstance(home_team, dict)
            or away_team.get("abbrev") != away.team
            or home_team.get("abbrev") != home.team
            or not isinstance(period, dict)
            or period.get("periodType") not in {"REG", "OT", "SO"}):
        raise HistoricalOfficialFinalError("Equipes ou periode incompatibles.")
    away_score, home_score = away_team.get("score"), home_team.get("score")
    if (type(away_score) is not int or type(home_score) is not int
            or min(away_score, home_score) < 0 or away_score == home_score):
        raise HistoricalOfficialFinalError("Vainqueur final absent.")
    tied = home.goals_for == away.goals_for
    if tied:
        source_score = home.goals_for
        if (period["periodType"] not in {"OT", "SO"}
                or min(home_score, away_score) != source_score
                or max(home_score, away_score) != source_score + 1):
            raise HistoricalOfficialFinalError("Score final non raccordable au score nul source.")
    elif (home_score, away_score) != (home.goals_for, away.goals_for):
        raise HistoricalOfficialFinalError("Score officiel et score source divergents.")
    return {
        "game_date": home.game_date.isoformat(), "home_abbr": home.team,
        "away_abbr": away.team, "source_home_score": home.goals_for,
        "source_away_score": away.goals_for, "home_score": home_score,
        "away_score": away_score, "final_type": period["periodType"],
        "winner_abbr": home.team if home_score > away_score else away.team,
        "source_was_tied": tied,
    }


def _receipt(source: VerifiedTeamCapture, game_id: int, observed: datetime,
             raw: bytes, final: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA, "status": "OFFICIAL_HISTORICAL_FINAL_OBSERVED",
        "source_url": BASE_URL + str(game_id) + "/landing",
        "game_id": game_id, "observed_at_utc": _stamp(observed),
        "source_capture_sha256": source.response_sha256,
        "source_observed_at_utc": _stamp(source.observed_at_utc),
        "response_sha256": sha256(raw).hexdigest(), "policy_sha256": _policy_hash(),
        **final, "historical_pregame_availability_proven": False,
        "origin_independently_verified": False,
        "training_permitted": False, "prediction_publication_permitted": False,
    }


def _as_result(path: Path, receipt: dict[str, object]) -> HistoricalFinalReceipt:
    return HistoricalFinalReceipt(
        path, receipt["game_id"], _utc(receipt["observed_at_utc"]),
        receipt["response_sha256"], receipt["winner_abbr"],
        receipt["home_score"], receipt["away_score"],
        receipt["final_type"], receipt["source_was_tied"],
    )


def capture_historical_final(
    team_slot: Path, game_id: int, *, explicit_manual_run: bool = False,
    transport: Callable | None = None, root: Path = ROOT,
    team_root: Path = TEAM_ROOT,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> HistoricalFinalReceipt:
    """One explicit HTTPS GET and append-only receipt; no model or database access."""
    if explicit_manual_run is not True or transport is None:
        raise HistoricalOfficialFinalError("Capture manuelle explicite requise.")
    source = verify_team_capture(team_slot, allowed_root=team_root)
    _source_game(source, game_id)
    _policy_hash()
    started = now()
    _stamp(started)
    url = BASE_URL + str(game_id) + "/landing"
    try:
        with transport(url, timeout=15) as response:
            if (response.geturl() != url or response.status != 200
                    or response.headers.get("Content-Type", "").split(";", 1)[0].lower() != "application/json"):
                raise HistoricalOfficialFinalError("Reponse NHL redirigee ou invalide.")
            raw = response.read(MAX_BYTES + 1)
    except HistoricalOfficialFinalError:
        raise
    except (OSError, ValueError) as error:
        raise HistoricalOfficialFinalError("Collecte officielle echouee.") from error
    observed = now()
    if observed < started:
        raise HistoricalOfficialFinalError("Horloge de collecte incoherente.")
    _stamp(observed)
    final = _final(raw, source, game_id, observed)
    receipt = _receipt(source, game_id, observed, raw, final)
    key = observed.strftime("%Y%m%dT%H%M%S%fZ") + "-" + receipt["response_sha256"][:16]
    slot = Path(root) / str(game_id // 1_000_000) / str(game_id) / key
    try:
        slot.mkdir(parents=True, exist_ok=False)
        with (slot / "response.json").open("xb") as output:
            output.write(raw)
        with (slot / "receipt.json").open("xb") as output:
            output.write((json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
        with (slot / "COMPLETED").open("xb"):
            pass
    except OSError as error:
        raise HistoricalOfficialFinalError("Archivage officiel incomplet.") from error
    return _as_result(slot, receipt)


def _verify(slot: Path, source: VerifiedTeamCapture, root: Path) -> HistoricalFinalReceipt:
    path = Path(slot).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not (path / "COMPLETED").is_file():
        raise HistoricalOfficialFinalError("Capture historique hors perimetre ou incomplete.")
    try:
        raw = (path / "response.json").read_bytes()
        receipt_raw = (path / "receipt.json").read_bytes()
        if len(receipt_raw) > 8192:
            raise HistoricalOfficialFinalError("Recu trop volumineux.")
        receipt = json.loads(receipt_raw, object_pairs_hook=_unique)
        if not isinstance(receipt, dict) or type(receipt.get("game_id")) is not int:
            raise HistoricalOfficialFinalError("Recu historique invalide.")
        observed = _utc(receipt.get("observed_at_utc"))
        game_id = receipt["game_id"]
        final = _final(raw, source, game_id, observed)
        expected = _receipt(source, game_id, observed, raw, final)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise HistoricalOfficialFinalError("Capture historique illisible ou invalide.") from error
    expected_key = observed.strftime("%Y%m%dT%H%M%S%fZ") + "-" + expected["response_sha256"][:16]
    if (receipt != expected or path.name != expected_key or path.parent.name != str(game_id)
            or path.parent.parent.name != str(game_id // 1_000_000)):
        raise HistoricalOfficialFinalError("Recu et resultat historique divergents.")
    return _as_result(path, expected)


def verify_historical_final(slot: Path, team_slot: Path, *, root: Path = ROOT,
                            team_root: Path = TEAM_ROOT) -> HistoricalFinalReceipt:
    source = verify_team_capture(team_slot, allowed_root=team_root)
    return _verify(slot, source, root)


def reconcile_historical_finals(team_slot: Path, final_slots: Sequence[Path], *,
                                root: Path = ROOT, team_root: Path = TEAM_ROOT
                                ) -> HistoricalFinalReconciliation:
    """Count only verified independent finals, never inferred source winners."""
    source = verify_team_capture(team_slot, allowed_root=team_root)
    audit = audit_reference_history(source.history)
    receipts = tuple(_verify(slot, source, root) for slot in final_slots)
    ids = [item.game_id for item in receipts]
    if len(ids) != len(set(ids)):
        raise HistoricalOfficialFinalError("Plusieurs captures pour un meme match.")
    tied = set(audit.tied_score_game_ids)
    resolved = tuple(sorted(item.game_id for item in receipts if item.game_id in tied))
    decisive = tuple(sorted(item.game_id for item in receipts if item.game_id not in tied))
    unresolved = tuple(sorted(tied - set(resolved)))
    proof = {
        "schema_version": SCHEMA, "source_capture_sha256": source.response_sha256,
        "verified_finals": sorted((item.game_id, item.response_sha256, item.observed_at_utc.isoformat()) for item in receipts),
        "resolved_tied_game_ids": resolved, "unresolved_tied_game_ids": unresolved,
        "matched_decisive_game_ids": decisive,
    }
    digest = sha256(json.dumps(proof, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return HistoricalFinalReconciliation(
        source.response_sha256, len(receipts), resolved, unresolved, decisive,
        digest, final_capture_coverage_complete=len(receipts) == audit.regular_games,
    )
