"""NHL-28: append-only import of current-season rows from a verified team capture.

The imported rows remain descriptive. Their effective availability is the import
instant, never the historical game date or an asserted provider timestamp.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

from .contracts import require_utc
from .database import NHL_DATA_ROOT, PROJECT_ROOT
from .moneypuck_five_season import REFERENCE_SEASONS
from .moneypuck_team_capture import (
    DEFAULT_ROOT as CAPTURE_ROOT, VerifiedTeamCapture, verify_team_capture,
)
from .moneypuck_team_import import (
    NHLMoneyPuckImportError, TeamGameRow, import_team_games,
)
from .public_schedule_candidates import ScheduledGame
from .temporal_policy import build_information_cutoff

SCHEMA_VERSION = "nhl_current_season_import_v1"
POLICY_PATH = PROJECT_ROOT / "nhl_protocols/data/nhl28_current_season_import_v1.json"
DEFAULT_ROOT = NHL_DATA_ROOT / "current_season_imports"
_MAX_RECEIPT_BYTES = 64 * 1024
_MAX_ROWS_BYTES = 16 * 1024 * 1024


class NHLCurrentSeasonImportError(ValueError):
    """Current-season evidence is incomplete, altered or temporally unsafe."""


@dataclass(frozen=True, slots=True)
class CurrentSeasonImport:
    path: Path
    season: int
    source_capture_id: str
    source_response_sha256: str
    source_observed_at_utc: datetime
    imported_at_utc: datetime
    effective_available_at_utc: datetime
    regular_rows: tuple[TeamGameRow, ...]
    regular_game_count: int
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def _utc(value: datetime) -> datetime:
    try:
        return require_utc(value, field_name="current-season NHL")
    except ValueError as error:
        raise NHLCurrentSeasonImportError("Horodatage UTC requis.") from error


def _timestamp(value: datetime) -> str:
    return _utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise NHLCurrentSeasonImportError("Horodatage d'import invalide.")
    try:
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as error:
        raise NHLCurrentSeasonImportError("Horodatage d'import invalide.") from error


def _policy_hash() -> str:
    try:
        raw = POLICY_PATH.read_bytes()
        document = json.loads(raw)
    except (OSError, ValueError) as error:
        raise NHLCurrentSeasonImportError("Protocole saison courante illisible.") from error
    expected = {
        "schema_version": SCHEMA_VERSION,
        "source_capture_protocol": "nhl_moneypuck_team_capture_v1",
        "storage_mode": "append_only_local_files",
        "availability_rule": "max(source_capture_observed_at_utc,imported_at_utc)<=information_cutoff_utc",
        "historical_backtest_asof_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
        "mlb_database_mutation": False,
        "nhl_historical_database_mutation": False,
    }
    if not isinstance(document, dict) or any(document.get(k) != v for k, v in expected.items()):
        raise NHLCurrentSeasonImportError("Protocole saison courante incompatible.")
    return sha256(raw).hexdigest()


def _rows(capture: VerifiedTeamCapture, season: int) -> tuple[TeamGameRow, ...]:
    if type(season) is not int or season <= max(REFERENCE_SEASONS) or season > 2099:
        raise NHLCurrentSeasonImportError("Saison courante invalide.")
    for name in ("response.csv", "receipt.json", "COMPLETED"):
        if (capture.path / name).is_symlink():
            raise NHLCurrentSeasonImportError("Preuve source sous forme de lien symbolique.")
    try:
        snapshot = import_team_games(capture.path / "response.csv", season=season)
    except NHLMoneyPuckImportError as error:
        raise NHLCurrentSeasonImportError("Lignes de saison courante invalides.") from error
    if snapshot.file_sha256 != capture.response_sha256:
        raise NHLCurrentSeasonImportError("CSV source modifié pendant l'import.")
    regular = tuple(row for row in snapshot.rows if (row.game_id // 10_000) % 100 == 2)
    if not regular:
        raise NHLCurrentSeasonImportError("Aucun match régulier de saison courante.")
    groups: dict[tuple[int, str], list[TeamGameRow]] = defaultdict(list)
    situations_by_game: dict[int, set[str]] = defaultdict(set)
    for row in regular:
        if row.game_date >= capture.observed_at_utc.date():
            raise NHLCurrentSeasonImportError("Match source non antérieur à la capture.")
        groups[(row.game_id, row.situation)].append(row)
        situations_by_game[row.game_id].add(row.situation)
    for (game_id, _), pair in groups.items():
        if len(pair) != 2 or {r.home_or_away for r in pair} != {"HOME", "AWAY"}:
            raise NHLCurrentSeasonImportError(f"Paire d'équipes incomplète: {game_id}.")
        home = next(r for r in pair if r.home_or_away == "HOME")
        away = next(r for r in pair if r.home_or_away == "AWAY")
        if (home.team == away.team or home.team != away.opponent
                or away.team != home.opponent or home.game_date != away.game_date):
            raise NHLCurrentSeasonImportError(f"Identité des équipes incohérente: {game_id}.")
    if any(situations != {"all", "5on5"} for situations in situations_by_game.values()):
        raise NHLCurrentSeasonImportError("Situations NHL incomplètes.")
    for game_id in situations_by_game:
        identities = [
            {(r.team, r.opponent, r.home_or_away, r.game_date) for r in groups[(game_id, situation)]}
            for situation in ("all", "5on5")
        ]
        if identities[0] != identities[1]:
            raise NHLCurrentSeasonImportError(f"Situations contradictoires: {game_id}.")
    return tuple(sorted(regular, key=lambda r: (r.game_date, r.game_id, r.team, r.situation)))


def _row_record(row: TeamGameRow) -> dict[str, object]:
    return {
        "game_id": row.game_id, "season": row.season,
        "game_date": row.game_date.isoformat(), "team": row.team,
        "opponent": row.opponent, "home_or_away": row.home_or_away,
        "situation": row.situation,
        "x_goals_for": str(row.x_goals_for), "x_goals_against": str(row.x_goals_against),
        "goals_for": row.goals_for, "goals_against": row.goals_against,
        "shots_on_goal_for": str(row.shots_on_goal_for),
        "shots_on_goal_against": str(row.shots_on_goal_against),
    }


def _rows_bytes(rows: tuple[TeamGameRow, ...]) -> bytes:
    return (json.dumps([_row_record(row) for row in rows], sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _capture(slot: Path, capture_root: Path) -> VerifiedTeamCapture:
    try:
        return verify_team_capture(slot, allowed_root=capture_root)
    except (OSError, ValueError) as error:
        raise NHLCurrentSeasonImportError("Capture MoneyPuck source invalide.") from error


def import_current_season_capture(
    source_slot: Path, *, season: int = 2026, explicit_manual_run: bool = False,
    root: Path = DEFAULT_ROOT, capture_root: Path = CAPTURE_ROOT,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> CurrentSeasonImport:
    """Seal verified regular-season rows without touching either database."""
    if explicit_manual_run is not True:
        raise NHLCurrentSeasonImportError("Import manuel explicite requis.")
    policy_sha = _policy_hash()
    capture = _capture(Path(source_slot), Path(capture_root))
    rows = _rows(capture, season)
    imported = _utc(now())
    if imported < capture.observed_at_utc:
        raise NHLCurrentSeasonImportError("Import antérieur à la capture.")
    raw_rows = _rows_bytes(rows)
    rows_sha = sha256(raw_rows).hexdigest()
    relative = capture.path.relative_to(Path(capture_root).resolve()).as_posix()
    key = capture.path.name + "-" + rows_sha[:16]
    path = Path(root) / str(season) / key
    if path.exists():
        existing = verify_current_season_import(path, root=root, capture_root=capture_root)
        if existing.source_response_sha256 != capture.response_sha256:
            raise NHLCurrentSeasonImportError("Import existant divergent.")
        return existing
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "DESCRIPTIVE_IMPORT_ONLY",
        "season": season,
        "source_capture_relative": relative,
        "source_capture_id": capture.path.name,
        "source_response_sha256": capture.response_sha256,
        "source_observed_at_utc": _timestamp(capture.observed_at_utc),
        "imported_at_utc": _timestamp(imported),
        "effective_available_at_utc": _timestamp(imported),
        "regular_row_count": len(rows),
        "regular_game_count": len({row.game_id for row in rows}),
        "rows_sha256": rows_sha,
        "policy_sha256": policy_sha,
        "historical_backtest_asof_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }
    try:
        path.mkdir(parents=True, exist_ok=False)
        with (path / "rows.json").open("xb") as output:
            output.write(raw_rows)
        with (path / "receipt.json").open("xb") as output:
            output.write((json.dumps(receipt, sort_keys=True, separators=(",", ":"))
                          + "\n").encode("utf-8"))
        with (path / "COMPLETED").open("xb"):
            pass
    except OSError as error:
        raise NHLCurrentSeasonImportError("Import incomplet; créneau inutilisable.") from error
    return CurrentSeasonImport(path, season, capture.path.name, capture.response_sha256,
                               capture.observed_at_utc, imported, imported, rows,
                               receipt["regular_game_count"])


def verify_current_season_import(
    slot: Path, *, root: Path = DEFAULT_ROOT, capture_root: Path = CAPTURE_ROOT,
) -> CurrentSeasonImport:
    """Rebuild rows from the sealed source and reject altered import artefacts."""
    candidate = Path(slot)
    if candidate.is_symlink():
        raise NHLCurrentSeasonImportError("Créneau d'import sous forme de lien symbolique.")
    path = candidate.resolve()
    if (not path.is_relative_to(Path(root).resolve())
            or not (path / "COMPLETED").is_file()):
        raise NHLCurrentSeasonImportError("Créneau d'import hors périmètre ou incomplet.")
    try:
        for name in ("rows.json", "receipt.json", "COMPLETED"):
            if (path / name).is_symlink():
                raise NHLCurrentSeasonImportError("Preuve d'import sous forme de lien symbolique.")
        receipt_raw = (path / "receipt.json").read_bytes()
        rows_raw = (path / "rows.json").read_bytes()
        if len(receipt_raw) > _MAX_RECEIPT_BYTES or len(rows_raw) > _MAX_ROWS_BYTES:
            raise NHLCurrentSeasonImportError("Import trop volumineux.")
        receipt = json.loads(receipt_raw)
        season = receipt["season"]
        source = Path(capture_root) / receipt["source_capture_relative"]
        capture = _capture(source, Path(capture_root))
        if receipt["source_capture_relative"] != capture.path.relative_to(Path(capture_root).resolve()).as_posix():
            raise NHLCurrentSeasonImportError("Chemin source non canonique.")
        rows = _rows(capture, season)
        imported = _parse_timestamp(receipt["imported_at_utc"])
        observed = _parse_timestamp(receipt["source_observed_at_utc"])
        expected = {
            "schema_version": SCHEMA_VERSION,
            "status": "DESCRIPTIVE_IMPORT_ONLY",
            "source_capture_id": capture.path.name,
            "source_response_sha256": capture.response_sha256,
            "source_observed_at_utc": _timestamp(capture.observed_at_utc),
            "effective_available_at_utc": _timestamp(imported),
            "regular_row_count": len(rows),
            "regular_game_count": len({row.game_id for row in rows}),
            "rows_sha256": sha256(rows_raw).hexdigest(),
            "policy_sha256": _policy_hash(),
            "historical_backtest_asof_proven": False,
            "training_permitted": False,
            "prediction_publication_permitted": False,
        }
        if (not isinstance(receipt, dict) or type(season) is not int
                or observed != capture.observed_at_utc or imported < observed
                or rows_raw != _rows_bytes(rows)
                or receipt["rows_sha256"] != sha256(rows_raw).hexdigest()
                or any(receipt.get(k) != v for k, v in expected.items())
                or path.name != capture.path.name + "-" + receipt["rows_sha256"][:16]
                or path.parent.name != str(season)):
            raise NHLCurrentSeasonImportError("Reçu d'import incohérent.")
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError) as error:
        if isinstance(error, NHLCurrentSeasonImportError):
            raise
        raise NHLCurrentSeasonImportError("Import saison courante invalide.") from error
    return CurrentSeasonImport(path, season, capture.path.name, capture.response_sha256,
                               observed, imported, imported, rows,
                               len({row.game_id for row in rows}))


def assess_current_season_before_game(
    imported: CurrentSeasonImport, game: ScheduledGame, *, lead_minutes: int,
) -> dict[str, object]:
    """Count only prior regular games after the import passed the cutoff."""
    if not isinstance(imported, CurrentSeasonImport) or not isinstance(game, ScheduledGame):
        raise NHLCurrentSeasonImportError("Import ou match NHL invalide.")
    if game.season // 10_000 != imported.season:
        raise NHLCurrentSeasonImportError("Saison cible différente de l'import.")
    if (imported.training_permitted or imported.prediction_publication_permitted
            or imported.effective_available_at_utc < imported.source_observed_at_utc):
        raise NHLCurrentSeasonImportError("Import non admissible à l'audit descriptif.")
    cutoff = build_information_cutoff(game.start_utc, lead_minutes=lead_minutes)
    available = imported.effective_available_at_utc <= cutoff
    counts = {game.away_abbr: 0, game.home_abbr: 0}
    if available:
        for row in imported.regular_rows:
            if (row.situation == "all" and row.team in counts
                    and row.game_date < game.start_utc.date()
                    and row.game_id != game.game_id):
                counts[row.team] += 1
    return {
        "game_id": game.game_id,
        "current_season_import_id": imported.path.name,
        "current_season_import_effective_at_utc": _timestamp(imported.effective_available_at_utc),
        "current_season_import_before_cutoff": available,
        "prior_regular_games_by_team": counts,
        "training_permitted": False,
        "prediction_publication_permitted": False,
    }
