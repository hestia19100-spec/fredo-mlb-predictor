"""Journal NHL dédié aux captures MoneyPuck réelles, sans activation du modèle.

Les données ne sont éligibles à une lecture pré-match qu'après leur capture ET
leur import effectifs. La date historique d'un match source n'est pas une preuve
qu'une statistique était disponible à cette date.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from .contracts import require_utc
from .database import NHL_DATA_ROOT, PROJECT_ROOT, validate_nhl_database_path
from .moneypuck_team_capture import DEFAULT_ROOT, VerifiedTeamCapture, verify_team_capture
from .moneypuck_team_import import TeamGameRow
from .public_schedule_candidates import ScheduledGame
from .temporal_policy import build_information_cutoff

DATABASE_PATH = NHL_DATA_ROOT / "team_history.db"
SCHEMA_VERSION = 1
POLICY_PATH = PROJECT_ROOT / "nhl_protocols/data/nhl24_team_history_store_v1.json"


class NHLHistoryStoreError(ValueError):
    """Une capture ou une lecture temporelle ne respecte pas le contrat NHL."""


@dataclass(frozen=True, slots=True)
class HistoryImport:
    capture_id: str
    created: bool
    row_count: int
    response_sha256: str
    effective_available_at_utc: datetime


@dataclass(frozen=True, slots=True)
class PregameTeamHistory:
    target_game_id: int
    information_cutoff_utc: datetime
    capture_id: str
    response_sha256: str
    effective_available_at_utc: datetime
    away_rows: tuple[TeamGameRow, ...]
    home_rows: tuple[TeamGameRow, ...]
    regular_coverage_complete: bool
    historical_backtest_asof_proven: bool = False
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


SCHEMA_SQL = {
    "nhl_history_captures": """CREATE TABLE nhl_history_captures (
        capture_id TEXT PRIMARY KEY,
        relative_slot TEXT NOT NULL UNIQUE,
        receipt_sha256 TEXT NOT NULL CHECK(length(receipt_sha256) = 64),
        response_sha256 TEXT NOT NULL CHECK(length(response_sha256) = 64),
        observed_at_utc TEXT NOT NULL,
        imported_at_utc TEXT NOT NULL,
        effective_available_at_utc TEXT NOT NULL,
        code_commit TEXT NOT NULL CHECK(length(code_commit) = 40),
        acquisition_mode TEXT NOT NULL CHECK(acquisition_mode IN ('https_get', 'local_file_intake')),
        source_page TEXT NOT NULL,
        attribution TEXT NOT NULL,
        regular_coverage_complete INTEGER NOT NULL CHECK(regular_coverage_complete IN (0, 1)),
        regular_row_count INTEGER NOT NULL CHECK(regular_row_count >= 0),
        playoff_row_count INTEGER NOT NULL CHECK(playoff_row_count >= 0)
    ) STRICT""",
    "nhl_history_team_rows": """CREATE TABLE nhl_history_team_rows (
        capture_id TEXT NOT NULL REFERENCES nhl_history_captures(capture_id),
        source_game_id INTEGER NOT NULL CHECK(source_game_id > 0),
        season INTEGER NOT NULL,
        game_date TEXT NOT NULL,
        game_type TEXT NOT NULL CHECK(game_type IN ('REGULAR', 'PLAYOFF')),
        team TEXT NOT NULL,
        opponent TEXT NOT NULL,
        home_or_away TEXT NOT NULL CHECK(home_or_away IN ('HOME', 'AWAY')),
        situation TEXT NOT NULL CHECK(situation IN ('all', '5on5')),
        x_goals_for TEXT NOT NULL,
        x_goals_against TEXT NOT NULL,
        goals_for INTEGER NOT NULL CHECK(goals_for >= 0),
        goals_against INTEGER NOT NULL CHECK(goals_against >= 0),
        shots_on_goal_for TEXT NOT NULL,
        shots_on_goal_against TEXT NOT NULL,
        PRIMARY KEY(capture_id, source_game_id, team, situation)
    ) STRICT""",
    "nhl_history_captures_no_update": """CREATE TRIGGER nhl_history_captures_no_update
        BEFORE UPDATE ON nhl_history_captures
        BEGIN SELECT RAISE(ABORT, 'nhl_history_captures is append-only'); END""",
    "nhl_history_captures_no_delete": """CREATE TRIGGER nhl_history_captures_no_delete
        BEFORE DELETE ON nhl_history_captures
        BEGIN SELECT RAISE(ABORT, 'nhl_history_captures is append-only'); END""",
    "nhl_history_team_rows_no_update": """CREATE TRIGGER nhl_history_team_rows_no_update
        BEFORE UPDATE ON nhl_history_team_rows
        BEGIN SELECT RAISE(ABORT, 'nhl_history_team_rows is append-only'); END""",
    "nhl_history_team_rows_no_delete": """CREATE TRIGGER nhl_history_team_rows_no_delete
        BEFORE DELETE ON nhl_history_team_rows
        BEGIN SELECT RAISE(ABORT, 'nhl_history_team_rows is append-only'); END""",
}


def _clock_utc() -> datetime:
    return datetime.now(timezone.utc)


def _verify_protocol() -> None:
    try:
        document = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise NHLHistoryStoreError("Protocole de stockage NHL illisible.") from error
    expected = {
        "schema_version": "nhl_team_history_store_v1",
        "database": "data/nhl/team_history.db",
        "source_capture_protocol": "nhl_moneypuck_team_capture_v1",
        "availability_rule": "max(capture_observed_at_utc,imported_at_utc)<=information_cutoff_utc",
        "schedule_rule": "schedule_observed_at_utc<=information_cutoff_utc",
        "historical_backtest_asof_proven": False,
        "training_permitted": False,
        "prediction_publication_permitted": False,
        "mlb_database_mutation": False,
        "synthetic_nhl_journal_mutation": False,
    }
    if not isinstance(document, dict) or any(document.get(key) != value for key, value in expected.items()):
        raise NHLHistoryStoreError("Protocole de stockage NHL incompatible.")


def _utc_text(value: datetime) -> str:
    require_utc(value, field_name="horodatage NHL")
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith(("Z", "+00:00")):
        raise NHLHistoryStoreError("Horodatage de capture non canonique.")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return require_utc(result, field_name="horodatage NHL")
    except ValueError as error:
        raise NHLHistoryStoreError("Horodatage de capture invalide.") from error


def _database_path(database_path: Path, allowed_root: Path) -> Path:
    validated = validate_nhl_database_path(database_path, allowed_root=allowed_root)
    if validated.name != "team_history.db" or Path(database_path).is_symlink():
        raise NHLHistoryStoreError("Base d'historique NHL dédiée requise.")
    return validated


def _connect(database_path: Path, allowed_root: Path, *, read_only: bool) -> sqlite3.Connection:
    path = _database_path(database_path, allowed_root)
    if read_only:
        if not path.is_file():
            raise NHLHistoryStoreError("Base d'historique NHL absente.")
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only = ON")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _verify_schema(connection: sqlite3.Connection) -> None:
    actual = {row["name"]: row["sql"] for row in connection.execute(
        "SELECT name, sql FROM sqlite_master WHERE type IN ('table', 'trigger') "
        "AND name NOT LIKE 'sqlite_%'"
    )}
    if actual != SCHEMA_SQL or connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        raise NHLHistoryStoreError("Schéma de l'historique NHL inattendu.")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise NHLHistoryStoreError("Clé étrangère NHL invalide.")


def _initialize(database_path: Path, allowed_root: Path) -> None:
    try:
        with closing(_connect(database_path, allowed_root, read_only=False)) as connection, connection:
            objects = connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type IN ('table', 'trigger') "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
            if objects == 0:
                for statement in SCHEMA_SQL.values():
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            _verify_schema(connection)
    except sqlite3.Error as error:
        raise NHLHistoryStoreError("Initialisation de la base NHL échouée.") from error


def verify_history_database(database_path: Path = DATABASE_PATH, *, allowed_root: Path = NHL_DATA_ROOT) -> None:
    """Audit en lecture seule, sans créer de fichier."""
    try:
        with closing(_connect(database_path, allowed_root, read_only=True)) as connection:
            _verify_schema(connection)
    except sqlite3.Error as error:
        raise NHLHistoryStoreError("Audit de la base NHL échoué.") from error


def _row_values(capture_id: str, game_type: str, row: TeamGameRow) -> tuple[object, ...]:
    return (capture_id, row.game_id, row.season, row.game_date.isoformat(),
            game_type, row.team, row.opponent, row.home_or_away, row.situation,
            str(row.x_goals_for), str(row.x_goals_against), row.goals_for,
            row.goals_against, str(row.shots_on_goal_for), str(row.shots_on_goal_against))


def import_verified_team_capture(
    slot: Path, *, database_path: Path = DATABASE_PATH,
    allowed_root: Path = NHL_DATA_ROOT, capture_root: Path = DEFAULT_ROOT,
) -> HistoryImport:
    """Importe explicitement une capture vérifiée dans la seule base historique NHL."""
    capture: VerifiedTeamCapture = verify_team_capture(slot, allowed_root=capture_root)
    _verify_protocol()
    for name in ("response.csv", "receipt.json", "COMPLETED"):
        if (capture.path / name).is_symlink():
            raise NHLHistoryStoreError("Un fichier de preuve NHL est un lien symbolique.")
    receipt_bytes = (capture.path / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    started = _parse_utc(receipt["request_started_at_utc"])
    if started > capture.observed_at_utc:
        raise NHLHistoryStoreError("La réception précède la demande NHL.")
    imported_at = _clock_utc()
    if imported_at < capture.observed_at_utc:
        raise NHLHistoryStoreError("Horloge d'import antérieure à la capture.")
    rows = (("REGULAR", row) for row in capture.history.regular_rows)
    rows = list(rows) + [("PLAYOFF", row) for row in capture.history.playoff_rows]
    if any(row.game_date >= capture.observed_at_utc.date() for _, row in rows):
        raise NHLHistoryStoreError("Une ligne source n'est pas antérieure à sa capture.")
    capture_id = capture.path.name
    relative_slot = capture.path.relative_to(Path(capture_root).resolve()).as_posix()
    effective = max(capture.observed_at_utc, imported_at)
    receipt_digest = sha256(receipt_bytes).hexdigest()
    _initialize(database_path, allowed_root)
    try:
        with closing(_connect(database_path, allowed_root, read_only=False)) as connection, connection:
            _verify_schema(connection)
            previous = connection.execute(
                "SELECT * FROM nhl_history_captures WHERE capture_id = ?", (capture_id,)
            ).fetchone()
            if previous is not None:
                count = connection.execute(
                    "SELECT COUNT(*) FROM nhl_history_team_rows WHERE capture_id = ?", (capture_id,)
                ).fetchone()[0]
                if (previous["receipt_sha256"] != receipt_digest
                    or previous["response_sha256"] != capture.response_sha256
                    or previous["relative_slot"] != relative_slot or count != len(rows)):
                    raise NHLHistoryStoreError("Capture déjà importée mais divergente.")
                return HistoryImport(capture_id, False, count, capture.response_sha256,
                                     _parse_utc(previous["effective_available_at_utc"]))
            connection.execute(
                """INSERT INTO nhl_history_captures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (capture_id, relative_slot, receipt_digest, capture.response_sha256,
                 _utc_text(capture.observed_at_utc), _utc_text(imported_at),
                 _utc_text(effective), receipt["code_commit"], receipt["acquisition_mode"],
                 receipt["source_page"], receipt["attribution"],
                 int(capture.history.regular_coverage_complete),
                 len(capture.history.regular_rows), len(capture.history.playoff_rows)),
            )
            connection.executemany(
                """INSERT INTO nhl_history_team_rows VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (_row_values(capture_id, game_type, row) for game_type, row in rows),
            )
            count = connection.execute(
                "SELECT COUNT(*) FROM nhl_history_team_rows WHERE capture_id = ?", (capture_id,)
            ).fetchone()[0]
            if count != len(rows):
                raise NHLHistoryStoreError("Import NHL partiel.")
    except sqlite3.Error as error:
        raise NHLHistoryStoreError("Import de la capture NHL échoué.") from error
    return HistoryImport(capture_id, True, len(rows), capture.response_sha256, effective)


def _team_row(row: sqlite3.Row) -> TeamGameRow:
    from datetime import date
    return TeamGameRow(
        game_id=row["source_game_id"], season=row["season"],
        game_date=date.fromisoformat(row["game_date"]), team=row["team"],
        opponent=row["opponent"], home_or_away=row["home_or_away"],
        situation=row["situation"], x_goals_for=Decimal(row["x_goals_for"]),
        x_goals_against=Decimal(row["x_goals_against"]),
        goals_for=row["goals_for"], goals_against=row["goals_against"],
        shots_on_goal_for=Decimal(row["shots_on_goal_for"]),
        shots_on_goal_against=Decimal(row["shots_on_goal_against"]),
    )


def load_pregame_team_history(
    game: ScheduledGame, *, schedule_observed_at_utc: datetime,
    lead_minutes: int, database_path: Path = DATABASE_PATH,
    allowed_root: Path = NHL_DATA_ROOT,
) -> PregameTeamHistory:
    """Relit uniquement les matchs réguliers prouvés avant le cutoff cible."""
    _verify_protocol()
    if not isinstance(game, ScheduledGame) or game.away_abbr == game.home_abbr:
        raise NHLHistoryStoreError("Match cible NHL invalide.")
    require_utc(schedule_observed_at_utc, field_name="calendrier NHL")
    cutoff = build_information_cutoff(game.start_utc, lead_minutes=lead_minutes)
    if schedule_observed_at_utc > cutoff:
        raise NHLHistoryStoreError("Calendrier reçu après le cutoff.")
    verify_history_database(database_path, allowed_root=allowed_root)
    try:
        with closing(_connect(database_path, allowed_root, read_only=True)) as connection:
            proof = connection.execute(
                "SELECT * FROM nhl_history_captures WHERE effective_available_at_utc <= ? "
                "ORDER BY effective_available_at_utc DESC, capture_id DESC LIMIT 1",
                (_utc_text(cutoff),),
            ).fetchone()
            if proof is None:
                raise NHLHistoryStoreError("Aucune capture historique avant le cutoff.")
            stored = connection.execute(
                """SELECT * FROM nhl_history_team_rows
                   WHERE capture_id = ? AND game_type = 'REGULAR'
                   AND game_date < ? AND source_game_id <> ?
                   AND team IN (?, ?)
                   ORDER BY game_date, source_game_id, team, situation""",
                (proof["capture_id"], game.start_utc.date().isoformat(), game.game_id,
                 game.away_abbr, game.home_abbr),
            ).fetchall()
    except sqlite3.Error as error:
        raise NHLHistoryStoreError("Lecture pré-match NHL échouée.") from error
    away = tuple(_team_row(row) for row in stored if row["team"] == game.away_abbr)
    home = tuple(_team_row(row) for row in stored if row["team"] == game.home_abbr)
    return PregameTeamHistory(
        target_game_id=game.game_id, information_cutoff_utc=cutoff,
        capture_id=proof["capture_id"], response_sha256=proof["response_sha256"],
        effective_available_at_utc=_parse_utc(proof["effective_available_at_utc"]),
        away_rows=away, home_rows=home,
        regular_coverage_complete=bool(proof["regular_coverage_complete"]),
    )


def load_verified_schedule_team_history(
    schedule_slot: Path, *, target_game_id: int, lead_minutes: int,
    database_path: Path = DATABASE_PATH, allowed_root: Path = NHL_DATA_ROOT,
) -> PregameTeamHistory:
    """Rattache l'historique à un match d'une capture de calendrier scellée."""
    from .schedule_capture_readiness import audit_schedule_capture

    if type(target_game_id) is not int or target_game_id <= 0:
        raise NHLHistoryStoreError("Identifiant de match cible invalide.")
    schedule = audit_schedule_capture(schedule_slot)
    matches = [item for item in schedule["games"] if item["game_id"] == target_game_id]
    if len(matches) != 1:
        raise NHLHistoryStoreError("Match cible absent du calendrier NHL vérifié.")
    item = matches[0]
    game = ScheduledGame(
        game_id=item["game_id"], season=item["season"],
        start_utc=_parse_utc(item["scheduled_start_utc"]),
        away_abbr=item["away_abbr"], home_abbr=item["home_abbr"],
    )
    return load_pregame_team_history(
        game, schedule_observed_at_utc=_parse_utc(schedule["observed_at_utc"]),
        lead_minutes=lead_minutes, database_path=database_path,
        allowed_root=allowed_root,
    )
