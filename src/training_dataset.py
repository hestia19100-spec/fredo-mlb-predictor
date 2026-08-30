"""Construit le premier jeu d’apprentissage MLB sans utiliser le jour cible."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
import csv
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import io
from pathlib import Path
import sqlite3
from typing import Iterable

from src.database import DATABASE_PATH


DATASET_VERSION = "mlb_team_form_v1"
CALENDAR_POLICY = "J_MINUS_1_BY_OFFICIAL_DATE"
TEMPORAL_VERDICT = "UNVERIFIABLE"
TEMPORAL_REASON = (
    "Les variables utilisent uniquement des résultats dont official_date "
    "est strictement antérieure au match cible. La table games ne conserve "
    "cependant pas l’instant réel auquel un résultat suspendu est devenu "
    "définitif : la disponibilité historique exacte ne peut donc pas encore "
    "être certifiée."
)

FINAL_STATUS_CODES = frozenset({"F"})
FINAL_STATUS_DETAILS = frozenset(
    {
        "FINAL",
        "GAME OVER",
        "COMPLETED EARLY",
    }
)

REQUIRED_TABLES = frozenset({"games", "ingestion_runs", "teams"})
REQUIRED_GAME_COLUMNS = frozenset(
    {
        "game_id",
        "season",
        "official_date",
        "game_type",
        "status_code",
        "status_detail",
        "away_team_id",
        "home_team_id",
        "away_score",
        "home_score",
    }
)
REQUIRED_TEAM_COLUMNS = frozenset({"team_id"})

METADATA_COLUMNS = (
    "game_id",
    "season",
    "official_date",
    "feature_as_of_date",
    "away_team_id",
    "home_team_id",
    "away_max_source_date",
    "home_max_source_date",
)

FEATURE_COLUMNS = (
    "away_games_before",
    "away_win_pct_before",
    "away_runs_scored_per_game_before",
    "away_runs_allowed_per_game_before",
    "home_games_before",
    "home_win_pct_before",
    "home_runs_scored_per_game_before",
    "home_runs_allowed_per_game_before",
)

TARGET_COLUMN = "home_win"
CSV_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + (TARGET_COLUMN,)


class TrainingDatasetError(RuntimeError):
    """Erreur empêchant de produire un dataset fiable."""


@dataclass(frozen=True, slots=True)
class HistoricalGame:
    """Résultat final minimal utilisé pour créer l’historique."""

    game_id: int
    season: int
    official_date: date
    away_team_id: int
    home_team_id: int
    away_score: int
    home_score: int


@dataclass(frozen=True, slots=True)
class TeamSnapshot:
    """État d’une équipe arrêté avant une journée de matchs."""

    games: int
    win_pct: float
    runs_scored_per_game: float
    runs_allowed_per_game: float
    max_source_date: date


@dataclass(slots=True)
class TeamState:
    """Cumuls internes d’une équipe pour une seule saison."""

    games: int = 0
    wins: int = 0
    runs_scored: int = 0
    runs_allowed: int = 0
    max_source_date: date | None = None

    def snapshot(self) -> TeamSnapshot:
        """Fige les cumuls actuels sous forme de moyennes."""
        if self.games <= 0 or self.max_source_date is None:
            raise TrainingDatasetError(
                "Impossible de créer un instantané sans historique."
            )

        return TeamSnapshot(
            games=self.games,
            win_pct=self.wins / self.games,
            runs_scored_per_game=self.runs_scored / self.games,
            runs_allowed_per_game=self.runs_allowed / self.games,
            max_source_date=self.max_source_date,
        )

    def add_game(
        self,
        *,
        game_date: date,
        runs_scored: int,
        runs_allowed: int,
    ) -> None:
        """Ajoute un résultat après la création des lignes de la journée."""
        if self.max_source_date is not None and game_date < self.max_source_date:
            raise TrainingDatasetError(
                "Les résultats ne sont pas ajoutés dans l’ordre chronologique."
            )

        self.games += 1
        self.wins += int(runs_scored > runs_allowed)
        self.runs_scored += runs_scored
        self.runs_allowed += runs_allowed
        self.max_source_date = game_date


@dataclass(frozen=True, slots=True)
class TrainingRow:
    """Une cible et ses variables disponibles selon la règle J-1."""

    game_id: int
    season: int
    official_date: date
    feature_as_of_date: date
    away_team_id: int
    home_team_id: int
    away_max_source_date: date
    home_max_source_date: date
    away_games_before: int
    away_win_pct_before: float
    away_runs_scored_per_game_before: float
    away_runs_allowed_per_game_before: float
    home_games_before: int
    home_win_pct_before: float
    home_runs_scored_per_game_before: float
    home_runs_allowed_per_game_before: float
    home_win: int


@dataclass(frozen=True, slots=True)
class SeasonDatasetSummary:
    """Résumé des lignes réellement conservées pour une saison."""

    season: int
    rows: int
    home_wins: int
    first_date: date
    last_date: date


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    """Dataset déterministe et informations nécessaires à son audit."""

    version: str
    cutoff_date: date
    minimum_history_games: int
    source_snapshot_sha256: str
    calendar_policy: str
    temporal_verdict: str
    temporal_reason: str
    eligible_games: int
    skipped_insufficient_history: int
    rows: tuple[TrainingRow, ...]
    season_summaries: tuple[SeasonDatasetSummary, ...]
    csv_sha256: str


@dataclass(frozen=True, slots=True)
class DatasetExport:
    """Résultat de l’écriture ou de la vérification du CSV."""

    path: Path
    sha256: str
    size_bytes: int
    already_existed: bool


def _normalize_text(value: object) -> str:
    """Normalise les statuts avant comparaison."""
    if value is None:
        return ""

    return str(value).strip().upper()


def _is_integer(value: object) -> bool:
    """Reconnaît un entier SQLite tout en refusant les booléens."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_final(row: sqlite3.Row) -> bool:
    """Reconnaît les matchs terminés acceptés par le projet."""
    return (
        _normalize_text(row["status_code"]) in FINAL_STATUS_CODES
        or _normalize_text(row["status_detail"])
        in FINAL_STATUS_DETAILS
    )


def _parse_date(value: object, *, field_name: str) -> date:
    """Valide une date ISO obligatoire."""
    if not isinstance(value, str):
        raise TrainingDatasetError(
            f"{field_name} doit être une date ISO."
        )

    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise TrainingDatasetError(
            f"{field_name} invalide : {value}."
        ) from error


def _readonly_uri(path: Path) -> str:
    """Produit une URI SQLite qui interdit toute écriture à la source."""
    return path.resolve().as_uri() + "?mode=ro"


def _table_names(connection: sqlite3.Connection) -> set[str]:
    """Retourne les tables du snapshot logique."""
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """
    ).fetchall()

    return {str(row[0]) for row in rows}


def _table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    """Retourne les colonnes d’une table connue."""
    rows = connection.execute(
        f'PRAGMA table_info("{table_name}")'
    ).fetchall()

    return {str(row[1]) for row in rows}


def _validate_game(row: sqlite3.Row) -> HistoricalGame:
    """Transforme un résultat final en objet contrôlé."""
    game_id = row["game_id"]
    season = row["season"]
    away_team_id = row["away_team_id"]
    home_team_id = row["home_team_id"]
    away_score = row["away_score"]
    home_score = row["home_score"]

    if not _is_integer(game_id) or game_id <= 0:
        raise TrainingDatasetError("Identifiant de match invalide.")
    if not _is_integer(season) or season <= 0:
        raise TrainingDatasetError(
            f"Saison invalide pour le match {game_id}."
        )
    if (
        not _is_integer(away_team_id)
        or away_team_id <= 0
        or not _is_integer(home_team_id)
        or home_team_id <= 0
        or away_team_id == home_team_id
    ):
        raise TrainingDatasetError(
            f"Équipes invalides pour le match {game_id}."
        )
    if (
        not _is_integer(away_score)
        or not _is_integer(home_score)
        or away_score < 0
        or home_score < 0
        or away_score == home_score
    ):
        raise TrainingDatasetError(
            f"Scores finaux invalides pour le match {game_id}."
        )

    official_date = _parse_date(
        row["official_date"],
        field_name=f"official_date du match {game_id}",
    )

    if official_date.year != season:
        raise TrainingDatasetError(
            f"La date et la saison du match {game_id} sont incohérentes."
        )

    return HistoricalGame(
        game_id=int(game_id),
        season=int(season),
        official_date=official_date,
        away_team_id=int(away_team_id),
        home_team_id=int(home_team_id),
        away_score=int(away_score),
        home_score=int(home_score),
    )


def _load_games(
    *,
    database_path: Path,
    cutoff_date: date,
) -> tuple[list[HistoricalGame], str]:
    """Charge un snapshot cohérent sans modifier la base originale."""
    database_path = Path(database_path)
    if not database_path.is_file():
        raise TrainingDatasetError(
            f"Base SQLite absente : {database_path}"
        )

    try:
        with closing(
            sqlite3.connect(
                _readonly_uri(database_path),
                uri=True,
                timeout=30,
            )
        ) as source_connection:
            with closing(sqlite3.connect(":memory:")) as connection:
                source_connection.backup(connection)
                snapshot_sha256 = hashlib.sha256(
                    connection.serialize()
                ).hexdigest()
                connection.row_factory = sqlite3.Row

                missing_tables = REQUIRED_TABLES - _table_names(connection)
                if missing_tables:
                    raise TrainingDatasetError(
                        "Tables obligatoires absentes : "
                        + ", ".join(sorted(missing_tables))
                    )

                missing_columns = (
                    REQUIRED_GAME_COLUMNS
                    - _table_columns(connection, "games")
                )
                if missing_columns:
                    raise TrainingDatasetError(
                        "Colonnes games obligatoires absentes : "
                        + ", ".join(sorted(missing_columns))
                    )

                missing_team_columns = (
                    REQUIRED_TEAM_COLUMNS
                    - _table_columns(connection, "teams")
                )
                if missing_team_columns:
                    raise TrainingDatasetError(
                        "Colonnes teams obligatoires absentes : "
                        + ", ".join(sorted(missing_team_columns))
                    )

                team_ids = {
                    int(row[0])
                    for row in connection.execute(
                        "SELECT team_id FROM teams"
                    ).fetchall()
                    if _is_integer(row[0]) and row[0] > 0
                }

                active_ingestions = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM ingestion_runs
                        WHERE status = 'started'
                        """
                    ).fetchone()[0]
                )
                if active_ingestions:
                    raise TrainingDatasetError(
                        "Une collecte est encore active ; relance après sa fin."
                    )

                integrity_messages = [
                    str(row[0])
                    for row in connection.execute(
                        "PRAGMA integrity_check"
                    ).fetchall()
                    if str(row[0]).lower() != "ok"
                ]
                if integrity_messages:
                    raise TrainingDatasetError(
                        "Le contrôle d’intégrité SQLite a échoué."
                    )

                foreign_key_violations = connection.execute(
                    "PRAGMA foreign_key_check"
                ).fetchall()
                if foreign_key_violations:
                    raise TrainingDatasetError(
                        "SQLite signale des références étrangères invalides."
                    )

                raw_rows = connection.execute(
                    """
                    SELECT
                        game_id,
                        season,
                        official_date,
                        game_type,
                        status_code,
                        status_detail,
                        away_team_id,
                        home_team_id,
                        away_score,
                        home_score
                    FROM games
                    """
                ).fetchall()

    except sqlite3.Error as error:
        raise TrainingDatasetError(
            f"Lecture SQLite impossible : {error}"
        ) from error

    games: list[HistoricalGame] = []
    seen_game_ids: set[int] = set()

    for row in raw_rows:
        if row["game_type"] != "R" or not _is_final(row):
            continue

        official_date = _parse_date(
            row["official_date"],
            field_name=f"official_date du match {row['game_id']}",
        )
        if official_date > cutoff_date:
            continue

        game = _validate_game(row)
        if (
            game.away_team_id not in team_ids
            or game.home_team_id not in team_ids
        ):
            raise TrainingDatasetError(
                f"Équipe inconnue pour le match {game.game_id}."
            )
        if game.game_id in seen_game_ids:
            raise TrainingDatasetError(
                f"Identifiant de match dupliqué : {game.game_id}."
            )

        seen_game_ids.add(game.game_id)
        games.append(game)

    games.sort(
        key=lambda game: (
            game.season,
            game.official_date,
            game.game_id,
        )
    )

    if not games:
        raise TrainingDatasetError(
            "Aucun résultat final de saison régulière avant la coupure."
        )

    return games, snapshot_sha256


def _make_training_row(
    *,
    game: HistoricalGame,
    away_snapshot: TeamSnapshot,
    home_snapshot: TeamSnapshot,
) -> TrainingRow:
    """Assemble une ligne après vérification de l’antériorité."""
    if (
        away_snapshot.max_source_date >= game.official_date
        or home_snapshot.max_source_date >= game.official_date
    ):
        raise TrainingDatasetError(
            f"Fuite temporelle détectée pour le match {game.game_id}."
        )

    return TrainingRow(
        game_id=game.game_id,
        season=game.season,
        official_date=game.official_date,
        feature_as_of_date=game.official_date - timedelta(days=1),
        away_team_id=game.away_team_id,
        home_team_id=game.home_team_id,
        away_max_source_date=away_snapshot.max_source_date,
        home_max_source_date=home_snapshot.max_source_date,
        away_games_before=away_snapshot.games,
        away_win_pct_before=away_snapshot.win_pct,
        away_runs_scored_per_game_before=(
            away_snapshot.runs_scored_per_game
        ),
        away_runs_allowed_per_game_before=(
            away_snapshot.runs_allowed_per_game
        ),
        home_games_before=home_snapshot.games,
        home_win_pct_before=home_snapshot.win_pct,
        home_runs_scored_per_game_before=(
            home_snapshot.runs_scored_per_game
        ),
        home_runs_allowed_per_game_before=(
            home_snapshot.runs_allowed_per_game
        ),
        home_win=int(game.home_score > game.away_score),
    )


def _update_states(
    states: dict[int, TeamState],
    games: Iterable[HistoricalGame],
) -> None:
    """Ajoute tous les résultats du jour après la création des cibles."""
    for game in games:
        away_state = states.setdefault(game.away_team_id, TeamState())
        home_state = states.setdefault(game.home_team_id, TeamState())

        away_state.add_game(
            game_date=game.official_date,
            runs_scored=game.away_score,
            runs_allowed=game.home_score,
        )
        home_state.add_game(
            game_date=game.official_date,
            runs_scored=game.home_score,
            runs_allowed=game.away_score,
        )


def _season_summaries(
    rows: tuple[TrainingRow, ...],
) -> tuple[SeasonDatasetSummary, ...]:
    """Résume la cible obtenue dans chaque saison."""
    grouped: dict[int, list[TrainingRow]] = defaultdict(list)
    for row in rows:
        grouped[row.season].append(row)

    summaries: list[SeasonDatasetSummary] = []
    for season in sorted(grouped):
        season_rows = grouped[season]
        summaries.append(
            SeasonDatasetSummary(
                season=season,
                rows=len(season_rows),
                home_wins=sum(row.home_win for row in season_rows),
                first_date=min(row.official_date for row in season_rows),
                last_date=max(row.official_date for row in season_rows),
            )
        )

    return tuple(summaries)


def _row_to_csv(row: TrainingRow) -> dict[str, object]:
    """Formate une ligne de manière stable pour le CSV."""
    return {
        "game_id": row.game_id,
        "season": row.season,
        "official_date": row.official_date.isoformat(),
        "feature_as_of_date": row.feature_as_of_date.isoformat(),
        "away_team_id": row.away_team_id,
        "home_team_id": row.home_team_id,
        "away_max_source_date": row.away_max_source_date.isoformat(),
        "home_max_source_date": row.home_max_source_date.isoformat(),
        "away_games_before": row.away_games_before,
        "away_win_pct_before": f"{row.away_win_pct_before:.6f}",
        "away_runs_scored_per_game_before": (
            f"{row.away_runs_scored_per_game_before:.6f}"
        ),
        "away_runs_allowed_per_game_before": (
            f"{row.away_runs_allowed_per_game_before:.6f}"
        ),
        "home_games_before": row.home_games_before,
        "home_win_pct_before": f"{row.home_win_pct_before:.6f}",
        "home_runs_scored_per_game_before": (
            f"{row.home_runs_scored_per_game_before:.6f}"
        ),
        "home_runs_allowed_per_game_before": (
            f"{row.home_runs_allowed_per_game_before:.6f}"
        ),
        "home_win": row.home_win,
    }


def render_training_csv(rows: Iterable[TrainingRow]) -> bytes:
    """Produit les octets CSV canoniques utilisés pour le SHA-256."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=CSV_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(_row_to_csv(row))

    return output.getvalue().encode("utf-8")


def build_training_dataset(
    *,
    cutoff_date: date,
    minimum_history_games: int = 10,
    database_path: Path = DATABASE_PATH,
) -> TrainingDataset:
    """Construit les lignes avec une mise à jour différée par journée."""
    if type(cutoff_date) is not date:
        raise ValueError("cutoff_date doit être une date.")

    if (
        not _is_integer(minimum_history_games)
        or minimum_history_games < 1
        or minimum_history_games > 162
    ):
        raise ValueError(
            "minimum_history_games doit être compris entre 1 et 162."
        )

    games, source_snapshot_sha256 = _load_games(
        database_path=database_path,
        cutoff_date=cutoff_date,
    )

    rows: list[TrainingRow] = []
    skipped_insufficient_history = 0
    current_season: int | None = None
    states: dict[int, TeamState] = {}
    index = 0

    while index < len(games):
        first_game = games[index]
        day_key = (first_game.season, first_game.official_date)
        day_games: list[HistoricalGame] = []

        while index < len(games):
            candidate = games[index]
            candidate_key = (
                candidate.season,
                candidate.official_date,
            )
            if candidate_key != day_key:
                break

            day_games.append(candidate)
            index += 1

        if current_season != first_game.season:
            current_season = first_game.season
            states = {}

        for game in day_games:
            away_state = states.get(game.away_team_id)
            home_state = states.get(game.home_team_id)

            if (
                away_state is None
                or home_state is None
                or away_state.games < minimum_history_games
                or home_state.games < minimum_history_games
            ):
                skipped_insufficient_history += 1
                continue

            rows.append(
                _make_training_row(
                    game=game,
                    away_snapshot=away_state.snapshot(),
                    home_snapshot=home_state.snapshot(),
                )
            )

        _update_states(states, day_games)

    ordered_rows = tuple(rows)
    if not ordered_rows:
        raise TrainingDatasetError(
            "Aucune ligne ne possède assez d’historique avant la coupure."
        )

    csv_content = render_training_csv(ordered_rows)

    return TrainingDataset(
        version=DATASET_VERSION,
        cutoff_date=cutoff_date,
        minimum_history_games=minimum_history_games,
        source_snapshot_sha256=source_snapshot_sha256,
        calendar_policy=CALENDAR_POLICY,
        temporal_verdict=TEMPORAL_VERDICT,
        temporal_reason=TEMPORAL_REASON,
        eligible_games=len(games),
        skipped_insufficient_history=skipped_insufficient_history,
        rows=ordered_rows,
        season_summaries=_season_summaries(ordered_rows),
        csv_sha256=hashlib.sha256(csv_content).hexdigest(),
    )


def write_training_dataset(
    dataset: TrainingDataset,
    output_path: Path,
) -> DatasetExport:
    """Écrit le CSV sans jamais remplacer un contenu différent."""
    output_path = Path(output_path)
    content = render_training_csv(dataset.rows)
    content_sha256 = hashlib.sha256(content).hexdigest()

    if content_sha256 != dataset.csv_sha256:
        raise TrainingDatasetError(
            "Le contenu CSV ne correspond plus au dataset construit."
        )

    if output_path.exists():
        try:
            existing_content = output_path.read_bytes()
        except OSError as error:
            raise TrainingDatasetError(
                f"Lecture du CSV existant impossible : {error}"
            ) from error

        existing_sha256 = hashlib.sha256(existing_content).hexdigest()
        if existing_sha256 != content_sha256:
            raise TrainingDatasetError(
                "Le fichier de sortie existe déjà avec un contenu différent."
            )

        return DatasetExport(
            path=output_path.resolve(),
            sha256=content_sha256,
            size_bytes=len(content),
            already_existed=True,
        )

    created_output = False
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("xb") as destination:
            created_output = True
            destination.write(content)
            destination.flush()
    except OSError as error:
        if created_output:
            try:
                output_path.unlink()
            except OSError:
                pass
        raise TrainingDatasetError(
            f"Écriture du CSV impossible : {error}"
        ) from error

    return DatasetExport(
        path=output_path.resolve(),
        sha256=content_sha256,
        size_bytes=len(content),
        already_existed=False,
    )


def display_dataset(dataset: TrainingDataset) -> None:
    """Affiche un résumé pédagogique du dataset."""
    home_wins = sum(row.home_win for row in dataset.rows)
    home_win_rate = home_wins / len(dataset.rows)

    print("Jeu d’apprentissage MLB")
    print(f"Version : {dataset.version}")
    print(f"Date de coupure : {dataset.cutoff_date.isoformat()}")
    print(f"Politique calendrier : {dataset.calendar_policy}")
    print(
        "Disponibilité historique réelle : "
        f"{dataset.temporal_verdict}"
    )
    print(f"SHA-256 du snapshot source : {dataset.source_snapshot_sha256}")
    print(f"Matchs finaux admissibles : {dataset.eligible_games}")
    print(
        "Matchs écartés faute d’historique : "
        f"{dataset.skipped_insufficient_history}"
    )
    print(f"Lignes conservées : {len(dataset.rows)}")
    print(f"Victoires à domicile : {home_wins} ({home_win_rate:.2%})")
    print(f"SHA-256 du CSV canonique : {dataset.csv_sha256}")

    print("\nPar saison")
    print("Saison | Lignes | Victoires domicile | Taux domicile | Début | Fin")
    for item in dataset.season_summaries:
        rate = item.home_wins / item.rows
        print(
            f"{item.season} | {item.rows} | {item.home_wins} | "
            f"{rate:.2%} | {item.first_date.isoformat()} | "
            f"{item.last_date.isoformat()}"
        )

    print("\nLimite temporelle")
    print(dataset.temporal_reason)
    print(
        "Aucun lanceur, score du match cible, statut final, stade ou "
        "horaire n’est utilisé comme variable."
    )


def _argument_date(value: str) -> date:
    """Valide une date passée en argument."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "La date doit respecter le format AAAA-MM-JJ."
        ) from error


def main() -> None:
    """Construit, affiche et exporte éventuellement le dataset."""
    parser = argparse.ArgumentParser(
        description=(
            "Construit le dataset MLB team-form avec une règle stricte J-1."
        )
    )
    parser.add_argument(
        "--cutoff-date",
        required=True,
        type=_argument_date,
        help="Dernière date officielle incluse, au format AAAA-MM-JJ.",
    )
    parser.add_argument(
        "--minimum-history-games",
        type=int,
        default=10,
        help="Matchs antérieurs requis pour chacune des deux équipes.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help="Chemin de la base SQLite source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Chemin CSV facultatif ; aucun fichier n’est écrit sans lui.",
    )
    arguments = parser.parse_args()

    try:
        dataset = build_training_dataset(
            cutoff_date=arguments.cutoff_date,
            minimum_history_games=arguments.minimum_history_games,
            database_path=arguments.database,
        )
        display_dataset(dataset)

        if arguments.output is None:
            print("\nMode aperçu : aucun fichier CSV n’a été écrit.")
        else:
            export = write_training_dataset(dataset, arguments.output)
            action = "vérifié" if export.already_existed else "créé"
            print(f"\nCSV {action} : {export.path}")
            print(f"Taille : {export.size_bytes} octets")
            print(f"SHA-256 : {export.sha256}")
    except (TrainingDatasetError, ValueError) as error:
        parser.exit(
            status=1,
            message=f"Échec du dataset MLB : {error}\n",
        )


if __name__ == "__main__":
    main()
