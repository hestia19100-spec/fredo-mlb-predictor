"""Tests du registre des candidats et des variables J-1 shadow v2.

Toutes les donnees sont synthetiques et tous les slots vivent dans des
repertoires temporaires. Aucun modele ni calendrier MLB reel n'est utilise.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager
from copy import deepcopy
import builtins
import csv
from dataclasses import replace
from datetime import date, timedelta
import gzip
import hashlib
import inspect
import io
import json
from multiprocessing import get_context
from pathlib import Path
import sqlite3
import sys
from threading import Event
import unittest
from unittest.mock import patch

import src.shadow_prediction as shadow_prediction
from src.shadow_prediction import (
    CANDIDATE_LEDGER_FILENAME,
    FEATURES_FILENAME,
    ShadowCandidateFeaturesPublication,
    ShadowPredictionError,
    ShadowPredictionSlotConsumedError,
    build_and_publish_candidate_ledger_and_features,
)

try:
    from tests import test_shadow_prediction_source_snapshot as source_tests
except ImportError:
    import test_shadow_prediction_source_snapshot as source_tests


TARGET_DATE = source_tests.TARGET_DATE
TARGET = date.fromisoformat(TARGET_DATE)
AS_OF_DATE = (TARGET - timedelta(days=1)).isoformat()
PROTOCOL_SHA256 = source_tests.PROTOCOL_SHA256
EXCLUSION_REASONS = (
    "POSTPONED",
    "CANCELLED",
    "START_TIME_MISSING",
    "INSUFFICIENT_BOTH_HISTORY",
    "INSUFFICIENT_AWAY_HISTORY",
    "INSUFFICIENT_HOME_HISTORY",
)
LEDGER_COLUMNS = (
    "batch_id", "game_id", "occurrence_key", "season", "official_date",
    "away_team_id", "home_team_id", "scheduled_start_utc", "status_code",
    "abstract_state", "detailed_state", "eligibility_status",
    "exclusion_reason",
)
FEATURE_COLUMNS = (
    "prediction_id", "batch_id", "game_id", "occurrence_key", "season",
    "official_date", "away_team_id", "home_team_id", "scheduled_start_utc",
    "feature_as_of_date", "away_max_source_date", "home_max_source_date",
    "away_games_before", "away_win_pct_before",
    "away_runs_scored_per_game_before", "away_runs_allowed_per_game_before",
    "home_games_before", "home_win_pct_before",
    "home_runs_scored_per_game_before", "home_runs_allowed_per_game_before",
    "feature_row_sha256",
)
FEATURE_VALUE_COLUMNS = FEATURE_COLUMNS[12:20]
FORBIDDEN_FIELDS = {
    "away_score", "home_score", "home_win", "target", "outcome", "odds",
    "implied_probability", "pick", "stake", "edge", "expected_value",
    "profit", "roi", "venue", "probable_pitcher",
}
SCORES = (
    (5, 2), (1, 4), (6, 1), (2, 3), (4, 1),
    (3, 5), (7, 2), (0, 3), (2, 1), (4, 2),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_gzip(value: object) -> bytes:
    stream = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=9
    ) as compressed:
        compressed.write(_canonical_json(value) + b"\n")
    return stream.getvalue()


def _identifier(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _csv_bytes(columns, rows) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(
        stream, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if row[column] is None else row[column]
                         for column in columns])
    return stream.getvalue().encode("utf-8")


def _read_csv(path: Path):
    content = path.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
    return content, tuple(reader.fieldnames or ()), list(reader)


def _attempt_candidate_features_in_child(
    reservation, activation_publication, source_publication, project, results
) -> None:
    """Vrai concurrent isole: il ne doit ni attendre ni reparer le slot."""
    try:
        build_and_publish_candidate_ledger_and_features(
            reservation,
            activation_publication,
            source_publication,
            project_directory=project,
        )
        results.put(("success", None))
    except ShadowPredictionSlotConsumedError:
        results.put(("consumed", None))
    except BaseException as error:
        results.put(("error", type(error).__name__ + ": " + str(error)))


class ShadowCandidateFeaturesTests(unittest.TestCase):
    """Le quatrieme et cinquieme fichiers restent lies au snapshot publie."""

    def setUp(self) -> None:
        # Composer la fixture, sans heriter de ses 34 tests ni les importer
        # comme classe visible par unittest dans ce module.
        self.fixture = source_tests.ShadowSourceSnapshotTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.project = self.fixture.project
        self.slot = self.fixture.slot
        self.reservation = self.fixture.reservation
        self.activation_publication = self.fixture.activation_publication
        self.source_publication = None
        self._set_histories({101: 10, 102: 12, 103: 10, 104: 10})

    def _set_histories(self, counts: dict[int, int]) -> None:
        """Chaque equipe joue contre un adversaire synthetique distinct."""
        with closing(sqlite3.connect(self.fixture.database_path)) as connection, connection:
            connection.execute("DELETE FROM games")
            for team_id, count in counts.items():
                opponent_id = team_id + 1000
                connection.execute(
                    "INSERT OR IGNORE INTO teams (team_id, name, abbreviation) "
                    "VALUES (?, ?, ?)",
                    (opponent_id, f"Adversaire {team_id}", f"X{team_id}"),
                )
                for index in range(count):
                    team_score, opponent_score = SCORES[index % len(SCORES)]
                    if index % 2 == 0:
                        away_id, home_id = team_id, opponent_id
                        away_score, home_score = team_score, opponent_score
                    else:
                        away_id, home_id = opponent_id, team_id
                        away_score, home_score = opponent_score, team_score
                    connection.execute(
                        "INSERT INTO games (game_id, season, official_date, "
                        "game_type, status_code, status_detail, away_team_id, "
                        "home_team_id, away_score, home_score) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            team_id * 1000 + index,
                            2026,
                            (TARGET - timedelta(days=count - index)).isoformat(),
                            "R", "F", "Final", away_id, home_id,
                            away_score, home_score,
                        ),
                    )

    def _prepare(self, games=None):
        result = self.fixture._new_success_result(games=games)
        publication, ingestion_mock = self.fixture._capture(result)
        self.assertEqual(ingestion_mock.call_count, 1)
        self.source_publication = publication
        return publication

    def _build(self, source_publication=None):
        selected = (
            self.source_publication
            if source_publication is None else source_publication
        )
        self.assertIsNotNone(selected)
        return build_and_publish_candidate_ledger_and_features(
            self.reservation,
            self.activation_publication,
            selected,
            project_directory=self.project,
        )

    def _predecessor_bytes(self):
        return {path.name: path.read_bytes() for path in self.slot.iterdir()}

    def _assert_predecessors_unchanged(self, before):
        for name, content in before.items():
            self.assertEqual((self.slot / name).read_bytes(), content, name)

    def _assert_no_csv(self):
        self.assertFalse((self.slot / CANDIDATE_LEDGER_FILENAME).exists())
        self.assertFalse((self.slot / FEATURES_FILENAME).exists())

    def _payload(self):
        return json.loads(
            gzip.decompress(self.source_publication.snapshot_path.read_bytes())
        )

    @contextmanager
    def _mutated_payload(self, mutation):
        original_publication = self.source_publication
        original_bytes = original_publication.snapshot_path.read_bytes()
        payload = deepcopy(json.loads(gzip.decompress(original_bytes)))
        mutation(payload)
        changed_bytes = _canonical_gzip(payload)
        original_publication.snapshot_path.write_bytes(changed_bytes)
        changed_publication = replace(
            original_publication,
            snapshot_sha256=hashlib.sha256(changed_bytes).hexdigest(),
            snapshot_size_bytes=len(changed_bytes),
        )
        try:
            yield changed_publication
        finally:
            original_publication.snapshot_path.write_bytes(original_bytes)

    def _assert_payload_refused(self, mutation):
        with self._mutated_payload(mutation) as publication:
            with self.assertRaises(ShadowPredictionError):
                self._build(publication)
            self._assert_no_csv()

    def _expected_ledger_row(self, game, reason=None):
        return {
            "batch_id": self.reservation.batch_id,
            "game_id": game.game_id,
            "occurrence_key": _identifier([
                "shadow_occurrence_v2", game.game_id, game.official_date,
                game.game_datetime_utc,
            ]),
            "season": game.season,
            "official_date": game.official_date,
            "away_team_id": game.away_team_id,
            "home_team_id": game.home_team_id,
            "scheduled_start_utc": game.game_datetime_utc,
            "status_code": game.status_code,
            "abstract_state": game.abstract_state,
            "detailed_state": game.status_detail,
            "eligibility_status": "ELIGIBLE" if reason is None else "EXCLUDED",
            "exclusion_reason": reason,
        }

    def _expected_feature_row(self, game, away_count=10, home_count=10):
        ledger = self._expected_ledger_row(game)
        values = {
            "prediction_id": _identifier([
                "shadow_prediction_v2", PROTOCOL_SHA256, game.game_id,
                game.official_date, game.game_datetime_utc,
            ]),
            **{column: ledger[column] for column in (
                "batch_id", "game_id", "occurrence_key", "season",
                "official_date", "away_team_id", "home_team_id",
                "scheduled_start_utc",
            )},
            "feature_as_of_date": AS_OF_DATE,
            "away_max_source_date": AS_OF_DATE,
            "home_max_source_date": AS_OF_DATE,
        }
        for side, count in (("away", away_count), ("home", home_count)):
            selected_scores = [SCORES[index % len(SCORES)]
                               for index in range(count)]
            wins = sum(scored > allowed for scored, allowed in selected_scores)
            values[f"{side}_games_before"] = count
            values[f"{side}_win_pct_before"] = format(wins / count, ".6f")
            values[f"{side}_runs_scored_per_game_before"] = format(
                sum(scored for scored, _ in selected_scores) / count, ".6f"
            )
            values[f"{side}_runs_allowed_per_game_before"] = format(
                sum(allowed for _, allowed in selected_scores) / count, ".6f"
            )
        values["feature_row_sha256"] = _identifier([
            "shadow_feature_row_v2",
            [values[column] for column in FEATURE_COLUMNS[:-1]],
        ])
        return values

    def test_exact_bytes_schemas_identifiers_hashes_and_publication_proof(self):
        games = self.fixture._default_games()
        self._prepare(games)
        predecessors = self._predecessor_bytes()

        publication = self._build()

        self.assertIs(type(publication), ShadowCandidateFeaturesPublication)
        ordered = sorted(games, key=lambda game: (
            game.game_datetime_utc is None,
            game.game_datetime_utc or "", game.game_id,
        ))
        reasons = {9003: "POSTPONED", 9004: "CANCELLED"}
        expected_ledger = [self._expected_ledger_row(game, reasons.get(game.game_id))
                           for game in ordered]
        expected_features = [
            self._expected_feature_row(ordered[0], 10, 12),
            self._expected_feature_row(ordered[1], 10, 10),
        ]
        ledger_bytes = _csv_bytes(LEDGER_COLUMNS, expected_ledger)
        feature_bytes = _csv_bytes(FEATURE_COLUMNS, expected_features)
        self.assertEqual(publication.slot_path, self.slot)
        self.assertEqual(publication.candidate_ledger_path,
                         self.slot / CANDIDATE_LEDGER_FILENAME)
        self.assertEqual(publication.features_path, self.slot / FEATURES_FILENAME)
        self.assertEqual(publication.candidate_ledger_path.read_bytes(), ledger_bytes)
        self.assertEqual(publication.features_path.read_bytes(), feature_bytes)
        self.assertEqual(publication.candidate_ledger_sha256,
                         hashlib.sha256(ledger_bytes).hexdigest())
        self.assertEqual(publication.features_sha256,
                         hashlib.sha256(feature_bytes).hexdigest())
        self.assertEqual(publication.candidate_ledger_size_bytes, len(ledger_bytes))
        self.assertEqual(publication.features_size_bytes, len(feature_bytes))
        self.assertEqual(publication.candidate_row_count, 4)
        self.assertEqual(publication.feature_row_count, 2)
        self.assertEqual(publication.eligible_game_count, 2)
        self.assertEqual(publication.excluded_games_by_reason,
                         tuple((reason, int(reason in {"POSTPONED", "CANCELLED"}))
                               for reason in EXCLUSION_REASONS))
        self.assertEqual(publication.earliest_eligible_scheduled_start_utc,
                         "2026-09-03T18:00:00Z")
        for path, relative in (
            (publication.candidate_ledger_path,
             publication.candidate_ledger_relative_path),
            (publication.features_path, publication.features_relative_path),
        ):
            self.assertEqual(relative, path.relative_to(self.project).as_posix())
        for content in (ledger_bytes, feature_bytes):
            self.assertNotIn(b"\r", content)
            self.assertFalse(content.startswith(b"\xef\xbb\xbf"))
            self.assertTrue(content.endswith(b"\n"))
        self._assert_predecessors_unchanged(predecessors)
        self.assertEqual({path.name for path in self.slot.iterdir()}, {
            *predecessors, CANDIDATE_LEDGER_FILENAME, FEATURES_FILENAME,
        })

    def test_all_exclusion_reasons_and_exact_history_boundaries(self):
        self._set_histories({101: 10, 102: 10, 103: 9, 104: 9})
        overrides = (
            {},
            {"away_team_id": 103, "home_team_id": 104},
            {"away_team_id": 103, "home_team_id": 101},
            {"away_team_id": 101, "home_team_id": 104},
            {"away_team_id": 103, "home_team_id": 104,
             "game_datetime_utc": None},
            {"away_team_id": 103, "home_team_id": 104,
             "status_code": "DR", "status_detail": "Postponed",
             "abstract_state": "Final", "game_datetime_utc": None},
            {"away_team_id": 103, "home_team_id": 104,
             "status_code": "CI", "status_detail": "Cancelled",
             "abstract_state": "Final", "game_datetime_utc": None},
        )
        games = tuple(source_tests._target_game(game_id=9100 + index, **values)
                      for index, values in enumerate(overrides))
        self._prepare(games)
        publication = self._build()
        _, _, ledger = _read_csv(publication.candidate_ledger_path)
        reasons = {int(row["game_id"]): row["exclusion_reason"] for row in ledger}
        self.assertEqual(reasons, {
            9100: "", 9101: "INSUFFICIENT_BOTH_HISTORY",
            9102: "INSUFFICIENT_AWAY_HISTORY", 9103: "INSUFFICIENT_HOME_HISTORY",
            9104: "START_TIME_MISSING", 9105: "POSTPONED", 9106: "CANCELLED",
        })
        self.assertEqual(publication.eligible_game_count, 1)
        self.assertEqual(publication.excluded_games_by_reason,
                         tuple((reason, 1) for reason in EXCLUSION_REASONS))
        self.assertEqual(publication.candidate_row_count,
                         publication.eligible_game_count + sum(
                             count for _, count in publication.excluded_games_by_reason
                         ))

    def test_doubleheader_games_share_unchanged_j_minus_one_team_states(self):
        games = (
            source_tests._target_game(game_id=9002, doubleheader="Y", game_number=2,
                                      game_datetime_utc="2026-09-03T22:00:00Z"),
            source_tests._target_game(game_id=9001, doubleheader="Y", game_number=1),
        )
        self._prepare(games)
        publication = self._build()
        _, _, rows = _read_csv(publication.features_path)
        self.assertEqual([row["game_id"] for row in rows], ["9001", "9002"])
        self.assertEqual([rows[0][column] for column in FEATURE_VALUE_COLUMNS],
                         [rows[1][column] for column in FEATURE_VALUE_COLUMNS])
        self.assertEqual(rows[0]["away_games_before"], "10")
        self.assertEqual(rows[1]["away_games_before"], "10")
        for column in ("prediction_id", "occurrence_key", "feature_row_sha256"):
            self.assertNotEqual(rows[0][column], rows[1][column])
        self.assertNotIn("game_number", rows[0])
        self.assertNotIn("doubleheader", rows[0])

    def test_same_start_time_is_sorted_by_numeric_game_id(self):
        games = (source_tests._target_game(game_id=100),
                 source_tests._target_game(game_id=20))
        self._prepare(games)
        publication = self._build()
        for path in (publication.candidate_ledger_path, publication.features_path):
            _, _, rows = _read_csv(path)
            self.assertEqual([row["game_id"] for row in rows], ["20", "100"])

    def test_midnight_utc_does_not_change_official_date_or_as_of_date(self):
        self._prepare((source_tests._target_game(
            game_datetime_utc="2026-09-04T01:10:00Z"
        ),))
        _, _, rows = _read_csv(self._build().features_path)
        self.assertEqual(rows[0]["official_date"], TARGET_DATE)
        self.assertEqual(rows[0]["feature_as_of_date"], AS_OF_DATE)

    def test_max_source_dates_are_team_specific_not_as_of_substitutes(self):
        with closing(sqlite3.connect(self.fixture.database_path)) as connection, connection:
            connection.execute(
                "UPDATE games SET official_date = date(official_date, '-2 days') "
                "WHERE away_team_id = 101 OR home_team_id = 101"
            )
        self._prepare((source_tests._target_game(),))
        _, _, rows = _read_csv(self._build().features_path)
        self.assertEqual(rows[0]["away_max_source_date"],
                         (TARGET - timedelta(days=3)).isoformat())
        self.assertEqual(rows[0]["home_max_source_date"], AS_OF_DATE)
        self.assertEqual(rows[0]["feature_as_of_date"], AS_OF_DATE)

    def test_prior_season_and_target_day_results_cannot_supply_history(self):
        self._set_histories({101: 9, 102: 10})
        with closing(sqlite3.connect(self.fixture.database_path)) as connection, connection:
            for game_id, season, official_date in (
                (80001, 2025, "2025-09-02"), (80002, 2026, TARGET_DATE),
                (80003, 2026, (TARGET + timedelta(days=1)).isoformat()),
            ):
                connection.execute(
                    "INSERT INTO games (game_id, season, official_date, game_type, "
                    "status_code, status_detail, away_team_id, home_team_id, "
                    "away_score, home_score) VALUES (?, ?, ?, 'R', 'F', 'Final', "
                    "101, 102, 99, 0)",
                    (game_id, season, official_date),
                )
        self._prepare((source_tests._target_game(),))
        publication = self._build()
        _, _, ledger = _read_csv(publication.candidate_ledger_path)
        self.assertEqual(ledger[0]["exclusion_reason"], "INSUFFICIENT_AWAY_HISTORY")
        self.assertEqual(publication.feature_row_count, 0)

    def test_live_database_changes_after_snapshot_never_change_features(self):
        game = source_tests._target_game()
        self._prepare((game,))
        with closing(sqlite3.connect(self.fixture.database_path)) as connection, connection:
            connection.execute("UPDATE games SET away_score = 100, home_score = 0")
            connection.execute("DELETE FROM teams WHERE team_id = 101")
        forbidden = AssertionError("Lecture externe interdite pendant les variables")
        with (
            patch.object(shadow_prediction.sqlite3, "connect", side_effect=forbidden),
            patch.object(shadow_prediction.requests, "get", side_effect=forbidden),
            patch.object(shadow_prediction, "run_observed_schedule_ingestion",
                         side_effect=forbidden),
            patch.object(shadow_prediction, "verify_raw_archive", side_effect=forbidden),
        ):
            publication = self._build()
        self.assertEqual(publication.features_path.read_bytes(), _csv_bytes(
            FEATURE_COLUMNS, [self._expected_feature_row(game, 10, 12)]
        ))

    def test_io_guard_allows_only_five_slot_files_and_no_model_loading(self):
        self._prepare((source_tests._target_game(),))
        allowed_paths = {
            self.slot / "RESERVED",
            self.activation_publication.evidence_path,
            self.source_publication.snapshot_path,
            self.slot / CANDIDATE_LEDGER_FILENAME,
            self.slot / FEATURES_FILENAME,
        }
        reads = []
        original_read_bytes = Path.read_bytes
        original_import = builtins.__import__
        forbidden_roots = {"joblib", "pickle", "numpy", "pandas", "sklearn", "scipy"}

        def guarded_read_bytes(path):
            if path not in allowed_paths:
                raise AssertionError(f"Lecture hors du snapshot/slot interdite : {path}")
            reads.append(path)
            return original_read_bytes(path)

        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in forbidden_roots:
                raise AssertionError(f"Import modele/numerique interdit : {name}")
            return original_import(name, *args, **kwargs)

        forbidden = AssertionError("Chargement ou source externe interdite")
        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "read_bytes", guarded_read_bytes))
            stack.enter_context(patch.object(builtins, "__import__", guarded_import))
            stack.enter_context(patch.object(shadow_prediction.sqlite3, "connect",
                                             side_effect=forbidden))
            stack.enter_context(patch.object(shadow_prediction.requests, "get",
                                             side_effect=forbidden))
            stack.enter_context(patch.object(shadow_prediction,
                "run_observed_schedule_ingestion", side_effect=forbidden))
            stack.enter_context(patch.object(shadow_prediction, "verify_raw_archive",
                                             side_effect=forbidden))
            # Ne pas importer joblib pour le test. S'il est deja present
            # apres d'autres suites, ses chargeurs sont aussi neutralises.
            for module_name in ("pickle", "joblib"):
                module = sys.modules.get(module_name)
                if module is not None:
                    for loader_name in ("load", "loads"):
                        if hasattr(module, loader_name):
                            stack.enter_context(patch.object(
                                module, loader_name, side_effect=forbidden
                            ))
            publication = self._build()
        self.assertEqual(set(reads), allowed_paths)
        self.assertEqual(publication.feature_row_count, 1)
        self.assertFalse((self.slot / "predictions.csv").exists())

    def test_empty_schedule_publishes_two_header_only_csvs(self):
        self._prepare(())
        publication = self._build()
        self.assertEqual(publication.candidate_ledger_path.read_bytes(),
                         _csv_bytes(LEDGER_COLUMNS, []))
        self.assertEqual(publication.features_path.read_bytes(),
                         _csv_bytes(FEATURE_COLUMNS, []))
        self.assertEqual(publication.candidate_row_count, 0)
        self.assertEqual(publication.feature_row_count, 0)
        self.assertEqual(publication.eligible_game_count, 0)
        self.assertIsNone(publication.earliest_eligible_scheduled_start_utc)
        self.assertEqual(publication.excluded_games_by_reason,
                         tuple((reason, 0) for reason in EXCLUSION_REASONS))

    def test_zero_history_never_divides_or_builds_feature_values(self):
        self._set_histories({})
        self._prepare((source_tests._target_game(),))
        with patch.object(shadow_prediction, "_format_feature_rate",
                          side_effect=AssertionError("Division/formatage premature")):
            publication = self._build()
        self.assertEqual(publication.features_path.read_bytes(),
                         _csv_bytes(FEATURE_COLUMNS, []))
        _, _, ledger = _read_csv(publication.candidate_ledger_path)
        self.assertEqual(ledger[0]["exclusion_reason"], "INSUFFICIENT_BOTH_HISTORY")

    def test_status_normalization_only_classifies_and_preserves_raw_text(self):
        games = (
            source_tests._target_game(game_id=9001, status_code=" s ",
                status_detail=" scheduled ", abstract_state=" preview "),
            source_tests._target_game(game_id=9002, status_code=" p ",
                status_detail=" pre-game ", abstract_state=" preview "),
        )
        self._prepare(games)
        publication = self._build()
        _, _, ledger = _read_csv(publication.candidate_ledger_path)
        self.assertEqual([row["eligibility_status"] for row in ledger],
                         ["ELIGIBLE", "ELIGIBLE"])
        self.assertEqual(ledger[0]["status_code"], " s ")
        self.assertEqual(ledger[0]["detailed_state"], " scheduled ")
        self.assertEqual(ledger[1]["abstract_state"], " preview ")

    def test_exact_postponed_and_cancelled_tables_precede_final_abstract_state(self):
        games = tuple(source_tests._target_game(
            game_id=9200 + index, status_code=code, status_detail=detail,
            abstract_state="Final", game_datetime_utc=None,
        ) for index, (code, detail) in enumerate((
            ("D", "Postponed"), ("DI", "Postponed"), ("DR", "Postponed"),
            ("C", "Cancelled"), ("CI", "Cancelled"), ("CR", "Cancelled"),
        )))
        self._prepare(games)
        publication = self._build()
        _, _, ledger = _read_csv(publication.candidate_ledger_path)
        self.assertEqual([row["exclusion_reason"] for row in ledger],
                         ["POSTPONED"] * 3 + ["CANCELLED"] * 3)
        self.assertTrue(all(row["scheduled_start_utc"] == "" for row in ledger))
        self.assertEqual(publication.feature_row_count, 0)

    def test_unknown_live_final_and_conflicting_target_states_are_rejected(self):
        self._prepare((source_tests._target_game(),))
        for changes in (
            {"status_code": "I", "abstract_state": "Live", "detailed_state": "In Progress"},
            {"status_code": "F", "abstract_state": "Final", "detailed_state": "Final"},
            {"status_code": "S", "detailed_state": "Postponed"},
            {"status_code": "DR", "detailed_state": "Cancelled"},
            {"status_code": "P", "detailed_state": "Scheduled"},
            {"status_code": "X", "game_datetime_utc": None},
        ):
            with self.subTest(changes=changes):
                self._assert_payload_refused(
                    lambda payload, changes=changes:
                    payload["target_schedule"][0].update(changes)
                )

    def test_target_schema_cannot_carry_scores_outcomes_or_other_features(self):
        self._prepare((source_tests._target_game(),))
        for field in sorted(FORBIDDEN_FIELDS):
            with self.subTest(field=field):
                self._assert_payload_refused(
                    lambda payload, field=field:
                    payload["target_schedule"][0].update({field: 999})
                )

    def test_invalid_source_temporal_season_type_and_finality_are_rejected(self):
        self._prepare((source_tests._target_game(),))
        for changes in (
            {"official_date": TARGET_DATE},
            {"official_date": (TARGET + timedelta(days=1)).isoformat()},
            {"official_date": "2026-02-30"},
            {"season": 2025}, {"game_type": "P"},
            {"status_code": "S", "status_detail": "Scheduled"},
        ):
            with self.subTest(changes=changes):
                self._assert_payload_refused(
                    lambda payload, changes=changes:
                    payload["source_final_games"][0].update(changes)
                )

    def test_source_integer_domains_scores_identity_and_duplicates_are_exact(self):
        self._prepare((source_tests._target_game(),))
        for field in ("game_id", "season", "away_team_id", "home_team_id",
                      "away_score", "home_score"):
            for invalid in (True, 1.0, "1", None):
                with self.subTest(field=field, invalid=invalid):
                    self._assert_payload_refused(
                        lambda payload, field=field, invalid=invalid:
                        payload["source_final_games"][0].update({field: invalid})
                    )
        for changes in ({"away_score": -1}, {"home_score": -1},
                        {"away_score": 0, "home_score": 0},
                        {"away_team_id": 102, "home_team_id": 102},
                        {"away_team_id": 999999}):
            with self.subTest(changes=changes):
                self._assert_payload_refused(
                    lambda payload, changes=changes:
                    payload["source_final_games"][0].update(changes)
                )
        self._assert_payload_refused(lambda payload:
            payload["source_final_games"].append(deepcopy(payload["source_final_games"][0])))

    def test_completed_early_sources_contribute_normally(self):
        with closing(sqlite3.connect(self.fixture.database_path)) as connection, connection:
            connection.execute(
                "UPDATE games SET status_code = 'FR', status_detail = 'Completed Early'"
            )
        game = source_tests._target_game()
        self._prepare((game,))
        self.assertEqual(self._build().features_path.read_bytes(), _csv_bytes(
            FEATURE_COLUMNS, [self._expected_feature_row(game, 10, 12)]
        ))

    def test_nonfinite_feature_computation_is_rejected_before_publication(self):
        self._prepare((source_tests._target_game(),))
        self._assert_payload_refused(lambda payload:
            payload["source_final_games"][0].update({"away_score": 10 ** 400}))

    def test_source_and_target_order_uniqueness_and_team_references_are_exact(self):
        self._prepare()
        mutations = (
            lambda payload: payload["source_final_games"].reverse(),
            lambda payload: payload["target_schedule"].reverse(),
            lambda payload: payload["teams"].reverse(),
            lambda payload: payload["target_schedule"].append(
                deepcopy(payload["target_schedule"][0])),
            lambda payload: payload["teams"].append(deepcopy(payload["teams"][0])),
            lambda payload: payload["target_schedule"][0].update({"away_team_id": 999999}),
            lambda payload: payload["target_schedule"][0].update({"away_team_id": 102}),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(case=index):
                self._assert_payload_refused(mutation)

    def test_snapshot_and_nested_schemas_are_exact(self):
        self._prepare((source_tests._target_game(),))
        mutations = (
            lambda payload: payload.update({"unexpected": True}),
            lambda payload: payload.pop("teams"),
            lambda payload: payload["teams"][0].update({"odds": 2.0}),
            lambda payload: payload["source_final_games"][0].update({"venue_id": 5}),
            lambda payload: payload["schedule_ingestion"].update({"unknown": 0}),
            lambda payload: payload["sqlite_snapshot"].update({"unknown": 0}),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(case=index):
                self._assert_payload_refused(mutation)

    def test_snapshot_binding_and_information_cutoff_are_reverified(self):
        self._prepare((source_tests._target_game(),))
        mutations = (
            lambda payload: payload.update({"batch_id": "f" * 64}),
            lambda payload: payload.update({"target_official_date": "2026-09-04"}),
            lambda payload: payload.update({"information_cutoff_utc": "2026-09-03T09:00:00Z"}),
            lambda payload: payload["target_schedule"][0].update({
                "game_datetime_utc": source_tests.INFORMATION_CUTOFF_UTC}),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(case=index):
                self._assert_payload_refused(mutation)

    def test_forged_publication_hash_size_paths_or_metadata_are_rejected(self):
        self._prepare((source_tests._target_game(),))
        originals = self._predecessor_bytes()
        changes = (
            {"snapshot_sha256": "f" * 64},
            {"snapshot_size_bytes": self.source_publication.snapshot_size_bytes + 1},
            {"snapshot_path": self.project / "elsewhere.json.gz"},
            {"snapshot_relative_path": "../source_snapshot.json.gz"},
            {"slot_path": self.project},
            {"sqlite_snapshot_sha256": "f" * 64},
            {"schedule_ingestion_run_id": True},
            {"information_cutoff_utc": "2026-09-03T09:00:00Z"},
        )
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(ShadowPredictionError):
                    self._build(replace(self.source_publication, **change))
                self._assert_no_csv()
                self._assert_predecessors_unchanged(originals)

    def test_corrupted_or_noncanonical_source_bytes_are_rejected(self):
        self._prepare((source_tests._target_game(),))
        original = self.source_publication.snapshot_path.read_bytes()
        payload = self._payload()
        noncanonical_json = json.dumps(payload, indent=2).encode("utf-8")
        cases = (
            b"not a gzip archive",
            original + original,
            gzip.compress(_canonical_json(payload) + b"\n", mtime=123),
            gzip.compress(noncanonical_json, mtime=0),
        )
        for content in cases:
            with self.subTest(prefix=content[:20]):
                self.source_publication.snapshot_path.write_bytes(content)
                proof = replace(self.source_publication,
                    snapshot_sha256=hashlib.sha256(content).hexdigest(),
                    snapshot_size_bytes=len(content))
                try:
                    with self.assertRaises(ShadowPredictionError):
                        self._build(proof)
                    self._assert_no_csv()
                finally:
                    self.source_publication.snapshot_path.write_bytes(original)

    def test_duplicate_json_keys_are_rejected_even_with_matching_file_hash(self):
        self._prepare((source_tests._target_game(),))
        original = self.source_publication.snapshot_path.read_bytes()
        raw = gzip.decompress(original)
        raw = b'{"schema_version":1,' + raw[1:]
        stream = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream,
                           mtime=0, compresslevel=9) as compressed:
            compressed.write(raw)
        content = stream.getvalue()
        self.source_publication.snapshot_path.write_bytes(content)
        proof = replace(self.source_publication,
            snapshot_sha256=hashlib.sha256(content).hexdigest(),
            snapshot_size_bytes=len(content))
        with self.assertRaises(ShadowPredictionError):
            self._build(proof)
        self._assert_no_csv()

    def test_invalid_deflate_block_is_wrapped_as_shadow_prediction_error(self):
        self._prepare((source_tests._target_game(),))
        original = self.source_publication.snapshot_path.read_bytes()
        corrupted = bytearray(original)
        self.assertGreater(len(corrupted), 10)
        # Apres les dix octets d'en-tete gzip, BTYPE=3 est interdit par DEFLATE.
        corrupted[10] = 7
        content = bytes(corrupted)
        self.source_publication.snapshot_path.write_bytes(content)
        proof = replace(self.source_publication,
            snapshot_sha256=hashlib.sha256(content).hexdigest(),
            snapshot_size_bytes=len(content))
        with self.assertRaises(ShadowPredictionError):
            self._build(proof)
        self._assert_no_csv()

    def test_preexisting_fourth_fifth_or_terminal_path_never_gets_repaired(self):
        self._prepare((source_tests._target_game(),))
        predecessors = self._predecessor_bytes()
        for name in (CANDIDATE_LEDGER_FILENAME, FEATURES_FILENAME,
                     "predictions.csv", "FAILED.json", "COMPLETED", "unknown"):
            with self.subTest(name=name):
                path = self.slot / name
                path.write_bytes(b"must remain untouched\n")
                try:
                    with patch.object(shadow_prediction, "_build_candidate_feature_rows",
                        side_effect=AssertionError("Calcul interdit pour slot consomme")):
                        with self.assertRaises(ShadowPredictionSlotConsumedError):
                            self._build()
                    self.assertEqual(path.read_bytes(), b"must remain untouched\n")
                    self._assert_predecessors_unchanged(predecessors)
                finally:
                    path.unlink()

    def test_preexisting_output_directory_is_not_replaced(self):
        self._prepare((source_tests._target_game(),))
        destination = self.slot / CANDIDATE_LEDGER_FILENAME
        destination.mkdir()
        with self.assertRaises(ShadowPredictionSlotConsumedError):
            self._build()
        self.assertTrue(destination.is_dir())
        self.assertFalse((self.slot / FEATURES_FILENAME).exists())

    def test_source_symbolic_link_is_rejected_without_reading_its_target(self):
        self._prepare((source_tests._target_game(),))
        path = self.source_publication.snapshot_path
        original = path.read_bytes()
        external = self.project / "external-snapshot.json.gz"
        external.write_bytes(original)
        path.unlink()
        try:
            path.symlink_to(external)
        except (OSError, NotImplementedError):
            path.write_bytes(original)
            self.skipTest("Les liens symboliques ne sont pas disponibles ici.")
        with self.assertRaises(ShadowPredictionError):
            self._build()
        self.assertEqual(external.read_bytes(), original)
        self._assert_no_csv()

    def test_both_csvs_are_built_before_first_exclusive_publication(self):
        self._prepare((source_tests._target_game(),))
        with (
            patch.object(shadow_prediction, "_build_candidate_feature_rows",
                         side_effect=ShadowPredictionError("Variables invalides")),
            patch.object(shadow_prediction, "_publish_exclusive_verified") as publish,
        ):
            with self.assertRaises(ShadowPredictionError):
                self._build()
        publish.assert_not_called()
        self._assert_no_csv()

    def test_second_csv_serialization_failure_precedes_first_publication(self):
        self._prepare((source_tests._target_game(),))
        predecessors = self._predecessor_bytes()
        original_csv = shadow_prediction._canonical_csv_bytes
        serializations = []

        def fail_second(columns, rows):
            serializations.append(tuple(columns))
            if len(serializations) == 2:
                raise ShadowPredictionError("La seconde serialisation a echoue")
            return original_csv(columns, rows)

        with (
            patch.object(shadow_prediction, "_canonical_csv_bytes",
                         side_effect=fail_second),
            patch.object(shadow_prediction, "_publish_exclusive_verified") as publish,
        ):
            with self.assertRaises(ShadowPredictionError):
                self._build()
        self.assertEqual(serializations, [LEDGER_COLUMNS, FEATURE_COLUMNS])
        publish.assert_not_called()
        self._assert_no_csv()
        self.assertEqual(self._predecessor_bytes(), predecessors)

    def test_snapshot_mutation_during_calculation_blocks_both_publications(self):
        self._prepare((source_tests._target_game(),))
        original_builder = shadow_prediction._build_candidate_feature_rows
        original = self.source_publication.snapshot_path.read_bytes()

        def change_after_build(*args, **kwargs):
            result = original_builder(*args, **kwargs)
            self.source_publication.snapshot_path.write_bytes(original + b"changed")
            return result

        with patch.object(shadow_prediction, "_build_candidate_feature_rows",
                          side_effect=change_after_build):
            with self.assertRaises(ShadowPredictionError):
                self._build()
        self._assert_no_csv()

    def test_success_files_are_published_exclusively_in_exact_order(self):
        self._prepare((source_tests._target_game(),))
        original_publish = shadow_prediction._publish_exclusive_verified
        names = []

        def observe(destination, content):
            names.append(destination.name)
            self.assertFalse(destination.exists())
            if destination.name == FEATURES_FILENAME:
                self.assertTrue((self.slot / CANDIDATE_LEDGER_FILENAME).is_file())
            return original_publish(destination, content)

        with patch.object(shadow_prediction, "_publish_exclusive_verified",
                          side_effect=observe):
            self._build()
        self.assertEqual(names, [CANDIDATE_LEDGER_FILENAME, FEATURES_FILENAME])

    def _assert_predecessor_mutation_between_links_blocks_features(self, filename):
        self._prepare((source_tests._target_game(),))
        originals = self._predecessor_bytes()
        target = self.slot / filename
        original_publish = shadow_prediction._publish_exclusive_verified
        published_names = []

        def mutate_after_first_link(destination, content):
            result = original_publish(destination, content)
            published_names.append(destination.name)
            if destination.name == CANDIDATE_LEDGER_FILENAME:
                # Mutation en place, sans troncature ni extension : le verrou
                # Windows de RESERVED est situe juste apres son EOF initial.
                with target.open("r+b", buffering=0) as stream:
                    stream.seek(0)
                    stream.write(bytes([originals[filename][0] ^ 1]))
            return result

        with patch.object(shadow_prediction, "_publish_exclusive_verified",
                          side_effect=mutate_after_first_link):
            with self.assertRaises(ShadowPredictionError):
                self._build()
        self.assertEqual(published_names, [CANDIDATE_LEDGER_FILENAME])
        self.assertTrue((self.slot / CANDIDATE_LEDGER_FILENAME).is_file())
        self.assertFalse((self.slot / FEATURES_FILENAME).exists())
        self.assertNotEqual(target.read_bytes(), originals[filename])
        for name, content in originals.items():
            if name != filename:
                self.assertEqual((self.slot / name).read_bytes(), content)
        partial = self._predecessor_bytes()
        with self.assertRaises(ShadowPredictionSlotConsumedError):
            self._build()
        self.assertEqual(self._predecessor_bytes(), partial)

    def test_source_mutation_between_links_blocks_second_publication(self):
        self._assert_predecessor_mutation_between_links_blocks_features(
            shadow_prediction.SOURCE_SNAPSHOT_FILENAME
        )

    def test_activation_mutation_between_links_blocks_second_publication(self):
        self._assert_predecessor_mutation_between_links_blocks_features(
            shadow_prediction.ACTIVATION_REVERIFICATION_FILENAME
        )

    def test_reserved_mutation_between_links_blocks_second_publication(self):
        self._assert_predecessor_mutation_between_links_blocks_features("RESERVED")

    def test_failure_before_first_link_leaves_only_the_three_predecessors(self):
        self._prepare((source_tests._target_game(),))
        predecessors = self._predecessor_bytes()
        with patch.object(shadow_prediction, "_publish_exclusive_verified",
                          side_effect=ShadowPredictionError("Avant premier lien")):
            with self.assertRaises(ShadowPredictionError):
                self._build()
        self.assertEqual(self._predecessor_bytes(), predecessors)

    def test_failure_after_candidate_link_preserves_partial_slot_permanently(self):
        self._prepare((source_tests._target_game(),))
        original_publish = shadow_prediction._publish_exclusive_verified

        def fail_after_candidate(destination, content):
            result = original_publish(destination, content)
            if destination.name == CANDIDATE_LEDGER_FILENAME:
                raise ShadowPredictionError("Apres lien du registre")
            return result

        with patch.object(shadow_prediction, "_publish_exclusive_verified",
                          side_effect=fail_after_candidate):
            with self.assertRaises(ShadowPredictionError):
                self._build()
        self.assertTrue((self.slot / CANDIDATE_LEDGER_FILENAME).is_file())
        self.assertFalse((self.slot / FEATURES_FILENAME).exists())
        partial = self._predecessor_bytes()
        with self.assertRaises(ShadowPredictionSlotConsumedError):
            self._build()
        self.assertEqual(self._predecessor_bytes(), partial)

    def test_failure_before_features_link_cannot_repair_the_published_ledger(self):
        self._prepare((source_tests._target_game(),))
        original_publish = shadow_prediction._publish_exclusive_verified

        def fail_second(destination, content):
            if destination.name == FEATURES_FILENAME:
                raise ShadowPredictionError("Avant lien des variables")
            return original_publish(destination, content)

        with patch.object(shadow_prediction, "_publish_exclusive_verified",
                          side_effect=fail_second):
            with self.assertRaises(ShadowPredictionError):
                self._build()
        partial = self._predecessor_bytes()
        self.assertIn(CANDIDATE_LEDGER_FILENAME, partial)
        self.assertNotIn(FEATURES_FILENAME, partial)
        with self.assertRaises(ShadowPredictionSlotConsumedError):
            self._build()
        self.assertEqual(self._predecessor_bytes(), partial)

    def test_failure_after_features_link_preserves_both_files_without_retry(self):
        self._prepare((source_tests._target_game(),))
        original_publish = shadow_prediction._publish_exclusive_verified

        def fail_after_second(destination, content):
            result = original_publish(destination, content)
            if destination.name == FEATURES_FILENAME:
                raise ShadowPredictionError("Apres lien des variables")
            return result

        with patch.object(shadow_prediction, "_publish_exclusive_verified",
                          side_effect=fail_after_second):
            with self.assertRaises(ShadowPredictionError):
                self._build()
        preserved = self._predecessor_bytes()
        self.assertIn(CANDIDATE_LEDGER_FILENAME, preserved)
        self.assertIn(FEATURES_FILENAME, preserved)
        with self.assertRaises(ShadowPredictionSlotConsumedError):
            self._build()
        self.assertEqual(self._predecessor_bytes(), preserved)

    def test_second_success_call_never_overwrites_or_reuses_any_output(self):
        self._prepare((source_tests._target_game(),))
        self._build()
        original = self._predecessor_bytes()
        with self.assertRaises(ShadowPredictionSlotConsumedError):
            self._build()
        self.assertEqual(self._predecessor_bytes(), original)

    def test_concurrent_threads_have_exactly_one_builder_and_one_winner(self):
        self._prepare((source_tests._target_game(),))
        entered, release = Event(), Event()
        original_builder = shadow_prediction._build_candidate_feature_rows

        def hold_builder(*args, **kwargs):
            entered.set()
            if not release.wait(15):
                raise AssertionError("Le test concurrent n'a pas libere le gagnant")
            return original_builder(*args, **kwargs)

        with patch.object(shadow_prediction, "_build_candidate_feature_rows",
                          side_effect=hold_builder) as builder:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(self._build)
                try:
                    self.assertTrue(entered.wait(15))
                    second = pool.submit(self._build)
                    with self.assertRaises(ShadowPredictionSlotConsumedError):
                        second.result(timeout=15)
                finally:
                    release.set()
                publication = first.result(timeout=15)
            self.assertEqual(builder.call_count, 1)
        self.assertEqual(publication.feature_row_count, 1)
        self.assertEqual(len(tuple(self.slot.iterdir())), 5)

    def test_concurrent_process_cannot_build_or_modify_the_owned_slot(self):
        self._prepare((source_tests._target_game(),))
        entered, release = Event(), Event()
        original_builder = shadow_prediction._build_candidate_feature_rows

        def hold_builder(*args, **kwargs):
            entered.set()
            if not release.wait(25):
                raise AssertionError("Le processus concurrent n'a pas termine")
            return original_builder(*args, **kwargs)

        context = get_context("spawn")
        results = context.Queue()
        child = context.Process(target=_attempt_candidate_features_in_child, args=(
            self.reservation, self.activation_publication, self.source_publication,
            self.project, results,
        ))
        with patch.object(shadow_prediction, "_build_candidate_feature_rows",
                          side_effect=hold_builder):
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(self._build)
                try:
                    self.assertTrue(entered.wait(15))
                    before = self._predecessor_bytes()
                    child.start()
                    outcome = results.get(timeout=20)
                    child.join(timeout=10)
                    self.assertEqual(outcome, ("consumed", None))
                    self.assertEqual(child.exitcode, 0)
                    self.assertEqual(self._predecessor_bytes(), before)
                finally:
                    release.set()
                    if child.pid is not None and child.is_alive():
                        child.terminate()
                        child.join(timeout=5)
                first.result(timeout=15)
        results.close()
        results.join_thread()
        self.assertEqual(len(tuple(self.slot.iterdir())), 5)

    def test_public_api_has_no_model_source_override_or_new_execution_switch(self):
        signature = inspect.signature(build_and_publish_candidate_ledger_and_features)
        self.assertEqual(tuple(signature.parameters), (
            "reservation", "activation_publication", "source_publication",
            "project_directory",
        ))
        self.assertEqual(signature.parameters["project_directory"].kind,
                         inspect.Parameter.KEYWORD_ONLY)
        self._prepare((source_tests._target_game(),))
        self._build()
        self.assertFalse(FORBIDDEN_FIELDS.intersection(LEDGER_COLUMNS))
        self.assertFalse(FORBIDDEN_FIELDS.intersection(FEATURE_COLUMNS))
        for forbidden in ("predictions.csv", "receipt.json", "COMPLETED", "FAILED.json"):
            self.assertFalse((self.slot / forbidden).exists())
        self.assertFalse((self.project / "models").exists())


if __name__ == "__main__":
    unittest.main()
