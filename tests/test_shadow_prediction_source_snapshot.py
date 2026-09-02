"""Tests du snapshot source prospectif du mode fantome MLB v2."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timezone
import gzip
import hashlib
import inspect
import json
from multiprocessing import get_context
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

import src.shadow_prediction as shadow_prediction
from src.database import get_connection, initialize_database
from src.ingestion_service import (
    INGESTION_SOURCE,
    ScheduleIngestionResult,
    build_schedule_request_parameters,
)
from src.mlb_api import MLBAPIError, MLBAPIRetryableError, ScheduledGame
from src.shadow_prediction import (
    ACTIVATION_REVERIFICATION_FILENAME,
    EXPECTED_MODEL_ARTIFACT_SHA256,
    EXPECTED_SHADOW_PROTOCOL_SHA256,
    GITHUB_COMPARE_URL_TEMPLATE,
    SHADOW_RESULT_ROOT_RELATIVE_PATH,
    SOURCE_SNAPSHOT_FILENAME,
    ShadowPredictionError,
    ShadowPredictionSlotConsumedError,
    capture_and_publish_source_snapshot,
    fetch_activation_reverification_evidence,
    publish_activation_reverification_evidence,
    reserve_shadow_prediction_slot,
)


TARGET_DATE = "2026-09-03"
TARGET = date.fromisoformat(TARGET_DATE)
PROTOCOL_SHA256 = EXPECTED_SHADOW_PROTOCOL_SHA256
EXECUTION_MANIFEST_SHA256 = "2" * 64
MODEL_ARTIFACT_SHA256 = EXPECTED_MODEL_ARTIFACT_SHA256
RUNTIME_CODE_COMMIT = "4" * 40
ACTIVATION_COMMIT = "5" * 40
ACTIVATION_DATE_RAW = "Wed, 02 Sep 2026 08:00:00 GMT"
RESERVED_AT_UTC = "2026-09-03T09:59:00Z"
MLB_DATE_RAW = "Thu, 03 Sep 2026 10:00:01 GMT"
MLB_DATE_UTC = "2026-09-03T10:00:01Z"
MLB_RECEIVED_AT_UTC = "2026-09-03T10:00:02Z"
INFORMATION_CUTOFF_UTC = "2026-09-03T10:00:03Z"

SOURCE_SNAPSHOT_KEYS = {
    "schema_version",
    "batch_id",
    "target_official_date",
    "created_at_utc",
    "information_cutoff_utc",
    "schedule_ingestion",
    "sqlite_snapshot",
    "teams",
    "target_schedule",
    "source_final_games",
}
SCHEDULE_INGESTION_KEYS = {
    "run_id",
    "source",
    "requested_start_date",
    "requested_end_date",
    "game_types",
    "request_parameters_json",
    "completed_at_utc",
    "raw_archive_path",
    "raw_archive_sha256",
    "response_effective_url",
    "response_status_code",
    "response_redirect_count",
    "mlb_http_date_header_raw",
    "mlb_http_date_utc",
    "mlb_http_response_received_at_utc",
    "response_body_sha256",
}
SQLITE_SNAPSHOT_KEYS = {
    "source_database_path",
    "sha256",
    "size_bytes",
    "foreign_key_violation_count",
    "active_ingestion_count",
}
TEAM_KEYS = {"team_id", "name", "abbreviation"}
TARGET_SCHEDULE_KEYS = {
    "game_id",
    "season",
    "official_date",
    "game_datetime_utc",
    "game_type",
    "status_code",
    "abstract_state",
    "detailed_state",
    "away_team_id",
    "home_team_id",
    "doubleheader",
    "game_number",
}
SOURCE_FINAL_GAME_KEYS = {
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


def _compare_body() -> bytes:
    return json.dumps(
        {
            "base_commit": {"sha": ACTIVATION_COMMIT},
            "merge_base_commit": {"sha": ACTIVATION_COMMIT},
            "status": "ahead",
            "ahead_by": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class _FakeGitHubResponse:
    status_code = 200
    history: list[object] = []
    url = GITHUB_COMPARE_URL_TEMPLATE.format(
        expected_commit=ACTIVATION_COMMIT
    )
    content = _compare_body()
    headers = {
        "date": ACTIVATION_DATE_RAW,
        "content-type": "application/json; charset=utf-8",
        "etag": 'W/"source-snapshot-test"',
        "x-github-request-id": "SOURCE:SNAPSHOT:TEST",
    }


def _target_game(**overrides: object) -> ScheduledGame:
    values: dict[str, object] = {
        "game_id": 9001,
        "season": 2026,
        "official_date": TARGET_DATE,
        "game_datetime_utc": "2026-09-03T18:00:00Z",
        "game_type": "R",
        "status_code": "S",
        "status_detail": "Scheduled",
        "away_team_id": 101,
        "away_team_name": "Alpha",
        "home_team_id": 102,
        "home_team_name": "Bravo",
        "away_score": None,
        "home_score": None,
        "venue_id": 77,
        "venue_name": "Stade interdit dans le snapshot cible",
        "doubleheader": "N",
        "game_number": 1,
        "away_probable_pitcher_id": 8001,
        "away_probable_pitcher_name": "Lanceur exterieur interdit",
        "home_probable_pitcher_id": 8002,
        "home_probable_pitcher_name": "Lanceur domicile interdit",
        "abstract_state": "Preview",
    }
    values.update(overrides)
    return ScheduledGame(**values)  # type: ignore[arg-type]


def _attempt_source_stage_lock_in_child(
    reservation,
    project_directory: Path,
    results,
) -> None:
    """Essaie le verrou depuis un vrai second processus."""
    try:
        with shadow_prediction._hold_source_snapshot_stage_lock(
            reservation,
            project_directory=project_directory,
        ):
            results.put("acquired")
    except ShadowPredictionSlotConsumedError:
        results.put("consumed")


class ShadowSourceSnapshotTests(unittest.TestCase):
    """Le troisieme fichier doit figer uniquement les sources permises."""

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project = Path(self.temporary_directory.name)
        self.data_directory = self.project / "data"
        self.database_path = self.data_directory / "fredo_mlb.db"
        initialize_database(self.database_path)
        self._insert_seed_rows()
        self.next_run_id = 100

        result_root = self.project.joinpath(
            *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts
        )
        result_root.mkdir(parents=True)
        with (
            patch.object(
                shadow_prediction.requests,
                "get",
                return_value=_FakeGitHubResponse(),
            ),
            patch.object(
                shadow_prediction,
                "_utc_now",
                return_value=datetime(
                    2026,
                    9,
                    2,
                    8,
                    0,
                    2,
                    tzinfo=timezone.utc,
                ),
            ),
        ):
            evidence = fetch_activation_reverification_evidence(
                ACTIVATION_COMMIT
            )
        self.reservation = reserve_shadow_prediction_slot(
            TARGET_DATE,
            reserved_at_utc=RESERVED_AT_UTC,
            runtime_code_commit=RUNTIME_CODE_COMMIT,
            shadow_protocol_sha256=PROTOCOL_SHA256,
            execution_manifest_sha256=EXECUTION_MANIFEST_SHA256,
            model_artifact_sha256=MODEL_ARTIFACT_SHA256,
            project_directory=self.project,
        )
        self.activation_publication = (
            publish_activation_reverification_evidence(
                self.reservation,
                evidence,
                project_directory=self.project,
            )
        )
        self.slot = self.reservation.slot_path

    def _insert_seed_rows(self) -> None:
        teams = (
            (104, "Delta", "DEL"),
            (101, "Alpha", "ALP"),
            (202, "Foxtrot", None),
            (103, "Charlie", "CHA"),
            (102, "Bravo", "BRA"),
            (201, "Echo", "ECH"),
        )
        games = (
            # Deux matchs finaux 2026 strictement anterieurs et admissibles.
            (7002, 2026, "2026-09-02", "R", "FR", "Completed Early", 103, 104, 4, 2),
            (7001, 2026, "2026-08-30", "R", "F", "Final", 101, 102, 1, 3),
            # Tous les suivants doivent etre absents de source_final_games.
            (7003, 2026, TARGET_DATE, "R", "F", "Final", 101, 103, 6, 2),
            (7004, 2026, "2026-09-04", "R", "F", "Final", 102, 104, 2, 1),
            (7005, 2026, "2026-09-01", "R", "S", "Scheduled", 101, 104, None, None),
            (7006, 2025, "2025-09-01", "R", "F", "Final", 101, 102, 5, 1),
            (7007, 2026, "2026-09-01", "P", "F", "Final", 103, 104, 2, 0),
        )
        with get_connection(self.database_path) as connection:
            connection.executemany(
                """
                INSERT INTO teams (team_id, name, abbreviation)
                VALUES (?, ?, ?)
                """,
                teams,
            )
            connection.executemany(
                """
                INSERT INTO games (
                    game_id, season, official_date, game_type,
                    status_code, status_detail, away_team_id,
                    home_team_id, away_score, home_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                games,
            )

    def _default_games(self) -> tuple[ScheduledGame, ...]:
        return (
            _target_game(
                game_id=9004,
                game_datetime_utc=None,
                status_code="CI",
                status_detail="Cancelled",
                abstract_state="Final",
                away_team_id=201,
                home_team_id=202,
            ),
            _target_game(
                game_id=9002,
                game_datetime_utc="2026-09-03T19:00:00Z",
                status_code="P",
                status_detail="Pre-Game",
                away_score=0,
                home_score=0,
                away_team_id=103,
                home_team_id=104,
            ),
            _target_game(
                game_id=9003,
                game_datetime_utc="2026-09-03T20:00:00Z",
                status_code="DR",
                status_detail="Postponed",
                abstract_state="Final",
                away_team_id=102,
                home_team_id=201,
            ),
            _target_game(),
        )

    def _request_parameters_json(self) -> str:
        parameters = build_schedule_request_parameters(
            start_date=TARGET,
            end_date=TARGET,
            game_types=("R",),
        )
        return json.dumps(
            parameters,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _insert_error_run(self, message: str) -> int:
        run_id = self.next_run_id
        self.next_run_id += 1
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO ingestion_runs (
                    run_id, source, requested_start_date,
                    requested_end_date, game_types,
                    request_parameters_json, started_at_utc,
                    completed_at_utc, status, records_received,
                    records_saved, raw_response_path, response_sha256,
                    code_version, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    INGESTION_SOURCE,
                    TARGET_DATE,
                    TARGET_DATE,
                    "R",
                    self._request_parameters_json(),
                    "2026-09-03T10:00:00+00:00",
                    "2026-09-03T10:00:01+00:00",
                    "error",
                    0,
                    0,
                    None,
                    None,
                    RUNTIME_CODE_COMMIT,
                    message,
                ),
            )
        return run_id

    def _new_success_result(
        self,
        *,
        games: tuple[ScheduledGame, ...] | None = None,
    ) -> ScheduleIngestionResult:
        selected_games = self._default_games() if games is None else games
        run_id = self.next_run_id
        self.next_run_id += 1
        raw_body = (
            b'{"dates":[],"source_snapshot_test_run":'
            + str(run_id).encode("ascii")
            + b"}\n"
        )
        response_sha256 = hashlib.sha256(raw_body).hexdigest()
        relative_path = (
            "data/raw/mlb_schedule/2026/"
            f"source-snapshot-test-{run_id}.json.gz"
        )
        archive_path = self.project.joinpath(*relative_path.split("/"))
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(archive_path, "wb") as archive:
            archive.write(raw_body)

        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO ingestion_runs (
                    run_id, source, requested_start_date,
                    requested_end_date, game_types,
                    request_parameters_json, started_at_utc,
                    completed_at_utc, status, records_received,
                    records_saved, raw_response_path, response_sha256,
                    code_version, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    INGESTION_SOURCE,
                    TARGET_DATE,
                    TARGET_DATE,
                    "R",
                    self._request_parameters_json(),
                    "2026-09-03T10:00:00+00:00",
                    "2026-09-03T10:00:02+00:00",
                    "success",
                    len(selected_games),
                    len(selected_games),
                    relative_path,
                    response_sha256,
                    RUNTIME_CODE_COMMIT,
                    None,
                ),
            )

        effective_url = (
            "https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&startDate={TARGET_DATE}&endDate={TARGET_DATE}"
            "&gameTypes=R&hydrate=probablePitcher"
        )
        return ScheduleIngestionResult(
            run_id=run_id,
            start_date=TARGET,
            end_date=TARGET,
            games_received=len(selected_games),
            games_saved=len(selected_games),
            archive_relative_path=relative_path,
            response_sha256=response_sha256,
            code_version=RUNTIME_CODE_COMMIT,
            games=selected_games,
            response_effective_url=effective_url,
            response_status_code=200,
            response_redirect_count=0,
            mlb_http_date_header_raw=MLB_DATE_RAW,
            mlb_http_date_utc=MLB_DATE_UTC,
            mlb_http_response_received_at_utc=MLB_RECEIVED_AT_UTC,
            response_body_sha256=response_sha256,
        )

    @staticmethod
    def _as_datetime(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def _capture(
        self,
        result: ScheduleIngestionResult,
        *,
        cutoff: str = INFORMATION_CUTOFF_UTC,
        sleeper=None,
    ):
        effective_sleeper = (
            sleeper
            if sleeper is not None
            else lambda seconds: self.fail(
                f"Pause reseau inattendue : {seconds}"
            )
        )
        with (
            patch.object(
                shadow_prediction,
                "run_observed_schedule_ingestion",
                return_value=result,
            ) as ingestion_mock,
            patch.object(
                shadow_prediction,
                "_utc_now",
                return_value=self._as_datetime(cutoff),
            ),
            patch.object(
                shadow_prediction.time,
                "sleep",
                side_effect=effective_sleeper,
            ),
        ):
            publication = capture_and_publish_source_snapshot(
                self.reservation,
                self.activation_publication,
                database_path=self.database_path,
                data_directory=self.data_directory,
                project_directory=self.project,
            )
        return publication, ingestion_mock

    def _read_snapshot(self) -> tuple[bytes, bytes, dict[str, object]]:
        compressed = (self.slot / SOURCE_SNAPSHOT_FILENAME).read_bytes()
        canonical_json = gzip.decompress(compressed)
        payload = json.loads(canonical_json.decode("utf-8"))
        self.assertIs(type(payload), dict)
        return compressed, canonical_json, payload

    def _assert_rejected(
        self,
        result: ScheduleIngestionResult,
        *,
        cutoff: str = INFORMATION_CUTOFF_UTC,
    ) -> None:
        with (
            patch.object(
                shadow_prediction,
                "run_observed_schedule_ingestion",
                return_value=result,
            ),
            patch.object(
                shadow_prediction,
                "_utc_now",
                return_value=self._as_datetime(cutoff),
            ),
        ):
            with self.assertRaises(ShadowPredictionError):
                capture_and_publish_source_snapshot(
                    self.reservation,
                    self.activation_publication,
                    database_path=self.database_path,
                    data_directory=self.data_directory,
                    project_directory=self.project,
                )
        self.assertFalse((self.slot / SOURCE_SNAPSHOT_FILENAME).exists())

    def test_success_has_exact_schema_redaction_order_and_hashes(self) -> None:
        """Le snapshot reussi doit etre canonique, minimal et verifiable."""
        result = self._new_success_result()

        publication, ingestion_mock = self._capture(result)
        compressed, canonical_json, payload = self._read_snapshot()

        ingestion_mock.assert_called_once_with(
            start_date=TARGET,
            end_date=TARGET,
            game_types=("R",),
            database_path=self.database_path,
            data_directory=self.data_directory,
            code_version=RUNTIME_CODE_COMMIT,
        )
        self.assertEqual(set(payload), SOURCE_SNAPSHOT_KEYS)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["batch_id"], self.reservation.batch_id)
        self.assertEqual(payload["target_official_date"], TARGET_DATE)
        self.assertEqual(payload["created_at_utc"], INFORMATION_CUTOFF_UTC)
        self.assertEqual(
            payload["information_cutoff_utc"],
            INFORMATION_CUTOFF_UTC,
        )

        schedule_ingestion = payload["schedule_ingestion"]
        sqlite_snapshot = payload["sqlite_snapshot"]
        teams = payload["teams"]
        target_schedule = payload["target_schedule"]
        source_games = payload["source_final_games"]
        self.assertEqual(set(schedule_ingestion), SCHEDULE_INGESTION_KEYS)
        self.assertEqual(set(sqlite_snapshot), SQLITE_SNAPSHOT_KEYS)
        self.assertTrue(all(set(row) == TEAM_KEYS for row in teams))
        self.assertTrue(
            all(set(row) == TARGET_SCHEDULE_KEYS for row in target_schedule)
        )
        self.assertTrue(
            all(set(row) == SOURCE_FINAL_GAME_KEYS for row in source_games)
        )
        self.assertEqual(
            [row["team_id"] for row in teams],
            sorted(row["team_id"] for row in teams),
        )
        self.assertEqual(
            [row["game_id"] for row in target_schedule],
            [9001, 9002, 9003, 9004],
        )
        self.assertEqual(
            [(row["official_date"], row["game_id"]) for row in source_games],
            [("2026-08-30", 7001), ("2026-09-02", 7002)],
        )

        forbidden_target_fields = {
            "away_score",
            "home_score",
            "venue_id",
            "venue_name",
            "away_probable_pitcher_id",
            "away_probable_pitcher_name",
            "home_probable_pitcher_id",
            "home_probable_pitcher_name",
            "odds",
        }
        self.assertTrue(
            all(
                not forbidden_target_fields.intersection(row)
                for row in target_schedule
            )
        )
        self.assertEqual(
            [(row["away_score"], row["home_score"]) for row in source_games],
            [(1, 3), (4, 2)],
        )

        self.assertEqual(
            schedule_ingestion["response_body_sha256"],
            schedule_ingestion["raw_archive_sha256"],
        )
        self.assertEqual(schedule_ingestion["run_id"], result.run_id)
        self.assertEqual(
            sqlite_snapshot["source_database_path"],
            "data/fredo_mlb.db",
        )
        self.assertEqual(sqlite_snapshot["active_ingestion_count"], 0)
        self.assertEqual(sqlite_snapshot["foreign_key_violation_count"], 0)
        self.assertEqual(
            sqlite_snapshot["sha256"],
            publication.sqlite_snapshot_sha256,
        )
        self.assertEqual(
            sqlite_snapshot["size_bytes"],
            publication.sqlite_snapshot_size_bytes,
        )
        self.assertGreater(sqlite_snapshot["size_bytes"], 0)

        self.assertEqual(
            canonical_json,
            shadow_prediction._canonical_json_file_bytes(payload),
        )
        self.assertEqual(
            compressed,
            shadow_prediction._canonical_gzip_bytes(canonical_json),
        )
        self.assertEqual(
            hashlib.sha256(compressed).hexdigest(),
            publication.snapshot_sha256,
        )
        self.assertEqual(publication.snapshot_size_bytes, len(compressed))
        self.assertEqual(publication.schedule_attempts, 1)
        self.assertEqual(
            publication.snapshot_relative_path,
            (
                SHADOW_RESULT_ROOT_RELATIVE_PATH
                / TARGET_DATE
                / SOURCE_SNAPSHOT_FILENAME
            ).as_posix(),
        )
        self.assertEqual(
            sorted(path.name for path in self.slot.iterdir()),
            sorted(
                [
                    "RESERVED",
                    ACTIVATION_REVERIFICATION_FILENAME,
                    SOURCE_SNAPSHOT_FILENAME,
                ]
            ),
        )

    def test_successful_first_attempt_has_zero_seconds_before_it(self) -> None:
        """Le premier appel part immediatement, sans sleep prealable."""
        result = self._new_success_result(games=())
        sleeps: list[float] = []

        publication, ingestion_mock = self._capture(
            result,
            sleeper=sleeps.append,
        )

        self.assertEqual(ingestion_mock.call_count, 1)
        self.assertEqual(sleeps, [])
        self.assertEqual(publication.schedule_attempts, 1)

    def test_only_strict_j_minus_one_final_regular_games_are_sources(self) -> None:
        """Jour cible, futur, autre saison, non final et non R sont exclus."""
        result = self._new_success_result(games=())
        self._capture(result)
        _, _, payload = self._read_snapshot()

        source_games = payload["source_final_games"]
        self.assertEqual([row["game_id"] for row in source_games], [7001, 7002])
        self.assertTrue(
            all(row["season"] == 2026 for row in source_games)
        )
        self.assertTrue(
            all(row["game_type"] == "R" for row in source_games)
        )
        self.assertTrue(
            all(row["official_date"] < TARGET_DATE for row in source_games)
        )

    def test_allowed_target_statuses_and_missing_time_are_preserved(self) -> None:
        """Scheduled, Pre-Game, report et annulation restent observables."""
        result = self._new_success_result()
        self._capture(result)
        _, _, payload = self._read_snapshot()

        by_id = {row["game_id"]: row for row in payload["target_schedule"]}
        self.assertEqual(by_id[9001]["detailed_state"], "Scheduled")
        self.assertEqual(by_id[9002]["detailed_state"], "Pre-Game")
        self.assertEqual(by_id[9003]["detailed_state"], "Postponed")
        self.assertEqual(by_id[9004]["detailed_state"], "Cancelled")
        self.assertIsNone(by_id[9004]["game_datetime_utc"])

    def test_live_final_and_unknown_target_statuses_are_rejected(self) -> None:
        """Un statut commence, termine ou inconnu ferme la preparation."""
        cases = (
            {"status_code": "I", "status_detail": "In Progress", "abstract_state": "Live"},
            {"status_code": "F", "status_detail": "Final", "abstract_state": "Final"},
            {"status_code": "X", "status_detail": "Mystery", "abstract_state": "Preview"},
        )
        for values in cases:
            with self.subTest(values=values):
                result = self._new_success_result(
                    games=(_target_game(**values),)
                )
                self._assert_rejected(result)

    def test_target_identity_date_type_and_uniqueness_are_exact(self) -> None:
        """Le calendrier cible reste 2026, du jour, R et sans doublon."""
        cases = (
            (_target_game(season=2025),),
            (_target_game(official_date="2026-09-02"),),
            (_target_game(game_type="P"),),
            (_target_game(home_team_id=101),),
            (_target_game(), _target_game()),
        )
        for games in cases:
            with self.subTest(games=games):
                result = self._new_success_result(games=games)
                self._assert_rejected(result)

    def test_target_scores_must_be_both_null_or_both_zero(self) -> None:
        """Aucun score renseigne ou unilateral ne peut entrer au snapshot."""
        cases = ((0, None), (None, 0), (1, 0), (0, 2), (-1, 0))
        for away_score, home_score in cases:
            with self.subTest(scores=(away_score, home_score)):
                result = self._new_success_result(
                    games=(
                        _target_game(
                            away_score=away_score,
                            home_score=home_score,
                        ),
                    )
                )
                self._assert_rejected(result)

    def test_every_known_target_start_must_be_after_cutoff(self) -> None:
        """Un horaire egal ou anterieur au cutoff invalide tout le lot."""
        for start in (
            "2026-09-03T10:00:02Z",
            INFORMATION_CUTOFF_UTC,
        ):
            with self.subTest(start=start):
                result = self._new_success_result(
                    games=(_target_game(game_datetime_utc=start),)
                )
                self._assert_rejected(result)

    def test_schedule_age_range_is_inclusive_from_zero_to_900(self) -> None:
        """Une observation vieille de 901 s echoue; 900 s est acceptee."""
        rejected = self._new_success_result(games=())
        self._assert_rejected(rejected, cutoff="2026-09-03T10:15:03Z")

        accepted = self._new_success_result(games=())
        publication, _ = self._capture(
            accepted,
            cutoff="2026-09-03T10:15:02Z",
        )
        self.assertEqual(
            publication.information_cutoff_utc,
            "2026-09-03T10:15:02Z",
        )

    def test_schedule_age_zero_seconds_is_accepted(self) -> None:
        """La borne basse exacte autorise observation et cutoff identiques."""
        result = self._new_success_result(games=())

        publication, _ = self._capture(
            result,
            cutoff=MLB_RECEIVED_AT_UTC,
        )

        self.assertEqual(
            publication.information_cutoff_utc,
            MLB_RECEIVED_AT_UTC,
        )

    def test_schedule_observation_must_follow_reservation(self) -> None:
        """Une ancienne reponse MLB ne peut pas servir au slot reserve."""
        result = replace(
            self._new_success_result(),
            mlb_http_response_received_at_utc="2026-09-03T09:58:59Z",
        )
        self._assert_rejected(result)

    def test_ingestion_completion_must_follow_http_observation(self) -> None:
        """Le journal ne peut pas finir avant la reponse HTTP qu'il prouve."""
        result = self._new_success_result(games=())
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                UPDATE ingestion_runs
                SET completed_at_utc = ?
                WHERE run_id = ?
                """,
                ("2026-09-03T10:00:01+00:00", result.run_id),
            )

        self._assert_rejected(result)

    def test_http_envelope_and_query_must_match_exact_contract(self) -> None:
        """HTTPS, hote, route, query, 200 et zero redirection sont figes."""
        base = self._new_success_result()
        cases = (
            replace(base, response_status_code=503),
            replace(base, response_status_code=True),
            replace(base, response_redirect_count=1),
            replace(base, response_effective_url=base.response_effective_url.replace("https://", "http://")),
            replace(base, response_effective_url=base.response_effective_url.replace("statsapi.mlb.com", "example.com")),
            replace(base, response_effective_url=base.response_effective_url + "&sportId=1"),
            replace(base, response_effective_url=base.response_effective_url.replace("gameTypes=R", "gameTypes=P")),
        )
        for result in cases:
            with self.subTest(result=result):
                self._assert_rejected(result)

    def test_http_date_hash_and_clock_provenance_are_exact(self) -> None:
        """Date, empreinte du corps et ecart d'horloge sont controles."""
        base = self._new_success_result()
        cases = (
            replace(base, response_body_sha256="a" * 64),
            replace(base, mlb_http_date_header_raw="2026-09-03T10:00:01Z"),
            replace(base, mlb_http_date_utc="2026-09-03T10:00:02Z"),
            replace(
                base,
                mlb_http_response_received_at_utc=(
                    "2026-09-03T10:05:02Z"
                ),
            ),
        )
        for result in cases:
            with self.subTest(result=result):
                self._assert_rejected(result, cutoff="2026-09-03T10:05:03Z")

    def test_ingestion_result_and_sqlite_journal_must_match(self) -> None:
        """Le resultat en memoire ne peut pas diverger du journal audite."""
        base = self._new_success_result()
        cases = (
            replace(base, start_date=date(2026, 9, 2)),
            replace(base, code_version="6" * 40),
            replace(base, games_saved=0),
            replace(base, archive_relative_path="data/raw/missing.json.gz"),
        )
        for result in cases:
            with self.subTest(result=result):
                self._assert_rejected(result)

        with get_connection(self.database_path) as connection:
            connection.execute(
                "UPDATE ingestion_runs SET source = ? WHERE run_id = ?",
                ("source_inconnue", base.run_id),
            )
        self._assert_rejected(base)

    def test_corrupted_raw_archive_is_rejected_without_third_file(self) -> None:
        """Le journal seul ne suffit jamais si l'archive brute a change."""
        result = self._new_success_result()
        archive_path = self.project.joinpath(
            *result.archive_relative_path.split("/")
        )
        with gzip.open(archive_path, "wb") as archive:
            archive.write(b'{"forged":true}\n')

        self._assert_rejected(result)

    def test_active_ingestion_blocks_before_fresh_ingestion(self) -> None:
        """Une collecte started empeche meme l'appel MLB du slot."""
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO ingestion_runs (
                    run_id, source, requested_start_date,
                    requested_end_date, game_types,
                    request_parameters_json, started_at_utc,
                    status, code_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    99,
                    INGESTION_SOURCE,
                    TARGET_DATE,
                    TARGET_DATE,
                    "R",
                    self._request_parameters_json(),
                    "2026-09-03T09:59:01+00:00",
                    "started",
                    RUNTIME_CODE_COMMIT,
                ),
            )

        with patch.object(
            shadow_prediction,
            "run_observed_schedule_ingestion",
            side_effect=AssertionError("appel MLB interdit"),
        ) as ingestion_mock:
            with self.assertRaises(ShadowPredictionError):
                capture_and_publish_source_snapshot(
                    self.reservation,
                    self.activation_publication,
                    database_path=self.database_path,
                    data_directory=self.data_directory,
                    project_directory=self.project,
                )
        ingestion_mock.assert_not_called()
        self.assertFalse((self.slot / SOURCE_SNAPSHOT_FILENAME).exists())

    def test_only_canonical_project_database_can_be_official(self) -> None:
        """Une base preparee parallele ne peut pas remplacer la vraie base."""
        alternate_database = self.data_directory / "prepared-history.db"
        initialize_database(alternate_database)

        with patch.object(
            shadow_prediction,
            "run_observed_schedule_ingestion",
            side_effect=AssertionError("appel MLB interdit"),
        ) as ingestion_mock:
            with self.assertRaises(ShadowPredictionError):
                capture_and_publish_source_snapshot(
                    self.reservation,
                    self.activation_publication,
                    database_path=alternate_database,
                    data_directory=self.data_directory,
                    project_directory=self.project,
                )

        ingestion_mock.assert_not_called()
        self.assertFalse((self.slot / SOURCE_SNAPSHOT_FILENAME).exists())

    def test_active_ingestion_started_during_schedule_is_in_snapshot(self) -> None:
        """Le backup recontrole aussi une collecte demarree apres preflight."""
        result = self._new_success_result(games=())

        def ingestion(**_arguments: object) -> ScheduleIngestionResult:
            with get_connection(self.database_path) as connection:
                connection.execute(
                    """
                    INSERT INTO ingestion_runs (
                        run_id, source, requested_start_date,
                        requested_end_date, game_types,
                        request_parameters_json, started_at_utc,
                        status, code_version
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        999,
                        INGESTION_SOURCE,
                        TARGET_DATE,
                        TARGET_DATE,
                        "R",
                        self._request_parameters_json(),
                        "2026-09-03T10:00:02+00:00",
                        "started",
                        RUNTIME_CODE_COMMIT,
                    ),
                )
            return result

        with (
            patch.object(
                shadow_prediction,
                "run_observed_schedule_ingestion",
                side_effect=ingestion,
            ),
            patch.object(
                shadow_prediction,
                "_utc_now",
                return_value=self._as_datetime(INFORMATION_CUTOFF_UTC),
            ),
        ):
            with self.assertRaises(ShadowPredictionError):
                capture_and_publish_source_snapshot(
                    self.reservation,
                    self.activation_publication,
                    database_path=self.database_path,
                    data_directory=self.data_directory,
                    project_directory=self.project,
                )
        self.assertFalse((self.slot / SOURCE_SNAPSHOT_FILENAME).exists())

    def test_foreign_key_violation_in_sqlite_snapshot_is_rejected(self) -> None:
        """Le backup officiel doit passer PRAGMA foreign_key_check."""
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute(
                """
                INSERT INTO games (
                    game_id, season, official_date, game_type,
                    status_code, status_detail, away_team_id,
                    home_team_id, away_score, home_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    7999,
                    2026,
                    "2026-09-01",
                    "R",
                    "F",
                    "Final",
                    9901,
                    9902,
                    1,
                    0,
                ),
            )
        result = self._new_success_result(games=())
        self._assert_rejected(result)

    def test_malformed_source_final_game_is_rejected(self) -> None:
        """Un score final nul ou egalite ne devient jamais une variable."""
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO games (
                    game_id, season, official_date, game_type,
                    status_code, status_detail, away_team_id,
                    home_team_id, away_score, home_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    7998,
                    2026,
                    "2026-09-01",
                    "R",
                    "F",
                    "Final",
                    101,
                    102,
                    2,
                    2,
                ),
            )
        result = self._new_success_result(games=())
        self._assert_rejected(result)

    def test_missing_team_identity_is_rejected(self) -> None:
        """Toutes les equipes du calendrier doivent exister dans teams."""
        result = self._new_success_result(
            games=(
                _target_game(
                    away_team_id=909,
                    home_team_id=102,
                ),
            )
        )
        self._assert_rejected(result)

    def test_raw_archive_symbolic_link_is_rejected(self) -> None:
        """Une archive officielle ne peut pas etre substituee par symlink."""
        result = self._new_success_result(games=())
        archive_path = self.project.joinpath(
            *result.archive_relative_path.split("/")
        )
        alternate_path = archive_path.with_name("alternate.json.gz")
        alternate_path.write_bytes(archive_path.read_bytes())
        archive_path.unlink()
        try:
            archive_path.symlink_to(alternate_path.name)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"Liens symboliques indisponibles : {error}")

        self._assert_rejected(result)

    def test_retry_policy_uses_zero_then_one_then_two_seconds(self) -> None:
        """Seules les erreurs MLB temporaires utilisent les trois essais."""
        calls: list[str] = []
        sleeps: list[float] = []

        def first() -> ScheduleIngestionResult:
            calls.append("attempt-1")
            self._insert_error_run("temporary-1")
            raise MLBAPIRetryableError("temporary-1")

        def second() -> ScheduleIngestionResult:
            calls.append("attempt-2")
            self._insert_error_run("temporary-2")
            raise MLBAPIRetryableError("temporary-2")

        def third() -> ScheduleIngestionResult:
            calls.append("attempt-3")
            return self._new_success_result(games=())

        attempts = iter((first, second, third))

        def operation(**_arguments: object) -> ScheduleIngestionResult:
            return next(attempts)()

        with (
            patch.object(
                shadow_prediction,
                "run_observed_schedule_ingestion",
                side_effect=operation,
            ),
            patch.object(
                shadow_prediction,
                "_utc_now",
                return_value=self._as_datetime(INFORMATION_CUTOFF_UTC),
            ),
            patch.object(
                shadow_prediction.time,
                "sleep",
                side_effect=sleeps.append,
            ),
        ):
            publication = capture_and_publish_source_snapshot(
                self.reservation,
                self.activation_publication,
                database_path=self.database_path,
                data_directory=self.data_directory,
                project_directory=self.project,
            )

        self.assertEqual(calls, ["attempt-1", "attempt-2", "attempt-3"])
        self.assertEqual(sleeps, [1.0, 2.0])
        self.assertEqual(publication.schedule_attempts, 3)
        with get_connection(self.database_path) as connection:
            statuses = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT status
                    FROM ingestion_runs
                    WHERE run_id >= 100
                    ORDER BY run_id
                    """
                )
            ]
        self.assertEqual(statuses, ["error", "error", "success"])

    def test_retry_policy_has_no_runtime_override_parameter(self) -> None:
        """La politique 0/1/2 figee ne peut pas etre changee par appelant."""
        parameters = inspect.signature(
            capture_and_publish_source_snapshot
        ).parameters
        self.assertNotIn("sleep_function", parameters)
        self.assertNotIn("_sleep_function", parameters)

    def test_non_retryable_error_is_not_retried_or_delayed(self) -> None:
        """Une erreur de contenu MLB remonte immediatement sans pause."""
        sleeps: list[float] = []
        with (
            patch.object(
                shadow_prediction,
                "run_observed_schedule_ingestion",
                side_effect=MLBAPIError("contenu invalide"),
            ) as ingestion_mock,
            patch.object(
                shadow_prediction.time,
                "sleep",
                side_effect=sleeps.append,
            ),
        ):
            with self.assertRaises(MLBAPIError):
                capture_and_publish_source_snapshot(
                    self.reservation,
                    self.activation_publication,
                    database_path=self.database_path,
                    data_directory=self.data_directory,
                    project_directory=self.project,
                )
        self.assertEqual(ingestion_mock.call_count, 1)
        self.assertEqual(sleeps, [])
        self.assertFalse((self.slot / SOURCE_SNAPSHOT_FILENAME).exists())

    def test_retryable_error_is_limited_to_three_attempts(self) -> None:
        """La troisieme erreur temporaire est terminale sans publication."""
        sleeps: list[float] = []
        with (
            patch.object(
                shadow_prediction,
                "run_observed_schedule_ingestion",
                side_effect=MLBAPIRetryableError("toujours indisponible"),
            ) as ingestion_mock,
            patch.object(
                shadow_prediction.time,
                "sleep",
                side_effect=sleeps.append,
            ),
        ):
            with self.assertRaises(MLBAPIRetryableError):
                capture_and_publish_source_snapshot(
                    self.reservation,
                    self.activation_publication,
                    database_path=self.database_path,
                    data_directory=self.data_directory,
                    project_directory=self.project,
                )
        self.assertEqual(ingestion_mock.call_count, 3)
        self.assertEqual(sleeps, [1.0, 2.0])
        self.assertFalse((self.slot / SOURCE_SNAPSHOT_FILENAME).exists())

    def test_extra_or_changed_predecessor_blocks_before_sources(self) -> None:
        """Le snapshot doit etre exactement le troisieme fichier du slot."""
        unexpected = self.slot / "unexpected.bin"
        unexpected.write_bytes(b"immutable\n")
        before = unexpected.read_bytes()

        with patch.object(
            shadow_prediction,
            "run_observed_schedule_ingestion",
            side_effect=AssertionError("appel MLB interdit"),
        ) as ingestion_mock:
            with self.assertRaises(ShadowPredictionSlotConsumedError):
                capture_and_publish_source_snapshot(
                    self.reservation,
                    self.activation_publication,
                    database_path=self.database_path,
                    data_directory=self.data_directory,
                    project_directory=self.project,
                )
        ingestion_mock.assert_not_called()
        self.assertEqual(unexpected.read_bytes(), before)
        self.assertFalse((self.slot / SOURCE_SNAPSHOT_FILENAME).exists())

    def test_second_call_never_reuses_overwrites_or_repairs_snapshot(self) -> None:
        """Une publication reussie consomme definitivement le troisieme lien."""
        result = self._new_success_result()
        publication, _ = self._capture(result)
        before = publication.snapshot_path.read_bytes()

        with patch.object(
            shadow_prediction,
            "run_observed_schedule_ingestion",
            side_effect=AssertionError("appel MLB interdit"),
        ) as ingestion_mock:
            with self.assertRaises(ShadowPredictionSlotConsumedError):
                capture_and_publish_source_snapshot(
                    self.reservation,
                    self.activation_publication,
                    database_path=self.database_path,
                    data_directory=self.data_directory,
                    project_directory=self.project,
                )
        ingestion_mock.assert_not_called()
        self.assertEqual(publication.snapshot_path.read_bytes(), before)

    def test_concurrent_publishers_have_exactly_one_winner(self) -> None:
        """Un seul thread collecte; les autres perdent avant toute source."""
        result = self._new_success_result(games=())
        ingestion_started = Event()
        release_ingestion = Event()

        def observed_ingestion(**_arguments: object):
            ingestion_started.set()
            if not release_ingestion.wait(timeout=10):
                raise AssertionError("Le test n'a pas libere la collecte.")
            return result

        def attempt(_index: int) -> str:
            try:
                capture_and_publish_source_snapshot(
                    self.reservation,
                    self.activation_publication,
                    database_path=self.database_path,
                    data_directory=self.data_directory,
                    project_directory=self.project,
                )
            except ShadowPredictionSlotConsumedError:
                return "consumed"
            return "published"

        with (
            patch.object(
                shadow_prediction,
                "run_observed_schedule_ingestion",
                side_effect=observed_ingestion,
            ) as ingestion_mock,
            patch.object(
                shadow_prediction,
                "_utc_now",
                return_value=self._as_datetime(INFORMATION_CUTOFF_UTC),
            ),
            ThreadPoolExecutor(max_workers=8) as executor,
        ):
            winner = executor.submit(attempt, 0)
            if not ingestion_started.wait(timeout=5):
                release_ingestion.set()
                self.fail("Le premier producteur n'a pas atteint la source.")
            losers = [executor.submit(attempt, index) for index in range(1, 8)]
            try:
                loser_outcomes = [future.result(timeout=5) for future in losers]
            finally:
                release_ingestion.set()
            outcomes = [winner.result(timeout=5), *loser_outcomes]

        self.assertEqual(outcomes.count("published"), 1)
        self.assertEqual(outcomes.count("consumed"), 7)
        self.assertEqual(ingestion_mock.call_count, 1)
        snapshot = self.slot / SOURCE_SNAPSHOT_FILENAME
        self.assertTrue(snapshot.is_file())
        self.assertTrue(gzip.decompress(snapshot.read_bytes()).endswith(b"\n"))

    def test_source_stage_lock_excludes_a_second_process(self) -> None:
        """Le verrou RESERVED doit aussi departager deux processus."""
        context = get_context("spawn")
        results = context.Queue()
        process = context.Process(
            target=_attempt_source_stage_lock_in_child,
            args=(self.reservation, self.project, results),
        )

        with shadow_prediction._hold_source_snapshot_stage_lock(
            self.reservation,
            project_directory=self.project,
        ):
            process.start()
            process.join(timeout=10)

        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            self.fail("Le second processus est reste bloque sur RESERVED.")
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(results.get(timeout=5), "consumed")
        results.close()
        results.join_thread()

    def test_temporary_sqlite_backup_lives_until_verified_publication(self) -> None:
        """Le fichier temporaire subsiste jusqu'au lien officiel verifie."""
        result = self._new_success_result(games=())
        actual_open = shadow_prediction._open_sqlite_read_only
        actual_publish = shadow_prediction._publish_exclusive_verified
        temporary_snapshots: list[Path] = []

        def recording_open(path: Path):
            if path != self.database_path:
                temporary_snapshots.append(path)
            return actual_open(path)

        def checking_publish(path: Path, content: bytes) -> str:
            if path.name == SOURCE_SNAPSHOT_FILENAME:
                self.assertEqual(len(temporary_snapshots), 1)
                self.assertTrue(temporary_snapshots[0].is_file())
                self.assertGreater(temporary_snapshots[0].stat().st_size, 0)
            return actual_publish(path, content)

        with (
            patch.object(
                shadow_prediction,
                "_open_sqlite_read_only",
                side_effect=recording_open,
            ),
            patch.object(
                shadow_prediction,
                "_publish_exclusive_verified",
                side_effect=checking_publish,
            ),
        ):
            self._capture(result)

        self.assertEqual(len(temporary_snapshots), 1)
        self.assertFalse(temporary_snapshots[0].exists())

    def test_temporary_sqlite_backup_cannot_change_during_extraction(
        self,
    ) -> None:
        """Lignes et empreinte doivent provenir des memes octets SQLite."""
        result = self._new_success_result(games=())
        actual_reader = shadow_prediction._read_snapshot_rows

        def mutate_after_read(connection, **arguments):
            rows = actual_reader(connection, **arguments)
            database_row = connection.execute(
                "PRAGMA database_list"
            ).fetchone()
            snapshot_path = Path(database_row[2])
            with snapshot_path.open("ab") as snapshot_file:
                snapshot_file.write(b"alteration-apres-extraction")
            return rows

        with patch.object(
            shadow_prediction,
            "_read_snapshot_rows",
            side_effect=mutate_after_read,
        ):
            self._assert_rejected(result)

    def test_source_stage_creates_no_prediction_or_model_output(self) -> None:
        """Ce jalon ne charge ni modele ni dataset et ne predit rien."""
        result = self._new_success_result()
        self._capture(result)

        self.assertEqual(
            sorted(path.name for path in self.slot.iterdir()),
            sorted(
                [
                    "RESERVED",
                    ACTIVATION_REVERIFICATION_FILENAME,
                    SOURCE_SNAPSHOT_FILENAME,
                ]
            ),
        )
        for forbidden in (
            "candidate_ledger.csv",
            "features.csv",
            "predictions.csv",
            "receipt.json",
            "COMPLETED",
        ):
            self.assertFalse((self.slot / forbidden).exists())

        source = "\n".join(
            inspect.getsource(function)
            for function in (
                shadow_prediction.capture_and_publish_source_snapshot,
                shadow_prediction._read_snapshot_rows,
                shadow_prediction._build_and_publish_source_snapshot,
            )
        ).lower()
        for forbidden_token in (
            "joblib",
            "predict_proba",
            "read_csv",
            "calibrated_classifier",
            "betting",
            "odds",
        ):
            self.assertNotIn(forbidden_token, source)


if __name__ == "__main__":
    unittest.main()
