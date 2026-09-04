"""Publication canonique du sixieme fichier shadow v2, donnees synthetiques."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import hashlib
import inspect
import io
from pathlib import Path
import sqlite3
from threading import Event
import unittest
from unittest.mock import patch

from src import shadow_prediction as shadow

try:
    from tests import test_shadow_prediction_candidate_features as candidate_tests
except ImportError:
    import test_shadow_prediction_candidate_features as candidate_tests


PREDICTION_COLUMNS = (
    "prediction_id", "batch_id", "game_id", "occurrence_key", "season",
    "official_date_at_prediction", "away_team_id", "home_team_id",
    "scheduled_start_utc_at_prediction", "information_cutoff_utc",
    "issued_at_utc", "feature_as_of_date", "away_max_source_date",
    "home_max_source_date", "away_games_before", "away_win_pct_before",
    "away_runs_scored_per_game_before", "away_runs_allowed_per_game_before",
    "home_games_before", "home_win_pct_before",
    "home_runs_scored_per_game_before", "home_runs_allowed_per_game_before",
    "p_home_win", "p_away_win", "model_version", "artifact_sha256",
    "protocol_sha256", "code_commit",
)
ISSUED_AT = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)
ISSUED_TEXT = "2026-09-03T12:00:00Z"
STATE_HASH = "7" * 64
RUNTIME_VERSIONS = tuple(sorted((key, "test-version") for key in (
    "python", "numpy", "pandas", "scipy", "scikit_learn", "joblib",
)))


def _csv_bytes(columns, rows) -> bytes:
    destination = io.StringIO(newline="")
    writer = csv.writer(
        destination, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    writer.writerow(columns)
    writer.writerows(rows)
    return destination.getvalue().encode("utf-8")


def _typed_feature_rows(path: Path) -> tuple[tuple[object, ...], ...]:
    with path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.reader(source))
    if tuple(rows[0]) != candidate_tests.FEATURE_COLUMNS:
        raise AssertionError("schema de fixture inattendu")
    integer_indexes = {2, 4, 6, 7, 12, 16}
    return tuple(
        tuple(int(value) if index in integer_indexes else value
              for index, value in enumerate(row))
        for row in rows[1:]
    )


class ShadowPredictionsPublicationTests(unittest.TestCase):
    """Le CSV de probabilites reste lie aux cinq fichiers deja figes."""

    def setUp(self) -> None:
        self.fixture = candidate_tests.ShadowCandidateFeaturesTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.project = self.fixture.project
        self.slot = self.fixture.slot
        self.source = self.fixture._prepare()
        self.candidates = self.fixture._build()
        self.feature_rows = _typed_feature_rows(self.candidates.features_path)
        self.model_result = self._model_result(self.feature_rows)

    def _model_result(self, rows, probabilities=None):
        selected = tuple(
            (1.0 - home, home)
            for home in ([0.8, 0.25][:len(rows)])
        ) if probabilities is None else probabilities
        return shadow.ShadowModelProbabilities(
            feature_rows=tuple(rows),
            probabilities=tuple(selected),
            artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
            artifact_state_sha256_before=STATE_HASH,
            artifact_state_sha256_after=STATE_HASH,
            runtime_versions=RUNTIME_VERSIONS,
            approved_deserialization_warning_count=1,
            approved_state_serialization_warning_count=2,
            approved_compatibility_warning_count=3,
        )

    def _publish(self, model_result=Ellipsis):
        selected = self.model_result if model_result is Ellipsis else model_result
        with patch.object(shadow, "_utc_now", return_value=ISSUED_AT):
            return shadow._build_and_publish_shadow_predictions(
                self.fixture.reservation,
                self.fixture.activation_publication,
                self.source,
                self.candidates,
                selected,
                project_directory=self.project,
            )

    def _expected_rows(self, probabilities=None):
        selected = self.model_result.probabilities if probabilities is None else probabilities
        cutoff = self.source.information_cutoff_utc
        reserved = self.fixture.reservation.reserved_marker
        return [
            [
                *row[:9], cutoff, ISSUED_TEXT, *row[9:20],
                format(pair[1], ".17g"), format(pair[0], ".17g"),
                "logistic_team_form_v1_platt",
                shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
                shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
                reserved["runtime_code_commit"],
            ]
            for row, pair in zip(self.feature_rows, selected)
        ]

    def _assert_no_prediction_file(self):
        self.assertFalse((self.slot / "predictions.csv").exists())

    def test_exact_schema_bytes_order_hash_and_intermediate_proof(self):
        before = {path.name: path.read_bytes() for path in self.slot.iterdir()}
        result = self._publish()
        expected = _csv_bytes(PREDICTION_COLUMNS, self._expected_rows())
        self.assertEqual(result.predictions_path.read_bytes(), expected)
        self.assertEqual(result.predictions_sha256, hashlib.sha256(expected).hexdigest())
        self.assertEqual(result.predictions_size_bytes, len(expected))
        self.assertEqual(result.prediction_row_count, len(self.feature_rows))
        self.assertEqual(result.issued_at_utc, ISSUED_TEXT)
        self.assertEqual(result.earliest_predicted_scheduled_start_utc,
                         self.feature_rows[0][8])
        self.assertEqual(result.predictions_relative_path,
                         "shadow_results/logistic_team_form_v1_platt_shadow_v2/"
                         + candidate_tests.TARGET_DATE + "/predictions.csv")
        self.assertEqual(result.model_version, "logistic_team_form_v1_platt")
        self.assertEqual(result.artifact_sha256, shadow.EXPECTED_MODEL_ARTIFACT_SHA256)
        self.assertEqual(result.protocol_sha256, shadow.EXPECTED_SHADOW_PROTOCOL_SHA256)
        self.assertEqual(result.code_commit,
                         self.fixture.reservation.reserved_marker["runtime_code_commit"])
        self.assertEqual(result.artifact_state_sha256_before, STATE_HASH)
        self.assertEqual(result.artifact_state_sha256_after, STATE_HASH)
        self.assertEqual(result.runtime_versions, RUNTIME_VERSIONS)
        self.assertEqual(result.approved_deserialization_warning_count, 1)
        self.assertEqual(result.approved_state_serialization_warning_count, 2)
        self.assertEqual(result.approved_compatibility_warning_count, 3)
        self.assertEqual(result.predict_proba_calls, 1)
        self.assertTrue(result.artifact_state_unchanged)
        self.assertTrue(result.official_prediction_created)
        self.assertFalse(result.receipt_created)
        self.assertFalse(result.slot_completed)
        self.assertEqual(result.unexpected_warning_count, 0)
        self.assertEqual(result.warning_policy_id,
                         "VERIFIED_JOBLIB_NUMPY_COMPATIBILITY_V1")
        for name, content in before.items():
            self.assertEqual((self.slot / name).read_bytes(), content, name)
        self.assertEqual(set(path.name for path in self.slot.iterdir()), {
            "RESERVED", "activation_reverification.remote.json.gz",
            "source_snapshot.json.gz", "candidate_ledger.csv", "features.csv",
            "predictions.csv",
        })
        with self.assertRaises(FrozenInstanceError):
            result.prediction_row_count = 99

    def test_values_use_exact_complement_and_seventeen_significant_digits(self):
        probabilities = ((1.0 - 0.12345678901234567, 0.12345678901234567),
                         (1.0, 0.0))
        result = self._publish(self._model_result(self.feature_rows, probabilities))
        with result.predictions_path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]["p_home_win"], format(probabilities[0][1], ".17g"))
        self.assertEqual(rows[0]["p_away_win"], format(probabilities[0][0], ".17g"))
        self.assertEqual((rows[1]["p_home_win"], rows[1]["p_away_win"]), ("0", "1"))
        self.assertEqual([int(row["game_id"]) for row in rows],
                         [row[2] for row in self.feature_rows])

    def test_empty_eligible_batch_publishes_header_only_without_model_result(self):
        # Nouvelle fixture propre pour remplacer le lot non vide du setUp.
        fixture = candidate_tests.ShadowCandidateFeaturesTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        source = fixture._prepare(games=())
        candidates = fixture._build()
        self.assertEqual(candidates.feature_row_count, 0)
        with patch.object(shadow, "_utc_now", return_value=ISSUED_AT):
            result = shadow._build_and_publish_shadow_predictions(
                fixture.reservation, fixture.activation_publication, source,
                candidates, None, project_directory=fixture.project,
            )
        self.assertEqual(result.predictions_path.read_bytes(),
                         _csv_bytes(PREDICTION_COLUMNS, []))
        self.assertEqual(result.prediction_row_count, 0)
        self.assertIsNone(result.earliest_predicted_scheduled_start_utc)
        self.assertIsNone(result.artifact_state_sha256_before)
        self.assertIsNone(result.artifact_state_sha256_after)
        self.assertEqual(result.runtime_versions, ())
        self.assertEqual(result.approved_compatibility_warning_count, 0)
        self.assertEqual(result.predict_proba_calls, 0)

    def test_nonempty_requires_result_and_empty_forbids_one(self):
        with self.assertRaises(shadow.ShadowPredictionError):
            self._publish(None)
        self._assert_no_prediction_file()

        fixture = candidate_tests.ShadowCandidateFeaturesTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        source = fixture._prepare(games=())
        candidates = fixture._build()
        with patch.object(shadow, "_utc_now", return_value=ISSUED_AT):
            with self.assertRaises(shadow.ShadowPredictionError):
                shadow._build_and_publish_shadow_predictions(
                    fixture.reservation, fixture.activation_publication, source,
                    candidates, self.model_result, project_directory=fixture.project,
                )
        self.assertFalse((fixture.slot / "predictions.csv").exists())

    def test_feature_identity_and_probability_count_must_match_exactly(self):
        mutations = (
            replace(self.model_result, feature_rows=self.model_result.feature_rows[::-1]),
            replace(self.model_result, probabilities=self.model_result.probabilities[:1]),
        )
        for value in mutations:
            with self.subTest(value=value):
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._publish(value)
                self._assert_no_prediction_file()

    def test_probability_pairs_are_exact_floats_finite_bounded_and_complements(self):
        invalid = (
            ((0.2, 0.8), (0.5,)),
            ((0.2, 0.8), [0.75, 0.25]),
            ((0, 1.0), (0.75, 0.25)),
            ((float("nan"), 0.5), (0.75, 0.25)),
            ((-0.0, 1.0), (0.75, 0.25)),
            ((0.3, 0.8), (0.75, 0.25)),
            ((0.19999999999999996, 0.7), (0.75, 0.25)),
        )
        for probabilities in invalid:
            with self.subTest(probabilities=probabilities):
                result = replace(self.model_result, probabilities=probabilities)
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._publish(result)
                self._assert_no_prediction_file()

    def test_every_model_invariant_is_revalidated_before_publication(self):
        mutations = {
            "artifact_sha256": "8" * 64,
            "artifact_state_sha256_before": "6" * 64,
            "artifact_state_unchanged": False,
            "predict_proba_calls": 2,
            "unexpected_warning_count": 1,
            "warning_policy_id": "OTHER",
            "execution_ready": True,
            "activation_verified": True,
            "execution_manifest_verified": True,
            "official_prediction_created": True,
            "approved_deserialization_warning_count": -1,
            "approved_state_serialization_warning_count": -1,
            "approved_compatibility_warning_count": 4,
            "runtime_versions": RUNTIME_VERSIONS[:-1],
        }
        for name, value in mutations.items():
            with self.subTest(name=name):
                if name in {
                    "artifact_state_unchanged", "predict_proba_calls",
                    "unexpected_warning_count", "warning_policy_id",
                    "execution_ready", "activation_verified",
                    "execution_manifest_verified", "official_prediction_created",
                }:
                    changed = deepcopy(self.model_result)
                    object.__setattr__(changed, name, value)
                else:
                    changed = replace(self.model_result, **{name: value})
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._publish(changed)
                self._assert_no_prediction_file()

    def test_issued_time_is_captured_once_after_validation(self):
        with patch.object(shadow, "_utc_now", return_value=ISSUED_AT) as clock:
            shadow._build_and_publish_shadow_predictions(
                self.fixture.reservation, self.fixture.activation_publication,
                self.source, self.candidates, self.model_result,
                project_directory=self.project,
            )
        clock.assert_called_once_with()

    def test_time_before_cutoff_or_too_late_is_rejected_without_csv(self):
        values = (
            datetime(2026, 9, 3, 9, 59, 59, tzinfo=timezone.utc),
            datetime(2026, 9, 3, 16, 0, 1, tzinfo=timezone.utc),
        )
        for instant in values:
            with self.subTest(instant=instant):
                with patch.object(shadow, "_utc_now", return_value=instant):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        shadow._build_and_publish_shadow_predictions(
                            self.fixture.reservation,
                            self.fixture.activation_publication,
                            self.source, self.candidates, self.model_result,
                            project_directory=self.project,
                        )
                self._assert_no_prediction_file()

    def test_two_hour_boundary_is_inclusive_at_prediction_publication(self):
        boundary = datetime(2026, 9, 3, 16, 0, 0, tzinfo=timezone.utc)
        with patch.object(shadow, "_utc_now", return_value=boundary):
            result = shadow._build_and_publish_shadow_predictions(
                self.fixture.reservation, self.fixture.activation_publication,
                self.source, self.candidates, self.model_result,
                project_directory=self.project,
            )
        self.assertEqual(result.issued_at_utc, "2026-09-03T16:00:00Z")

    def test_forged_candidate_publication_fields_are_rejected(self):
        mutations = {
            "candidate_ledger_sha256": "9" * 64,
            "features_sha256": "9" * 64,
            "candidate_ledger_size_bytes": self.candidates.candidate_ledger_size_bytes + 1,
            "features_size_bytes": self.candidates.features_size_bytes + 1,
            "candidate_row_count": self.candidates.candidate_row_count + 1,
            "feature_row_count": self.candidates.feature_row_count + 1,
            "eligible_game_count": self.candidates.eligible_game_count + 1,
            "excluded_games_by_reason": (),
            "earliest_eligible_scheduled_start_utc": "2026-09-03T17:00:00Z",
            "slot_path": self.project,
            "features_path": self.slot / "other.csv",
            "features_relative_path": "other.csv",
        }
        for name, value in mutations.items():
            with self.subTest(name=name):
                forged = replace(self.candidates, **{name: value})
                with self.assertRaises(shadow.ShadowPredictionError):
                    with patch.object(shadow, "_utc_now", return_value=ISSUED_AT):
                        shadow._build_and_publish_shadow_predictions(
                            self.fixture.reservation,
                            self.fixture.activation_publication,
                            self.source, forged, self.model_result,
                            project_directory=self.project,
                        )
                self._assert_no_prediction_file()

    def test_changed_source_candidate_or_features_bytes_are_rejected(self):
        paths = (self.source.snapshot_path,
                 self.candidates.candidate_ledger_path,
                 self.candidates.features_path)
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"x")
                try:
                    with self.assertRaises(shadow.ShadowPredictionError):
                        self._publish()
                    self._assert_no_prediction_file()
                finally:
                    path.write_bytes(original)

    def test_preexisting_sixth_or_terminal_path_is_never_overwritten(self):
        for name in ("predictions.csv", "receipt.json", "COMPLETED", "FAILED.json"):
            with self.subTest(name=name):
                path = self.slot / name
                original = b"foreign\n"
                path.write_bytes(original)
                try:
                    with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                        self._publish()
                    self.assertEqual(path.read_bytes(), original)
                finally:
                    path.unlink()

    def test_second_publication_never_reuses_or_overwrites_predictions(self):
        first = self._publish()
        before = first.predictions_path.read_bytes()
        with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
            self._publish()
        self.assertEqual(first.predictions_path.read_bytes(), before)

    def test_failure_after_exclusive_link_preserves_file_and_prevents_repair(self):
        actual = shadow._publish_exclusive_verified
        def publish_then_fail(destination, content):
            digest = actual(destination, content)
            if destination.name == "predictions.csv":
                raise OSError("synthetic post-link failure")
            return digest
        with patch.object(shadow, "_publish_exclusive_verified",
                          side_effect=publish_then_fail):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._publish()
        path = self.slot / "predictions.csv"
        preserved = path.read_bytes()
        with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
            self._publish()
        self.assertEqual(path.read_bytes(), preserved)

    def test_failure_before_link_leaves_exact_five_predecessors(self):
        names = set(path.name for path in self.slot.iterdir())
        with patch.object(shadow, "_publish_exclusive_verified",
                          side_effect=OSError("synthetic pre-link failure")):
            with self.assertRaises(shadow.ShadowPredictionError):
                self._publish()
        self.assertEqual(set(path.name for path in self.slot.iterdir()), names)
        self._assert_no_prediction_file()

    def test_predecessor_mutation_during_csv_build_blocks_publication(self):
        original = self.candidates.features_path.read_bytes()
        actual = shadow._canonical_csv_bytes
        def mutate_after_build(columns, rows):
            content = actual(columns, rows)
            if tuple(columns) == PREDICTION_COLUMNS:
                self.candidates.features_path.write_bytes(original + b"changed")
            return content
        try:
            with patch.object(shadow, "_canonical_csv_bytes",
                              side_effect=mutate_after_build):
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._publish()
            self._assert_no_prediction_file()
        finally:
            self.candidates.features_path.write_bytes(original)

    def test_predecessor_mutation_after_link_keeps_partial_slot_terminal(self):
        original = self.candidates.features_path.read_bytes()
        actual = shadow._publish_exclusive_verified
        def mutate_after_link(destination, content):
            digest = actual(destination, content)
            if destination.name == "predictions.csv":
                self.candidates.features_path.write_bytes(original + b"changed")
            return digest
        try:
            with patch.object(shadow, "_publish_exclusive_verified",
                              side_effect=mutate_after_link):
                with self.assertRaises(shadow.ShadowPredictionError):
                    self._publish()
            preserved = (self.slot / "predictions.csv").read_bytes()
            with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                self._publish()
            self.assertEqual((self.slot / "predictions.csv").read_bytes(), preserved)
        finally:
            self.candidates.features_path.write_bytes(original)

    def test_concurrent_publishers_have_exactly_one_winner(self):
        entered = Event()
        release = Event()
        actual = shadow._canonical_csv_bytes
        def slow_predictions(columns, rows):
            if tuple(columns) == PREDICTION_COLUMNS:
                entered.set()
                if not release.wait(5):
                    raise AssertionError("timeout")
            return actual(columns, rows)
        def attempt():
            try:
                self._publish()
                return "success"
            except shadow.ShadowPredictionSlotConsumedError:
                return "consumed"
        with patch.object(shadow, "_canonical_csv_bytes", side_effect=slow_predictions):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(attempt)
                self.assertTrue(entered.wait(5))
                second = pool.submit(attempt)
                second_result = second.result(timeout=5)
                release.set()
                first_result = first.result(timeout=5)
        self.assertCountEqual((first_result, second_result), ("success", "consumed"))
        self.assertTrue((self.slot / "predictions.csv").is_file())

    def test_publication_never_loads_calls_or_trains_model_or_reads_external_sources(self):
        with (
            patch.object(shadow, "_load_frozen_shadow_model",
                         side_effect=AssertionError("chargement interdit")) as loader,
            patch.object(shadow, "_predict_frozen_shadow_model_once",
                         side_effect=AssertionError("appel modele interdit")) as predict,
            patch.object(shadow.requests, "get",
                         side_effect=AssertionError("reseau interdit")) as network,
            patch.object(sqlite3, "connect",
                         side_effect=AssertionError("SQLite interdit")) as database,
        ):
            self._publish()
        loader.assert_not_called()
        predict.assert_not_called()
        network.assert_not_called()
        database.assert_not_called()

    def test_output_has_no_score_outcome_odds_bet_or_metric_field(self):
        result = self._publish()
        with result.predictions_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.reader(source)
            header = next(reader)
        self.assertEqual(tuple(header), PREDICTION_COLUMNS)
        forbidden = {
            "away_score", "home_score", "home_win", "target", "outcome",
            "odds", "implied_probability", "pick", "stake", "edge",
            "expected_value", "profit", "roi", "venue", "probable_pitcher",
        }
        self.assertTrue(forbidden.isdisjoint(header))

    def test_function_is_private_and_has_no_clock_model_or_policy_override(self):
        signature = inspect.signature(shadow._build_and_publish_shadow_predictions)
        self.assertEqual(list(signature.parameters), [
            "reservation", "activation_publication", "source_publication",
            "candidate_publication", "model_probabilities", "project_directory",
        ])
        self.assertNotIn("build_and_publish_shadow_predictions", shadow.__dict__)
        for forbidden in (
            "issued_at_utc", "clock", "predict", "model", "artifact_path",
            "retry", "columns", "serializer", "overwrite",
        ):
            self.assertNotIn(forbidden, signature.parameters)


if __name__ == "__main__":
    unittest.main()
