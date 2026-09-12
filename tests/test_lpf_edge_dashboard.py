"""Tests de la consultation en lecture seule des prédictions LPF Edge."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.lpf_edge_dashboard import (
    ADJUDICATION_COLUMNS,
    CERTIFICATION_ROOT,
    PREDICTION_COLUMNS,
    PREDICTION_ROOT,
    SCORING_ROOT,
    LPFEdgeDashboardError,
    list_certified_prediction_dates,
    load_certified_prediction_day,
    load_latest_score_summary,
    load_team_names,
)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


class LPFEdgeDashboardTests(unittest.TestCase):
    """La page ne montre que des octets certifiés et intègres."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name)
        self.target = date(2026, 9, 11)
        self.batch_id = "a" * 64
        self.result_root = self.project / PREDICTION_ROOT / self.target.isoformat()
        self.certification_root = self.project / CERTIFICATION_ROOT
        self.result_root.mkdir(parents=True)
        self.certification_root.mkdir(parents=True)
        (self.result_root / "COMPLETED").write_bytes(b"complete\n")

        row = {column: "x" for column in PREDICTION_COLUMNS}
        row.update(
            {
                "prediction_id": "b" * 64,
                "batch_id": self.batch_id,
                "game_id": "123",
                "occurrence_key": "c" * 64,
                "season": "2026",
                "official_date_at_prediction": self.target.isoformat(),
                "away_team_id": "10",
                "home_team_id": "20",
                "scheduled_start_utc_at_prediction": "2026-09-11T18:20:00Z",
                "information_cutoff_utc": "2026-09-11T06:18:00Z",
                "issued_at_utc": "2026-09-11T06:19:00Z",
                "feature_as_of_date": "2026-09-10",
                "away_max_source_date": "2026-09-10",
                "home_max_source_date": "2026-09-10",
                "away_games_before": "145",
                "home_games_before": "145",
                "p_home_win": "0.625",
                "p_away_win": "0.375",
            }
        )
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
        self.prediction_bytes = stream.getvalue().encode("utf-8")
        (self.result_root / "predictions.csv").write_bytes(self.prediction_bytes)

        prediction_sha = hashlib.sha256(self.prediction_bytes).hexdigest()
        receipt = {
            "batch": {
                "batch_id": self.batch_id,
                "status": "COMPLETED_WITH_PREDICTIONS",
                "target_official_date": self.target.isoformat(),
            },
            "counts": {"predicted_games": 1},
            "output_hashes": {"predictions_sha256": prediction_sha},
        }
        self.receipt_bytes = canonical_json(receipt)
        (self.result_root / "receipt.json").write_bytes(self.receipt_bytes)
        receipt_sha = hashlib.sha256(self.receipt_bytes).hexdigest()

        certification = {
            "batch_id": self.batch_id,
            "remote_http_date_utc": "2026-09-11T06:23:34Z",
            "remote_publication_lead_minutes": "716.433333",
            "remote_response_status_code": 200,
            "results_commit": "d" * 40,
            "results_tree_file_hashes": [
                {
                    "path": "predictions.csv",
                    "sha256": prediction_sha,
                    "size_bytes": len(self.prediction_bytes),
                },
                {
                    "path": "receipt.json",
                    "sha256": receipt_sha,
                    "size_bytes": len(self.receipt_bytes),
                },
            ],
            "status": "PROSPECTIVELY_CERTIFIED_REMOTE_MAIN_BEFORE_GAMES",
            "target_official_date": self.target.isoformat(),
        }
        self.certification_path = self.certification_root / f"{self.target}.json"
        self.certification_path.write_bytes(canonical_json(certification))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_score_observation(
        self,
        *,
        correct: bool = True,
        actual_winner_override: str | None = None,
    ) -> Path:
        checkpoint = date(2026, 9, 12)
        observation_id = "e" * 64
        slot = (
            self.project
            / SCORING_ROOT
            / self.target.isoformat()
            / "observations"
            / checkpoint.isoformat()
        )
        slot.mkdir(parents=True)

        away_score, home_score = ((3, 5) if correct else (5, 3))
        actual_winner = "HOME" if correct else "AWAY"
        row = {column: "" for column in ADJUDICATION_COLUMNS}
        row.update(
            {
                "prediction_id": "b" * 64,
                "batch_id": self.batch_id,
                "game_id": "123",
                "occurrence_key": "c" * 64,
                "target_official_date": self.target.isoformat(),
                "away_team_id": "10",
                "home_team_id": "20",
                "p_away_win": "0.375",
                "p_home_win": "0.625",
                "predicted_side": "HOME",
                "outcome_status": "SCORED_FINAL",
                "away_score": str(away_score),
                "home_score": str(home_score),
                "home_win": "1" if correct else "0",
                "actual_winner": actual_winner_override or actual_winner,
                "classification_correct": "1" if correct else "0",
                "individual_log_loss": "0.470003629246",
                "individual_brier_score": "0.140625000000",
                "status_code_normalized": "F",
                "status_detail_normalized": "FINAL",
                "final_official_date": self.target.isoformat(),
                "outcome_http_date_utc": "2026-09-12T06:00:00Z",
                "outcome_response_received_at_utc": "2026-09-12T06:00:01Z",
                "outcome_evidence_sha256": "f" * 64,
            }
        )
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream,
            fieldnames=ADJUDICATION_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(row)
        adjudications_bytes = stream.getvalue().encode("utf-8")
        (slot / "adjudications.csv").write_bytes(adjudications_bytes)

        report = {
            "accuracy": 1.0 if correct else 0.0,
            "certified_prediction_count": 1,
            "checkpoint_utc_date": checkpoint.isoformat(),
            "correct_count": 1 if correct else 0,
            "disclaimer": (
                "PROVISIONAL_DAILY_NO_VERDICT_SMALL_SAMPLE_"
                "DO_NOT_CHANGE_MODEL_OR_PROTOCOL"
            ),
            "incorrect_count": 0 if correct else 1,
            "mean_brier_score": 0.140625,
            "mean_log_loss": 0.470003629246,
            "observation_id": observation_id,
            "pending_count": 0,
            "protocol_id": (
                "logistic_team_form_v1_platt_shadow_v2_2026_scoring_v1"
            ),
            "report_schema_version": 1,
            "scored_count": 1,
            "status": "PROVISIONAL_DAILY_NO_VERDICT",
            "target_official_date": self.target.isoformat(),
            "void_count": 0,
        }
        report_bytes = canonical_json(report)
        (slot / "daily_report.json").write_bytes(report_bytes)
        counts = {
            "certified_prediction_count": 1,
            "correct_count": 1 if correct else 0,
            "incorrect_count": 0 if correct else 1,
            "pending_count": 0,
            "scored_count": 1,
            "void_count": 0,
        }
        receipt = {
            "adjudications_sha256": hashlib.sha256(
                adjudications_bytes
            ).hexdigest(),
            "checkpoint_utc_date": checkpoint.isoformat(),
            "counts": counts,
            "daily_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "observation_id": observation_id,
            "target_official_date": self.target.isoformat(),
        }
        receipt_bytes = canonical_json(receipt)
        (slot / "observation_receipt.json").write_bytes(receipt_bytes)
        completed = {
            "observation_id": observation_id,
            "observation_receipt_sha256": hashlib.sha256(
                receipt_bytes
            ).hexdigest(),
        }
        (slot / "COMPLETED").write_bytes(canonical_json(completed))
        return slot

    def test_discovery_and_loading_of_certified_day(self) -> None:
        self.assertEqual(
            list_certified_prediction_dates(project_directory=self.project),
            [self.target],
        )
        day = load_certified_prediction_day(
            self.target, project_directory=self.project
        )
        self.assertEqual(day.batch_id, self.batch_id)
        self.assertEqual(len(day.predictions), 1)
        self.assertEqual(day.predictions[0].predicted_side, "HOME")
        self.assertEqual(str(day.predictions[0].predicted_probability), "0.625")

    def test_tampered_predictions_are_rejected(self) -> None:
        path = self.result_root / "predictions.csv"
        path.write_bytes(path.read_bytes().replace(b"0.625", b"0.624"))
        with self.assertRaisesRegex(
            LPFEdgeDashboardError, "diffère de sa certification"
        ):
            load_certified_prediction_day(
                self.target, project_directory=self.project
            )

    def test_canonical_float_rounding_is_accepted(self) -> None:
        path = self.result_root / "predictions.csv"
        rounded = self.prediction_bytes.replace(
            b"0.625", b"0.10000000000000001"
        ).replace(b"0.375", b"0.90000000000000002")
        path.write_bytes(rounded)

        receipt = json.loads(self.receipt_bytes)
        prediction_sha = hashlib.sha256(rounded).hexdigest()
        receipt["output_hashes"]["predictions_sha256"] = prediction_sha
        receipt_bytes = canonical_json(receipt)
        (self.result_root / "receipt.json").write_bytes(receipt_bytes)

        certification = json.loads(self.certification_path.read_bytes())
        for item in certification["results_tree_file_hashes"]:
            if item["path"] == "predictions.csv":
                item["sha256"] = prediction_sha
                item["size_bytes"] = len(rounded)
            if item["path"] == "receipt.json":
                item["sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
                item["size_bytes"] = len(receipt_bytes)
        self.certification_path.write_bytes(canonical_json(certification))

        day = load_certified_prediction_day(
            self.target, project_directory=self.project
        )
        self.assertEqual(len(day.predictions), 1)
        self.assertEqual(float(day.predictions[0].p_home_win), 0.1)

    def test_certification_with_less_than_one_hour_is_rejected(self) -> None:
        certification = json.loads(self.certification_path.read_bytes())
        certification["remote_publication_lead_minutes"] = "59.999"
        self.certification_path.write_bytes(canonical_json(certification))
        with self.assertRaisesRegex(LPFEdgeDashboardError, "une heure d'avance"):
            load_certified_prediction_day(
                self.target, project_directory=self.project
            )

    def test_team_names_are_read_without_creating_a_database(self) -> None:
        self.assertEqual(
            load_team_names([10, 20], project_directory=self.project), {}
        )
        self.assertFalse((self.project / "data").exists())

        database_path = self.project / "data/fredo_mlb.db"
        database_path.parent.mkdir()
        connection = sqlite3.connect(database_path)
        connection.execute("CREATE TABLE teams (team_id INTEGER, name TEXT)")
        connection.executemany(
            "INSERT INTO teams VALUES (?, ?)",
            [(10, "Visiteurs"), (20, "Domicile")],
        )
        connection.commit()
        connection.close()
        before = database_path.read_bytes()
        self.assertEqual(
            load_team_names([10, 20], project_directory=self.project),
            {10: "Visiteurs", 20: "Domicile"},
        )
        self.assertEqual(database_path.read_bytes(), before)

    def test_results_are_absent_until_a_completed_report_exists(self) -> None:
        self.assertIsNone(
            load_latest_score_summary(
                self.target, project_directory=self.project
            )
        )
        self._write_score_observation(correct=True)
        day = load_certified_prediction_day(
            self.target, project_directory=self.project
        )
        summary = load_latest_score_summary(
            self.target,
            certified_predictions=day.predictions,
            project_directory=self.project,
        )
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.scored_count, 1)
        self.assertEqual(summary.accuracy, 1.0)
        self.assertEqual(len(summary.results), 1)
        self.assertEqual(summary.results[0].display_status, "Réussie")
        self.assertEqual(summary.results[0].display_tone, "correct")
        self.assertEqual(summary.results[0].home_score, 5)

    def test_incorrect_prediction_is_exposed_as_red_result(self) -> None:
        self._write_score_observation(correct=False)
        day = load_certified_prediction_day(
            self.target, project_directory=self.project
        )
        summary = load_latest_score_summary(
            self.target,
            certified_predictions=day.predictions,
            project_directory=self.project,
        )
        assert summary is not None
        self.assertEqual(summary.correct_count, 0)
        self.assertEqual(summary.incorrect_count, 1)
        self.assertEqual(summary.results[0].display_status, "Ratée")
        self.assertEqual(summary.results[0].display_tone, "incorrect")
        self.assertEqual(summary.results[0].actual_winner, "AWAY")

    def test_score_and_winner_disagreement_is_rejected(self) -> None:
        self._write_score_observation(
            correct=True,
            actual_winner_override="AWAY",
        )
        day = load_certified_prediction_day(
            self.target, project_directory=self.project
        )
        with self.assertRaisesRegex(
            LPFEdgeDashboardError,
            "vainqueur ne correspond pas au score final",
        ):
            load_latest_score_summary(
                self.target,
                certified_predictions=day.predictions,
                project_directory=self.project,
            )

    def test_reader_has_no_network_or_model_dependency(self) -> None:
        source = Path(__file__).parents[1].joinpath(
            "src", "lpf_edge_dashboard.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("import requests", source)
        self.assertNotIn("import joblib", source)
        self.assertNotIn("predict_proba", source)

    def test_streamlit_page_enables_only_controlled_daily_actions(self) -> None:
        source = Path(__file__).parents[1].joinpath(
            "pages", "1_LPF_Edge.py"
        ).read_text(encoding="utf-8")
        self.assertIn("st.button", source)
        self.assertIn("refresh_daily_mlb_data", source)
        self.assertIn("execute_daily_prediction_publication", source)
        self.assertIn("DailyPredictionAutomationError", source)
        self.assertIn("execute_daily_results_publication", source)
        self.assertIn("DailyResultsAutomationError", source)
        self.assertIn("prediction_clicked", source)
        self.assertIn(
            "disabled=not daily.prediction_action.can_execute",
            source,
        )
        self.assertIn("lpf_edge_prediction_success", source)
        self.assertIn("results_clicked", source)
        self.assertIn(
            "disabled=not daily.results_action.can_execute",
            source,
        )
        self.assertIn("lpf_edge_results_success", source)
        self.assertIn("publication.observation_id", source)
        self.assertIn("publication.results_commit", source)
        self.assertIn("publication.target_date.strftime", source)
        self.assertIn("st.rerun()", source)
        self.assertNotIn("run_schedule_ingestion", source)
        self.assertNotIn("shadow_prediction", source)
        self.assertNotIn("shadow_certification", source)
        self.assertNotIn("shadow_scoring", source)
        self.assertNotIn("Cette page ne relance jamais le modèle", source)
        self.assertIn("une action explicite", source)
        self.assertIn("daily.results_action.label", source)
        self.assertIn("Centre d’actions quotidien", source)
        self.assertIn("lpf-result-correct", source)
        self.assertIn("lpf-result-incorrect", source)
        self.assertIn("Prédictions réussies", source)
        self.assertIn("Score final (ext. – dom.)", source)


if __name__ == "__main__":
    unittest.main()
