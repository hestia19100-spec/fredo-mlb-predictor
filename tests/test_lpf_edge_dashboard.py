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
        slot = (
            self.project
            / SCORING_ROOT
            / self.target.isoformat()
            / "observations"
            / "2026-09-12"
        )
        slot.mkdir(parents=True)
        (slot / "COMPLETED").write_text("complete\n", encoding="utf-8")
        report = {
            "accuracy": 0.8,
            "checkpoint_utc_date": "2026-09-12",
            "correct_count": 4,
            "incorrect_count": 1,
            "mean_brier_score": 0.2,
            "mean_log_loss": 0.6,
            "pending_count": 0,
            "scored_count": 5,
            "target_official_date": self.target.isoformat(),
            "void_count": 0,
        }
        (slot / "daily_report.json").write_bytes(canonical_json(report))
        summary = load_latest_score_summary(
            self.target, project_directory=self.project
        )
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.scored_count, 5)
        self.assertEqual(summary.accuracy, 0.8)

    def test_reader_has_no_network_or_model_dependency(self) -> None:
        source = Path(__file__).parents[1].joinpath(
            "src", "lpf_edge_dashboard.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("import requests", source)
        self.assertNotIn("import joblib", source)
        self.assertNotIn("predict_proba", source)

    def test_streamlit_page_is_strictly_read_only(self) -> None:
        source = Path(__file__).parents[1].joinpath(
            "pages", "1_LPF_Edge.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("st.button", source)
        self.assertNotIn("run_schedule_ingestion", source)
        self.assertNotIn("shadow_prediction", source)
        self.assertNotIn("shadow_certification", source)


if __name__ == "__main__":
    unittest.main()
