"""Regression tests for the offline public-schedule NHL reconciliation."""
from __future__ import annotations

import csv
import io
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.nhl.public_schedule_reconciliation import (
    PublicScheduleReconciliationError,
    reconcile_public_schedules,
)

REPORT = (Path(__file__).resolve().parents[2] / "nhl_protocols" / "data"
          / "nhl15_real_archive_report_2026-10-03.json")
COLUMNS = ["game_id", "season", "game_type", "game_date", "game_state",
           "home_team_abbr", "away_team_abbr", "home_score", "away_score"]
OBSERVED = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def fixture():
    report = REPORT.read_bytes()
    ids = json.loads(report)["tied_score_game_ids"]
    rows = {season: [] for season in range(2022, 2027)}
    for game_id in ids:
        season = game_id // 1_000_000 + 1
        rows[season].append({
            "game_id": str(game_id), "season": str(season), "game_type": "R",
            "game_date": f"{season - 1}-10-15", "game_state": "OFF",
            "home_team_abbr": "BOS", "away_team_abbr": "NYR",
            "home_score": "3", "away_score": "2",
        })
    return report, rows


def archive_bytes(rows):
    result = io.StringIO()
    writer = csv.DictWriter(result, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return result.getvalue().encode("utf-8")


def run_audit(report, rows, observed=OBSERVED):
    return reconcile_public_schedules(
        report, {season: archive_bytes(batch) for season, batch in rows.items()},
        observed_at_utc=observed)


class PublicScheduleReconciliationTests(unittest.TestCase):
    def test_all_475_resolved_without_opening_any_gate(self):
        report, rows = fixture()
        audit = run_audit(report, rows)
        self.assertEqual(audit.resolved_count, 475)
        self.assertEqual(audit.finals[0].winner, "BOS")
        self.assertEqual(len(audit.archive_sha256_by_season), 5)
        summary = audit.summary()
        self.assertFalse(summary["source_is_direct_nhl_gamecenter_evidence"])
        self.assertFalse(summary["independent_final_labels_proven"])
        self.assertFalse(summary["historical_pregame_availability_proven"])
        self.assertFalse(summary["training_permitted"])
        self.assertFalse(summary["prediction_publication_permitted"])
        self.assertEqual(summary["resolved_count"], 475)
        self.assertEqual(len(summary["outcomes_sha256"]), 64)
        self.assertNotIn("finals", summary)

    def test_missing_match_is_rejected(self):
        report, rows = fixture()
        rows[2022].pop()
        with self.assertRaisesRegex(PublicScheduleReconciliationError, "Missing 1"):
            run_audit(report, rows)

    def test_duplicate_match_is_rejected(self):
        report, rows = fixture()
        rows[2022].append(rows[2022][0].copy())
        with self.assertRaisesRegex(PublicScheduleReconciliationError, "Duplicate"):
            run_audit(report, rows)

    def test_non_final_or_wrong_game_type_is_rejected(self):
        for field, value in (("game_state", "LIVE"), ("game_type", "P"),
                             ("season", "2018")):
            with self.subTest(field=field):
                report, rows = fixture()
                rows[2022][0][field] = value
                with self.assertRaisesRegex(PublicScheduleReconciliationError,
                                            "Non-final or wrong-season"):
                    run_audit(report, rows)

    def test_tie_and_missing_score_are_rejected(self):
        for score in ("3", ""):
            with self.subTest(score=score):
                report, rows = fixture()
                rows[2022][0]["away_score"] = score
                with self.assertRaises(PublicScheduleReconciliationError):
                    run_audit(report, rows)

    def test_two_observed_release_season_conventions_are_supported(self):
        report, rows = fixture()
        rows[2022][0]["season"] = "2021"
        self.assertEqual(run_audit(report, rows).resolved_count, 475)

    def test_missing_archive_or_naive_timestamp_is_rejected(self):
        report, rows = fixture()
        del rows[2026]
        with self.assertRaisesRegex(PublicScheduleReconciliationError,
                                    "five release seasons"):
            run_audit(report, rows)
        report, rows = fixture()
        with self.assertRaisesRegex(PublicScheduleReconciliationError,
                                    "timezone-aware"):
            run_audit(report, rows, datetime(2026, 10, 3))

    def test_nhl15_gates_cannot_be_overridden(self):
        report, rows = fixture()
        changed = json.loads(report)
        changed["training_permitted"] = True
        with self.assertRaisesRegex(PublicScheduleReconciliationError,
                                    "gates must remain closed"):
            run_audit(json.dumps(changed).encode(), rows)

    def test_missing_and_malformed_columns_are_rejected(self):
        report, rows = fixture()
        archives = {season: archive_bytes(batch) for season, batch in rows.items()}
        archives[2022] = b"game_id,season\n2021020006,2022\n"
        with self.assertRaisesRegex(PublicScheduleReconciliationError,
                                    "Invalid release columns"):
            reconcile_public_schedules(report, archives, observed_at_utc=OBSERVED)


    def test_checked_in_manifest_matches_nhl15_and_keeps_gates_closed(self):
        from hashlib import sha256

        manifest_path = REPORT.with_name("nhl17_public_schedule_audit_v1.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["nhl15_report_sha256"],
                         sha256(REPORT.read_bytes()).hexdigest())
        self.assertEqual(manifest["resolved_count"], 475)
        self.assertEqual(manifest["resolved_count_by_season"],
                         {"2022": 102, "2023": 95, "2024": 82,
                          "2025": 77, "2026": 119})
        self.assertFalse(manifest["source_is_direct_nhl_gamecenter_evidence"])
        self.assertFalse(manifest["training_permitted"])
        self.assertFalse(manifest["prediction_publication_permitted"])


if __name__ == "__main__":
    unittest.main()
