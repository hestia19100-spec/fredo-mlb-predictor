"""Cohérence du rapport daté issu du CSV MoneyPuck réel."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

REPORT = (Path(__file__).resolve().parents[2]
          / "nhl_protocols/data/nhl15_real_archive_report_2026-10-03.json")


class NHL15RealReportTests(unittest.TestCase):
    def test_coverage_score_partition_and_closed_gates(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["regular_games"], 6560)
        self.assertEqual([item["regular_games"] for item in report["seasons"]],
                         [1312] * 5)
        self.assertEqual(report["feature_candidate_rows"], 13120)
        self.assertTrue(report["regular_coverage_complete"])
        self.assertEqual(report["mirrored_decisive_scores"]
                         + len(report["tied_score_game_ids"])
                         + len(report["inconsistent_score_game_ids"]), 6560)
        self.assertEqual(len(set(report["tied_score_game_ids"])),
                         len(report["tied_score_game_ids"]))
        self.assertEqual(report["attribution"], "Données : MoneyPuck.com")
        for key in ("historical_pregame_availability_proven",
                    "independent_final_labels_proven", "training_permitted",
                    "prediction_publication_permitted"):
            self.assertIs(report[key], False, key)
        for key in ("source_file_sha256", "audit_sha256"):
            self.assertEqual(len(report[key]), 64)


if __name__ == "__main__":
    unittest.main()
