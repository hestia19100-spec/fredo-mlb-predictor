"""The public NHL dashboard only exposes coherent aggregate coverage."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.nhl.public_dashboard import load_nhl_coverage


class NHLCoverageTests(unittest.TestCase):
    def test_repository_report_is_non_predictive(self) -> None:
        coverage = load_nhl_coverage()
        self.assertEqual(len(coverage.seasons), 5)
        self.assertEqual(coverage.regular_games, 6560)
        self.assertEqual(coverage.full_lookback_games, 6128)

    def test_inconsistent_total_fails_closed(self) -> None:
        from src.nhl.public_dashboard import REPORT_PATH

        payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        payload["regular_games"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_nhl_coverage(path)

    def test_changed_publication_status_requires_review(self) -> None:
        from src.nhl.public_dashboard import REPORT_PATH

        payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        payload["prediction_publication_permitted"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_nhl_coverage(path)


if __name__ == "__main__":
    unittest.main()
