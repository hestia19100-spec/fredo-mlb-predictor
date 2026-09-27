"""Le complément se publie après le contrôle officiel, sans second scoring."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src import lpf_edge_daily_operations as operations
from src.lpf_edge_postponement_reconciliation import ReconciliationPublication
from tests.test_lpf_edge_postponement_reconciliation import (
    prediction_day, score_summary,
)


class ReconciliationDailyOperationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)

    def test_publishes_only_new_complementary_files(self):
        day = prediction_day()
        target = self.project / "outcome_reconciliations" / "lpf_edge_mlb_postponements_v1" / "2026-09-22" / "2026-09-28"
        publication = ReconciliationPublication(
            target,
            (target / "schedule_response.json.gz", target / "reconciliation.json", target / "COMPLETED"),
            (),
        )
        git = operations.GitWorkspaceState(
            True, "main", "a" * 40, "a" * 40, True, True,
        )
        with (
            patch.object(operations, "_utc_now", return_value=datetime(2026, 9, 28, 8, tzinfo=timezone.utc)),
            patch.object(operations, "inspect_git_workspace", return_value=git),
            patch.object(operations, "list_certified_prediction_dates", return_value=[day.target_date]),
            patch.object(operations, "load_certified_prediction_day", return_value=day),
            patch.object(operations, "load_latest_score_summary", return_value=score_summary()),
            patch("src.lpf_edge_postponement_reconciliation.load_resolutions", return_value={}),
            patch("src.lpf_edge_postponement_reconciliation.create_reconciliation", return_value=publication) as create,
            patch.object(operations, "_commit_and_push_exact_paths", return_value="b" * 40) as publish,
        ):
            result = operations.execute_postponement_reconciliation(
                project_directory=self.project,
            )
        self.assertEqual(("b" * 40, 0), result)
        create.assert_called_once()
        self.assertEqual(3, len(publish.call_args.kwargs["expected_paths"]))
        self.assertEqual("a" * 40, publish.call_args.kwargs["expected_parent_commit"])

    def test_known_resolution_prevents_another_collection(self):
        day = prediction_day()
        git = operations.GitWorkspaceState(
            True, "main", "a" * 40, "a" * 40, True, True,
        )
        with (
            patch.object(operations, "_utc_now", return_value=datetime(2026, 9, 28, 8, tzinfo=timezone.utc)),
            patch.object(operations, "inspect_git_workspace", return_value=git),
            patch.object(operations, "list_certified_prediction_dates", return_value=[day.target_date]),
            patch.object(operations, "load_certified_prediction_day", return_value=day),
            patch.object(operations, "load_latest_score_summary", return_value=score_summary()),
            patch("src.lpf_edge_postponement_reconciliation.load_resolutions", return_value={824785: object()}),
            patch("src.lpf_edge_postponement_reconciliation.create_reconciliation") as create,
        ):
            result = operations.execute_postponement_reconciliation(
                project_directory=self.project,
            )
        self.assertIsNone(result)
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
