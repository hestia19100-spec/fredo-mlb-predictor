"""NHL-41: a failed source request cannot become a false no-games report."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import date, datetime, timezone
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from src.nhl import manual_prospective_day as module
from src import nhl_manual_prospective_day_cli as cli
from src.nhl.current_season_import import NHLCurrentSeasonImportError
from src.nhl.public_schedule_capture import PublicScheduleCaptureError


class ManualProspectiveDayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.when = datetime(2026, 10, 9, 11, 54, tzinfo=timezone.utc)
        self.target = date(2026, 10, 9)
        self.team = SimpleNamespace(
            path=Path("team-slot"), response_sha256="a" * 64,
            observed_at_utc=self.when,
        )
        self.imported = SimpleNamespace(path=Path("import-slot"), regular_game_count=65)
        self.schedule = SimpleNamespace(
            path=Path("schedule-slot"), response_sha256="b" * 64,
            game_count=2,
        )
        self.receipt = {
            "target_date": self.target.isoformat(),
            "response_sha256": self.schedule.response_sha256,
            "future_regular_games": [{"game_id": 1}, {"game_id": 2}],
        }
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.capture = self.stack.enter_context(patch.object(
            module, "capture_team_history", return_value=self.team,
        ))
        self.verify_team = self.stack.enter_context(patch.object(
            module, "verify_team_capture", return_value=self.team,
        ))
        self.import_season = self.stack.enter_context(patch.object(
            module, "import_current_season_capture", return_value=self.imported,
        ))
        self.verify_import = self.stack.enter_context(patch.object(
            module, "verify_current_season_import", return_value=self.imported,
        ))
        self.capture_schedule = self.stack.enter_context(patch.object(
            module, "capture_public_schedule", return_value=self.schedule,
        ))
        self.verify_schedule = self.stack.enter_context(patch.object(
            module, "verify_public_schedule_capture", return_value=self.receipt,
        ))

    def run_day(self) -> dict[str, object]:
        return module.run_manual_prospective_day(
            self.target, season=2026, code_commit="c" * 40,
            explicit_manual_run=True,
            team_transport=lambda *_args, **_kwargs: None,
            schedule_transport=lambda *_args, **_kwargs: None,
            now=lambda: self.when,
        )

    def test_valid_manual_day_preserves_both_sources_without_model_rights(self) -> None:
        report = self.run_day()
        self.assertEqual(report["team_capture_status"], "VERIFIED")
        self.assertEqual(report["current_season_import_status"], "VERIFIED_DESCRIPTIVE_ONLY")
        self.assertEqual(report["current_season_regular_games"], 65)
        self.assertEqual(report["schedule_status"], "FUTURE_REGULAR_GAMES_VERIFIED")
        self.assertEqual(report["status"], "CAPTURES_VERIFIED")
        self.assertEqual(report["future_regular_games"], 2)
        self.assertFalse(report["training_permitted"])
        self.assertFalse(report["prediction_publication_permitted"])
        self.capture.assert_called_once()
        self.capture_schedule.assert_called_once()

    def test_http_403_is_unknown_not_no_games(self) -> None:
        cause = HTTPError("https://api-web.nhle.com/v1/schedule/2026-10-09", 403,
                          "Forbidden", {}, None)
        error = PublicScheduleCaptureError("Collecte NHL échouée.")
        error.__cause__ = cause
        self.capture_schedule.side_effect = error
        report = self.run_day()
        self.assertEqual(report["team_capture_status"], "VERIFIED")
        self.assertEqual(report["current_season_import_status"], "VERIFIED_DESCRIPTIVE_ONLY")
        self.assertEqual(report["schedule_status"], "NOT_VERIFIED")
        self.assertEqual(report["schedule_http_status"], 403)
        self.assertEqual(report["status"], "PARTIAL_CAPTURE")
        self.assertIsNone(report["future_regular_games"])
        self.verify_schedule.assert_not_called()

    def test_zero_future_games_does_not_claim_no_games_today(self) -> None:
        self.schedule.game_count = 0
        self.receipt["future_regular_games"] = []
        report = self.run_day()
        self.assertEqual(report["schedule_status"], "NO_FUTURE_REGULAR_GAMES_VERIFIED")
        self.assertEqual(report["future_regular_games"], 0)
        self.assertNotIn("NO_GAMES_TODAY", report.values())

    def test_import_failure_keeps_capture_and_attempts_schedule(self) -> None:
        self.import_season.side_effect = NHLCurrentSeasonImportError(
            "Aucun match régulier de saison courante.",
        )
        report = self.run_day()
        self.assertEqual(report["team_capture_status"], "VERIFIED")
        self.assertEqual(report["current_season_import_status"], "NO_COMPLETED_REGULAR_GAMES_YET")
        self.assertEqual(report["schedule_status"], "FUTURE_REGULAR_GAMES_VERIFIED")
        self.assertEqual(report["status"], "PARTIAL_CAPTURE")
        self.verify_import.assert_not_called()
        self.capture_schedule.assert_called_once()

    def test_other_import_failure_is_not_disguised_as_opening_day(self) -> None:
        self.import_season.side_effect = NHLCurrentSeasonImportError("CSV corrompu.")
        report = self.run_day()
        self.assertEqual(report["current_season_import_status"], "IMPORT_REJECTED")
        self.assertEqual(report["status"], "PARTIAL_CAPTURE")

    def test_invalid_schedule_receipt_is_not_accepted(self) -> None:
        self.receipt["target_date"] = "2026-10-10"
        with self.assertRaises(module.NHLManualProspectiveDayError):
            self.run_day()

    def test_manual_flag_and_season_are_checked_before_network(self) -> None:
        for target, season, manual in (
            (self.target, 2026, False),
            (self.target, 2025, True),
            (date(2027, 2, 1), 2027, True),
        ):
            with self.subTest(target=target, season=season, manual=manual):
                with self.assertRaises(module.NHLManualProspectiveDayError):
                    module.run_manual_prospective_day(
                        target, season=season, code_commit="c" * 40,
                        explicit_manual_run=manual,
                        team_transport=lambda *_args, **_kwargs: None,
                        schedule_transport=lambda *_args, **_kwargs: None,
                    )
        self.capture.assert_not_called()
        self.capture_schedule.assert_not_called()

    def test_policy_rejects_automatic_collection(self) -> None:
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            altered = json.loads(module.POLICY.read_text(encoding="utf-8"))
            altered["automatic_daily_collection_allowed"] = True
            policy = Path(directory) / "policy.json"
            policy.write_text(json.dumps(altered), encoding="utf-8")
            with patch.object(module, "POLICY", policy):
                with self.assertRaises(module.NHLManualProspectiveDayError):
                    self.run_day()
        self.capture.assert_not_called()

    def test_cli_returns_nonzero_for_missing_schedule(self) -> None:
        with patch.object(cli.subprocess, "check_output", return_value="c" * 40), \
             patch.object(cli, "run_manual_prospective_day", return_value={
                 "status": "PARTIAL_CAPTURE", "schedule_status": "NOT_VERIFIED",
                 "future_regular_games": None,
             }), patch("sys.stdout", new_callable=StringIO) as output:
            code = cli.main(["--date", "2026-10-09", "--season", "2026", "--manual"])
        self.assertEqual(code, 1)
        self.assertIsNone(json.loads(output.getvalue())["future_regular_games"])


if __name__ == "__main__":
    unittest.main()
