"""Tests fermes du moteur quotidien de resultats LPF Edge."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import lpf_edge_daily_operations as operations
from src import shadow_scoring


TODAY = date(2026, 9, 13)
TARGET = date(2026, 9, 11)
NOW = datetime(2026, 9, 13, 8, 0, tzinfo=timezone.utc)
RESULTS_COMMIT = "a" * 40
OBSERVATION_ID = "b" * 64


class LPFEdgeDailyResultsExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.slot = (
            self.project
            / operations.SCORING_ROOT
            / TARGET.isoformat()
            / "observations"
            / TODAY.isoformat()
        )

    def _action(
        self,
        state: operations.DailyActionState = operations.DailyActionState.READY,
        message: str = "Pret.",
    ) -> operations.DailyAction:
        return operations.DailyAction(
            key="results",
            label="Recuperer et verifier les resultats",
            state=state,
            message=message,
            can_execute=state is operations.DailyActionState.READY,
        )

    def _overview(self, action: operations.DailyAction | None = None) -> mock.Mock:
        return mock.Mock(
            results_action=action or self._action(),
            results_target_date=TARGET,
            git=mock.Mock(head_commit="e" * 40),
            scoring_runtime_preparation_required=False,
        )

    def _scoring(
        self,
        *,
        slot_path: Path | None = None,
        outcome: str = "COMPLETED",
        filenames: tuple[str, ...] = operations.SCORING_SUCCESS_FILENAMES,
    ) -> operations._ScoringEngineOutcome:
        return operations._ScoringEngineOutcome(
            slot_path=slot_path or self.slot,
            observation_id=OBSERVATION_ID,
            outcome=outcome,
            expected_filenames=filenames,
        )

    def _run(self) -> operations.DailyResultsPublication:
        return operations.execute_daily_results_publication(
            TODAY,
            project_directory=self.project,
        )

    def test_success_executes_exact_three_stage_chain(self) -> None:
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._overview(),
            ) as preflight,
            mock.patch.object(
                operations,
                "_execute_scoring_engine",
                return_value=self._scoring(),
            ) as scoring,
            mock.patch.object(
                operations,
                "_commit_and_push_exact_paths",
                return_value=RESULTS_COMMIT,
            ) as publish,
        ):
            result = self._run()

        self.assertEqual(result.target_date, TARGET)
        self.assertEqual(result.checkpoint_date, TODAY)
        self.assertEqual(result.observation_id, OBSERVATION_ID)
        self.assertEqual(result.outcome, "COMPLETED")
        self.assertEqual(result.results_commit, RESULTS_COMMIT)
        self.assertEqual(len(result.results_paths), 6)
        preflight.assert_called_once_with(
            TODAY,
            now_utc=NOW,
            project_directory=self.project.resolve(),
        )
        scoring.assert_called_once_with(
            TARGET,
            TODAY,
            project_directory=self.project.resolve(),
        )
        arguments = publish.call_args.kwargs
        self.assertEqual(arguments["expected_parent_commit"], "e" * 40)
        self.assertEqual(arguments["stage"], operations.DailyResultsStage.RESULTS_PUBLICATION)
        self.assertIs(arguments["error_type"], operations.DailyResultsAutomationError)
        self.assertEqual(set(arguments["expected_paths"]), set(result.results_paths))

    def test_certification_head_is_advanced_by_empty_runtime_commit(self) -> None:
        overview = self._overview()
        overview.scoring_runtime_preparation_required = True
        runtime_commit = "f" * 40
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=overview,
            ),
            mock.patch.object(
                operations,
                "_prepare_scoring_runtime_if_needed",
                return_value=runtime_commit,
            ) as prepare,
            mock.patch.object(
                operations,
                "_execute_scoring_engine",
                return_value=self._scoring(),
            ),
            mock.patch.object(
                operations,
                "_commit_and_push_exact_paths",
                return_value=RESULTS_COMMIT,
            ) as publish,
        ):
            self._run()

        prepare.assert_called_once_with(
            project_directory=self.project.resolve(),
            expected_parent_commit="e" * 40,
            scoring_target=TARGET,
            required=True,
        )
        self.assertEqual(
            publish.call_args.kwargs["expected_parent_commit"],
            runtime_commit,
        )

    def test_runtime_preparation_is_noop_when_head_is_already_later(self) -> None:
        with mock.patch.object(operations, "_run_git_mutation") as run_git:
            result = operations._prepare_scoring_runtime_if_needed(
                project_directory=self.project,
                expected_parent_commit="e" * 40,
                scoring_target=TARGET,
                required=False,
            )
        self.assertEqual(result, "e" * 40)
        run_git.assert_not_called()

    def test_runtime_preparation_creates_and_pushes_exact_empty_commit(self) -> None:
        runtime_commit = "f" * 40
        outputs = iter(
            (
                "main",
                "e" * 40,
                "e" * 40,
                "",
                "",
                runtime_commit,
                "",
                "",
                runtime_commit,
                "",
            )
        )
        with mock.patch.object(
            operations,
            "_run_git_mutation",
            side_effect=lambda *args, **kwargs: next(outputs),
        ) as run_git:
            result = operations._prepare_scoring_runtime_if_needed(
                project_directory=self.project,
                expected_parent_commit="e" * 40,
                scoring_target=TARGET,
                required=True,
            )

        self.assertEqual(result, runtime_commit)
        arguments = [call.args[1] for call in run_git.call_args_list]
        self.assertIn(
            (
                "commit",
                "--allow-empty",
                "-m",
                "Preparer le runtime scoring MLB du 11 septembre 2026",
            ),
            arguments,
        )
        self.assertIn(("push", "origin", "main"), arguments)
        self.assertIn(
            (
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                runtime_commit,
            ),
            arguments,
        )

    def test_runtime_preparation_failure_stops_before_scoring(self) -> None:
        overview = self._overview()
        overview.scoring_runtime_preparation_required = True
        failure = operations.DailyResultsAutomationError(
            operations.DailyResultsStage.RUNTIME_PREPARATION,
            "push du runtime impossible",
        )
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=overview,
            ),
            mock.patch.object(
                operations,
                "_prepare_scoring_runtime_if_needed",
                side_effect=failure,
            ),
            mock.patch.object(operations, "_execute_scoring_engine") as scoring,
        ):
            with self.assertRaises(operations.DailyResultsAutomationError) as caught:
                self._run()
        self.assertIs(caught.exception, failure)
        scoring.assert_not_called()

    def test_runtime_requirement_is_idempotent_after_empty_commit(self) -> None:
        prospective_commit = "e" * 40
        with mock.patch.object(
            operations,
            "_run_git",
            return_value=prospective_commit,
        ):
            self.assertTrue(
                operations._scoring_runtime_preparation_is_required(
                    TARGET,
                    head_commit=prospective_commit,
                    project_directory=self.project,
                )
            )
            self.assertFalse(
                operations._scoring_runtime_preparation_is_required(
                    TARGET,
                    head_commit="f" * 40,
                    project_directory=self.project,
                )
            )

    def test_final_results_and_market_verdict_are_published_atomically(self) -> None:
        settlement_slot = (
            self.project
            / operations.MARKET_SETTLEMENT_ROOT
            / TARGET.isoformat()
        )
        settlement = operations.MarketSettlementPublication(
            target_date=TARGET,
            slot_path=settlement_slot,
            settlement_path=settlement_slot / "market_evaluation.json",
            completed_path=settlement_slot / "COMPLETED",
            settlement_sha256="c" * 64,
            snapshot_sha256="d" * 64,
            evaluated_count=1,
            correct_count=1,
            missing_market_count=0,
            void_count=0,
            theoretical_net_units=Decimal("0.8"),
            theoretical_roi_percent=Decimal("80"),
        )
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._overview(),
            ),
            mock.patch.object(
                operations,
                "_execute_scoring_engine",
                return_value=self._scoring(),
            ),
            mock.patch.object(
                operations,
                "_create_market_settlement_if_ready",
                return_value=settlement,
            ) as seal,
            mock.patch.object(
                operations,
                "_commit_and_push_exact_paths",
                return_value=RESULTS_COMMIT,
            ) as publish,
        ):
            publication = self._run()

        seal.assert_called_once_with(
            TARGET,
            observation_id=OBSERVATION_ID,
            publication_parent_commit="e" * 40,
            project_directory=self.project.resolve(),
        )
        self.assertEqual(publication.market_settlement_sha256, "c" * 64)
        self.assertEqual(len(publication.market_settlement_paths), 2)
        expected = set(publication.results_paths) | set(
            publication.market_settlement_paths
        )
        self.assertEqual(set(publish.call_args.kwargs["expected_paths"]), expected)

    def test_pending_or_absent_journal_keeps_original_results_publication(self) -> None:
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._overview(),
            ),
            mock.patch.object(
                operations,
                "_execute_scoring_engine",
                return_value=self._scoring(),
            ),
            mock.patch.object(
                operations,
                "_create_market_settlement_if_ready",
                return_value=None,
            ),
            mock.patch.object(
                operations,
                "_commit_and_push_exact_paths",
                return_value=RESULTS_COMMIT,
            ) as publish,
        ):
            publication = self._run()

        self.assertEqual(publication.market_settlement_paths, ())
        self.assertIsNone(publication.market_settlement_sha256)
        self.assertEqual(
            set(publish.call_args.kwargs["expected_paths"]),
            set(publication.results_paths),
        )

    def test_controlled_mlb_failure_is_published_as_two_exact_proofs(self) -> None:
        failed = self._scoring(
            outcome="FAILED",
            filenames=operations.SCORING_FAILURE_FILENAMES,
        )
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._overview(),
            ),
            mock.patch.object(
                operations,
                "_execute_scoring_engine",
                return_value=failed,
            ),
            mock.patch.object(
                operations,
                "_commit_and_push_exact_paths",
                return_value=RESULTS_COMMIT,
            ),
        ):
            result = self._run()
        self.assertEqual(result.outcome, "FAILED")
        self.assertEqual(len(result.results_paths), 2)
        self.assertTrue(result.results_paths[0].endswith("/FAILED.json"))
        self.assertTrue(result.results_paths[1].endswith("/RESERVED"))

    def test_preflight_block_stops_before_scoring_and_git(self) -> None:
        blocked = self._action(
            operations.DailyActionState.TOO_EARLY,
            "Le controle ouvre plus tard.",
        )
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._overview(blocked),
            ),
            mock.patch.object(operations, "_execute_scoring_engine") as scoring,
            mock.patch.object(operations, "_commit_and_push_exact_paths") as publish,
        ):
            with self.assertRaisesRegex(
                operations.DailyResultsAutomationError,
                "ouvre plus tard",
            ) as caught:
                self._run()
        self.assertEqual(caught.exception.stage, operations.DailyResultsStage.PREFLIGHT)
        scoring.assert_not_called()
        publish.assert_not_called()

    def test_scoring_failure_stops_before_git(self) -> None:
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._overview(),
            ),
            mock.patch.object(
                operations,
                "_execute_scoring_engine",
                side_effect=RuntimeError("echec simule"),
            ),
            mock.patch.object(operations, "_commit_and_push_exact_paths") as publish,
        ):
            with self.assertRaises(operations.DailyResultsAutomationError) as caught:
                self._run()
        self.assertEqual(caught.exception.stage, operations.DailyResultsStage.SCORING)
        publish.assert_not_called()

    def test_publication_failure_is_not_hidden(self) -> None:
        failure = operations.DailyResultsAutomationError(
            operations.DailyResultsStage.RESULTS_PUBLICATION,
            "push impossible",
        )
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._overview(),
            ),
            mock.patch.object(
                operations,
                "_execute_scoring_engine",
                return_value=self._scoring(),
            ),
            mock.patch.object(
                operations,
                "_commit_and_push_exact_paths",
                side_effect=failure,
            ),
        ):
            with self.assertRaises(operations.DailyResultsAutomationError) as caught:
                self._run()
        self.assertIs(caught.exception, failure)

    def test_foreign_slot_is_rejected_before_publication(self) -> None:
        foreign = self._scoring(slot_path=self.project / "foreign")
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._overview(),
            ),
            mock.patch.object(
                operations,
                "_execute_scoring_engine",
                return_value=foreign,
            ),
            mock.patch.object(operations, "_commit_and_push_exact_paths") as publish,
        ):
            with self.assertRaisesRegex(
                operations.DailyResultsAutomationError,
                "dossier de scoring inattendu",
            ):
                self._run()
        publish.assert_not_called()

    def test_forged_file_list_is_rejected_before_publication(self) -> None:
        forged = self._scoring(filenames=("COMPLETED", "intrus.txt"))
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._overview(),
            ),
            mock.patch.object(
                operations,
                "_execute_scoring_engine",
                return_value=forged,
            ),
            mock.patch.object(operations, "_commit_and_push_exact_paths") as publish,
        ):
            with self.assertRaisesRegex(
                operations.DailyResultsAutomationError,
                "liste de preuves inattendue",
            ):
                self._run()
        publish.assert_not_called()

    def test_scoring_adapter_accepts_only_exact_completion_type(self) -> None:
        completion = mock.Mock(
            observation_id=OBSERVATION_ID,
            target_official_date=TARGET.isoformat(),
            checkpoint_utc_date=TODAY.isoformat(),
        )
        publication = shadow_scoring.ScoringObservationCompletionPublication(
            receipt_publication=mock.Mock(),
            completion=completion,
            completed_path=self.slot / "COMPLETED",
            completed_sha256="c" * 64,
            completed_size_bytes=1,
        )
        with mock.patch.object(
            shadow_scoring,
            "execute_scoring_observation",
            return_value=publication,
        ) as execute:
            result = operations._execute_scoring_engine(
                TARGET,
                TODAY,
                project_directory=self.project,
            )
        execute.assert_called_once_with(
            TARGET,
            TODAY,
            project_directory=self.project,
        )
        self.assertEqual(result.outcome, "COMPLETED")
        self.assertEqual(result.expected_filenames, operations.SCORING_SUCCESS_FILENAMES)

    def test_scoring_adapter_preserves_controlled_failure(self) -> None:
        reservation = mock.Mock(
            slot_path=self.slot,
            observation_id=OBSERVATION_ID,
        )
        failure = shadow_scoring.ScoringObservationFailure(
            reservation=reservation,
            failed_path=self.slot / "FAILED.json",
            failed_marker={},
            failed_marker_sha256="d" * 64,
            stage="OUTCOME_REQUEST",
            error_type="ScoringOutcomeRequestError",
        )
        with mock.patch.object(
            shadow_scoring,
            "execute_scoring_observation",
            return_value=failure,
        ):
            result = operations._execute_scoring_engine(
                TARGET,
                TODAY,
                project_directory=self.project,
            )
        self.assertEqual(result.outcome, "FAILED")
        self.assertEqual(result.expected_filenames, operations.SCORING_FAILURE_FILENAMES)

    def test_scoring_adapter_rejects_foreign_terminal_path(self) -> None:
        completion = mock.Mock(
            observation_id=OBSERVATION_ID,
            target_official_date=TARGET.isoformat(),
            checkpoint_utc_date=TODAY.isoformat(),
        )
        publication = shadow_scoring.ScoringObservationCompletionPublication(
            receipt_publication=mock.Mock(),
            completion=completion,
            completed_path=self.slot / "FOREIGN",
            completed_sha256="c" * 64,
            completed_size_bytes=1,
        )
        with mock.patch.object(
            shadow_scoring,
            "execute_scoring_observation",
            return_value=publication,
        ):
            with self.assertRaisesRegex(
                operations.DailyResultsAutomationError,
                "preuve de completion incoherente",
            ):
                operations._execute_scoring_engine(
                    TARGET,
                    TODAY,
                    project_directory=self.project,
                )

    def test_foreign_engine_result_is_rejected(self) -> None:
        with mock.patch.object(
            shadow_scoring,
            "execute_scoring_observation",
            return_value=mock.Mock(),
        ):
            with self.assertRaisesRegex(
                operations.DailyResultsAutomationError,
                "fermeture de scoring exacte",
            ):
                operations._execute_scoring_engine(
                    TARGET,
                    TODAY,
                    project_directory=self.project,
                )

    def test_public_signature_exposes_no_clock_engine_git_or_network_override(self) -> None:
        signature = inspect.signature(operations.execute_daily_results_publication)
        self.assertEqual(
            tuple(signature.parameters),
            ("target_date", "project_directory"),
        )
        forbidden = {"clock", "now", "git", "engine", "network", "session"}
        self.assertTrue(forbidden.isdisjoint(signature.parameters))


if __name__ == "__main__":
    unittest.main()
