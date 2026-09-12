"""Tests fermes du moteur quotidien de prediction LPF Edge."""

from __future__ import annotations

from datetime import date, datetime, timezone
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import lpf_edge_daily_operations as operations


TARGET = date(2026, 9, 13)
NOW = datetime(2026, 9, 13, 8, 0, tzinfo=timezone.utc)
RESULTS_COMMIT = "a" * 40
CERTIFICATION_COMMIT = "b" * 40
BATCH_ID = "c" * 64


class LPFEdgeDailyPredictionExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.database = self.project / "data/fredo_mlb.db"
        self.data_directory = self.project / "data"
        self.slot = (
            self.project
            / operations.PREDICTION_ROOT
            / TARGET.isoformat()
        )
        self.certification = (
            self.project
            / operations.CERTIFICATION_ROOT
            / f"{TARGET.isoformat()}.json"
        )
        self.evidence = (
            self.project
            / operations.CERTIFICATION_ROOT
            / f"{TARGET.isoformat()}.remote.json.gz"
        )

    def _action(
        self,
        state: operations.DailyActionState = operations.DailyActionState.READY,
        message: str = "Prêt.",
    ) -> operations.DailyAction:
        return operations.DailyAction(
            key="predictions",
            label="Créer les prédictions du jour",
            state=state,
            message=message,
            can_execute=state is operations.DailyActionState.READY,
        )

    def _overview(self, action: operations.DailyAction | None = None) -> mock.Mock:
        return mock.Mock(
            prediction_action=action or self._action(),
            git=mock.Mock(head_commit="e" * 40),
        )

    def _prediction(self) -> operations._PredictionEngineOutcome:
        return operations._PredictionEngineOutcome(
            slot_path=self.slot,
            batch_id=BATCH_ID,
            batch_status="COMPLETED_WITH_PREDICTIONS",
            receipt_sha256="d" * 64,
        )

    def _certification(self, **changes: object) -> operations._CertificationOutcome:
        values: dict[str, object] = {
            "batch_id": BATCH_ID,
            "results_commit": RESULTS_COMMIT,
            "certification_path": self.certification,
            "evidence_path": self.evidence,
        }
        values.update(changes)
        return operations._CertificationOutcome(**values)

    def _run(self) -> operations.DailyPredictionPublication:
        return operations.execute_daily_prediction_publication(
            TARGET,
            project_directory=self.project,
            database_path=self.database,
            data_directory=self.data_directory,
        )

    def test_success_executes_exact_five_stage_chain(self) -> None:
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._overview(),
            ) as preflight,
            mock.patch.object(
                operations,
                "_execute_prediction_engine",
                return_value=self._prediction(),
            ) as prediction,
            mock.patch.object(
                operations,
                "_commit_and_push_exact_paths",
                side_effect=(RESULTS_COMMIT, CERTIFICATION_COMMIT),
            ) as publish,
            mock.patch.object(
                operations,
                "_certify_prediction",
                return_value=self._certification(),
            ) as certify,
        ):
            result = self._run()

        self.assertEqual(result.target_date, TARGET)
        self.assertEqual(result.batch_id, BATCH_ID)
        self.assertEqual(result.results_commit, RESULTS_COMMIT)
        self.assertEqual(result.certification_commit, CERTIFICATION_COMMIT)
        self.assertEqual(len(result.results_paths), 8)
        self.assertEqual(len(result.certification_paths), 2)
        preflight.assert_called_once_with(
            TARGET,
            now_utc=NOW,
            project_directory=self.project.resolve(),
            database_path=self.database,
        )
        prediction.assert_called_once_with(
            TARGET,
            project_directory=self.project.resolve(),
            database_path=self.database,
            data_directory=self.data_directory,
        )
        certify.assert_called_once_with(
            TARGET,
            RESULTS_COMMIT,
            project_directory=self.project.resolve(),
        )
        self.assertEqual(publish.call_count, 2)
        first = publish.call_args_list[0].kwargs
        second = publish.call_args_list[1].kwargs
        self.assertEqual(first["stage"], operations.DailyPredictionStage.RESULTS_PUBLICATION)
        self.assertEqual(second["stage"], operations.DailyPredictionStage.CERTIFICATION_PUBLICATION)
        self.assertEqual(first["expected_parent_commit"], "e" * 40)
        self.assertEqual(second["expected_parent_commit"], RESULTS_COMMIT)
        self.assertEqual(set(first["expected_paths"]), set(result.results_paths))
        self.assertEqual(set(second["expected_paths"]), set(result.certification_paths))

    def test_preflight_block_stops_before_prediction_and_git(self) -> None:
        blocked = self._action(
            operations.DailyActionState.TOO_LATE,
            "Le délai obligatoire est dépassé.",
        )
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._overview(blocked),
            ),
            mock.patch.object(operations, "_execute_prediction_engine") as prediction,
            mock.patch.object(operations, "_commit_and_push_exact_paths") as publish,
            mock.patch.object(operations, "_certify_prediction") as certify,
        ):
            with self.assertRaisesRegex(
                operations.DailyPredictionAutomationError,
                "délai obligatoire",
            ) as caught:
                self._run()
        self.assertEqual(caught.exception.stage, operations.DailyPredictionStage.PREFLIGHT)
        prediction.assert_not_called()
        publish.assert_not_called()
        certify.assert_not_called()

    def test_prediction_failure_stops_all_later_stages(self) -> None:
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._overview(),
            ),
            mock.patch.object(
                operations,
                "_execute_prediction_engine",
                side_effect=RuntimeError("échec simulé"),
            ),
            mock.patch.object(operations, "_commit_and_push_exact_paths") as publish,
            mock.patch.object(operations, "_certify_prediction") as certify,
        ):
            with self.assertRaises(operations.DailyPredictionAutomationError) as caught:
                self._run()
        self.assertEqual(caught.exception.stage, operations.DailyPredictionStage.PREDICTION)
        publish.assert_not_called()
        certify.assert_not_called()

    def test_results_publication_failure_never_starts_certification(self) -> None:
        failure = operations.DailyPredictionAutomationError(
            operations.DailyPredictionStage.RESULTS_PUBLICATION,
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
                "_execute_prediction_engine",
                return_value=self._prediction(),
            ),
            mock.patch.object(
                operations,
                "_commit_and_push_exact_paths",
                side_effect=failure,
            ),
            mock.patch.object(operations, "_certify_prediction") as certify,
        ):
            with self.assertRaises(operations.DailyPredictionAutomationError) as caught:
                self._run()
        self.assertIs(caught.exception, failure)
        certify.assert_not_called()

    def test_certification_failure_stops_before_second_commit(self) -> None:
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._overview(),
            ),
            mock.patch.object(
                operations,
                "_execute_prediction_engine",
                return_value=self._prediction(),
            ),
            mock.patch.object(
                operations,
                "_commit_and_push_exact_paths",
                return_value=RESULTS_COMMIT,
            ) as publish,
            mock.patch.object(
                operations,
                "_certify_prediction",
                side_effect=RuntimeError("preuve distante absente"),
            ),
        ):
            with self.assertRaises(operations.DailyPredictionAutomationError) as caught:
                self._run()
        self.assertEqual(caught.exception.stage, operations.DailyPredictionStage.CERTIFICATION)
        self.assertEqual(publish.call_count, 1)

    def test_foreign_certification_is_rejected_before_second_commit(self) -> None:
        foreign = self._certification(batch_id="f" * 64)
        with (
            mock.patch.object(operations, "_utc_now", return_value=NOW),
            mock.patch.object(
                operations,
                "inspect_daily_operations",
                return_value=self._overview(),
            ),
            mock.patch.object(
                operations,
                "_execute_prediction_engine",
                return_value=self._prediction(),
            ),
            mock.patch.object(
                operations,
                "_commit_and_push_exact_paths",
                return_value=RESULTS_COMMIT,
            ) as publish,
            mock.patch.object(
                operations,
                "_certify_prediction",
                return_value=foreign,
            ),
        ):
            with self.assertRaisesRegex(
                operations.DailyPredictionAutomationError,
                "ne correspond pas au lot",
            ):
                self._run()
        self.assertEqual(publish.call_count, 1)

    def test_git_publication_uses_exact_paths_commit_and_push(self) -> None:
        expected = ("proofs/a", "proofs/b")
        listing = "\n".join(expected)
        outputs = iter(
            (
                "main", RESULTS_COMMIT, RESULTS_COMMIT,
                listing, "", "",
                "",
                "", "", listing,
                "", "", RESULTS_COMMIT,
                "", RESULTS_COMMIT, "",
            )
        )
        with mock.patch.object(
            operations,
            "_run_git_mutation",
            side_effect=lambda *_args, **_kwargs: next(outputs),
        ) as run_git:
            commit = operations._commit_and_push_exact_paths(
                project_directory=self.project,
                expected_paths=expected,
                expected_parent_commit=RESULTS_COMMIT,
                commit_message="Commit contrôlé",
                stage=operations.DailyPredictionStage.RESULTS_PUBLICATION,
            )
        self.assertEqual(commit, RESULTS_COMMIT)
        self.assertEqual(run_git.call_count, 16)
        commands = [call.args[1] for call in run_git.call_args_list]
        self.assertIn(("add", "--", *expected), commands)
        self.assertIn(("commit", "-m", "Commit contrôlé"), commands)
        self.assertIn(("push", "origin", "main"), commands)

    def test_unexpected_git_path_blocks_before_add(self) -> None:
        outputs = iter(
            (
                "main", RESULTS_COMMIT, RESULTS_COMMIT,
                "proofs/a\nforeign.txt", "", "",
            )
        )
        with mock.patch.object(
            operations,
            "_run_git_mutation",
            side_effect=lambda *_args, **_kwargs: next(outputs),
        ) as run_git:
            with self.assertRaisesRegex(
                operations.DailyPredictionAutomationError,
                "ne correspondent pas exactement",
            ):
                operations._commit_and_push_exact_paths(
                    project_directory=self.project,
                    expected_paths=("proofs/a",),
                    expected_parent_commit=RESULTS_COMMIT,
                    commit_message="Interdit",
                    stage=operations.DailyPredictionStage.RESULTS_PUBLICATION,
                )
        self.assertEqual(run_git.call_count, 6)

    def test_changed_git_base_blocks_before_staging(self) -> None:
        outputs = iter(("main", "f" * 40, RESULTS_COMMIT))
        with mock.patch.object(
            operations,
            "_run_git_mutation",
            side_effect=lambda *_args, **_kwargs: next(outputs),
        ) as run_git:
            with self.assertRaisesRegex(
                operations.DailyPredictionAutomationError,
                "base Git a change",
            ):
                operations._commit_and_push_exact_paths(
                    project_directory=self.project,
                    expected_paths=("proofs/a",),
                    expected_parent_commit=RESULTS_COMMIT,
                    commit_message="Interdit",
                    stage=operations.DailyPredictionStage.RESULTS_PUBLICATION,
                )
        self.assertEqual(run_git.call_count, 3)

    def test_public_signature_exposes_no_clock_engine_or_git_override(self) -> None:
        signature = inspect.signature(operations.execute_daily_prediction_publication)
        self.assertEqual(
            tuple(signature.parameters),
            ("target_date", "project_directory", "database_path", "data_directory"),
        )
        forbidden = {"clock", "now", "git", "engine", "certification", "session"}
        self.assertTrue(forbidden.isdisjoint(signature.parameters))


if __name__ == "__main__":
    unittest.main()
