"""Tests fermes du moteur quotidien de prediction LPF Edge."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
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
        self.market_slot = (
            self.project
            / operations.MARKET_SNAPSHOT_ROOT
            / TARGET.isoformat()
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

    def _market_snapshot(self) -> operations.MarketSnapshotPublication:
        return operations.MarketSnapshotPublication(
            target_date=TARGET,
            slot_path=self.market_slot,
            snapshot_path=self.market_slot / "market_snapshot.json",
            completed_path=self.market_slot / "COMPLETED",
            snapshot_sha256="2" * 64,
            odds_run_id=4,
            row_count=15,
            comparable_count=15,
        )

    def test_success_publishes_certification_and_snapshot_together(self) -> None:
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
            mock.patch.object(
                operations,
                "_create_certified_market_snapshot",
                return_value=self._market_snapshot(),
            ) as snapshot,
        ):
            result = self._run()

        self.assertEqual(result.target_date, TARGET)
        self.assertEqual(result.batch_id, BATCH_ID)
        self.assertEqual(result.results_commit, RESULTS_COMMIT)
        self.assertEqual(result.certification_commit, CERTIFICATION_COMMIT)
        self.assertEqual(len(result.results_paths), 8)
        self.assertEqual(len(result.certification_paths), 2)
        self.assertEqual(len(result.market_snapshot_paths), 2)
        self.assertEqual(result.market_snapshot_sha256, "2" * 64)
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
        snapshot.assert_called_once_with(
            TARGET,
            project_directory=self.project.resolve(),
            database_path=self.database,
        )
        self.assertEqual(publish.call_count, 2)
        first = publish.call_args_list[0].kwargs
        second = publish.call_args_list[1].kwargs
        self.assertEqual(first["stage"], operations.DailyPredictionStage.RESULTS_PUBLICATION)
        self.assertEqual(second["stage"], operations.DailyPredictionStage.CERTIFICATION_PUBLICATION)
        self.assertEqual(first["expected_parent_commit"], "e" * 40)
        self.assertEqual(second["expected_parent_commit"], RESULTS_COMMIT)
        self.assertEqual(set(first["expected_paths"]), set(result.results_paths))
        self.assertEqual(
            set(second["expected_paths"]),
            set(result.certification_paths + result.market_snapshot_paths),
        )

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

    def test_market_snapshot_failure_stops_before_certification_commit(self) -> None:
        failure = operations.DailyPredictionAutomationError(
            operations.DailyPredictionStage.MARKET_SNAPSHOT,
            "Journal impossible.",
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
                return_value=RESULTS_COMMIT,
            ) as publish,
            mock.patch.object(
                operations,
                "_certify_prediction",
                return_value=self._certification(),
            ),
            mock.patch.object(
                operations,
                "_create_certified_market_snapshot",
                side_effect=failure,
            ),
        ):
            with self.assertRaises(
                operations.DailyPredictionAutomationError
            ) as raised:
                self._run()

        self.assertIs(raised.exception, failure)
        self.assertEqual(publish.call_count, 1)

    def test_market_snapshot_uses_only_odds_before_certification(self) -> None:
        self.database.parent.mkdir(parents=True)
        self.database.write_bytes(b"database")
        self.certification.parent.mkdir(parents=True)
        self.certification.write_bytes(b"certification\n")
        self.evidence.write_bytes(b"evidence")
        certified_at = datetime(2026, 9, 13, 12, 1, tzinfo=timezone.utc)
        prediction_day = mock.Mock(
            target_date=TARGET,
            certified_at_utc=certified_at,
            predictions=(
                mock.Mock(home_team_id=20, away_team_id=10),
            ),
        )
        display = mock.Mock()
        publication = self._market_snapshot()
        with (
            mock.patch.object(
                operations,
                "load_certified_prediction_day",
                return_value=prediction_day,
            ),
            mock.patch.object(
                operations,
                "load_latest_moneyline_odds_display",
                return_value=display,
            ) as load_odds,
            mock.patch.object(
                operations,
                "load_team_names",
                return_value={10: "Extérieur", 20: "Domicile"},
            ) as load_names,
            mock.patch.object(
                operations,
                "create_market_snapshot_publication",
                return_value=publication,
            ) as create_snapshot,
        ):
            result = operations._create_certified_market_snapshot(
                TARGET,
                project_directory=self.project.resolve(),
                database_path=self.database,
            )

        self.assertIs(result, publication)
        load_odds.assert_called_once_with(
            TARGET,
            database_path=self.database,
            required_region="fr",
            completed_at_or_before_utc=certified_at,
        )
        load_names.assert_called_once_with(
            {10, 20},
            project_directory=self.project.resolve(),
        )
        create_snapshot.assert_called_once_with(
            prediction_day,
            display,
            team_names={10: "Extérieur", 20: "Domicile"},
            certification_sha256=hashlib.sha256(
                b"certification\n"
            ).hexdigest(),
            certification_evidence_sha256=hashlib.sha256(
                b"evidence"
            ).hexdigest(),
            project_directory=self.project.resolve(),
        )

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
