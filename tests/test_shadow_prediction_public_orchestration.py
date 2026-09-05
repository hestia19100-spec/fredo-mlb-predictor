"""Tests de l'orchestration publique ordonnee du mode fantome MLB v2."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import src.shadow_prediction as shadow


TARGET_DATE = "2026-09-05"
STARTED_AT = datetime(2026, 9, 4, 10, 0, 0, tzinfo=timezone.utc)
RESERVED_AT = datetime(2026, 9, 4, 10, 0, 3, tzinfo=timezone.utc)
RUNTIME_COMMIT = "1" * 40
SERVICE_COMMIT = "2" * 40
SERVICE_SHA256 = "3" * 64
MANIFEST_SHA256 = "4" * 64
MANIFEST_COMMIT = "5" * 40
ACTIVATION_SHA256 = "6" * 64
ACTIVATION_COMMIT = "7" * 40
RUNTIME_VERSIONS = tuple(
    sorted(
        {
            "joblib": "1.5.2",
            "numpy": "2.5.2",
            "pandas": "3.0.5",
            "python": "3.12.1",
            "scikit_learn": "1.9.0",
            "scipy": "1.16.2",
        }.items()
    )
)


def _authority(**changes: object) -> shadow.ShadowExecutionAuthority:
    values: dict[str, object] = {
        "runtime_code_commit": RUNTIME_COMMIT,
        "shadow_service_code_commit": SERVICE_COMMIT,
        "shadow_service_module_sha256": SERVICE_SHA256,
        "shadow_protocol_sha256": shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
        "execution_manifest_sha256": MANIFEST_SHA256,
        "execution_manifest_introduction_commit": MANIFEST_COMMIT,
        "activation_sha256": ACTIVATION_SHA256,
        "activation_introduction_commit": ACTIVATION_COMMIT,
        "activation_verified_at_utc": "2026-09-03T09:00:00Z",
        "minimum_target_official_date": "2026-09-03",
        "runtime_versions": RUNTIME_VERSIONS,
    }
    values.update(changes)
    return shadow.ShadowExecutionAuthority(**values)


def _evidence() -> shadow.ShadowActivationReverificationEvidence:
    response = SimpleNamespace(
        status_code=200,
        history=[],
        url=shadow.GITHUB_COMPARE_URL_TEMPLATE.format(
            expected_commit=ACTIVATION_COMMIT
        ),
        content=shadow._canonical_json_bytes(
            {
                "base_commit": {"sha": ACTIVATION_COMMIT},
                "merge_base_commit": {"sha": ACTIVATION_COMMIT},
                "status": "ahead",
            }
        ),
        headers={
            "date": "Fri, 04 Sep 2026 10:00:00 GMT",
            "content-type": "application/json; charset=utf-8",
            "etag": 'W/"orchestration"',
            "x-github-request-id": "ORCH:1234:5678:9ABC:DEF0",
        },
    )
    with (
        patch.object(shadow.requests, "get", return_value=response),
        patch.object(
            shadow,
            "_utc_now",
            return_value=datetime(
                2026,
                9,
                4,
                10,
                0,
                2,
                tzinfo=timezone.utc,
            ),
        ),
    ):
        return shadow.fetch_activation_reverification_evidence(
            ACTIVATION_COMMIT
        )


class ShadowPredictionPublicOrchestrationTests(unittest.TestCase):
    """Le point public est le seul composeur des preuves d'un nouveau slot."""

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project = Path(self.temporary_directory.name)
        self.slot = self.project.joinpath(
            *shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH.parts,
            TARGET_DATE,
        )
        self.authority = _authority()
        self.evidence = _evidence()
        self.reservation = SimpleNamespace(name="reservation")
        self.activation_publication = SimpleNamespace(name="activation")
        self.completion = SimpleNamespace(name="completion")

    def _inspection(
        self,
        state: shadow.ShadowPredictionSlotState,
        receipt: dict[str, object] | None = None,
    ) -> shadow.ShadowPredictionSlotInspection:
        return shadow.ShadowPredictionSlotInspection(
            state=state,
            slot_path=self.slot,
            slot_key="8" * 64,
            batch_id="9" * 64,
            receipt=receipt,
        )

    def _base_patches(self, *, present: bool, state):
        return (
            patch.object(
                shadow,
                "_inspect_shadow_prediction_slot_presence_first",
                return_value=(self.project, self.slot, present),
            ),
            patch.object(
                shadow,
                "verify_shadow_execution_authority",
                return_value=self.authority,
            ),
            patch.object(
                shadow,
                "inspect_shadow_prediction_slot",
                return_value=self._inspection(state),
            ),
        )

    def test_new_slot_runs_every_stage_in_the_only_allowed_order(self) -> None:
        events: list[str] = []
        clock_values = iter((STARTED_AT, RESERVED_AT))

        def first_inspection(*args, **kwargs):
            events.append("FIRST_INSPECTION")
            return self.project, self.slot, False

        def clock():
            name = "STARTED_AT" if not events or events[-1] == "FIRST_INSPECTION" else "RESERVED_AT"
            events.append(name)
            return next(clock_values)

        def authority(*args, **kwargs):
            events.append("AUTHORITY")
            return self.authority

        def second_inspection(*args, **kwargs):
            events.append("SECOND_INSPECTION")
            return self._inspection(shadow.ShadowPredictionSlotState.ABSENT)

        def fetch(*args, **kwargs):
            events.append("REMOTE_REVERIFICATION")
            return self.evidence

        def reserve(*args, **kwargs):
            events.append("RESERVATION")
            return self.reservation

        def publish(*args, **kwargs):
            events.append("ACTIVATION_PUBLICATION")
            return self.activation_publication

        def execute(*args, **kwargs):
            events.append("RESERVED_EXECUTION")
            return self.completion

        with (
            patch.object(
                shadow,
                "_inspect_shadow_prediction_slot_presence_first",
                side_effect=first_inspection,
            ),
            patch.object(shadow, "_utc_now", side_effect=clock),
            patch.object(
                shadow,
                "verify_shadow_execution_authority",
                side_effect=authority,
            ),
            patch.object(
                shadow,
                "inspect_shadow_prediction_slot",
                side_effect=second_inspection,
            ) as inspect_mock,
            patch.object(
                shadow,
                "fetch_activation_reverification_evidence",
                side_effect=fetch,
            ) as fetch_mock,
            patch.object(
                shadow,
                "reserve_shadow_prediction_slot",
                side_effect=reserve,
            ) as reserve_mock,
            patch.object(
                shadow,
                "publish_activation_reverification_evidence",
                side_effect=publish,
            ) as publish_mock,
            patch.object(
                shadow,
                "_execute_reserved_shadow_prediction",
                side_effect=execute,
            ) as execute_mock,
        ):
            result = shadow.execute_shadow_prediction(
                TARGET_DATE,
                project_directory=self.project,
                database_path=self.project / "data" / "fredo_mlb.db",
                data_directory=self.project / "data",
            )

        self.assertIs(result, self.completion)
        self.assertEqual(
            events,
            [
                "FIRST_INSPECTION",
                "STARTED_AT",
                "AUTHORITY",
                "SECOND_INSPECTION",
                "REMOTE_REVERIFICATION",
                "RESERVED_AT",
                "RESERVATION",
                "ACTIVATION_PUBLICATION",
                "RESERVED_EXECUTION",
            ],
        )
        inspect_mock.assert_called_once_with(
            datetime(2026, 9, 5).date(),
            shadow_protocol_sha256=shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
            execution_manifest_sha256=MANIFEST_SHA256,
            model_artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
            project_directory=self.project,
        )
        fetch_mock.assert_called_once_with(ACTIVATION_COMMIT)
        reserve_kwargs = reserve_mock.call_args.kwargs
        self.assertEqual(reserve_kwargs["reserved_at_utc"], "2026-09-04T10:00:03Z")
        self.assertEqual(reserve_kwargs["runtime_code_commit"], RUNTIME_COMMIT)
        self.assertEqual(reserve_kwargs["execution_manifest_sha256"], MANIFEST_SHA256)
        publish_mock.assert_called_once_with(
            self.reservation,
            self.evidence,
            project_directory=self.project,
        )
        context = execute_mock.call_args.args[2]
        self.assertIs(type(context), shadow.ShadowReceiptExecutionContext)
        self.assertEqual(context.started_at_utc, "2026-09-04T10:00:00Z")
        self.assertEqual(context.activation_sha256, ACTIVATION_SHA256)
        self.assertEqual(context.runtime_versions, RUNTIME_VERSIONS)

    def test_exact_duplicate_returns_receipt_without_clock_or_remote(self) -> None:
        receipt = {"existing": "receipt"}
        with (
            patch.object(
                shadow,
                "_inspect_shadow_prediction_slot_presence_first",
                return_value=(self.project, self.slot, True),
            ) as first_mock,
            patch.object(
                shadow,
                "verify_shadow_execution_authority",
                return_value=self.authority,
            ) as authority_mock,
            patch.object(
                shadow,
                "inspect_shadow_prediction_slot",
                return_value=self._inspection(
                    shadow.ShadowPredictionSlotState.COMPLETED_EXACT,
                    receipt,
                ),
            ) as inspect_mock,
            patch.object(
                shadow,
                "_utc_now",
                side_effect=AssertionError("new time forbidden"),
            ),
            patch.object(
                shadow,
                "fetch_activation_reverification_evidence",
                side_effect=AssertionError("network forbidden"),
            ),
            patch.object(
                shadow,
                "reserve_shadow_prediction_slot",
                side_effect=AssertionError("reservation forbidden"),
            ),
            patch.object(
                shadow,
                "_execute_reserved_shadow_prediction",
                side_effect=AssertionError("model path forbidden"),
            ),
        ):
            result = shadow.execute_shadow_prediction(
                TARGET_DATE,
                project_directory=self.project,
            )

        self.assertIs(result, receipt)
        first_mock.assert_called_once()
        authority_mock.assert_called_once()
        inspect_mock.assert_called_once()

    def test_every_nonexact_preexisting_slot_is_terminal(self) -> None:
        states = tuple(
            state
            for state in shadow.ShadowPredictionSlotState
            if state is not shadow.ShadowPredictionSlotState.COMPLETED_EXACT
        )
        for state in states:
            with self.subTest(state=state.value), self._base_patches(
                present=True,
                state=state,
            )[0], self._base_patches(present=True, state=state)[1], patch.object(
                shadow,
                "inspect_shadow_prediction_slot",
                return_value=self._inspection(state),
            ), patch.object(
                shadow,
                "_utc_now",
                side_effect=AssertionError("clock forbidden"),
            ), patch.object(
                shadow,
                "fetch_activation_reverification_evidence",
                side_effect=AssertionError("network forbidden"),
            ):
                with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                    shadow.execute_shadow_prediction(
                        TARGET_DATE,
                        project_directory=self.project,
                    )

    def test_race_after_absent_inspection_stops_before_network(self) -> None:
        with (
            patch.object(
                shadow,
                "_inspect_shadow_prediction_slot_presence_first",
                return_value=(self.project, self.slot, False),
            ),
            patch.object(shadow, "_utc_now", return_value=STARTED_AT),
            patch.object(
                shadow,
                "verify_shadow_execution_authority",
                return_value=self.authority,
            ),
            patch.object(
                shadow,
                "inspect_shadow_prediction_slot",
                return_value=self._inspection(
                    shadow.ShadowPredictionSlotState.INCOMPLETE_CONSUMED
                ),
            ),
            patch.object(
                shadow,
                "fetch_activation_reverification_evidence",
                side_effect=AssertionError("network forbidden"),
            ),
            patch.object(
                shadow,
                "reserve_shadow_prediction_slot",
                side_effect=AssertionError("reservation forbidden"),
            ),
        ):
            with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                shadow.execute_shadow_prediction(
                    TARGET_DATE,
                    project_directory=self.project,
                )

    def test_authority_failure_is_before_remote_and_reservation(self) -> None:
        events: list[str] = []
        with (
            patch.object(
                shadow,
                "_inspect_shadow_prediction_slot_presence_first",
                side_effect=lambda *a, **k: (
                    events.append("inspect")
                    or (self.project, self.slot, False)
                ),
            ),
            patch.object(
                shadow,
                "_utc_now",
                side_effect=lambda: events.append("started") or STARTED_AT,
            ),
            patch.object(
                shadow,
                "verify_shadow_execution_authority",
                side_effect=lambda *a, **k: (
                    events.append("authority")
                    or (_ for _ in ()).throw(shadow.ShadowPredictionError("bad"))
                ),
            ),
            patch.object(
                shadow,
                "fetch_activation_reverification_evidence",
                side_effect=AssertionError("network forbidden"),
            ),
            patch.object(
                shadow,
                "reserve_shadow_prediction_slot",
                side_effect=AssertionError("reservation forbidden"),
            ),
        ):
            with self.assertRaisesRegex(shadow.ShadowPredictionError, "bad"):
                shadow.execute_shadow_prediction(
                    TARGET_DATE,
                    project_directory=self.project,
                )
        self.assertEqual(events, ["inspect", "started", "authority"])

    def test_forged_authority_stops_before_remote_and_reservation(self) -> None:
        forged_values = (
            SimpleNamespace(),
            replace(self.authority, shadow_protocol_sha256="f" * 64),
            replace(
                self.authority,
                activation_verified_at_utc="2026-09-04T10:00:01Z",
            ),
            replace(
                self.authority,
                runtime_versions=tuple(reversed(RUNTIME_VERSIONS)),
            ),
        )
        for forged in forged_values:
            with self.subTest(forged=type(forged).__name__):
                with (
                    patch.object(
                        shadow,
                        "_inspect_shadow_prediction_slot_presence_first",
                        return_value=(self.project, self.slot, False),
                    ),
                    patch.object(
                        shadow,
                        "_utc_now",
                        return_value=STARTED_AT,
                    ),
                    patch.object(
                        shadow,
                        "verify_shadow_execution_authority",
                        return_value=forged,
                    ),
                    patch.object(
                        shadow,
                        "inspect_shadow_prediction_slot",
                        return_value=self._inspection(
                            shadow.ShadowPredictionSlotState.ABSENT
                        ),
                    ),
                    patch.object(
                        shadow,
                        "fetch_activation_reverification_evidence",
                        side_effect=AssertionError("network forbidden"),
                    ),
                    patch.object(
                        shadow,
                        "reserve_shadow_prediction_slot",
                        side_effect=AssertionError("reservation forbidden"),
                    ),
                ):
                    with self.assertRaises(shadow.ShadowPredictionError):
                        shadow.execute_shadow_prediction(
                            TARGET_DATE,
                            project_directory=self.project,
                        )

    def test_remote_failure_creates_no_reservation_or_failed_marker(self) -> None:
        first, authority, second = self._base_patches(
            present=False,
            state=shadow.ShadowPredictionSlotState.ABSENT,
        )
        with (
            first,
            authority,
            second,
            patch.object(shadow, "_utc_now", return_value=STARTED_AT),
            patch.object(
                shadow,
                "fetch_activation_reverification_evidence",
                side_effect=shadow.ShadowPredictionError("offline"),
            ),
            patch.object(
                shadow,
                "reserve_shadow_prediction_slot",
                side_effect=AssertionError("reservation forbidden"),
            ),
            patch.object(
                shadow,
                "fail_shadow_prediction_slot",
                side_effect=AssertionError("FAILED forbidden"),
            ),
        ):
            with self.assertRaisesRegex(shadow.ShadowPredictionError, "offline"):
                shadow.execute_shadow_prediction(
                    TARGET_DATE,
                    project_directory=self.project,
                )

    def test_invalid_remote_time_stops_before_reservation(self) -> None:
        late = replace(
            self.evidence,
            response_received_at_utc="2026-09-04T10:00:04Z",
        )
        first, authority, second = self._base_patches(
            present=False,
            state=shadow.ShadowPredictionSlotState.ABSENT,
        )
        with (
            first,
            authority,
            second,
            patch.object(
                shadow,
                "_utc_now",
                side_effect=(STARTED_AT, RESERVED_AT),
            ),
            patch.object(
                shadow,
                "fetch_activation_reverification_evidence",
                return_value=late,
            ),
            patch.object(
                shadow,
                "reserve_shadow_prediction_slot",
                side_effect=AssertionError("reservation forbidden"),
            ),
            patch.object(
                shadow,
                "fail_shadow_prediction_slot",
                side_effect=AssertionError("FAILED forbidden"),
            ),
        ):
            with self.assertRaises(shadow.ShadowPredictionError):
                shadow.execute_shadow_prediction(
                    TARGET_DATE,
                    project_directory=self.project,
                )

    def test_reservation_conflict_is_not_repaired_or_failed(self) -> None:
        first, authority, second = self._base_patches(
            present=False,
            state=shadow.ShadowPredictionSlotState.ABSENT,
        )
        with (
            first,
            authority,
            second,
            patch.object(
                shadow,
                "_utc_now",
                side_effect=(STARTED_AT, RESERVED_AT),
            ),
            patch.object(
                shadow,
                "fetch_activation_reverification_evidence",
                return_value=self.evidence,
            ),
            patch.object(
                shadow,
                "reserve_shadow_prediction_slot",
                side_effect=shadow.ShadowPredictionSlotConsumedError("lost"),
            ),
            patch.object(
                shadow,
                "fail_shadow_prediction_slot",
                side_effect=AssertionError("FAILED forbidden"),
            ),
        ):
            with self.assertRaisesRegex(
                shadow.ShadowPredictionSlotConsumedError,
                "lost",
            ):
                shadow.execute_shadow_prediction(
                    TARGET_DATE,
                    project_directory=self.project,
                )

    def test_concurrent_public_executions_have_exactly_one_winner(self) -> None:
        result_root = self.project.joinpath(
            *shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH.parts
        )
        result_root.mkdir(parents=True)
        first_barrier = threading.Barrier(2)
        detailed_barrier = threading.Barrier(2)
        remote_barrier = threading.Barrier(2)
        local_clock = threading.local()
        original_first = shadow._inspect_shadow_prediction_slot_presence_first
        original_detailed = shadow.inspect_shadow_prediction_slot

        def first(*args, **kwargs):
            first_barrier.wait(timeout=10)
            return original_first(*args, **kwargs)

        def detailed(*args, **kwargs):
            detailed_barrier.wait(timeout=10)
            return original_detailed(*args, **kwargs)

        def clock():
            count = getattr(local_clock, "count", 0)
            local_clock.count = count + 1
            return STARTED_AT if count == 0 else RESERVED_AT

        def remote(*args, **kwargs):
            remote_barrier.wait(timeout=10)
            return self.evidence

        def worker() -> str:
            try:
                result = shadow.execute_shadow_prediction(
                    TARGET_DATE,
                    project_directory=self.project,
                )
            except shadow.ShadowPredictionSlotConsumedError:
                return "consumed"
            self.assertIs(result, self.completion)
            return "completed"

        with (
            patch.object(
                shadow,
                "_inspect_shadow_prediction_slot_presence_first",
                side_effect=first,
            ),
            patch.object(shadow, "_utc_now", side_effect=clock),
            patch.object(
                shadow,
                "verify_shadow_execution_authority",
                return_value=self.authority,
            ),
            patch.object(
                shadow,
                "inspect_shadow_prediction_slot",
                side_effect=detailed,
            ),
            patch.object(
                shadow,
                "fetch_activation_reverification_evidence",
                side_effect=remote,
            ),
            patch.object(
                shadow,
                "_execute_reserved_shadow_prediction",
                return_value=self.completion,
            ) as execute_mock,
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(lambda _: worker(), range(2)))

        self.assertCountEqual(outcomes, ["completed", "consumed"])
        execute_mock.assert_called_once()
        self.assertTrue((self.slot / "RESERVED").is_file())
        self.assertTrue(
            (self.slot / shadow.ACTIVATION_REVERIFICATION_FILENAME).is_file()
        )

    def test_activation_publication_failure_closes_reserved_slot(self) -> None:
        publication_error = shadow.ShadowPredictionError("cannot publish")
        first, authority, second = self._base_patches(
            present=False,
            state=shadow.ShadowPredictionSlotState.ABSENT,
        )
        with (
            first,
            authority,
            second,
            patch.object(
                shadow,
                "_utc_now",
                side_effect=(STARTED_AT, RESERVED_AT),
            ),
            patch.object(
                shadow,
                "fetch_activation_reverification_evidence",
                return_value=self.evidence,
            ),
            patch.object(
                shadow,
                "reserve_shadow_prediction_slot",
                return_value=self.reservation,
            ),
            patch.object(
                shadow,
                "publish_activation_reverification_evidence",
                side_effect=publication_error,
            ),
            patch.object(
                shadow,
                "_fail_after_public_reservation",
            ) as fail_mock,
            patch.object(
                shadow,
                "_execute_reserved_shadow_prediction",
                side_effect=AssertionError("execution forbidden"),
            ),
        ):
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "cannot publish",
            ):
                shadow.execute_shadow_prediction(
                    TARGET_DATE,
                    project_directory=self.project,
                )
        fail_mock.assert_called_once_with(
            self.reservation,
            stage=shadow._EXECUTION_STAGE_ACTIVATION_PUBLICATION,
            error=publication_error,
            project_directory=self.project,
        )

    def test_invalid_target_stops_before_result_tree_access(self) -> None:
        for value in ("2025-09-05", "2026-02-30", datetime.now(timezone.utc)):
            with self.subTest(value=value), patch.object(
                shadow,
                "_inspect_shadow_prediction_slot_presence_first",
                side_effect=AssertionError("slot access forbidden"),
            ):
                with self.assertRaises(shadow.ShadowPredictionError):
                    shadow.execute_shadow_prediction(
                        value,
                        project_directory=self.project,
                    )

    def test_first_presence_inspection_uses_only_lstat(self) -> None:
        result_root = self.project.joinpath(
            *shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH.parts
        )
        with patch.object(
            Path,
            "read_bytes",
            side_effect=AssertionError("file read forbidden"),
        ):
            project, slot, present = (
                shadow._inspect_shadow_prediction_slot_presence_first(
                    datetime(2026, 9, 5).date(),
                    project_directory=self.project,
                )
            )
            self.assertEqual(project, self.project)
            self.assertEqual(slot, self.slot)
            self.assertFalse(present)
            result_root.mkdir(parents=True)
            self.slot.mkdir()
            _, _, present = (
                shadow._inspect_shadow_prediction_slot_presence_first(
                    datetime(2026, 9, 5).date(),
                    project_directory=self.project,
                )
            )
            self.assertTrue(present)

    def test_public_signature_has_no_authority_or_clock_override(self) -> None:
        parameters = inspect.signature(
            shadow.execute_shadow_prediction
        ).parameters
        self.assertEqual(
            tuple(parameters),
            (
                "target_official_date",
                "project_directory",
                "database_path",
                "data_directory",
            ),
        )
        forbidden = {
            "runtime_code_commit",
            "shadow_protocol_sha256",
            "execution_manifest_sha256",
            "model_artifact_sha256",
            "started_at_utc",
            "reserved_at_utc",
            "authority",
            "evidence",
            "retry_policy",
            "sleep",
            "clock",
            "model",
        }
        self.assertTrue(forbidden.isdisjoint(parameters))


if __name__ == "__main__":
    unittest.main()
