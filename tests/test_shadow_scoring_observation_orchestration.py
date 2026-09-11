"""Tests de l'orchestration complete d'une observation de scoring MLB."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, timezone
import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import shadow_prediction as shadow
from src import shadow_scoring as scoring
from src import shadow_scoring_registration as registration


TARGET = date(2026, 9, 10)
CHECKPOINT = date(2026, 9, 11)
STARTED = datetime(2026, 9, 11, 6, 1, 0, tzinfo=timezone.utc)
RESERVED = datetime(2026, 9, 11, 6, 1, 1, tzinfo=timezone.utc)
FINALIZED = datetime(2026, 9, 11, 6, 1, 3, tzinfo=timezone.utc)
COMPLETED = datetime(2026, 9, 11, 6, 1, 4, tzinfo=timezone.utc)


class SyntheticStageError(RuntimeError):
    """Erreur inattendue injectee apres la reservation."""


class ShadowScoringObservationOrchestrationTests(unittest.TestCase):
    """L'orchestrateur relie les primitives sans ouvrir de raccourci."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name).resolve()
        self.output_root = self.project.joinpath(
            *scoring.SCORING_OUTPUT_ROOT_RELATIVE_PATH.parts
        )
        self.output_root.mkdir(parents=True)
        (self.output_root / ".gitkeep").write_text("\n", encoding="utf-8")
        self.authority = scoring.ScoringExecutionAuthority(
            protocol={},
            registration={},
            scoring_protocol_sha256=(
                registration.EXPECTED_SCORING_PROTOCOL_SHA256
            ),
            protocol_introduction_commit=(
                scoring.EXPECTED_PROTOCOL_INTRODUCTION_COMMIT
            ),
            registration_commit=scoring.EXPECTED_REGISTRATION_COMMIT,
            registration_sha256=scoring.EXPECTED_REGISTRATION_SHA256,
            registration_remote_evidence_sha256=(
                scoring.EXPECTED_REGISTRATION_REMOTE_EVIDENCE_SHA256
            ),
            runtime_code_commit="a" * 40,
            scoring_engine_sha256="b" * 64,
        )
        self.prediction = scoring.CertifiedScoringPrediction(
            prediction_id="c" * 64,
            batch_id="d" * 64,
            game_id=123,
            occurrence_key="e" * 64,
            season=2026,
            target_official_date=TARGET.isoformat(),
            away_team_id=10,
            home_team_id=20,
            p_home_win="0.60000000000000009",
            p_away_win="0.39999999999999991",
        )
        self.source = self._source(
            scoring.ScoringPredictionSourceStatus.CERTIFIED_NONEMPTY
        )
        observation_id = scoring.build_observation_id(
            scoring_protocol_sha256=(
                registration.EXPECTED_SCORING_PROTOCOL_SHA256
            ),
            target_official_date=TARGET,
            checkpoint_utc_date=CHECKPOINT,
        )
        self.slot = (
            self.output_root
            / TARGET.isoformat()
            / "observations"
            / CHECKPOINT.isoformat()
        )
        marker = {
            "marker_schema_version": 1,
            "protocol_id": registration.EXPECTED_PROTOCOL_ID,
            "scoring_protocol_sha256": (
                registration.EXPECTED_SCORING_PROTOCOL_SHA256
            ),
            "target_official_date": TARGET.isoformat(),
            "checkpoint_utc_date": CHECKPOINT.isoformat(),
            "observation_id": observation_id,
            "reserved_at_utc": "2026-09-11T06:01:01Z",
            "runtime_code_commit": self.authority.runtime_code_commit,
        }
        marker_bytes = shadow._canonical_json_file_bytes(marker)
        self.reservation = scoring.ScoringObservationReservation(
            slot_path=self.slot,
            target_official_date=TARGET.isoformat(),
            checkpoint_utc_date=CHECKPOINT.isoformat(),
            observation_id=observation_id,
            reserved_marker=marker,
            reserved_marker_sha256=hashlib.sha256(marker_bytes).hexdigest(),
        )
        self.evidence = scoring.ScoringOutcomeObservationEvidence(
            target_official_date=TARGET.isoformat(),
            checkpoint_utc_date=CHECKPOINT.isoformat(),
            outcome_http_date_utc="2026-09-11T06:01:02Z",
            response_received_at_utc="2026-09-11T06:01:02Z",
            response_body_sha256="f" * 64,
            flattened_occurrence_count=1,
            raw_evidence={},
            canonical_json_bytes=b"evidence\n",
            canonical_gzip_bytes=b"gzip",
            canonical_gzip_sha256="1" * 64,
        )
        self.evidence_publication = scoring.ScoringOutcomeEvidencePublication(
            reservation=self.reservation,
            evidence_path=self.slot / scoring.OUTCOME_EVIDENCE_FILENAME,
            evidence_sha256="1" * 64,
            evidence_size_bytes=4,
            outcome_http_date_utc=self.evidence.outcome_http_date_utc,
            response_received_at_utc=self.evidence.response_received_at_utc,
            response_body_sha256=self.evidence.response_body_sha256,
            flattened_occurrence_count=1,
            evidence=self.evidence,
        )

    def _source(
        self,
        status: scoring.ScoringPredictionSourceStatus,
    ) -> scoring.ImmutableScoringPredictionSource:
        certified = (
            status is scoring.ScoringPredictionSourceStatus.CERTIFIED_NONEMPTY
        )
        return scoring.ImmutableScoringPredictionSource(
            target_official_date=TARGET.isoformat(),
            status=status,
            results_commit="2" * 40 if certified else None,
            certification_commit="3" * 40 if certified else None,
            batch_id="d" * 64 if certified else None,
            predictions=(self.prediction,) if certified else (),
            predictions_sha256="4" * 64 if certified else None,
            receipt_sha256="5" * 64 if certified else None,
            certification_sha256="6" * 64 if certified else None,
            raw_certification_evidence_sha256="7" * 64 if certified else None,
            status_reason="VALID" if certified else "NOT_CERTIFIED",
        )

    @contextmanager
    def _pipeline(
        self,
        *,
        source: scoring.ImmutableScoringPredictionSource | None = None,
        capture_result: object | None = None,
        fail_stage: str | None = None,
    ):
        calls: list[str] = []
        source_result = self.source if source is None else source
        capture_value = (
            self.evidence_publication
            if capture_result is None
            else capture_result
        )
        batch = object()
        documents = object()
        receipt = object()
        completion = object()

        def action(name: str, result: object):
            def invoke(*_args, **_kwargs):
                calls.append(name)
                if name == fail_stage:
                    raise SyntheticStageError(f"echec {name}")
                return result

            return invoke

        initial = scoring.ScoringObservationSlotInspection(
            state=scoring.ScoringObservationSlotState.ABSENT,
            slot_path=self.slot,
            target_official_date=TARGET.isoformat(),
            checkpoint_utc_date=CHECKPOINT.isoformat(),
            observation_id=self.reservation.observation_id,
        )
        with ExitStack() as stack:
            mocks = {
                "inspect": stack.enter_context(mock.patch.object(
                    scoring,
                    "inspect_scoring_observation_slot_presence_first",
                    side_effect=action("inspect", initial),
                )),
                "authority": stack.enter_context(mock.patch.object(
                    scoring,
                    "verify_scoring_execution_authority",
                    side_effect=action("authority", self.authority),
                )),
                "source": stack.enter_context(mock.patch.object(
                    scoring,
                    "load_immutable_scoring_prediction_source",
                    side_effect=action("source", source_result),
                )),
                "reserve": stack.enter_context(mock.patch.object(
                    scoring,
                    "reserve_scoring_observation_slot",
                    side_effect=action("reserve", self.reservation),
                )),
                "capture": stack.enter_context(mock.patch.object(
                    scoring,
                    "capture_and_publish_scoring_outcome_evidence",
                    side_effect=action("capture", capture_value),
                )),
                "adjudicate": stack.enter_context(mock.patch.object(
                    scoring,
                    "adjudicate_scoring_predictions",
                    side_effect=action("adjudicate", batch),
                )),
                "documents": stack.enter_context(mock.patch.object(
                    scoring,
                    "publish_scoring_observation_documents",
                    side_effect=action("documents", documents),
                )),
                "receipt": stack.enter_context(mock.patch.object(
                    scoring,
                    "publish_scoring_observation_receipt",
                    side_effect=action("receipt", receipt),
                )),
                "complete": stack.enter_context(mock.patch.object(
                    scoring,
                    "publish_scoring_observation_completion",
                    side_effect=action("complete", completion),
                )),
                "clock": stack.enter_context(mock.patch.object(
                    scoring,
                    "_utc_now",
                    side_effect=[STARTED, RESERVED, FINALIZED, COMPLETED],
                )),
            }
            yield calls, mocks, batch, documents, receipt, completion

    def _execute(self):
        return scoring.execute_scoring_observation(
            TARGET,
            CHECKPOINT,
            project_directory=self.project,
        )

    def test_success_path_calls_every_primitive_once_in_exact_order(self) -> None:
        with self._pipeline() as (calls, _mocks, *_results):
            result = self._execute()
        self.assertIs(result, _results[-1])
        self.assertEqual(
            calls,
            [
                "inspect",
                "authority",
                "source",
                "reserve",
                "capture",
                "adjudicate",
                "documents",
                "receipt",
                "complete",
            ],
        )

    def test_exact_arguments_link_every_stage(self) -> None:
        with self._pipeline() as (
            _calls,
            mocks,
            batch,
            documents,
            receipt,
            _completion,
        ):
            self._execute()
        mocks["inspect"].assert_called_once_with(
            TARGET, CHECKPOINT, project_directory=self.project
        )
        mocks["authority"].assert_called_once_with(
            project_directory=self.project
        )
        mocks["source"].assert_called_once_with(
            self.authority, TARGET, project_directory=self.project
        )
        mocks["reserve"].assert_called_once_with(
            self.authority,
            TARGET,
            CHECKPOINT,
            reserved_at_utc="2026-09-11T06:01:01Z",
            project_directory=self.project,
        )
        mocks["capture"].assert_called_once_with(
            self.reservation, project_directory=self.project
        )
        mocks["adjudicate"].assert_called_once_with(
            self.source.predictions, self.evidence
        )
        mocks["documents"].assert_called_once_with(
            self.authority,
            self.reservation,
            self.evidence_publication,
            self.source,
            batch,
            project_directory=self.project,
        )
        mocks["receipt"].assert_called_once_with(
            self.authority,
            self.reservation,
            self.evidence_publication,
            documents,
            started_at_utc=STARTED,
            receipt_finalized_at_utc=FINALIZED,
            project_directory=self.project,
        )
        mocks["complete"].assert_called_once_with(
            self.authority,
            receipt,
            completed_at_utc=COMPLETED,
            project_directory=self.project,
        )

    def test_controlled_capture_failure_is_terminal(self) -> None:
        failure = scoring.ScoringObservationFailure(
            reservation=self.reservation,
            failed_path=self.slot / scoring.FAILED_FILENAME,
            failed_marker={},
            failed_marker_sha256="8" * 64,
            stage="OUTCOME_REQUEST",
            error_type="ScoringOutcomeRequestError",
        )
        with self._pipeline(capture_result=failure) as (calls, mocks, *_):
            result = self._execute()
        self.assertIs(result, failure)
        self.assertEqual(
            calls,
            ["inspect", "authority", "source", "reserve", "capture"],
        )
        mocks["adjudicate"].assert_not_called()
        self.assertEqual(mocks["clock"].call_count, 2)

    def test_consumed_slot_stops_before_clock_git_source_or_mlb(self) -> None:
        self.slot.mkdir(parents=True)
        for name in (
            "_utc_now",
            "verify_scoring_execution_authority",
            "load_immutable_scoring_prediction_source",
            "capture_and_publish_scoring_outcome_evidence",
        ):
            with self.subTest(name=name), mock.patch.object(
                scoring, name, side_effect=AssertionError(name)
            ) as protected:
                with self.assertRaises(shadow.ShadowPredictionSlotConsumedError):
                    self._execute()
                protected.assert_not_called()

    def test_noncertified_source_stops_before_reservation_and_mlb(self) -> None:
        invalid = self._source(scoring.ScoringPredictionSourceStatus.LOCAL_ONLY)
        with self._pipeline(source=invalid) as (calls, mocks, *_):
            with self.assertRaisesRegex(
                scoring.ScoringAdjudicationError,
                "certifiee avec predictions",
            ):
                self._execute()
        self.assertEqual(calls, ["inspect", "authority", "source"])
        mocks["reserve"].assert_not_called()
        mocks["capture"].assert_not_called()
        self.assertEqual(mocks["clock"].call_count, 1)

    def test_authority_or_source_error_stops_before_reservation(self) -> None:
        for stage, expected_calls in (
            ("authority", ["inspect", "authority"]),
            ("source", ["inspect", "authority", "source"]),
        ):
            with self.subTest(stage=stage):
                with self._pipeline(fail_stage=stage) as (calls, mocks, *_):
                    with self.assertRaisesRegex(
                        SyntheticStageError, f"echec {stage}"
                    ):
                        self._execute()
                self.assertEqual(calls, expected_calls)
                mocks["reserve"].assert_not_called()
                mocks["capture"].assert_not_called()

    def test_backward_clock_is_rejected_before_reservation(self) -> None:
        with self._pipeline() as (calls, mocks, *_):
            mocks["clock"].side_effect = [
                STARTED,
                datetime(2026, 9, 11, 6, 0, 59, tzinfo=timezone.utc),
            ]
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "horloge a recule",
            ):
                self._execute()
        self.assertEqual(calls, ["inspect", "authority", "source"])
        mocks["reserve"].assert_not_called()
        mocks["capture"].assert_not_called()

    def test_nonexact_capture_result_stops_before_adjudication(self) -> None:
        with self._pipeline(capture_result=object()) as (calls, mocks, *_):
            with self.assertRaisesRegex(
                scoring.ScoringAdjudicationError,
                "aucune publication exacte",
            ):
                self._execute()
        self.assertEqual(
            calls,
            ["inspect", "authority", "source", "reserve", "capture"],
        )
        mocks["adjudicate"].assert_not_called()

    def test_every_post_reservation_error_stops_later_publications(self) -> None:
        stages = ("capture", "adjudicate", "documents", "receipt", "complete")
        order = [
            "inspect",
            "authority",
            "source",
            "reserve",
            "capture",
            "adjudicate",
            "documents",
            "receipt",
            "complete",
        ]
        for stage in stages:
            with self.subTest(stage=stage):
                with self._pipeline(fail_stage=stage) as (calls, _mocks, *_):
                    with self.assertRaisesRegex(
                        SyntheticStageError, f"echec {stage}"
                    ):
                        self._execute()
                self.assertEqual(calls, order[: order.index(stage) + 1])

    def test_post_evidence_error_never_attempts_failed_or_retry(self) -> None:
        with (
            self._pipeline(fail_stage="documents") as (_calls, _mocks, *_),
            mock.patch.object(
                scoring,
                "fail_scoring_observation_slot",
                side_effect=AssertionError("FAILED interdit"),
            ) as failed,
        ):
            with self.assertRaises(SyntheticStageError):
                self._execute()
        failed.assert_not_called()

    def test_keyboard_interrupt_is_never_hidden(self) -> None:
        with self._pipeline() as (_calls, mocks, *_):
            mocks["capture"].side_effect = KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                self._execute()
        mocks["adjudicate"].assert_not_called()

    def test_atomic_concurrency_has_exactly_one_winner(self) -> None:
        now = datetime(2026, 9, 11, 6, 2, 0, tzinfo=timezone.utc)

        def controlled_failure(reservation, *, project_directory):
            return scoring.fail_scoring_observation_slot(
                reservation,
                failed_at_utc="2026-09-11T06:02:00Z",
                stage="OUTCOME_REQUEST",
                error=scoring.ScoringOutcomeRequestError("MLB indisponible"),
                project_directory=project_directory,
            )

        def execute():
            try:
                return self._execute()
            except shadow.ShadowPredictionSlotConsumedError as error:
                return error

        with (
            mock.patch.object(
                scoring,
                "verify_scoring_execution_authority",
                return_value=self.authority,
            ),
            mock.patch.object(
                scoring,
                "load_immutable_scoring_prediction_source",
                return_value=self.source,
            ),
            mock.patch.object(scoring, "_utc_now", return_value=now),
            mock.patch.object(
                scoring,
                "capture_and_publish_scoring_outcome_evidence",
                side_effect=controlled_failure,
            ),
            mock.patch.object(
                scoring,
                "adjudicate_scoring_predictions",
                side_effect=AssertionError("adjudication interdite"),
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = list(executor.map(lambda _index: execute(), range(2)))

        self.assertEqual(
            sum(type(item) is scoring.ScoringObservationFailure for item in results),
            1,
        )
        self.assertEqual(
            sum(isinstance(item, shadow.ShadowPredictionSlotConsumedError) for item in results),
            1,
        )
        self.assertEqual(
            sorted(path.name for path in self.slot.iterdir()),
            [scoring.FAILED_FILENAME, "RESERVED"],
        )

    def test_public_signature_exposes_no_clock_network_or_model_override(self) -> None:
        parameters = tuple(
            inspect.signature(scoring.execute_scoring_observation).parameters
        )
        self.assertEqual(
            parameters,
            (
                "target_official_date",
                "checkpoint_utc_date",
                "project_directory",
            ),
        )
        forbidden = {
            "clock",
            "request",
            "response",
            "model",
            "predictor",
            "database_path",
            "retry",
        }
        self.assertFalse(forbidden & set(parameters))

    def test_orchestrator_never_calls_sqlite_model_or_prediction(self) -> None:
        protected_names = (
            "_load_frozen_shadow_model",
            "_predict_frozen_shadow_model_once",
            "execute_shadow_prediction",
        )
        with ExitStack() as stack:
            protected = [
                stack.enter_context(mock.patch.object(
                    shadow,
                    name,
                    side_effect=AssertionError(name),
                ))
                for name in protected_names
            ]
            with self._pipeline():
                self._execute()
        for operation in protected:
            operation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
