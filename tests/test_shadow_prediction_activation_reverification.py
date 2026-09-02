"""Tests de la preuve GitHub par slot fantome MLB v2."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import requests

import src.shadow_prediction as shadow_prediction
from src.shadow_prediction import (
    ACTIVATION_REVERIFICATION_FILENAME,
    EXPECTED_MODEL_ARTIFACT_SHA256,
    EXPECTED_SHADOW_PROTOCOL_SHA256,
    GITHUB_COMPARE_URL_TEMPLATE,
    GITHUB_REMOTE_REF,
    GITHUB_REQUEST_HEADERS,
    GITHUB_REQUEST_TIMEOUT_SECONDS,
    SHADOW_RESULT_ROOT_RELATIVE_PATH,
    ShadowActivationReverificationEvidence,
    ShadowPredictionError,
    ShadowPredictionSlotConsumedError,
    fetch_activation_reverification_evidence,
    publish_activation_reverification_evidence,
    reserve_shadow_prediction_slot,
)


PROTOCOL_SHA256 = EXPECTED_SHADOW_PROTOCOL_SHA256
EXECUTION_MANIFEST_SHA256 = "2" * 64
MODEL_ARTIFACT_SHA256 = EXPECTED_MODEL_ARTIFACT_SHA256
RUNTIME_CODE_COMMIT = "4" * 40
ACTIVATION_COMMIT = "5" * 40
TARGET_DATE = "2026-09-03"
EVIDENCE_HTTP_DATE = "Tue, 01 Sep 2026 18:00:00 GMT"
EVIDENCE_HTTP_DATE_UTC = "2026-09-01T18:00:00Z"
EVIDENCE_RECEIVED_AT_UTC = "2026-09-01T18:00:02Z"
RESERVED_AT_UTC = "2026-09-02T18:00:00Z"


def _canonical_json_file_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _compare_body(
    *,
    base_commit: str = ACTIVATION_COMMIT,
    merge_base_commit: str = ACTIVATION_COMMIT,
    status: str = "ahead",
) -> bytes:
    return json.dumps(
        {
            "base_commit": {"sha": base_commit},
            "merge_base_commit": {"sha": merge_base_commit},
            "status": status,
            "ahead_by": 3,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: object = 200,
        history: object = None,
        url: object = None,
        content: object = None,
        headers: object = None,
    ) -> None:
        request_url = GITHUB_COMPARE_URL_TEMPLATE.format(
            expected_commit=ACTIVATION_COMMIT
        )
        self.status_code = status_code
        self.history = [] if history is None else history
        self.url = request_url if url is None else url
        self.content = _compare_body() if content is None else content
        self.headers = (
            {
                "date": EVIDENCE_HTTP_DATE,
                "content-type": "application/json; charset=utf-8",
                "etag": 'W/"evidence-etag"',
                "x-github-request-id": "ABC0:1234:5678:9ABC:DEF0",
            }
            if headers is None
            else headers
        )


def _process_publication_worker(
    reservation,
    evidence,
    project_directory: str,
    start_gate,
    result_queue,
    index: int,
) -> None:
    """Tente la meme publication depuis un processus independant."""
    try:
        start_gate.wait(timeout=15)
        publish_activation_reverification_evidence(
            reservation,
            evidence,
            project_directory=Path(project_directory),
        )
    except ShadowPredictionSlotConsumedError:
        result_queue.put(("consumed", index))
    except BaseException as error:
        result_queue.put(("error", type(error).__name__, str(error)))
    else:
        result_queue.put(("published", index))


class ShadowActivationReverificationTests(unittest.TestCase):
    """La preuve distante doit preceder puis suivre RESERVED exactement."""

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project = Path(self.temporary_directory.name)
        self.result_root = self.project.joinpath(
            *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts
        )
        self.result_root.mkdir(parents=True)
        self.slot = self.result_root / TARGET_DATE
        self.evidence = self._fetch()
        self.reservation = self._reserve()

    def _fetch(
        self,
        *,
        response: _FakeResponse | None = None,
        now: datetime | None = None,
    ) -> ShadowActivationReverificationEvidence:
        effective_response = response or _FakeResponse()
        effective_now = now or datetime(
            2026,
            9,
            1,
            18,
            0,
            2,
            900000,
            tzinfo=timezone.utc,
        )
        with (
            patch.object(
                shadow_prediction.requests,
                "get",
                return_value=effective_response,
            ),
            patch.object(
                shadow_prediction,
                "_utc_now",
                return_value=effective_now,
            ),
        ):
            return fetch_activation_reverification_evidence(
                ACTIVATION_COMMIT
            )

    def _reserve(self, *, project: Path | None = None):
        target_project = project if project is not None else self.project
        return reserve_shadow_prediction_slot(
            TARGET_DATE,
            reserved_at_utc=RESERVED_AT_UTC,
            runtime_code_commit=RUNTIME_CODE_COMMIT,
            shadow_protocol_sha256=PROTOCOL_SHA256,
            execution_manifest_sha256=EXECUTION_MANIFEST_SHA256,
            model_artifact_sha256=MODEL_ARTIFACT_SHA256,
            project_directory=target_project,
        )

    def _publish(self, **overrides):
        arguments: dict[str, object] = {
            "reservation": self.reservation,
            "evidence": self.evidence,
            "project_directory": self.project,
        }
        arguments.update(overrides)
        return publish_activation_reverification_evidence(**arguments)

    def test_fetch_builds_exact_canonical_evidence_before_reservation(
        self,
    ) -> None:
        response = _FakeResponse()
        reserved_before = (self.slot / "RESERVED").read_bytes()
        request_url = GITHUB_COMPARE_URL_TEMPLATE.format(
            expected_commit=ACTIVATION_COMMIT
        )
        with (
            patch.object(
                shadow_prediction.requests,
                "get",
                return_value=response,
            ) as get_mock,
            patch.object(
                shadow_prediction,
                "_utc_now",
                return_value=datetime(
                    2026,
                    9,
                    1,
                    18,
                    0,
                    2,
                    987654,
                    tzinfo=timezone.utc,
                ),
            ),
        ):
            evidence = fetch_activation_reverification_evidence(
                ACTIVATION_COMMIT
            )

        get_mock.assert_called_once_with(
            request_url,
            headers=dict(GITHUB_REQUEST_HEADERS),
            timeout=GITHUB_REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
            verify=True,
        )
        expected_body_sha256 = hashlib.sha256(response.content).hexdigest()
        expected_raw = {
            "evidence_schema_version": 1,
            "request_url": request_url,
            "request_method": "GET",
            "application_request_headers": dict(GITHUB_REQUEST_HEADERS),
            "effective_url": request_url,
            "response_status_code": 200,
            "response_redirect_count": 0,
            "selected_response_headers": dict(response.headers),
            "response_received_at_utc": EVIDENCE_RECEIVED_AT_UTC,
            "response_body_base64": base64.b64encode(
                response.content
            ).decode("ascii"),
            "response_body_sha256": expected_body_sha256,
        }
        expected_json = _canonical_json_file_bytes(expected_raw)
        self.assertEqual(evidence.activation_introduction_commit, ACTIVATION_COMMIT)
        self.assertEqual(evidence.activation_remote_ref, GITHUB_REMOTE_REF)
        self.assertEqual(
            evidence.activation_remote_reverified_at_utc,
            EVIDENCE_HTTP_DATE_UTC,
        )
        self.assertEqual(
            evidence.response_received_at_utc,
            EVIDENCE_RECEIVED_AT_UTC,
        )
        self.assertEqual(evidence.response_body_sha256, expected_body_sha256)
        self.assertEqual(evidence.raw_evidence, expected_raw)
        self.assertEqual(evidence.canonical_json_bytes, expected_json)
        self.assertEqual(
            gzip.decompress(evidence.canonical_gzip_bytes),
            expected_json,
        )
        self.assertEqual(
            evidence.canonical_gzip_sha256,
            hashlib.sha256(evidence.canonical_gzip_bytes).hexdigest(),
        )
        self.assertEqual((self.slot / "RESERVED").read_bytes(), reserved_before)
        self.assertEqual(
            sorted(path.name for path in self.slot.iterdir()),
            ["RESERVED"],
        )

    def test_invalid_commit_is_rejected_before_network(self) -> None:
        for value in (None, True, "5" * 39, "A" * 40):
            with self.subTest(value=value):
                with patch.object(
                    shadow_prediction.requests,
                    "get",
                    side_effect=AssertionError("reseau interdit"),
                ) as get_mock:
                    with self.assertRaises(ShadowPredictionError):
                        fetch_activation_reverification_evidence(value)
                get_mock.assert_not_called()

    def test_request_failure_is_closed_without_any_file(self) -> None:
        before = sorted(path.as_posix() for path in self.project.rglob("*"))
        with patch.object(
            shadow_prediction.requests,
            "get",
            side_effect=requests.Timeout("timeout"),
        ):
            with self.assertRaises(ShadowPredictionError):
                fetch_activation_reverification_evidence(ACTIVATION_COMMIT)
        self.assertEqual(
            sorted(path.as_posix() for path in self.project.rglob("*")),
            before,
        )

    def test_http_envelope_must_match_exact_remote_contract(self) -> None:
        request_url = GITHUB_COMPARE_URL_TEMPLATE.format(
            expected_commit=ACTIVATION_COMMIT
        )
        cases = (
            _FakeResponse(status_code=201),
            _FakeResponse(status_code=True),
            _FakeResponse(history=[object()]),
            _FakeResponse(history=()),
            _FakeResponse(url=request_url + "/redirected"),
            _FakeResponse(content=b""),
            _FakeResponse(content=bytearray(b"not exact bytes")),
            _FakeResponse(headers={}),
            _FakeResponse(headers={"date": "2026-09-01T18:00:00Z"}),
            _FakeResponse(
                headers={
                    "date": EVIDENCE_HTTP_DATE,
                    "etag": 123,
                }
            ),
        )
        for response in cases:
            with self.subTest(response=response.__dict__):
                with self.assertRaises(ShadowPredictionError):
                    self._fetch(response=response)

    def test_compare_body_must_name_exact_activation_commit(self) -> None:
        duplicate_body = (
            b'{"base_commit":{"sha":"'
            + ACTIVATION_COMMIT.encode("ascii")
            + b'"},"base_commit":{"sha":"'
            + ACTIVATION_COMMIT.encode("ascii")
            + b'"},"merge_base_commit":{"sha":"'
            + ACTIVATION_COMMIT.encode("ascii")
            + b'"},"status":"ahead"}'
        )
        bodies = (
            b"not-json",
            b"[]",
            _compare_body(base_commit="6" * 40),
            _compare_body(merge_base_commit="6" * 40),
            _compare_body(status="behind"),
            duplicate_body,
        )
        for body in bodies:
            with self.subTest(body=body):
                with self.assertRaises(ShadowPredictionError):
                    self._fetch(response=_FakeResponse(content=body))

    def test_identical_status_and_absent_optional_headers_are_allowed(
        self,
    ) -> None:
        evidence = self._fetch(
            response=_FakeResponse(
                content=_compare_body(status="identical"),
                headers={"date": EVIDENCE_HTTP_DATE},
            )
        )
        self.assertEqual(
            evidence.raw_evidence["selected_response_headers"],
            {
                "date": EVIDENCE_HTTP_DATE,
                "content-type": None,
                "etag": None,
                "x-github-request-id": None,
            },
        )

    def test_fetch_never_reads_project_model_mlb_or_sqlite(self) -> None:
        with patch.object(
            Path,
            "read_bytes",
            side_effect=AssertionError("lecture de fichier interdite"),
        ):
            evidence = self._fetch()
        self.assertEqual(
            evidence.activation_introduction_commit,
            ACTIVATION_COMMIT,
        )

    def test_publication_is_exact_second_file_and_preserves_reserved(
        self,
    ) -> None:
        reserved_before = (self.slot / "RESERVED").read_bytes()

        publication = self._publish()

        evidence_path = self.slot / ACTIVATION_REVERIFICATION_FILENAME
        expected_relative = (
            SHADOW_RESULT_ROOT_RELATIVE_PATH
            / TARGET_DATE
            / ACTIVATION_REVERIFICATION_FILENAME
        ).as_posix()
        self.assertEqual(publication.slot_path, self.slot)
        self.assertEqual(publication.evidence_path, evidence_path)
        self.assertEqual(publication.evidence_relative_path, expected_relative)
        self.assertEqual(
            publication.evidence_sha256,
            self.evidence.canonical_gzip_sha256,
        )
        self.assertEqual(
            publication.activation_introduction_commit,
            ACTIVATION_COMMIT,
        )
        self.assertEqual(publication.activation_remote_ref, GITHUB_REMOTE_REF)
        self.assertEqual(
            publication.activation_remote_reverified_at_utc,
            EVIDENCE_HTTP_DATE_UTC,
        )
        self.assertEqual(
            publication.response_received_at_utc,
            EVIDENCE_RECEIVED_AT_UTC,
        )
        self.assertEqual(
            publication.response_body_sha256,
            self.evidence.response_body_sha256,
        )
        self.assertEqual(evidence_path.read_bytes(), self.evidence.canonical_gzip_bytes)
        self.assertEqual((self.slot / "RESERVED").read_bytes(), reserved_before)
        self.assertEqual(
            sorted(path.name for path in self.slot.iterdir()),
            ["RESERVED", ACTIVATION_REVERIFICATION_FILENAME],
        )

    def test_publication_reads_only_reserved_and_published_evidence(
        self,
    ) -> None:
        actual_read_bytes = Path.read_bytes
        read_names: list[str] = []

        def recording_read_bytes(path: Path) -> bytes:
            read_names.append(path.name)
            return actual_read_bytes(path)

        with (
            patch.object(Path, "read_bytes", new=recording_read_bytes),
            patch.object(
                shadow_prediction.requests,
                "get",
                side_effect=AssertionError("reseau interdit"),
            ),
        ):
            self._publish()

        self.assertEqual(
            read_names,
            ["RESERVED", ACTIVATION_REVERIFICATION_FILENAME],
        )

    def test_forged_evidence_is_rejected_before_slot_access(self) -> None:
        changed_raw = dict(self.evidence.raw_evidence)
        changed_raw["request_method"] = "POST"
        invalid_cases = (
            None,
            replace(self.evidence, activation_introduction_commit="6" * 40),
            replace(self.evidence, activation_remote_ref="refs/heads/dev"),
            replace(self.evidence, response_body_sha256="6" * 64),
            replace(self.evidence, raw_evidence=changed_raw),
            replace(self.evidence, canonical_json_bytes=b"forged\n"),
            replace(self.evidence, canonical_gzip_bytes=b"forged"),
            replace(self.evidence, canonical_gzip_sha256="6" * 64),
        )
        for evidence in invalid_cases:
            with self.subTest(evidence=evidence):
                with patch.object(
                    shadow_prediction,
                    "_require_shadow_result_root",
                    side_effect=AssertionError("acces disque interdit"),
                ):
                    with self.assertRaises(ShadowPredictionError):
                        self._publish(evidence=evidence)
        self.assertEqual(
            sorted(path.name for path in self.slot.iterdir()),
            ["RESERVED"],
        )

    def test_temporal_evidence_must_precede_reservation(self) -> None:
        cases = (
            replace(
                self.evidence,
                response_received_at_utc="2026-09-02T18:00:01Z",
                raw_evidence={
                    **self.evidence.raw_evidence,
                    "response_received_at_utc": "2026-09-02T18:00:01Z",
                },
            ),
            replace(
                self.evidence,
                activation_remote_reverified_at_utc=(
                    "2026-09-02T18:00:01Z"
                ),
                raw_evidence={
                    **self.evidence.raw_evidence,
                    "selected_response_headers": {
                        **self.evidence.raw_evidence[
                            "selected_response_headers"
                        ],
                        "date": "Wed, 02 Sep 2026 18:00:01 GMT",
                    },
                },
            ),
        )
        canonical_cases = []
        for evidence in cases:
            raw = evidence.raw_evidence
            canonical_json = _canonical_json_file_bytes(raw)
            canonical_gzip = shadow_prediction._canonical_gzip_bytes(
                canonical_json
            )
            canonical_cases.append(
                replace(
                    evidence,
                    canonical_json_bytes=canonical_json,
                    canonical_gzip_bytes=canonical_gzip,
                    canonical_gzip_sha256=hashlib.sha256(
                        canonical_gzip
                    ).hexdigest(),
                )
            )

        for evidence in canonical_cases:
            with self.subTest(evidence=evidence):
                with patch.object(
                    shadow_prediction,
                    "_require_shadow_result_root",
                    side_effect=AssertionError("acces disque interdit"),
                ):
                    with self.assertRaises(ShadowPredictionError):
                        self._publish(evidence=evidence)

    def test_immediate_order_rejects_every_preexisting_second_path(
        self,
    ) -> None:
        for name in (
            ACTIVATION_REVERIFICATION_FILENAME,
            "source_snapshot.json.gz",
            "FAILED.json",
            "COMPLETED",
            "unexpected.bin",
        ):
            with self.subTest(name=name), TemporaryDirectory() as directory:
                project = Path(directory)
                root = project.joinpath(*SHADOW_RESULT_ROOT_RELATIVE_PATH.parts)
                root.mkdir(parents=True)
                reservation = self._reserve(project=project)
                extra = reservation.slot_path / name
                extra.write_bytes(b"immutable\n")
                before = extra.read_bytes()

                with self.assertRaises(ShadowPredictionSlotConsumedError):
                    publish_activation_reverification_evidence(
                        reservation,
                        self.evidence,
                        project_directory=project,
                    )

                self.assertEqual(extra.read_bytes(), before)
                self.assertEqual(
                    sorted(path.name for path in reservation.slot_path.iterdir()),
                    sorted(["RESERVED", name]),
                )

    def test_missing_noncanonical_or_foreign_reserved_cannot_be_repaired(
        self,
    ) -> None:
        reserved_path = self.slot / "RESERVED"
        reserved_path.unlink()
        with self.assertRaises(ShadowPredictionSlotConsumedError):
            self._publish()
        self.assertFalse(reserved_path.exists())

        with TemporaryDirectory() as directory:
            project = Path(directory)
            root = project.joinpath(*SHADOW_RESULT_ROOT_RELATIVE_PATH.parts)
            root.mkdir(parents=True)
            reservation = self._reserve(project=project)
            path = reservation.slot_path / "RESERVED"
            path.write_bytes(path.read_bytes() + b"\n")
            before = path.read_bytes()
            with self.assertRaises(ShadowPredictionError):
                publish_activation_reverification_evidence(
                    reservation,
                    self.evidence,
                    project_directory=project,
                )
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(
                (reservation.slot_path / ACTIVATION_REVERIFICATION_FILENAME).exists()
            )

        with TemporaryDirectory() as directory:
            other_project = Path(directory)
            other_project.joinpath(
                *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts
            ).mkdir(parents=True)
            with self.assertRaises(ShadowPredictionError):
                self._publish(project_directory=other_project)
            self.assertFalse((root / TARGET_DATE).exists())

    def test_second_publication_never_reuses_or_overwrites(self) -> None:
        first = self._publish()
        before = first.evidence_path.read_bytes()

        with self.assertRaises(ShadowPredictionSlotConsumedError):
            self._publish()

        self.assertEqual(first.evidence_path.read_bytes(), before)
        self.assertEqual(
            hashlib.sha256(before).hexdigest(),
            first.evidence_sha256,
        )

    def test_concurrent_publishers_have_exactly_one_winner(self) -> None:
        def attempt(index: int) -> tuple[str, int]:
            try:
                self._publish()
            except ShadowPredictionSlotConsumedError:
                return "consumed", index
            return "published", index

        with ThreadPoolExecutor(max_workers=12) as executor:
            outcomes = list(executor.map(attempt, range(12)))

        self.assertEqual(
            sum(status == "published" for status, _ in outcomes),
            1,
        )
        self.assertEqual(
            sum(status == "consumed" for status, _ in outcomes),
            11,
        )
        self.assertEqual(
            (self.slot / ACTIVATION_REVERIFICATION_FILENAME).read_bytes(),
            self.evidence.canonical_gzip_bytes,
        )

    def test_concurrent_processes_have_exactly_one_winner(self) -> None:
        context = multiprocessing.get_context("spawn")
        start_gate = context.Event()
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_process_publication_worker,
                args=(
                    self.reservation,
                    self.evidence,
                    str(self.project),
                    start_gate,
                    result_queue,
                    index,
                ),
            )
            for index in range(6)
        ]
        for process in processes:
            process.start()
        start_gate.set()
        outcomes = []
        try:
            for _ in processes:
                outcomes.append(result_queue.get(timeout=30))
        except Empty as error:
            self.fail(f"Un processus de publication n'a pas repondu : {error}")
        finally:
            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
            result_queue.close()
            result_queue.join_thread()

        self.assertTrue(all(process.exitcode == 0 for process in processes))
        self.assertFalse(
            [outcome for outcome in outcomes if outcome[0] == "error"],
            outcomes,
        )
        self.assertEqual(
            sum(outcome[0] == "published" for outcome in outcomes),
            1,
        )
        self.assertEqual(
            sum(outcome[0] == "consumed" for outcome in outcomes),
            5,
        )

    def test_failure_before_link_preserves_reserved_slot(self) -> None:
        reserved_before = (self.slot / "RESERVED").read_bytes()
        with patch.object(
            shadow_prediction.os,
            "link",
            side_effect=OSError("publication impossible"),
        ):
            with self.assertRaises(OSError):
                self._publish()

        self.assertEqual((self.slot / "RESERVED").read_bytes(), reserved_before)
        self.assertEqual(
            sorted(path.name for path in self.slot.iterdir()),
            ["RESERVED"],
        )

    def test_failure_after_link_preserves_published_evidence(self) -> None:
        with patch.object(
            shadow_prediction,
            "_fsync_parent_directory",
            side_effect=OSError("fsync impossible"),
        ):
            with self.assertRaises(OSError):
                self._publish()

        path = self.slot / ACTIVATION_REVERIFICATION_FILENAME
        self.assertEqual(path.read_bytes(), self.evidence.canonical_gzip_bytes)
        with self.assertRaises(ShadowPredictionSlotConsumedError):
            self._publish()


if __name__ == "__main__":
    unittest.main()
