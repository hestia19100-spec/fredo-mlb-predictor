"""Tests de la capture brute MLB et de la fermeture des echecs."""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import requests

from src import shadow_prediction as shadow
from src import shadow_scoring as scoring
from src import shadow_scoring_registration as registration


RUNTIME_COMMIT = "d" * 40
TARGET_DATE = "2026-09-10"
CHECKPOINT_DATE = "2026-09-11"
RESERVED_AT = "2026-09-11T06:00:00Z"
RESPONSE_AT = datetime(2026, 9, 11, 6, 0, 2, tzinfo=timezone.utc)
HTTP_DATE = "Fri, 11 Sep 2026 06:00:02 GMT"


class _Response:
    """Reponse requests minimale et entierement controlee."""

    def __init__(
        self,
        *,
        content: bytes,
        url: str,
        status_code: int = 200,
        history: tuple[object, ...] = (),
        headers: dict[str, str] | None = None,
    ) -> None:
        self.content = content
        self.url = url
        self.status_code = status_code
        self.history = history
        self.headers = headers or {
            "Date": HTTP_DATE,
            "Content-Type": "application/json; charset=utf-8",
        }


class ShadowScoringOutcomeCaptureTests(unittest.TestCase):
    """Une observation publie une preuve brute ou un echec terminal."""

    def _authority(self) -> scoring.ScoringExecutionAuthority:
        return scoring.ScoringExecutionAuthority(
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
            runtime_code_commit=RUNTIME_COMMIT,
            scoring_engine_sha256="e" * 64,
        )

    def _project(self, temporary: str) -> Path:
        project = Path(temporary)
        project.joinpath(
            *scoring.SCORING_OUTPUT_ROOT_RELATIVE_PATH.parts
        ).mkdir(parents=True)
        return project

    def _reservation(
        self,
        project: Path,
    ) -> scoring.ScoringObservationReservation:
        return scoring.reserve_scoring_observation_slot(
            self._authority(),
            TARGET_DATE,
            CHECKPOINT_DATE,
            reserved_at_utc=RESERVED_AT,
            project_directory=project,
        )

    def _payload_bytes(self, *, game_count: int = 1) -> bytes:
        games = [
            {
                "gamePk": 900000 + index,
                "status": {
                    "abstractGameState": "Final",
                    "detailedState": "Final",
                },
            }
            for index in range(game_count)
        ]
        return shadow._canonical_json_bytes(
            {
                "dates": [{"date": TARGET_DATE, "games": games}],
                "totalGames": game_count,
            }
        )

    def _response(
        self,
        *,
        content: bytes | None = None,
        url: str | None = None,
        status_code: int = 200,
        history: tuple[object, ...] = (),
        headers: dict[str, str] | None = None,
    ) -> _Response:
        return _Response(
            content=self._payload_bytes() if content is None else content,
            url=(
                scoring._expected_outcome_request_url(TARGET_DATE)
                if url is None
                else url
            ),
            status_code=status_code,
            history=history,
            headers=headers,
        )

    def _fetch_evidence(
        self,
        response: _Response | None = None,
    ) -> scoring.ScoringOutcomeObservationEvidence:
        with (
            mock.patch.object(
                scoring.requests,
                "get",
                return_value=response or self._response(),
            ),
            mock.patch.object(scoring, "_utc_now", return_value=RESPONSE_AT),
        ):
            return scoring.fetch_scoring_outcome_evidence(
                TARGET_DATE,
                CHECKPOINT_DATE,
            )

    def test_exact_single_mlb_request_builds_canonical_evidence(self) -> None:
        """La capture fait une requete exacte et conserve le corps original."""
        response = self._response(content=self._payload_bytes(game_count=2))
        with (
            mock.patch.object(
                scoring.requests,
                "get",
                return_value=response,
            ) as get,
            mock.patch.object(scoring, "_utc_now", return_value=RESPONSE_AT),
        ):
            evidence = scoring.fetch_scoring_outcome_evidence(
                TARGET_DATE,
                CHECKPOINT_DATE,
            )

        get.assert_called_once_with(
            scoring.MLB_SCHEDULE_URL,
            params={
                "sportId": 1,
                "startDate": TARGET_DATE,
                "endDate": TARGET_DATE,
                "gameTypes": "R",
                "hydrate": "probablePitcher",
            },
            headers=scoring.OUTCOME_REQUEST_HEADERS,
            timeout=scoring.OUTCOME_REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
            verify=True,
        )
        self.assertEqual(evidence.flattened_occurrence_count, 2)
        self.assertEqual(evidence.response_received_at_utc, "2026-09-11T06:00:02Z")
        self.assertEqual(
            evidence.response_body_sha256,
            hashlib.sha256(response.content).hexdigest(),
        )
        self.assertEqual(
            gzip.decompress(evidence.canonical_gzip_bytes),
            evidence.canonical_json_bytes,
        )
        self.assertEqual(
            evidence.canonical_json_bytes,
            shadow._canonical_json_file_bytes(evidence.raw_evidence),
        )

    def test_http_envelope_must_be_exact(self) -> None:
        """Statut, redirection et URL effective sont fermes."""
        cases = (
            (self._response(status_code=503), "statut HTTP 200"),
            (self._response(history=(object(),)), "aucune redirection"),
            (
                self._response(url="https://statsapi.mlb.com/other"),
                "URL MLB effective",
            ),
        )
        for response, message in cases:
            with self.subTest(message=message):
                with (
                    mock.patch.object(
                        scoring.requests,
                        "get",
                        return_value=response,
                    ) as get,
                    mock.patch.object(
                        scoring,
                        "_utc_now",
                        return_value=RESPONSE_AT,
                    ),
                ):
                    with self.assertRaisesRegex(
                        scoring.ScoringOutcomeRequestError,
                        message,
                    ):
                        scoring.fetch_scoring_outcome_evidence(
                            TARGET_DATE,
                            CHECKPOINT_DATE,
                        )
                get.assert_called_once()

    def test_http_date_must_match_the_local_clock(self) -> None:
        """Une preuve temporelle eloignee de plus de cinq minutes est refusee."""
        response = self._response(
            headers={
                "Date": "Fri, 11 Sep 2026 05:54:59 GMT",
                "Content-Type": "application/json",
            }
        )
        with (
            mock.patch.object(scoring.requests, "get", return_value=response),
            mock.patch.object(scoring, "_utc_now", return_value=RESPONSE_AT),
        ):
            with self.assertRaisesRegex(
                scoring.ScoringOutcomeValidationError,
                "plus de 300 secondes",
            ):
                scoring.fetch_scoring_outcome_evidence(
                    TARGET_DATE,
                    CHECKPOINT_DATE,
                )

    def test_request_exception_is_typed_and_never_retried(self) -> None:
        """Une panne reseau reste une tentative unique et un echec controle."""
        with mock.patch.object(
            scoring.requests,
            "get",
            side_effect=requests.Timeout("delai depasse"),
        ) as get:
            with self.assertRaises(scoring.ScoringOutcomeRequestError):
                scoring.fetch_scoring_outcome_evidence(
                    TARGET_DATE,
                    CHECKPOINT_DATE,
                )
        get.assert_called_once()

    def test_raw_payload_is_structurally_validated(self) -> None:
        """JSON ambigu, listes invalides et total faux sont refuses."""
        invalid_payloads = (
            b"",
            b"[]",
            b'{"dates":[],"dates":[]}',
            b'{"dates":{},"totalGames":0}',
            b'{"dates":[{"games":{}}],"totalGames":0}',
            b'{"dates":[{"games":[null]}],"totalGames":1}',
            b'{"dates":[{"games":[]}],"totalGames":1}',
            b'{"dates":[{"games":[]}],"totalGames":true}',
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self._response(content=payload)
                with (
                    mock.patch.object(
                        scoring.requests,
                        "get",
                        return_value=response,
                    ),
                    mock.patch.object(
                        scoring,
                        "_utc_now",
                        return_value=RESPONSE_AT,
                    ),
                ):
                    with self.assertRaises(
                        scoring.ScoringOutcomeValidationError
                    ):
                        scoring.fetch_scoring_outcome_evidence(
                            TARGET_DATE,
                            CHECKPOINT_DATE,
                        )

    def test_success_publication_is_the_exact_second_file(self) -> None:
        """Apres RESERVED, seule la preuve MLB brute est ajoutee."""
        evidence = self._fetch_evidence()
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(temporary)
            reservation = self._reservation(project)
            publication = scoring.publish_scoring_outcome_evidence(
                reservation,
                evidence,
                project_directory=project,
            )

            self.assertEqual(
                sorted(path.name for path in reservation.slot_path.iterdir()),
                ["RESERVED", scoring.OUTCOME_EVIDENCE_FILENAME],
            )
            persisted = publication.evidence_path.read_bytes()
            self.assertEqual(persisted, evidence.canonical_gzip_bytes)
            self.assertEqual(
                hashlib.sha256(persisted).hexdigest(),
                evidence.canonical_gzip_sha256,
            )
            self.assertEqual(
                json.loads(gzip.decompress(persisted)),
                evidence.raw_evidence,
            )

    def test_capture_observes_reserved_before_the_network_call(self) -> None:
        """MLB ne peut etre lu qu'apres la publication verifiee de RESERVED."""
        evidence = self._fetch_evidence()
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(temporary)
            reservation = self._reservation(project)
            observed_names: list[list[str]] = []

            def fetch(*_args: object) -> scoring.ScoringOutcomeObservationEvidence:
                observed_names.append(
                    sorted(path.name for path in reservation.slot_path.iterdir())
                )
                return evidence

            with mock.patch.object(
                scoring,
                "fetch_scoring_outcome_evidence",
                side_effect=fetch,
            ) as fetch_mock:
                result = scoring.capture_and_publish_scoring_outcome_evidence(
                    reservation,
                    project_directory=project,
                )

            self.assertIsInstance(
                result,
                scoring.ScoringOutcomeEvidencePublication,
            )
            self.assertEqual(observed_names, [["RESERVED"]])
            fetch_mock.assert_called_once_with(TARGET_DATE, CHECKPOINT_DATE)

    def test_controlled_request_failure_publishes_only_failed(self) -> None:
        """Une panne MLB ferme le creneau sans preuve brute partielle."""
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(temporary)
            reservation = self._reservation(project)
            error = scoring.ScoringOutcomeRequestError("MLB indisponible")
            with (
                mock.patch.object(
                    scoring,
                    "fetch_scoring_outcome_evidence",
                    side_effect=error,
                ) as fetch,
                mock.patch.object(
                    scoring,
                    "_utc_now",
                    return_value=datetime(
                        2026, 9, 11, 6, 0, 3, tzinfo=timezone.utc
                    ),
                ),
            ):
                result = scoring.capture_and_publish_scoring_outcome_evidence(
                    reservation,
                    project_directory=project,
                )

            fetch.assert_called_once()
            self.assertIsInstance(result, scoring.ScoringObservationFailure)
            self.assertEqual(result.stage, "OUTCOME_REQUEST")
            self.assertEqual(
                sorted(path.name for path in reservation.slot_path.iterdir()),
                [scoring.FAILED_FILENAME, "RESERVED"],
            )
            failed = shadow._read_canonical_json_object(result.failed_path)
            self.assertIsNotNone(failed)
            self.assertEqual(failed[0], result.failed_marker)
            self.assertEqual(failed[0]["error_type"], type(error).__name__)
            self.assertEqual(failed[0]["error_message"], str(error))

    def test_controlled_validation_failure_has_the_exact_stage(self) -> None:
        """Une reponse invalide est distinguee d'une panne de transport."""
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(temporary)
            reservation = self._reservation(project)
            with (
                mock.patch.object(
                    scoring,
                    "fetch_scoring_outcome_evidence",
                    side_effect=scoring.ScoringOutcomeValidationError(
                        "JSON MLB invalide"
                    ),
                ),
                mock.patch.object(
                    scoring,
                    "_utc_now",
                    return_value=datetime(
                        2026, 9, 11, 6, 0, 3, tzinfo=timezone.utc
                    ),
                ),
            ):
                result = scoring.capture_and_publish_scoring_outcome_evidence(
                    reservation,
                    project_directory=project,
                )
            self.assertIsInstance(result, scoring.ScoringObservationFailure)
            self.assertEqual(result.stage, "OUTCOME_VALIDATION")
            self.assertEqual(
                result.failed_marker["error_type"],
                "ScoringOutcomeValidationError",
            )

    def test_unexpected_programming_error_is_never_hidden(self) -> None:
        """Un bug reste visible et ne fabrique pas un faux echec metier."""
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(temporary)
            reservation = self._reservation(project)
            with mock.patch.object(
                scoring,
                "fetch_scoring_outcome_evidence",
                side_effect=RuntimeError("bug visible"),
            ):
                with self.assertRaisesRegex(RuntimeError, "bug visible"):
                    scoring.capture_and_publish_scoring_outcome_evidence(
                        reservation,
                        project_directory=project,
                    )
            self.assertEqual(
                sorted(path.name for path in reservation.slot_path.iterdir()),
                ["RESERVED"],
            )

    def test_forged_evidence_is_rejected_before_publication(self) -> None:
        """Une preuve modifiee ne peut jamais devenir le second fichier."""
        evidence = self._fetch_evidence()
        forged = replace(evidence, checkpoint_utc_date="2026-09-12")
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(temporary)
            reservation = self._reservation(project)
            with self.assertRaises(shadow.ShadowPredictionError):
                scoring.publish_scoring_outcome_evidence(
                    reservation,
                    forged,
                    project_directory=project,
                )
            self.assertEqual(
                sorted(path.name for path in reservation.slot_path.iterdir()),
                ["RESERVED"],
            )

    def test_concurrent_publishers_have_exactly_one_winner(self) -> None:
        """Deux publications simultanees ne partagent jamais le creneau."""
        evidence = self._fetch_evidence()
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(temporary)
            reservation = self._reservation(project)
            barrier = threading.Barrier(2)

            def publish() -> str:
                barrier.wait()
                try:
                    scoring.publish_scoring_outcome_evidence(
                        reservation,
                        evidence,
                        project_directory=project,
                    )
                except (
                    shadow.ShadowPredictionSlotConsumedError,
                    shadow.ShadowPublicationConflictError,
                ):
                    return "conflict"
                return "winner"

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _index: publish(), range(2)))

            self.assertEqual(results.count("winner"), 1)
            self.assertEqual(results.count("conflict"), 1)
            self.assertEqual(
                sorted(path.name for path in reservation.slot_path.iterdir()),
                ["RESERVED", scoring.OUTCOME_EVIDENCE_FILENAME],
            )

    def test_failure_marker_accepts_only_matching_controlled_errors(self) -> None:
        """FAILED ne peut ni mentir sur l'etape ni masquer un autre bug."""
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(temporary)
            reservation = self._reservation(project)
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "ne correspond pas",
            ):
                scoring.fail_scoring_observation_slot(
                    reservation,
                    failed_at_utc="2026-09-11T06:00:03Z",
                    stage="OUTCOME_VALIDATION",
                    error=scoring.ScoringOutcomeRequestError("panne"),
                    project_directory=project,
                )
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "echec MLB controle",
            ):
                scoring.fail_scoring_observation_slot(
                    reservation,
                    failed_at_utc="2026-09-11T06:00:03Z",
                    stage="OUTCOME_REQUEST",
                    error=RuntimeError("bug"),  # type: ignore[arg-type]
                    project_directory=project,
                )
            self.assertEqual(
                sorted(path.name for path in reservation.slot_path.iterdir()),
                ["RESERVED"],
            )

    def test_capture_layer_imports_no_sqlite_or_model_runtime(self) -> None:
        """Cette couche reseau ne lit toujours ni base ni modele."""
        source = Path(scoring.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("sqlite3", imports)
        self.assertNotIn("joblib", imports)
        self.assertNotIn("src.database", imports)
        self.assertNotIn("src.calibrated_model", imports)


if __name__ == "__main__":
    unittest.main()
