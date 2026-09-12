"""Tests du chargement Git-only des predictions certifiees shadow v2."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import shadow_certification as certification
from src import shadow_prediction as shadow
from src import shadow_scoring as scoring
from src import shadow_scoring_registration as registration


PROJECT = Path(__file__).resolve().parents[1]
TARGET_DATE = "2026-09-10"
NEXT_DATE = "2026-09-11"
RESULTS_COMMIT = registration.EXPECTED_FIRST_RESULTS_COMMIT
CERTIFICATION_COMMIT = registration.EXPECTED_FIRST_CERTIFICATION_COMMIT
FAKE_RUNTIME_COMMIT = "f" * 40


class ShadowScoringPredictionSourceTests(unittest.TestCase):
    """La cohorte ne peut provenir que de blobs Git certifies et immuables."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (
                PROJECT
                / "scoring_protocols"
                / "logistic_team_form_v1_platt_shadow_v2_2026_v1.json"
            ).read_text(encoding="utf-8")
        )

    def _authority(self) -> scoring.ScoringExecutionAuthority:
        return scoring.ScoringExecutionAuthority(
            protocol=self.protocol,
            registration={},
            scoring_protocol_sha256=registration.EXPECTED_SCORING_PROTOCOL_SHA256,
            protocol_introduction_commit=(
                scoring.EXPECTED_PROTOCOL_INTRODUCTION_COMMIT
            ),
            registration_commit=scoring.EXPECTED_REGISTRATION_COMMIT,
            registration_sha256=scoring.EXPECTED_REGISTRATION_SHA256,
            registration_remote_evidence_sha256=(
                scoring.EXPECTED_REGISTRATION_REMOTE_EVIDENCE_SHA256
            ),
            runtime_code_commit=FAKE_RUNTIME_COMMIT,
            scoring_engine_sha256="0" * 64,
        )

    def _shadow_authority(
        self,
        *,
        target: str = TARGET_DATE,
    ) -> shadow.ShadowExecutionAuthority:
        return shadow.ShadowExecutionAuthority(
            runtime_code_commit=FAKE_RUNTIME_COMMIT,
            shadow_service_code_commit="1" * 40,
            shadow_service_module_sha256="2" * 64,
            shadow_protocol_sha256=shadow.EXPECTED_SHADOW_PROTOCOL_SHA256,
            execution_manifest_sha256=(
                self.protocol["authorities"]["execution_manifest"]["sha256"]
            ),
            execution_manifest_introduction_commit="3" * 40,
            activation_sha256="4" * 64,
            activation_introduction_commit="5" * 40,
            activation_verified_at_utc="2026-09-06T08:00:00Z",
            minimum_target_official_date=target,
            runtime_versions=(),
        )

    def _load_real_first_source(
        self,
    ) -> scoring.ImmutableScoringPredictionSource:
        with (
            mock.patch.object(
                scoring,
                "_require_prediction_source_authorities",
                return_value=self._shadow_authority(),
            ),
            mock.patch.object(shadow, "_require_git_ancestor"),
        ):
            return scoring.load_immutable_scoring_prediction_source(
                self._authority(),
                TARGET_DATE,
                project_directory=PROJECT,
            )

    def _real_batch(self) -> scoring._ValidatedImmutableShadowBatch:
        prefix = (
            shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH / TARGET_DATE
        ).as_posix()
        blobs = {
            filename: shadow._git_blob_at_commit(
                PROJECT,
                RESULTS_COMMIT,
                f"{prefix}/{filename}",
            )
            for filename in certification.EXPECTED_RESULT_FILENAMES
        }
        with mock.patch.object(shadow, "_require_git_ancestor"):
            return scoring._validate_immutable_result_blobs(
                PROJECT,
                TARGET_DATE,
                RESULTS_COMMIT,
                self._shadow_authority(),
                blobs,
            )

    def _empty_blobs(self) -> dict[str, bytes]:
        authority = self._shadow_authority(target=NEXT_DATE)
        slot_key = shadow.build_slot_key(
            shadow_protocol_sha256=authority.shadow_protocol_sha256,
            target_official_date=NEXT_DATE,
        )
        batch_id = shadow.build_batch_id(
            slot_key=slot_key,
            execution_manifest_sha256=authority.execution_manifest_sha256,
            model_artifact_sha256=shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
        )
        reserved = {
            "marker_schema_version": 1,
            "batch_id": batch_id,
            "slot_key": slot_key,
            "target_official_date": NEXT_DATE,
            "reserved_at_utc": "2026-09-11T08:00:00Z",
            "shadow_protocol_sha256": authority.shadow_protocol_sha256,
            "execution_manifest_sha256": authority.execution_manifest_sha256,
            "runtime_code_commit": "6" * 40,
        }
        blobs: dict[str, bytes] = {
            "RESERVED": shadow._canonical_json_file_bytes(reserved),
            shadow.ACTIVATION_REVERIFICATION_FILENAME: b"activation",
            shadow.SOURCE_SNAPSHOT_FILENAME: b"snapshot",
            shadow.CANDIDATE_LEDGER_FILENAME: b"ledger",
            shadow.FEATURES_FILENAME: b"features",
            shadow.PREDICTIONS_FILENAME: shadow._canonical_csv_bytes(
                shadow._PREDICTIONS_COLUMNS,
                [],
            ),
        }

        def section(name: str) -> dict[str, object]:
            return {
                key: None for key in shadow._RECEIPT_SECTION_KEYS[name]
            }

        batch = section("batch")
        batch.update(
            {
                "batch_id": batch_id,
                "slot_key": slot_key,
                "target_official_date": NEXT_DATE,
                "status": "COMPLETED_NO_ELIGIBLE_GAMES",
                "earliest_predicted_scheduled_start_utc": None,
            }
        )
        lineage = section("lineage")
        lineage.update(
            {
                "runtime_code_commit": reserved["runtime_code_commit"],
                "shadow_protocol_sha256": authority.shadow_protocol_sha256,
                "execution_manifest_sha256": authority.execution_manifest_sha256,
                "model_artifact_sha256": shadow.EXPECTED_MODEL_ARTIFACT_SHA256,
            }
        )
        times = section("times")
        times.update(
            {
                "reserved_at_utc": reserved["reserved_at_utc"],
                "receipt_finalized_at_utc": "2026-09-11T08:01:00Z",
            }
        )
        counts = section("counts")
        counts.update(
            {
                "schedule_games": 0,
                "eligible_games": 0,
                "predicted_games": 0,
                "excluded_games_by_reason": {
                    key: 0 for key in shadow._EXCLUDED_GAMES_BY_REASON_KEYS
                },
            }
        )
        source = section("source")
        source.update(
            {
                "sqlite_snapshot_sha256": "7" * 64,
                "source_snapshot_sha256": hashlib.sha256(
                    blobs[shadow.SOURCE_SNAPSHOT_FILENAME]
                ).hexdigest(),
                "schedule_raw_archive_sha256": "8" * 64,
            }
        )
        output_hashes = section("output_hashes")
        for field, filename in certification._OUTPUT_HASH_TO_FILENAME.items():
            output_hashes[field] = hashlib.sha256(blobs[filename]).hexdigest()
        receipt = {
            "receipt_schema_version": 1,
            "batch": batch,
            "activation": section("activation"),
            "times": times,
            "schedule_http_response": section("schedule_http_response"),
            "lineage": lineage,
            "source": source,
            "counts": counts,
            "output_hashes": output_hashes,
            "model_invariants": section("model_invariants"),
            "negative_attestations": {
                key: True
                for key in shadow._RECEIPT_SECTION_KEYS[
                    "negative_attestations"
                ]
            },
            "runtime_versions": section("runtime_versions"),
        }
        receipt_bytes = shadow._canonical_json_file_bytes(receipt)
        blobs[shadow.RECEIPT_FILENAME] = receipt_bytes
        completed = {
            "marker_schema_version": 1,
            "batch_id": batch_id,
            "receipt_path": (
                shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH
                / NEXT_DATE
                / shadow.RECEIPT_FILENAME
            ).as_posix(),
            "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "completed_at_utc": "2026-09-11T08:02:00Z",
        }
        blobs[shadow.COMPLETED_FILENAME] = shadow._canonical_json_file_bytes(
            completed
        )
        return blobs

    def test_statuses_and_paths_match_the_frozen_protocol(self) -> None:
        """Le moteur ne cree aucune nouvelle interpretation du protocole."""
        source = self.protocol["prediction_source"]
        intent = self.protocol["intent_to_observe"]
        self.assertEqual(
            {status.value for status in scoring.ScoringPredictionSourceStatus},
            set(intent["allowed_date_statuses"]),
        )
        self.assertEqual(
            certification.EXPECTED_RESULT_FILENAMES,
            tuple(source["required_result_files_lexicographic"]),
        )
        self.assertEqual(
            source["source_of_truth"],
            "EXACT_IMMUTABLE_GIT_BLOBS_FROM_CERTIFIED_RESULTS_COMMIT",
        )
        self.assertTrue(
            source["working_tree_sqlite_and_regenerated_predictions_forbidden"]
        )

    def test_real_first_batch_is_loaded_from_its_immutable_commits(self) -> None:
        """Le lot publie le 10 septembre reproduit ses cinq lignes certifiees."""
        source = self._load_real_first_source()
        self.assertEqual(
            source.status,
            scoring.ScoringPredictionSourceStatus.CERTIFIED_NONEMPTY,
        )
        self.assertEqual(source.results_commit, RESULTS_COMMIT)
        self.assertEqual(source.certification_commit, CERTIFICATION_COMMIT)
        self.assertEqual(source.batch_id, registration.EXPECTED_FIRST_BATCH_ID)
        self.assertEqual(len(source.predictions), 5)
        self.assertEqual(
            source.predictions_sha256,
            registration.EXPECTED_FIRST_PREDICTIONS_SHA256,
        )
        self.assertEqual(
            source.receipt_sha256,
            registration.EXPECTED_FIRST_RECEIPT_SHA256,
        )
        self.assertEqual(
            source.certification_sha256,
            registration.EXPECTED_FIRST_CERTIFICATION_SHA256,
        )
        self.assertEqual(
            {prediction.batch_id for prediction in source.predictions},
            {registration.EXPECTED_FIRST_BATCH_ID},
        )
        self.assertEqual(
            {prediction.target_official_date for prediction in source.predictions},
            {TARGET_DATE},
        )

    def test_working_tree_files_are_never_read_as_prediction_source(self) -> None:
        """Meme une lecture Path directe est interdite pendant ce chargement."""
        authority = self._authority()
        forbidden = AssertionError("lecture de fichier de travail interdite")
        with (
            mock.patch.object(
                scoring,
                "_require_prediction_source_authorities",
                return_value=self._shadow_authority(),
            ),
            mock.patch.object(shadow, "_require_git_ancestor"),
            mock.patch.object(Path, "read_bytes", side_effect=forbidden),
            mock.patch.object(scoring.requests, "get", side_effect=forbidden) as net,
            mock.patch.object(
                shadow,
                "_read_regular_project_file",
                side_effect=forbidden,
            ),
        ):
            source = scoring.load_immutable_scoring_prediction_source(
                authority,
                TARGET_DATE,
                project_directory=PROJECT,
            )
        self.assertEqual(len(source.predictions), 5)
        net.assert_not_called()

    def test_mutated_result_history_is_a_structural_abort(self) -> None:
        """Un second commit touchant le lot ne peut pas etre ignore."""
        real_history = scoring._git_history
        result_root = (
            shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH / TARGET_DATE
        ).as_posix()

        def altered(project: Path, arguments: tuple[str, ...], *, field_name: str):
            if arguments == ("log", "--format=%H", "--", result_root):
                return ("e" * 40, RESULTS_COMMIT)
            return real_history(project, arguments, field_name=field_name)

        with (
            mock.patch.object(scoring, "_git_history", side_effect=altered),
            mock.patch.object(shadow, "_require_git_ancestor"),
        ):
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "INVALID_OR_MUTATED",
            ):
                scoring._discover_result_commit(
                    PROJECT,
                    TARGET_DATE,
                    runtime_commit=FAKE_RUNTIME_COMMIT,
                )

    def test_multiple_completed_introductions_abort(self) -> None:
        """Supprimer puis recreer COMPLETED ne reouvre jamais une date."""
        completed = (
            shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH
            / TARGET_DATE
            / shadow.COMPLETED_FILENAME
        ).as_posix()

        def histories(_project: Path, arguments: tuple[str, ...], *, field_name: str):
            del field_name
            if completed in arguments:
                return (RESULTS_COMMIT, "e" * 40)
            return (RESULTS_COMMIT,)

        with mock.patch.object(scoring, "_git_history", side_effect=histories):
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "unique commit",
            ):
                scoring._discover_result_commit(
                    PROJECT,
                    TARGET_DATE,
                    runtime_commit=FAKE_RUNTIME_COMMIT,
                )

    def test_absent_and_incomplete_dates_remain_visible(self) -> None:
        """Une date jamais publiee est MISSED; une racine partielle est FAILED."""
        for create_root, expected in (
            (False, scoring.ScoringPredictionSourceStatus.MISSED),
            (True, scoring.ScoringPredictionSourceStatus.FAILED),
        ):
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    if create_root:
                        result = root.joinpath(
                            *shadow.SHADOW_RESULT_ROOT_RELATIVE_PATH.parts,
                            NEXT_DATE,
                        )
                        result.mkdir(parents=True)
                    with mock.patch.object(
                        scoring,
                        "_git_history",
                        return_value=(),
                    ):
                        status, commit = scoring._discover_result_commit(
                            root,
                            NEXT_DATE,
                            runtime_commit=FAKE_RUNTIME_COMMIT,
                        )
                    self.assertEqual(status, expected)
                    self.assertIsNone(commit)

    def test_orphan_certification_is_never_downgraded_to_missed(self) -> None:
        """Une preuve sans lot est une anomalie, pas une absence innocente."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path, certification_path = scoring._certification_paths_for_target(
                NEXT_DATE
            )
            path = root.joinpath(*certification_path.parts)
            path.parent.mkdir(parents=True)
            path.write_bytes(b"orphelin")
            with mock.patch.object(scoring, "_git_history", return_value=()):
                with self.assertRaisesRegex(
                    shadow.ShadowPredictionError,
                    "certification existe sans lot",
                ):
                    scoring._discover_result_commit(
                        root,
                        NEXT_DATE,
                        runtime_commit=FAKE_RUNTIME_COMMIT,
                    )
            self.assertFalse(root.joinpath(*raw_path.parts).exists())

    def test_empty_completed_batch_is_valid_without_model_or_certificate(self) -> None:
        """Un CSV avec seul en-tete devient exactement COMPLETED_EMPTY."""
        blobs = self._empty_blobs()
        with mock.patch.object(shadow, "_require_git_ancestor") as ancestor:
            batch = scoring._validate_immutable_result_blobs(
                PROJECT,
                NEXT_DATE,
                "9" * 40,
                self._shadow_authority(target=NEXT_DATE),
                blobs,
            )
        self.assertEqual(batch.status, "COMPLETED_NO_ELIGIBLE_GAMES")
        self.assertIsNone(batch.earliest_predicted_start_utc)
        self.assertEqual(scoring._read_certified_predictions_from_blob(batch), ())
        ancestor.assert_called_once()

    def test_empty_batch_with_extra_prediction_row_is_rejected(self) -> None:
        """Le statut vide ne peut masquer une probabilite surnumeraire."""
        blobs = self._empty_blobs()
        blobs[shadow.PREDICTIONS_FILENAME] += b"fraud\n"
        with mock.patch.object(shadow, "_require_git_ancestor"):
            with self.assertRaises(shadow.ShadowPredictionError):
                scoring._validate_immutable_result_blobs(
                    PROJECT,
                    NEXT_DATE,
                    "9" * 40,
                    self._shadow_authority(target=NEXT_DATE),
                    blobs,
                )

    def test_nonempty_without_certification_is_local_only_and_excluded(self) -> None:
        """Une probabilite locale ne traverse jamais la cohorte confirmatoire."""
        real_batch = self._real_batch()
        shifted_batch = replace(
            real_batch,
            target_official_date=NEXT_DATE,
            results_commit="9" * 40,
        )
        sample = scoring.CertifiedScoringPrediction(
            prediction_id="1" * 64,
            batch_id=shifted_batch.batch_id,
            game_id=1,
            occurrence_key="2" * 64,
            season=2026,
            target_official_date=NEXT_DATE,
            away_team_id=10,
            home_team_id=20,
            p_home_win="0.6",
            p_away_win="0.4",
        )
        with (
            mock.patch.object(
                scoring,
                "_require_prediction_source_authorities",
                return_value=self._shadow_authority(target=NEXT_DATE),
            ),
            mock.patch.object(
                scoring,
                "_discover_result_commit",
                return_value=(
                    scoring.ScoringPredictionSourceStatus.CERTIFIED_NONEMPTY,
                    shifted_batch.results_commit,
                ),
            ),
            mock.patch.object(scoring, "_read_result_blobs", return_value={}),
            mock.patch.object(
                scoring,
                "_validate_immutable_result_blobs",
                return_value=shifted_batch,
            ),
            mock.patch.object(
                scoring,
                "_read_certified_predictions_from_blob",
                return_value=(sample,),
            ),
            mock.patch.object(
                scoring,
                "_read_and_validate_certification",
                return_value=("", "", ""),
            ),
        ):
            source = scoring.load_immutable_scoring_prediction_source(
                self._authority(),
                NEXT_DATE,
                project_directory=PROJECT,
            )
        self.assertEqual(
            source.status,
            scoring.ScoringPredictionSourceStatus.LOCAL_ONLY,
        )
        self.assertEqual(source.predictions, ())
        self.assertEqual(source.results_commit, shifted_batch.results_commit)

    def test_mutated_certification_blob_aborts_instead_of_local_only(self) -> None:
        """Une certification presente mais alteree ne peut jamais etre ignoree."""
        real_blob = shadow._git_blob_at_commit

        def corrupted(project: Path, commit: str, relative: str) -> bytes:
            content = real_blob(project, commit, relative)
            if relative.endswith(f"/{TARGET_DATE}.json"):
                return content.replace(b"606.800000", b"606.800001")
            return content

        with (
            mock.patch.object(
                scoring,
                "_require_prediction_source_authorities",
                return_value=self._shadow_authority(),
            ),
            mock.patch.object(shadow, "_require_git_ancestor"),
            mock.patch.object(
                shadow,
                "_git_blob_at_commit",
                side_effect=corrupted,
            ),
        ):
            with self.assertRaises(shadow.ShadowPredictionError):
                scoring.load_immutable_scoring_prediction_source(
                    self._authority(),
                    TARGET_DATE,
                    project_directory=PROJECT,
                )

    def test_certification_and_raw_evidence_must_share_one_commit(self) -> None:
        """Une preuve ajoutee plus tard ne peut pas reparer la certification."""
        real_history = scoring._git_history
        raw_path, _ = scoring._certification_paths_for_target(TARGET_DATE)

        def altered(project: Path, arguments: tuple[str, ...], *, field_name: str):
            if (
                "--diff-filter=A" in arguments
                and raw_path.as_posix() in arguments
            ):
                return ("e" * 40,)
            return real_history(project, arguments, field_name=field_name)

        batch = self._real_batch()
        with mock.patch.object(scoring, "_git_history", side_effect=altered):
            with self.assertRaisesRegex(
                shadow.ShadowPredictionError,
                "INVALID_OR_MUTATED",
            ):
                scoring._read_and_validate_certification(
                    PROJECT,
                    batch,
                    runtime_commit=FAKE_RUNTIME_COMMIT,
                )

    def test_prediction_identity_or_probability_tampering_is_rejected(self) -> None:
        """Identifiant, equipes et probabilites doivent rester lies au blob."""
        batch = self._real_batch()
        lines = batch.predictions_bytes.decode("utf-8").splitlines()
        cells = lines[1].split(",")
        variants = (
            (0, "0" * 64),
            (7, cells[6]),
            (23, cells[22]),
        )
        for index, replacement in variants:
            with self.subTest(index=index):
                changed = list(cells)
                changed[index] = replacement
                rows = [line.split(",") for line in lines[1:]]
                rows[0] = changed
                forged = shadow._canonical_csv_bytes(
                    shadow._PREDICTIONS_COLUMNS,
                    rows,
                )
                with self.assertRaises(shadow.ShadowPredictionError):
                    scoring._read_certified_predictions_from_blob(
                        replace(batch, predictions_bytes=forged)
                    )

    def test_canonical_float_complements_survive_decimal_rendering(self) -> None:
        """Le texte canonique peut sommer a 1 plus ou moins 6e-17."""
        batch = self._real_batch()
        original_rows = [
            line.split(",")
            for line in batch.predictions_bytes.decode("utf-8").splitlines()[1:]
        ]
        probability_pairs = (
            ("0.46839344147795409", "0.53160655852204597"),
            ("0.4806587455251356", "0.51934125447486434"),
        )
        for p_home, p_away in probability_pairs:
            with self.subTest(p_home=p_home, p_away=p_away):
                rows = [list(row) for row in original_rows]
                rows[0][22] = p_home
                rows[0][23] = p_away
                rendered = shadow._canonical_csv_bytes(
                    shadow._PREDICTIONS_COLUMNS,
                    rows,
                )
                predictions = scoring._read_certified_predictions_from_blob(
                    replace(batch, predictions_bytes=rendered)
                )
                self.assertEqual(predictions[0].p_home_win, p_home)
                self.assertEqual(predictions[0].p_away_win, p_away)

    def test_loader_has_no_outcome_network_sqlite_or_model_side_effect(self) -> None:
        """La barriere de provenance ne consulte aucune source du lendemain."""
        source_text = inspect.getsource(
            scoring.load_immutable_scoring_prediction_source
        )
        self.assertNotIn("requests", source_text)
        self.assertNotIn("sqlite", source_text.lower())
        self.assertNotIn("joblib", source_text.lower())
        self.assertNotIn("predict_proba", source_text)
        forbidden = AssertionError("effet externe interdit")
        with (
            mock.patch.object(
                scoring,
                "_require_prediction_source_authorities",
                return_value=self._shadow_authority(),
            ),
            mock.patch.object(shadow, "_require_git_ancestor"),
            mock.patch.object(scoring.requests, "get", side_effect=forbidden) as net,
        ):
            loaded = scoring.load_immutable_scoring_prediction_source(
                self._authority(),
                TARGET_DATE,
                project_directory=PROJECT,
            )
        self.assertEqual(len(loaded.predictions), 5)
        net.assert_not_called()


if __name__ == "__main__":
    unittest.main()
