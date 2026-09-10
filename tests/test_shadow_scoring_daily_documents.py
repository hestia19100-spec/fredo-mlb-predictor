"""Tests des sorties quotidiennes canoniques du scoring shadow MLB."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

from src import shadow_prediction as shadow
from src import shadow_scoring as scoring
from src import shadow_scoring_registration as registration


PROJECT = Path(__file__).resolve().parents[1]
TARGET = "2026-09-10"
CHECKPOINT = "2026-09-11"
EVIDENCE_SHA256 = "f" * 64
BATCH_ID = "c" * 64


class ShadowScoringDailyDocumentsTests(unittest.TestCase):
    """CSV et rapport ne peuvent diverger de la cohorte certifiee."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (
                PROJECT
                / "scoring_protocols"
                / "logistic_team_form_v1_platt_shadow_v2_2026_v1.json"
            ).read_text(encoding="utf-8")
        )

    def _authority(
        self,
        *,
        protocol: dict[str, object] | None = None,
    ) -> scoring.ScoringExecutionAuthority:
        return scoring.ScoringExecutionAuthority(
            protocol=self.protocol if protocol is None else protocol,
            registration={},
            scoring_protocol_sha256=registration.EXPECTED_SCORING_PROTOCOL_SHA256,
            protocol_introduction_commit="1" * 40,
            registration_commit="2" * 40,
            registration_sha256="3" * 64,
            registration_remote_evidence_sha256="4" * 64,
            runtime_code_commit="5" * 40,
            scoring_engine_sha256="6" * 64,
        )

    def _reservation(self) -> scoring.ScoringObservationReservation:
        observation_id = scoring.build_observation_id(
            scoring_protocol_sha256=registration.EXPECTED_SCORING_PROTOCOL_SHA256,
            target_official_date=date.fromisoformat(TARGET),
            checkpoint_utc_date=date.fromisoformat(CHECKPOINT),
        )
        marker = {
            "marker_schema_version": 1,
            "protocol_id": registration.EXPECTED_PROTOCOL_ID,
            "scoring_protocol_sha256": (
                registration.EXPECTED_SCORING_PROTOCOL_SHA256
            ),
            "target_official_date": TARGET,
            "checkpoint_utc_date": CHECKPOINT,
            "observation_id": observation_id,
            "reserved_at_utc": "2026-09-11T06:00:01Z",
            "runtime_code_commit": "5" * 40,
        }
        marker_bytes = shadow._canonical_json_file_bytes(marker)
        return scoring.ScoringObservationReservation(
            slot_path=Path("unused-observation-slot"),
            target_official_date=TARGET,
            checkpoint_utc_date=CHECKPOINT,
            observation_id=observation_id,
            reserved_marker=marker,
            reserved_marker_sha256=hashlib.sha256(marker_bytes).hexdigest(),
        )

    def _predictions(self) -> tuple[scoring.CertifiedScoringPrediction, ...]:
        return (
            scoring.CertifiedScoringPrediction(
                prediction_id="a" * 64,
                batch_id=BATCH_ID,
                game_id=1001,
                occurrence_key="d" * 64,
                season=2026,
                target_official_date=TARGET,
                away_team_id=10,
                home_team_id=20,
                p_home_win="0.600000000000",
                p_away_win="0.400000000000",
            ),
            scoring.CertifiedScoringPrediction(
                prediction_id="b" * 64,
                batch_id=BATCH_ID,
                game_id=1002,
                occurrence_key="e" * 64,
                season=2026,
                target_official_date=TARGET,
                away_team_id=30,
                home_team_id=40,
                p_home_win="0.400000000000",
                p_away_win="0.600000000000",
            ),
        )

    def _source(self) -> scoring.ImmutableScoringPredictionSource:
        return scoring.ImmutableScoringPredictionSource(
            target_official_date=TARGET,
            status=scoring.ScoringPredictionSourceStatus.CERTIFIED_NONEMPTY,
            results_commit="7" * 40,
            certification_commit="8" * 40,
            batch_id=BATCH_ID,
            predictions=self._predictions(),
            predictions_sha256="9" * 64,
            receipt_sha256="a" * 64,
            certification_sha256="b" * 64,
            raw_certification_evidence_sha256="c" * 64,
            status_reason="VALID_IMMUTABLE_CERTIFIED_NONEMPTY_BATCH",
        )

    def _row(
        self,
        prediction: scoring.CertifiedScoringPrediction,
        *,
        home_score: int,
        away_score: int,
    ) -> scoring.ScoringAdjudication:
        home_win = int(home_score > away_score)
        predicted_side = (
            "HOME" if prediction.p_home_win >= "0.500000000000" else "AWAY"
        )
        actual_winner = "HOME" if home_win else "AWAY"
        log_loss, brier = scoring._individual_probability_metrics(
            scoring.Decimal(prediction.p_home_win),
            home_win,
        )
        return scoring.ScoringAdjudication(
            prediction_id=prediction.prediction_id,
            batch_id=prediction.batch_id,
            game_id=prediction.game_id,
            occurrence_key=prediction.occurrence_key,
            target_official_date=prediction.target_official_date,
            away_team_id=prediction.away_team_id,
            home_team_id=prediction.home_team_id,
            p_away_win=prediction.p_away_win,
            p_home_win=prediction.p_home_win,
            predicted_side=predicted_side,
            outcome_status="SCORED_FINAL",
            away_score=away_score,
            home_score=home_score,
            home_win=home_win,
            actual_winner=actual_winner,
            classification_correct=int(predicted_side == actual_winner),
            individual_log_loss=log_loss,
            individual_brier_score=brier,
            status_code_normalized="F",
            status_detail_normalized="FINAL",
            final_official_date=TARGET,
            outcome_http_date_utc="2026-09-11T06:01:00Z",
            outcome_response_received_at_utc="2026-09-11T06:01:01Z",
            outcome_evidence_sha256=EVIDENCE_SHA256,
        )

    def _batch(self) -> scoring.ScoringAdjudicationBatch:
        predictions = self._predictions()
        rows = (
            self._row(predictions[0], home_score=5, away_score=3),
            self._row(predictions[1], home_score=2, away_score=4),
        )
        return scoring.ScoringAdjudicationBatch(
            target_official_date=TARGET,
            checkpoint_utc_date=CHECKPOINT,
            outcome_evidence_sha256=EVIDENCE_SHA256,
            adjudications=rows,
            certified_prediction_count=2,
            scored_count=2,
            void_count=0,
            pending_count=0,
            correct_count=2,
            incorrect_count=0,
        )

    def _build(self) -> scoring.ScoringObservationDocuments:
        return scoring.build_scoring_observation_documents(
            self._authority(),
            self._reservation(),
            self._source(),
            self._batch(),
        )

    def test_exact_canonical_csv_and_daily_report(self) -> None:
        """Les deux documents suivent les schemas et formules preenregistres."""
        documents = self._build()
        rows = self._batch().adjudications
        expected_csv = shadow._canonical_csv_bytes(
            scoring._ADJUDICATION_COLUMNS,
            tuple(row.as_csv_row() for row in rows),
        )
        expected_report = {
            "report_schema_version": 1,
            "status": "PROVISIONAL_DAILY_NO_VERDICT",
            "protocol_id": registration.EXPECTED_PROTOCOL_ID,
            "target_official_date": TARGET,
            "checkpoint_utc_date": CHECKPOINT,
            "observation_id": self._reservation().observation_id,
            "certified_prediction_count": 2,
            "scored_count": 2,
            "void_count": 0,
            "pending_count": 0,
            "correct_count": 2,
            "incorrect_count": 0,
            "accuracy": 1.0,
            "mean_log_loss": 0.510825623766,
            "mean_brier_score": 0.16,
            "disclaimer": (
                "PROVISIONAL_DAILY_NO_VERDICT_SMALL_SAMPLE_"
                "DO_NOT_CHANGE_MODEL_OR_PROTOCOL"
            ),
        }
        self.assertEqual(documents.adjudications_csv_bytes, expected_csv)
        self.assertEqual(documents.daily_report, expected_report)
        self.assertEqual(
            documents.daily_report_bytes,
            shadow._canonical_json_file_bytes(expected_report),
        )
        self.assertTrue(documents.adjudications_csv_bytes.endswith(b"\n"))
        self.assertTrue(documents.daily_report_bytes.endswith(b"\n"))
        self.assertEqual(
            documents.adjudications_sha256,
            hashlib.sha256(expected_csv).hexdigest(),
        )

    def test_zero_scored_rows_have_exactly_three_null_metrics(self) -> None:
        """Une observation encore pending ne fabrique aucune metrique."""
        prediction = self._predictions()[0]
        row = replace(
            self._row(prediction, home_score=5, away_score=3),
            outcome_status="PENDING_NONTERMINAL",
            away_score=None,
            home_score=None,
            home_win=None,
            actual_winner=None,
            classification_correct=None,
            individual_log_loss=None,
            individual_brier_score=None,
            status_code_normalized="I",
            status_detail_normalized="IN PROGRESS",
            final_official_date=None,
        )
        source = replace(self._source(), predictions=(prediction,))
        batch = replace(
            self._batch(),
            adjudications=(row,),
            certified_prediction_count=1,
            scored_count=0,
            pending_count=1,
            correct_count=0,
        )
        documents = scoring.build_scoring_observation_documents(
            self._authority(), self._reservation(), source, batch
        )
        self.assertIsNone(documents.daily_report["accuracy"])
        self.assertIsNone(documents.daily_report["mean_log_loss"])
        self.assertIsNone(documents.daily_report["mean_brier_score"])

    def test_documents_are_byte_deterministic(self) -> None:
        """Deux constructions identiques produisent les memes octets."""
        first = self._build()
        second = self._build()
        self.assertEqual(first.adjudications_csv_bytes, second.adjudications_csv_bytes)
        self.assertEqual(first.daily_report_bytes, second.daily_report_bytes)
        self.assertEqual(first.adjudications_sha256, second.adjudications_sha256)
        self.assertEqual(first.daily_report_sha256, second.daily_report_sha256)

    def test_protocol_output_contract_cannot_drift(self) -> None:
        """Colonnes, statut et disclaimer restent ceux du protocole fige."""
        protocol = copy.deepcopy(self.protocol)
        protocol["daily_reporting"]["status"] = "FINAL"
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "diverge du protocole fige",
        ):
            scoring.build_scoring_observation_documents(
                self._authority(protocol=protocol),
                self._reservation(),
                self._source(),
                self._batch(),
            )

    def test_only_certified_nonempty_source_can_be_reported(self) -> None:
        """LOCAL_ONLY et les dates sans prediction ne traversent pas la barriere."""
        for status in (
            scoring.ScoringPredictionSourceStatus.LOCAL_ONLY,
            scoring.ScoringPredictionSourceStatus.COMPLETED_EMPTY,
            scoring.ScoringPredictionSourceStatus.FAILED,
            scoring.ScoringPredictionSourceStatus.MISSED,
        ):
            with self.subTest(status=status):
                with self.assertRaisesRegex(
                    scoring.ScoringAdjudicationError,
                    "cohorte certifiee non vide",
                ):
                    scoring.build_scoring_observation_documents(
                        self._authority(),
                        self._reservation(),
                        replace(self._source(), status=status),
                        self._batch(),
                    )

    def test_reservation_identity_and_hash_are_mandatory(self) -> None:
        """Un autre creneau ne peut recevoir ces documents."""
        reservation = replace(
            self._reservation(),
            reserved_marker_sha256="0" * 64,
        )
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "empreinte de la reservation",
        ):
            scoring.build_scoring_observation_documents(
                self._authority(), reservation, self._source(), self._batch()
            )

    def test_every_row_must_match_its_certified_prediction(self) -> None:
        """Equipe, identifiant et probabilite ne peuvent changer au scoring."""
        rows = list(self._batch().adjudications)
        rows[0] = replace(rows[0], home_team_id=999)
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "diverge de sa prediction certifiee",
        ):
            scoring.build_scoring_observation_documents(
                self._authority(),
                self._reservation(),
                self._source(),
                replace(self._batch(), adjudications=tuple(rows)),
            )

    def test_rows_must_be_in_prediction_id_order(self) -> None:
        """L'ordre canonique interdit deux CSV differents pour le meme lot."""
        batch = self._batch()
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "ne couvrent pas exactement",
        ):
            scoring.build_scoring_observation_documents(
                self._authority(),
                self._reservation(),
                self._source(),
                replace(batch, adjudications=tuple(reversed(batch.adjudications))),
            )

    def test_counts_are_recomputed_from_rows(self) -> None:
        """Le rapport ne fait jamais confiance a un compteur fourni."""
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "compteurs quotidiens",
        ):
            scoring.build_scoring_observation_documents(
                self._authority(),
                self._reservation(),
                self._source(),
                replace(self._batch(), correct_count=1, incorrect_count=1),
            )

    def test_individual_metrics_are_recomputed(self) -> None:
        """Une metrique modifiee ne peut entrer dans le rapport."""
        batch = self._batch()
        rows = list(batch.adjudications)
        rows[0] = replace(rows[0], individual_log_loss="0.000000000000")
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "metriques individuelles",
        ):
            scoring.build_scoring_observation_documents(
                self._authority(),
                self._reservation(),
                self._source(),
                replace(batch, adjudications=tuple(rows)),
            )

    def test_non_scored_row_cannot_hide_a_metric(self) -> None:
        """Les champs de mesure sont vides hors SCORED_FINAL."""
        batch = self._batch()
        rows = list(batch.adjudications)
        rows[0] = replace(
            rows[0],
            outcome_status="PENDING_NONTERMINAL",
            home_win=None,
            actual_winner=None,
            classification_correct=None,
        )
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "metrique interdite",
        ):
            scoring.build_scoring_observation_documents(
                self._authority(),
                self._reservation(),
                self._source(),
                replace(
                    batch,
                    adjudications=tuple(rows),
                    scored_count=1,
                    pending_count=1,
                    correct_count=1,
                ),
            )

    def test_outcome_evidence_hash_must_cover_every_row(self) -> None:
        """Une ligne ne peut provenir d'une autre observation MLB."""
        batch = self._batch()
        rows = list(batch.adjudications)
        rows[1] = replace(rows[1], outcome_evidence_sha256="0" * 64)
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "provenance invalide",
        ):
            scoring.build_scoring_observation_documents(
                self._authority(),
                self._reservation(),
                self._source(),
                replace(batch, adjudications=tuple(rows)),
            )

    def test_builder_has_no_io_network_model_or_prediction_side_effect(self) -> None:
        """Cette etape pure ne lit ni resultat, ni MLB, ni modele."""
        forbidden = AssertionError("effet de bord interdit")
        with (
            mock.patch.object(Path, "read_bytes", side_effect=forbidden),
            mock.patch.object(Path, "write_bytes", side_effect=forbidden),
            mock.patch.object(scoring.requests, "get", side_effect=forbidden),
            mock.patch.object(
                scoring,
                "load_immutable_scoring_prediction_source",
                side_effect=forbidden,
            ),
            mock.patch.object(
                scoring.shadow,
                "_publish_exclusive_verified",
                side_effect=forbidden,
            ),
        ):
            documents = self._build()
        self.assertEqual(documents.daily_report["scored_count"], 2)


if __name__ == "__main__":
    unittest.main()
