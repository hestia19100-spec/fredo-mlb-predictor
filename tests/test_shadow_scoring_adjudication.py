"""Tests de l'adjudication MLB pure du protocole prospectif 2026."""

from __future__ import annotations

import base64
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import unittest
from unittest import mock

from src import shadow_prediction as shadow
from src import shadow_scoring as scoring


TARGET_DATE = "2026-09-10"
CHECKPOINT_DATE = "2026-09-11"
BATCH_ID = "a" * 64
HTTP_DATE = "Fri, 11 Sep 2026 06:00:02 GMT"
HTTP_DATE_UTC = "2026-09-11T06:00:02Z"


class ShadowScoringAdjudicationTests(unittest.TestCase):
    """Chaque prediction est conservee et adjugee par les regles figees."""

    def _game(
        self,
        game_id: int,
        *,
        season: int | str = "2026",
        game_type: str = "R",
        away_team_id: int = 10,
        home_team_id: int = 20,
        status_code: str = "F",
        status_detail: str = "Final",
        official_date: str = TARGET_DATE,
        away_score: object = 3,
        home_score: object = 5,
    ) -> dict[str, object]:
        return {
            "gamePk": game_id,
            "season": season,
            "gameType": game_type,
            "officialDate": official_date,
            "status": {
                "statusCode": status_code,
                "detailedState": status_detail,
            },
            "teams": {
                "away": {
                    "team": {"id": away_team_id},
                    "score": away_score,
                },
                "home": {
                    "team": {"id": home_team_id},
                    "score": home_score,
                },
            },
        }

    def _evidence(
        self,
        games: list[dict[str, object]],
        *,
        checkpoint_date: str = CHECKPOINT_DATE,
        date_blocks: list[dict[str, object]] | None = None,
    ) -> scoring.ScoringOutcomeObservationEvidence:
        blocks = (
            [{"date": TARGET_DATE, "games": games}]
            if date_blocks is None
            else date_blocks
        )
        occurrence_count = sum(len(block["games"]) for block in blocks)
        body = shadow._canonical_json_bytes(
            {
                "dates": blocks,
                "totalGames": occurrence_count,
            }
        )
        target_url = scoring._expected_outcome_request_url(TARGET_DATE)
        raw = {
            "evidence_schema_version": 1,
            "target_official_date": TARGET_DATE,
            "checkpoint_utc_date": checkpoint_date,
            "request_url": target_url,
            "request_method": "GET",
            "application_request_headers": dict(
                scoring.OUTCOME_REQUEST_HEADERS
            ),
            "effective_url": target_url,
            "response_status_code": 200,
            "response_redirect_count": 0,
            "selected_response_headers": {
                "date": HTTP_DATE,
                "content-type": "application/json",
            },
            "response_received_at_utc": HTTP_DATE_UTC,
            "response_body_base64": base64.b64encode(body).decode("ascii"),
            "response_body_sha256": hashlib.sha256(body).hexdigest(),
        }
        canonical_json = shadow._canonical_json_file_bytes(raw)
        canonical_gzip = shadow._canonical_gzip_bytes(canonical_json)
        return scoring.ScoringOutcomeObservationEvidence(
            target_official_date=TARGET_DATE,
            checkpoint_utc_date=checkpoint_date,
            outcome_http_date_utc=HTTP_DATE_UTC,
            response_received_at_utc=HTTP_DATE_UTC,
            response_body_sha256=hashlib.sha256(body).hexdigest(),
            flattened_occurrence_count=occurrence_count,
            raw_evidence=raw,
            canonical_json_bytes=canonical_json,
            canonical_gzip_bytes=canonical_gzip,
            canonical_gzip_sha256=hashlib.sha256(canonical_gzip).hexdigest(),
        )

    def _prediction(
        self,
        game_id: int,
        *,
        prediction_hex: str = "1",
        away_team_id: int = 10,
        home_team_id: int = 20,
        p_home_win: str = "0.60000000000000000",
        p_away_win: str = "0.40000000000000000",
    ) -> scoring.CertifiedScoringPrediction:
        occurrence_key = hashlib.sha256(
            f"occurrence-{game_id}".encode("ascii")
        ).hexdigest()
        return scoring.CertifiedScoringPrediction(
            prediction_id=prediction_hex * 64,
            batch_id=BATCH_ID,
            game_id=game_id,
            occurrence_key=occurrence_key,
            season=2026,
            target_official_date=TARGET_DATE,
            away_team_id=away_team_id,
            home_team_id=home_team_id,
            p_home_win=p_home_win,
            p_away_win=p_away_win,
        )

    def test_home_final_is_scored_with_exact_individual_metrics(self) -> None:
        """Une finale le jour cible produit une cible et deux metriques."""
        prediction = self._prediction(1)
        batch = scoring.adjudicate_scoring_predictions(
            [prediction],
            self._evidence([self._game(1)]),
        )
        row = batch.adjudications[0]

        self.assertEqual(row.outcome_status, "SCORED_FINAL")
        self.assertEqual((row.away_score, row.home_score), (3, 5))
        self.assertEqual(row.home_win, 1)
        self.assertEqual(row.actual_winner, "HOME")
        self.assertEqual(row.predicted_side, "HOME")
        self.assertEqual(row.classification_correct, 1)
        self.assertEqual(row.individual_log_loss, "0.510825623766")
        self.assertEqual(row.individual_brier_score, "0.160000000000")
        self.assertEqual(
            (batch.certified_prediction_count, batch.scored_count),
            (1, 1),
        )
        self.assertEqual(
            (batch.void_count, batch.pending_count),
            (0, 0),
        )
        self.assertEqual((batch.correct_count, batch.incorrect_count), (1, 0))

    def test_away_final_and_probability_tie_are_deterministic(self) -> None:
        """Le seuil exactement egal a 0,5 choisit HOME comme preenregistre."""
        prediction = self._prediction(
            2,
            p_home_win="0.50000000000000000",
            p_away_win="0.50000000000000000",
        )
        batch = scoring.adjudicate_scoring_predictions(
            [prediction],
            self._evidence([self._game(2, away_score=7, home_score=2)]),
        )
        row = batch.adjudications[0]
        self.assertEqual(row.actual_winner, "AWAY")
        self.assertEqual(row.predicted_side, "HOME")
        self.assertEqual(row.classification_correct, 0)
        self.assertEqual(row.individual_log_loss, "0.693147180560")
        self.assertEqual(row.individual_brier_score, "0.250000000000")
        self.assertEqual((batch.correct_count, batch.incorrect_count), (0, 1))

    def test_status_normalization_is_exact(self) -> None:
        """Espaces et casse suivent strictement la formule preenregistree."""
        batch = scoring.adjudicate_scoring_predictions(
            [self._prediction(3)],
            self._evidence(
                [
                    self._game(
                        3,
                        status_code="  f  ",
                        status_detail="  completed\t early  ",
                    )
                ]
            ),
        )
        row = batch.adjudications[0]
        self.assertEqual(row.status_code_normalized, "F")
        self.assertEqual(row.status_detail_normalized, "COMPLETED EARLY")

    def test_missing_nonterminal_and_postponed_are_pending(self) -> None:
        """Avant la date limite, aucune absence ne devient un resultat."""
        predictions = [
            self._prediction(10, prediction_hex="1"),
            self._prediction(11, prediction_hex="2"),
            self._prediction(12, prediction_hex="3"),
        ]
        evidence = self._evidence(
            [
                self._game(
                    11,
                    status_code="I",
                    status_detail="In Progress",
                ),
                self._game(
                    12,
                    status_code="DR",
                    status_detail="Postponed",
                ),
            ]
        )
        batch = scoring.adjudicate_scoring_predictions(predictions, evidence)
        self.assertEqual(
            [row.outcome_status for row in batch.adjudications],
            [
                "PENDING_MISSING_FROM_OBSERVATION",
                "PENDING_NONTERMINAL",
                "PENDING_POSTPONED",
            ],
        )
        self.assertEqual((batch.scored_count, batch.void_count), (0, 0))
        self.assertEqual(batch.pending_count, 3)
        for row in batch.adjudications:
            self.assertIsNone(row.classification_correct)
            self.assertIsNone(row.individual_log_loss)
            self.assertIsNone(row.individual_brier_score)

    def test_deadline_converts_every_pending_family(self) -> None:
        """Le checkpoint final applique seulement les conversions figees."""
        predictions = [
            self._prediction(20, prediction_hex="1"),
            self._prediction(21, prediction_hex="2"),
            self._prediction(22, prediction_hex="3"),
        ]
        evidence = self._evidence(
            [
                self._game(
                    21,
                    status_code="I",
                    status_detail="Delayed",
                ),
                self._game(
                    22,
                    status_code="DI",
                    status_detail="Postponed",
                ),
            ],
            checkpoint_date="2026-10-12",
        )
        batch = scoring.adjudicate_scoring_predictions(predictions, evidence)
        self.assertEqual(
            [row.outcome_status for row in batch.adjudications],
            [
                "VOID_UNRESOLVED_AT_DEADLINE",
                "VOID_UNRESOLVED_AT_DEADLINE",
                "VOID_POSTPONED_AT_DEADLINE",
            ],
        )
        self.assertEqual((batch.void_count, batch.pending_count), (3, 0))

    def test_cancelled_is_terminal_without_metrics(self) -> None:
        """Une annulation est conservee mais ne devient jamais une cible."""
        batch = scoring.adjudicate_scoring_predictions(
            [self._prediction(30)],
            self._evidence(
                [
                    self._game(
                        30,
                        status_code="CI",
                        status_detail="Cancelled",
                    )
                ]
            ),
        )
        row = batch.adjudications[0]
        self.assertEqual(row.outcome_status, "VOID_CANCELLED")
        self.assertEqual(batch.void_count, 1)
        self.assertIsNone(row.away_score)
        self.assertIsNone(row.home_score)
        self.assertIsNone(row.actual_winner)

    def test_rescheduled_final_is_void_but_keeps_audit_scores(self) -> None:
        """Une finale deplacee explique le void sans calculer de metrique."""
        batch = scoring.adjudicate_scoring_predictions(
            [self._prediction(31)],
            self._evidence(
                [self._game(31, official_date="2026-09-12")]
            ),
        )
        row = batch.adjudications[0]
        self.assertEqual(
            row.outcome_status,
            "VOID_RESCHEDULED_OFFICIAL_DATE",
        )
        self.assertEqual(row.final_official_date, "2026-09-12")
        self.assertEqual((row.away_score, row.home_score), (3, 5))
        self.assertIsNone(row.home_win)
        self.assertIsNone(row.actual_winner)
        self.assertIsNone(row.individual_log_loss)

    def test_later_consistent_final_overrides_nonterminal(self) -> None:
        """Un match suspendu puis final est reduit a sa finale coherente."""
        evidence = self._evidence(
            [],
            date_blocks=[
                {
                    "date": TARGET_DATE,
                    "games": [
                        self._game(
                            40,
                            status_code="S",
                            status_detail="Suspended",
                        )
                    ],
                },
                {
                    "date": "2026-09-11",
                    "games": [self._game(40)],
                },
            ],
        )
        batch = scoring.adjudicate_scoring_predictions(
            [self._prediction(40)],
            evidence,
        )
        self.assertEqual(batch.adjudications[0].outcome_status, "SCORED_FINAL")

    def test_conflicting_final_occurrences_abort(self) -> None:
        """Scores ou dates finales differents interdisent l'adjudication."""
        variants = (
            self._game(41, away_score=4, home_score=5),
            self._game(41, official_date="2026-09-12"),
        )
        for conflicting in variants:
            with self.subTest(conflicting=conflicting):
                with self.assertRaisesRegex(
                    scoring.ScoringAdjudicationError,
                    "CONFLICTING_FINAL_SCORES_OR_FINAL_OFFICIAL_DATES",
                ):
                    scoring.adjudicate_scoring_predictions(
                        [self._prediction(41)],
                        self._evidence([self._game(41), conflicting]),
                    )

    def test_duplicate_occurrence_identity_must_be_exact(self) -> None:
        """Un gamePk partage par deux identites est une erreur structurelle."""
        conflicts = (
            self._game(42, season=2026),
            self._game(42, away_team_id=11),
            self._game(42, home_team_id=21),
        )
        for conflicting in conflicts:
            with self.subTest(conflicting=conflicting):
                with self.assertRaisesRegex(
                    scoring.ScoringAdjudicationError,
                    "CONFLICTING_MLB_OCCURRENCE_IDENTITY",
                ):
                    scoring.adjudicate_scoring_predictions(
                        [self._prediction(42)],
                        self._evidence([self._game(42), conflicting]),
                    )

    def test_occurrence_status_family_conflict_aborts(self) -> None:
        """Code final et detail reporte ne peuvent coexister."""
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "STATUS_FAMILY_CONFLICT_WITHIN_OCCURRENCE",
        ):
            scoring.adjudicate_scoring_predictions(
                [self._prediction(43)],
                self._evidence(
                    [
                        self._game(
                            43,
                            status_code="F",
                            status_detail="Postponed",
                        )
                    ]
                ),
            )

    def test_cancelled_can_follow_postponed_but_not_nonterminal(self) -> None:
        """La reduction annulee accepte seulement les familles prevues."""
        postponed = self._game(
            44,
            status_code="DR",
            status_detail="Postponed",
        )
        cancelled = self._game(
            44,
            status_code="CR",
            status_detail="Cancelled",
        )
        batch = scoring.adjudicate_scoring_predictions(
            [self._prediction(44)],
            self._evidence([postponed, cancelled]),
        )
        self.assertEqual(batch.adjudications[0].outcome_status, "VOID_CANCELLED")

        nonterminal = self._game(
            44,
            status_code="I",
            status_detail="In Progress",
        )
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "contredit",
        ):
            scoring.adjudicate_scoring_predictions(
                [self._prediction(44)],
                self._evidence([cancelled, nonterminal]),
            )

    def test_nonterminal_precedes_postponed_without_silent_void(self) -> None:
        """Sans finale ou annulation, un etat actif reste prioritaire."""
        batch = scoring.adjudicate_scoring_predictions(
            [self._prediction(45)],
            self._evidence(
                [
                    self._game(
                        45,
                        status_code="DR",
                        status_detail="Postponed",
                    ),
                    self._game(
                        45,
                        status_code="I",
                        status_detail="In Progress",
                    ),
                ]
            ),
        )
        self.assertEqual(
            batch.adjudications[0].outcome_status,
            "PENDING_NONTERMINAL",
        )

    def test_final_scores_are_strict_json_integers_nonnegative_not_tied(self) -> None:
        """Aucune finale douteuse ne peut entrer dans les metriques."""
        invalid_scores = (
            (None, 5),
            (True, 5),
            (3.0, 5),
            (-1, 5),
            (5, 5),
        )
        for away_score, home_score in invalid_scores:
            with self.subTest(away=away_score, home=home_score):
                with self.assertRaisesRegex(
                    scoring.ScoringAdjudicationError,
                    "FINAL_SCORE_MISSING_NEGATIVE_BOOLEAN_NONINTEGER_OR_TIED",
                ):
                    scoring.adjudicate_scoring_predictions(
                        [self._prediction(50)],
                        self._evidence(
                            [
                                self._game(
                                    50,
                                    away_score=away_score,
                                    home_score=home_score,
                                )
                            ]
                        ),
                    )

    def test_prediction_and_outcome_identities_must_match(self) -> None:
        """Saison, type et equipes ne sont jamais rapproches approximativement."""
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "TEAM_IDENTITY_MISMATCH",
        ):
            scoring.adjudicate_scoring_predictions(
                [self._prediction(51, away_team_id=99)],
                self._evidence([self._game(51)]),
            )
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "SEASON_MISMATCH",
        ):
            scoring.adjudicate_scoring_predictions(
                [replace(self._prediction(51), season=2025)],
                self._evidence([self._game(51)]),
            )
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "GAME_TYPE_MISMATCH",
        ):
            scoring.adjudicate_scoring_predictions(
                [self._prediction(51)],
                self._evidence([self._game(51, game_type="P")]),
            )

    def test_duplicate_predictions_and_game_ids_abort(self) -> None:
        """Une prediction ou un match ne peut compter deux fois."""
        first = self._prediction(60, prediction_hex="1")
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "DUPLICATE_PREDICTION_ID",
        ):
            scoring.adjudicate_scoring_predictions(
                [first, replace(first, game_id=61)],
                self._evidence([self._game(60), self._game(61)]),
            )
        with self.assertRaisesRegex(
            scoring.ScoringAdjudicationError,
            "DUPLICATE_GAME_ID_WITHIN_BATCH",
        ):
            scoring.adjudicate_scoring_predictions(
                [first, replace(first, prediction_id="2" * 64)],
                self._evidence([self._game(60)]),
            )

    def test_probabilities_are_exact_complementary_canonical_text(self) -> None:
        """Les probabilites sont copiees sans conversion ni approximation."""
        valid = self._prediction(
            70,
            p_home_win="0.6",
            p_away_win="0.4",
        )
        row = scoring.adjudicate_scoring_predictions(
            [valid],
            self._evidence([self._game(70)]),
        ).adjudications[0]
        self.assertEqual((row.p_home_win, row.p_away_win), ("0.6", "0.4"))

        invalid_pairs = (
            ("0.6", "0.5"),
            ("6e-1", "0.4"),
            ("-0.1", "1.1"),
            ("NaN", "NaN"),
        )
        for home, away in invalid_pairs:
            with self.subTest(home=home, away=away):
                with self.assertRaises(scoring.ScoringAdjudicationError):
                    scoring.adjudicate_scoring_predictions(
                        [
                            self._prediction(
                                70,
                                p_home_win=home,
                                p_away_win=away,
                            )
                        ],
                        self._evidence([self._game(70)]),
                    )

    def test_every_prediction_is_retained_in_prediction_id_order(self) -> None:
        """Le resultat ne peut filtrer ni reordonner la cohorte selon l'issue."""
        predictions = [
            self._prediction(81, prediction_hex="f"),
            self._prediction(80, prediction_hex="0"),
            self._prediction(82, prediction_hex="8"),
        ]
        batch = scoring.adjudicate_scoring_predictions(
            predictions,
            self._evidence(
                [
                    self._game(81),
                    self._game(
                        82,
                        status_code="DR",
                        status_detail="Postponed",
                    ),
                ]
            ),
        )
        self.assertEqual(
            [row.prediction_id for row in batch.adjudications],
            sorted(prediction.prediction_id for prediction in predictions),
        )
        self.assertEqual(batch.certified_prediction_count, 3)
        self.assertEqual(
            batch.scored_count + batch.void_count + batch.pending_count,
            3,
        )

    def test_adjudication_csv_projection_has_the_exact_schema_order(self) -> None:
        """La projection prepare exactement les 24 colonnes figees."""
        row = scoring.adjudicate_scoring_predictions(
            [self._prediction(90)],
            self._evidence([self._game(90)]),
        ).adjudications[0]
        values = row.as_csv_row()
        self.assertEqual(len(values), len(scoring._ADJUDICATION_COLUMNS))
        self.assertEqual(scoring._ADJUDICATION_COLUMNS[0], "prediction_id")
        self.assertEqual(scoring._ADJUDICATION_COLUMNS[-1], "outcome_evidence_sha256")
        self.assertEqual(values[0], row.prediction_id)
        self.assertEqual(values[-1], row.outcome_evidence_sha256)

    def test_tampered_evidence_aborts_before_any_adjudication(self) -> None:
        """La reduction relit et revalide toute la chaine de preuve brute."""
        evidence = self._evidence([self._game(91)])
        forged = replace(evidence, response_body_sha256="0" * 64)
        with self.assertRaises(shadow.ShadowPredictionError):
            scoring.adjudicate_scoring_predictions(
                [self._prediction(91)],
                forged,
            )

    def test_pure_adjudication_reads_no_network_sqlite_or_model(self) -> None:
        """Apres l'archive, l'adjudication reste entierement deterministe."""
        evidence = self._evidence([self._game(92)])
        prediction = self._prediction(92)
        forbidden = AssertionError("source interdite")
        with (
            mock.patch.object(scoring.requests, "get", side_effect=forbidden),
            mock.patch.object(
                scoring.shadow.subprocess,
                "run",
                side_effect=forbidden,
            ),
            mock.patch.object(
                scoring.shadow,
                "_read_regular_project_file",
                side_effect=forbidden,
            ),
        ):
            batch = scoring.adjudicate_scoring_predictions(
                [prediction],
                evidence,
            )
        self.assertEqual(batch.scored_count, 1)

        source = Path(scoring.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import sqlite3", source)
        self.assertNotIn("import joblib", source)
        self.assertNotIn("from src.database", source)


if __name__ == "__main__":
    unittest.main()
