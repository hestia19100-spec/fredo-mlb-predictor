"""Tests fermes du protocole prospectif de scoring MLB 2026."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import unittest

from src import shadow_scoring_registration as registration


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_DIRECTORY.joinpath(
    *registration.SCORING_PROTOCOL_RELATIVE_PATH.parts
)


def _load_protocol() -> tuple[bytes, dict[str, object]]:
    content = PROTOCOL_PATH.read_bytes()
    return content, json.loads(content.decode("utf-8"))


class ShadowScoringProtocolTests(unittest.TestCase):
    """Les choix de scoring doivent etre exacts avant le premier resultat."""

    def test_protocol_is_canonical_and_hash_is_frozen(self) -> None:
        """Le fichier est un JSON canonique lie a une empreinte unique."""
        content, protocol = _load_protocol()
        canonical = (
            json.dumps(
                protocol,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        self.assertEqual(content, canonical)
        self.assertEqual(
            hashlib.sha256(content).hexdigest(),
            registration.EXPECTED_SCORING_PROTOCOL_SHA256,
        )
        registration._validate_scoring_protocol(protocol)

    def test_horizon_is_fixed_to_all_eighteen_regular_season_dates(self) -> None:
        """Aucun arret anticipe ou choix de jour apres resultat n'est possible."""
        _, protocol = _load_protocol()
        self.assertEqual(
            protocol["fixed_horizon"],
            {
                "calendar_days": 18,
                "game_type": "R",
                "no_early_stop": True,
                "season": 2026,
                "target_official_date_end": "2026-09-27",
                "target_official_date_start": "2026-09-10",
            },
        )
        dates = protocol["intent_to_observe"]["dates"]
        self.assertEqual(len(dates), 18)
        self.assertEqual(dates[0], "2026-09-10")
        self.assertEqual(dates[-1], "2026-09-27")
        self.assertEqual(len(set(dates)), 18)
        self.assertEqual(
            [date.fromisoformat(value).day for value in dates],
            list(range(10, 28)),
        )

    def test_first_batch_is_bound_before_any_outcome(self) -> None:
        """Le lot deja certifie est nomme par tous ses identifiants exacts."""
        _, protocol = _load_protocol()
        first = protocol["authorities"]["first_certified_batch"]
        self.assertEqual(
            first,
            {
                "batch_id": registration.EXPECTED_FIRST_BATCH_ID,
                "certification_commit": (
                    registration.EXPECTED_FIRST_CERTIFICATION_COMMIT
                ),
                "certification_path": (
                    registration.EXPECTED_FIRST_CERTIFICATION_PATH
                ),
                "certification_sha256": (
                    registration.EXPECTED_FIRST_CERTIFICATION_SHA256
                ),
                "earliest_predicted_start_utc": (
                    registration.EXPECTED_FIRST_START_UTC
                ),
                "predictions_sha256": (
                    registration.EXPECTED_FIRST_PREDICTIONS_SHA256
                ),
                "receipt_sha256": registration.EXPECTED_FIRST_RECEIPT_SHA256,
                "results_commit": registration.EXPECTED_FIRST_RESULTS_COMMIT,
                "target_official_date": registration.EXPECTED_FIRST_TARGET_DATE,
            },
        )
        boundary = protocol["prospective_boundary"]
        first_start = datetime.fromisoformat(
            boundary["first_predicted_start_utc"].replace("Z", "+00:00")
        )
        self.assertEqual(first_start.tzinfo, timezone.utc)
        self.assertTrue(boundary["outcomes_must_not_be_read_before_remote_registration"])
        self.assertEqual(
            boundary["minimum_remote_registration_lead_seconds"], 3600
        )

    def test_prediction_cohort_is_complete_and_git_only(self) -> None:
        """Toutes les predictions certifiees viennent de blobs immuables."""
        _, protocol = _load_protocol()
        cohort = protocol["prediction_cohort"]
        source = protocol["prediction_source"]
        self.assertEqual(
            cohort["inclusion"],
            "ALL_ROWS_FROM_EVERY_NONEMPTY_COMPLETED_EXACT_BATCH_WITH_VALID_PROSPECTIVE_CERTIFICATION",
        )
        self.assertTrue(cohort["no_confidence_probability_team_or_outcome_filter"])
        self.assertTrue(cohort["no_retroactive_shadow_batch_or_certification"])
        self.assertEqual(
            source["source_of_truth"],
            "EXACT_IMMUTABLE_GIT_BLOBS_FROM_CERTIFIED_RESULTS_COMMIT",
        )
        self.assertTrue(source["working_tree_sqlite_and_regenerated_predictions_forbidden"])
        self.assertEqual(
            source["required_result_files_lexicographic"],
            [
                "COMPLETED",
                "RESERVED",
                "activation_reverification.remote.json.gz",
                "candidate_ledger.csv",
                "features.csv",
                "predictions.csv",
                "receipt.json",
                "source_snapshot.json.gz",
            ],
        )
        discovery = source["discovery_algorithm_exact"]
        self.assertEqual(len(discovery), 11)
        self.assertIn("ITERATE_EXACTLY_INTENT_TO_OBSERVE_DATES", discovery[0])
        self.assertIn("git_show", discovery[4])
        self.assertIn("COMPLETED_EMPTY", discovery[5])
        self.assertIn("LOCAL_ONLY", discovery[6])
        self.assertIn("STRUCTURAL_ABORT", discovery[8])
        self.assertIn("raw_remote_evidence_path", discovery[9])
        self.assertIn("SAME_UNIQUE_INTRODUCTION_COMMIT", discovery[9])
        self.assertIn("git_show", discovery[9])
        self.assertIn("CANONICAL_JSON", discovery[9])

    def test_outcomes_use_fresh_archived_mlb_evidence_not_mutable_sqlite(self) -> None:
        """Le score final devra etre prouve par une nouvelle reponse brute."""
        _, protocol = _load_protocol()
        observation = protocol["outcome_observation"]
        self.assertEqual(
            observation["source_of_truth"],
            "FRESH_VERIFIED_ARCHIVED_RAW_MLB_RESPONSE_NOT_MUTABLE_GAMES_TABLE",
        )
        self.assertEqual(
            observation["database_role"],
            "SECONDARY_INTEGRITY_CHECK_ONLY_NOT_SOURCE_OF_TRUTH",
        )
        self.assertEqual(
            observation["exact_query_parameters_per_target_date"],
            {
                "endDate": "TARGET_OFFICIAL_DATE",
                "gameTypes": "R",
                "hydrate": "probablePitcher",
                "sportId": 1,
                "startDate": "TARGET_OFFICIAL_DATE",
            },
        )
        self.assertEqual(observation["http_requirements"]["status_code"], 200)
        self.assertTrue(observation["http_requirements"]["no_redirect"])
        self.assertTrue(observation["raw_body_and_http_envelope_archived"])

    def test_schedule_edge_cases_are_predeclared(self) -> None:
        """Reports, annulations et suspensions ne pourront pas etre choisis."""
        _, protocol = _load_protocol()
        rules = protocol["outcome_adjudication"]
        self.assertEqual(
            rules["result_by_reduced_family"]["POSTPONED_BEFORE_DEADLINE"],
            "PENDING_POSTPONED",
        )
        self.assertEqual(
            rules["result_by_reduced_family"]["CANCELLED"],
            "VOID_CANCELLED",
        )
        self.assertEqual(
            rules["result_by_reduced_family"]["FINAL_DIFFERENT_OFFICIAL_DATE"],
            "VOID_RESCHEDULED_OFFICIAL_DATE",
        )
        self.assertIn("SUSPENDED_OR_DELAYED_NONTERMINAL", rules["suspended_rule"])
        self.assertEqual(
            rules["deadline_conversion"]["PENDING_POSTPONED"],
            "VOID_POSTPONED_AT_DEADLINE",
        )
        self.assertEqual(
            rules["deadline_conversion"]["PENDING_NONTERMINAL"],
            "VOID_UNRESOLVED_AT_DEADLINE",
        )
        self.assertIn(
            "DUPLICATE_GAME_ID_WITHIN_BATCH",
            protocol["identity_and_linkage"]["abort_conditions"],
        )
        self.assertFalse(protocol["identity_and_linkage"]["silent_exclusion_allowed"])

    def test_adjudication_has_exact_status_precedence_and_immutable_transitions(self) -> None:
        """Une observation tardive ne peut ni changer ni choisir un resultat."""
        _, protocol = _load_protocol()
        rules = protocol["outcome_adjudication"]
        self.assertEqual(
            rules["precedence_after_duplicate_reduction"],
            ["FINAL", "CANCELLED", "NONTERMINAL", "POSTPONED", "MISSING"],
        )
        family = rules["occurrence_family_classification"]
        self.assertEqual(family["final_codes"], ["F", "FG", "FO", "FR"])
        self.assertIn("SET_SIZE_GREATER_THAN_ONE_ABORTS", family["formula"])
        reduction = rules["duplicate_occurrence_reduction_exact"]
        self.assertIn("SAME_officialDate", reduction[2])
        self.assertIn("CANCELLED_OR_POSTPONED", reduction[3])
        terminal = rules["terminal_immutability"]
        self.assertIn("SCORED_FINAL", terminal["terminal_statuses"])
        self.assertIn("ANY_CHANGE_ABORTS", terminal["later_observation_rule"])

    def test_pending_results_use_new_append_only_observation_directories(self) -> None:
        """Un match en attente evolue par ajout, jamais par reecriture."""
        _, protocol = _load_protocol()
        observation = protocol["outcome_observation"]
        checkpoints = observation["checkpoint_policy"]
        publication = protocol["output_publication"]
        self.assertEqual(
            checkpoints["checkpoint_key"], "UTC_CALENDAR_DATE_YYYY_MM_DD"
        )
        self.assertIn("WHILE_ANY_PREDICTION", checkpoints["next_checkpoint_formula"])
        self.assertIn("AT_MOST_ONE_IMMUTABLE_OBSERVATION_SLOT", checkpoints["one_attempt_maximum"])
        self.assertIn(
            "observations/CHECKPOINT_UTC_DATE",
            publication["observation_path_template"],
        )
        self.assertEqual(
            publication["observation_success_write_order_exact"],
            [
                "RESERVED",
                "outcome_observation.remote.json.gz",
                "adjudications.csv",
                "daily_report.json",
                "observation_receipt.json",
                "COMPLETED",
            ],
        )
        self.assertTrue(
            protocol["daily_reporting"][
                "pending_rows_can_transition_only_in_a_new_observation_directory"
            ]
        )
        self.assertTrue(protocol["daily_reporting"]["terminal_rows_are_never_rewritten"])
        metric_rules = protocol["daily_reporting"]["metric_value_rules"]
        self.assertEqual(
            metric_rules[
                "accuracy_mean_log_loss_and_mean_brier_score_when_scored_count_is_zero"
            ],
            "JSON_NULL",
        )
        self.assertIn("IF_AND_ONLY_IF", metric_rules["null_equivalence"])

    def test_output_schemas_and_final_cohort_ledger_are_exact(self) -> None:
        """Chaque future preuve a un schema ferme et une place unique."""
        _, protocol = _load_protocol()
        publication = protocol["output_publication"]
        schemas = publication["schemas"]
        self.assertEqual(
            schemas["adjudications_csv_columns_exact_order"],
            protocol["per_prediction_output"]["adjudication_columns_exact_order"],
        )
        self.assertEqual(
            schemas["settled_predictions_csv_columns_exact_order"][-1],
            "terminal_observation_id",
        )
        self.assertEqual(
            publication["target_settlement"]["write_order_exact"],
            [
                "RESERVED",
                "settled_predictions.csv",
                "settlement_receipt.json",
                "SETTLED.json",
            ],
        )
        self.assertEqual(
            publication["final_publication"]["write_order_exact"],
            [
                "RESERVED",
                "intent_ledger.csv",
                "settled_predictions.csv",
                "final_report.json",
                "final_receipt.json",
                "FINALIZED",
            ],
        )
        self.assertEqual(
            schemas["cohort_intent_ledger_csv_columns_exact_order"][0],
            "target_official_date",
        )
        self.assertEqual(protocol["intent_to_observe"]["ledger_row_count"], 18)
        self.assertNotIn(
            "observation_receipt_sha256",
            schemas["daily_report_keys_exact_set"],
        )
        self.assertIn(
            "receipt_finalized_at_utc",
            schemas["observation_receipt_keys_exact_set"],
        )
        self.assertIn("NO_EARLY_STOP", schemas["final_negative_attestations_keys_exact_set"])
        self.assertIn(
            "target_official_date",
            schemas["settlement_reserved_marker_keys_exact_set"],
        )
        self.assertIn(
            "horizon_start",
            schemas["final_reserved_marker_keys_exact_set"],
        )
        self.assertIn(
            "NO_PRECEDING_FILE_CONTAINS",
            publication["observation_hash_direction"],
        )

    def test_tomorrow_report_is_descriptive_and_cannot_change_the_trial(self) -> None:
        """Fred peut voir juste/faux sans transformer n=5 en conclusion."""
        _, protocol = _load_protocol()
        daily = protocol["daily_reporting"]
        self.assertTrue(daily["publication_allowed"])
        self.assertEqual(daily["status"], "PROVISIONAL_DAILY_NO_VERDICT")
        self.assertFalse(daily["cumulative_metrics_before_final_report_allowed"])
        self.assertIn("classification_correct", daily["individual_fields_allowed"])
        self.assertIn("individual_log_loss", daily["individual_fields_allowed"])
        self.assertIn("individual_brier_score", daily["individual_fields_allowed"])
        self.assertIn("SMALL_SAMPLE", daily["minimum_disclaimer"])
        self.assertTrue(protocol["fixed_horizon"]["no_early_stop"])

    def test_primary_metric_reference_and_probability_rules_are_exact(self) -> None:
        """La comparaison probabiliste reprend la reference historique figee."""
        _, protocol = _load_protocol()
        metrics = protocol["aggregate_metrics"]
        self.assertEqual(metrics["primary"]["metric"], "MEAN_LOG_LOSS")
        self.assertEqual(
            metrics["primary"]["formula"],
            "(reference_log_loss-model_log_loss)/reference_log_loss",
        )
        self.assertEqual(metrics["reference"]["home_wins"], 3623)
        self.assertEqual(metrics["reference"]["games"], 6815)
        self.assertFalse(metrics["reference"]["reestimated_on_2026"])
        self.assertEqual(metrics["probability_handling"]["log_loss_clip_epsilon"], 1e-15)
        self.assertEqual(metrics["probability_handling"]["logarithm"], "NATURAL")
        self.assertTrue(metrics["brier_score"]["uses_unclipped_probability"])

    def test_uncertainty_and_final_verdict_cannot_be_moved_after_results(self) -> None:
        """Bootstrap, couverture et seuils sont tous fixes aujourd'hui."""
        _, protocol = _load_protocol()
        uncertainty = protocol["uncertainty"]
        self.assertEqual(uncertainty["calendar_axis_days"], 18)
        self.assertEqual(uncertainty["block_length_days"], 7)
        self.assertEqual(uncertainty["expected_complete_blocks_per_replicate"], 2)
        self.assertEqual(uncertainty["expected_final_partial_block_days"], 4)
        self.assertEqual(uncertainty["blocks_drawn_per_replicate_exact"], 3)
        self.assertEqual(
            uncertainty["sampled_start_index_domain_inclusive"], [0, 11]
        )
        self.assertEqual(uncertainty["sampled_day_slots_per_replicate_exact"], 18)
        self.assertIn("FIRST_4_DAYS", uncertainty["draw_algorithm_exact"])
        self.assertEqual(uncertainty["replications"], 5000)
        self.assertEqual(uncertainty["random_seed"], 42)
        self.assertTrue(uncertainty["include_days_without_scorable_games"])

        verdict = protocol["verdict"]
        self.assertEqual(verdict["coverage_gate"]["minimum_scorable_games"], 100)
        self.assertEqual(
            verdict["coverage_gate"]["minimum_operationally_valid_dates"], 18
        )
        self.assertEqual(verdict["coverage_gate"]["maximum_missed_dates"], 0)
        self.assertEqual(verdict["coverage_gate"]["maximum_failed_dates"], 0)
        self.assertEqual(verdict["coverage_gate"]["maximum_local_only_dates"], 0)
        self.assertEqual(
            verdict["coverage_gate"][
                "minimum_scorable_fraction_of_certified_predictions"
            ],
            0.95,
        )
        self.assertEqual(
            verdict["decision_order"][0]["label"],
            "INSUFFICIENT_PROSPECTIVE_COVERAGE",
        )
        self.assertIn("relative_log_loss_improvement >= 0.005", verdict["primary_supported"])

    def test_publication_is_separate_append_only_and_no_betting_is_allowed(self) -> None:
        """Le scoring ne touche jamais les preuves shadow et ne parle pas ROI."""
        _, protocol = _load_protocol()
        publication = protocol["output_publication"]
        self.assertTrue(publication["root"].startswith("shadow_scores/"))
        self.assertEqual(
            publication["observation_success_write_order_exact"][0], "RESERVED"
        )
        self.assertEqual(
            publication["observation_success_write_order_exact"][-1], "COMPLETED"
        )
        self.assertIn("EXCLUSIVE_CREATE", publication["publication_semantics"])
        forbidden = set(protocol["forbidden_actions"])
        self.assertIn(
            "MODIFY_SHADOW_V2_PROTOCOL_ENGINE_RESULTS_OR_CERTIFICATIONS",
            forbidden,
        )
        self.assertIn(
            "USE_ODDS_IMPLIED_PROBABILITY_EDGE_EXPECTED_VALUE_STAKE_PROFIT_ROI_OR_BET_RECOMMENDATION",
            forbidden,
        )
        self.assertIn(
            "CLAIM_BETTING_PROFITABILITY_WITHOUT_SEPARATE_TIMESTAMPED_PREMATCH_ODDS_PROTOCOL",
            forbidden,
        )


if __name__ == "__main__":
    unittest.main()
