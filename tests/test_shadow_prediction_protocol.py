"""Verrouille le protocole des prédictions fantômes prospectives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import unittest


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SHADOW_PROTOCOL_PATH = (
    PROJECT_DIRECTORY
    / "shadow_protocols"
    / "logistic_team_form_v1_platt_shadow_v1.json"
)
ARTIFACT_MANIFEST_PATH = (
    PROJECT_DIRECTORY
    / "model_artifacts"
    / "logistic_team_form_v1_platt.json"
)
MODEL_PROTOCOL_PATH = (
    PROJECT_DIRECTORY
    / "model_protocols"
    / "logistic_team_form_v1.json"
)
EVALUATION_PROTOCOL_PATH = (
    PROJECT_DIRECTORY
    / "evaluation_protocols"
    / "logistic_team_form_v1_platt_2025.json"
)

EXPECTED_SHADOW_PROTOCOL_SHA256 = (
    "dad8bd60b45400ede52952f6ac7c58de"
    "71fe75843cb0c010980fdad00ac1f8a7"
)
EXPECTED_ARTIFACT_SHA256 = (
    "e0d4d2421ba076072c7ef8b3bc97dd9"
    "a341e26c62828a0ad9ba43f30da15ff55"
)
EXPECTED_ARTIFACT_MANIFEST_SHA256 = (
    "a7375d5376baa043b3365cff713cb717"
    "010ad812efee472b594fa454306c96ae"
)
EXPECTED_MODEL_PROTOCOL_SHA256 = (
    "c4cb1af750619967514d37ae3a5a47a6"
    "a04255aeaccb20c5e94533dc4d138451"
)
EXPECTED_EVALUATION_PROTOCOL_SHA256 = (
    "f6dbace5d25d92c5d0ec3c9ae16962ab"
    "03e022439177342bd28d607afba7b1a9"
)
EXPECTED_DATASET_SHA256 = (
    "2a24c1a22a919acfc4ea545f8025f59c"
    "86d15873aaf09cdbdcef66236cd0a73e"
)
EXPECTED_EVALUATION_PREDICTIONS_SHA256 = (
    "4396b9aa645e9842cd6f0cbed663e2e0"
    "d1ada2487195a09eb081a4a198033b25"
)
EXPECTED_EVALUATION_REPORT_SHA256 = (
    "3f3d71baa122a4ac5f1690356e1c1656"
    "e812f500436dd6838d1421ba36522e70"
)
EXPECTED_FEATURES = [
    "away_games_before",
    "away_win_pct_before",
    "away_runs_scored_per_game_before",
    "away_runs_allowed_per_game_before",
    "home_games_before",
    "home_win_pct_before",
    "home_runs_scored_per_game_before",
    "home_runs_allowed_per_game_before",
]


def _reject_duplicate_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Refuse les clés JSON dupliquées au lieu d'accepter la dernière."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError(f"Clé JSON dupliquée : {key}.")
        result[key] = value
    return result


def _read_json(path: Path) -> tuple[bytes, dict[str, object]]:
    """Lit strictement un objet JSON et conserve ses octets."""
    content = path.read_bytes()
    payload = json.loads(
        content,
        object_pairs_hook=_reject_duplicate_object,
    )
    if not isinstance(payload, dict):
        raise AssertionError(f"Objet JSON attendu dans {path}.")
    return content, payload


def _safe_relative_path(value: str) -> PurePosixPath:
    """Valide un chemin suivi par le dépôt."""
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise AssertionError(f"Chemin relatif sûr attendu : {value}.")
    return path


class ShadowPredictionProtocolTests(unittest.TestCase):
    """Empêche le backfill, l'adaptation et la réécriture des probabilités."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.shadow_bytes, cls.shadow = _read_json(SHADOW_PROTOCOL_PATH)
        cls.manifest_bytes, cls.manifest = _read_json(
            ARTIFACT_MANIFEST_PATH
        )
        cls.model_protocol_bytes, cls.model_protocol = _read_json(
            MODEL_PROTOCOL_PATH
        )
        cls.evaluation_bytes, cls.evaluation_protocol = _read_json(
            EVALUATION_PROTOCOL_PATH
        )

    def test_protocol_is_exact_and_registered_before_first_prediction(
        self,
    ) -> None:
        """Le protocole doit être daté et figé avant toute probabilité."""
        self.assertEqual(
            hashlib.sha256(self.shadow_bytes).hexdigest(),
            EXPECTED_SHADOW_PROTOCOL_SHA256,
        )
        self.assertEqual(
            set(self.shadow),
            {
                "shadow_prediction_protocol_version",
                "registered_on",
                "registered_after_results_commit",
                "status",
                "first_shadow_prediction_created_at_registration",
                "purpose",
                "validated_lineage",
                "execution_freeze",
                "activation_contract",
                "operating_mode",
                "daily_batch",
                "source_snapshot",
                "candidate_games",
                "feature_contract",
                "model_use",
                "canonicalization",
                "idempotency",
                "schedule_change_policy",
                "output_contract",
                "receipt_contract",
                "prospective_certification",
                "scoring_and_betting",
                "preflight_invariants",
                "failure_policy",
                "registered_limitations",
            },
        )
        self.assertEqual(self.shadow["shadow_prediction_protocol_version"], 1)
        self.assertEqual(self.shadow["registered_on"], "2026-08-31")
        self.assertEqual(
            self.shadow["status"],
            "REGISTERED_BEFORE_FIRST_SHADOW_PREDICTION",
        )
        self.assertIs(
            self.shadow["first_shadow_prediction_created_at_registration"],
            False,
        )
        self.assertEqual(
            self.shadow["purpose"],
            "PROSPECTIVE_DAILY_SHADOW_PROBABILITIES_WITHOUT_ODDS_OR_BETS",
        )

    def test_frozen_model_identity_matches_registered_sources(self) -> None:
        """Le seul artefact autorisé doit être celui validé sur 2025."""
        lineage = self.shadow["validated_lineage"]
        artifact = lineage["model_artifact"]
        dataset = lineage["historical_dataset_lineage_only"]

        self.assertEqual(
            hashlib.sha256(self.manifest_bytes).hexdigest(),
            EXPECTED_ARTIFACT_MANIFEST_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(self.model_protocol_bytes).hexdigest(),
            EXPECTED_MODEL_PROTOCOL_SHA256,
        )
        self.assertEqual(artifact["sha256"], EXPECTED_ARTIFACT_SHA256)
        self.assertEqual(artifact["size_bytes"], 1589)
        self.assertEqual(
            artifact["code_commit"],
            "1d13f8f361de0865f722a8560f1efda8e95badbc",
        )
        self.assertEqual(
            lineage["artifact_manifest"]["sha256"],
            EXPECTED_ARTIFACT_MANIFEST_SHA256,
        )
        self.assertEqual(
            lineage["model_protocol"]["sha256"],
            EXPECTED_MODEL_PROTOCOL_SHA256,
        )
        self.assertEqual(dataset["sha256"], EXPECTED_DATASET_SHA256)
        self.assertIs(dataset["must_not_be_loaded_for_shadow_prediction"], True)
        self.assertEqual(
            self.manifest["model"]["feature_columns"],
            EXPECTED_FEATURES,
        )

    def test_2025_validation_provenance_is_exact(self) -> None:
        """Le passage au prospectif doit partir du verdict publié."""
        evaluation = self.shadow["validated_lineage"]["sealed_evaluation"]

        self.assertEqual(
            hashlib.sha256(self.evaluation_bytes).hexdigest(),
            EXPECTED_EVALUATION_PROTOCOL_SHA256,
        )
        self.assertEqual(
            evaluation["protocol_sha256"],
            EXPECTED_EVALUATION_PROTOCOL_SHA256,
        )
        self.assertEqual(
            evaluation["evaluation_code_commit"],
            "47d77269ff43cef8cef0fb5d0a3302cdf1ac1d7f",
        )
        self.assertEqual(
            evaluation["results_commit"],
            "7d64a6e5c03e36900cbf36f7c10182232c126baa",
        )
        self.assertEqual(
            self.shadow["registered_after_results_commit"],
            evaluation["results_commit"],
        )
        self.assertEqual(
            evaluation["predictions_sha256"],
            EXPECTED_EVALUATION_PREDICTIONS_SHA256,
        )
        self.assertEqual(
            evaluation["report_sha256"],
            EXPECTED_EVALUATION_REPORT_SHA256,
        )
        self.assertEqual(evaluation["report_status"], "COMPLETED_AND_VERIFIED")
        self.assertEqual(evaluation["evaluated_games"], 2276)
        self.assertEqual(evaluation["verdict"], "SIGNAL_VALIDE")
        self.assertIs(evaluation["artifact_state_unchanged"], True)
        self.assertEqual(evaluation["model_fit_calls"], 0)
        self.assertEqual(evaluation["season_2026_predictions"], 0)
        self.assertEqual(evaluation["season_2026_metrics"], 0)
        for rounded_name in (
            "model_log_loss",
            "reference_log_loss",
            "relative_log_loss_improvement",
            "bootstrap_95_percent_interval",
        ):
            self.assertNotIn(rounded_name, evaluation)

    def test_execution_manifest_has_no_git_commit_cycle(self) -> None:
        """Les commits quotidiens peuvent avancer sans changer le moteur."""
        freeze = self.shadow["execution_freeze"]

        self.assertEqual(
            set(freeze["manifest_must_bind"]),
            {
                "SHADOW_SERVICE_CODE_COMMIT",
                "SHADOW_SERVICE_MODULE_SHA256",
                "TRANSITIVE_RUNTIME_FILE_SHA256_MAP",
                "SHADOW_PROTOCOL_SHA256",
                "MODEL_ARTIFACT_SHA256",
                "REQUIREMENTS_SHA256",
                "TEST_SUITE_RESULT",
                "RUNTIME_VERSIONS",
                "MINIMUM_TARGET_OFFICIAL_DATE",
            },
        )
        self.assertIs(
            freeze["runtime_head_must_descend_from_service_code_commit"],
            True,
        )
        self.assertIs(
            freeze["runtime_files_must_match_execution_manifest_hashes"],
            True,
        )
        self.assertNotIn("runtime_head_must_equal_manifest_code_commit", freeze)
        self.assertIs(freeze["new_slot_requires_clean_git_worktree"], True)
        self.assertIs(
            freeze["existing_slot_is_inspected_before_clean_worktree_check"],
            True,
        )
        self.assertEqual(freeze["explicit_cli_flag_required"], "--execute-shadow")
        self.assertEqual(
            freeze["default_mode"],
            "PREVIEW_WITHOUT_MODEL_OR_PREDICTIONS",
        )
        for field in (
            "preview_must_not_deserialize_model",
            "preview_must_not_call_predict_proba",
            "preview_must_not_reserve_output_slot",
            "preview_must_not_create_output_files",
        ):
            self.assertIs(freeze[field], True)

    def test_activation_date_is_explicit_and_cannot_be_overridden(self) -> None:
        """Le premier jour admissible doit venir du manifeste déjà publié."""
        activation = self.shadow["activation_contract"]
        operating = self.shadow["operating_mode"]

        self.assertEqual(
            activation["minimum_target_official_date_source"],
            "execution_manifest.minimum_target_official_date",
        )
        self.assertIs(
            activation["minimum_target_official_date_must_be_explicit"],
            True,
        )
        self.assertIs(
            activation["execution_manifest_introduction_commit_must_be_on_remote_main"],
            True,
        )
        self.assertIs(
            activation["execution_manifest_must_equal_blob_at_introduction_commit"],
            True,
        )
        self.assertIs(
            activation["execution_manifest_updates_forbidden_for_v1"],
            True,
        )
        self.assertIs(
            activation["all_execution_values_must_be_read_from_introduction_blob"],
            True,
        )
        self.assertEqual(
            activation["minimum_target_official_date_rule"],
            "minimum_target_official_date >= max(shadow_protocol.registered_on, utc_date(execution_manifest_remote_visibility_verified_at_utc))",
        )
        self.assertEqual(
            activation["target_official_date_rule"],
            "target_official_date >= minimum_target_official_date",
        )
        self.assertIs(activation["no_runtime_override_allowed"], True)
        self.assertIs(operating["strictly_prospective"], True)
        self.assertIs(operating["historical_backfill_allowed"], False)
        self.assertIs(
            operating["past_2026_games_may_be_labeled_as_shadow_predictions"],
            False,
        )
        self.assertIs(
            operating["future_2026_games_may_be_predicted_prospectively"],
            True,
        )

    def test_slot_is_consumed_before_official_inputs_are_read(self) -> None:
        """Impossible de choisir après coup le snapshot officiel préféré."""
        idempotency = self.shadow["idempotency"]
        snapshot = self.shadow["source_snapshot"]

        self.assertIs(
            idempotency["slot_inspection_is_first_operation_before_any_official_input_read"],
            True,
        )
        self.assertIs(
            idempotency["new_slot_is_atomically_reserved_before_any_schedule_api_or_database_read"],
            True,
        )
        self.assertIs(
            snapshot["official_slot_must_be_reserved_before_schedule_api_or_database_read"],
            True,
        )
        self.assertEqual(idempotency["write_mode"], "EXCLUSIVE_NO_OVERWRITE")
        self.assertEqual(
            idempotency["exact_completed_identity_fields"],
            [
                "slot_key",
                "target_official_date",
                "shadow_protocol_sha256",
                "execution_manifest_sha256",
                "model_artifact_sha256",
            ],
        )
        self.assertEqual(
            idempotency["completed_exact_duplicate"],
            "RETURN_EXISTING_RECEIPT_WITHOUT_MODEL_CALL",
        )
        self.assertEqual(
            idempotency["incomplete_or_failed_slot"],
            "PERMANENTLY_FAILED_NO_REUSE_NO_DELETE",
        )
        self.assertIs(idempotency["supplemental_official_runs_allowed"], False)

    def test_entire_batch_finishes_two_hours_before_first_game(self) -> None:
        """Aucun résultat de la journée cible ne peut déjà exister."""
        batch = self.shadow["daily_batch"]
        lead = batch["local_completion_lead_rule"]
        operating = self.shadow["operating_mode"]

        self.assertIs(
            batch["all_target_schedule_games_must_be_observed_before_any_target_game_starts"],
            True,
        )
        self.assertEqual(
            batch["required_time_order"],
            "schedule_observed_at_utc <= information_cutoff_utc <= issued_at_utc <= completed_at_utc",
        )
        self.assertEqual(
            lead["applies_to"],
            "EARLIEST_SCHEDULED_START_AMONG_ALL_PREDICTED_GAMES",
        )
        self.assertEqual(
            lead["formula"],
            "completed_at_utc <= earliest_predicted_scheduled_start_utc - 120 minutes",
        )
        self.assertIs(lead["operator_is_inclusive"], True)
        self.assertEqual(operating["minimum_local_completion_lead_minutes"], 120)
        self.assertIs(
            batch["clock_evidence"]["mlb_http_response_received_at_utc_required"],
            True,
        )
        self.assertEqual(
            batch["clock_evidence"]["clock_skew_formula"],
            "system_received_at_utc - parsed_mlb_http_date_utc",
        )

    def test_pregame_zero_scores_are_tolerated_but_never_tracked(self) -> None:
        """MLB peut annoncer 0-0 avant match sans que ce soit une cible."""
        candidates = self.shadow["candidate_games"]

        self.assertEqual(
            candidates["accepted_pregame_score_states"],
            ["BOTH_NULL", "BOTH_ZERO"],
        )
        self.assertEqual(
            candidates["nonzero_or_one_sided_pregame_score_policy"],
            "FAIL_ENTIRE_RESERVED_SLOT",
        )
        self.assertIs(
            candidates["target_score_values_must_be_redacted_from_tracked_target_rows"],
            True,
        )
        self.assertEqual(
            candidates["started_live_or_final_target_game_policy"],
            "FAIL_ENTIRE_RESERVED_SLOT",
        )
        self.assertEqual(
            set(candidates["allowed_exclusion_reasons"]),
            {
                "POSTPONED",
                "CANCELLED",
                "START_TIME_MISSING",
                "INSUFFICIENT_AWAY_HISTORY",
                "INSUFFICIENT_HOME_HISTORY",
            },
        )
        self.assertEqual(
            candidates["wrong_sport_type_date_duplicate_or_unknown_state_policy"],
            "FAIL_ENTIRE_RESERVED_SLOT",
        )

    def test_features_are_exactly_the_validated_j_minus_one_features(self) -> None:
        """Aucun résultat du jour, lanceur ou cote ne peut entrer au modèle."""
        features = self.shadow["feature_contract"]

        self.assertEqual(features["dataset_semantics"], "mlb_team_form_v1")
        self.assertEqual(features["calendar_policy"], "J_MINUS_1_BY_OFFICIAL_DATE")
        self.assertEqual(features["minimum_history_games_per_team"], 10)
        self.assertIs(features["reset_team_state_each_season"], True)
        self.assertIs(features["same_day_games_never_feed_each_other"], True)
        self.assertEqual(
            features["allowed_source_rule"],
            "source_game.season == target_game.season and source_game.official_date < target_game.official_date",
        )
        self.assertEqual(
            features["accepted_final_rule"],
            "status_code in accepted_final_status_codes OR upper(trim(status_detail)) in accepted_final_status_details_casefolded",
        )
        self.assertEqual(
            features["feature_as_of_date_rule"],
            "target_official_date - 1 calendar day",
        )
        self.assertEqual(features["feature_columns_in_exact_order"], EXPECTED_FEATURES)
        self.assertIs(
            features["target_scores_and_outcomes_forbidden_as_features"],
            True,
        )
        self.assertEqual(
            features["feature_quantization"],
            {
                "games_before": "INTEGER",
                "rates_and_averages": "PYTHON_FIXED_POINT_6_DECIMALS_THEN_PARSE_AS_FLOAT",
                "decimal_separator": ".",
            },
        )
        for forbidden in (
            "PROBABLE_PITCHER",
            "TARGET_SCORE",
            "TARGET_OUTCOME",
            "ODDS",
        ):
            self.assertIn(forbidden, features["forbidden_features"])

    def test_model_is_prediction_only_and_cannot_be_adapted(self) -> None:
        """Fit, recalibrage, sélection et modèle de secours sont interdits."""
        model_use = self.shadow["model_use"]

        self.assertIs(
            model_use["verify_artifact_sha256_before_deserialization"],
            True,
        )
        self.assertEqual(model_use["expected_classes"], [0, 1])
        self.assertEqual(
            model_use["home_probability_source"],
            "calibrated_classifier.predict_proba(X) for class 1",
        )
        for field in (
            "fit_allowed",
            "partial_fit_allowed",
            "recalibration_allowed",
            "threshold_tuning_allowed",
            "feature_selection_allowed",
            "fallback_model_allowed",
            "automatic_latest_model_selection_allowed",
        ):
            self.assertIs(model_use[field], False)
        self.assertEqual(
            model_use["maximum_predict_proba_calls_per_nonempty_batch"],
            1,
        )
        self.assertIs(
            model_use["artifact_serialized_state_before_and_after_must_match"],
            True,
        )

    def test_hashes_and_identifiers_have_unambiguous_encoding(self) -> None:
        """Deux exécutions conformes doivent fabriquer les mêmes octets."""
        canonical = self.shadow["canonicalization"]
        snapshot = self.shadow["source_snapshot"]

        self.assertEqual(canonical["text_encoding"], "UTF-8")
        self.assertEqual(canonical["line_ending"], "LF")
        self.assertEqual(canonical["utc_timestamp_format"], "RFC3339_SECONDS_Z")
        self.assertEqual(
            canonical["identifier_preimage_format"],
            "CANONICAL_JSON_ARRAY_WITH_DOMAIN_TAG",
        )
        self.assertEqual(canonical["slot_key_components"][0], "shadow_slot_v1")
        self.assertEqual(
            canonical["occurrence_key_components"][0],
            "shadow_occurrence_v1",
        )
        self.assertEqual(
            canonical["prediction_id_components"][0],
            "shadow_prediction_v1",
        )
        self.assertEqual(canonical["feature_rate_format_before_model_parse"], ".6f")
        self.assertEqual(
            snapshot["canonical_gzip"],
            {
                "implementation": "gzip.GzipFile",
                "filename": "",
                "mtime": 0,
                "compresslevel": 9,
            },
        )

    def test_all_tracked_data_schemas_are_exact_and_safe(self) -> None:
        """Les scores cibles et les cotes n'ont aucun champ où se cacher."""
        outputs = self.shadow["output_contract"]

        self.assertEqual(
            set(outputs["source_snapshot_keys_exact_set"]),
            {
                "schema_version",
                "batch_id",
                "target_official_date",
                "created_at_utc",
                "information_cutoff_utc",
                "schedule_ingestion",
                "sqlite_snapshot",
                "teams",
                "target_schedule",
                "source_final_games",
            },
        )
        self.assertEqual(
            set(outputs["target_schedule_row_keys_exact_set"]),
            {
                "game_id",
                "season",
                "official_date",
                "game_datetime_utc",
                "game_type",
                "abstract_state",
                "detailed_state",
                "away_team_id",
                "home_team_id",
                "doubleheader",
                "game_number",
            },
        )
        self.assertEqual(
            set(outputs["schedule_ingestion_keys_exact_set"]),
            {
                "run_id",
                "completed_at_utc",
                "raw_archive_path",
                "raw_archive_sha256",
                "mlb_http_date_utc",
                "mlb_http_response_received_at_utc",
            },
        )
        self.assertEqual(
            set(outputs["sqlite_snapshot_keys_exact_set"]),
            {
                "source_database_path",
                "sha256",
                "size_bytes",
                "foreign_key_violation_count",
                "active_ingestion_count",
            },
        )
        self.assertEqual(
            set(outputs["team_row_keys_exact_set"]),
            {"team_id", "name", "abbreviation"},
        )
        self.assertEqual(
            set(outputs["source_final_game_row_keys_exact_set"]),
            {
                "game_id",
                "season",
                "official_date",
                "status_code",
                "status_detail",
                "away_team_id",
                "home_team_id",
                "away_score",
                "home_score",
            },
        )
        self.assertEqual(
            outputs["candidate_ledger_columns_in_exact_order"],
            [
                "batch_id",
                "game_id",
                "occurrence_key",
                "season",
                "official_date",
                "away_team_id",
                "home_team_id",
                "scheduled_start_utc",
                "abstract_state",
                "detailed_state",
                "eligibility_status",
                "exclusion_reason",
            ],
        )
        self.assertEqual(
            outputs["features_columns_in_exact_order"],
            [
                "prediction_id",
                "batch_id",
                "game_id",
                "occurrence_key",
                "season",
                "official_date",
                "away_team_id",
                "home_team_id",
                "scheduled_start_utc",
                "feature_as_of_date",
                "away_max_source_date",
                "home_max_source_date",
                *EXPECTED_FEATURES,
                "feature_row_sha256",
            ],
        )
        self.assertEqual(
            outputs["predictions_columns_in_exact_order"],
            [
                "prediction_id",
                "batch_id",
                "game_id",
                "occurrence_key",
                "season",
                "official_date_at_prediction",
                "away_team_id",
                "home_team_id",
                "scheduled_start_utc_at_prediction",
                "information_cutoff_utc",
                "issued_at_utc",
                "feature_as_of_date",
                "away_max_source_date",
                "home_max_source_date",
                *EXPECTED_FEATURES,
                "p_home_win",
                "p_away_win",
                "model_version",
                "artifact_sha256",
                "protocol_sha256",
                "code_commit",
            ],
        )
        forbidden = set(outputs["forbidden_exact_target_or_prediction_field_names"])
        for schema_name in (
            "candidate_ledger_columns_in_exact_order",
            "features_columns_in_exact_order",
            "predictions_columns_in_exact_order",
        ):
            self.assertTrue(forbidden.isdisjoint(outputs[schema_name]))
        self.assertTrue(
            forbidden.isdisjoint(outputs["target_schedule_row_keys_exact_set"])
        )
        self.assertEqual(
            set(outputs["reserved_marker_keys_exact_set"]),
            {
                "marker_schema_version",
                "batch_id",
                "slot_key",
                "target_official_date",
                "reserved_at_utc",
                "shadow_protocol_sha256",
                "execution_manifest_sha256",
                "runtime_code_commit",
            },
        )
        self.assertEqual(
            set(outputs["completed_marker_keys_exact_set"]),
            {
                "marker_schema_version",
                "batch_id",
                "receipt_path",
                "receipt_sha256",
                "completed_at_utc",
            },
        )
        self.assertEqual(
            set(outputs["failed_marker_keys_exact_set"]),
            {
                "marker_schema_version",
                "batch_id",
                "slot_key",
                "target_official_date",
                "failed_at_utc",
                "stage",
                "error_type",
                "error_message",
                "shadow_protocol_sha256",
                "execution_manifest_sha256",
                "runtime_code_commit",
            },
        )
        self.assertIs(
            outputs["source_final_game_scores_are_the_only_score_fields_allowed"],
            True,
        )
        self.assertEqual(outputs["failure_marker"], "FAILED.json")
        self.assertIs(outputs["completed_or_failed_marker_is_terminal"], True)

    def test_empty_eligible_batch_is_terminal_and_unambiguous(self) -> None:
        """Une journée sans match admissible consomme le slot sans modèle."""
        empty = self.shadow["output_contract"]["empty_eligible_batch"]

        self.assertEqual(empty["status"], "COMPLETED_NO_ELIGIBLE_GAMES")
        self.assertEqual(empty["predict_proba_calls"], 0)
        self.assertIsNone(empty["earliest_predicted_scheduled_start_utc"])
        self.assertEqual(
            empty["features_and_predictions_csv"],
            "HEADER_ONLY",
        )
        self.assertIs(empty["receipt_and_completed_marker_required"], True)
        self.assertIs(
            empty["prospective_remote_certification_required"],
            False,
        )
        self.assertIs(empty["slot_remains_permanently_consumed"], True)

    def test_output_paths_are_tracked_relative_and_never_overwritten(self) -> None:
        """Résultats et preuve distante doivent rester hors des dossiers ignorés."""
        outputs = self.shadow["output_contract"]
        idempotency = self.shadow["idempotency"]
        certification = self.shadow["prospective_certification"]

        result_root = _safe_relative_path(outputs["result_root"])
        result_template = _safe_relative_path(
            idempotency["result_directory_template"]
        )
        certificate = _safe_relative_path(
            certification["certification_path_template"]
        )
        evidence = _safe_relative_path(
            certification["raw_remote_evidence_path_template"]
        )
        self.assertEqual(result_root.parts[0], "shadow_results")
        self.assertEqual(result_template.parts[0], "shadow_results")
        self.assertEqual(certificate.parts[0], "shadow_certifications")
        self.assertEqual(evidence.parts[0], "shadow_certifications")
        self.assertIs(outputs["result_root_must_be_tracked_by_git"], True)
        self.assertIs(outputs["git_check_ignore_must_report_not_ignored"], True)
        self.assertIs(
            certification["certification_root_must_be_tracked_by_git"],
            True,
        )
        self.assertEqual(idempotency["write_mode"], "EXCLUSIVE_NO_OVERWRITE")
        self.assertEqual(
            certification["certification_write_mode"],
            "APPEND_ONLY_EXCLUSIVE_CREATE",
        )
        self.assertEqual(
            certification["raw_remote_evidence_write_mode"],
            "APPEND_ONLY_EXCLUSIVE_CREATE",
        )

    def test_receipt_schema_is_complete_and_auditable(self) -> None:
        """Chaque lot doit porter ses dates, sources, empreintes et compteurs."""
        receipt = self.shadow["receipt_contract"]

        self.assertEqual(
            set(receipt["top_level_keys_exact_set"]),
            {
                "receipt_schema_version",
                "batch",
                "activation",
                "times",
                "lineage",
                "source",
                "counts",
                "output_hashes",
                "model_invariants",
                "negative_attestations",
                "runtime_versions",
            },
        )
        self.assertEqual(
            set(receipt["batch_keys_exact_set"]),
            {
                "batch_id",
                "slot_key",
                "target_official_date",
                "status",
                "earliest_predicted_scheduled_start_utc",
            },
        )
        self.assertEqual(
            set(receipt["activation_keys_exact_set"]),
            {
                "execution_manifest_introduction_commit",
                "activation_verified_at_utc",
                "minimum_target_official_date",
            },
        )
        self.assertEqual(
            set(receipt["times_keys_exact_set"]),
            {
                "started_at_utc",
                "schedule_observed_at_utc",
                "information_cutoff_utc",
                "issued_at_utc",
                "completed_at_utc",
                "mlb_http_date_utc",
                "mlb_http_response_received_at_utc",
                "clock_skew_seconds",
            },
        )
        self.assertEqual(
            set(receipt["lineage_keys_exact_set"]),
            {
                "runtime_code_commit",
                "shadow_service_module_sha256",
                "shadow_protocol_sha256",
                "execution_manifest_sha256",
                "model_artifact_sha256",
                "artifact_manifest_sha256",
                "model_protocol_sha256",
                "evaluation_protocol_sha256",
                "evaluation_report_sha256",
                "evaluation_results_commit",
            },
        )
        self.assertEqual(
            set(receipt["source_keys_exact_set"]),
            {
                "sqlite_snapshot_sha256",
                "sqlite_snapshot_size_bytes",
                "source_snapshot_path",
                "source_snapshot_sha256",
                "schedule_ingestion_run_id",
                "schedule_ingestion_completed_at_utc",
                "schedule_raw_archive_path",
                "schedule_raw_archive_sha256",
            },
        )
        self.assertEqual(
            set(receipt["output_hashes_keys_exact_set"]),
            {
                "candidate_ledger_sha256",
                "features_sha256",
                "predictions_sha256",
            },
        )
        self.assertEqual(
            set(receipt["counts_keys_exact_set"]),
            {
                "schedule_games",
                "eligible_games",
                "predicted_games",
                "excluded_games_by_reason",
            },
        )
        self.assertEqual(
            set(receipt["excluded_games_by_reason_keys_exact_set"]),
            set(self.shadow["candidate_games"]["allowed_exclusion_reasons"]),
        )
        self.assertEqual(
            set(receipt["model_invariants_keys_exact_set"]),
            {
                "fit_calls",
                "partial_fit_calls",
                "recalibration_calls",
                "threshold_tuning_calls",
                "feature_selection_calls",
                "predict_proba_calls",
                "artifact_state_sha256_before",
                "artifact_state_sha256_after",
                "artifact_state_unchanged",
            },
        )
        self.assertEqual(
            receipt["model_invariant_required_values"],
            {
                "fit_calls": 0,
                "partial_fit_calls": 0,
                "recalibration_calls": 0,
                "threshold_tuning_calls": 0,
                "feature_selection_calls": 0,
                "artifact_state_unchanged": True,
            },
        )
        self.assertEqual(
            receipt["predict_proba_calls_rule"],
            "1 if predicted_games > 0 else 0",
        )
        self.assertEqual(
            receipt["artifact_state_hash_rule"],
            "artifact_state_sha256_before == artifact_state_sha256_after",
        )
        self.assertEqual(
            set(receipt["negative_attestations_keys_exact_set"]),
            {
                "ALL_TARGET_GAMES_UNSTARTED_AT_INFORMATION_CUTOFF",
                "TARGET_OUTCOMES_NOT_AVAILABLE_AT_INFORMATION_CUTOFF",
                "TARGET_SCORE_VALUES_REDACTED_FROM_TRACKED_TARGET_ROWS",
                "TARGET_SCORES_NOT_USED_AS_FEATURES",
                "ODDS_NOT_READ",
                "BETTING_RECOMMENDATIONS_NOT_COMPUTED",
                "SEASON_2026_METRICS_NOT_COMPUTED",
            },
        )
        self.assertIs(receipt["negative_attestations_required_value"], True)
        self.assertEqual(
            set(receipt["runtime_versions_keys_exact_set"]),
            {"python", "numpy", "pandas", "scipy", "scikit_learn", "joblib"},
        )
        self.assertIs(receipt["game_ids_must_be_unique"], True)
        self.assertEqual(
            receipt["probability_pairs_must_sum_to_one_with_absolute_tolerance"],
            1e-12,
        )

    def test_remote_certification_is_separate_and_precedes_games(self) -> None:
        """La publication distante ne modifie jamais le lot terminé."""
        certification = self.shadow["prospective_certification"]
        operating = self.shadow["operating_mode"]

        self.assertEqual(
            certification["claim_level"],
            "REMOTE_SERVER_ATTESTED_NOT_CRYPTOGRAPHICALLY_TIMESTAMPED",
        )
        self.assertIs(
            certification["certification_is_outside_completed_batch_directory"],
            True,
        )
        self.assertEqual(
            certification["remote_time_must_precede_each_predicted_game_by_at_least_minutes"],
            operating["minimum_remote_publication_lead_minutes"],
        )
        self.assertEqual(
            set(certification["certification_keys_exact_set"]),
            {
                "certification_schema_version",
                "target_official_date",
                "batch_id",
                "results_commit",
                "results_remote_ref",
                "results_tree_file_hashes",
                "remote_query_url",
                "remote_http_date_utc",
                "remote_response_received_at_utc",
                "remote_response_body_sha256",
                "raw_remote_evidence_path",
                "raw_remote_evidence_sha256",
                "earliest_predicted_start_utc",
                "remote_publication_lead_minutes",
                "status",
                "claim_level",
            },
        )
        self.assertEqual(
            set(certification["raw_remote_evidence_keys_exact_set"]),
            {
                "evidence_schema_version",
                "request_url",
                "request_method",
                "response_status_code",
                "selected_response_headers",
                "response_received_at_utc",
                "response_body_base64",
                "response_body_sha256",
            },
        )
        self.assertEqual(
            set(certification["selected_response_header_keys_exact_set"]),
            {"date", "content-type", "etag", "x-github-request-id"},
        )
        self.assertEqual(
            certification["remote_evidence_gzip"],
            {
                "implementation": "gzip.GzipFile",
                "filename": "",
                "mtime": 0,
                "compresslevel": 9,
            },
        )
        self.assertEqual(
            set(certification["results_tree_file_hashes_entry_keys_exact_set"]),
            {"path", "sha256", "size_bytes"},
        )
        self.assertEqual(
            certification["results_tree_file_hashes_expected_relative_paths"],
            self.shadow["output_contract"]["success_files_in_write_order"],
        )
        self.assertIs(
            certification["certification_must_reference_preexisting_results_commit"],
            True,
        )
        self.assertIs(
            certification["remote_response_body_must_name_exact_results_commit"],
            True,
        )
        self.assertIs(certification["retroactive_certification_allowed"], False)
        self.assertIs(
            certification["remote_evidence_never_changes_original_predictions"],
            True,
        )

    def test_shadow_mode_forbids_results_odds_bets_and_metrics(self) -> None:
        """Cette phase ne doit produire qu'une probabilité par équipe."""
        operating = self.shadow["operating_mode"]
        betting = self.shadow["scoring_and_betting"]

        self.assertEqual(operating["name"], "PROBABILITIES_ONLY_SHADOW_MODE")
        for field in (
            "outcome_collection_enabled",
            "season_2026_metrics_allowed",
            "odds_input_allowed",
            "implied_probability_allowed",
            "edge_allowed",
            "expected_value_allowed",
            "bet_recommendation_allowed",
            "stake_allowed",
            "roi_allowed",
            "profitability_claim_allowed",
        ):
            self.assertIs(betting[field], False)
        self.assertIs(
            betting["later_scoring_requires_separate_frozen_protocol"],
            True,
        )
        self.assertIs(
            betting["later_odds_analysis_requires_timestamped_prematch_odds"],
            True,
        )

    def test_preflight_and_failures_are_closed(self) -> None:
        """Toute incohérence consomme ou arrête clairement le lot."""
        invariants = set(self.shadow["preflight_invariants"])
        failure = self.shadow["failure_policy"]

        for invariant in (
            "INSPECT_EXISTING_SLOT_BEFORE_ANY_OFFICIAL_INPUT_READ",
            "REQUIRE_RUNTIME_HEAD_DESCENDS_FROM_SERVICE_COMMIT",
            "REQUIRE_TRANSITIVE_RUNTIME_FILES_MATCH_EXECUTION_MANIFEST_HASHES",
            "REQUIRE_EXECUTION_MANIFEST_EQUALS_IMMUTABLE_INTRODUCTION_BLOB",
            "REQUIRE_TARGET_DATE_AT_OR_AFTER_EXPLICIT_MINIMUM_TARGET_DATE",
            "REQUIRE_MINIMUM_TARGET_DATE_NOT_BEFORE_REGISTRATION_OR_MANIFEST_REMOTE_VISIBILITY_DATE",
            "REQUIRE_ALL_TARGET_GAMES_UNSTARTED_AT_INFORMATION_CUTOFF",
            "REQUIRE_COMPLETION_120_MINUTES_BEFORE_EARLIEST_PREDICTED_START",
            "REQUIRE_PREGAME_SCORES_BOTH_NULL_OR_BOTH_ZERO",
            "REQUIRE_ALL_TRACKED_OUTPUT_SCHEMAS_EXACT",
            "REQUIRE_NO_HISTORICAL_2026_SHADOW_BACKFILL",
        ):
            self.assertIn(invariant, invariants)
        self.assertEqual(
            failure["failure_before_reservation"],
            "ABORT_WITHOUT_OFFICIAL_INPUT_READ",
        )
        self.assertEqual(
            failure["failure_after_reservation"],
            "WRITE_FAILED_JSON_AND_NEVER_REUSE_SLOT",
        )
        self.assertEqual(
            failure["structural_or_temporal_failure"],
            "ABORT_WITHOUT_PREDICTIONS",
        )
        self.assertIs(failure["partial_prediction_publication_allowed"], False)
        self.assertIs(
            failure["only_registered_network_retries_inside_reserved_slot_allowed"],
            True,
        )
        self.assertIs(failure["automatic_output_deletion_allowed"], False)
        self.assertIn(
            "NO_BETTING_PROFITABILITY_EVIDENCE",
            self.shadow["registered_limitations"],
        )


if __name__ == "__main__":
    unittest.main()
