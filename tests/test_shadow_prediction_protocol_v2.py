"""Verrouille la version 2 du protocole des prédictions fantômes MLB."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SHADOW_V1_PATH = (
    PROJECT_DIRECTORY
    / "shadow_protocols"
    / "logistic_team_form_v1_platt_shadow_v1.json"
)
SHADOW_V2_PATH = (
    PROJECT_DIRECTORY
    / "shadow_protocols"
    / "logistic_team_form_v1_platt_shadow_v2.json"
)

EXPECTED_SHADOW_V1_SHA256 = (
    "dad8bd60b45400ede52952f6ac7c58de"
    "71fe75843cb0c010980fdad00ac1f8a7"
)
EXPECTED_SHADOW_V2_SHA256 = (
    "4dcae9e85bb9ed5b3f4a9f961491d872"
    "56965c42e97d11e70f31f02f52cc49c9"
)

EXPECTED_TOP_LEVEL_KEYS = {
    "shadow_prediction_protocol_version",
    "registered_on",
    "registered_after_results_commit",
    "status",
    "first_shadow_prediction_created_at_registration",
    "purpose",
    "supersedes_unexecuted_protocol",
    "remote_repository",
    "validated_lineage",
    "execution_freeze",
    "activation_contract",
    "activation_evidence",
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
}

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
    """Refuse les clés JSON dupliquées au lieu d'écraser une valeur."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError(f"Clé JSON dupliquée : {key}.")
        result[key] = value
    return result


def _read_json(path: Path) -> tuple[bytes, dict[str, object]]:
    """Lit strictement un objet JSON et conserve ses octets exacts."""
    content = path.read_bytes()
    payload = json.loads(
        content,
        object_pairs_hook=_reject_duplicate_object,
    )
    if not isinstance(payload, dict):
        raise AssertionError(f"Objet JSON attendu dans {path}.")
    return content, payload


def _sha256(path: Path) -> str:
    """Calcule l'empreinte d'un fichier suivi par Git."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    """Reproduit exactement la canonicalisation enregistrée."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _all_strings(value: object) -> list[str]:
    """Énumère récursivement les chaînes d'un objet JSON."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_all_strings(item))
        return result
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            result.append(str(key))
            result.extend(_all_strings(item))
        return result
    return []


def _iter_named_lists(
    value: object,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], list[object]]]:
    """Retourne toutes les listes JSON avec leur chemin de clés."""
    result: list[tuple[tuple[str, ...], list[object]]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = (*path, str(key))
            if isinstance(item, list):
                result.append((child_path, item))
            result.extend(_iter_named_lists(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_iter_named_lists(item, (*path, str(index))))
    return result


class ShadowPredictionProtocolV2Tests(unittest.TestCase):
    """Empêche toute ambiguïté avant la première exécution v2."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.v1_content, cls.v1 = _read_json(SHADOW_V1_PATH)
        cls.v2_content, cls.v2 = _read_json(SHADOW_V2_PATH)

    def test_v2_is_exact_and_registered_before_any_prediction(self) -> None:
        """Les octets et l'état initial du protocole v2 sont figés."""
        self.assertEqual(
            hashlib.sha256(self.v2_content).hexdigest(),
            EXPECTED_SHADOW_V2_SHA256,
        )
        self.assertEqual(set(self.v2), EXPECTED_TOP_LEVEL_KEYS)
        self.assertEqual(self.v2["shadow_prediction_protocol_version"], 2)
        self.assertEqual(self.v2["registered_on"], "2026-08-31")
        self.assertEqual(
            self.v2["status"],
            "REGISTERED_BEFORE_FIRST_SHADOW_V2_PREDICTION",
        )
        self.assertFalse(
            self.v2["first_shadow_prediction_created_at_registration"]
        )
        self.assertTrue(self.v2_content.endswith(b"\n"))
        self.assertFalse(self.v2_content.endswith(b"\n\n"))
        self.assertNotIn(b"\r", self.v2_content)
        self.assertFalse(self.v2_content.startswith(b"\xef\xbb\xbf"))

    def test_every_exact_set_and_order_list_has_unique_values(self) -> None:
        """Une liste déclarée exacte ne peut masquer aucun doublon."""
        exact_suffixes = (
            "keys_exact_set",
            "columns_in_exact_order",
            "order_exact",
            "status_domain_exact",
            "expected_relative_paths",
        )
        checked = 0
        for path, values in _iter_named_lists(self.v2):
            if not path[-1].endswith(exact_suffixes):
                continue
            self.assertTrue(
                all(isinstance(item, str) for item in values),
                ".".join(path),
            )
            self.assertEqual(
                len(values),
                len(set(values)),
                ".".join(path),
            )
            checked += 1
        self.assertGreaterEqual(checked, 20)

    def test_v1_is_immutable_superseded_and_never_executed(self) -> None:
        """La v1 reste une preuve historique mais ne peut jamais tourner."""
        self.assertEqual(
            hashlib.sha256(self.v1_content).hexdigest(),
            EXPECTED_SHADOW_V1_SHA256,
        )
        self.assertEqual(
            self.v2["supersedes_unexecuted_protocol"],
            {
                "path": (
                    "shadow_protocols/"
                    "logistic_team_form_v1_platt_shadow_v1.json"
                ),
                "sha256": EXPECTED_SHADOW_V1_SHA256,
                "registration_commit": (
                    "56298f43e88b6d276aba1d52cb8170c87df259b7"
                ),
                "execution_manifest_created": False,
                "shadow_prediction_created": False,
                "v1_execution_is_permanently_forbidden": True,
                "reason": (
                    "CLOSE_ACTIVATION_WARNING_IDENTIFIER_AND_"
                    "ORDERING_AMBIGUITIES_BEFORE_FIRST_RUN"
                ),
            },
        )
        v1_references = [
            value
            for value in _all_strings(self.v2)
            if "shadow_v1" in value
        ]
        self.assertEqual(
            v1_references,
            [
                "shadow_protocols/"
                "logistic_team_form_v1_platt_shadow_v1.json"
            ],
        )
        forbidden_v1_execution_paths = [
            PROJECT_DIRECTORY
            / "shadow_protocols"
            / "logistic_team_form_v1_platt_shadow_v1_execution.json",
            PROJECT_DIRECTORY
            / "shadow_results"
            / "logistic_team_form_v1_platt_shadow_v1",
            PROJECT_DIRECTORY
            / "shadow_certifications"
            / "logistic_team_form_v1_platt_shadow_v1",
        ]
        for path in forbidden_v1_execution_paths:
            self.assertFalse(path.exists(), str(path))

    def test_validated_lineage_is_unchanged(self) -> None:
        """La v2 utilise exactement le modèle qui a validé 2025."""
        lineage = self.v2["validated_lineage"]
        expected_files = {
            "artifact_manifest": (
                "model_artifacts/logistic_team_form_v1_platt.json",
                "a7375d5376baa043b3365cff713cb717"
                "010ad812efee472b594fa454306c96ae",
            ),
            "model_protocol": (
                "model_protocols/logistic_team_form_v1.json",
                "c4cb1af750619967514d37ae3a5a47a6"
                "a04255aeaccb20c5e94533dc4d138451",
            ),
        }
        for key, (relative_path, expected_sha256) in expected_files.items():
            self.assertEqual(lineage[key]["path"], relative_path)
            self.assertEqual(lineage[key]["sha256"], expected_sha256)
            self.assertEqual(
                _sha256(PROJECT_DIRECTORY / relative_path),
                expected_sha256,
            )

        artifact = lineage["model_artifact"]
        self.assertEqual(
            artifact,
            {
                "path": "models/logistic_team_form_v1_platt.joblib",
                "sha256": (
                    "e0d4d2421ba076072c7ef8b3bc97dd9a"
                    "341e26c62828a0ad9ba43f30da15ff55"
                ),
                "size_bytes": 1589,
                "code_commit": (
                    "1d13f8f361de0865f722a8560f1efda8e95badbc"
                ),
                "model_version": "logistic_team_form_v1_platt",
            },
        )
        model_path = PROJECT_DIRECTORY / artifact["path"]
        if model_path.exists():
            self.assertEqual(model_path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_sha256(model_path), artifact["sha256"])
        dataset = lineage["historical_dataset_lineage_only"]
        self.assertEqual(dataset["row_count"], 13253)
        self.assertEqual(
            dataset["sha256"],
            "2a24c1a22a919acfc4ea545f8025f59c"
            "86d15873aaf09cdbdcef66236cd0a73e",
        )
        self.assertTrue(dataset["must_not_be_loaded_for_shadow_prediction"])
        dataset_path = PROJECT_DIRECTORY / dataset["path"]
        if dataset_path.exists():
            self.assertEqual(_sha256(dataset_path), dataset["sha256"])
            with dataset_path.open("rb") as handle:
                self.assertEqual(sum(1 for _ in handle) - 1, dataset["row_count"])

        evaluation = lineage["sealed_evaluation"]
        for path_key, sha_key in (
            ("protocol_path", "protocol_sha256"),
            ("predictions_path", "predictions_sha256"),
            ("report_path", "report_sha256"),
        ):
            self.assertEqual(
                _sha256(PROJECT_DIRECTORY / evaluation[path_key]),
                evaluation[sha_key],
            )
        self.assertEqual(evaluation["report_status"], "COMPLETED_AND_VERIFIED")
        self.assertEqual(evaluation["verdict"], "SIGNAL_VALIDE")
        self.assertTrue(evaluation["artifact_state_unchanged"])
        self.assertEqual(evaluation["model_fit_calls"], 0)
        self.assertEqual(evaluation["season_2026_predictions"], 0)
        self.assertEqual(evaluation["season_2026_metrics"], 0)

    def test_execution_manifest_schema_and_runtime_closure_are_exact(self) -> None:
        """Le futur service devra être figé sans cycle de commit ni dépendance cachée."""
        freeze = self.v2["execution_freeze"]
        self.assertEqual(freeze["execution_manifest_schema_version_exact"], 1)
        self.assertEqual(
            freeze["execution_manifest_status_exact"],
            "FROZEN_BEFORE_SHADOW_V2_ACTIVATION",
        )
        self.assertEqual(
            set(freeze["execution_manifest_keys_exact_set"]),
            {
                "execution_manifest_schema_version",
                "status",
                "shadow_service_code_commit",
                "shadow_service_module_path",
                "shadow_service_module_sha256",
                "transitive_runtime_file_sha256_map",
                "shadow_protocol_path",
                "shadow_protocol_sha256",
                "model_artifact_path",
                "model_artifact_sha256",
                "model_artifact_size_bytes",
                "requirements_path",
                "requirements_sha256",
                "test_suite_result",
                "runtime_versions",
                "minimum_target_official_date",
                "created_at_utc",
            },
        )
        self.assertEqual(
            freeze["shadow_service_module_path_exact"],
            "src/shadow_prediction.py",
        )
        minimum_paths = freeze[
            "transitive_runtime_file_sha256_map_minimum_paths"
        ]
        self.assertEqual(minimum_paths, sorted(minimum_paths))
        self.assertIn("src/shadow_prediction.py", minimum_paths)
        self.assertIn("src/mlb_api.py", minimum_paths)
        self.assertIn("src/calibrated_model.py", minimum_paths)
        self.assertIn("complete recursive closure", freeze["transitive_runtime_file_sha256_map_rule"])
        self.assertTrue(
            freeze[
                "shadow_service_code_commit_must_precede_execution_manifest_introduction_commit"
            ]
        )
        self.assertEqual(
            freeze["test_suite_result_required_values"]["command"],
            "python -m unittest discover -s tests -v",
        )
        self.assertEqual(
            freeze["runtime_versions_keys_exact_set"],
            ["python", "numpy", "pandas", "scipy", "scikit_learn", "joblib"],
        )

    def test_activation_is_persistent_remote_and_precedes_daily_slots(self) -> None:
        """L'activation distante laisse une preuve immuable et réutilisable."""
        freeze = self.v2["execution_freeze"]
        contract = self.v2["activation_contract"]
        evidence = self.v2["activation_evidence"]

        self.assertEqual(freeze["activation_cli_flag_required"], "--activate-shadow")
        self.assertEqual(
            freeze["manifest_path"],
            "shadow_protocols/"
            "logistic_team_form_v1_platt_shadow_v2_execution.json",
        )
        self.assertEqual(
            contract["activation_path"],
            "shadow_activations/"
            "logistic_team_form_v1_platt_shadow_v2/activation.json",
        )
        self.assertEqual(
            contract["activation_raw_remote_evidence_path"],
            "shadow_activations/logistic_team_form_v1_platt_shadow_v2/"
            "execution_manifest.remote.json.gz",
        )
        for key in (
            "activation_mode_must_not_reserve_daily_slot",
            "activation_mode_must_not_read_mlb_or_sqlite",
            "activation_mode_must_not_deserialize_or_predict",
            "activation_files_must_be_committed_and_pushed_before_first_official_batch",
            "activation_introduction_commit_remote_visibility_must_be_reverified_for_each_new_slot",
            "activation_must_equal_blob_at_introduction_commit",
            "activation_updates_forbidden_for_v2",
            "activation_remote_response_must_name_exact_activation_introduction_commit",
            "activation_remote_blob_must_equal_immutable_introduction_blob_for_each_new_slot",
            "activation_remote_reverification_required_for_each_new_slot",
            "no_runtime_override_allowed",
        ):
            self.assertTrue(contract[key], key)

        self.assertEqual(
            set(evidence["activation_keys_exact_set"]),
            {
                "activation_schema_version",
                "status",
                "shadow_protocol_path",
                "shadow_protocol_sha256",
                "execution_manifest_path",
                "execution_manifest_sha256",
                "execution_manifest_introduction_commit",
                "execution_manifest_remote_ref",
                "execution_manifest_remote_query_url",
                "execution_manifest_remote_effective_url",
                "execution_manifest_remote_response_status_code",
                "execution_manifest_remote_response_redirect_count",
                "execution_manifest_remote_http_date_utc",
                "execution_manifest_remote_response_received_at_utc",
                "execution_manifest_remote_response_body_sha256",
                "raw_remote_evidence_path",
                "raw_remote_evidence_sha256",
                "minimum_target_official_date",
                "created_at_utc",
                "claim_level",
            },
        )
        self.assertEqual(evidence["remote_request_method"], "GET")
        self.assertEqual(evidence["remote_response_status_code_required"], 200)
        self.assertEqual(evidence["remote_response_redirect_count_required"], 0)
        self.assertTrue(
            evidence[
                "remote_request_and_response_must_follow_remote_repository_contract"
            ]
        )
        self.assertEqual(
            set(evidence["raw_remote_evidence_keys_exact_set"]),
            {
                "evidence_schema_version",
                "request_url",
                "request_method",
                "application_request_headers",
                "effective_url",
                "response_status_code",
                "response_redirect_count",
                "selected_response_headers",
                "response_received_at_utc",
                "response_body_base64",
                "response_body_sha256",
            },
        )
        self.assertEqual(
            evidence["selected_response_header_keys_exact_set"],
            ["date", "content-type", "etag", "x-github-request-id"],
        )
        self.assertEqual(
            evidence["claim_level"],
            "REMOTE_SERVER_ATTESTED_NOT_CRYPTOGRAPHICALLY_TIMESTAMPED",
        )
        activation_bindings = evidence[
            "activation_json_to_raw_remote_evidence_invariants_exact"
        ]
        self.assertEqual(len(activation_bindings), 12)
        self.assertTrue(
            any("application_request_headers_exact" in rule for rule in activation_bindings)
        )
        self.assertTrue(
            any("parse_IMF_FIXDATE_GMT" in rule for rule in activation_bindings)
        )
        self.assertTrue(
            any("sha256(persisted raw gzip bytes)" in rule for rule in activation_bindings)
        )
        self.assertEqual(
            contract["per_slot_reverification_raw_evidence_filename"],
            "activation_reverification.remote.json.gz",
        )
        self.assertTrue(
            contract[
                "per_slot_reverification_evidence_bytes_must_be_built_before_reservation"
            ]
        )
        self.assertTrue(
            contract[
                "per_slot_reverification_evidence_must_be_written_exclusively_immediately_after_reserved_marker"
            ]
        )
        self.assertTrue(
            contract[
                "activation_remote_reverified_at_utc_must_not_follow_reserved_at_utc"
            ]
        )
        self.assertIn("utc_date(reserved_at_utc)", contract["target_official_date_rule"])
        self.assertNotIn("slot_reserved_at_utc", contract["target_official_date_rule"])

    def test_remote_repository_identity_and_compare_contract_are_exact(self) -> None:
        """Chaque preuve distante vise uniquement le dépôt GitHub enregistré."""
        remote = self.v2["remote_repository"]
        self.assertEqual(remote["provider"], "GITHUB")
        self.assertEqual(remote["owner"], "hestia19100-spec")
        self.assertEqual(remote["repository"], "fredo-mlb-predictor")
        self.assertEqual(remote["branch"], "main")
        self.assertEqual(remote["full_ref"], "refs/heads/main")
        self.assertEqual(
            remote["compare_api_url_template"],
            "https://api.github.com/repos/hestia19100-spec/"
            "fredo-mlb-predictor/compare/{expected_commit}...main",
        )
        self.assertEqual(remote["request_method"], "GET")
        self.assertEqual(
            remote["application_request_headers_exact"],
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "fredo-mlb-predictor-shadow-v2/1.0",
            },
        )
        self.assertFalse(remote["authorization_header_allowed"])
        self.assertFalse(remote["cookies_allowed"])
        self.assertTrue(remote["authentication_is_forbidden_for_v2"])
        self.assertTrue(remote["effective_url_must_equal_request_url"])
        self.assertEqual(remote["response_status_code_exact"], 200)
        self.assertEqual(remote["response_redirect_count_exact"], 0)
        self.assertEqual(
            remote["response_body_json_rules"],
            {
                "base_commit.sha": "expected_commit",
                "merge_base_commit.sha": "expected_commit",
                "status_allowed_values": ["ahead", "identical"],
            },
        )
        self.assertTrue(remote["tls_certificate_verification_required"])
        self.assertTrue(
            self.v2["activation_contract"]
            ["per_slot_reverification_must_follow_remote_repository_contract"]
        )
        self.assertIn(
            "remote_repository.compare_api_url_template",
            self.v2["prospective_certification"]["remote_visibility_check"],
        )

    def test_warning_policy_allows_only_the_registered_compatibility_warning(self) -> None:
        """Seul l'avertissement joblib/NumPy déjà vérifié est toléré."""
        policy = self.v2["model_use"]["warning_policy"]
        self.assertEqual(
            policy,
            {
                "policy_id": "VERIFIED_JOBLIB_NUMPY_COMPATIBILITY_V1",
                "default": "ABORT_ENTIRE_RUN",
                "approved_load_or_state_serialization_warning": {
                    "category": "DeprecationWarning",
                    "message_regex": (
                        "Setting the shape on a NumPy array has been "
                        "deprecated.*"
                    ),
                    "module_regex": "joblib\\.numpy_pickle",
                    "allowed_stages": [
                        "ARTIFACT_DESERIALIZATION",
                        "ARTIFACT_STATE_SERIALIZATION",
                    ],
                    "requires_exact_manifest_runtime_versions": True,
                },
                "matching_implementation": (
                    "inside warnings.catch_warnings(record=True), call "
                    "warnings.simplefilter('error') first, then "
                    "warnings.filterwarnings('always', "
                    "message=message_regex, category=DeprecationWarning, "
                    "module=module_regex, append=False); execute exactly one "
                    "allowed stage inside that context"
                ),
                "message_regex_semantics": (
                    "Python warnings.filterwarnings semantics: "
                    "case-insensitive re.match from the start against "
                    "str(warning message); the registered trailing .* is "
                    "intentional"
                ),
                "module_regex_semantics": (
                    "Python warnings.filterwarnings semantics: "
                    "case-sensitive re.match from the start against the "
                    "fully qualified module name supplied by the warnings "
                    "machinery"
                ),
                "approved_warning_count_semantics": (
                    "length of the recorded WarningMessage list produced by "
                    "the exact matching_implementation; any warning not "
                    "selected by the approved filter is raised as an "
                    "exception by the error filter"
                ),
                "approved_warning_count_must_be_recorded": True,
                "warnings_during_predict_proba": "ABORT_ENTIRE_RUN",
                "all_other_warnings": "ABORT_ENTIRE_RUN",
            },
        )
        receipt = self.v2["receipt_contract"]
        self.assertEqual(
            receipt["model_invariant_required_values"]["warning_policy_id"],
            policy["policy_id"],
        )
        self.assertEqual(
            receipt["model_invariant_required_values"]["unexpected_warning_count"],
            0,
        )

    def test_v2_identifiers_feature_hashes_and_orders_are_unambiguous(self) -> None:
        """Chaque identifiant et chaque ordre de ligne a une définition unique."""
        canonical = self.v2["canonicalization"]
        self.assertEqual(
            canonical["slot_key_components"],
            ["shadow_slot_v2", "shadow_protocol_sha256", "target_official_date"],
        )
        self.assertEqual(
            canonical["batch_id_components"],
            [
                "shadow_batch_v2",
                "slot_key",
                "execution_manifest_sha256",
                "model_artifact_sha256",
            ],
        )
        self.assertEqual(canonical["occurrence_key_components"][0], "shadow_occurrence_v2")
        self.assertEqual(canonical["prediction_id_components"][0], "shadow_prediction_v2")
        self.assertEqual(
            canonical["feature_row_sha256_components"],
            [
                "shadow_feature_row_v2",
                "feature_row_values_in_features_columns_exact_order_"
                "excluding_feature_row_sha256",
            ],
        )
        formulas = canonical["identifier_formulas_exact"]
        self.assertEqual(
            set(formulas),
            {
                "slot_key",
                "batch_id",
                "occurrence_key",
                "prediction_id",
                "feature_row_sha256",
            },
        )
        for name, formula in formulas.items():
            self.assertIn("runtime_", formula, name)
            self.assertNotIn("_components))", formula, name)

        known_answers = canonical["known_answer_vectors"]
        self.assertEqual(
            set(known_answers),
            {
                "slot_key",
                "batch_id",
                "occurrence_key",
                "occurrence_key_missing_start",
                "prediction_id",
                "feature_row_sha256",
            },
        )
        for name, vector in known_answers.items():
            self.assertEqual(
                _canonical_json_sha256(vector["preimage"]),
                vector["sha256"],
                name,
            )
        self.assertEqual(
            known_answers["occurrence_key_missing_start"]["preimage"][-1],
            None,
        )
        source_mapping = canonical["identifier_source_field_mapping_exact"]
        self.assertIn(
            "target_schedule.official_date",
            source_mapping["runtime_official_date_at_snapshot"],
        )
        self.assertIn(
            "target_schedule.game_datetime_utc",
            source_mapping[
                "runtime_scheduled_start_utc_at_snapshot_or_null"
            ],
        )
        self.assertIn(
            "JSON null forbidden",
            source_mapping["runtime_scheduled_start_utc_at_snapshot"],
        )

        idempotency = self.v2["idempotency"]
        self.assertIn("using runtime values", idempotency["slot_key_formula"])
        self.assertIn("using runtime values", idempotency["batch_id_formula"])
        self.assertNotIn(
            "canonical_json(batch_id_components)",
            idempotency["batch_id_formula"],
        )

        ordering = canonical["row_ordering"]
        self.assertEqual(ordering["sort_direction_for_every_term"], "ASCENDING")
        self.assertTrue(ordering["integer_identifiers_sort_numerically"])
        self.assertEqual(ordering["teams"], ["team_id"])
        self.assertEqual(ordering["source_final_games"], ["official_date", "game_id"])
        self.assertEqual(
            ordering["target_schedule"],
            ["missing_time_flag", "game_datetime_utc", "game_id"],
        )
        self.assertEqual(
            ordering["candidate_ledger"],
            ["missing_time_flag", "scheduled_start_utc", "game_id"],
        )
        self.assertEqual(ordering["features"], ["scheduled_start_utc", "game_id"])
        self.assertEqual(
            ordering["predictions"],
            ["scheduled_start_utc_at_prediction", "game_id"],
        )
        self.assertTrue(ordering["ordering_terms_are_derived_and_not_additional_output_fields"])
        self.assertEqual(
            ordering["missing_time_sort_value"],
            "EMPTY_STRING_AFTER_MISSING_TIME_FLAG",
        )

    def test_missing_time_and_both_insufficient_histories_are_closed(self) -> None:
        """Les deux cas autrefois ambigus ont une issue déterministe."""
        candidates = self.v2["candidate_games"]
        operating = self.v2["operating_mode"]
        self.assertEqual(operating["target_season_exact"], 2026)
        self.assertTrue(
            operating["future_other_seasons_require_new_registered_protocol"]
        )
        self.assertEqual(candidates["season_must_equal"], 2026)
        self.assertFalse(candidates["game_datetime_utc_required"])
        self.assertIsNone(candidates["missing_game_datetime_utc_canonical_value"])
        self.assertTrue(candidates["missing_start_time_is_exclusion"])
        self.assertTrue(
            candidates[
                "postponed_and_cancelled_classification_precedes_abstract_state_validation"
            ]
        )
        self.assertTrue(
            candidates[
                "required_abstract_state_applies_only_after_postponed_and_cancelled_exclusion"
            ]
        )
        self.assertEqual(
            candidates["allowed_exclusion_reasons"],
            [
                "POSTPONED",
                "CANCELLED",
                "START_TIME_MISSING",
                "INSUFFICIENT_BOTH_HISTORY",
                "INSUFFICIENT_AWAY_HISTORY",
                "INSUFFICIENT_HOME_HISTORY",
            ],
        )
        self.assertEqual(
            candidates["insufficient_history_reason_priority"],
            [
                "INSUFFICIENT_BOTH_HISTORY",
                "INSUFFICIENT_AWAY_HISTORY",
                "INSUFFICIENT_HOME_HISTORY",
            ],
        )
        self.assertIn("BOTH when away and home", candidates["insufficient_history_reason_rule"])
        self.assertEqual(
            candidates["classification_order_exact"],
            [
                "VALIDATE_STRUCTURE_IDENTITY_SPORT_TYPE_SEASON_DATE_AND_PREGAME_SCORE_PAIR",
                "MATCH_EXACT_POSTPONED_PAIR",
                "MATCH_EXACT_CANCELLED_PAIR",
                "FAIL_IF_LIVE_OR_FINAL_ABSTRACT_STATE",
                "MATCH_EXACT_SCHEDULED_OR_PRE_GAME_TRIPLET_OTHERWISE_FAIL_UNKNOWN_STATE",
                "EXCLUDE_IF_START_TIME_MISSING",
                "EXCLUDE_IF_BOTH_TEAMS_HAVE_INSUFFICIENT_HISTORY",
                "EXCLUDE_IF_ONLY_AWAY_TEAM_HAS_INSUFFICIENT_HISTORY",
                "EXCLUDE_IF_ONLY_HOME_TEAM_HAS_INSUFFICIENT_HISTORY",
                "MARK_ELIGIBLE",
            ],
        )
        tables = candidates["classification_tables"]
        self.assertEqual(
            tables["POSTPONED"]["normalized_status_codes"],
            ["D", "DI", "DR"],
        )
        self.assertEqual(
            tables["CANCELLED"]["normalized_status_codes"],
            ["C", "CI", "CR"],
        )
        self.assertEqual(
            tables["SCHEDULED"]["normalized_status_codes"],
            ["S"],
        )
        self.assertEqual(
            tables["PRE_GAME"]["normalized_status_codes"],
            ["P"],
        )
        self.assertEqual(
            candidates["eligibility_status_domain_exact"],
            ["ELIGIBLE", "EXCLUDED"],
        )
        self.assertIsNone(candidates["eligible_exclusion_reason_canonical_value"])
        output = self.v2["output_contract"]
        self.assertIn("status_code", output["target_schedule_row_keys_exact_set"])
        self.assertIn("status_code", output["candidate_ledger_columns_in_exact_order"])
        self.assertEqual(
            output["candidate_ledger_empty_field_rules"],
            {
                "exclusion_reason_when_eligibility_status_is_ELIGIBLE": (
                    "empty CSV field encoding internal JSON null"
                ),
                "scheduled_start_utc_when_target_schedule_game_datetime_utc_is_null": (
                    "empty CSV field encoding internal JSON null regardless of "
                    "exclusion_reason"
                ),
                "all_other_fields": "NONEMPTY",
            },
        )

    def test_feature_contract_is_exactly_the_validated_j_minus_one_contract(self) -> None:
        """La v2 ne change aucune variable du modèle validé."""
        features = self.v2["feature_contract"]
        self.assertEqual(features["dataset_semantics"], "mlb_team_form_v1")
        self.assertEqual(features["calendar_policy"], "J_MINUS_1_BY_OFFICIAL_DATE")
        self.assertEqual(features["minimum_history_games_per_team"], 10)
        self.assertEqual(features["feature_columns_in_exact_order"], EXPECTED_FEATURES)
        self.assertTrue(features["same_day_games_never_feed_each_other"])
        self.assertTrue(features["target_scores_and_outcomes_forbidden_as_features"])
        self.assertTrue(features["current_season_results_must_never_fit_or_recalibrate_model"])
        validation = features["source_game_validation"]
        self.assertEqual(validation["game_type_exact"], "R")
        self.assertTrue(validation["official_date_must_be_strictly_before_target_official_date"])
        self.assertFalse(validation["tied_final_scores_allowed"])
        self.assertFalse(validation["duplicate_game_ids_allowed"])
        self.assertEqual(
            set(features["team_state_update_formulas"]),
            {
                "away_team.games",
                "away_team.wins",
                "away_team.runs_scored",
                "away_team.runs_allowed",
                "away_team.max_source_date",
                "home_team.games",
                "home_team.wins",
                "home_team.runs_scored",
                "home_team.runs_allowed",
                "home_team.max_source_date",
            },
        )
        self.assertEqual(
            features["feature_value_formulas"],
            {
                "away_games_before": "away_team.games",
                "away_win_pct_before": "away_team.wins / away_team.games",
                "away_runs_scored_per_game_before": (
                    "away_team.runs_scored / away_team.games"
                ),
                "away_runs_allowed_per_game_before": (
                    "away_team.runs_allowed / away_team.games"
                ),
                "home_games_before": "home_team.games",
                "home_win_pct_before": "home_team.wins / home_team.games",
                "home_runs_scored_per_game_before": (
                    "home_team.runs_scored / home_team.games"
                ),
                "home_runs_allowed_per_game_before": (
                    "home_team.runs_allowed / home_team.games"
                ),
            },
        )
        self.assertEqual(
            features["minimum_history_rule_exact"],
            "away_team.games >= 10 AND home_team.games >= 10 before any "
            "division or feature row creation",
        )
        quantization = features["feature_quantization"]
        self.assertEqual(quantization["matrix_dtype"], "numpy.float64")
        self.assertEqual(
            quantization["matrix_column_order"],
            "feature_columns_in_exact_order",
        )
        self.assertIn("format(binary64_value, '.6f')", quantization["rates_and_averages"])
        self.assertIn(
            "game_type",
            self.v2["output_contract"]["source_final_game_row_keys_exact_set"],
        )

    def test_model_deserialization_and_state_hash_are_fully_specified(self) -> None:
        """Le modèle est chargé et comparé sans appel d'entraînement caché."""
        model = self.v2["model_use"]
        deserialization = model["artifact_deserialization_contract"]
        self.assertEqual(
            deserialization["loader"],
            "joblib.load(io.BytesIO(trusted_bytes))",
        )
        self.assertIn("__main__.CalibratedModelArtifact", deserialization["legacy_type_alias"])
        self.assertEqual(
            deserialization["artifact_type_exact"],
            "src.calibrated_model.CalibratedModelArtifact",
        )
        state = model["artifact_state_sha256_method"]
        self.assertEqual(state["serializer"], "joblib.dump")
        self.assertEqual(state["destination"], "io.BytesIO")
        self.assertEqual(state["compress"], 3)
        self.assertEqual(
            state["digest"],
            "sha256(buffer.getvalue()).hexdigest()",
        )
        self.assertTrue(state["computed_immediately_before_predict_proba"])
        self.assertTrue(state["computed_immediately_after_predict_proba"])
        for forbidden_fit in (
            "fit_allowed",
            "partial_fit_allowed",
            "recalibration_allowed",
            "threshold_tuning_allowed",
            "feature_selection_allowed",
            "fallback_model_allowed",
        ):
            self.assertFalse(model[forbidden_fit], forbidden_fit)
        self.assertEqual(model["maximum_predict_proba_calls_per_nonempty_batch"], 1)

    def test_result_receipt_and_certification_paths_are_v2_and_tracked(self) -> None:
        """Les sorties v2 sont append-only, séparées et vérifiables."""
        output = self.v2["output_contract"]
        receipt = self.v2["receipt_contract"]
        certification = self.v2["prospective_certification"]

        self.assertEqual(
            output["result_root"],
            "shadow_results/logistic_team_form_v1_platt_shadow_v2",
        )
        self.assertTrue(output["result_root_must_be_tracked_by_git"])
        expected_result_files = [
            "RESERVED",
            "activation_reverification.remote.json.gz",
            "source_snapshot.json.gz",
            "candidate_ledger.csv",
            "features.csv",
            "predictions.csv",
            "receipt.json",
            "COMPLETED",
        ]
        self.assertEqual(
            output["success_files_in_write_order"],
            expected_result_files,
        )
        self.assertEqual(self.v2["idempotency"]["write_mode"], "EXCLUSIVE_NO_OVERWRITE")
        self.assertEqual(
            certification["certification_path_template"],
            "shadow_certifications/logistic_team_form_v1_platt_shadow_v2/"
            "YYYY-MM-DD.json",
        )
        self.assertEqual(
            certification["raw_remote_evidence_path_template"],
            "shadow_certifications/logistic_team_form_v1_platt_shadow_v2/"
            "YYYY-MM-DD.remote.json.gz",
        )
        self.assertTrue(certification["certification_root_must_be_tracked_by_git"])
        self.assertTrue(certification["certification_is_outside_completed_batch_directory"])
        self.assertEqual(certification["certification_write_mode"], "APPEND_ONLY_EXCLUSIVE_CREATE")
        self.assertEqual(certification["raw_remote_evidence_write_mode"], "APPEND_ONLY_EXCLUSIVE_CREATE")

        self.assertEqual(
            set(receipt["top_level_keys_exact_set"]),
            {
                "receipt_schema_version",
                "batch",
                "activation",
                "times",
                "schedule_http_response",
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
            set(receipt["activation_keys_exact_set"]),
            {
                "execution_manifest_introduction_commit",
                "activation_introduction_commit",
                "activation_path",
                "activation_sha256",
                "activation_verified_at_utc",
                "activation_remote_reverified_at_utc",
                "activation_remote_ref",
                "activation_remote_reverification_query_url",
                "activation_remote_reverification_effective_url",
                "activation_remote_reverification_status_code",
                "activation_remote_reverification_redirect_count",
                "activation_remote_reverification_response_received_at_utc",
                "activation_remote_reverification_response_body_sha256",
                "activation_remote_reverification_evidence_path",
                "activation_remote_reverification_evidence_sha256",
                "minimum_target_official_date",
            },
        )
        activation_reverification_bindings = receipt[
            "activation_reverification_to_raw_remote_evidence_invariants_exact"
        ]
        self.assertEqual(len(activation_reverification_bindings), 12)
        self.assertTrue(
            any(
                "application_request_headers_exact" in rule
                for rule in activation_reverification_bindings
            )
        )
        self.assertTrue(
            any(
                "output_hashes.activation_reverification_evidence_sha256"
                in rule
                for rule in activation_reverification_bindings
            )
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
                "remote_effective_url",
                "remote_response_status_code",
                "remote_response_redirect_count",
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
            certification["results_tree_file_hashes_expected_relative_paths"],
            sorted(expected_result_files),
        )
        self.assertEqual(
            certification["results_tree_file_hashes_order"],
            "LEXICOGRAPHIC_BY_RELATIVE_PATH",
        )
        self.assertEqual(
            set(certification["raw_remote_evidence_keys_exact_set"]),
            set(self.v2["activation_evidence"]["raw_remote_evidence_keys_exact_set"]),
        )
        self.assertEqual(certification["remote_request_method_exact"], "GET")
        self.assertEqual(certification["remote_response_status_code_exact"], 200)
        self.assertEqual(certification["remote_response_redirect_count_exact"], 0)
        self.assertTrue(certification["remote_effective_url_must_equal_query_url"])
        self.assertEqual(
            certification["certification_write_order_exact"],
            ["RAW_REMOTE_EVIDENCE_GZIP", "CERTIFICATION_JSON"],
        )
        self.assertEqual(
            certification["certification_status_exact"],
            "PROSPECTIVELY_CERTIFIED_REMOTE_MAIN_BEFORE_GAMES",
        )
        self.assertIn(
            ">= 3600 exact integer seconds",
            certification["remote_lead_threshold_rule"],
        )
        self.assertIn(
            "decimal.ROUND_HALF_EVEN",
            certification["remote_publication_lead_minutes_formula"],
        )
        self.assertEqual(
            certification["remote_publication_lead_minutes_type"],
            "JSON string matching ^[0-9]+\\.[0-9]{6}$",
        )
        certification_bindings = certification[
            "certification_json_to_raw_remote_evidence_invariants_exact"
        ]
        self.assertEqual(len(certification_bindings), 12)
        self.assertTrue(
            any("application_request_headers_exact" in rule for rule in certification_bindings)
        )
        self.assertTrue(
            any("parse_IMF_FIXDATE_GMT" in rule for rule in certification_bindings)
        )
        batch_bindings = certification[
            "certification_to_completed_batch_invariants_exact"
        ]
        self.assertEqual(len(batch_bindings), 6)
        self.assertTrue(
            any(
                "minimum predictions.scheduled_start_utc_at_prediction" in rule
                for rule in batch_bindings
            )
        )
        self.assertIn(
            "create both raw evidence gzip and certification JSON only when",
            certification["certification_file_creation_policy"],
        )
        self.assertIn(
            "preserve the orphan gzip",
            certification["partial_certification_failure_policy"],
        )
        self.assertIn(
            "certification is forbidden",
            certification["empty_batch_certification_policy"],
        )
        self.assertIn(
            "git show results_commit:",
            certification["results_tree_file_hashes_source_of_truth"],
        )
        self.assertTrue(
            any(
                "current working tree is forbidden" in rule
                for rule in [certification["results_tree_file_hashes_source_of_truth"]]
            )
        )
        self.assertEqual(
            len(certification["results_tree_file_hashes_validation_rules"]),
            6,
        )

        self.assertEqual(
            output["schema_versions_exact"],
            {
                "execution_manifest_schema_version": 1,
                "activation_schema_version": 1,
                "remote_evidence_schema_version": 1,
                "reserved_marker_schema_version": 1,
                "failed_marker_schema_version": 1,
                "completed_marker_schema_version": 1,
                "source_snapshot_schema_version": 1,
                "receipt_schema_version": 1,
                "certification_schema_version": 1,
            },
        )
        self.assertEqual(
            output["batch_status_domain_exact"],
            ["COMPLETED_WITH_PREDICTIONS", "COMPLETED_NO_ELIGIBLE_GAMES"],
        )
        empty = output["empty_eligible_batch"]
        self.assertEqual(empty["predict_proba_calls"], 0)
        self.assertEqual(empty["features_and_predictions_csv"], "HEADER_ONLY")
        self.assertFalse(empty["prospective_remote_certification_required"])
        empty_invariants = receipt["empty_batch_model_invariant_values"]
        self.assertIsNone(empty_invariants["artifact_state_sha256_before"])
        self.assertIsNone(empty_invariants["artifact_state_sha256_after"])
        self.assertTrue(empty_invariants["artifact_state_unchanged"])
        self.assertEqual(empty_invariants["predict_proba_calls"], 0)
        self.assertIn(
            "when predicted_games == 0",
            receipt["artifact_state_hash_rule"],
        )

    def test_receipt_counts_are_conserved_with_every_exclusion_reason(self) -> None:
        """Aucun match du calendrier ne peut disparaître des compteurs."""
        receipt = self.v2["receipt_contract"]
        self.assertEqual(
            receipt["excluded_games_by_reason_keys_exact_set"],
            self.v2["candidate_games"]["allowed_exclusion_reasons"],
        )
        self.assertEqual(
            receipt["count_conservation_rules"],
            [
                "schedule_games == eligible_games + "
                "sum(excluded_games_by_reason.values())",
                "predicted_games == eligible_games",
            ],
        )
        self.assertEqual(
            receipt["predict_proba_calls_rule"],
            "1 if predicted_games > 0 else 0",
        )
        row_invariants = receipt["row_count_and_identity_invariants_exact"]
        self.assertEqual(len(row_invariants), 8)
        self.assertIn(
            "counts.schedule_games == len(source_snapshot.target_schedule) "
            "== data_row_count(candidate_ledger.csv)",
            row_invariants,
        )
        self.assertTrue(
            any(
                "ordered (game_id, occurrence_key)" in rule
                and "features.csv" in rule
                and "predictions.csv" in rule
                for rule in row_invariants
            )
        )
        self.assertTrue(
            any("counts.eligible_games == 0" in rule for rule in row_invariants)
        )
        cross_invariants = receipt[
            "cross_section_and_cross_file_invariants_exact"
        ]
        for required_fragment in (
            "candidate_ledger_sha256",
            "features_sha256",
            "predictions_sha256",
            "schedule_http_response.date_header_utc",
            "activation_remote_reverification_evidence_sha256",
            "COMPLETED.receipt_sha256",
            "validated_lineage.sealed_evaluation.report_sha256",
        ):
            self.assertTrue(
                any(required_fragment in rule for rule in cross_invariants),
                required_fragment,
            )

    def test_target_schemas_have_no_outcome_odds_or_betting_fields(self) -> None:
        """La phase fantôme ne produit que des probabilités prospectives."""
        output = self.v2["output_contract"]
        forbidden = set(output["forbidden_exact_target_or_prediction_field_names"])
        self.assertTrue(
            {
                "away_score",
                "home_score",
                "home_win",
                "outcome",
                "odds",
                "pick",
                "stake",
                "edge",
                "expected_value",
                "profit",
                "roi",
            }.issubset(forbidden)
        )
        target_columns = (
            set(output["target_schedule_row_keys_exact_set"])
            | set(output["candidate_ledger_columns_in_exact_order"])
            | set(output["features_columns_in_exact_order"])
            | set(output["predictions_columns_in_exact_order"])
        )
        self.assertFalse(forbidden & target_columns)

        betting = self.v2["scoring_and_betting"]
        forbidden_switches = {
            key: value
            for key, value in betting.items()
            if key.endswith("_allowed") or key.endswith("_enabled")
        }
        self.assertTrue(forbidden_switches)
        self.assertTrue(all(value is False for value in forbidden_switches.values()))
        self.assertFalse(betting["season_2026_metrics_allowed"])
        self.assertFalse(betting["profitability_claim_allowed"])

        attestations = set(
            self.v2["receipt_contract"]["negative_attestations_keys_exact_set"]
        )
        self.assertIn("ODDS_NOT_READ", attestations)
        self.assertIn("BETTING_RECOMMENDATIONS_NOT_COMPUTED", attestations)
        self.assertIn("SEASON_2026_METRICS_NOT_COMPUTED", attestations)
        self.assertTrue(
            self.v2["receipt_contract"]["negative_attestations_required_value"]
        )

    def test_http_observation_is_durable_and_preflight_is_closed(self) -> None:
        """La date serveur MLB doit être prouvée, normalisée et conservée."""
        clock = self.v2["daily_batch"]["clock_evidence"]
        source = self.v2["source_snapshot"]
        output = self.v2["output_contract"]
        receipt = self.v2["receipt_contract"]
        preflight = set(self.v2["preflight_invariants"])

        expected_schedule_keys = {
            "run_id",
            "source",
            "requested_start_date",
            "requested_end_date",
            "game_types",
            "request_parameters_json",
            "completed_at_utc",
            "raw_archive_path",
            "raw_archive_sha256",
            "response_effective_url",
            "response_status_code",
            "response_redirect_count",
            "mlb_http_date_header_raw",
            "mlb_http_date_utc",
            "mlb_http_response_received_at_utc",
            "response_body_sha256",
        }
        self.assertEqual(
            set(output["schedule_ingestion_keys_exact_set"]),
            expected_schedule_keys,
        )
        self.assertEqual(clock["mlb_response_effective_url_scheme_exact"], "https")
        self.assertEqual(
            clock["mlb_response_effective_url_host_exact"],
            "statsapi.mlb.com",
        )
        self.assertEqual(
            clock["mlb_response_effective_url_path_exact"],
            "/api/v1/schedule",
        )
        self.assertEqual(clock["mlb_response_status_code_exact"], 200)
        self.assertEqual(clock["mlb_response_redirect_count_exact"], 0)
        self.assertEqual(clock["mlb_http_date_header_raw_format"], "IMF_FIXDATE_GMT")
        self.assertEqual(
            clock["clock_skew_formula"],
            "mlb_http_response_received_at_utc - mlb_http_date_utc",
        )
        self.assertEqual(
            self.v2["daily_batch"]["schedule_request_parameters_exact"],
            {
                "sportId": 1,
                "startDate": "runtime_target_official_date",
                "endDate": "runtime_target_official_date",
                "gameTypes": "R",
                "hydrate": "probablePitcher",
            },
        )
        self.assertIn(
            "duplicate query names forbidden",
            self.v2["daily_batch"]["schedule_effective_url_query_validation"],
        )
        self.assertEqual(
            self.v2["daily_batch"]["schedule_observed_at_utc_formula"],
            "mlb_http_response_received_at_utc",
        )
        self.assertEqual(
            self.v2["daily_batch"]["schedule_age_seconds_allowed_range_inclusive"],
            [0, 900],
        )
        self.assertEqual(
            source["schedule_age_formula"],
            "information_cutoff_utc - "
            "schedule_ingestion.mlb_http_response_received_at_utc",
        )
        self.assertTrue(
            source["schedule_age_must_be_nonnegative_and_at_most_900_seconds"]
        )
        self.assertTrue(
            source[
                "schedule_http_observation_must_be_persisted_in_canonical_source_subset"
            ]
        )
        self.assertTrue(
            source["schedule_http_body_sha256_must_equal_raw_archive_sha256"]
        )
        self.assertTrue(
            {
                "REQUIRE_MLB_SCHEDULE_RESPONSE_HTTPS_EXACT_HOST_PATH_STATUS_AND_NO_REDIRECT",
                "REQUIRE_EXACT_MLB_SCHEDULE_QUERY_PARAMETERS_AND_INGESTION_PROVENANCE",
                "REQUIRE_MLB_DATE_HEADER_RAW_AND_NORMALIZED_WITHIN_CLOCK_SKEW_LIMIT",
                "REQUIRE_MLB_HTTP_BODY_SHA256_EQUALS_ARCHIVED_RAW_RESPONSE_SHA256",
            }.issubset(preflight)
        )
        self.assertEqual(
            receipt["schedule_http_response_keys_exact_set"],
            [
                "effective_url",
                "status_code",
                "redirect_count",
                "date_header_raw",
                "date_header_utc",
                "received_at_utc",
                "body_sha256",
            ],
        )
        self.assertEqual(
            receipt["times_keys_exact_set"],
            [
                "started_at_utc",
                "reserved_at_utc",
                "schedule_observed_at_utc",
                "information_cutoff_utc",
                "issued_at_utc",
                "receipt_finalized_at_utc",
                "mlb_http_date_utc",
                "mlb_http_response_received_at_utc",
                "clock_skew_seconds",
                "schedule_age_seconds",
            ],
        )
        self.assertEqual(
            receipt["times_invariant_rules"],
            [
                "started_at_utc <= activation_reverification_raw."
                "response_received_at_utc <= reserved_at_utc",
                "reserved_at_utc == RESERVED.reserved_at_utc",
                "schedule_observed_at_utc == mlb_http_response_received_at_utc",
                "schedule_age_seconds == information_cutoff_utc - "
                "schedule_observed_at_utc",
                "0 <= schedule_age_seconds <= 900",
                "reserved_at_utc <= schedule_observed_at_utc <= "
                "information_cutoff_utc <= issued_at_utc <= "
                "receipt_finalized_at_utc",
                "receipt_finalized_at_utc <= COMPLETED.completed_at_utc",
            ],
        )
        self.assertIn(
            "activation_reverification_evidence_sha256",
            receipt["output_hashes_keys_exact_set"],
        )
        self.assertTrue(
            {
                "schedule_source",
                "schedule_requested_start_date",
                "schedule_requested_end_date",
                "schedule_game_types",
                "schedule_request_parameters_json",
            }.issubset(set(receipt["source_keys_exact_set"]))
        )


if __name__ == "__main__":
    unittest.main()
