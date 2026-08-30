"""Verrouille le manifeste du modèle avant l'ouverture de 2025."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import unittest

from src.baseline_model import (
    MODEL_VERSION,
    SEALED_RECENT_SEASONS,
    SEALED_TEST_SEASONS,
)
from src.calibrated_model import (
    ARTIFACT_FORMAT_VERSION,
    BASE_TRAINING_SEASONS,
    CALIBRATED_MODEL_VERSION,
    CALIBRATION_METHOD,
    CALIBRATION_SEASON,
    EXPECTED_PROTOCOL_SHA256,
)
from src.training_dataset import DATASET_VERSION, FEATURE_COLUMNS


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_DIRECTORY
    / "model_artifacts"
    / "logistic_team_form_v1_platt.json"
)
PROTOCOL_PATH = (
    PROJECT_DIRECTORY
    / "model_protocols"
    / "logistic_team_form_v1.json"
)
REQUIREMENTS_PATH = PROJECT_DIRECTORY / "requirements.txt"

EXPECTED_MANIFEST_SHA256 = (
    "a7375d5376baa043b3365cff713cb717"
    "010ad812efee472b594fa454306c96ae"
)
EXPECTED_ARTIFACT_SHA256 = (
    "e0d4d2421ba076072c7ef8b3bc97dd9"
    "a341e26c62828a0ad9ba43f30da15ff55"
)
EXPECTED_DATASET_SHA256 = (
    "2a24c1a22a919acfc4ea545f8025f59c"
    "86d15873aaf09cdbdcef66236cd0a73e"
)
EXPECTED_CODE_VERSION = (
    "1d13f8f361de0865f722a8560f1efda8e95badbc"
)


def _read_json(path: Path) -> tuple[bytes, dict[str, object]]:
    """Lit un JSON suivi par Git sans dépendre des fichiers ignorés."""
    content = path.read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise AssertionError(f"Objet JSON attendu dans {path}.")
    return content, payload


def _pinned_requirements(path: Path) -> dict[str, str]:
    """Relit uniquement les versions écrites avec un double égal."""
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if separator != "==" or not name or not version:
            raise AssertionError(
                f"Dépendance non figée dans requirements.txt : {line}"
            )
        pins[name] = version
    return pins


class ModelArtifactManifestTests(unittest.TestCase):
    """Empêche toute mutation silencieuse avant le test scellé."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_bytes, cls.manifest = _read_json(MANIFEST_PATH)
        cls.protocol_bytes, cls.protocol = _read_json(PROTOCOL_PATH)

    def test_manifest_file_is_exactly_frozen_before_2025(self) -> None:
        """Le moindre octet modifié doit invalider l'enregistrement."""
        actual_sha256 = hashlib.sha256(
            self.manifest_bytes
        ).hexdigest()
        self.assertEqual(actual_sha256, EXPECTED_MANIFEST_SHA256)
        self.assertEqual(self.manifest["manifest_version"], 1)
        self.assertEqual(
            self.manifest["status"],
            "FROZEN_BEFORE_SEALED_TEST",
        )
        self.assertEqual(self.manifest["registered_on"], "2026-08-30")

    def test_artifact_identity_and_safe_path_are_fixed(self) -> None:
        """Chemin, empreinte, taille et commit doivent rester exacts."""
        artifact = self.manifest["artifact"]
        self.assertIsInstance(artifact, dict)
        artifact_path = PurePosixPath(artifact["path"])

        self.assertFalse(artifact_path.is_absolute())
        self.assertNotIn("..", artifact_path.parts)
        self.assertEqual(
            artifact_path,
            PurePosixPath(
                "models/logistic_team_form_v1_platt.joblib"
            ),
        )
        self.assertEqual(artifact["serializer"], "joblib")
        self.assertEqual(artifact["compression_level"], 3)
        self.assertEqual(
            artifact["artifact_format_version"],
            ARTIFACT_FORMAT_VERSION,
        )
        self.assertEqual(
            artifact["sha256"],
            EXPECTED_ARTIFACT_SHA256,
        )
        self.assertEqual(artifact["size_bytes"], 1589)
        self.assertEqual(
            artifact["code_version"],
            EXPECTED_CODE_VERSION,
        )
        self.assertRegex(
            artifact["code_version"],
            re.compile(r"[0-9a-f]{40}\Z"),
        )
        self.assertIs(
            artifact["round_trip_verified_after_write"],
            True,
        )
        self.assertIs(artifact["external_copy_verified"], True)

    def test_model_contract_matches_the_implemented_constants(self) -> None:
        """Version, calibration et variables ne peuvent plus changer."""
        model = self.manifest["model"]
        self.assertEqual(
            model["calibrated_model_version"],
            CALIBRATED_MODEL_VERSION,
        )
        self.assertEqual(model["base_model_version"], MODEL_VERSION)
        self.assertEqual(
            model["calibration_method"],
            CALIBRATION_METHOD,
        )
        self.assertIs(
            model["base_model_unchanged_during_calibration"],
            True,
        )
        self.assertEqual(
            tuple(model["feature_columns"]),
            FEATURE_COLUMNS,
        )

    def test_dataset_and_protocol_provenance_are_consistent(self) -> None:
        """Les deux sources approuvées gardent leur identité exacte."""
        dataset = self.manifest["dataset"]
        protocol_reference = self.manifest["protocol"]

        self.assertEqual(dataset["version"], DATASET_VERSION)
        self.assertEqual(
            dataset["path"],
            "data/processed/mlb_team_form_v1_2026-08-29.csv",
        )
        self.assertEqual(dataset["sha256"], EXPECTED_DATASET_SHA256)
        self.assertEqual(dataset["size_bytes"], 1680593)
        self.assertEqual(dataset["row_count"], 13253)
        self.assertEqual(dataset["cutoff_date"], "2026-08-29")

        actual_protocol_sha256 = hashlib.sha256(
            self.protocol_bytes
        ).hexdigest()
        self.assertEqual(
            actual_protocol_sha256,
            EXPECTED_PROTOCOL_SHA256,
        )
        self.assertEqual(
            protocol_reference["path"],
            "model_protocols/logistic_team_form_v1.json",
        )
        self.assertEqual(
            protocol_reference["sha256"],
            EXPECTED_PROTOCOL_SHA256,
        )
        self.assertEqual(
            self.protocol["dataset"]["sha256"],
            EXPECTED_DATASET_SHA256,
        )
        self.assertEqual(
            self.protocol["dataset"]["row_count"],
            dataset["row_count"],
        )
        self.assertEqual(
            tuple(self.protocol["features"]),
            FEATURE_COLUMNS,
        )

    def test_chronological_cohorts_are_disjoint_and_complete(self) -> None:
        """2021-2023, 2024, 2025 et 2026 ne se chevauchent pas."""
        chronology = self.manifest["chronology"]
        training = tuple(chronology["base_training_seasons"])
        calibration = (chronology["calibration_season"],)
        sealed_test = tuple(chronology["sealed_test_seasons"])
        recent = tuple(chronology["recent_seasons"])
        all_seasons = training + calibration + sealed_test + recent

        self.assertEqual(training, BASE_TRAINING_SEASONS)
        self.assertEqual(calibration, (CALIBRATION_SEASON,))
        self.assertEqual(sealed_test, SEALED_TEST_SEASONS)
        self.assertEqual(recent, SEALED_RECENT_SEASONS)
        self.assertEqual(len(all_seasons), len(set(all_seasons)))
        self.assertEqual(chronology["base_training_rows"], 6815)
        self.assertEqual(chronology["calibration_rows"], 2271)
        self.assertEqual(chronology["sealed_test_rows"], 2276)
        self.assertEqual(chronology["recent_rows"], 1891)
        self.assertEqual(
            chronology["base_training_rows"]
            + chronology["calibration_rows"]
            + chronology["sealed_test_rows"]
            + chronology["recent_rows"],
            self.manifest["dataset"]["row_count"],
        )
        self.assertIs(chronology["sealed_predictions_computed"], False)

    def test_reference_baseline_uses_only_2021_2023(self) -> None:
        """La comparaison future ne peut apprendre de 2024 ou 2025."""
        baseline = self.manifest["reference_baseline"]
        self.assertEqual(baseline["home_wins"], 3623)
        self.assertEqual(baseline["games"], 6815)
        self.assertEqual(
            tuple(baseline["training_seasons"]),
            BASE_TRAINING_SEASONS,
        )
        self.assertAlmostEqual(
            baseline["home_wins"] / baseline["games"],
            3623 / 6815,
            places=15,
        )
        self.assertIs(baseline["uses_calibration_season"], False)
        self.assertIs(baseline["uses_sealed_test_season"], False)
        self.assertIs(baseline["uses_recent_season"], False)

    def test_runtime_and_primary_dependency_pins_are_recorded(self) -> None:
        """L'environnement ayant produit le joblib reste explicite."""
        runtime = self.manifest["runtime"]
        pins = _pinned_requirements(REQUIREMENTS_PATH)

        self.assertEqual(
            runtime,
            {
                "python": "3.12.1",
                "numpy": "2.5.2",
                "pandas": "3.0.5",
                "scipy": "1.18.1",
                "scikit_learn": "1.9.0",
                "joblib": "1.5.3",
            },
        )
        self.assertEqual(pins["numpy"], runtime["numpy"])
        self.assertEqual(pins["pandas"], runtime["pandas"])
        self.assertEqual(
            pins["scikit-learn"],
            runtime["scikit_learn"],
        )
        self.assertEqual(pins["joblib"], runtime["joblib"])

    def test_loading_policy_requires_hash_before_deserialization(self) -> None:
        """Un joblib inconnu ne doit jamais être désérialisé."""
        policy = self.manifest["loading_policy"]
        self.assertIs(policy["pickle_based_format"], True)
        self.assertIs(
            policy["verify_sha256_before_deserialization"],
            True,
        )
        self.assertIs(policy["accept_untrusted_artifact"], False)


if __name__ == "__main__":
    unittest.main()
