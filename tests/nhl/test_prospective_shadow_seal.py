"""A research forecast may only be sealed in the separate private ops tree."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from src.nhl.prospective_shadow_observation import ProspectiveShadowObservation, _digest
from src.nhl.prospective_shadow_seal import (
    NHLPrivateShadowSealError, seal_private_shadow_observation,
)


def observation() -> ProspectiveShadowObservation:
    fields = {
        "game_id": 2026020099,
        "scheduled_start_utc": "2026-10-10T20:00:00.000000Z",
        "observed_at_utc": "2026-10-09T12:00:00.000000Z",
        "information_cutoff_utc": "2026-10-10T19:00:00.000000Z",
        "away_abbr": "NYR", "home_abbr": "BOS",
        "home_win_probability": 0.7, "away_win_probability": 0.3,
        "schedule_sha256": "a" * 64, "current_import_sha256": "b" * 64,
        "training_source_sha256": "c" * 64, "training_examples_sha256": "d" * 64,
        "feature_sha256": "e" * 64, "policy_sha256": "f" * 64,
        "official_tie_labels_in_training": 200,
        "status": "PRIVATE_RESEARCH_SHADOW_ONLY",
        "production_training_permitted": False,
        "prediction_publication_permitted": False,
        "betting_recommendation_permitted": False,
    }
    return ProspectiveShadowObservation(**fields, observation_sha256=_digest(fields))


class PrivateShadowSealTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.private = Path(temporary.name) / "fredo-lpf-edge-private-ops"
        self.private.mkdir()
        (self.private / ".git").mkdir()

    def test_seal_once_without_overwrite(self):
        result = seal_private_shadow_observation(
            observation(), private_repository_root=self.private,
            explicit_private_seal=True)
        self.assertTrue((result / "COMPLETED").is_file())
        data = json.loads((result / "observation.json").read_text())
        self.assertEqual(data["game_id"], 2026020099)
        self.assertFalse(data["prediction_publication_permitted"])
        with self.assertRaisesRegex(NHLPrivateShadowSealError, "déjà scellé"):
            seal_private_shadow_observation(
                observation(), private_repository_root=self.private,
                explicit_private_seal=True)

    def test_exact_one_hour_boundary_can_be_sealed(self):
        fields = asdict(observation())
        fields["observed_at_utc"] = fields["information_cutoff_utc"]
        fields.pop("observation_sha256")
        at_cutoff = ProspectiveShadowObservation(
            **fields, observation_sha256=_digest(fields))
        result = seal_private_shadow_observation(
            at_cutoff, private_repository_root=self.private,
            explicit_private_seal=True)
        self.assertTrue((result / "COMPLETED").is_file())

    def test_reject_late_or_tampered_observation(self):
        with self.assertRaisesRegex(NHLPrivateShadowSealError, "incohérente"):
            seal_private_shadow_observation(
                replace(observation(), home_win_probability=0.9),
                private_repository_root=self.private, explicit_private_seal=True)
        invalid = asdict(observation())
        invalid["home_win_probability"] = 1.1
        invalid["away_win_probability"] = -0.1
        invalid.pop("observation_sha256")
        bad_probability = ProspectiveShadowObservation(
            **invalid, observation_sha256=_digest(invalid))
        with self.assertRaisesRegex(NHLPrivateShadowSealError, "incohérente"):
            seal_private_shadow_observation(
                bad_probability, private_repository_root=self.private,
                explicit_private_seal=True)
        late = replace(observation(), observed_at_utc="2026-10-10T19:01:00.000000Z")
        with self.assertRaisesRegex(NHLPrivateShadowSealError, "incohérente"):
            seal_private_shadow_observation(
                late, private_repository_root=self.private, explicit_private_seal=True)

    def test_explicit_private_repository_required(self):
        with self.assertRaisesRegex(NHLPrivateShadowSealError, "explicite"):
            seal_private_shadow_observation(
                observation(), private_repository_root=self.private)
        wrong = self.private.parent / "fredo-mlb-predictor"
        wrong.mkdir()
        (wrong / ".git").mkdir()
        with self.assertRaisesRegex(NHLPrivateShadowSealError, "distinct"):
            seal_private_shadow_observation(
                observation(), private_repository_root=wrong,
                explicit_private_seal=True)


if __name__ == "__main__":
    unittest.main()
