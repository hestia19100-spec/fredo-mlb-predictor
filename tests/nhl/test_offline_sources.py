from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.nhl.offline_sources import (
    NHLOfflineSourceError,
    ObservationKind,
    SYNTHETIC_FIXTURE_MARKER,
    TemporalStatus,
    ValueState,
    assess_observation_for_cutoff,
    canonical_json_bytes,
    eligible_observations,
    load_offline_fixture,
    validate_observation_for_cutoff,
)
from src.nhl.source_registry import require_enabled_provider


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "nhl" / "fixtures" / "source_registry"


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


class NHLOfflineSourceTests(unittest.TestCase):
    def fixture(self, name: str):
        return load_offline_fixture(FIXTURE_ROOT / name)

    def write_variant(self, name: str, change, *, rehash: bool = False) -> Path:
        document = json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))
        change(document)
        if rehash:
            document["response_sha256"] = hashlib.sha256(
                canonical_json_bytes(document["payload"])
            ).hexdigest()
        temporary = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        )
        with temporary:
            json.dump(document, temporary, ensure_ascii=False, sort_keys=True)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        return Path(temporary.name)

    def test_every_valid_fixture_is_deterministic_and_fully_proven(self) -> None:
        paths = sorted(FIXTURE_ROOT.glob("*_valid.json"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            with self.subTest(path=path.name):
                first = load_offline_fixture(path)
                second = load_offline_fixture(path)
                self.assertEqual(first, second)
                self.assertEqual(first.proof.fixture_id, first.observations[0].fixture_id)
                self.assertEqual(
                    first.proof.response_sha256,
                    first.observations[0].evidence.response_sha256,
                )
                self.assertTrue(
                    all(
                        item.temporal_status is TemporalStatus.UNASSESSED
                        for item in first.observations
                    )
                )

    def test_schedule_preserves_canonical_and_rescheduled_times(self) -> None:
        bundle = self.fixture("public_schedule_valid.json")
        regular, rescheduled = bundle.observations
        self.assertEqual(regular.kind, ObservationKind.SCHEDULED_GAME)
        self.assertFalse(regular.value["rescheduled"])
        self.assertTrue(rescheduled.value["rescheduled"])
        self.assertEqual(
            rescheduled.value["original_start_utc"],
            "2026-10-01T22:00:00Z",
        )

    def test_non_final_game_state_is_not_a_final_result(self) -> None:
        observation = self.fixture("public_game_state_valid.json").observations[0]
        self.assertEqual(observation.kind, ObservationKind.GAME_STATE)
        self.assertFalse(observation.value["is_final"])

    def test_prior_final_result_and_team_context_keep_source_game(self) -> None:
        result = self.fixture("public_final_result_valid.json").observations[0]
        stats = self.fixture("stats_team_history_valid.json").observations[0]
        self.assertEqual(result.source_game_id, 2025021312)
        self.assertEqual(result.value["game_state"], "FINAL")
        self.assertEqual(stats.source_game_id, 2025021312)
        self.assertEqual(stats.value["shots_for"], 34)
        self.assertIn("rest_hours", stats.value)
        self.assertIn("travel_km_since_previous_game", stats.value)

    def test_goalie_states_remain_observed_and_unknown_is_not_imputed(self) -> None:
        observations = self.fixture("pregame_goalies_valid.json").observations
        by_id = {item.observation_id: item for item in observations}
        self.assertEqual(
            by_id["obs.goalie.away.probable"].value["status"],
            "PROBABLE",
        )
        self.assertEqual(
            by_id["obs.goalie.home.confirmed.early"].value["status"],
            "CONFIRMED",
        )
        unknown = by_id["obs.goalie.unknown"]
        self.assertEqual(unknown.value_state, ValueState.UNKNOWN)
        self.assertIsNone(unknown.value)

    def test_effective_availability_uses_later_source_timestamp(self) -> None:
        observations = self.fixture("pregame_goalies_valid.json").observations
        late = next(item for item in observations if item.observation_id.endswith("late"))
        self.assertEqual(late.effective_available_at_utc, utc("2026-10-01T22:45:00Z"))
        self.assertEqual(
            assess_observation_for_cutoff(
                late,
                information_cutoff_utc=utc("2026-10-01T22:00:00Z"),
                scheduled_start_utc=utc("2026-10-01T23:00:00Z"),
            ),
            TemporalStatus.AFTER_CUTOFF,
        )
        with self.assertRaises(NHLOfflineSourceError):
            validate_observation_for_cutoff(
                late,
                information_cutoff_utc=utc("2026-10-01T22:00:00Z"),
                scheduled_start_utc=utc("2026-10-01T23:00:00Z"),
            )

    def test_early_goalie_and_injury_are_eligible_but_late_updates_are_not(self) -> None:
        goalie = self.fixture("pregame_goalies_valid.json").observations
        injury = self.fixture("player_availability_valid.json").observations
        cutoff = utc("2026-10-01T22:00:00Z")
        start = utc("2026-10-01T23:00:00Z")
        selected_goalies = eligible_observations(
            tuple(item for item in goalie if not item.observation_id.endswith("late")),
            target_game_id=2026020001,
            information_cutoff_utc=cutoff,
            scheduled_start_utc=start,
        )
        self.assertEqual(len(selected_goalies), 2)
        self.assertEqual(
            assess_observation_for_cutoff(
                injury[0],
                information_cutoff_utc=cutoff,
                scheduled_start_utc=start,
            ),
            TemporalStatus.ELIGIBLE,
        )
        self.assertEqual(
            assess_observation_for_cutoff(
                injury[1],
                information_cutoff_utc=cutoff,
                scheduled_start_utc=start,
            ),
            TemporalStatus.AFTER_CUTOFF,
        )

    def test_fixture_from_disabled_provider_can_be_read_but_not_called(self) -> None:
        bundle = self.fixture("public_schedule_valid.json")
        with self.assertRaises(ValueError):
            require_enabled_provider(bundle.proof.provider)

    def test_bad_hash_is_rejected(self) -> None:
        path = self.write_variant(
            "public_schedule_valid.json",
            lambda document: document["payload"]["records"][0]["value"].update(
                {"home_team_id": 99}
            ),
        )
        with self.assertRaisesRegex(NHLOfflineSourceError, "empreinte"):
            load_offline_fixture(path)

    def test_unknown_provider_is_rejected(self) -> None:
        path = self.write_variant(
            "public_schedule_valid.json",
            lambda document: (
                document.__setitem__("provider", "unknown_provider"),
                document["request"].__setitem__("provider", "unknown_provider"),
            ),
        )
        with self.assertRaisesRegex(NHLOfflineSourceError, "invalide"):
            load_offline_fixture(path)

    def test_non_utc_timestamp_is_rejected(self) -> None:
        path = self.write_variant(
            "public_schedule_valid.json",
            lambda document: document.__setitem__(
                "observed_at_utc", "2026-09-30T14:00:00+02:00"
            ),
        )
        with self.assertRaisesRegex(NHLOfflineSourceError, "UTC"):
            load_offline_fixture(path)

    def test_target_game_as_source_game_is_rejected(self) -> None:
        path = self.write_variant(
            "stats_team_history_valid.json",
            lambda document: document["payload"]["records"][0].__setitem__(
                "source_game_id", 2026020001
            ),
            rehash=True,
        )
        with self.assertRaisesRegex(NHLOfflineSourceError, "propre match source"):
            load_offline_fixture(path)

    def test_unknown_value_cannot_silently_become_known(self) -> None:
        path = self.write_variant(
            "pregame_goalies_valid.json",
            lambda document: document["payload"]["records"][3].__setitem__(
                "value", {"goalie_id": 9999, "status": "CONFIRMED", "team_id": 30}
            ),
            rehash=True,
        )
        with self.assertRaisesRegex(NHLOfflineSourceError, "UNKNOWN"):
            load_offline_fixture(path)

    def test_target_play_by_play_is_rejected(self) -> None:
        path = self.write_variant(
            "public_schedule_valid.json",
            lambda document: document["payload"]["records"][0]["value"].update(
                {"play_by_play": []}
            ),
            rehash=True,
        )
        with self.assertRaisesRegex(NHLOfflineSourceError, "play-by-play"):
            load_offline_fixture(path)

    def test_cutoff_must_be_strictly_before_start(self) -> None:
        observation = self.fixture("public_schedule_valid.json").observations[0]
        with self.assertRaisesRegex(NHLOfflineSourceError, "précéder"):
            assess_observation_for_cutoff(
                observation,
                information_cutoff_utc=utc("2026-10-01T23:00:00Z"),
                scheduled_start_utc=utc("2026-10-01T23:00:00Z"),
            )

    def test_marker_is_required(self) -> None:
        path = self.write_variant(
            "public_schedule_valid.json",
            lambda document: document.__setitem__("fixture_marker", "REAL_DATA"),
        )
        with self.assertRaisesRegex(NHLOfflineSourceError, "synthétique"):
            load_offline_fixture(path)
        self.assertEqual(SYNTHETIC_FIXTURE_MARKER, "SYNTHETIC_OFFLINE_FIXTURE")


if __name__ == "__main__":
    unittest.main()
