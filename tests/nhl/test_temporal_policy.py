from __future__ import annotations

from datetime import date, datetime, timezone
import unittest

from src.nhl.contracts import (
    FeatureKind,
    FeatureProvenance,
    GoalieObservation,
    GoalieStatus,
    PlayerAvailabilityObservation,
    PlayerAvailabilityStatus,
    ScheduledGameObservation,
    SourceEvidence,
    SourceGameResult,
)
from src.nhl.temporal_policy import (
    NHLTemporalLeakageError,
    build_information_cutoff,
    validate_feature_provenance,
    validate_goalie_observation,
    validate_information_cutoff,
    validate_player_availability_observation,
)


UTC = timezone.utc
COMMIT = "8" * 40
SHA256 = "a" * 64


def evidence(
    observation_id: str,
    observed_at: datetime,
    *,
    updated_at: datetime | None = None,
) -> SourceEvidence:
    return SourceEvidence(
        provider="nhl-web",
        observation_id=observation_id,
        observed_at_utc=observed_at,
        response_sha256=SHA256,
        code_commit=COMMIT,
        source_updated_at_utc=updated_at,
    )


class TemporalPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.start = datetime(2026, 10, 2, 0, 0, tzinfo=UTC)
        self.cutoff = datetime(2026, 10, 1, 23, 0, tzinfo=UTC)
        self.schedule_evidence = evidence(
            "nhl-web:schedule:1",
            datetime(2026, 10, 1, 18, 0, tzinfo=UTC),
        )
        self.game = ScheduledGameObservation(
            game_id=2026020001,
            season_id=20262027,
            official_date=date(2026, 10, 1),
            away_team_id=10,
            home_team_id=20,
            scheduled_start_utc=self.start,
            evidence=self.schedule_evidence,
        )
        self.prior_result = SourceGameResult(
            game_id=2026020000,
            scheduled_start_utc=datetime(2026, 9, 30, 23, 0, tzinfo=UTC),
            final_observed_at_utc=datetime(2026, 10, 1, 2, 0, tzinfo=UTC),
        )

    def test_builds_deterministic_cutoff(self) -> None:
        self.assertEqual(
            build_information_cutoff(self.start, lead_minutes=60),
            self.cutoff,
        )

    def test_cutoff_lead_must_be_positive_integer(self) -> None:
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_information_cutoff(self.start, lead_minutes=value)  # type: ignore[arg-type]

    def test_accepts_cutoff_strictly_before_start(self) -> None:
        validate_information_cutoff(self.game, self.cutoff)

    def test_rejects_cutoff_at_start(self) -> None:
        with self.assertRaises(NHLTemporalLeakageError):
            validate_information_cutoff(self.game, self.start)

    def test_rejects_schedule_observed_after_cutoff(self) -> None:
        late_game = ScheduledGameObservation(
            game_id=self.game.game_id,
            season_id=self.game.season_id,
            official_date=self.game.official_date,
            away_team_id=self.game.away_team_id,
            home_team_id=self.game.home_team_id,
            scheduled_start_utc=self.start,
            evidence=evidence(
                "nhl-web:schedule:late",
                datetime(2026, 10, 1, 23, 1, tzinfo=UTC),
            ),
        )
        with self.assertRaises(NHLTemporalLeakageError):
            validate_information_cutoff(late_game, self.cutoff)

    def test_accepts_prior_results_final_before_cutoff(self) -> None:
        provenance = FeatureProvenance(
            target_game_id=self.game.game_id,
            feature_kind=FeatureKind.TEAM_FORM,
            evidence=(self.schedule_evidence,),
            source_games=(self.prior_result,),
        )
        validate_feature_provenance(
            game=self.game,
            provenance=provenance,
            information_cutoff_utc=self.cutoff,
        )

    def test_rejects_target_game_as_feature_source(self) -> None:
        target_result = SourceGameResult(
            game_id=self.game.game_id,
            scheduled_start_utc=datetime(2026, 10, 1, 20, 0, tzinfo=UTC),
            final_observed_at_utc=datetime(2026, 10, 1, 22, 30, tzinfo=UTC),
        )
        provenance = FeatureProvenance(
            target_game_id=self.game.game_id,
            feature_kind=FeatureKind.GOAL_FORM,
            evidence=(self.schedule_evidence,),
            source_games=(target_result,),
        )
        with self.assertRaises(NHLTemporalLeakageError):
            validate_feature_provenance(
                game=self.game,
                provenance=provenance,
                information_cutoff_utc=self.cutoff,
            )

    def test_rejects_result_final_after_cutoff(self) -> None:
        late_result = SourceGameResult(
            game_id=2026020000,
            scheduled_start_utc=datetime(2026, 10, 1, 21, 0, tzinfo=UTC),
            final_observed_at_utc=datetime(2026, 10, 1, 23, 1, tzinfo=UTC),
        )
        provenance = FeatureProvenance(
            target_game_id=self.game.game_id,
            feature_kind=FeatureKind.GOALIE_FORM,
            evidence=(self.schedule_evidence,),
            source_games=(late_result,),
        )
        with self.assertRaises(NHLTemporalLeakageError):
            validate_feature_provenance(
                game=self.game,
                provenance=provenance,
                information_cutoff_utc=self.cutoff,
            )

    def test_rejects_feature_evidence_observed_after_cutoff(self) -> None:
        late_evidence = evidence(
            "nhl-web:stats:late",
            datetime(2026, 10, 1, 23, 1, tzinfo=UTC),
        )
        provenance = FeatureProvenance(
            target_game_id=self.game.game_id,
            feature_kind=FeatureKind.POWER_PLAY,
            evidence=(late_evidence,),
        )
        with self.assertRaises(NHLTemporalLeakageError):
            validate_feature_provenance(
                game=self.game,
                provenance=provenance,
                information_cutoff_utc=self.cutoff,
            )

    def test_rejects_provider_update_after_cutoff(self) -> None:
        late_update = evidence(
            "sportsdataio:goalie:late",
            datetime(2026, 10, 1, 22, 30, tzinfo=UTC),
            updated_at=datetime(2026, 10, 1, 23, 1, tzinfo=UTC),
        )
        provenance = FeatureProvenance(
            target_game_id=self.game.game_id,
            feature_kind=FeatureKind.GOALIE_STATUS,
            evidence=(late_update,),
        )
        with self.assertRaises(NHLTemporalLeakageError):
            validate_feature_provenance(
                game=self.game,
                provenance=provenance,
                information_cutoff_utc=self.cutoff,
            )

    def test_rejects_provenance_for_another_target(self) -> None:
        provenance = FeatureProvenance(
            target_game_id=2026029999,
            feature_kind=FeatureKind.REST,
            evidence=(self.schedule_evidence,),
        )
        with self.assertRaises(NHLTemporalLeakageError):
            validate_feature_provenance(
                game=self.game,
                provenance=provenance,
                information_cutoff_utc=self.cutoff,
            )

    def test_accepts_probable_goalie_observed_before_cutoff(self) -> None:
        goalie = GoalieObservation(
            game_id=self.game.game_id,
            team_id=self.game.home_team_id,
            status=GoalieStatus.PROBABLE,
            goalie_id=8470001,
            evidence=evidence(
                "sportsdataio:goalie:1",
                datetime(2026, 10, 1, 22, 45, tzinfo=UTC),
            ),
        )
        validate_goalie_observation(
            game=self.game,
            goalie=goalie,
            information_cutoff_utc=self.cutoff,
        )

    def test_rejects_goalie_observed_after_cutoff(self) -> None:
        goalie = GoalieObservation(
            game_id=self.game.game_id,
            team_id=self.game.home_team_id,
            status=GoalieStatus.CONFIRMED,
            goalie_id=8470001,
            evidence=evidence(
                "sportsdataio:goalie:late",
                datetime(2026, 10, 1, 23, 1, tzinfo=UTC),
            ),
        )
        with self.assertRaises(NHLTemporalLeakageError):
            validate_goalie_observation(
                game=self.game,
                goalie=goalie,
                information_cutoff_utc=self.cutoff,
            )

    def test_rejects_goalie_from_other_team(self) -> None:
        goalie = GoalieObservation(
            game_id=self.game.game_id,
            team_id=999,
            status=GoalieStatus.PROBABLE,
            goalie_id=8470001,
            evidence=self.schedule_evidence,
        )
        with self.assertRaises(NHLTemporalLeakageError):
            validate_goalie_observation(
                game=self.game,
                goalie=goalie,
                information_cutoff_utc=self.cutoff,
            )

    def test_accepts_unknown_goalie_without_postgame_identity(self) -> None:
        goalie = GoalieObservation(
            game_id=self.game.game_id,
            team_id=self.game.away_team_id,
            status=GoalieStatus.UNKNOWN,
            evidence=self.schedule_evidence,
        )
        validate_goalie_observation(
            game=self.game,
            goalie=goalie,
            information_cutoff_utc=self.cutoff,
        )

    def test_rejects_late_player_availability(self) -> None:
        availability = PlayerAvailabilityObservation(
            game_id=self.game.game_id,
            team_id=self.game.away_team_id,
            player_id=8470002,
            status=PlayerAvailabilityStatus.OUT,
            evidence=evidence(
                "sportsdataio:injury:late",
                datetime(2026, 10, 1, 23, 2, tzinfo=UTC),
            ),
        )
        with self.assertRaises(NHLTemporalLeakageError):
            validate_player_availability_observation(
                game=self.game,
                availability=availability,
                information_cutoff_utc=self.cutoff,
            )


if __name__ == "__main__":
    unittest.main()
