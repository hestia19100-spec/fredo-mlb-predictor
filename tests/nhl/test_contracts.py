from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import unittest

from src.nhl.contracts import (
    FeatureKind,
    FeatureProvenance,
    GoalieObservation,
    GoalieStatus,
    NHLContractError,
    PlayerAvailabilityObservation,
    PlayerAvailabilityStatus,
    ScheduledGameObservation,
    SourceEvidence,
    SourceGameResult,
)


UTC = timezone.utc
COMMIT = "8" * 40
SHA256 = "a" * 64


def evidence(
    *,
    observed_at: datetime | None = None,
    updated_at: datetime | None = None,
    observation_id: str = "nhl-web:game:1",
) -> SourceEvidence:
    return SourceEvidence(
        provider="nhl-web",
        observation_id=observation_id,
        observed_at_utc=observed_at
        or datetime(2026, 10, 1, 18, 0, tzinfo=UTC),
        response_sha256=SHA256,
        code_commit=COMMIT,
        source_updated_at_utc=updated_at,
    )


class SourceEvidenceTests(unittest.TestCase):
    def test_accepts_canonical_evidence(self) -> None:
        value = evidence()
        self.assertEqual(value.provider, "nhl-web")
        self.assertEqual(value.effective_available_at_utc, value.observed_at_utc)

    def test_effective_availability_uses_later_provider_update(self) -> None:
        updated = datetime(2026, 10, 1, 18, 5, tzinfo=UTC)
        value = evidence(updated_at=updated)
        self.assertEqual(value.effective_available_at_utc, updated)

    def test_rejects_noncanonical_provider(self) -> None:
        with self.assertRaises(NHLContractError):
            SourceEvidence(
                provider="NHL Web",
                observation_id="nhl-web:game:1",
                observed_at_utc=datetime(2026, 10, 1, 18, 0, tzinfo=UTC),
                response_sha256=SHA256,
                code_commit=COMMIT,
            )

    def test_rejects_naive_observation_time(self) -> None:
        with self.assertRaises(NHLContractError):
            SourceEvidence(
                provider="nhl-web",
                observation_id="nhl-web:game:1",
                observed_at_utc=datetime(2026, 10, 1, 18, 0),
                response_sha256=SHA256,
                code_commit=COMMIT,
            )

    def test_rejects_non_utc_observation_time(self) -> None:
        with self.assertRaises(NHLContractError):
            SourceEvidence(
                provider="nhl-web",
                observation_id="nhl-web:game:1",
                observed_at_utc=datetime(
                    2026,
                    10,
                    1,
                    20,
                    0,
                    tzinfo=timezone(timedelta(hours=2)),
                ),
                response_sha256=SHA256,
                code_commit=COMMIT,
            )

    def test_rejects_invalid_sha256(self) -> None:
        with self.assertRaises(NHLContractError):
            SourceEvidence(
                provider="nhl-web",
                observation_id="nhl-web:game:1",
                observed_at_utc=datetime(2026, 10, 1, 18, 0, tzinfo=UTC),
                response_sha256="A" * 64,
                code_commit=COMMIT,
            )

    def test_rejects_invalid_git_commit(self) -> None:
        with self.assertRaises(NHLContractError):
            SourceEvidence(
                provider="nhl-web",
                observation_id="nhl-web:game:1",
                observed_at_utc=datetime(2026, 10, 1, 18, 0, tzinfo=UTC),
                response_sha256=SHA256,
                code_commit="8" * 39,
            )


class DomainContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_evidence = evidence()

    def test_game_requires_distinct_teams(self) -> None:
        with self.assertRaises(NHLContractError):
            ScheduledGameObservation(
                game_id=2026020001,
                season_id=20262027,
                official_date=date(2026, 10, 1),
                away_team_id=10,
                home_team_id=10,
                scheduled_start_utc=datetime(2026, 10, 2, 0, 0, tzinfo=UTC),
                evidence=self.source_evidence,
            )

    def test_game_rejects_datetime_as_official_date(self) -> None:
        with self.assertRaises(NHLContractError):
            ScheduledGameObservation(
                game_id=2026020001,
                season_id=20262027,
                official_date=datetime(2026, 10, 1, tzinfo=UTC),  # type: ignore[arg-type]
                away_team_id=10,
                home_team_id=20,
                scheduled_start_utc=datetime(2026, 10, 2, 0, 0, tzinfo=UTC),
                evidence=self.source_evidence,
            )

    def test_source_result_cannot_be_final_before_start(self) -> None:
        with self.assertRaises(NHLContractError):
            SourceGameResult(
                game_id=2026020000,
                scheduled_start_utc=datetime(2026, 10, 1, 1, 0, tzinfo=UTC),
                final_observed_at_utc=datetime(
                    2026,
                    10,
                    1,
                    0,
                    59,
                    tzinfo=UTC,
                ),
            )

    def test_feature_requires_evidence(self) -> None:
        with self.assertRaises(NHLContractError):
            FeatureProvenance(
                target_game_id=2026020001,
                feature_kind=FeatureKind.TEAM_FORM,
                evidence=(),
            )

    def test_feature_rejects_mutable_evidence_collection(self) -> None:
        with self.assertRaises(NHLContractError):
            FeatureProvenance(
                target_game_id=2026020001,
                feature_kind=FeatureKind.TEAM_FORM,
                evidence=[self.source_evidence],  # type: ignore[arg-type]
            )

    def test_feature_rejects_duplicate_evidence(self) -> None:
        with self.assertRaises(NHLContractError):
            FeatureProvenance(
                target_game_id=2026020001,
                feature_kind=FeatureKind.TEAM_FORM,
                evidence=(self.source_evidence, self.source_evidence),
            )

    def test_feature_rejects_duplicate_source_game(self) -> None:
        result = SourceGameResult(
            game_id=2026020000,
            scheduled_start_utc=datetime(2026, 10, 1, 0, 0, tzinfo=UTC),
            final_observed_at_utc=datetime(2026, 10, 1, 3, 0, tzinfo=UTC),
        )
        with self.assertRaises(NHLContractError):
            FeatureProvenance(
                target_game_id=2026020001,
                feature_kind=FeatureKind.TEAM_FORM,
                evidence=(self.source_evidence,),
                source_games=(result, result),
            )

    def test_unknown_goalie_must_not_have_identity(self) -> None:
        with self.assertRaises(NHLContractError):
            GoalieObservation(
                game_id=2026020001,
                team_id=10,
                status=GoalieStatus.UNKNOWN,
                evidence=self.source_evidence,
                goalie_id=8470001,
            )

    def test_probable_goalie_requires_identity(self) -> None:
        with self.assertRaises(NHLContractError):
            GoalieObservation(
                game_id=2026020001,
                team_id=10,
                status=GoalieStatus.PROBABLE,
                evidence=self.source_evidence,
            )

    def test_unknown_goalie_without_identity_is_valid(self) -> None:
        value = GoalieObservation(
            game_id=2026020001,
            team_id=10,
            status=GoalieStatus.UNKNOWN,
            evidence=self.source_evidence,
        )
        self.assertIsNone(value.goalie_id)

    def test_player_availability_requires_positive_player_id(self) -> None:
        with self.assertRaises(NHLContractError):
            PlayerAvailabilityObservation(
                game_id=2026020001,
                team_id=10,
                player_id=0,
                status=PlayerAvailabilityStatus.OUT,
                evidence=self.source_evidence,
            )


if __name__ == "__main__":
    unittest.main()
