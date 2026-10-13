"""Tests de préparation MLB 2027 : référence figée ou abstention."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import unittest

from src.mlb_2027_reference_readiness import (
    MLB2027ReadinessError,
    REFERENCE_ARTIFACT_SHA256,
    REFERENCE_MODEL_VERSION,
    PriorFinalGame,
    UpcomingGame,
    plan_reference_readiness,
)
from src.mlb_2027_timing import MLB2027TimingError

DAY = date(2027, 4, 15)
START = datetime(2027, 4, 16, 0, 0, tzinfo=timezone.utc)
RECEIPT = "a" * 64
ROOT = Path(__file__).resolve().parents[1]


def finals(team_id: int, count: int, id_offset: int) -> tuple[PriorFinalGame, ...]:
    return tuple(
        PriorFinalGame(
            game_id=id_offset + index,
            official_date=date(2027, 4, 1) + timedelta(days=index),
            away_team_id=team_id,
            home_team_id=id_offset + 100 + index,
            away_score=4,
            home_score=2,
            status="FINAL",
        )
        for index in range(count)
    )


def target() -> UpcomingGame:
    return UpcomingGame(
        game_id=9000,
        official_date=DAY,
        away_team_id=1,
        home_team_id=2,
        scheduled_start_utc=START,
        status="SCHEDULED",
    )


def plan(
    history: tuple[PriorFinalGame, ...],
    games: tuple[UpcomingGame, ...] | None = None,
):
    return plan_reference_readiness(
        DAY,
        (target(),) if games is None else games,
        history,
        source_receipt_sha256=RECEIPT,
    )


class MLB2027ReferenceReadinessTests(unittest.TestCase):
    def test_early_season_abstains_without_a_model_substitution(self) -> None:
        result = plan(())
        self.assertEqual(result.eligible_count, 0)
        self.assertEqual(result.games[0].status, "ABSTAIN")
        self.assertEqual(result.games[0].reason, "INSUFFICIENT_BOTH_HISTORY")
        self.assertIsNone(result.games[0].model_version)
        self.assertEqual(result.lead_window.latest_completion_utc,
                         START - timedelta(hours=1))

    def test_nine_games_for_one_team_still_abstains(self) -> None:
        result = plan(finals(1, 10, 1000) + finals(2, 9, 2000))
        self.assertEqual(result.games[0].reason, "INSUFFICIENT_HOME_HISTORY")
        self.assertEqual(result.games[0].status, "ABSTAIN")
        self.assertEqual(result.games[0].away_same_season_finals, 10)
        self.assertEqual(result.games[0].home_same_season_finals, 9)

    def test_ten_games_for_each_team_uses_the_frozen_reference(self) -> None:
        result = plan(finals(1, 10, 1000) + finals(2, 10, 2000))
        self.assertEqual(result.eligible_count, 1)
        self.assertEqual(result.games[0].status, "REFERENCE_ELIGIBLE")
        self.assertEqual(result.games[0].model_version, REFERENCE_MODEL_VERSION)
        self.assertEqual(result.games[0].artifact_sha256, REFERENCE_ARTIFACT_SHA256)
        self.assertEqual(result.source_receipt_sha256, RECEIPT)

    def test_future_or_previous_season_game_is_not_history(self) -> None:
        base = finals(1, 1, 1000)[0]
        for official_date in (DAY, date(2026, 9, 30)):
            with self.subTest(official_date=official_date):
                with self.assertRaises(MLB2027ReadinessError):
                    plan((replace(base, official_date=official_date),))

    def test_duplicate_game_or_unfinished_game_fails_closed(self) -> None:
        prior = finals(1, 1, 1000)[0]
        with self.assertRaises(MLB2027ReadinessError):
            plan((prior, prior))
        with self.assertRaises(MLB2027ReadinessError):
            plan((), (replace(target(), status="FINAL"),))
        with self.assertRaises(MLB2027ReadinessError):
            plan((), (target(), target()))

    def test_unverified_clock_or_receipt_fails_closed(self) -> None:
        with self.assertRaises(MLB2027TimingError):
            plan((), (replace(target(), scheduled_start_utc=START.replace(tzinfo=None)),))
        with self.assertRaises(MLB2027ReadinessError):
            plan_reference_readiness(DAY, (target(),), (), source_receipt_sha256="bad")

    def test_unexpected_input_type_is_a_controlled_error(self) -> None:
        with self.assertRaises(MLB2027ReadinessError):
            plan((), (object(),))

    def test_2026_protocol_and_model_artifact_are_not_rewritten(self) -> None:
        protocol = json.loads((
            ROOT / "shadow_protocols" /
            "logistic_team_form_v1_platt_shadow_v2.json"
        ).read_text(encoding="utf-8"))
        manifest = json.loads((
            ROOT / "model_artifacts" / "logistic_team_form_v1_platt.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(protocol["operating_mode"]["target_season_exact"], 2026)
        self.assertEqual(
            protocol["operating_mode"]["minimum_local_completion_lead_minutes"], 120
        )
        self.assertEqual(manifest["artifact"]["sha256"], REFERENCE_ARTIFACT_SHA256)


if __name__ == "__main__":
    unittest.main()
