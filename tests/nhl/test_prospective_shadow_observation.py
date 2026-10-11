"""Safety checks for private, pregame-only NHL research observations."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.nhl.current_season_import import CurrentSeasonImport
from src.nhl.moneypuck_team_import import TeamGameRow
from src.nhl.prospective_shadow_observation import (
    NHLProspectiveShadowError, POLICY_PATH, _current_features,
    build_prospective_shadow_observation,
)
from src.nhl.exploratory_chronological_backtest import POLICY_PATH as RESEARCH_POLICY

UTC = timezone.utc
GAME_ID = 2026020099
START = datetime(2026, 10, 10, 20, tzinfo=UTC)
NOW = datetime(2026, 10, 9, 12, tzinfo=UTC)


def _row(day: int, team: str, side: str, situation: str) -> TeamGameRow:
    goals = day + (1 if side == "HOME" else 0)
    return TeamGameRow(
        game_id=2026020000 + day, season=2026,
        game_date=date(2026, 10, day), team=team,
        opponent="BOS" if team == "NYR" else "NYR",
        home_or_away=side, situation=situation,
        x_goals_for=Decimal(str(goals)) / 2,
        x_goals_against=Decimal("1.5"),
        goals_for=goals, goals_against=1,
        shots_on_goal_for=Decimal("30"), shots_on_goal_against=Decimal("25"),
    )


def _imported(days: int = 5) -> CurrentSeasonImport:
    rows = tuple(
        _row(day, team, side, situation)
        for day in range(1, days + 1)
        for team, side in (("NYR", "AWAY"), ("BOS", "HOME"))
        for situation in ("all", "5on5")
    )
    received = datetime(2026, 10, 8, 12, tzinfo=UTC)
    return CurrentSeasonImport(
        Path("/private/import"), 2026, "capture", "a" * 64,
        received, received, received, rows, days,
    )


def _schedule(observed_at: str = "2026-10-08T11:00:00Z") -> dict:
    return {
        "observed_at_utc": observed_at,
        "response_sha256": "b" * 64,
        "games": [{
            "game_id": GAME_ID, "season": 20262027,
            "scheduled_start_utc": "2026-10-10T20:00:00Z",
            "away_abbr": "NYR", "home_abbr": "BOS",
        }],
    }


class _StubModel:
    def predict_proba(self, features):
        assert features.shape == (1, 6)
        return np.asarray([[0.3, 0.7]])


class ProspectiveShadowTests(unittest.TestCase):
    def run_shadow(self, *, imported=None, schedule=None, now=NOW, explicit=True):
        with (
            patch("src.nhl.prospective_shadow_observation.audit_schedule_capture",
                  return_value=_schedule() if schedule is None else schedule),
            patch("src.nhl.prospective_shadow_observation.verify_current_season_import",
                  return_value=_imported() if imported is None else imported),
            patch("src.nhl.prospective_shadow_observation._fit_verified_research_model",
                  return_value=(_StubModel(), "c" * 64, "d" * 64, 200)) as fitted,
        ):
            result = build_prospective_shadow_observation(
                Path("/sealed/schedule"), Path("/sealed/import"),
                target_game_id=GAME_ID, explicit_shadow_run=explicit,
                now=lambda: now,
            )
            return result, fitted.call_count

    def test_shadow_probability_is_private_and_provenance_bound(self):
        result, fit_count = self.run_shadow()
        self.assertEqual(fit_count, 1)
        self.assertEqual(result.home_win_probability, 0.7)
        self.assertAlmostEqual(result.away_win_probability, 0.3)
        self.assertEqual(result.information_cutoff_utc, "2026-10-10T19:00:00.000000Z")
        self.assertEqual(result.status, "PRIVATE_RESEARCH_SHADOW_ONLY")
        self.assertEqual(len(result.observation_sha256), 64)
        self.assertFalse(result.production_training_permitted)
        self.assertFalse(result.prediction_publication_permitted)
        self.assertFalse(result.betting_recommendation_permitted)
        altered = replace(_imported(), source_response_sha256="e" * 64)
        changed, _ = self.run_shadow(imported=altered)
        self.assertNotEqual(result.feature_sha256, changed.feature_sha256)
        self.assertNotEqual(result.observation_sha256, changed.observation_sha256)

    def test_no_implicit_run_or_late_prediction(self):
        with self.assertRaisesRegex(NHLProspectiveShadowError, "explicite"):
            self.run_shadow(explicit=False)
        with self.assertRaisesRegex(NHLProspectiveShadowError, "limite pré-match"):
            self.run_shadow(now=datetime(2026, 10, 10, 19, 0, 1, tzinfo=UTC))
        with self.assertRaisesRegex(NHLProspectiveShadowError, "limite pré-match"):
            self.run_shadow(schedule=_schedule("2026-10-10T19:00:01Z"))

    def test_late_import_and_missing_five_games_fail_closed(self):
        late = replace(_imported(),
                       imported_at_utc=datetime(2026, 10, 10, 19, 1, tzinfo=UTC),
                       effective_available_at_utc=datetime(2026, 10, 10, 19, 1, tzinfo=UTC))
        with self.assertRaisesRegex(NHLProspectiveShadowError, "Statistiques reçues"):
            self.run_shadow(imported=late)
        with self.assertRaisesRegex(NHLProspectiveShadowError, "Cinq matchs"):
            self.run_shadow(imported=_imported(days=4))

    def test_training_feature_parity_and_target_exclusion(self):
        from src.nhl.public_schedule_candidates import ScheduledGame
        game = ScheduledGame(GAME_ID, 20262027, START, "NYR", "BOS")
        vector, away_ids, home_ids = _current_features(_imported(), game)
        self.assertEqual(vector, (1.0, 0.0, 0.5, 0.0, 0.5, 0.0))
        self.assertEqual(away_ids, (2026020005, 2026020004, 2026020003,
                                    2026020002, 2026020001))
        self.assertEqual(home_ids, away_ids)
        leaked = replace(_imported(), regular_rows=_imported().regular_rows +
                         (replace(_imported().regular_rows[0], game_id=GAME_ID),))
        with self.assertRaisesRegex(NHLProspectiveShadowError, "match cible"):
            _current_features(leaked, game)

    def test_shadow_policy_cannot_enable_subscriber_publication(self):
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        old = json.loads(RESEARCH_POLICY.read_text(encoding="utf-8"))
        self.assertFalse(policy["subscriber_publication_permitted"])
        self.assertFalse(old["per_game_predictions_allowed"])
        self.assertFalse(old["prediction_publication_permitted"])
        policy["subscriber_publication_permitted"] = True
        with TemporaryDirectory() as folder:
            bad = Path(folder) / "unsafe.json"
            bad.write_text(json.dumps(policy), encoding="utf-8")
            with patch("src.nhl.prospective_shadow_observation.POLICY_PATH", bad):
                with self.assertRaisesRegex(NHLProspectiveShadowError, "incompatible"):
                    self.run_shadow()


if __name__ == "__main__":
    unittest.main()
