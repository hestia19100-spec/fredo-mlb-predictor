"""Preuves hors ligne de la neutralisation complémentaire des reports."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlencode

from src.lpf_edge_dashboard import (
    SCORING_ROOT, CertifiedPrediction, CertifiedPredictionDay,
    DailyScoreResult, DailyScoreSummary,
)
from src.lpf_edge_postponement_reconciliation import (
    API_URL, PostponementReconciliationError,
    create_reconciliation, load_resolutions,
)


TARGET = date(2026, 9, 22)
CHECKPOINT = date(2026, 9, 27)


def prediction_day() -> CertifiedPredictionDay:
    prediction = CertifiedPrediction(
        prediction_id="p" * 64, batch_id="b" * 64, game_id=824785,
        official_date=TARGET, away_team_id=141, home_team_id=110,
        scheduled_start_utc=datetime(2026, 9, 22, 23, tzinfo=timezone.utc),
        issued_at_utc=datetime(2026, 9, 22, 13, tzinfo=timezone.utc),
        p_home_win=Decimal("0.6"), p_away_win=Decimal("0.4"),
    )
    return CertifiedPredictionDay(
        target_date=TARGET, batch_id="b" * 64,
        results_commit="c" * 40,
        certified_at_utc=datetime(2026, 9, 22, 14, tzinfo=timezone.utc),
        remote_lead_minutes=Decimal("540"),
        predictions_sha256="a" * 64, receipt_sha256="f" * 64,
        predictions=(prediction,),
    )


def score_summary() -> DailyScoreSummary:
    result = DailyScoreResult(
        prediction_id="p" * 64, game_id=824785,
        away_team_id=141, home_team_id=110, predicted_side="HOME",
        predicted_probability=Decimal("0.6"),
        outcome_status="PENDING_POSTPONED", away_score=None,
        home_score=None, actual_winner=None, classification_correct=None,
    )
    return DailyScoreSummary(
        checkpoint_date=CHECKPOINT, scored_count=0, void_count=0,
        pending_count=1, correct_count=0, incorrect_count=0,
        accuracy=None, mean_log_loss=None, mean_brier_score=None,
        results=(result,),
    )


def response_for(*, official_date="2026-09-23", game_id=824785,
                 away_id=141, home_id=110, away_score=2, home_score=4):
    game = {
        "gamePk": game_id, "season": "2026", "gameType": "R",
        "officialDate": official_date,
        "status": {"statusCode": "F", "detailedState": "Final"},
        "teams": {
            "away": {"team": {"id": away_id}, "score": away_score},
            "home": {"team": {"id": home_id}, "score": home_score},
        },
    }
    url = API_URL + "?" + urlencode({
        "sportId": 1, "startDate": TARGET.isoformat(),
        "endDate": CHECKPOINT.isoformat(), "gameTypes": "R",
    })
    return SimpleNamespace(
        status_code=200, url=url,
        headers={"Date": "Sun, 27 Sep 2026 07:56:09 GMT"},
        content=json.dumps({"dates": [{"games": [game]}]}).encode(),
    )


class PostponementReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        slot = self.project / SCORING_ROOT / TARGET.isoformat() / "observations" / CHECKPOINT.isoformat()
        slot.mkdir(parents=True)
        (slot / "daily_report.json").write_bytes(b"original report\n")
        (slot / "adjudications.csv").write_text(
            "prediction_id,game_id,outcome_status\n"
            + f"{'p' * 64},824785,PENDING_POSTPONED\n",
            encoding="utf-8",
        )
        self.day = prediction_day()
        self.score = score_summary()

    def create(self, response=None):
        return create_reconciliation(
            self.day, self.score, project_directory=self.project,
            checkpoint_date=CHECKPOINT,
            response=response if response is not None else response_for(),
        )

    def test_final_other_date_is_audited_and_neutralized_only_in_complement(self):
        publication = self.create()
        self.assertEqual(3, len(publication.paths))
        self.assertEqual((824785,), tuple(x.game_id for x in publication.resolved))
        loaded = load_resolutions(self.day, self.score, project_directory=self.project)
        self.assertEqual(date(2026, 9, 23), loaded[824785].final_official_date)
        self.assertEqual((2, 4), (loaded[824785].away_score, loaded[824785].home_score))
        self.assertEqual("PENDING_POSTPONED", self.score.results[0].outcome_status)
        with self.assertRaises(PostponementReconciliationError):
            self.create()

    def test_wrong_team_consumes_slot_without_publishing_evidence(self):
        with self.assertRaises(PostponementReconciliationError):
            self.create(response_for(home_id=111))
        slot = self.project / "outcome_reconciliations" / "lpf_edge_mlb_postponements_v1" / TARGET.isoformat() / CHECKPOINT.isoformat()
        self.assertTrue(slot.is_dir())
        self.assertEqual([], list(slot.iterdir()))
        with patch("src.lpf_edge_postponement_reconciliation.urlopen") as network:
            with self.assertRaises(PostponementReconciliationError):
                create_reconciliation(
                    self.day, self.score, project_directory=self.project,
                    checkpoint_date=CHECKPOINT,
                )
            network.assert_not_called()

    def test_same_official_date_consumes_slot_without_publishing_evidence(self):
        with self.assertRaises(PostponementReconciliationError):
            self.create(response_for(official_date=TARGET.isoformat()))
        self.assertEqual({}, load_resolutions(self.day, self.score,
                                               project_directory=self.project))

    def test_other_game_id_remains_pending(self):
        publication = self.create(response_for(game_id=824784))
        self.assertEqual((), publication.resolved)
        self.assertEqual({}, load_resolutions(self.day, self.score,
                                               project_directory=self.project))

    def test_conflicting_final_is_rejected_before_publishing(self):
        response = response_for()
        payload = json.loads(response.content)
        conflict = json.loads(json.dumps(payload["dates"][0]["games"][0]))
        conflict["teams"]["home"]["score"] = 5
        payload["dates"][0]["games"].append(conflict)
        response.content = json.dumps(payload).encode()
        with self.assertRaises(PostponementReconciliationError):
            self.create(response)
        self.assertEqual({}, load_resolutions(self.day, self.score,
                                               project_directory=self.project))

    def test_redirect_is_rejected_before_publishing(self):
        response = response_for()
        response.url += "&unexpected=true"
        with self.assertRaises(PostponementReconciliationError):
            self.create(response)
        self.assertEqual({}, load_resolutions(self.day, self.score,
                                               project_directory=self.project))

    def test_corrupt_raw_archive_fails_closed(self):
        publication = self.create()
        publication.paths[0].write_bytes(b"not a gzip")
        with self.assertRaises(PostponementReconciliationError):
            load_resolutions(self.day, self.score, project_directory=self.project)

    def test_official_terminal_result_takes_priority(self):
        self.create()
        terminal = replace(self.score, results=(replace(
            self.score.results[0], outcome_status="VOID_POSTPONED_AT_DEADLINE",
        ),), pending_count=0, void_count=1)
        self.assertEqual({}, load_resolutions(self.day, terminal,
                                               project_directory=self.project))


if __name__ == "__main__":
    unittest.main()
