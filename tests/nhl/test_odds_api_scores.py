"""Offline NHL score validation and pregame-event reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import unittest

from src.nhl.odds_api_candidates import parse_nhl_events
from src.nhl.odds_api_scores import (
    NHLScoreEvidenceError,
    parse_nhl_final_scores,
    reconcile_pregame_events,
)


EVENT_ID = "a" * 32
OBSERVED = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
CAPTURED = datetime(2026, 10, 8, 20, tzinfo=timezone.utc)


def _row() -> dict:
    return {
        "id": EVENT_ID, "sport_key": "icehockey_nhl",
        "commence_time": "2026-10-08T23:00:00Z",
        "away_team": "Away", "home_team": "Home", "completed": True,
        "scores": [{"name": "Home", "score": "3"}, {"name": "Away", "score": "2"}],
        "last_update": "2026-10-09T02:00:00Z",
    }


def _raw(rows: list[dict]) -> bytes:
    return json.dumps(rows).encode("utf-8")


class NHLScoreEvidenceTests(unittest.TestCase):
    def test_final_and_exact_pregame_match(self):
        row = _row()
        finals = parse_nhl_final_scores(_raw([row]), OBSERVED)
        self.assertEqual(len(finals), 1)
        self.assertEqual((finals[0].away_score, finals[0].home_score, finals[0].winner_name), (2, 3, "Home"))
        pregame = parse_nhl_events(_raw([row]))
        self.assertEqual(reconcile_pregame_events(pregame, CAPTURED, finals), finals)

    def test_live_score_never_becomes_final(self):
        row = _row()
        row["completed"] = False
        row["scores"][0]["score"] = "7"
        self.assertEqual(parse_nhl_final_scores(_raw([row]), OBSERVED), ())

    def test_score_team_mismatch_rejected(self):
        row = _row()
        row["scores"][1]["name"] = "Other"
        with self.assertRaisesRegex(NHLScoreEvidenceError, "différentes"):
            parse_nhl_final_scores(_raw([row]), OBSERVED)

    def test_tied_final_rejected(self):
        row = _row()
        row["scores"][1]["score"] = "3"
        with self.assertRaisesRegex(NHLScoreEvidenceError, "indéterminé"):
            parse_nhl_final_scores(_raw([row]), OBSERVED)

    def test_future_score_update_rejected(self):
        row = _row()
        row["last_update"] = "2026-10-10T00:00:00Z"
        with self.assertRaisesRegex(NHLScoreEvidenceError, "incohérent"):
            parse_nhl_final_scores(_raw([row]), OBSERVED)

    def test_duplicate_event_rejected(self):
        row = _row()
        with self.assertRaises(NHLScoreEvidenceError):
            parse_nhl_final_scores(_raw([row, row]), OBSERVED)

    def test_changed_start_rejected_instead_of_silent_postponement_match(self):
        row = _row()
        finals = parse_nhl_final_scores(_raw([row]), OBSERVED)
        row["commence_time"] = "2026-10-08T21:00:00Z"
        pregame = parse_nhl_events(_raw([row]))
        with self.assertRaisesRegex(NHLScoreEvidenceError, "divergent"):
            reconcile_pregame_events(pregame, CAPTURED, finals)

    def test_after_cutoff_capture_rejected(self):
        row = _row()
        finals = parse_nhl_final_scores(_raw([row]), OBSERVED)
        pregame = parse_nhl_events(_raw([row]))
        late = datetime(2026, 10, 8, 22, 1, tzinfo=timezone.utc)
        with self.assertRaisesRegex(NHLScoreEvidenceError, "limite"):
            reconcile_pregame_events(pregame, late, finals)

    def test_unknown_final_is_not_imputed_to_other_match(self):
        row = _row()
        finals = parse_nhl_final_scores(_raw([row]), OBSERVED)
        row["id"] = "b" * 32
        pregame = parse_nhl_events(_raw([row]))
        self.assertEqual(reconcile_pregame_events(pregame, CAPTURED, finals), ())


if __name__ == "__main__":
    unittest.main()
