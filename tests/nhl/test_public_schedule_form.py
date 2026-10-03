"""Offline regression tests for score-only, date-lagged NHL form."""
from __future__ import annotations

import csv
import io
import json
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.nhl.public_schedule_form import (
    PublicScheduleFormError, build_public_schedule_form,
)
from src.nhl.public_schedule_reconciliation import reconcile_public_schedules

ROOT = Path(__file__).resolve().parents[2] / "nhl_protocols" / "data"
REPORT = (ROOT / "nhl15_real_archive_report_2026-10-03.json").read_bytes()
OBSERVED = datetime(2026, 10, 3, tzinfo=timezone.utc)
COLUMNS = ["game_id", "season", "game_type", "game_date", "game_state",
           "home_team_abbr", "away_team_abbr", "home_score", "away_score"]


def synthetic_rows():
    rows = {}
    for season in range(2022, 2027):
        start = date(season - 1, 10, 1)
        rows[season] = [
            {"game_id": str((season - 1) * 1_000_000 + 20_000 + number),
             "season": str(season), "game_type": "R",
             "game_date": (start + timedelta(days=(number - 1) // 10)).isoformat(),
             "game_state": "OFF", "home_team_abbr": "BOS",
             "away_team_abbr": "NYR", "home_score": "3", "away_score": "2"}
            for number in range(1, 1313)
        ]
    return rows


def encode(rows):
    archives = {}
    for season, batch in rows.items():
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(batch)
        archives[season] = buffer.getvalue().encode("utf-8")
    return archives


def fixture(rows=None):
    archives = encode(synthetic_rows() if rows is None else rows)
    manifest = reconcile_public_schedules(
        REPORT, archives, observed_at_utc=OBSERVED).summary()
    return archives, json.dumps(manifest).encode("utf-8")


def build(archives, manifest, **kwargs):
    return build_public_schedule_form(
        REPORT, manifest, archives, observed_at_utc=OBSERVED, **kwargs)


class PublicScheduleFormTests(unittest.TestCase):
    def test_full_coverage_and_closed_gates(self):
        archives, manifest = fixture()
        audit = build(archives, manifest)
        self.assertEqual(len(audit.forms), 6560)
        summary = audit.summary()
        self.assertEqual(summary["regular_final_games_by_season"],
                         {str(s): 1312 for s in range(2022, 2027)})
        self.assertFalse(summary["historical_pregame_availability_proven"])
        self.assertFalse(summary["training_permitted"])
        self.assertFalse(summary["prediction_publication_permitted"])
        self.assertNotIn("forms", summary)

    def test_same_date_results_and_target_score_are_not_priors(self):
        archives, manifest = fixture()
        forms = {f.game_id: f for f in build(archives, manifest).forms}
        self.assertEqual(forms[2021020001].home_prior_games, 0)
        self.assertEqual(forms[2021020010].home_prior_games, 0)
        self.assertEqual(forms[2021020011].home_prior_games, 5)
        self.assertEqual(forms[2021020011].home_prior_wins, 5)
        self.assertEqual(forms[2021020011].away_prior_wins, 0)
        self.assertEqual(forms[2022020001].home_prior_games, 0)
        rows = synthetic_rows()
        rows[2022][9]["home_score"] = "9"
        changed_archives, changed_manifest = fixture(rows)
        changed = {f.game_id: f for f in build(
            changed_archives, changed_manifest).forms}
        self.assertEqual(changed[2021020010], forms[2021020010])
        self.assertNotEqual(changed[2021020011].home_prior_goal_diff,
                            forms[2021020011].home_prior_goal_diff)

    def test_mutable_release_is_rejected_by_pinned_hash(self):
        archives, manifest = fixture()
        modified = dict(archives)
        modified[2022] = modified[2022] + b" "
        with self.assertRaisesRegex(PublicScheduleFormError, "digest changed"):
            build(modified, manifest)

    def test_missing_regular_game_is_rejected_even_if_not_in_tied_set(self):
        rows = synthetic_rows()
        rows[2022].pop()
        archives, manifest = fixture(rows)
        with self.assertRaisesRegex(PublicScheduleFormError,
                                    "Incomplete regular season"):
            build(archives, manifest)

    def test_non_final_regular_game_is_rejected(self):
        rows = synthetic_rows()
        rows[2022][-1]["game_state"] = "LIVE"
        archives, manifest = fixture(rows)
        with self.assertRaisesRegex(PublicScheduleFormError,
                                    "Non-final or wrong-season"):
            build(archives, manifest)

    def test_invalid_lookback_and_manifest_gate_are_rejected(self):
        archives, manifest = fixture()
        for value in (0, 31, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(PublicScheduleFormError, "Lookback"):
                    build(archives, manifest, lookback_games=value)
        changed = json.loads(manifest)
        changed["training_permitted"] = True
        with self.assertRaisesRegex(PublicScheduleFormError, "gates differ"):
            build(archives, json.dumps(changed).encode())


    def test_duplicate_regular_game_is_rejected(self):
        rows = synthetic_rows()
        rows[2022][-1] = rows[2022][0].copy()
        archives, manifest = fixture(rows)
        with self.assertRaisesRegex(PublicScheduleFormError, "Duplicate regular game"):
            build(archives, manifest)

    def test_playoff_is_excluded_and_source_season_variants_are_accepted(self):
        rows = synthetic_rows()
        rows[2022][0]["season"] = "2021"
        rows[2022].append(dict(rows[2022][0], game_id="2021030001",
                               game_type="P"))
        archives, manifest = fixture(rows)
        self.assertEqual(len(build(archives, manifest).forms), 6560)

    def test_checked_in_manifest_is_linked_to_prior_audits(self):
        from hashlib import sha256

        summary = json.loads((ROOT / "nhl18_prior_score_form_audit_v1.json")
                             .read_text(encoding="utf-8"))
        self.assertEqual(summary["nhl15_report_sha256"], sha256(REPORT).hexdigest())
        self.assertEqual(summary["nhl17_manifest_sha256"], sha256(
            (ROOT / "nhl17_public_schedule_audit_v1.json").read_bytes()).hexdigest())
        self.assertEqual(summary["regular_final_games"], 6560)
        self.assertFalse(summary["source_is_historical_pregame_snapshot"])
        self.assertFalse(summary["training_permitted"])
        self.assertFalse(summary["prediction_publication_permitted"])


if __name__ == "__main__":
    unittest.main()
