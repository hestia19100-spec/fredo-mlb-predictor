"""Boundary checks for the isolated MLB 2027 one-hour contract."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import unittest

from src.mlb_2027_timing import (
    MLB2027TimingError, build_lead_window, is_prospectively_on_time,
)

UTC = timezone.utc
DAY = date(2027, 4, 1)
FIRST = datetime(2027, 4, 1, 20, 0, tzinfo=UTC)
LATER = datetime(2027, 4, 1, 22, 0, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[1]


class MLB2027TimingTests(unittest.TestCase):
    def test_earliest_start_sets_exact_hour_and_run_buffer(self) -> None:
        window = build_lead_window(DAY, [LATER, FIRST])
        self.assertEqual(window.earliest_start_utc, FIRST)
        self.assertEqual(window.latest_completion_utc, FIRST - timedelta(hours=1))
        self.assertEqual(window.latest_remote_publication_utc, FIRST - timedelta(hours=1))
        self.assertEqual(window.suggested_run_start_utc, FIRST - timedelta(minutes=90))
        self.assertTrue(is_prospectively_on_time(
            window, issued_at_utc=FIRST - timedelta(minutes=70),
            completed_at_utc=FIRST - timedelta(minutes=60),
            remote_http_date_utc=FIRST - timedelta(minutes=60),
        ))

    def test_one_second_late_publication_is_not_certified(self) -> None:
        window = build_lead_window(DAY, [FIRST])
        self.assertFalse(is_prospectively_on_time(
            window, issued_at_utc=FIRST - timedelta(minutes=70),
            completed_at_utc=FIRST - timedelta(minutes=65),
            remote_http_date_utc=FIRST - timedelta(minutes=60) + timedelta(seconds=1),
        ))

    def test_one_second_late_completion_is_not_certified(self) -> None:
        window = build_lead_window(DAY, [FIRST])
        self.assertFalse(is_prospectively_on_time(
            window, issued_at_utc=FIRST - timedelta(minutes=70),
            completed_at_utc=FIRST - timedelta(minutes=60) + timedelta(seconds=1),
            remote_http_date_utc=FIRST - timedelta(minutes=60) + timedelta(seconds=2),
        ))

    def test_bad_scope_or_clock_fails_closed(self) -> None:
        with self.assertRaises(MLB2027TimingError):
            build_lead_window(date(2026, 9, 20), [FIRST])
        with self.assertRaises(MLB2027TimingError):
            build_lead_window(DAY, [])
        with self.assertRaises(MLB2027TimingError):
            build_lead_window(DAY, [FIRST.replace(tzinfo=None)])
        with self.assertRaises(MLB2027TimingError):
            build_lead_window(DAY, [FIRST.astimezone(timezone(timedelta(hours=1)))])
        window = build_lead_window(DAY, [FIRST])
        with self.assertRaises(MLB2027TimingError):
            is_prospectively_on_time(
                window, issued_at_utc=FIRST - timedelta(minutes=61),
                completed_at_utc=FIRST - timedelta(minutes=65),
                remote_http_date_utc=FIRST - timedelta(minutes=60),
            )

    def test_2026_frozen_contract_remains_two_hours(self) -> None:
        protocol = json.loads((ROOT / "shadow_protocols"
            / "logistic_team_form_v1_platt_shadow_v2.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["operating_mode"]["target_season_exact"], 2026)
        self.assertEqual(protocol["operating_mode"]["minimum_local_completion_lead_minutes"], 120)


if __name__ == "__main__":
    unittest.main()
