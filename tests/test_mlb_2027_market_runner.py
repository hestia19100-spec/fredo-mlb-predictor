"""Tests du verrou à usage unique et du flux privé MLB 2027, sans réseau."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, sentinel
import unittest

from src.mlb_2027_market_runner import (
    MLB2027PrivateRunError,
    _reserve_one_odds_attempt,
    run_private_market_day,
)
from src.mlb_2027_remote_visibility import RemoteVisibility
from tests.test_mlb_2027_observed_source import TARGET, _observed


IN_WINDOW = datetime(2027, 1, 20, 21, 30, tzinfo=timezone.utc)
TOO_EARLY = datetime(2027, 1, 20, 20, 0, tzinfo=timezone.utc)


class MarketRunnerTests(unittest.TestCase):
    def test_reservation_is_pushed_once_before_any_api_call(self):
        with TemporaryDirectory() as folder:
            repo = Path(folder)
            calls = []

            def git(_repo, *args):
                calls.append(args)
                if args == ("diff", "--cached", "--name-only"):
                    return "mlb_2027_odds_attempts/2027-01-20/RESERVED.json"
                if args == ("rev-parse", "HEAD"):
                    return "a" * 40
                if args == ("ls-remote", "origin", "refs/heads/main"):
                    return "a" * 40 + "\trefs/heads/main"
                return ""

            with patch("src.mlb_2027_market_runner._git", side_effect=git):
                _reserve_one_odds_attempt(repo, TARGET, "b" * 64, IN_WINDOW)
                with self.assertRaises(MLB2027PrivateRunError):
                    _reserve_one_odds_attempt(repo, TARGET, "b" * 64, IN_WINDOW)
            self.assertIn(("push", "origin", "HEAD:refs/heads/main"), calls)
            self.assertTrue((repo / "mlb_2027_odds_attempts" / TARGET.isoformat() / "RESERVED.json").is_file())

    def test_outside_safe_window_spends_no_credit(self):
        with TemporaryDirectory() as folder:
            with (
                patch.dict("os.environ", {"GITHUB_TOKEN": "test", "THE_ODDS_API_KEY": "test"}),
                patch("src.mlb_2027_market_runner._check_private_checkout", return_value=Path(folder) / "mlb_2027_candidates"),
                patch("src.mlb_2027_market_runner.fetch_schedule_range_observed", return_value=_observed()),
                patch("src.mlb_2027_market_runner._now_utc", return_value=TOO_EARLY),
                patch("src.mlb_2027_market_runner._reserve_one_odds_attempt") as reserve,
                patch("src.mlb_2027_market_runner.fetch_mlb_moneyline_odds") as fetch_odds,
            ):
                with self.assertRaises(MLB2027PrivateRunError):
                    run_private_market_day(TARGET, private_repo=Path(folder))
                reserve.assert_not_called()
                fetch_odds.assert_not_called()

    def test_private_chain_collects_only_after_reservation(self):
        with TemporaryDirectory() as folder:
            repo = Path(folder)
            proof = RemoteVisibility(
                repository="hestia19100-spec/fredo-lpf-edge-private-ops",
                branch="main", path="mlb_2027_candidates/example/candidate.json",
                candidate_sha256="a" * 64, github_http_date_utc=IN_WINDOW,
                response_received_at_utc=IN_WINDOW,
            )
            events = []

            def reserve(*_args):
                events.append("reserve")

            def fetch():
                events.append("odds")
                return sentinel.odds

            with (
                patch.dict("os.environ", {"GITHUB_TOKEN": "test", "THE_ODDS_API_KEY": "test"}),
                patch("src.mlb_2027_market_runner._check_private_checkout", return_value=repo / "mlb_2027_candidates"),
                patch("src.mlb_2027_market_runner.fetch_schedule_range_observed", return_value=_observed()),
                patch("src.mlb_2027_market_runner._now_utc", side_effect=[IN_WINDOW, IN_WINDOW, IN_WINDOW]),
                patch("src.mlb_2027_market_runner._reserve_one_odds_attempt", side_effect=reserve),
                patch("src.mlb_2027_market_runner.fetch_mlb_moneyline_odds", side_effect=fetch) as fetch_odds,
                patch("src.mlb_2027_market_runner.match_mlb_2027_market", return_value=sentinel.market),
                patch("src.mlb_2027_market_runner.archive_reference_candidate", return_value=sentinel.archive) as archive,
                patch("src.mlb_2027_market_runner._commit_private_archive"),
                patch("src.mlb_2027_market_runner.verify_private_main_visibility", return_value=proof),
            ):
                result = run_private_market_day(TARGET, private_repo=repo)
            self.assertEqual(events, ["reserve", "odds"])
            fetch_odds.assert_called_once_with()
            self.assertIs(archive.call_args.kwargs["odds"], sentinel.odds)
            self.assertIs(archive.call_args.kwargs["market"], sentinel.market)
            self.assertEqual(result.status, "VERIFIED_PRIVATE_ONLY")


if __name__ == "__main__":
    unittest.main()
