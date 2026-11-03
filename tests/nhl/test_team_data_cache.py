"""Freshness, provenance and no-network consumer regression tests."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from nhl_operations.team_data_cache import (
    NoFreshTeamData, TeamDataCacheError, collect_or_reuse, require_fresh_team_capture,
    materialize_capture,
)
from nhl_operations.automatic_research_capture import prepare_automatic_research
from src.nhl.moneypuck_team_capture import capture_team_history
try:
    from .test_automated_research_capture import TeamResponse, ScheduleResponse, team_bytes, schedule_bytes, OBSERVED
except ImportError:  # Also support direct discovery with -s tests/nhl.
    from test_automated_research_capture import TeamResponse, ScheduleResponse, team_bytes, schedule_bytes, OBSERVED


class CacheTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / 'team'
        self.tmp = Path(tmp.name)
        self.clock = OBSERVED

    def capture(self, stamp=None, automatic=True):
        return capture_team_history(
            explicit_automated_run=automatic, explicit_manual_run=not automatic,
            transport=lambda *a, **k: TeamResponse(team_bytes()),
            code_commit='a'*40, root=self.root, now=lambda: stamp or self.clock)

    def test_valid_cache_preserves_time_and_skips_download(self):
        source = self.capture()
        opener = Mock(side_effect=AssertionError('Network forbidden'))
        verified, reused = collect_or_reuse(root=self.root, code_commit='a'*40,
            transport=opener, now=lambda: self.clock+timedelta(hours=2))
        self.assertTrue(reused)
        opener.assert_not_called()
        self.assertEqual(source.observed_at_utc, verified.observed_at_utc)
        self.assertEqual(source.response_sha256, verified.response_sha256)

    def test_explicit_refresh_keeps_previous_receipt_and_collects_a_new_one(self):
        previous=self.capture()
        later=self.clock+timedelta(minutes=10)
        found,reused=collect_or_reuse(root=self.root,code_commit='a'*40,
            transport=lambda *a,**k:TeamResponse(team_bytes()),
            force_refresh=True,now=lambda:later)
        self.assertFalse(reused)
        self.assertNotEqual(previous.path,found.path)
        self.assertTrue((previous.path/'COMPLETED').exists())
        self.assertEqual(found.observed_at_utc,later)

    def test_six_hour_boundary_and_stale_capture(self):
        self.capture()
        require_fresh_team_capture(self.root, observed=self.clock+timedelta(hours=6))
        with self.assertRaises(NoFreshTeamData):
            require_fresh_team_capture(self.root, observed=self.clock+timedelta(hours=6, seconds=1))

    def test_previous_paris_day_rejected_even_within_six_hours(self):
        stamp=datetime(2026, 10, 14, 21, tzinfo=timezone.utc)
        self.capture(stamp)
        with self.assertRaises(NoFreshTeamData):
            require_fresh_team_capture(self.root, observed=stamp+timedelta(hours=3))

    def test_previous_utc_day_same_paris_day_is_accepted(self):
        stamp=datetime(2026, 10, 14, 23, tzinfo=timezone.utc)
        source=self.capture(stamp)
        found=require_fresh_team_capture(self.root, observed=stamp+timedelta(hours=2))
        self.assertEqual(found.path, source.path)

    def test_newest_valid_capture_is_selected(self):
        self.capture()
        latest=self.capture(self.clock+timedelta(minutes=10))
        self.assertEqual(require_fresh_team_capture(self.root,
            observed=self.clock+timedelta(minutes=11)).path, latest.path)

    def test_corrupt_cache_is_not_hidden_by_a_new_download(self):
        source=self.capture()
        (source.path/'response.csv').write_bytes(b'altered')
        opener=Mock()
        with self.assertRaises(TeamDataCacheError):
            collect_or_reuse(root=self.root, code_commit='a'*40, transport=opener, now=lambda:self.clock)
        opener.assert_not_called()

    def test_future_and_manual_captures_cannot_be_reused(self):
        self.capture(self.clock+timedelta(seconds=1))
        with self.assertRaises(TeamDataCacheError):
            require_fresh_team_capture(self.root, observed=self.clock)
        with TemporaryDirectory() as directory:
            self.root=Path(directory)
            self.capture(automatic=False)
            with self.assertRaises(NoFreshTeamData):
                require_fresh_team_capture(self.root, observed=self.clock)

    def test_incomplete_and_symlink_cache_rejected(self):
        source=self.capture()
        (source.path/'COMPLETED').unlink()
        with self.assertRaises(NoFreshTeamData):
            require_fresh_team_capture(self.root, observed=self.clock)
        (source.path/'COMPLETED').symlink_to(source.path/'response.csv')
        with self.assertRaises(TeamDataCacheError):
            require_fresh_team_capture(self.root, observed=self.clock)

    def test_empty_cache_collects_with_long_bounded_budget(self):
        with patch('nhl_operations.team_data_cache.capture_team_history', wraps=capture_team_history) as collect:
            capture,reused=collect_or_reuse(root=self.root, code_commit='a'*40,
                transport=lambda *a, **k: TeamResponse(team_bytes()),now=lambda:self.clock)
        self.assertFalse(reused)
        self.assertEqual(collect.call_args.kwargs['download_attempt_seconds'],900)
        self.assertEqual(collect.call_args.kwargs['download_max_attempts'],2)
        self.assertTrue((capture.path/'COMPLETED').is_file())

    def test_calculation_uses_fresh_schedule_and_cached_stats_only(self):
        source=self.capture()
        calls=[]
        def transport(url,timeout):
            calls.append(url)
            if 'api-web.nhle.com' not in url:
                raise AssertionError('Consumer downloaded team CSV')
            return ScheduleResponse(schedule_bytes())
        result=prepare_automatic_research(code_commit='a'*40,explicit_automated_run=True,
            cached_team_only=True,team_root=self.root,schedule_root=self.tmp/'schedule',
            import_root=self.tmp/'imports',transport=transport,now=lambda:self.clock+timedelta(minutes=1))
        self.assertEqual(len(calls),1)
        self.assertEqual(result.team_slot, source.path)
        self.assertEqual(result.current_regular_games,1)

    def test_calculation_without_cache_never_falls_back_to_download(self):
        with self.assertRaises(NoFreshTeamData):
            prepare_automatic_research(code_commit='a'*40,explicit_automated_run=True,
                cached_team_only=True,team_root=self.root,schedule_root=self.tmp/'schedule',
                import_root=self.tmp/'imports',transport=lambda *a,**k:ScheduleResponse(schedule_bytes()),
                now=lambda:self.clock)

    def test_materialize_preserves_receipt_and_never_copies_old_slots(self):
        self.capture(self.clock-timedelta(days=1))
        source=self.capture()
        destination=self.tmp/'engine'/'team'
        found=materialize_capture(self.root,destination,observed=self.clock)
        self.assertEqual(found.observed_at_utc,source.observed_at_utc)
        self.assertEqual((found.path/'receipt.json').read_bytes(),(source.path/'receipt.json').read_bytes())
        self.assertEqual(len(list(destination.rglob('COMPLETED'))),1)
        # A repeated materialization verifies the existing copy rather than replacing it.
        again=materialize_capture(self.root,destination,observed=self.clock)
        self.assertEqual(found.path,again.path)
        (found.path/'response.csv').write_bytes(b'corrupt destination')
        with self.assertRaises(ValueError):
            materialize_capture(self.root,destination,observed=self.clock)
        self.assertEqual((found.path/'response.csv').read_bytes(),b'corrupt destination')

    def test_long_collection_does_not_make_old_data_fresh(self):
        # Restore time is not substituted for the actual receipt time.
        self.capture(self.clock-timedelta(hours=7))
        with self.assertRaises(NoFreshTeamData):
            materialize_capture(self.root,self.tmp/'engine-team',observed=self.clock)


if __name__=='__main__':
    unittest.main()
