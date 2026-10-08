"""Offline tests for the manual 50-game NHL historical-final batch."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.nhl.historical_final_batch import (
    HistoricalFinalBatchPlan, capture_historical_final_batch,
    plan_historical_final_batch,
)
from src.nhl.historical_official_final import HistoricalOfficialFinalError


class HistoricalFinalBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.team = Path('team')
        self.final = Path('final')
        self.digest = 'a' * 64
        self.ids = tuple(range(1, 13))

    def _plan(self, ids=None):
        return HistoricalFinalBatchPlan(
            'c' * 64, self.digest, self.ids if ids is None else ids,
            50, 'd' * 64,
        )

    def test_offline_plan_caps_at_fifty(self) -> None:
        backlog = SimpleNamespace(
            source_capture_sha256='c' * 64,
            backlog_sha256=self.digest,
            pending_tied_game_ids=tuple(range(1, 61)),
        )
        with patch('src.nhl.historical_final_batch.inspect_historical_final_backlog',
                   return_value=backlog) as inspect:
            plan = plan_historical_final_batch(self.team, root=self.final)
            self.assertEqual(plan.game_ids, tuple(range(1, 51)))
            self.assertFalse(plan.training_permitted)
            self.assertFalse(plan.prediction_publication_permitted)
            self.assertEqual(inspect.call_count, 1)
            for invalid in (0, 51, True, 1.0):
                with self.subTest(invalid=invalid), self.assertRaises(HistoricalOfficialFinalError):
                    plan_historical_final_batch(self.team, request_limit=invalid)
            self.assertEqual(inspect.call_count, 1)

    def test_fifty_game_plan_is_split_into_ten_game_pilots(self) -> None:
        next_id = 0
        transport_calls = []
        pauses = []
        chunks = []

        def pilot_plan(*_args, request_limit, **_kwargs):
            return SimpleNamespace(
                backlog_sha256='b' * 64,
                game_ids=self.ids[next_id:next_id + request_limit],
            )

        def pilot_capture(*_args, request_limit, transport, **_kwargs):
            nonlocal next_id
            ids = self.ids[next_id:next_id + request_limit]
            for game_id in ids:
                transport(str(game_id), timeout=15)
            next_id += len(ids)
            return tuple(SimpleNamespace(game_id=game_id) for game_id in ids)

        with patch('src.nhl.historical_final_batch.plan_historical_final_batch',
                   return_value=self._plan()), patch(
                   'src.nhl.historical_final_batch.plan_historical_final_pilot',
                   side_effect=pilot_plan) as pilot, patch(
                   'src.nhl.historical_final_batch.capture_historical_final_pilot',
                   side_effect=pilot_capture) as capture:
            receipts = capture_historical_final_batch(
                self.team, expected_backlog_sha256=self.digest,
                explicit_manual_run=True, transport=lambda url, timeout: transport_calls.append(url),
                sleep=pauses.append, on_chunk=chunks.append,
            )
        self.assertEqual(tuple(item.game_id for item in receipts), self.ids)
        self.assertEqual([call.kwargs['request_limit'] for call in pilot.call_args_list], [10, 2])
        self.assertEqual([call.kwargs['request_limit'] for call in capture.call_args_list], [10, 2])
        self.assertEqual(len(transport_calls), 12)
        self.assertEqual(pauses, [1.0] * 11)
        self.assertEqual([len(chunk) for chunk in chunks], [10, 2])

    def test_stale_hash_and_missing_manual_consent_block_network(self) -> None:
        with patch('src.nhl.historical_final_batch.plan_historical_final_batch',
                   return_value=self._plan()), patch(
                   'src.nhl.historical_final_batch.capture_historical_final_pilot') as capture:
            for digest, manual, transport in (
                ('b' * 64, True, lambda *_args, **_kwargs: None),
                (self.digest, False, lambda *_args, **_kwargs: None),
                (self.digest, True, None),
            ):
                with self.subTest(digest=digest, manual=manual), self.assertRaises(HistoricalOfficialFinalError):
                    capture_historical_final_batch(
                        self.team, expected_backlog_sha256=digest,
                        explicit_manual_run=manual, transport=transport,
                    )
            capture.assert_not_called()

    def test_changed_chunk_stops_without_network(self) -> None:
        wrong = SimpleNamespace(backlog_sha256='b' * 64, game_ids=(99,))
        with patch('src.nhl.historical_final_batch.plan_historical_final_batch',
                   return_value=self._plan((1,))), patch(
                   'src.nhl.historical_final_batch.plan_historical_final_pilot',
                   return_value=wrong), patch(
                   'src.nhl.historical_final_batch.capture_historical_final_pilot') as capture:
            with self.assertRaises(HistoricalOfficialFinalError):
                capture_historical_final_batch(
                    self.team, expected_backlog_sha256=self.digest,
                    explicit_manual_run=True, transport=lambda *_args, **_kwargs: None,
                    sleep=lambda _seconds: None,
                )
            capture.assert_not_called()


if __name__ == '__main__':
    unittest.main()
