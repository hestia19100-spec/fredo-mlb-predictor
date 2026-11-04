"""Real deadline and compressed-stream failure regressions; no network."""
from contextlib import nullcontext
import gzip
from io import BytesIO
import signal
import time
import unittest
from unittest.mock import Mock, patch

from nhl_operations import moneypuck_transfer as transfer
from src.nhl.moneypuck_team_import import TEAM_GAME_DOWNLOAD


class Response(BytesIO):
    status = 200
    def __init__(self, raw, encoding='identity', length=None):
        super().__init__(raw)
        self.headers = {'Content-Type': 'text/csv', 'Content-Encoding': encoding}
        if length is not None:
            self.headers['Content-Length'] = str(length)
    def geturl(self):
        return TEAM_GAME_DOWNLOAD


class TransferTests(unittest.TestCase):
    def test_compressed_transfer_returns_identical_csv_and_logs_progress(self):
        raw = b'a,b,c\n1,2,3\n' * 20000
        wire = gzip.compress(raw)
        opener = Mock(return_value=Response(wire, 'gzip', len(wire)))
        with patch.object(transfer.sys, 'stderr') as output:
            self.assertEqual(transfer._download_team_csv(opener), raw)
        self.assertIn('wire_bytes=', str(output.write.call_args_list))
        opener.assert_called_once_with(TEAM_GAME_DOWNLOAD, timeout=90)

    def test_chunk_reader_is_used_instead_of_a_full_blocking_read(self):
        response = Response(b'a,b\n1,2\n')
        response.read = Mock(side_effect=AssertionError('Full read forbidden'))
        self.assertEqual(transfer._download_team_csv(lambda *a, **k: response), b'a,b\n1,2\n')

    def test_truncated_or_corrupt_gzip_restarts_from_zero(self):
        raw = b'a,b\n1,2\n'
        for broken in (gzip.compress(raw)[:-5], b'not gzip'):
            with self.subTest(broken=broken[:8]):
                first = Response(broken, 'gzip')
                opener = Mock(side_effect=[first, Response(gzip.compress(raw), 'gzip')])
                with patch.object(transfer, 'sleep'):
                    self.assertEqual(transfer._download_team_csv(opener), raw)
                self.assertEqual(opener.call_count, 2)
                self.assertTrue(first.closed)

    def test_decompression_size_limit_is_enforced(self):
        opener = Mock(return_value=Response(gzip.compress(b'a' * 10000), 'gzip'))
        with patch.object(transfer, 'MAX_FILE_BYTES', 100):
            with self.assertRaisesRegex(transfer.MoneyPuckTransferError, 'volumineux'):
                transfer._download_team_csv(opener)
        opener.assert_called_once()

    def test_unknown_encoding_fails_without_retry(self):
        opener = Mock(return_value=Response(b'data', 'br'))
        with self.assertRaises(transfer.MoneyPuckTransferError):
            transfer._download_team_csv(opener)
        opener.assert_called_once()

    def test_monotonic_deadline_also_applies_without_signal(self):
        with patch.object(transfer, 'monotonic', return_value=121):
            with self.assertRaises(TimeoutError):
                transfer._stream_csv(Response(b'data'), attempt=1, started=0)

    @unittest.skipUnless(hasattr(signal, 'setitimer'), 'POSIX deadline')
    def test_wall_clock_interrupts_a_blocked_read_and_restores_alarm(self):
        previous = signal.getsignal(signal.SIGALRM)
        opener = Mock()
        def response(*args, **kwargs):
            item = Response(b'partial')
            item.read1 = lambda n: (time.sleep(1) or b'')
            return item
        opener.side_effect = response
        started = time.monotonic()
        with patch.object(transfer, '_ATTEMPT_SECONDS', 0.03), patch.object(transfer, 'sleep'):
            with self.assertRaisesRegex(transfer.MoneyPuckTransferError, '3 tentatives'):
                transfer._download_team_csv(opener)
        self.assertLess(time.monotonic() - started, 0.8)
        self.assertEqual(opener.call_count, 3)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    @unittest.skipUnless(hasattr(signal, 'setitimer'), 'POSIX deadline')
    def test_existing_caller_alarm_is_preserved(self):
        signal.setitimer(signal.ITIMER_REAL, 30)
        try:
            with transfer._wall_clock_limit(0.01):
                self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 29)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
