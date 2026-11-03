"""Stream and validate injected MoneyPuck responses with bounded retries."""
from contextlib import contextmanager
from http.client import HTTPException, IncompleteRead
import logging
import signal
import sys
from threading import current_thread, main_thread
from time import monotonic, sleep
from typing import Any, Callable
from urllib.error import HTTPError
import zlib

from src.nhl.moneypuck_team_import import MAX_FILE_BYTES, TEAM_GAME_DOWNLOAD


class MoneyPuckTransferError(ValueError):
    """The transport did not deliver a valid complete response."""


_LOGGER = logging.getLogger(__name__)
_DOWNLOAD_ATTEMPTS = 3
_ATTEMPT_SECONDS = 120
_IO_TIMEOUT_SECONDS = 90
_CHUNK_BYTES = 64 * 1024


@contextmanager
def _wall_clock_limit(seconds: float):
    """Interrupt even a trickling read on the Linux Actions main thread.

    Other callers retain the monotonic checks and per-I/O timeout. Do not
    replace an existing caller alarm, whose deadline must retain priority.
    """
    enabled = (hasattr(signal, 'setitimer') and current_thread() is main_thread()
               and signal.getitimer(signal.ITIMER_REAL)[0] == 0)
    if not enabled:
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError('MoneyPuck attempt wall-clock deadline reached')

    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _progress(attempt: int, wire_bytes: int, csv_bytes: int, started: float,
              attempts: int | None = None):
    total = _DOWNLOAD_ATTEMPTS if attempts is None else attempts
    print(f'MONEYPUCK_TRANSFER attempt={attempt}/{total} '
          f'wire_bytes={wire_bytes} csv_bytes={csv_bytes} '
          f'elapsed_seconds={monotonic() - started:.1f}', file=sys.stderr, flush=True)


def _stream_csv(response: Any, *, attempt: int, started: float,
                attempt_seconds: float | None = None,
                total_attempts: int | None = None) -> bytes:
    deadline = _ATTEMPT_SECONDS if attempt_seconds is None else attempt_seconds
    encoding = response.headers.get('Content-Encoding', 'identity').strip().lower()
    if encoding not in {'identity', 'gzip'}:
        raise MoneyPuckTransferError('Encodage MoneyPuck inattendu.')
    length = response.headers.get('Content-Length')
    expected = int(length) if length is not None else None
    if expected is not None and not 0 < expected <= MAX_FILE_BYTES:
        raise MoneyPuckTransferError('Taille annoncée du CSV MoneyPuck invalide.')
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == 'gzip' else None
    # HTTPResponse.read1 returns available data rather than filling the buffer.
    reader = getattr(response, 'read1', response.read)
    result = bytearray()
    wire_bytes = 0
    next_progress = started + 15
    while True:
        if monotonic() - started >= deadline:
            raise TimeoutError('MoneyPuck attempt deadline reached')
        chunk = reader(_CHUNK_BYTES)
        if not chunk:
            break
        wire_bytes += len(chunk)
        if wire_bytes > MAX_FILE_BYTES:
            raise MoneyPuckTransferError('Transfert MoneyPuck trop volumineux.')
        if decoder is not None:
            chunk = decoder.decompress(chunk, MAX_FILE_BYTES - len(result) + 1)
        result.extend(chunk)
        if len(result) > MAX_FILE_BYTES:
            raise MoneyPuckTransferError('Fichier MoneyPuck trop volumineux.')
        if monotonic() >= next_progress:
            _progress(attempt, wire_bytes, len(result), started, total_attempts)
            next_progress = monotonic() + 15
    if expected is not None and wire_bytes != expected:
        raise IncompleteRead(b'')
    if decoder is not None and (not decoder.eof or decoder.unused_data):
        raise IncompleteRead(b'')
    if not result:
        raise IncompleteRead(b'')
    _progress(attempt, wire_bytes, len(result), started, total_attempts)
    return bytes(result)


def _download_team_csv(transport: Callable[..., Any], *,
                       attempt_seconds: float | None = None,
                       max_attempts: int | None = None) -> bytes:
    """Restart interrupted transfers; never accept or append partial bytes."""
    seconds = _ATTEMPT_SECONDS if attempt_seconds is None else attempt_seconds
    attempts = _DOWNLOAD_ATTEMPTS if max_attempts is None else max_attempts
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not 0 < seconds <= 900 or type(attempts) is not int or not 1 <= attempts <= 3):
        raise MoneyPuckTransferError('Budget de collecte MoneyPuck invalide.')
    for attempt in range(1, attempts + 1):
        started = monotonic()
        print(f'MONEYPUCK_TRANSFER_START attempt={attempt}/{attempts} '
              f'deadline_seconds={seconds}', file=sys.stderr, flush=True)
        try:
            with _wall_clock_limit(seconds):
                with transport(TEAM_GAME_DOWNLOAD, timeout=_IO_TIMEOUT_SECONDS) as response:
                    if response.status != 200 or response.geturl() != TEAM_GAME_DOWNLOAD:
                        raise MoneyPuckTransferError('Réponse redirigée ou non réussie.')
                    content_type = response.headers.get('Content-Type', '').split(';', 1)[0].lower()
                    if content_type not in {'text/csv', 'text/plain', 'application/octet-stream'}:
                        raise MoneyPuckTransferError('Type de réponse inattendu.')
                    return _stream_csv(response, attempt=attempt, started=started, attempt_seconds=seconds,
                                       total_attempts=attempts)
        except HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504}:
                raise MoneyPuckTransferError('Collecte MoneyPuck refusée.') from error
            failure = error
        except (OSError, HTTPException, zlib.error) as error:
            failure = error
        _LOGGER.warning('MoneyPuck: tentative %s interrompue (%s) après %.1f secondes',
                        attempt, type(failure).__name__, monotonic() - started)
        if attempt == attempts:
            raise MoneyPuckTransferError(f'Collecte MoneyPuck échouée après {attempts} tentatives.') from failure
        sleep(2 ** attempt)
    raise AssertionError('Unreachable download state')
