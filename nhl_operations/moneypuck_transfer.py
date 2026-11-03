"""Bounded retries for an injected MoneyPuck transport; no HTTP client is opened here."""
from http.client import HTTPException, IncompleteRead
import logging
from time import sleep
from typing import Any, Callable
from urllib.error import HTTPError

from src.nhl.moneypuck_team_import import MAX_FILE_BYTES, TEAM_GAME_DOWNLOAD


class MoneyPuckTransferError(ValueError):
    """The transport did not deliver a valid complete response."""


_LOGGER = logging.getLogger(__name__)
_DOWNLOAD_ATTEMPTS = 3


def _download_team_csv(transport: Callable[..., Any]) -> bytes:
    """Restart interrupted transfers; never accept or append partial bytes."""
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            with transport(TEAM_GAME_DOWNLOAD, timeout=45) as response:
                if response.status != 200 or response.geturl() != TEAM_GAME_DOWNLOAD:
                    raise MoneyPuckTransferError("Réponse redirigée ou non réussie.")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                if content_type not in {"text/csv", "text/plain", "application/octet-stream"}:
                    raise MoneyPuckTransferError("Type de réponse inattendu.")
                length = response.headers.get("Content-Length")
                expected = int(length) if length is not None else None
                if expected is not None and not 0 < expected <= MAX_FILE_BYTES:
                    raise MoneyPuckTransferError("Taille annoncée du CSV MoneyPuck invalide.")
                raw = response.read(MAX_FILE_BYTES + 1)
                if len(raw) > MAX_FILE_BYTES:
                    raise MoneyPuckTransferError("Fichier MoneyPuck trop volumineux.")
                if expected is not None and len(raw) != expected:
                    raise IncompleteRead(b"")
                return raw
        except HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504}:
                raise MoneyPuckTransferError("Collecte MoneyPuck refusée.") from error
            failure = error
        except (OSError, HTTPException) as error:
            failure = error
        if attempt == _DOWNLOAD_ATTEMPTS:
            raise MoneyPuckTransferError("Collecte MoneyPuck échouée après 3 tentatives.") from failure
        _LOGGER.warning("MoneyPuck: transfert interrompu ou indisponible; nouvelle tentative %s/%s",
                        attempt + 1, _DOWNLOAD_ATTEMPTS)
        sleep(2 ** attempt)
    raise AssertionError("Unreachable download state")
