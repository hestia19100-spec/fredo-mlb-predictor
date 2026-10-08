"""Bounded, direct HTTPS transport for one manually requested NHL final.

Uses the system curl client without redirects, credentials or User-Agent spoofing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from .official_final_capture import MAX_BYTES

_URL = re.compile(r"https://api-web\.nhle\.com/v1/gamecenter/[0-9]{10}/landing\Z")


class NHLDirectHTTPError(ValueError):
    """A direct official response could not be obtained and bounded."""


@dataclass(frozen=True, slots=True)
class DirectJSONResponse:
    status: int
    headers: dict[str, str]
    url: str
    body: bytes

    def __enter__(self) -> DirectJSONResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def geturl(self) -> str:
        return self.url

    def read(self, size: int) -> bytes:
        if type(size) is not int or size < 0:
            raise NHLDirectHTTPError("Taille de lecture invalide.")
        return self.body[:size]


def curl_direct_nhl_json(url: str, timeout: int) -> DirectJSONResponse:
    """Perform one bounded GET; fail on redirects and any HTTP anomaly."""
    if not isinstance(url, str) or not _URL.fullmatch(url):
        raise NHLDirectHTTPError("URL NHL hors perimetre.")
    if type(timeout) is not int or not 1 <= timeout <= 15:
        raise NHLDirectHTTPError("Delai de capture invalide.")
    executable = shutil.which("curl")
    if executable is None:
        raise NHLDirectHTTPError("Client HTTPS systeme indisponible.")
    with TemporaryDirectory(prefix="nhl-direct-final-") as directory:
        body_path = Path(directory) / "body.json"
        arguments = [
            executable, "--silent", "--show-error", "--fail",
            "--proto", "=https", "--max-time", str(timeout),
            "--max-filesize", str(MAX_BYTES + 1), "--max-redirs", "0",
            "--output", str(body_path),
            "--write-out", "%{http_code}\n%{url_effective}\n%{content_type}\n",
            url,
        ]
        try:
            completed = subprocess.run(
                arguments, capture_output=True, check=False, timeout=timeout + 2,
            )
            metadata = completed.stdout.decode("ascii").splitlines()
            if completed.returncode != 0 or len(metadata) != 3:
                raise NHLDirectHTTPError("Reponse HTTPS indisponible.")
            status, effective_url, content_type = metadata
            if (status != "200" or effective_url != url
                    or content_type.split(";", 1)[0].strip().lower() != "application/json"):
                raise NHLDirectHTTPError("Statut, URL ou type de reponse invalide.")
            if not body_path.is_file() or not 0 < body_path.stat().st_size <= MAX_BYTES:
                raise NHLDirectHTTPError("Taille du resultat invalide.")
            body = body_path.read_bytes()
        except (OSError, UnicodeError, subprocess.TimeoutExpired) as error:
            raise NHLDirectHTTPError("Echec du transport HTTPS direct.") from error
    if not 0 < len(body) <= MAX_BYTES:
        raise NHLDirectHTTPError("Resultat incomplet ou trop grand.")
    return DirectJSONResponse(200, {"Content-Type": content_type}, url, body)
