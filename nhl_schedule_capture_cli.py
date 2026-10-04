"""Commande réseau isolée du cœur NHL pour une capture manuelle du calendrier."""
from __future__ import annotations

import argparse
from datetime import date
from urllib.request import HTTPRedirectHandler, Request, build_opener

from src.nhl.public_schedule_capture import BASE_URL, capture_public_schedule


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _transport(url: str, timeout: int):
    if not url.startswith(BASE_URL):
        raise ValueError("Adresse NHL non autorisée.")
    day = url[len(BASE_URL):]
    if date.fromisoformat(day).isoformat() != day:
        raise ValueError("Date NHL non canonique.")
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "LPF-Edge-Personal-Research/1.0",
        },
        method="GET",
    )
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture manuelle du calendrier NHL")
    parser.add_argument("target_date", type=date.fromisoformat)
    parser.add_argument("--capture", action="store_true", help="autoriser un GET réel pour cette exécution")
    arguments = parser.parse_args()
    if not arguments.capture:
        parser.error("--capture est requis pour effectuer un appel réseau")
    result = capture_public_schedule(
        arguments.target_date, explicit_manual_run=True, transport=_transport
    )
    print(result.path)
    print("Matchs réguliers futurs:", result.game_count)
    print("SHA-256:", result.response_sha256)


if __name__ == "__main__":
    main()
