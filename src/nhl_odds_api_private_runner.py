"""Capture-only NHL runner for a separate, private Git repository.

Every quota reservation is pushed *before* its provider request. A failed
commit or push stops the request. A lost runner can therefore lose evidence,
but it cannot forget credits already risked and silently retry that day.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Callable

import requests

from src.nhl_odds_api_daily import capture_nhl_daily_pregame, capture_nhl_daily_scores


class NHLPrivateRunnerError(RuntimeError):
    """The private capture could not be completed safely."""


def _git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True,
            text=True, timeout=45,
        )
    except (OSError, subprocess.SubprocessError):
        # Remote URLs and Git stderr may include credentials. Never echo them.
        raise NHLPrivateRunnerError("Opération Git privée échouée; aucun nouvel appel autorisé.") from None
    return result.stdout.strip()


class PrivateGitLedger:
    """Publish the quota ledger to the private default branch synchronously."""

    def __init__(self, repo: Path) -> None:
        self.repo = Path(repo).resolve(strict=True)
        if Path(_git(self.repo, "rev-parse", "--show-toplevel")).resolve() != self.repo:
            raise NHLPrivateRunnerError("La racine du dépôt privé est incorrecte.")
        if _git(self.repo, "symbolic-ref", "--short", "HEAD") != "main":
            raise NHLPrivateRunnerError("Le dépôt privé doit être sur sa branche main.")
        if _git(self.repo, "status", "--porcelain", "--untracked-files=all"):
            raise NHLPrivateRunnerError("Le dépôt privé contient déjà des changements.")
        self.ledger = self.repo / "state" / "nhl_daily_quota.sqlite"

    def _publish(self, message: str, *paths: Path) -> None:
        relative = []
        for path in paths:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(self.repo):
                raise NHLPrivateRunnerError("Fichier hors du dépôt privé.")
            relative.append(str(resolved.relative_to(self.repo)))
        _git(self.repo, "add", "-f", "--", *relative)
        _git(
            self.repo, "-c", "user.name=LPF Edge automation",
            "-c", "user.email=lpf-edge-automation@users.noreply.github.com",
            "commit", "-m", message,
        )
        _git(self.repo, "push", "origin", "HEAD:refs/heads/main")

    def publish_reservation(self) -> None:
        self._publish("Reserve NHL Odds API daily credit", self.ledger)

    def publish_capture(self, slot: Path) -> None:
        self._publish("Archive NHL capture-only evidence", self.ledger, slot)


def run_private_capture(
    repo: Path,
    mode: str,
    *,
    transport: Callable = requests.get,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Path:
    """Run one capture, never a model, wager, label or publication."""
    if mode not in {"pregame", "scores"}:
        raise NHLPrivateRunnerError("Mode NHL inconnu.")
    publisher = PrivateGitLedger(repo)
    capture_root = publisher.repo / "captures" / "nhl" / mode
    common = {
        "ledger_path": publisher.ledger,
        "root": capture_root,
        "transport": transport,
        "now": now,
        "on_reserved": publisher.publish_reservation,
    }
    if mode == "pregame":
        receipt = capture_nhl_daily_pregame(**common)
    else:
        receipt = capture_nhl_daily_scores(**common)
    publisher.publish_capture(receipt.path)
    return receipt.path


def main() -> int:
    parser = argparse.ArgumentParser(description="Collecte NHL privée, sans pronostic.")
    parser.add_argument("mode", choices=("pregame", "scores"))
    parser.add_argument("--private-repo", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_private_capture(args.private_repo, args.mode)
    except Exception:
        # Neither request URLs nor provider exception text may reach Action logs.
        parser.exit(1, "Collecte NHL interrompue; vérifier le dépôt privé et le quota.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
