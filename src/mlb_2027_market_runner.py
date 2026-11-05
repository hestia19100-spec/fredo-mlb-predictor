"""Exécute une seule collecte de cotes et un candidat MLB 2027 privé.

Le reçu de réservation est poussé avant tout appel payant. Une erreur après
réservation interdit une seconde collecte automatique le même jour.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
import json
from pathlib import Path

from src.mlb_api import fetch_schedule_range_observed
from src.mlb_2027_candidate_archive import archive_reference_candidate
from src.mlb_2027_market_match import match_mlb_2027_market
from src.mlb_2027_observed_source import verify_2027_observed_source
from src.mlb_2027_private_runner import (
    MLB2027PrivateRunError,
    PrivateRunResult,
    _check_private_checkout,
    _commit_private_archive,
    _git,
    _now_utc,
    _periods,
)
from src.mlb_2027_reference_predictor import calculate_reference_probabilities
from src.mlb_2027_remote_visibility import verify_private_main_visibility
from src.odds_api import fetch_mlb_moneyline_odds


def _reserve_one_odds_attempt(
    private_repo: Path,
    target_date: date,
    source_sha256: str,
    reserved_at_utc: datetime,
) -> None:
    """Pousse un verrou privé durable; il n'est jamais effacé par l'automate."""
    root = private_repo / "mlb_2027_odds_attempts"
    day = root / target_date.isoformat()
    if root.is_symlink() or day.exists() or day.is_symlink():
        raise MLB2027PrivateRunError("Collecte de cotes déjà réservée ou chemin non fiable.")
    root.mkdir(exist_ok=True)
    day.mkdir(exist_ok=False)
    receipt = {
        "schema_version": 1,
        "target_date": target_date.isoformat(),
        "source_receipt_sha256": source_sha256,
        "reserved_at_utc": reserved_at_utc.isoformat().replace("+00:00", "Z"),
        "status": "ONE_ATTEMPT_ONLY",
    }
    with (day / "RESERVED.json").open("xb") as output:
        output.write((json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    relative = (day / "RESERVED.json").relative_to(private_repo).as_posix()
    _git(private_repo, "add", "--", relative)
    if _git(private_repo, "diff", "--cached", "--name-only") != relative:
        raise MLB2027PrivateRunError("Réservation de cotes non isolée.")
    _git(
        private_repo, "-c", "user.name=LPF Edge automation",
        "-c", "user.email=lpf-edge-automation@users.noreply.github.com",
        "commit", "-m", f"Réserver une collecte de cotes MLB {target_date.isoformat()} (privé)",
    )
    _git(private_repo, "push", "origin", "HEAD:refs/heads/main")
    local_head = _git(private_repo, "rev-parse", "HEAD")
    remote = _git(private_repo, "ls-remote", "origin", "refs/heads/main")
    if remote.split("\t", 1)[0] != local_head:
        raise MLB2027PrivateRunError("Réservation privée non visible sur main.")


def run_private_market_day(target_date: date, *, private_repo: Path) -> PrivateRunResult:
    """Scelle source, probabilités et cotes; n'émet aucun choix public."""
    import os

    _periods(target_date)
    archive_root = _check_private_checkout(private_repo, target_date)
    private_repo = private_repo.resolve(strict=True)
    token = os.environ.get("GITHUB_TOKEN")
    if type(token) is not str or not token.strip():
        raise MLB2027PrivateRunError("Accès GitHub privé indisponible.")
    if not os.environ.get("THE_ODDS_API_KEY"):
        raise MLB2027PrivateRunError("Clé de cotes absente : aucune collecte lancée.")

    observations = tuple(
        fetch_schedule_range_observed(start, end, ("R",))
        for start, end in _periods(target_date)
    )
    source = verify_2027_observed_source(target_date, observations)
    calculation = calculate_reference_probabilities(
        target_date, source.upcoming_games, source.prior_final_games,
        source_receipt_sha256=source.receipt_sha256,
    )
    now = _now_utc()
    first = calculation.readiness.lead_window.earliest_start_utc
    if not (first - timedelta(minutes=95) <= now < first - timedelta(minutes=65)):
        raise MLB2027PrivateRunError("Hors de la fenêtre sûre H−95 à H−65 : aucun crédit utilisé.")
    _reserve_one_odds_attempt(private_repo, target_date, source.receipt_sha256, now)

    # Un seul appel, sans boucle de reprise. La réservation reste même si l'API échoue.
    odds = fetch_mlb_moneyline_odds()
    market = match_mlb_2027_market(source, odds)
    issued_at = _now_utc()
    archive = archive_reference_candidate(
        archive_root, source, calculation, issued_at_utc=issued_at,
        market=market, odds=odds,
    )
    completed_at = _now_utc()
    if completed_at > calculation.readiness.lead_window.latest_completion_utc:
        raise MLB2027PrivateRunError("Candidat avec cotes terminé après T−60.")
    _commit_private_archive(private_repo, archive)
    visibility = verify_private_main_visibility(
        archive, source, calculation,
        issued_at_utc=issued_at, completed_at_utc=completed_at,
        github_token=token,
    )
    return PrivateRunResult(target_date, archive, completed_at, visibility)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Candidat MLB 2027 avec cotes privées.")
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--private-repo", type=Path, required=True)
    args = parser.parse_args()
    result = run_private_market_day(args.date, private_repo=args.private_repo)
    print(f"{result.target_date.isoformat()}: {result.status}")


if __name__ == "__main__":
    main()
