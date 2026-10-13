"""Exécute un candidat MLB 2027 privé, sans publier de pronostic aux abonnés.

L'exécution est à usage unique par journée. Elle échoue sans label de succès si
l'archive n'est pas visible sur main du dépôt privé avant T−60.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess

from src.mlb_api import fetch_schedule_range_observed
from src.mlb_2027_candidate_archive import (
    CandidateArchive,
    archive_reference_candidate,
    verify_candidate_archive,
)
from src.mlb_2027_observed_source import verify_2027_observed_source
from src.mlb_2027_reference_predictor import calculate_reference_probabilities
from src.mlb_2027_remote_visibility import (
    PRIVATE_REPOSITORY,
    RemoteVisibility,
    verify_private_main_visibility,
)


class MLB2027PrivateRunError(ValueError):
    """L'exécution privée n'a pas produit de candidat vérifié."""


@dataclass(frozen=True, slots=True)
class PrivateRunResult:
    target_date: date
    archive: CandidateArchive
    completed_at_utc: datetime
    visibility: RemoteVisibility
    status: str = "VERIFIED_PRIVATE_ONLY"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _git(repo: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        # stderr may contain a credential-bearing remote URL.
        raise MLB2027PrivateRunError("Opération Git privée impossible.") from error
    return completed.stdout.strip()


def _periods(target_date: date) -> tuple[tuple[date, date], ...]:
    if type(target_date) is not date or target_date.year != 2027:
        raise MLB2027PrivateRunError("Seule une journée MLB 2027 est admise.")
    start = date(2027, 1, 1)
    ranges: list[tuple[date, date]] = []
    while start <= target_date:
        end = min(start + timedelta(days=30), target_date)
        ranges.append((start, end))
        start = end + timedelta(days=1)
    return tuple(ranges)


def _check_private_checkout(repo: Path, target_date: date) -> Path:
    if not isinstance(repo, Path) or repo.is_symlink() or not repo.is_dir():
        raise MLB2027PrivateRunError("Dépôt privé local absent.")
    resolved = repo.resolve(strict=True)
    if Path(_git(repo, "rev-parse", "--show-toplevel")).resolve() != resolved:
        raise MLB2027PrivateRunError("La racine du dépôt privé est ambiguë.")
    if _git(repo, "branch", "--show-current") != "main":
        raise MLB2027PrivateRunError("La branche privée doit être main.")
    remote = _git(repo, "remote", "get-url", "origin")
    if remote not in {
        f"https://github.com/{PRIVATE_REPOSITORY}.git",
        f"https://github.com/{PRIVATE_REPOSITORY}",
        f"git@github.com:{PRIVATE_REPOSITORY}.git",
    }:
        raise MLB2027PrivateRunError("Le dépôt distant n'est pas celui attendu.")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise MLB2027PrivateRunError("Le dépôt privé doit être propre.")
    archive_root = resolved / "mlb_2027_candidates"
    day = archive_root / target_date.isoformat()
    if archive_root.is_symlink() or day.exists() or day.is_symlink():
        raise MLB2027PrivateRunError("Journée déjà utilisée ou chemin non fiable.")
    return archive_root


def _commit_private_archive(repo: Path, archive: CandidateArchive) -> None:
    manifest = verify_candidate_archive(
        archive.path, expected_manifest_sha256=archive.manifest_sha256,
    )
    relative = archive.path.relative_to(repo)
    expected = {
        (relative / name).as_posix()
        for name in set(manifest["files"]) | {"manifest.json", "COMPLETED"}
    }
    _git(repo, "add", "--", relative.as_posix())
    staged = set(_git(repo, "diff", "--cached", "--name-only").splitlines())
    if staged != expected:
        raise MLB2027PrivateRunError("Les fichiers préparés ne correspondent pas à l'archive.")
    if _git(repo, "diff", "--name-only"):
        raise MLB2027PrivateRunError("Modifications privées non prévues.")
    _git(
        repo, "-c", "user.name=LPF Edge automation",
        "-c", "user.email=lpf-edge-automation@users.noreply.github.com",
        "commit", "-m", f"Archiver candidat MLB {archive.path.parent.name} (privé)",
    )
    # Un push non fast-forward échoue; aucune réécriture de l'historique.
    _git(repo, "push", "origin", "HEAD:refs/heads/main")


def run_private_reference_day(
    target_date: date,
    *,
    private_repo: Path,
    github_token: str | None = None,
) -> PrivateRunResult:
    """Collecte, calcule, scelle et revérifie une journée sur GitHub privé.

    Le temps de fin est capturé par le processus après l'archive, jamais fourni
    par l'appelant. Un échec peut laisser une archive privée, sans statut validé.
    """
    _periods(target_date)
    archive_root = _check_private_checkout(private_repo, target_date)
    token = github_token if github_token is not None else os.environ.get("GITHUB_TOKEN")
    if type(token) is not str or not token.strip():
        raise MLB2027PrivateRunError("Accès GitHub privé indisponible.")
    observations = tuple(
        fetch_schedule_range_observed(start, end, ("R",))
        for start, end in _periods(target_date)
    )
    source = verify_2027_observed_source(target_date, observations)
    issued_at = _now_utc()
    calculation = calculate_reference_probabilities(
        target_date, source.upcoming_games, source.prior_final_games,
        source_receipt_sha256=source.receipt_sha256,
    )
    archive = archive_reference_candidate(
        archive_root, source, calculation, issued_at_utc=issued_at,
    )
    completed_at = _now_utc()
    if completed_at > calculation.readiness.lead_window.latest_completion_utc:
        raise MLB2027PrivateRunError("Calcul ou archivage terminé après T−60.")
    _commit_private_archive(private_repo, archive)
    visibility = verify_private_main_visibility(
        archive, source, calculation,
        issued_at_utc=issued_at,
        completed_at_utc=completed_at,
        github_token=token,
    )
    return PrivateRunResult(
        target_date=target_date, archive=archive,
        completed_at_utc=completed_at, visibility=visibility,
    )
