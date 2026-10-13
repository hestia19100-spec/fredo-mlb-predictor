"""Contrôle indépendant de visibilité d'un candidat MLB 2027 sur GitHub.

La lecture vise exclusivement la branche main du dépôt privé d'exploitation.
Une réponse locale, un SHA de commit ou une heure fournie par l'appelant ne
suffisent pas. Cette vérification ne publie rien et ne choisit aucun pari.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path

import requests

from src.mlb_2027_candidate_archive import (
    CandidateArchive,
    verify_candidate_archive,
)
from src.mlb_2027_observed_source import (
    ObservedSourceBundle,
    candidate_has_prospective_timing,
)
from src.mlb_2027_reference_predictor import ReferenceCalculation


PRIVATE_REPOSITORY = "hestia19100-spec/fredo-lpf-edge-private-ops"
API_VERSION = "2026-03-10"


class MLB2027RemoteError(ValueError):
    """La publication distante ne peut pas être attestée."""


@dataclass(frozen=True, slots=True)
class RemoteVisibility:
    repository: str
    branch: str
    path: str
    candidate_sha256: str
    github_http_date_utc: datetime
    response_received_at_utc: datetime


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime, label: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise MLB2027RemoteError(f"{label} doit être UTC.")
    return value


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MLB2027RemoteError("Clé JSON GitHub dupliquée.")
        result[key] = value
    return result


def verify_private_main_visibility(
    archive: CandidateArchive,
    source: ObservedSourceBundle,
    calculation: ReferenceCalculation,
    *,
    issued_at_utc: datetime,
    completed_at_utc: datetime,
    github_token: str,
) -> RemoteVisibility:
    """Lit GitHub lui-même et exige les octets exacts visibles avant T-60.

    Le jeton ne figure jamais dans l'URL ni dans le reçu retourné. Cette
    méthode suppose que l'orchestrateur fournit une vraie heure de fin de
    calcul; elle ne certifie pas à elle seule l'origine de cette heure.
    """
    if (
        type(archive) is not CandidateArchive
        or type(source) is not ObservedSourceBundle
        or type(calculation) is not ReferenceCalculation
        or type(github_token) is not str
        or not github_token.strip()
    ):
        raise MLB2027RemoteError("Archive, calcul ou accès privé manquant.")
    issued = _utc(issued_at_utc, "émission")
    completed = _utc(completed_at_utc, "fin du calcul")
    if issued > completed:
        raise MLB2027RemoteError("Chronologie locale inversée.")
    window = calculation.readiness.lead_window
    if completed > window.latest_completion_utc:
        raise MLB2027RemoteError("Calcul terminé après T-60.")
    verify_candidate_archive(
        archive.path, expected_manifest_sha256=archive.manifest_sha256,
    )
    expected_path = Path(source.target_date.isoformat()) / source.receipt_sha256
    if archive.path.parent.name != expected_path.parent.name or archive.path.name != expected_path.name:
        raise MLB2027RemoteError("Créneau ne correspond pas à la source.")
    candidate_bytes = (archive.path / "candidate.json").read_bytes()
    try:
        candidate = json.loads(candidate_bytes, object_pairs_hook=_object_without_duplicates)
    except (UnicodeError, ValueError, TypeError) as error:
        raise MLB2027RemoteError("Candidat local illisible.") from error
    if (
        candidate.get("issued_at_utc") != issued.isoformat().replace("+00:00", "Z")
        or candidate.get("target_date") != source.target_date.isoformat()
        or candidate.get("source_receipt_sha256") != source.receipt_sha256
    ):
        raise MLB2027RemoteError("Candidat et horodatage local non liés.")

    path = (
        f"mlb_2027_candidates/{source.target_date.isoformat()}/"
        f"{source.receipt_sha256}/candidate.json"
    )
    url = f"https://api.github.com/repos/{PRIVATE_REPOSITORY}/contents/{path}?ref=main"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": "LPF-Edge-MLB-2027-evidence",
        "Cache-Control": "no-cache",
    }
    try:
        response = requests.get(
            url, headers=headers, timeout=10, allow_redirects=False,
        )
        received = _utc(_now_utc(), "réception GitHub")
    except (requests.RequestException, MLB2027RemoteError) as error:
        raise MLB2027RemoteError("Lecture GitHub impossible.") from error
    if (
        response.status_code != 200
        or response.url != url
        or response.history
        or received > window.latest_remote_publication_utc
    ):
        raise MLB2027RemoteError("Candidat absent de main, redirigé ou lu trop tard.")
    try:
        github_date = parsedate_to_datetime(response.headers["Date"]).astimezone(timezone.utc)
        if (
            github_date.utcoffset() != timedelta(0)
            or abs(received - github_date) > timedelta(minutes=5)
        ):
            raise MLB2027RemoteError("Horloge GitHub incohérente.")
        payload = json.loads(
            response.content.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
        )
        if (
            type(payload) is not dict
            or payload.get("type") != "file"
            or payload.get("path") != path
            or payload.get("encoding") != "base64"
            or type(payload.get("content")) is not str
            or type(payload.get("size")) is not int
            or type(payload.get("sha")) is not str
        ):
            raise MLB2027RemoteError("Objet GitHub inattendu.")
        remote_bytes = base64.b64decode(
            "".join(payload["content"].split()), validate=True,
        )
        git_blob_sha = hashlib.sha1(
            f"blob {len(candidate_bytes)}\0".encode("ascii") + candidate_bytes,
        ).hexdigest()
        if (
            payload["size"] != len(candidate_bytes)
            or payload["sha"] != git_blob_sha
            or remote_bytes != candidate_bytes
        ):
            raise MLB2027RemoteError("Octets distants différents du candidat local.")
    except (KeyError, ValueError, UnicodeError, TypeError, base64.binascii.Error) as error:
        if isinstance(error, MLB2027RemoteError):
            raise
        raise MLB2027RemoteError("Réponse GitHub illisible.") from error
    if not candidate_has_prospective_timing(
        source, calculation,
        issued_at_utc=issued,
        completed_at_utc=completed,
        remote_http_date_utc=github_date,
    ):
        raise MLB2027RemoteError("Publication distante après T-60.")
    return RemoteVisibility(
        repository=PRIVATE_REPOSITORY,
        branch="main",
        path=path,
        candidate_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
        github_http_date_utc=github_date,
        response_received_at_utc=received,
    )
