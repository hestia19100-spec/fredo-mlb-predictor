"""Read-only cumulative coverage of sealed NHL observations and official finals.

This is an evidence ledger, not a predictive-performance report. A final score
cannot be called a successful prediction when no prediction was sealed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from .database import PROJECT_ROOT
from .postgame_reconciliation import verify_postgame_reconciliation
from .prospective_checkpoint import verify_checkpoint

SCHEMA = "nhl34_prospective_coverage_v1"
POLICY = PROJECT_ROOT / "nhl_protocols/data/nhl34_prospective_coverage_v1.json"


class NHLProspectiveCoverageError(ValueError):
    """The selected evidence is missing, duplicated, or inconsistent."""


@dataclass(frozen=True, slots=True)
class ProspectiveEvidenceDay:
    checkpoint_path: Path
    manifest_path: Path | None = None
    capture_slots: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class ProspectiveDayCoverage:
    target_date: str
    checkpoint_sha256: str
    manifest_sha256: str | None
    game_count: int
    verified_final_count: int
    pending_count: int
    pending_game_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProspectiveCoverage:
    days: tuple[ProspectiveDayCoverage, ...]
    game_count: int
    verified_final_count: int
    pending_count: int
    audit_sha256: str
    training_permitted: bool = False
    prediction_publication_permitted: bool = False


def _policy_sha256() -> str:
    try:
        raw = POLICY.read_bytes()
        policy = json.loads(raw)
    except (OSError, ValueError) as error:
        raise NHLProspectiveCoverageError("Protocole NHL-34 illisible.") from error
    required = {
        "schema_version": SCHEMA, "read_only": True,
        "network_calls_allowed": False, "unreconciled_games_remain_pending": True,
        "duplicate_games_rejected": True, "training_permitted": False,
        "prediction_publication_permitted": False, "odds_ingestion": False,
        "mlb_database_mutation": False, "nhl_historical_database_mutation": False,
    }
    if not isinstance(policy, dict) or any(policy.get(k) != v for k, v in required.items()):
        raise NHLProspectiveCoverageError("Protocole NHL-34 incompatible.")
    return sha256(raw).hexdigest()


def audit_prospective_coverage(
    evidence_days: Iterable[ProspectiveEvidenceDay],
) -> ProspectiveCoverage:
    """Recheck source proofs; never infer a result from a schedule or a score page."""
    policy_sha = _policy_sha256()
    rows: list[ProspectiveDayCoverage] = []
    seen_dates: set[str] = set()
    seen_games: set[int] = set()
    seen_checkpoints: set[Path] = set()
    for evidence in evidence_days:
        if not isinstance(evidence, ProspectiveEvidenceDay):
            raise NHLProspectiveCoverageError("Journee de preuve invalide.")
        checkpoint_path = Path(evidence.checkpoint_path)
        resolved_path = checkpoint_path.resolve()
        if resolved_path in seen_checkpoints:
            raise NHLProspectiveCoverageError("Preuve avant match dupliquee.")
        seen_checkpoints.add(resolved_path)
        checkpoint = verify_checkpoint(checkpoint_path)
        target_date = checkpoint["target_date"]
        try:
            if date.fromisoformat(target_date).isoformat() != target_date:
                raise ValueError("Noncanonical date")
        except (TypeError, ValueError) as error:
            raise NHLProspectiveCoverageError("Date NHL non canonique.") from error
        game_ids = {game["game_id"] for game in checkpoint["games"]}
        if (len(game_ids) != len(checkpoint["games"])
                or any(type(game_id) is not int or game_id <= 0
                       for game_id in game_ids)):
            raise NHLProspectiveCoverageError("Match NHL invalide ou duplique.")
        if target_date in seen_dates or seen_games.intersection(game_ids):
            raise NHLProspectiveCoverageError("Journee ou match duplique.")
        seen_dates.add(target_date)
        seen_games.update(game_ids)
        if evidence.manifest_path is None:
            if evidence.capture_slots:
                raise NHLProspectiveCoverageError("Capture sans bilan d'apres-match.")
            final_count = 0
            pending_ids = tuple(sorted(game_ids))
            manifest_sha = None
        else:
            manifest = verify_postgame_reconciliation(
                Path(evidence.manifest_path), checkpoint_path,
                tuple(Path(slot) for slot in evidence.capture_slots),
            )
            if (manifest["target_date"] != target_date
                    or manifest["checkpoint_sha256"] != checkpoint["proof_sha256"]
                    or manifest["game_count"] != len(game_ids)
                    or {game["game_id"] for game in manifest["games"]} != game_ids):
                raise NHLProspectiveCoverageError("Bilan et preuve avant match divergents.")
            final_count = manifest["verified_final_count"]
            pending_ids = tuple(sorted(
                game["game_id"] for game in manifest["games"]
                if game["status"] == "PENDING_OFFICIAL_FINAL"
            ))
            if (final_count + len(pending_ids) != len(game_ids)
                    or len(pending_ids) != manifest["pending_count"]):
                raise NHLProspectiveCoverageError("Couverture finale incoherente.")
            manifest_sha = manifest["report_sha256"]
        rows.append(ProspectiveDayCoverage(
            target_date=target_date, checkpoint_sha256=checkpoint["proof_sha256"],
            manifest_sha256=manifest_sha, game_count=len(game_ids),
            verified_final_count=final_count, pending_count=len(pending_ids),
            pending_game_ids=pending_ids,
        ))
    if not rows:
        raise NHLProspectiveCoverageError("Aucune preuve prospective a auditer.")
    ordered = tuple(sorted(rows, key=lambda row: row.target_date))
    payload = {
        "schema_version": SCHEMA, "policy_sha256": policy_sha,
        "days": [asdict(row) for row in ordered],
    }
    audit_sha = sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return ProspectiveCoverage(
        days=ordered, game_count=sum(row.game_count for row in ordered),
        verified_final_count=sum(row.verified_final_count for row in ordered),
        pending_count=sum(row.pending_count for row in ordered),
        audit_sha256=audit_sha,
    )
