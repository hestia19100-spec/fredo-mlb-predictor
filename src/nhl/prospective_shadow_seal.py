"""Seal a shadow-only NHL observation in the separate private ops checkout.

Never write a per-game probability into the public model repository.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from .database import PROJECT_ROOT
from .prospective_shadow_observation import ProspectiveShadowObservation

PRIVATE_REPOSITORY_NAME = "fredo-lpf-edge-private-ops"


class NHLPrivateShadowSealError(ValueError):
    """The private receipt cannot be written without a safe provenance chain."""


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _validate(observation: ProspectiveShadowObservation) -> dict[str, object]:
    if not isinstance(observation, ProspectiveShadowObservation):
        raise NHLPrivateShadowSealError("Observation NHL absente.")
    record = asdict(observation)
    digest = record.pop("observation_sha256")
    expected = sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    try:
        observed = datetime.fromisoformat(record["observed_at_utc"].replace("Z", "+00:00"))
        cutoff = datetime.fromisoformat(record["information_cutoff_utc"].replace("Z", "+00:00"))
        start = datetime.fromisoformat(record["scheduled_start_utc"].replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise NHLPrivateShadowSealError("Heures de l'observation invalides.") from error
    probabilities = (record["home_win_probability"], record["away_win_probability"])
    if (type(record["game_id"]) is not int or record["game_id"] <= 0
            or any(type(p) is not float or not isfinite(p) or not 0 <= p <= 1
                   for p in probabilities)
            or abs(sum(probabilities) - 1.0) > 1e-12
            or digest != expected
            or not observed < cutoff < start
            or record["status"] != "PRIVATE_RESEARCH_SHADOW_ONLY"
            or record["production_training_permitted"] is not False
            or record["prediction_publication_permitted"] is not False
            or record["betting_recommendation_permitted"] is not False):
        raise NHLPrivateShadowSealError("Observation NHL incohérente ou publiable.")
    return asdict(observation)


def seal_private_shadow_observation(
    observation: ProspectiveShadowObservation, *, private_repository_root: Path,
    explicit_private_seal: bool = False,
) -> Path:
    """Create one immutable local receipt for later private Git publication."""
    if explicit_private_seal is not True:
        raise NHLPrivateShadowSealError("Scellement privé explicite requis.")
    payload = _validate(observation)
    root = Path(private_repository_root)
    if (root.is_symlink() or not root.is_dir()
            or root.name != PRIVATE_REPOSITORY_NAME
            or not (root / ".git").exists()
            or root.resolve().is_relative_to(PROJECT_ROOT.resolve())):
        raise NHLPrivateShadowSealError("Dépôt privé distinct non vérifié.")
    base = root / "captures" / "nhl" / "shadow"
    for part in (root / "captures", root / "captures" / "nhl", base):
        if part.is_symlink():
            raise NHLPrivateShadowSealError("Chemin privé symbolique interdit.")
    day = payload["scheduled_start_utc"][:10]
    slot = base / day / str(payload["game_id"])
    if slot.exists() or slot.is_symlink():
        raise NHLPrivateShadowSealError("Match déjà scellé ou créneau ambigu.")
    try:
        slot.mkdir(parents=True, exist_ok=False)
        with (slot / "observation.json").open("xb") as output:
            output.write(_canonical(payload))
        with (slot / "COMPLETED").open("xb"):
            pass
    except OSError as error:
        raise NHLPrivateShadowSealError("Scellement privé incomplet; ne pas utiliser ce créneau.") from error
    return slot
