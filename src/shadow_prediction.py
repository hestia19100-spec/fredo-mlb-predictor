"""Apercu local et inerte du protocole de prediction fantome MLB v2.

Ce premier jalon ne sait volontairement ni activer ni executer une
prediction. Il ne lit que le protocole JSON fige, puis valide statiquement
une date cible. Il n'importe aucun composant d'ingestion, de base de donnees
ou de modele.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SHADOW_PROTOCOL_RELATIVE_PATH = PurePosixPath(
    "shadow_protocols/logistic_team_form_v1_platt_shadow_v2.json"
)
DEFAULT_SHADOW_PROTOCOL_PATH = PROJECT_DIRECTORY.joinpath(
    *SHADOW_PROTOCOL_RELATIVE_PATH.parts
)

EXPECTED_SHADOW_PROTOCOL_SHA256 = (
    "4dcae9e85bb9ed5b3f4a9f961491d872"
    "56965c42e97d11e70f31f02f52cc49c9"
)
EXPECTED_SHADOW_PROTOCOL_VERSION = 2
EXPECTED_SHADOW_PROTOCOL_STATUS = (
    "REGISTERED_BEFORE_FIRST_SHADOW_V2_PREDICTION"
)
EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON = date(2026, 8, 31)
EXPECTED_TARGET_SEASON = 2026
EXPECTED_PREVIEW_MODE = "PREVIEW_WITHOUT_MODEL_OR_PREDICTIONS"

EXPECTED_SERVICE_MODULE_PATH = "src/shadow_prediction.py"
EXPECTED_MODEL_ARTIFACT_PATH = (
    "models/logistic_team_form_v1_platt.joblib"
)
EXPECTED_MODEL_ARTIFACT_SHA256 = (
    "e0d4d2421ba076072c7ef8b3bc97dd9a"
    "341e26c62828a0ad9ba43f30da15ff55"
)
EXPECTED_ARTIFACT_MANIFEST_PATH = (
    "model_artifacts/logistic_team_form_v1_platt.json"
)
EXPECTED_ARTIFACT_MANIFEST_SHA256 = (
    "a7375d5376baa043b3365cff713cb717"
    "010ad812efee472b594fa454306c96ae"
)
EXPECTED_MODEL_PROTOCOL_PATH = "model_protocols/logistic_team_form_v1.json"
EXPECTED_MODEL_PROTOCOL_SHA256 = (
    "c4cb1af750619967514d37ae3a5a47a6"
    "a04255aeaccb20c5e94533dc4d138451"
)

_CANONICAL_DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")


class ShadowPredictionError(RuntimeError):
    """Erreur qui interdit de produire meme un apercu fiable."""


@dataclass(frozen=True, slots=True)
class ShadowPredictionPreview:
    """Resultat deterministe d'un controle statique, sans execution."""

    mode: str
    shadow_protocol_version: int
    shadow_protocol_path: str
    shadow_protocol_sha256: str
    shadow_protocol_registered_on: str
    shadow_protocol_status: str
    target_official_date: str
    target_season: int
    target_validation_scope: str
    execution_ready: bool
    execution_manifest_read: bool
    activation_read: bool
    model_artifact_read: bool
    model_deserialized: bool
    sqlite_read: bool
    network_request_performed: bool
    output_slot_reserved: bool
    output_files_created: bool
    predictions_computed: bool

    def to_dict(self) -> dict[str, object]:
        """Retourne une representation JSON sans etat ni horodatage."""
        return asdict(self)

    def to_canonical_json(self) -> str:
        """Serialise l'apercu avec un encodage stable entre deux appels."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Refuse une ambiguite JSON au lieu de conserver la derniere cle."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ShadowPredictionError(
                f"Cle JSON dupliquee dans le protocole v2 : {key!r}."
            )
        result[key] = value
    return result


def _require_mapping(
    value: object,
    *,
    description: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ShadowPredictionError(
            f"Objet JSON attendu pour {description}."
        )
    return value


def _require_exact(
    mapping: Mapping[str, Any],
    key: str,
    expected: object,
    *,
    context: str,
) -> None:
    actual = mapping.get(key)
    if type(actual) is not type(expected) or actual != expected:
        raise ShadowPredictionError(
            f"Valeur v2 invalide pour {context}.{key} : "
            f"{actual!r}, attendu {expected!r}."
        )


def _read_frozen_protocol(
    project_directory: Path,
) -> tuple[dict[str, Any], str]:
    """Lit uniquement le chemin local impose, controle avant decodage."""
    project = Path(project_directory).expanduser().resolve()
    if not project.is_dir():
        raise ShadowPredictionError(
            f"Dossier du projet introuvable : {project}."
        )

    protocol_path = project.joinpath(*SHADOW_PROTOCOL_RELATIVE_PATH.parts)
    if protocol_path.is_symlink():
        raise ShadowPredictionError(
            "Le protocole v2 local ne peut pas etre un lien symbolique : "
            f"{SHADOW_PROTOCOL_RELATIVE_PATH.as_posix()}."
        )
    try:
        protocol_bytes = protocol_path.read_bytes()
    except OSError as error:
        raise ShadowPredictionError(
            "Protocole v2 introuvable au chemin fige "
            f"{SHADOW_PROTOCOL_RELATIVE_PATH.as_posix()}."
        ) from error

    actual_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    if actual_sha256 != EXPECTED_SHADOW_PROTOCOL_SHA256:
        raise ShadowPredictionError(
            "SHA-256 invalide pour le protocole v2 local : "
            f"{actual_sha256}, attendu {EXPECTED_SHADOW_PROTOCOL_SHA256}."
        )

    try:
        decoded = protocol_bytes.decode("utf-8")
        payload = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShadowPredictionError(
            "Le protocole v2 fige n'est pas un objet JSON UTF-8 valide."
        ) from error
    protocol = dict(
        _require_mapping(payload, description="le protocole v2")
    )
    return protocol, actual_sha256


def _validate_protocol_contract(protocol: Mapping[str, Any]) -> None:
    """Valide les verrous necessaires au seul mode apercu."""
    _require_exact(
        protocol,
        "shadow_prediction_protocol_version",
        EXPECTED_SHADOW_PROTOCOL_VERSION,
        context="protocol",
    )
    _require_exact(
        protocol,
        "registered_on",
        EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON.isoformat(),
        context="protocol",
    )
    _require_exact(
        protocol,
        "status",
        EXPECTED_SHADOW_PROTOCOL_STATUS,
        context="protocol",
    )
    _require_exact(
        protocol,
        "first_shadow_prediction_created_at_registration",
        False,
        context="protocol",
    )

    freeze = _require_mapping(
        protocol.get("execution_freeze"),
        description="protocol.execution_freeze",
    )
    _require_exact(
        freeze,
        "shadow_service_module_path_exact",
        EXPECTED_SERVICE_MODULE_PATH,
        context="protocol.execution_freeze",
    )
    _require_exact(
        freeze,
        "shadow_protocol_path_exact",
        SHADOW_PROTOCOL_RELATIVE_PATH.as_posix(),
        context="protocol.execution_freeze",
    )
    _require_exact(
        freeze,
        "default_mode",
        EXPECTED_PREVIEW_MODE,
        context="protocol.execution_freeze",
    )
    for key in (
        "preview_must_not_deserialize_model",
        "preview_must_not_call_predict_proba",
        "preview_must_not_reserve_output_slot",
        "preview_must_not_create_output_files",
    ):
        _require_exact(
            freeze,
            key,
            True,
            context="protocol.execution_freeze",
        )

    operating = _require_mapping(
        protocol.get("operating_mode"),
        description="protocol.operating_mode",
    )
    for key, expected in (
        ("strictly_prospective", True),
        ("historical_backfill_allowed", False),
        ("target_season_exact", EXPECTED_TARGET_SEASON),
        ("future_other_seasons_require_new_registered_protocol", True),
        ("time_basis", "UTC"),
    ):
        _require_exact(
            operating,
            key,
            expected,
            context="protocol.operating_mode",
        )

    candidates = _require_mapping(
        protocol.get("candidate_games"),
        description="protocol.candidate_games",
    )
    _require_exact(
        candidates,
        "season_must_equal",
        EXPECTED_TARGET_SEASON,
        context="protocol.candidate_games",
    )

    lineage = _require_mapping(
        protocol.get("validated_lineage"),
        description="protocol.validated_lineage",
    )
    _validate_declared_file(
        lineage,
        "model_artifact",
        expected_path=EXPECTED_MODEL_ARTIFACT_PATH,
        expected_sha256=EXPECTED_MODEL_ARTIFACT_SHA256,
    )
    _validate_declared_file(
        lineage,
        "artifact_manifest",
        expected_path=EXPECTED_ARTIFACT_MANIFEST_PATH,
        expected_sha256=EXPECTED_ARTIFACT_MANIFEST_SHA256,
    )
    _validate_declared_file(
        lineage,
        "model_protocol",
        expected_path=EXPECTED_MODEL_PROTOCOL_PATH,
        expected_sha256=EXPECTED_MODEL_PROTOCOL_SHA256,
    )
    historical = _require_mapping(
        lineage.get("historical_dataset_lineage_only"),
        description=(
            "protocol.validated_lineage.historical_dataset_lineage_only"
        ),
    )
    _require_exact(
        historical,
        "must_not_be_loaded_for_shadow_prediction",
        True,
        context=(
            "protocol.validated_lineage.historical_dataset_lineage_only"
        ),
    )


def _validate_declared_file(
    lineage: Mapping[str, Any],
    key: str,
    *,
    expected_path: str,
    expected_sha256: str,
) -> None:
    declaration = _require_mapping(
        lineage.get(key),
        description=f"protocol.validated_lineage.{key}",
    )
    context = f"protocol.validated_lineage.{key}"
    _require_exact(
        declaration,
        "path",
        expected_path,
        context=context,
    )
    _require_exact(
        declaration,
        "sha256",
        expected_sha256,
        context=context,
    )


def _parse_target_official_date(value: date | str) -> date:
    """Accepte une date ou une chaine ISO canonique, jamais un instant."""
    if isinstance(value, datetime):
        raise ShadowPredictionError(
            "target_official_date doit etre une date sans heure."
        )
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not _CANONICAL_DATE_PATTERN.fullmatch(
        value
    ):
        raise ShadowPredictionError(
            "target_official_date doit respecter exactement YYYY-MM-DD."
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ShadowPredictionError(
            f"target_official_date invalide : {value!r}."
        ) from error
    if parsed.isoformat() != value:
        raise ShadowPredictionError(
            "target_official_date doit etre une date ISO canonique."
        )
    return parsed


def preview_shadow_prediction(
    target_official_date: date | str,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowPredictionPreview:
    """Valide le protocole et la date sans ouvrir aucune entree officielle."""
    protocol, protocol_sha256 = _read_frozen_protocol(project_directory)
    _validate_protocol_contract(protocol)
    target = _parse_target_official_date(target_official_date)

    if target.year != EXPECTED_TARGET_SEASON:
        raise ShadowPredictionError(
            "Saison cible invalide : "
            f"{target.year}, attendu {EXPECTED_TARGET_SEASON}."
        )
    if target < EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON:
        raise ShadowPredictionError(
            "La date cible ne peut pas preceder l'enregistrement du "
            f"protocole v2 ({EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON})."
        )

    return ShadowPredictionPreview(
        mode=EXPECTED_PREVIEW_MODE,
        shadow_protocol_version=EXPECTED_SHADOW_PROTOCOL_VERSION,
        shadow_protocol_path=SHADOW_PROTOCOL_RELATIVE_PATH.as_posix(),
        shadow_protocol_sha256=protocol_sha256,
        shadow_protocol_registered_on=(
            EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON.isoformat()
        ),
        shadow_protocol_status=EXPECTED_SHADOW_PROTOCOL_STATUS,
        target_official_date=target.isoformat(),
        target_season=target.year,
        target_validation_scope="STATIC_PROTOCOL_ONLY",
        execution_ready=False,
        execution_manifest_read=False,
        activation_read=False,
        model_artifact_read=False,
        model_deserialized=False,
        sqlite_read=False,
        network_request_performed=False,
        output_slot_reserved=False,
        output_files_created=False,
        predictions_computed=False,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Valide localement le protocole fantome MLB v2 et une date "
            "cible. Ce jalon est un apercu sans modele ni prediction."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Affiche un JSON deterministe; aucun mode d'execution n'existe ici."""
    parser = _build_argument_parser()
    parser.add_argument(
        "--target-official-date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Date officielle MLB cible, obligatoirement en saison 2026.",
    )
    arguments = parser.parse_args(argv)
    try:
        preview = preview_shadow_prediction(arguments.target_official_date)
    except ShadowPredictionError as error:
        parser.error(str(error))
    print(preview.to_canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
