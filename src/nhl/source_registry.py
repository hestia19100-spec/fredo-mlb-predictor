"""Registre hors ligne des futures sources de données LPF Edge NHL.

Ce module décrit des fournisseurs candidats sans créer de client HTTP. Tous
les fournisseurs restent volontairement désactivés pendant NHL-02.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Mapping


class NHLSourceRegistryError(ValueError):
    """Signale un registre ou une requête source non conforme."""


class ProviderId(str, Enum):
    NHL_PUBLIC_WEB_API = "nhl_public_web_api"
    NHL_STATS_REST_API = "nhl_stats_rest_api"
    SPORTSDATAIO_CANDIDATE = "sportsdataio_candidate"
    EXPECTED_GOALS_CANDIDATE = "expected_goals_candidate"


class ProviderStatus(str, Enum):
    PLANNED_DISABLED = "PLANNED_DISABLED"
    CANDIDATE_DISABLED = "CANDIDATE_DISABLED"
    DISABLED_PENDING_RIGHTS_REVIEW = "DISABLED_PENDING_RIGHTS_REVIEW"


class SourceCapability(str, Enum):
    SCHEDULE = "SCHEDULE"
    TEAMS = "TEAMS"
    GAME_STATE = "GAME_STATE"
    FINAL_RESULTS = "FINAL_RESULTS"
    TEAM_STATISTICS = "TEAM_STATISTICS"
    PLAYER_STATISTICS = "PLAYER_STATISTICS"
    GOALIE_STATISTICS = "GOALIE_STATISTICS"
    PREGAME_GOALIE = "PREGAME_GOALIE"
    PLAYER_AVAILABILITY = "PLAYER_AVAILABILITY"
    EXPECTED_GOALS = "EXPECTED_GOALS"


class AuthenticationMode(str, Enum):
    NONE_EXPECTED = "NONE_EXPECTED"
    ENVIRONMENT_HEADER_ONLY = "ENVIRONMENT_HEADER_ONLY"
    NOT_DEFINED = "NOT_DEFINED"


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    provider_id: ProviderId
    display_name: str
    status: ProviderStatus
    capabilities: tuple[SourceCapability, ...]
    authentication: AuthenticationMode
    credential_environment_variable: str | None
    historical_availability: str
    licensing_status: str

    def __post_init__(self) -> None:
        if not self.display_name.strip():
            raise NHLSourceRegistryError("Le nom du fournisseur est requis.")
        if not self.capabilities or len(set(self.capabilities)) != len(
            self.capabilities
        ):
            raise NHLSourceRegistryError(
                "Les capacités doivent être uniques et non vides."
            )
        if tuple(sorted(self.capabilities, key=lambda item: item.value)) != (
            self.capabilities
        ):
            raise NHLSourceRegistryError(
                "Les capacités doivent être triées pour rester déterministes."
            )
        if self.authentication is AuthenticationMode.ENVIRONMENT_HEADER_ONLY:
            if not self.credential_environment_variable:
                raise NHLSourceRegistryError(
                    "La variable d'environnement du secret est requise."
                )
        elif self.credential_environment_variable is not None:
            raise NHLSourceRegistryError(
                "Un fournisseur sans secret ne doit pas déclarer de variable."
            )
        if not self.historical_availability or not self.licensing_status:
            raise NHLSourceRegistryError(
                "La disponibilité historique et la licence doivent être explicites."
            )

    @property
    def enabled(self) -> bool:
        return False


def _capabilities(*values: SourceCapability) -> tuple[SourceCapability, ...]:
    return tuple(sorted(values, key=lambda item: item.value))


PROVIDERS: tuple[ProviderDefinition, ...] = (
    ProviderDefinition(
        provider_id=ProviderId.EXPECTED_GOALS_CANDIDATE,
        display_name="Source xG à sélectionner",
        status=ProviderStatus.DISABLED_PENDING_RIGHTS_REVIEW,
        capabilities=_capabilities(SourceCapability.EXPECTED_GOALS),
        authentication=AuthenticationMode.NOT_DEFINED,
        credential_environment_variable=None,
        historical_availability="NOT_AUDITED",
        licensing_status="RIGHTS_AND_REPRODUCIBILITY_NOT_VALIDATED",
    ),
    ProviderDefinition(
        provider_id=ProviderId.NHL_PUBLIC_WEB_API,
        display_name="NHL Public Web API",
        status=ProviderStatus.PLANNED_DISABLED,
        capabilities=_capabilities(
            SourceCapability.FINAL_RESULTS,
            SourceCapability.GAME_STATE,
            SourceCapability.SCHEDULE,
            SourceCapability.TEAMS,
        ),
        authentication=AuthenticationMode.NONE_EXPECTED,
        credential_environment_variable=None,
        historical_availability="TO_AUDIT",
        licensing_status="PUBLIC_DOCUMENTATION_TO_REVIEW",
    ),
    ProviderDefinition(
        provider_id=ProviderId.NHL_STATS_REST_API,
        display_name="NHL Stats REST API",
        status=ProviderStatus.PLANNED_DISABLED,
        capabilities=_capabilities(
            SourceCapability.GOALIE_STATISTICS,
            SourceCapability.PLAYER_STATISTICS,
            SourceCapability.TEAM_STATISTICS,
        ),
        authentication=AuthenticationMode.NONE_EXPECTED,
        credential_environment_variable=None,
        historical_availability="TO_AUDIT",
        licensing_status="PUBLIC_DOCUMENTATION_TO_REVIEW",
    ),
    ProviderDefinition(
        provider_id=ProviderId.SPORTSDATAIO_CANDIDATE,
        display_name="SportsDataIO ou fournisseur équivalent",
        status=ProviderStatus.CANDIDATE_DISABLED,
        capabilities=_capabilities(
            SourceCapability.PLAYER_AVAILABILITY,
            SourceCapability.PREGAME_GOALIE,
        ),
        authentication=AuthenticationMode.ENVIRONMENT_HEADER_ONLY,
        credential_environment_variable="SPORTSDATAIO_API_KEY",
        historical_availability="TO_AUDIT",
        licensing_status="COST_QUOTA_AND_COMMERCIAL_RIGHTS_NOT_VALIDATED",
    ),
)

if tuple(item.provider_id.value for item in PROVIDERS) != tuple(
    sorted(item.provider_id.value for item in PROVIDERS)
):
    raise RuntimeError("Le registre NHL doit rester trié par identifiant.")

_PROVIDER_BY_ID = {item.provider_id: item for item in PROVIDERS}


def get_provider(provider_id: ProviderId | str) -> ProviderDefinition:
    """Retourne un fournisseur enregistré, sans l'activer."""

    try:
        canonical_id = (
            provider_id
            if isinstance(provider_id, ProviderId)
            else ProviderId(provider_id)
        )
        return _PROVIDER_BY_ID[canonical_id]
    except (KeyError, ValueError) as error:
        raise NHLSourceRegistryError(
            f"Fournisseur NHL non enregistré : {provider_id!r}."
        ) from error


def require_enabled_provider(provider_id: ProviderId | str) -> None:
    """Empêche tout appel tant que NHL-02 garde les sources désactivées."""

    provider = get_provider(provider_id)
    raise NHLSourceRegistryError(
        f"Le fournisseur {provider.provider_id.value} est désactivé "
        f"({provider.status.value})."
    )


_OPERATION_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_RESOURCE_PATTERN = re.compile(r"^/[A-Za-z0-9_./{}:-]{1,255}$")
_PARAMETER_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SENSITIVE_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
}


@dataclass(frozen=True, slots=True)
class CanonicalSourceRequest:
    """Description déterministe d'une requête future, sans URL ni secret."""

    provider: ProviderId
    capability: SourceCapability
    operation: str
    resource_path: str
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.provider, ProviderId):
            raise NHLSourceRegistryError("provider doit être un ProviderId.")
        if not isinstance(self.capability, SourceCapability):
            raise NHLSourceRegistryError(
                "capability doit être une SourceCapability."
            )
        provider = get_provider(self.provider)
        if self.capability not in provider.capabilities:
            raise NHLSourceRegistryError(
                "La capacité demandée n'est pas déclarée par le fournisseur."
            )
        if not _OPERATION_PATTERN.fullmatch(self.operation):
            raise NHLSourceRegistryError("operation n'est pas canonique.")
        if not _RESOURCE_PATTERN.fullmatch(self.resource_path):
            raise NHLSourceRegistryError(
                "resource_path doit être un chemin relatif sans hôte."
            )
        if not isinstance(self.parameters, tuple):
            raise NHLSourceRegistryError("parameters doit être un tuple immuable.")
        if tuple(sorted(self.parameters)) != self.parameters:
            raise NHLSourceRegistryError("Les paramètres doivent être triés.")
        names: list[str] = []
        for item in self.parameters:
            if not isinstance(item, tuple) or len(item) != 2:
                raise NHLSourceRegistryError("Chaque paramètre doit être une paire.")
            name, value = item
            if not _PARAMETER_PATTERN.fullmatch(name) or not isinstance(value, str):
                raise NHLSourceRegistryError("Paramètre canonique invalide.")
            if name.lower() in _SENSITIVE_NAMES or any(
                marker in name.lower() for marker in ("secret", "token", "key")
            ):
                raise NHLSourceRegistryError(
                    "Aucune information d'authentification n'est autorisée."
                )
            names.append(name)
        if len(names) != len(set(names)):
            raise NHLSourceRegistryError("Un paramètre ne peut pas être répété.")

    def canonical_document(self) -> dict[str, Any]:
        return {
            "capability": self.capability.value,
            "operation": self.operation,
            "parameters": [list(item) for item in self.parameters],
            "provider": self.provider.value,
            "resource_path": self.resource_path,
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_document(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


def registry_protocol_document() -> dict[str, Any]:
    """Retourne la représentation JSON exacte du registre en code."""

    return {
        "providers": [
            {
                "authentication": item.authentication.value,
                "capabilities": [value.value for value in item.capabilities],
                "credential_environment_variable": (
                    item.credential_environment_variable
                ),
                "display_name": item.display_name,
                "enabled": item.enabled,
                "historical_availability": item.historical_availability,
                "licensing_status": item.licensing_status,
                "provider_id": item.provider_id.value,
                "status": item.status.value,
            }
            for item in PROVIDERS
        ]
    }


def validate_registry_protocol(document: Mapping[str, Any]) -> None:
    """Vérifie que le protocole publié reflète exactement le registre local."""

    if not isinstance(document, Mapping):
        raise NHLSourceRegistryError("Le protocole doit être un objet JSON.")
    if document.get("providers") != registry_protocol_document()["providers"]:
        raise NHLSourceRegistryError(
            "Le protocole des fournisseurs diverge du registre en code."
        )


def load_and_validate_registry_protocol(path: Path) -> Mapping[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_registry_protocol(document)
    return document
