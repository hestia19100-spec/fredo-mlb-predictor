"""Lecture et normalisation strictement hors ligne de fixtures NHL-02.

Le module ne sait ni appeler un fournisseur, ni écrire en base. Il transforme
uniquement des preuves synthétiques déterministes en observations traçables.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from src.nhl.contracts import SourceEvidence, require_utc
from src.nhl.source_registry import (
    CanonicalSourceRequest,
    ProviderId,
    SourceCapability,
    get_provider,
)


SYNTHETIC_FIXTURE_MARKER = "SYNTHETIC_OFFLINE_FIXTURE"
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class NHLOfflineSourceError(ValueError):
    """Signale une preuve synthétique invalide ou temporellement inutilisable."""


class ValueState(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"


class ObservationKind(str, Enum):
    SCHEDULED_GAME = "SCHEDULED_GAME"
    GAME_STATE = "GAME_STATE"
    FINAL_RESULT = "FINAL_RESULT"
    TEAM_STATISTICS = "TEAM_STATISTICS"
    PLAYER_STATISTICS = "PLAYER_STATISTICS"
    GOALIE_STATISTICS = "GOALIE_STATISTICS"
    PREGAME_GOALIE = "PREGAME_GOALIE"
    PLAYER_AVAILABILITY = "PLAYER_AVAILABILITY"
    EXPECTED_GOALS = "EXPECTED_GOALS"


class TemporalStatus(str, Enum):
    UNASSESSED = "UNASSESSED"
    ELIGIBLE = "ELIGIBLE"
    AFTER_CUTOFF = "AFTER_CUTOFF"


_ALLOWED_KINDS_BY_CAPABILITY: dict[SourceCapability, frozenset[ObservationKind]] = {
    SourceCapability.SCHEDULE: frozenset({ObservationKind.SCHEDULED_GAME}),
    SourceCapability.TEAMS: frozenset({ObservationKind.SCHEDULED_GAME}),
    SourceCapability.GAME_STATE: frozenset({ObservationKind.GAME_STATE}),
    SourceCapability.FINAL_RESULTS: frozenset({ObservationKind.FINAL_RESULT}),
    SourceCapability.TEAM_STATISTICS: frozenset(
        {ObservationKind.TEAM_STATISTICS}
    ),
    SourceCapability.PLAYER_STATISTICS: frozenset(
        {ObservationKind.PLAYER_STATISTICS}
    ),
    SourceCapability.GOALIE_STATISTICS: frozenset(
        {ObservationKind.GOALIE_STATISTICS}
    ),
    SourceCapability.PREGAME_GOALIE: frozenset(
        {ObservationKind.PREGAME_GOALIE}
    ),
    SourceCapability.PLAYER_AVAILABILITY: frozenset(
        {ObservationKind.PLAYER_AVAILABILITY}
    ),
    SourceCapability.EXPECTED_GOALS: frozenset(
        {ObservationKind.EXPECTED_GOALS}
    ),
}


def canonical_json_bytes(value: Any) -> bytes:
    """Encode un document JSON de façon stable et sans espaces variables."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NHLOfflineSourceError("Le contenu n'est pas un JSON canonique.") from error


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_utc_timestamp(value: Any, *, field_name: str) -> datetime:
    """Accepte uniquement une chaîne ISO-8601 explicitement exprimée en UTC."""

    if not isinstance(value, str) or not value.endswith("Z"):
        raise NHLOfflineSourceError(f"{field_name} doit être une chaîne UTC en Z.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise NHLOfflineSourceError(f"{field_name} est invalide.") from error
    if parsed.utcoffset() != timedelta(0):
        raise NHLOfflineSourceError(f"{field_name} doit être en UTC.")
    return parsed


def _required_identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise NHLOfflineSourceError(f"{field_name} n'est pas canonique.")
    return value


def _positive_game_id(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NHLOfflineSourceError(f"{field_name} doit être un entier positif.")
    return value


@dataclass(frozen=True, slots=True)
class OfflineResponseProof:
    fixture_id: str
    provider: ProviderId
    capability: SourceCapability
    request_sha256: str
    response_sha256: str
    observed_at_utc: datetime
    source_updated_at_utc: datetime | None
    code_commit: str

    def __post_init__(self) -> None:
        _required_identifier(self.fixture_id, field_name="fixture_id")
        if not isinstance(self.provider, ProviderId):
            raise NHLOfflineSourceError("provider doit être un ProviderId.")
        if not isinstance(self.capability, SourceCapability):
            raise NHLOfflineSourceError(
                "capability doit être une SourceCapability."
            )
        get_provider(self.provider)
        for field_name, value in (
            ("request_sha256", self.request_sha256),
            ("response_sha256", self.response_sha256),
        ):
            if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
                raise NHLOfflineSourceError(f"{field_name} doit être un SHA-256.")
        if not _GIT_COMMIT_PATTERN.fullmatch(self.code_commit):
            raise NHLOfflineSourceError("code_commit doit être un SHA-1 Git.")
        require_utc(self.observed_at_utc, field_name="observed_at_utc")
        if self.source_updated_at_utc is not None:
            require_utc(
                self.source_updated_at_utc,
                field_name="source_updated_at_utc",
            )

    @property
    def effective_available_at_utc(self) -> datetime:
        if self.source_updated_at_utc is None:
            return self.observed_at_utc
        return max(self.observed_at_utc, self.source_updated_at_utc)


@dataclass(frozen=True, slots=True)
class NormalizedObservation:
    observation_id: str
    kind: ObservationKind
    target_game_id: int
    entity_id: str
    value_state: ValueState
    value: Any
    evidence: SourceEvidence
    fixture_id: str
    request_sha256: str
    source_game_id: int | None = None
    temporal_status: TemporalStatus = TemporalStatus.UNASSESSED

    def __post_init__(self) -> None:
        _required_identifier(self.observation_id, field_name="observation_id")
        if not isinstance(self.kind, ObservationKind):
            raise NHLOfflineSourceError("kind doit être un ObservationKind.")
        _positive_game_id(self.target_game_id, field_name="target_game_id")
        _required_identifier(self.entity_id, field_name="entity_id")
        if not isinstance(self.value_state, ValueState):
            raise NHLOfflineSourceError("value_state doit être explicite.")
        if self.value_state is ValueState.UNKNOWN and self.value is not None:
            raise NHLOfflineSourceError("Une valeur UNKNOWN doit rester nulle.")
        if self.value_state is ValueState.KNOWN and self.value is None:
            raise NHLOfflineSourceError("Une valeur KNOWN ne peut pas être nulle.")
        if self.value is not None:
            canonical_json_bytes(self.value)
        if not isinstance(self.evidence, SourceEvidence):
            raise NHLOfflineSourceError("evidence doit être une SourceEvidence.")
        _required_identifier(self.fixture_id, field_name="fixture_id")
        if not _SHA256_PATTERN.fullmatch(self.request_sha256):
            raise NHLOfflineSourceError("request_sha256 doit être un SHA-256.")
        if self.temporal_status is not TemporalStatus.UNASSESSED:
            raise NHLOfflineSourceError(
                "Une observation brute doit commencer avec le statut UNASSESSED."
            )
        if self.source_game_id is not None:
            _positive_game_id(self.source_game_id, field_name="source_game_id")
            if self.source_game_id == self.target_game_id:
                raise NHLOfflineSourceError(
                    "Le match cible ne peut pas être son propre match source."
                )

    @property
    def effective_available_at_utc(self) -> datetime:
        return self.evidence.effective_available_at_utc


@dataclass(frozen=True, slots=True)
class OfflineFixtureBundle:
    request: CanonicalSourceRequest
    proof: OfflineResponseProof
    observations: tuple[NormalizedObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, CanonicalSourceRequest):
            raise NHLOfflineSourceError("request doit être canonique.")
        if not isinstance(self.proof, OfflineResponseProof):
            raise NHLOfflineSourceError("proof est invalide.")
        if self.request.provider is not self.proof.provider:
            raise NHLOfflineSourceError("Le fournisseur de la preuve diverge.")
        if self.request.capability is not self.proof.capability:
            raise NHLOfflineSourceError("La capacité de la preuve diverge.")
        if sha256_hex(self.request.canonical_bytes()) != self.proof.request_sha256:
            raise NHLOfflineSourceError("L'empreinte de la requête diverge.")
        if not self.observations:
            raise NHLOfflineSourceError("Une fixture doit contenir une observation.")
        identifiers = [item.observation_id for item in self.observations]
        if identifiers != sorted(identifiers) or len(identifiers) != len(
            set(identifiers)
        ):
            raise NHLOfflineSourceError(
                "Les observations doivent être uniques et triées."
            )


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NHLOfflineSourceError(f"{field_name} doit être un objet JSON.")
    return value


def _build_request(document: Mapping[str, Any]) -> CanonicalSourceRequest:
    parameters = document.get("parameters")
    if not isinstance(parameters, list):
        raise NHLOfflineSourceError("request.parameters doit être une liste.")
    try:
        parameter_tuple = tuple(tuple(item) for item in parameters)
        return CanonicalSourceRequest(
            provider=ProviderId(document.get("provider")),
            capability=SourceCapability(document.get("capability")),
            operation=document.get("operation"),
            resource_path=document.get("resource_path"),
            parameters=parameter_tuple,
        )
    except (TypeError, ValueError) as error:
        raise NHLOfflineSourceError("La requête synthétique est invalide.") from error


def _build_observation(
    record: Mapping[str, Any],
    *,
    proof: OfflineResponseProof,
) -> NormalizedObservation:
    allowed_keys = {
        "entity_id",
        "kind",
        "observation_id",
        "source_game_id",
        "source_updated_at_utc",
        "target_game_id",
        "value",
        "value_state",
    }
    if set(record) - allowed_keys:
        raise NHLOfflineSourceError("Une observation contient un champ inattendu.")
    try:
        kind = ObservationKind(record.get("kind"))
        value_state = ValueState(record.get("value_state"))
    except ValueError as error:
        raise NHLOfflineSourceError("Type ou état d'observation inconnu.") from error
    if kind not in _ALLOWED_KINDS_BY_CAPABILITY[proof.capability]:
        raise NHLOfflineSourceError(
            "Le type d'observation ne correspond pas à la capacité source."
        )
    record_updated = record.get("source_updated_at_utc")
    updated_at = (
        proof.source_updated_at_utc
        if record_updated is None
        else parse_utc_timestamp(
            record_updated,
            field_name="record.source_updated_at_utc",
        )
    )
    observation_id = record.get("observation_id")
    evidence = SourceEvidence(
        provider=proof.provider.value,
        observation_id=observation_id,
        observed_at_utc=proof.observed_at_utc,
        response_sha256=proof.response_sha256,
        code_commit=proof.code_commit,
        source_updated_at_utc=updated_at,
    )
    observation = NormalizedObservation(
        observation_id=observation_id,
        kind=kind,
        target_game_id=record.get("target_game_id"),
        entity_id=record.get("entity_id"),
        value_state=value_state,
        value=record.get("value"),
        evidence=evidence,
        fixture_id=proof.fixture_id,
        request_sha256=proof.request_sha256,
        source_game_id=record.get("source_game_id"),
    )
    _validate_observation_semantics(observation)
    return observation


def _require_value_mapping(observation: NormalizedObservation) -> Mapping[str, Any]:
    return _require_mapping(
        observation.value,
        field_name=f"value de {observation.observation_id}",
    )


def _validate_observation_semantics(observation: NormalizedObservation) -> None:
    """Refuse les raccourcis qui pourraient transformer une fuite en donnée."""

    if observation.value_state is ValueState.UNKNOWN:
        return
    value = _require_value_mapping(observation)
    forbidden_target_fields = {"play_by_play", "target_score"}
    if set(value) & forbidden_target_fields:
        raise NHLOfflineSourceError(
            "Le score cible et le play-by-play cible sont interdits."
        )
    if observation.kind is not ObservationKind.FINAL_RESULT and set(value) & {
        "away_score",
        "home_score",
    }:
        raise NHLOfflineSourceError(
            "Un score n'est autorisé que pour un résultat source final."
        )
    if observation.kind is ObservationKind.SCHEDULED_GAME:
        required = {
            "away_team_id",
            "game_state",
            "game_type",
            "home_team_id",
            "official_date",
            "rescheduled",
            "scheduled_start_utc",
            "season_id",
        }
        if not required <= set(value):
            raise NHLOfflineSourceError("Le calendrier canonique est incomplet.")
        parse_utc_timestamp(
            value["scheduled_start_utc"],
            field_name="scheduled_start_utc",
        )
        if value["rescheduled"] and "original_start_utc" not in value:
            raise NHLOfflineSourceError(
                "Un match replanifié doit conserver son horaire initial."
            )
    elif observation.kind is ObservationKind.FINAL_RESULT:
        if observation.source_game_id is None:
            raise NHLOfflineSourceError(
                "Un résultat final doit identifier son match source."
            )
        if value.get("game_state") != "FINAL":
            raise NHLOfflineSourceError("Un résultat utilisé doit être final.")
        final_at = parse_utc_timestamp(
            value.get("final_observed_at_utc"),
            field_name="final_observed_at_utc",
        )
        if final_at > observation.effective_available_at_utc:
            raise NHLOfflineSourceError(
                "Le résultat est déclaré final après sa preuve de disponibilité."
            )
    elif observation.kind in {
        ObservationKind.TEAM_STATISTICS,
        ObservationKind.PLAYER_STATISTICS,
        ObservationKind.GOALIE_STATISTICS,
    } and observation.source_game_id is None:
        raise NHLOfflineSourceError(
            "Une statistique historique doit identifier son match source."
        )
    elif observation.kind is ObservationKind.PREGAME_GOALIE:
        if value.get("status") not in {"PROBABLE", "CONFIRMED"}:
            raise NHLOfflineSourceError(
                "Un gardien connu doit être probable ou confirmé."
            )
        if not isinstance(value.get("goalie_id"), int):
            raise NHLOfflineSourceError("Un gardien connu doit être identifié.")
    elif observation.kind is ObservationKind.PLAYER_AVAILABILITY:
        if value.get("status") not in {
            "AVAILABLE",
            "DOUBTFUL",
            "OUT",
            "PROBABLE",
            "QUESTIONABLE",
        }:
            raise NHLOfflineSourceError("Le statut du joueur est invalide.")


def load_offline_fixture(path: Path) -> OfflineFixtureBundle:
    """Relit une fixture synthétique, vérifie ses preuves et la normalise."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NHLOfflineSourceError("La fixture n'est pas un JSON lisible.") from error
    root = _require_mapping(document, field_name="fixture")
    expected_keys = {
        "capability",
        "code_commit",
        "fixture_id",
        "fixture_marker",
        "fixture_schema_version",
        "observed_at_utc",
        "payload",
        "provider",
        "request",
        "response_sha256",
        "source_updated_at_utc",
    }
    if set(root) != expected_keys:
        raise NHLOfflineSourceError("Les champs de la fixture ne sont pas exacts.")
    if root["fixture_marker"] != SYNTHETIC_FIXTURE_MARKER:
        raise NHLOfflineSourceError("La fixture n'est pas marquée synthétique.")
    if root["fixture_schema_version"] != 1:
        raise NHLOfflineSourceError("Version de fixture non prise en charge.")

    request = _build_request(_require_mapping(root["request"], field_name="request"))
    try:
        provider = ProviderId(root["provider"])
        capability = SourceCapability(root["capability"])
    except ValueError as error:
        raise NHLOfflineSourceError("Fournisseur ou capacité inconnu.") from error
    if request.provider is not provider or request.capability is not capability:
        raise NHLOfflineSourceError("La requête diverge de la preuve racine.")

    payload = _require_mapping(root["payload"], field_name="payload")
    records = payload.get("records")
    if set(payload) != {"records"} or not isinstance(records, list):
        raise NHLOfflineSourceError("payload.records doit être l'unique contenu.")
    response_sha256 = root["response_sha256"]
    if response_sha256 != sha256_hex(canonical_json_bytes(payload)):
        raise NHLOfflineSourceError("L'empreinte de la réponse est invalide.")

    source_updated_raw = root["source_updated_at_utc"]
    proof = OfflineResponseProof(
        fixture_id=root["fixture_id"],
        provider=provider,
        capability=capability,
        request_sha256=sha256_hex(request.canonical_bytes()),
        response_sha256=response_sha256,
        observed_at_utc=parse_utc_timestamp(
            root["observed_at_utc"],
            field_name="observed_at_utc",
        ),
        source_updated_at_utc=(
            None
            if source_updated_raw is None
            else parse_utc_timestamp(
                source_updated_raw,
                field_name="source_updated_at_utc",
            )
        ),
        code_commit=root["code_commit"],
    )
    observations = tuple(
        _build_observation(
            _require_mapping(record, field_name="record"),
            proof=proof,
        )
        for record in records
    )
    return OfflineFixtureBundle(
        request=request,
        proof=proof,
        observations=observations,
    )


def validate_observation_for_cutoff(
    observation: NormalizedObservation,
    *,
    information_cutoff_utc: datetime,
    scheduled_start_utc: datetime,
) -> None:
    """Applique la règle centrale NHL-02 sans fabriquer de valeur manquante."""

    require_utc(information_cutoff_utc, field_name="information_cutoff_utc")
    require_utc(scheduled_start_utc, field_name="scheduled_start_utc")
    status = assess_observation_for_cutoff(
        observation,
        information_cutoff_utc=information_cutoff_utc,
        scheduled_start_utc=scheduled_start_utc,
    )
    if information_cutoff_utc >= scheduled_start_utc:
        raise NHLOfflineSourceError(
            "Le cutoff doit précéder strictement le début du match."
        )
    if status is TemporalStatus.AFTER_CUTOFF:
        raise NHLOfflineSourceError(
            "L'observation n'était pas disponible au cutoff."
        )


def assess_observation_for_cutoff(
    observation: NormalizedObservation,
    *,
    information_cutoff_utc: datetime,
    scheduled_start_utc: datetime,
) -> TemporalStatus:
    """Produit un statut explicite sans altérer la preuve normalisée."""

    require_utc(information_cutoff_utc, field_name="information_cutoff_utc")
    require_utc(scheduled_start_utc, field_name="scheduled_start_utc")
    if information_cutoff_utc >= scheduled_start_utc:
        raise NHLOfflineSourceError(
            "Le cutoff doit précéder strictement le début du match."
        )
    if observation.effective_available_at_utc > information_cutoff_utc:
        return TemporalStatus.AFTER_CUTOFF
    return TemporalStatus.ELIGIBLE


def eligible_observations(
    observations: tuple[NormalizedObservation, ...],
    *,
    target_game_id: int,
    information_cutoff_utc: datetime,
    scheduled_start_utc: datetime,
) -> tuple[NormalizedObservation, ...]:
    """Retourne uniquement les observations du match cible disponibles à temps."""

    _positive_game_id(target_game_id, field_name="target_game_id")
    selected: list[NormalizedObservation] = []
    for observation in observations:
        if observation.target_game_id != target_game_id:
            continue
        validate_observation_for_cutoff(
            observation,
            information_cutoff_utc=information_cutoff_utc,
            scheduled_start_utc=scheduled_start_utc,
        )
        selected.append(observation)
    return tuple(selected)
