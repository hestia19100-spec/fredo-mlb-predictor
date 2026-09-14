"""Décisions documentaires NHL-03, sans accès aux fournisseurs.

Les constats sont volontairement conservateurs : une source ne devient jamais
appelable parce qu'elle figure dans cet audit. Une autorisation ultérieure,
distincte et explicite reste obligatoire.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping

from src.nhl.source_registry import PROVIDERS, ProviderId


class NHLSourceAuditError(ValueError):
    """Signale un audit incomplet ou une tentative d'activation prématurée."""


class AuditVerdict(str, Enum):
    BLOCKED = "BLOCKED"
    CONDITIONAL_HISTORICAL_ONLY = "CONDITIONAL_HISTORICAL_ONLY"


@dataclass(frozen=True, slots=True)
class OfficialDocument:
    url: str
    accessed_on: date
    supports: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.url.startswith("https://") or "." not in self.url[8:]:
            raise NHLSourceAuditError("Une preuve doit utiliser une URL HTTPS.")
        if type(self.accessed_on) is not date:
            raise NHLSourceAuditError("accessed_on doit être une date.")
        if not self.supports or any(not item.strip() for item in self.supports):
            raise NHLSourceAuditError("Chaque document doit soutenir un constat.")


@dataclass(frozen=True, slots=True)
class ProviderAudit:
    provider_id: ProviderId
    verdict: AuditVerdict
    automated_collection_allowed: bool
    production_use_allowed: bool
    rights_finding: str
    cost_finding: str
    quota_finding: str
    historical_finding: str
    timestamp_finding: str
    blockers: tuple[str, ...]
    next_authorized_step: str
    documents: tuple[OfficialDocument, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, ProviderId):
            raise NHLSourceAuditError("provider_id est invalide.")
        if self.automated_collection_allowed or self.production_use_allowed:
            raise NHLSourceAuditError("NHL-03 ne peut activer aucune source.")
        findings = (
            self.rights_finding,
            self.cost_finding,
            self.quota_finding,
            self.historical_finding,
            self.timestamp_finding,
            self.next_authorized_step,
        )
        if any(not value.strip() for value in findings):
            raise NHLSourceAuditError("Tous les axes d'audit sont obligatoires.")
        if not self.blockers or len(set(self.blockers)) != len(self.blockers):
            raise NHLSourceAuditError("Les blocages doivent être explicites.")
        if not self.documents:
            raise NHLSourceAuditError("Une décision doit citer un document officiel.")


_AUDIT_DATE = date(2026, 9, 14)
_NHL_TERMS = OfficialDocument(
    url="https://www.nhl.com/info/terms-of-service",
    accessed_on=_AUDIT_DATE,
    supports=(
        "AUTOMATED_EXTRACTION_REQUIRES_AUTHORIZATION",
        "CONTENT_USE_IS_RESTRICTED",
    ),
)
_SPORTSDATA_ACCESS = OfficialDocument(
    url="https://sportsdata.io/developers",
    accessed_on=_AUDIT_DATE,
    supports=(
        "DISCOVERY_LAB_IS_NEXT_DAY_DELAYED",
        "PERSONAL_TIERS_HAVE_DAILY_CALL_LIMITS",
        "REAL_TIME_REQUIRES_COMMERCIAL_AGREEMENT",
    ),
)
_SPORTSDATA_RIGHTS = OfficialDocument(
    url="https://sportsdata.io/help/data-rights-and-licensing-questions",
    accessed_on=_AUDIT_DATE,
    supports=(
        "PERSONAL_NONCOMMERCIAL_USE_AVAILABLE",
        "MODEL_AND_STORAGE_RIGHTS_DEPEND_ON_LICENSE_SCOPE",
    ),
)
_SPORTSDATA_DICTIONARY = OfficialDocument(
    url="https://sportsdata.io/developers/data-dictionary/nhl",
    accessed_on=_AUDIT_DATE,
    supports=("SOURCE_TIMESTAMPS_REQUIRE_FIELD_LEVEL_AUDIT",),
)
_MONEYPUCK_DATA = OfficialDocument(
    url="https://www.moneypuck.com/data.htm",
    accessed_on=_AUDIT_DATE,
    supports=(
        "DOWNLOADS_PERMITTED_FOR_NONCOMMERCIAL_USE_WITH_ATTRIBUTION",
        "HISTORICAL_SHOTS_AND_XG_AVAILABLE_FROM_2007_2008",
        "UNAPPROVED_SCRAPING_IS_FORBIDDEN",
    ),
)


AUDITS: tuple[ProviderAudit, ...] = (
    ProviderAudit(
        provider_id=ProviderId.EXPECTED_GOALS_CANDIDATE,
        verdict=AuditVerdict.CONDITIONAL_HISTORICAL_ONLY,
        automated_collection_allowed=False,
        production_use_allowed=False,
        rights_finding="MONEYPUCK_NONCOMMERCIAL_DOWNLOADS_WITH_ATTRIBUTION_ONLY",
        cost_finding="LISTED_DOWNLOADS_FREE_FOR_NONCOMMERCIAL_USE",
        quota_finding="NO_API_QUOTA_DOCUMENTED_FOR_LISTED_DOWNLOAD_FILES",
        historical_finding="SHOT_AND_XG_HISTORY_LISTED_FROM_2007_2008",
        timestamp_finding="NIGHTLY_UPDATE_STATED_BUT_ROW_AVAILABILITY_NOT_PROVEN",
        blockers=(
            "PROSPECTIVE_ROW_TIMESTAMPS_NOT_PROVEN",
            "REPRODUCIBLE_VERSIONED_SNAPSHOTS_NOT_PROVEN",
        ),
        next_authorized_step="TEST_DOWNLOADED_SYNTHETIC_SCHEMA_WITHOUT_REMOTE_ACCESS",
        documents=(_MONEYPUCK_DATA,),
    ),
    ProviderAudit(
        provider_id=ProviderId.NHL_PUBLIC_WEB_API,
        verdict=AuditVerdict.BLOCKED,
        automated_collection_allowed=False,
        production_use_allowed=False,
        rights_finding="NO_EXPLICIT_AUTOMATED_DATA_LICENSE_FOUND",
        cost_finding="NO_OFFICIAL_API_PRICING_FOUND",
        quota_finding="NO_OFFICIAL_API_QUOTA_FOUND",
        historical_finding="ENDPOINT_HISTORY_NOT_CONTRACTUALLY_DOCUMENTED",
        timestamp_finding="FIELD_LEVEL_AVAILABILITY_NOT_CONTRACTUALLY_DOCUMENTED",
        blockers=(
            "WRITTEN_AUTOMATION_AUTHORIZATION_MISSING",
            "API_SERVICE_TERMS_MISSING",
        ),
        next_authorized_step="REQUEST_WRITTEN_NHL_DATA_USE_AUTHORIZATION",
        documents=(_NHL_TERMS,),
    ),
    ProviderAudit(
        provider_id=ProviderId.NHL_STATS_REST_API,
        verdict=AuditVerdict.BLOCKED,
        automated_collection_allowed=False,
        production_use_allowed=False,
        rights_finding="NO_EXPLICIT_AUTOMATED_DATA_LICENSE_FOUND",
        cost_finding="NO_OFFICIAL_API_PRICING_FOUND",
        quota_finding="NO_OFFICIAL_API_QUOTA_FOUND",
        historical_finding="ENDPOINT_HISTORY_NOT_CONTRACTUALLY_DOCUMENTED",
        timestamp_finding="HISTORICAL_AS_OF_TIMESTAMPS_NOT_PROVEN",
        blockers=(
            "WRITTEN_AUTOMATION_AUTHORIZATION_MISSING",
            "HISTORICAL_POINT_IN_TIME_REPRODUCIBILITY_NOT_PROVEN",
        ),
        next_authorized_step="REQUEST_WRITTEN_NHL_DATA_USE_AUTHORIZATION",
        documents=(_NHL_TERMS,),
    ),
    ProviderAudit(
        provider_id=ProviderId.SPORTSDATAIO_CANDIDATE,
        verdict=AuditVerdict.CONDITIONAL_HISTORICAL_ONLY,
        automated_collection_allowed=False,
        production_use_allowed=False,
        rights_finding="PERSONAL_USE_EXISTS_BUT_EXACT_FEED_LICENSE_MUST_BE_CONFIRMED",
        cost_finding="DISCOVERY_LAB_LISTS_PAID_PERSONAL_TIERS_AND_FREE_PRIOR_SEASON",
        quota_finding="DISCOVERY_LAB_LISTS_100_TO_1000_CALLS_PER_DAY_BY_TIER",
        historical_finding="FREE_ACCESS_LIMITED_TO_LAST_SEASON",
        timestamp_finding="DISCOVERY_LAB_IS_NEXT_DAY_DELAYED_AND_USES_US_EASTERN_TIMES",
        blockers=(
            "NEXT_DAY_DELAY_UNSUITABLE_FOR_PREGAME_STATUS",
            "REAL_TIME_COMMERCIAL_QUOTE_NOT_OBTAINED",
            "FIELD_LEVEL_SOURCE_UPDATE_TIMESTAMPS_NOT_AUDITED",
        ),
        next_authorized_step="OBTAIN_WRITTEN_PERSONAL_USE_AND_FEED_SCOPE_CONFIRMATION",
        documents=(
            _SPORTSDATA_ACCESS,
            _SPORTSDATA_DICTIONARY,
            _SPORTSDATA_RIGHTS,
        ),
    ),
)

if tuple(item.provider_id.value for item in AUDITS) != tuple(
    sorted(item.provider_id.value for item in AUDITS)
):
    raise RuntimeError("Les audits doivent rester triés par fournisseur.")
if {item.provider_id for item in AUDITS} != {
    item.provider_id for item in PROVIDERS
}:
    raise RuntimeError("Chaque fournisseur NHL-02 doit avoir un audit NHL-03.")


def audit_document() -> dict[str, Any]:
    return {
        "audits": [
            {
                "automated_collection_allowed": audit.automated_collection_allowed,
                "blockers": list(audit.blockers),
                "cost_finding": audit.cost_finding,
                "documents": [
                    {
                        "accessed_on": document.accessed_on.isoformat(),
                        "supports": list(document.supports),
                        "url": document.url,
                    }
                    for document in audit.documents
                ],
                "historical_finding": audit.historical_finding,
                "next_authorized_step": audit.next_authorized_step,
                "production_use_allowed": audit.production_use_allowed,
                "provider_id": audit.provider_id.value,
                "quota_finding": audit.quota_finding,
                "rights_finding": audit.rights_finding,
                "timestamp_finding": audit.timestamp_finding,
                "verdict": audit.verdict.value,
            }
            for audit in AUDITS
        ]
    }


def validate_source_audit_protocol(document: Mapping[str, Any]) -> None:
    if not isinstance(document, Mapping):
        raise NHLSourceAuditError("Le protocole d'audit doit être un objet JSON.")
    if document.get("audits") != audit_document()["audits"]:
        raise NHLSourceAuditError("Le protocole diverge des décisions en code.")
    if document.get("status") != "DRAFT_RESEARCH_ONLY_NO_PROVIDER_ENABLED":
        raise NHLSourceAuditError("Le statut de sécurité NHL-03 est invalide.")


def load_and_validate_source_audit_protocol(path: Path) -> Mapping[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_source_audit_protocol(document)
    return document


def require_provider_activation_authorized(provider_id: ProviderId | str) -> None:
    """Refuse toute activation : NHL-03 est seulement un audit documentaire."""

    try:
        canonical = provider_id if isinstance(provider_id, ProviderId) else ProviderId(provider_id)
    except ValueError as error:
        raise NHLSourceAuditError("Fournisseur inconnu.") from error
    audit = next(item for item in AUDITS if item.provider_id is canonical)
    raise NHLSourceAuditError(
        f"Activation interdite pour {canonical.value}: {audit.verdict.value}."
    )
