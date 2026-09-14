from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.nhl.source_audit import (
    AUDITS,
    AuditVerdict,
    NHLSourceAuditError,
    load_and_validate_source_audit_protocol,
    require_provider_activation_authorized,
    validate_source_audit_protocol,
)
from src.nhl.source_registry import PROVIDERS, ProviderId


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    PROJECT_ROOT / "nhl_protocols" / "data" / "nhl_source_audit_v1.json"
)


class NHLSourceAuditTests(unittest.TestCase):
    def test_protocol_matches_immutable_code_decisions(self) -> None:
        document = load_and_validate_source_audit_protocol(PROTOCOL_PATH)
        self.assertEqual(document["protocol_schema_version"], 1)
        self.assertEqual(
            document["base_repository_commit"],
            "9b0b8d0ab3a3eb50caeee2c792a8d204ca490022",
        )

    def test_every_registered_provider_is_audited_once(self) -> None:
        self.assertEqual(
            [item.provider_id for item in AUDITS],
            sorted(
                (item.provider_id for item in PROVIDERS),
                key=lambda value: value.value,
            ),
        )

    def test_nhl_undocumented_endpoints_remain_blocked(self) -> None:
        by_provider = {item.provider_id: item for item in AUDITS}
        for provider_id in (
            ProviderId.NHL_PUBLIC_WEB_API,
            ProviderId.NHL_STATS_REST_API,
        ):
            with self.subTest(provider=provider_id.value):
                audit = by_provider[provider_id]
                self.assertEqual(audit.verdict, AuditVerdict.BLOCKED)
                self.assertIn(
                    "WRITTEN_AUTOMATION_AUTHORIZATION_MISSING",
                    audit.blockers,
                )

    def test_paid_candidate_is_not_suitable_for_live_pregame_on_personal_tier(self) -> None:
        audit = next(
            item
            for item in AUDITS
            if item.provider_id is ProviderId.SPORTSDATAIO_CANDIDATE
        )
        self.assertEqual(
            audit.verdict,
            AuditVerdict.CONDITIONAL_HISTORICAL_ONLY,
        )
        self.assertIn("NEXT_DAY_DELAY", audit.timestamp_finding)
        self.assertIn(
            "NEXT_DAY_DELAY_UNSUITABLE_FOR_PREGAME_STATUS",
            audit.blockers,
        )

    def test_xg_candidate_is_limited_to_documented_noncommercial_downloads(self) -> None:
        audit = next(
            item
            for item in AUDITS
            if item.provider_id is ProviderId.EXPECTED_GOALS_CANDIDATE
        )
        self.assertEqual(
            audit.verdict,
            AuditVerdict.CONDITIONAL_HISTORICAL_ONLY,
        )
        self.assertIn("NONCOMMERCIAL", audit.rights_finding)
        self.assertIn("2007_2008", audit.historical_finding)
        self.assertIn("TIMESTAMPS_NOT_PROVEN", audit.blockers[0])

    def test_no_audit_allows_collection_or_production(self) -> None:
        for audit in AUDITS:
            with self.subTest(provider=audit.provider_id.value):
                self.assertFalse(audit.automated_collection_allowed)
                self.assertFalse(audit.production_use_allowed)
                with self.assertRaisesRegex(NHLSourceAuditError, "interdite"):
                    require_provider_activation_authorized(audit.provider_id)

    def test_unknown_provider_cannot_be_activated(self) -> None:
        with self.assertRaisesRegex(NHLSourceAuditError, "inconnu"):
            require_provider_activation_authorized("unknown_provider")

    def test_only_official_primary_documents_support_decisions(self) -> None:
        allowed_hosts = ("www.nhl.com/", "sportsdata.io/", "www.moneypuck.com/")
        for audit in AUDITS:
            for document in audit.documents:
                with self.subTest(
                    provider=audit.provider_id.value,
                    url=document.url,
                ):
                    self.assertTrue(document.url.startswith("https://"))
                    self.assertTrue(
                        any(host in document.url for host in allowed_hosts)
                    )
                    self.assertEqual(document.accessed_on.isoformat(), "2026-09-14")

    def test_separate_validation_is_required_before_any_real_call(self) -> None:
        document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        decision = document["decision"]
        self.assertIsNone(decision["active_provider"])
        self.assertFalse(decision["first_real_call_allowed"])
        self.assertTrue(decision["activation_requires_separate_user_validation"])
        self.assertIsNone(decision["recommended_pregame_provider"])

    def test_tampered_decision_is_rejected(self) -> None:
        document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        document["audits"][0]["production_use_allowed"] = True
        with self.assertRaises(NHLSourceAuditError):
            validate_source_audit_protocol(document)


if __name__ == "__main__":
    unittest.main()
