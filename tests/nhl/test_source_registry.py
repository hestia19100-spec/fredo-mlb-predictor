from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.nhl.source_registry import (
    CanonicalSourceRequest,
    NHLSourceRegistryError,
    PROVIDERS,
    ProviderId,
    SourceCapability,
    get_provider,
    load_and_validate_registry_protocol,
    require_enabled_provider,
    validate_registry_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    PROJECT_ROOT / "nhl_protocols" / "data" / "nhl_source_registry_v1.json"
)


class NHLSourceRegistryTests(unittest.TestCase):
    def test_protocol_and_code_registry_are_identical(self) -> None:
        document = load_and_validate_registry_protocol(PROTOCOL_PATH)
        self.assertEqual(document["protocol_schema_version"], 1)
        self.assertEqual(
            document["status"],
            "DRAFT_OFFLINE_NO_PROVIDER_ENABLED",
        )

    def test_every_provider_is_explicitly_disabled(self) -> None:
        self.assertEqual(len(PROVIDERS), 4)
        self.assertTrue(all(not provider.enabled for provider in PROVIDERS))
        self.assertEqual(
            {provider.provider_id for provider in PROVIDERS},
            set(ProviderId),
        )
        for provider in PROVIDERS:
            with self.subTest(provider=provider.provider_id.value):
                with self.assertRaises(NHLSourceRegistryError):
                    require_enabled_provider(provider.provider_id)

    def test_required_capabilities_are_declared(self) -> None:
        public = get_provider(ProviderId.NHL_PUBLIC_WEB_API)
        self.assertTrue(
            {
                SourceCapability.SCHEDULE,
                SourceCapability.TEAMS,
                SourceCapability.GAME_STATE,
                SourceCapability.FINAL_RESULTS,
            }
            <= set(public.capabilities)
        )
        stats = get_provider(ProviderId.NHL_STATS_REST_API)
        self.assertTrue(
            {
                SourceCapability.TEAM_STATISTICS,
                SourceCapability.PLAYER_STATISTICS,
                SourceCapability.GOALIE_STATISTICS,
            }
            <= set(stats.capabilities)
        )
        context = get_provider(ProviderId.SPORTSDATAIO_CANDIDATE)
        self.assertTrue(
            {
                SourceCapability.PREGAME_GOALIE,
                SourceCapability.PLAYER_AVAILABILITY,
            }
            <= set(context.capabilities)
        )

    def test_request_is_deterministic_and_contains_no_host_or_secret(self) -> None:
        request = CanonicalSourceRequest(
            provider=ProviderId.NHL_PUBLIC_WEB_API,
            capability=SourceCapability.SCHEDULE,
            operation="schedule_by_date",
            resource_path="/v1/schedule/{date}",
            parameters=(("date", "2026-10-01"),),
        )
        self.assertEqual(request.canonical_bytes(), request.canonical_bytes())
        self.assertNotIn(b"http", request.canonical_bytes().lower())
        self.assertNotIn(b"secret", request.canonical_bytes().lower())

    def test_unknown_provider_wrong_capability_and_secret_are_rejected(self) -> None:
        with self.assertRaises(NHLSourceRegistryError):
            get_provider("unknown_provider")
        with self.assertRaises(NHLSourceRegistryError):
            CanonicalSourceRequest(
                provider=ProviderId.NHL_PUBLIC_WEB_API,
                capability=SourceCapability.PREGAME_GOALIE,
                operation="goalies",
                resource_path="/goalies",
            )
        with self.assertRaises(NHLSourceRegistryError):
            CanonicalSourceRequest(
                provider=ProviderId.NHL_PUBLIC_WEB_API,
                capability=SourceCapability.SCHEDULE,
                operation="schedule",
                resource_path="/schedule",
                parameters=(("api_key", "forbidden"),),
            )

    def test_protocol_requires_source_audits_before_selection(self) -> None:
        document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        gate = document["provider_selection_gate"]
        self.assertFalse(gate["selection_frozen"])
        self.assertTrue(gate["cost_audit_required"])
        self.assertTrue(gate["quota_audit_required"])
        self.assertTrue(gate["licence_and_commercial_rights_audit_required"])
        self.assertTrue(gate["historical_timestamp_audit_required"])

    def test_divergent_protocol_is_rejected(self) -> None:
        document = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        document["providers"][0]["enabled"] = True
        with self.assertRaises(NHLSourceRegistryError):
            validate_registry_protocol(document)


if __name__ == "__main__":
    unittest.main()
