"""Tests purs des identifiants enregistres du protocole fantome v2."""

from __future__ import annotations

from datetime import date
import math
from pathlib import Path
import socket
import sqlite3
import unittest
from unittest import mock
import urllib.request

from src import shadow_prediction


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
OFFICIAL_DATE = "2026-09-01"
SCHEDULED_START = "2026-09-01T23:10:00Z"
FEATURE_ROW_VALUES_ARGUMENT = (
    "feature_row_values_in_features_columns_exact_order_"
    "excluding_feature_row_sha256"
)

KNOWN_ANSWER_VECTORS: dict[str, tuple[list[object], str]] = {
    "slot_key": (
        ["shadow_slot_v2", SHA_A, OFFICIAL_DATE],
        "534a9e487bb6f9af8faba5682750b636"
        "eee84b07496613571b1cfbeccebc9676",
    ),
    "batch_id": (
        ["shadow_batch_v2", SHA_B, SHA_C, SHA_D],
        "bd6199cc0c8ab005be588fe03103811"
        "a5054b516037594f13d50497a162928bc",
    ),
    "occurrence_key": (
        [
            "shadow_occurrence_v2",
            123456,
            OFFICIAL_DATE,
            SCHEDULED_START,
        ],
        "a0b0c323cb249ffaa96762c36439fec1"
        "d37a7e1d2e345cc7734a3fee6da3a4e1",
    ),
    "occurrence_key_missing_start": (
        ["shadow_occurrence_v2", 123456, OFFICIAL_DATE, None],
        "50e4ab25b98b6d05d9183c680e21f031"
        "fd28f8ecee76c7c4511a3bb4638e8401",
    ),
    "prediction_id": (
        [
            "shadow_prediction_v2",
            SHA_A,
            123456,
            OFFICIAL_DATE,
            SCHEDULED_START,
        ],
        "6b3ab93aedeb2a0b0da158aa9a5f0e510"
        "75cb4560965b1d15d6e1b66242b9f21",
    ),
}


def _feature_row_values() -> list[object]:
    return [
        SHA_E,
        SHA_B,
        123456,
        SHA_F,
        2026,
        OFFICIAL_DATE,
        111,
        222,
        SCHEDULED_START,
        "2026-08-31",
        "2026-08-31",
        "2026-08-31",
        10,
        "0.500000",
        "4.250000",
        "3.750000",
        12,
        "0.583333",
        "4.500000",
        "4.000000",
    ]


def _feature_hash(values: list[object]) -> str:
    return shadow_prediction.build_feature_row_sha256(
        **{FEATURE_ROW_VALUES_ARGUMENT: values}
    )


class ShadowPredictionIdentifierTests(unittest.TestCase):
    """Verrouille les preimages, types et effets de bord des identifiants."""

    def test_canonical_json_bytes_match_registered_rule(self) -> None:
        value = {"z": [True, None, 1.5], "accent": "ete", "a": 2}
        self.assertEqual(
            shadow_prediction._canonical_json_bytes(value),
            b'{"a":2,"accent":"ete","z":[true,null,1.5]}',
        )
        self.assertFalse(
            shadow_prediction._canonical_json_bytes(value).endswith(b"\n")
        )

    def test_canonical_json_bytes_are_utf8_without_ascii_escaping(self) -> None:
        self.assertEqual(
            shadow_prediction._canonical_json_bytes(
                ["cafe\N{LATIN SMALL LETTER E WITH ACUTE}"]
            ),
            '["cafe\N{LATIN SMALL LETTER E WITH ACUTE}"]'.encode("utf-8"),
        )

    def test_canonical_json_rejects_non_json_and_ambiguous_values(self) -> None:
        circular: list[object] = []
        circular.append(circular)
        invalid_values: tuple[object, ...] = (
            math.nan,
            math.inf,
            -math.inf,
            (1, 2),
            {1: "integer key"},
            b"bytes",
            circular,
            "\ud800",
        )
        for value in invalid_values:
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(shadow_prediction.ShadowPredictionError):
                    shadow_prediction._canonical_json_bytes(value)

    def test_every_registered_preimage_has_its_known_hash(self) -> None:
        vectors = dict(KNOWN_ANSWER_VECTORS)
        feature_values = _feature_row_values()
        vectors["feature_row_sha256"] = (
            ["shadow_feature_row_v2", feature_values],
            "98bb3e09b94de8182eb9d28ac3a0ed0"
            "e1978ff8354298587b67a2b0b6616fd47",
        )
        for name, (preimage, expected) in vectors.items():
            with self.subTest(name=name):
                self.assertEqual(
                    shadow_prediction._sha256_identifier(preimage),
                    expected,
                )

    def test_public_builders_match_every_registered_known_answer(self) -> None:
        self.assertEqual(
            shadow_prediction.build_slot_key(
                shadow_protocol_sha256=SHA_A,
                target_official_date=OFFICIAL_DATE,
            ),
            KNOWN_ANSWER_VECTORS["slot_key"][1],
        )
        self.assertEqual(
            shadow_prediction.build_batch_id(
                slot_key=SHA_B,
                execution_manifest_sha256=SHA_C,
                model_artifact_sha256=SHA_D,
            ),
            KNOWN_ANSWER_VECTORS["batch_id"][1],
        )
        self.assertEqual(
            shadow_prediction.build_occurrence_key(
                game_id=123456,
                official_date_at_snapshot=OFFICIAL_DATE,
                scheduled_start_utc_at_snapshot_or_null=SCHEDULED_START,
            ),
            KNOWN_ANSWER_VECTORS["occurrence_key"][1],
        )
        self.assertEqual(
            shadow_prediction.build_occurrence_key(
                game_id=123456,
                official_date_at_snapshot=OFFICIAL_DATE,
                scheduled_start_utc_at_snapshot_or_null=None,
            ),
            KNOWN_ANSWER_VECTORS["occurrence_key_missing_start"][1],
        )
        self.assertEqual(
            shadow_prediction.build_prediction_id(
                shadow_protocol_sha256=SHA_A,
                game_id=123456,
                official_date_at_snapshot=OFFICIAL_DATE,
                scheduled_start_utc_at_snapshot=SCHEDULED_START,
            ),
            KNOWN_ANSWER_VECTORS["prediction_id"][1],
        )
        self.assertEqual(
            _feature_hash(_feature_row_values()),
            "98bb3e09b94de8182eb9d28ac3a0ed0"
            "e1978ff8354298587b67a2b0b6616fd47",
        )

    def test_builders_are_keyword_only(self) -> None:
        with self.assertRaises(TypeError):
            shadow_prediction.build_slot_key(SHA_A, OFFICIAL_DATE)  # type: ignore[misc]

    def test_hash_fields_require_lowercase_sha256_strings(self) -> None:
        invalid_hashes: tuple[object, ...] = (
            "a" * 63,
            "a" * 65,
            "A" * 64,
            "g" * 64,
            True,
            0,
            None,
        )
        for invalid in invalid_hashes:
            with self.subTest(invalid=invalid):
                with self.assertRaises(shadow_prediction.ShadowPredictionError):
                    shadow_prediction.build_slot_key(
                        shadow_protocol_sha256=invalid,  # type: ignore[arg-type]
                        target_official_date=OFFICIAL_DATE,
                    )

        for field in (
            "slot_key",
            "execution_manifest_sha256",
            "model_artifact_sha256",
        ):
            arguments: dict[str, object] = {
                "slot_key": SHA_B,
                "execution_manifest_sha256": SHA_C,
                "model_artifact_sha256": SHA_D,
            }
            arguments[field] = "A" * 64
            with self.subTest(field=field):
                with self.assertRaises(shadow_prediction.ShadowPredictionError):
                    shadow_prediction.build_batch_id(
                        **arguments  # type: ignore[arg-type]
                    )

    def test_game_id_requires_json_integer_and_rejects_boolean(self) -> None:
        for invalid in (True, False, 123456.0, "123456", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(shadow_prediction.ShadowPredictionError):
                    shadow_prediction.build_occurrence_key(
                        game_id=invalid,  # type: ignore[arg-type]
                        official_date_at_snapshot=OFFICIAL_DATE,
                        scheduled_start_utc_at_snapshot_or_null=SCHEDULED_START,
                    )
                with self.assertRaises(shadow_prediction.ShadowPredictionError):
                    shadow_prediction.build_prediction_id(
                        shadow_protocol_sha256=SHA_A,
                        game_id=invalid,  # type: ignore[arg-type]
                        official_date_at_snapshot=OFFICIAL_DATE,
                        scheduled_start_utc_at_snapshot=SCHEDULED_START,
                    )

    def test_every_feature_integer_rejects_boolean_and_float(self) -> None:
        for index in (2, 4, 6, 7, 12, 16):
            for invalid in (True, 1.0, "1"):
                values = _feature_row_values()
                values[index] = invalid
                with self.subTest(index=index, invalid=invalid):
                    with self.assertRaises(
                        shadow_prediction.ShadowPredictionError
                    ):
                        _feature_hash(values)

    def test_dates_require_canonical_valid_json_strings(self) -> None:
        invalid_dates: tuple[object, ...] = (
            "2026-9-01",
            "2026-09-1",
            "2026-02-29",
            "2026-09-01T00:00:00Z",
            " 2026-09-01",
            date(2026, 9, 1),
            True,
            20260901,
            None,
        )
        for invalid in invalid_dates:
            with self.subTest(invalid=invalid):
                with self.assertRaises(shadow_prediction.ShadowPredictionError):
                    shadow_prediction.build_slot_key(
                        shadow_protocol_sha256=SHA_A,
                        target_official_date=invalid,  # type: ignore[arg-type]
                    )
                with self.assertRaises(shadow_prediction.ShadowPredictionError):
                    shadow_prediction.build_occurrence_key(
                        game_id=123456,
                        official_date_at_snapshot=invalid,  # type: ignore[arg-type]
                        scheduled_start_utc_at_snapshot_or_null=SCHEDULED_START,
                    )

    def test_every_feature_date_is_validated(self) -> None:
        for index in (5, 9, 10, 11):
            values = _feature_row_values()
            values[index] = "2026-02-29"
            with self.subTest(index=index):
                with self.assertRaises(shadow_prediction.ShadowPredictionError):
                    _feature_hash(values)

    def test_timestamps_require_exact_valid_rfc3339_seconds_z(self) -> None:
        invalid_timestamps: tuple[object, ...] = (
            "2026-09-01T23:10Z",
            "2026-09-01T23:10:00.000Z",
            "2026-09-01T23:10:00+00:00",
            "2026-09-01t23:10:00z",
            "2026-09-01T24:10:00Z",
            "2026-02-29T23:10:00Z",
            True,
            0,
        )
        for invalid in invalid_timestamps:
            with self.subTest(invalid=invalid):
                with self.assertRaises(shadow_prediction.ShadowPredictionError):
                    shadow_prediction.build_occurrence_key(
                        game_id=123456,
                        official_date_at_snapshot=OFFICIAL_DATE,
                        scheduled_start_utc_at_snapshot_or_null=(
                            invalid  # type: ignore[arg-type]
                        ),
                    )
                with self.assertRaises(shadow_prediction.ShadowPredictionError):
                    shadow_prediction.build_prediction_id(
                        shadow_protocol_sha256=SHA_A,
                        game_id=123456,
                        official_date_at_snapshot=OFFICIAL_DATE,
                        scheduled_start_utc_at_snapshot=(
                            invalid  # type: ignore[arg-type]
                        ),
                    )

    def test_null_timestamp_is_allowed_only_for_occurrence(self) -> None:
        shadow_prediction.build_occurrence_key(
            game_id=123456,
            official_date_at_snapshot=OFFICIAL_DATE,
            scheduled_start_utc_at_snapshot_or_null=None,
        )
        with self.assertRaises(shadow_prediction.ShadowPredictionError):
            shadow_prediction.build_prediction_id(
                shadow_protocol_sha256=SHA_A,
                game_id=123456,
                official_date_at_snapshot=OFFICIAL_DATE,
                scheduled_start_utc_at_snapshot=None,  # type: ignore[arg-type]
            )
        values = _feature_row_values()
        values[8] = None
        with self.assertRaises(shadow_prediction.ShadowPredictionError):
            _feature_hash(values)

    def test_feature_row_requires_exactly_twenty_values_in_a_json_list(self) -> None:
        values = _feature_row_values()
        for invalid in (values[:-1], values + ["extra"], tuple(values)):
            with self.subTest(length=len(invalid)):
                with self.assertRaises(shadow_prediction.ShadowPredictionError):
                    _feature_hash(invalid)  # type: ignore[arg-type]

    def test_every_feature_hash_field_is_validated(self) -> None:
        for index in (0, 1, 3):
            values = _feature_row_values()
            values[index] = "A" * 64
            with self.subTest(index=index):
                with self.assertRaises(shadow_prediction.ShadowPredictionError):
                    _feature_hash(values)

    def test_every_fixed_six_field_requires_the_registered_string_format(
        self,
    ) -> None:
        invalid_values: tuple[object, ...] = (
            "0.50000",
            "0.5000000",
            "-0.500000",
            ".500000",
            "1e-6",
            0.5,
            True,
            None,
        )
        for index in (13, 14, 15, 17, 18, 19):
            for invalid in invalid_values:
                values = _feature_row_values()
                values[index] = invalid
                with self.subTest(index=index, invalid=invalid):
                    with self.assertRaises(
                        shadow_prediction.ShadowPredictionError
                    ):
                        _feature_hash(values)

    def test_identifiers_are_deterministic_and_component_sensitive(self) -> None:
        arguments = {
            "shadow_protocol_sha256": SHA_A,
            "game_id": 123456,
            "official_date_at_snapshot": OFFICIAL_DATE,
            "scheduled_start_utc_at_snapshot": SCHEDULED_START,
        }
        first = shadow_prediction.build_prediction_id(**arguments)
        self.assertEqual(
            first,
            shadow_prediction.build_prediction_id(**arguments),
        )
        changed = dict(arguments)
        changed["game_id"] = 123457
        self.assertNotEqual(
            first,
            shadow_prediction.build_prediction_id(**changed),
        )

        row = _feature_row_values()
        original_hash = _feature_hash(row)
        self.assertEqual(original_hash, _feature_hash(list(row)))
        row[19] = "4.000001"
        self.assertNotEqual(original_hash, _feature_hash(row))

    def test_identifier_builders_perform_no_io(self) -> None:
        forbidden = AssertionError("Entree-sortie interdite.")
        with (
            mock.patch("builtins.open", side_effect=forbidden),
            mock.patch("io.open", side_effect=forbidden),
            mock.patch.object(Path, "read_bytes", side_effect=forbidden),
            mock.patch.object(Path, "read_text", side_effect=forbidden),
            mock.patch.object(Path, "write_bytes", side_effect=forbidden),
            mock.patch.object(Path, "write_text", side_effect=forbidden),
            mock.patch.object(Path, "mkdir", side_effect=forbidden),
            mock.patch.object(Path, "touch", side_effect=forbidden),
            mock.patch.object(sqlite3, "connect", side_effect=forbidden),
            mock.patch.object(socket, "create_connection", side_effect=forbidden),
            mock.patch.object(urllib.request, "urlopen", side_effect=forbidden),
        ):
            shadow_prediction.build_slot_key(
                shadow_protocol_sha256=SHA_A,
                target_official_date=OFFICIAL_DATE,
            )
            shadow_prediction.build_batch_id(
                slot_key=SHA_B,
                execution_manifest_sha256=SHA_C,
                model_artifact_sha256=SHA_D,
            )
            shadow_prediction.build_occurrence_key(
                game_id=123456,
                official_date_at_snapshot=OFFICIAL_DATE,
                scheduled_start_utc_at_snapshot_or_null=None,
            )
            shadow_prediction.build_prediction_id(
                shadow_protocol_sha256=SHA_A,
                game_id=123456,
                official_date_at_snapshot=OFFICIAL_DATE,
                scheduled_start_utc_at_snapshot=SCHEDULED_START,
            )
            _feature_hash(_feature_row_values())


if __name__ == "__main__":
    unittest.main()
