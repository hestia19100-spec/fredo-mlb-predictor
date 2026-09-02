"""Tests des formats canoniques purs du protocole fantome MLB v2."""

from __future__ import annotations

import builtins
import gzip
import hashlib
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from src.shadow_prediction import (
    ShadowPredictionError,
    _canonical_csv_bytes,
    _canonical_gzip_bytes,
    _canonical_json_file_bytes,
    _format_feature_rate,
    _format_probability_float,
)


class ShadowPredictionSerializationTests(unittest.TestCase):
    """Verifie les octets, les refus et l'absence d'entrees-sorties."""

    def test_json_file_is_utf8_sorted_compact_with_one_trailing_lf(self) -> None:
        payload = {"z": "ete", "a": [1, None]}

        canonical = _canonical_json_file_bytes(payload)

        self.assertEqual(canonical, b'{"a":[1,null],"z":"ete"}\n')
        self.assertTrue(canonical.endswith(b"\n"))
        self.assertFalse(canonical.endswith(b"\n\n"))

    def test_json_file_preserves_utf8_without_ascii_escaping(self) -> None:
        canonical = _canonical_json_file_bytes(
            {"z": "\u00e9t\u00e9", "a": [1, None]}
        )

        self.assertEqual(
            canonical,
            b'{"a":[1,null],"z":"\xc3\xa9t\xc3\xa9"}\n',
        )
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(),
            "c3dab0adfc8a59ebbf3c6267b6f19480818a4b4c579aded7fa8a49848ddc1644",
        )

    def test_json_file_reuses_the_strict_json_value_contract(self) -> None:
        invalid_values = (
            ("tuple",),
            {1: "non-string key"},
            float("nan"),
            float("inf"),
            b"bytes",
        )

        for value in invalid_values:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ShadowPredictionError):
                    _canonical_json_file_bytes(value)

    def test_csv_exact_bytes_follow_registered_dialect(self) -> None:
        canonical = _canonical_csv_bytes(
            ["name", "value", "note"],
            [
                ["Fredo", 7, None],
                ["A, B", 0, "ligne\n2"],
                ['"club"', 11, "\u00e9t\u00e9"],
            ],
        )

        self.assertEqual(
            canonical,
            (
                b"name,value,note\n"
                b"Fredo,7,\n"
                b'"A, B",0,"ligne\n2"\n'
                b'"""club""",11,\xc3\xa9t\xc3\xa9\n'
            ),
        )
        self.assertNotIn(b"\r", canonical)

    def test_csv_header_only_is_terminal_and_deterministic(self) -> None:
        first = _canonical_csv_bytes(["prediction_id", "game_id"], [])
        second = _canonical_csv_bytes(
            ("prediction_id", "game_id"),
            (),
        )

        self.assertEqual(first, b"prediction_id,game_id\n")
        self.assertEqual(second, first)

    def test_csv_null_is_empty_and_zero_is_not_empty(self) -> None:
        canonical = _canonical_csv_bytes(
            ["missing", "zero", "text"],
            [[None, 0, ""]],
        )

        self.assertEqual(canonical, b"missing,zero,text\n,0,\n")

    def test_csv_schema_must_be_nonempty_unique_and_textual(self) -> None:
        invalid_columns = (
            [],
            ["game_id", "game_id"],
            [""],
            [1],
            ["bad\rname"],
            {"unordered"},
        )

        for columns in invalid_columns:
            with self.subTest(columns=repr(columns)):
                with self.assertRaises(ShadowPredictionError):
                    _canonical_csv_bytes(columns, [])  # type: ignore[arg-type]

    def test_csv_rows_must_be_ordered_and_have_exact_width(self) -> None:
        invalid_rows = (
            {("a", "b")},
            [[1]],
            [[1, 2, 3]],
            [{"a": 1, "b": 2}],
            ["ab"],
        )

        for rows in invalid_rows:
            with self.subTest(rows=repr(rows)):
                with self.assertRaises(ShadowPredictionError):
                    _canonical_csv_bytes(["a", "b"], rows)  # type: ignore[arg-type]

    def test_csv_rejects_implicit_or_ambiguous_field_conversion(self) -> None:
        invalid_values = (
            True,
            False,
            0.5,
            float("nan"),
            b"bytes",
            ["nested"],
            {"nested": "mapping"},
        )

        for value in invalid_values:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ShadowPredictionError):
                    _canonical_csv_bytes(["value"], [[value]])

    def test_csv_rejects_carriage_return_anywhere(self) -> None:
        with self.assertRaises(ShadowPredictionError):
            _canonical_csv_bytes(["value"], [["a\rb"]])

    def test_csv_rejects_invalid_utf8_scalar(self) -> None:
        with self.assertRaises(ShadowPredictionError):
            _canonical_csv_bytes(["value"], [["\ud800"]])

    def test_gzip_matches_registered_header_and_known_answer(self) -> None:
        json_bytes = _canonical_json_file_bytes(
            {"z": "\u00e9t\u00e9", "a": [1, None]}
        )

        compressed = _canonical_gzip_bytes(json_bytes)

        self.assertEqual(
            compressed.hex(),
            "1f8b08000000000002ffab564a54b28a36d4c92bcdc989d551aa52b2523a"
            "bcb2e4f04aa55a2e00210ddef91b000000",
        )
        self.assertEqual(
            hashlib.sha256(compressed).hexdigest(),
            "4fd808a058a890adbb39592a2fc421e3b0f48de1e9f3c8d55d46b3c221548360",
        )
        self.assertEqual(gzip.decompress(compressed), json_bytes)

    def test_gzip_is_deterministic_and_payload_sensitive(self) -> None:
        first = _canonical_gzip_bytes(b"proof\n")
        second = _canonical_gzip_bytes(b"proof\n")
        changed = _canonical_gzip_bytes(b"proof!\n")

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(first[0:4], bytes.fromhex("1f8b0800"))
        self.assertEqual(first[4:8], b"\x00\x00\x00\x00")
        self.assertEqual(first[9], 255)

    def test_gzip_accepts_exact_bytes_only(self) -> None:
        for value in (bytearray(b"x"), memoryview(b"x"), "x", None):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ShadowPredictionError):
                    _canonical_gzip_bytes(value)  # type: ignore[arg-type]

        self.assertEqual(gzip.decompress(_canonical_gzip_bytes(b"")), b"")

    def test_feature_rate_uses_exact_fixed_six_format(self) -> None:
        examples = (
            (0, "0.000000"),
            (1, "1.000000"),
            (0.5, "0.500000"),
            (4.25, "4.250000"),
            (1 / 3, "0.333333"),
        )

        for value, expected in examples:
            with self.subTest(value=value):
                self.assertEqual(_format_feature_rate(value), expected)

    def test_feature_rate_rejects_nonfinite_negative_and_ambiguous_values(self) -> None:
        invalid_values = (
            True,
            False,
            -1,
            -0.1,
            float("nan"),
            float("inf"),
            "0.500000",
            None,
        )

        for value in invalid_values:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ShadowPredictionError):
                    _format_feature_rate(value)

    def test_probability_uses_exact_seventeen_significant_digits(self) -> None:
        examples = (
            (0, "0"),
            (1, "1"),
            (0.5, "0.5"),
            (0.1, "0.10000000000000001"),
            (math.nextafter(1.0, 0.0), "0.99999999999999989"),
        )

        for value, expected in examples:
            with self.subTest(value=value):
                self.assertEqual(_format_probability_float(value), expected)

    def test_probability_rejects_out_of_range_nonfinite_and_ambiguous_values(
        self,
    ) -> None:
        invalid_values = (
            True,
            False,
            -1,
            -0.0,
            1.0000000001,
            float("nan"),
            float("inf"),
            "0.5",
            None,
        )

        for value in invalid_values:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ShadowPredictionError):
                    _format_probability_float(value)

    def test_all_serializers_are_pure_and_do_not_use_files(self) -> None:
        def forbidden(*args: object, **kwargs: object) -> object:
            raise AssertionError("entree-sortie fichier interdite")

        with (
            patch.object(builtins, "open", side_effect=forbidden),
            patch.object(Path, "open", side_effect=forbidden),
            patch.object(Path, "read_bytes", side_effect=forbidden),
            patch.object(Path, "write_bytes", side_effect=forbidden),
            patch.object(Path, "mkdir", side_effect=forbidden),
        ):
            json_bytes = _canonical_json_file_bytes({"a": 1})
            csv_bytes = _canonical_csv_bytes(["a"], [[1]])
            gzip_bytes = _canonical_gzip_bytes(json_bytes)

        self.assertEqual(json_bytes, b'{"a":1}\n')
        self.assertEqual(csv_bytes, b"a\n1\n")
        self.assertEqual(gzip.decompress(gzip_bytes), json_bytes)


if __name__ == "__main__":
    unittest.main()
