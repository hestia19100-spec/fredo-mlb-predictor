"""Tests de publication append-only du mode fantome MLB v2."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import src.shadow_prediction as shadow_prediction
from src.shadow_prediction import (
    ShadowPredictionError,
    ShadowPublicationConflictError,
    _fsync_parent_directory,
    _publish_exclusive_verified,
)


class ShadowPredictionPublicationTests(unittest.TestCase):
    """La publication doit etre atomique, verifiee et sans remplacement."""

    def test_publication_writes_exact_bytes_and_returns_their_hash(self) -> None:
        content = b'{"proof":"exact"}\n'

        with TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory)
            destination = parent / "receipt.json"

            published_sha256 = _publish_exclusive_verified(
                destination,
                content,
            )

            self.assertEqual(destination.read_bytes(), content)
            self.assertEqual(
                published_sha256,
                hashlib.sha256(content).hexdigest(),
            )
            self.assertEqual(
                sorted(path.name for path in parent.iterdir()),
                ["receipt.json"],
            )

    def test_existing_destination_is_never_overwritten(self) -> None:
        original = b"original\n"

        with TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory)
            destination = parent / "COMPLETED"
            destination.write_bytes(original)

            with self.assertRaises(ShadowPublicationConflictError):
                _publish_exclusive_verified(destination, b"replacement\n")

            self.assertEqual(destination.read_bytes(), original)
            self.assertEqual(
                sorted(path.name for path in parent.iterdir()),
                ["COMPLETED"],
            )

    def test_missing_parent_is_rejected_without_creating_any_path(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            missing_parent = root / "missing"
            destination = missing_parent / "receipt.json"

            with self.assertRaises(ShadowPredictionError):
                _publish_exclusive_verified(destination, b"proof\n")

            self.assertFalse(missing_parent.exists())

    def test_destination_and_content_types_are_strict(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory)

            with self.assertRaises(ShadowPredictionError):
                _publish_exclusive_verified(  # type: ignore[arg-type]
                    str(parent / "proof.json"),
                    b"proof\n",
                )

            for content in ("proof", bytearray(b"proof"), memoryview(b"proof")):
                with self.subTest(content_type=type(content).__name__):
                    with self.assertRaises(ShadowPredictionError):
                        _publish_exclusive_verified(
                            parent / "proof.json",
                            content,  # type: ignore[arg-type]
                        )

            self.assertEqual(list(parent.iterdir()), [])

    def test_file_is_fsynced_before_it_becomes_the_destination(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "proof.json"
            actual_fsync = os.fsync
            fsync_descriptors: list[int] = []

            def recording_fsync(descriptor: int) -> None:
                fsync_descriptors.append(descriptor)
                actual_fsync(descriptor)

            with patch.object(
                shadow_prediction.os,
                "fsync",
                side_effect=recording_fsync,
            ):
                _publish_exclusive_verified(destination, b"proof\n")

            self.assertGreaterEqual(len(fsync_descriptors), 1)
            self.assertEqual(destination.read_bytes(), b"proof\n")

    def test_posix_parent_directory_is_fsynced_and_closed(self) -> None:
        directory = Path("unused-by-mocks")

        with (
            patch.object(shadow_prediction.os, "name", "posix"),
            patch.object(shadow_prediction.os, "open", return_value=42) as open_mock,
            patch.object(shadow_prediction.os, "fsync") as fsync_mock,
            patch.object(shadow_prediction.os, "close") as close_mock,
        ):
            _fsync_parent_directory(directory)

        expected_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        open_mock.assert_called_once_with(directory, expected_flags)
        fsync_mock.assert_called_once_with(42)
        close_mock.assert_called_once_with(42)

    def test_posix_parent_descriptor_is_closed_when_fsync_fails(self) -> None:
        with (
            patch.object(shadow_prediction.os, "name", "posix"),
            patch.object(shadow_prediction.os, "open", return_value=73),
            patch.object(
                shadow_prediction.os,
                "fsync",
                side_effect=OSError("fsync failure"),
            ),
            patch.object(shadow_prediction.os, "close") as close_mock,
        ):
            with self.assertRaises(OSError):
                _fsync_parent_directory(Path("unused-by-mocks"))

        close_mock.assert_called_once_with(73)

    def test_failed_atomic_link_leaves_no_destination_or_temporary_file(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory)
            destination = parent / "proof.json"

            with patch.object(
                shadow_prediction.os,
                "link",
                side_effect=OSError("link failure"),
            ):
                with self.assertRaises(OSError):
                    _publish_exclusive_verified(destination, b"proof\n")

            self.assertFalse(destination.exists())
            self.assertEqual(list(parent.iterdir()), [])

    def test_failed_file_fsync_leaves_no_destination_or_temporary_file(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory)
            destination = parent / "proof.json"

            with patch.object(
                shadow_prediction.os,
                "fsync",
                side_effect=OSError("file fsync failure"),
            ):
                with self.assertRaises(OSError):
                    _publish_exclusive_verified(destination, b"proof\n")

            self.assertFalse(destination.exists())
            self.assertEqual(list(parent.iterdir()), [])

    def test_failure_after_atomic_link_never_removes_destination(self) -> None:
        content = b"durable proof\n"

        with TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory)
            destination = parent / "proof.json"

            with patch.object(
                shadow_prediction,
                "_fsync_parent_directory",
                side_effect=OSError("directory fsync failure"),
            ):
                with self.assertRaises(OSError):
                    _publish_exclusive_verified(destination, content)

            self.assertTrue(destination.exists())
            self.assertEqual(destination.read_bytes(), content)
            self.assertEqual(
                sorted(path.name for path in parent.iterdir()),
                ["proof.json"],
            )

    def test_corruption_detected_during_reread_keeps_published_evidence(
        self,
    ) -> None:
        content = b"official proof\n"

        with TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "proof.json"
            original_read_bytes = Path.read_bytes

            def corrupted_read(path: Path) -> bytes:
                if path == destination:
                    return b"corrupted reread"
                return original_read_bytes(path)

            with patch.object(Path, "read_bytes", new=corrupted_read):
                with self.assertRaises(ShadowPredictionError):
                    _publish_exclusive_verified(destination, content)

            self.assertTrue(destination.exists())
            self.assertEqual(destination.read_bytes(), content)

    def test_concurrent_publishers_have_exactly_one_winner(self) -> None:
        payloads = [f"candidate-{index}\n".encode("ascii") for index in range(8)]

        with TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory)
            destination = parent / "RESERVED"

            def publish(payload: bytes) -> tuple[str, bytes]:
                try:
                    result = _publish_exclusive_verified(destination, payload)
                    return ("published", result.encode("ascii"))
                except ShadowPublicationConflictError:
                    return ("conflict", payload)

            with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
                outcomes = list(executor.map(publish, payloads))

            self.assertEqual(
                sum(status == "published" for status, _ in outcomes),
                1,
            )
            self.assertEqual(
                sum(status == "conflict" for status, _ in outcomes),
                len(payloads) - 1,
            )
            persisted = destination.read_bytes()
            self.assertIn(persisted, payloads)
            self.assertEqual(
                sorted(path.name for path in parent.iterdir()),
                ["RESERVED"],
            )


if __name__ == "__main__":
    unittest.main()
