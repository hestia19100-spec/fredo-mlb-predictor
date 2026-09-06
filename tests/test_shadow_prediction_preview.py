"""Tests comportementaux du jalon apercu des predictions fantomes v2."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timezone
import io
import json
from pathlib import Path
import shutil
import socket
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest import mock
import urllib.request

from src import shadow_prediction


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    PROJECT_DIRECTORY
    / "shadow_protocols"
    / "logistic_team_form_v1_platt_shadow_v2.json"
)


class ShadowPredictionPreviewTests(unittest.TestCase):
    """Garantit que l'apercu reste local, inerte et reproductible."""

    def test_preview_reads_only_protocol_and_performs_no_forbidden_io(
        self,
    ) -> None:
        real_io_open = io.open
        opened_for_read: list[Path] = []
        forbidden_joblib = types.ModuleType("joblib")
        forbidden_joblib.load = mock.Mock(  # type: ignore[attr-defined]
            side_effect=AssertionError("Lecture joblib interdite.")
        )

        def guarded_open(
            file: object,
            mode: str = "r",
            *args: object,
            **kwargs: object,
        ) -> object:
            if any(marker in mode for marker in ("w", "a", "x", "+")):
                raise AssertionError(f"Ecriture interdite : {file!r}.")
            opened_for_read.append(Path(file).resolve())
            return real_io_open(file, mode, *args, **kwargs)

        with (
            mock.patch("io.open", side_effect=guarded_open),
            mock.patch.object(
                Path,
                "mkdir",
                side_effect=AssertionError("Creation de dossier interdite."),
            ),
            mock.patch.object(
                Path,
                "write_bytes",
                side_effect=AssertionError("Ecriture interdite."),
            ),
            mock.patch.object(
                Path,
                "write_text",
                side_effect=AssertionError("Ecriture interdite."),
            ),
            mock.patch.object(
                Path,
                "touch",
                side_effect=AssertionError("Creation interdite."),
            ),
            mock.patch.dict(
                sys.modules,
                {"joblib": forbidden_joblib},
            ),
            mock.patch.object(
                sqlite3,
                "connect",
                side_effect=AssertionError("Lecture SQLite interdite."),
            ),
            mock.patch.object(
                socket,
                "create_connection",
                side_effect=AssertionError("Reseau interdit."),
            ),
            mock.patch.object(
                urllib.request,
                "urlopen",
                side_effect=AssertionError("Reseau interdit."),
            ),
        ):
            preview = shadow_prediction.preview_shadow_prediction(
                "2026-09-01"
            )

        self.assertEqual(opened_for_read, [PROTOCOL_PATH.resolve()])
        self.assertFalse(preview.execution_ready)
        self.assertFalse(preview.execution_manifest_read)
        self.assertFalse(preview.activation_read)
        self.assertFalse(preview.model_artifact_read)
        self.assertFalse(preview.model_deserialized)
        self.assertFalse(preview.sqlite_read)
        self.assertFalse(preview.network_request_performed)
        self.assertFalse(preview.output_slot_reserved)
        self.assertFalse(preview.output_files_created)
        self.assertFalse(preview.predictions_computed)

    def test_preview_does_not_create_result_directory_or_any_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            protocol_copy = project.joinpath(
                *shadow_prediction.SHADOW_PROTOCOL_RELATIVE_PATH.parts
            )
            protocol_copy.parent.mkdir(parents=True)
            shutil.copyfile(PROTOCOL_PATH, protocol_copy)
            before = self._filesystem_snapshot(project)

            shadow_prediction.preview_shadow_prediction(
                date(2026, 9, 1),
                project_directory=project,
            )

            self.assertEqual(self._filesystem_snapshot(project), before)
            self.assertFalse((project / "shadow_results").exists())

    def test_target_date_must_be_canonical_and_calendar_valid(self) -> None:
        invalid_values: tuple[object, ...] = (
            "2026-9-01",
            "2026-09-1",
            " 2026-09-01",
            "2026-02-29",
            "not-a-date",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            20260901,
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(shadow_prediction.ShadowPredictionError):
                    shadow_prediction.preview_shadow_prediction(value)  # type: ignore[arg-type]

    def test_target_season_is_exactly_2026(self) -> None:
        for value in ("2025-09-01", "2027-04-01"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    shadow_prediction.ShadowPredictionError,
                    "Saison cible invalide",
                ):
                    shadow_prediction.preview_shadow_prediction(value)

    def test_target_date_cannot_precede_protocol_registration(self) -> None:
        with self.assertRaisesRegex(
            shadow_prediction.ShadowPredictionError,
            "preceder l'enregistrement",
        ):
            shadow_prediction.preview_shadow_prediction("2026-08-30")

    def test_missing_or_modified_frozen_protocol_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_project = Path(temporary_directory)
            with self.assertRaisesRegex(
                shadow_prediction.ShadowPredictionError,
                "chemin fige",
            ):
                shadow_prediction.preview_shadow_prediction(
                    "2026-09-01",
                    project_directory=missing_project,
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            modified_project = Path(temporary_directory)
            modified_protocol = modified_project.joinpath(
                *shadow_prediction.SHADOW_PROTOCOL_RELATIVE_PATH.parts
            )
            modified_protocol.parent.mkdir(parents=True)
            modified_protocol.write_bytes(PROTOCOL_PATH.read_bytes() + b" ")
            with self.assertRaisesRegex(
                shadow_prediction.ShadowPredictionError,
                "SHA-256 invalide",
            ):
                shadow_prediction.preview_shadow_prediction(
                    "2026-09-01",
                    project_directory=modified_project,
                )

    def test_preview_and_json_are_deterministic(self) -> None:
        first = shadow_prediction.preview_shadow_prediction("2026-09-01")
        second = shadow_prediction.preview_shadow_prediction("2026-09-01")

        self.assertEqual(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.to_canonical_json(), second.to_canonical_json())
        self.assertEqual(
            json.loads(first.to_canonical_json()),
            first.to_dict(),
        )
        self.assertNotIn("created_at", first.to_canonical_json())
        self.assertNotIn("now", first.to_canonical_json())

    def test_cli_outputs_only_the_same_deterministic_preview(self) -> None:
        outputs: list[str] = []
        for _ in range(2):
            stream = io.StringIO()
            with redirect_stdout(stream):
                return_code = shadow_prediction.main(
                    ["--target-official-date", "2026-09-01"]
                )
            self.assertEqual(return_code, 0)
            outputs.append(stream.getvalue())

        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(
            json.loads(outputs[0]),
            shadow_prediction.preview_shadow_prediction(
                "2026-09-01"
            ).to_dict(),
        )

    def test_cli_activation_cannot_be_mixed_with_a_daily_date(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            shadow_prediction.main(
                [
                    "--target-official-date",
                    "2026-09-01",
                    "--activate-shadow",
                ]
            )
        self.assertIn(
            "--target-official-date ne doit pas etre utilise",
            stderr.getvalue(),
        )

    @staticmethod
    def _filesystem_snapshot(root: Path) -> tuple[tuple[str, bool, bytes], ...]:
        entries: list[tuple[str, bool, bytes]] = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                entries.append((relative, True, b""))
            else:
                entries.append((relative, False, path.read_bytes()))
        return tuple(entries)


if __name__ == "__main__":
    unittest.main()
