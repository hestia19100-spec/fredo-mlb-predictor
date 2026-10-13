"""Tests de lecture GitHub indépendante, sans réseau ni publication réelle."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from email.utils import format_datetime
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from src.mlb_2027_candidate_archive import archive_reference_candidate
from src.mlb_2027_remote_visibility import (
    MLB2027RemoteError,
    PRIVATE_REPOSITORY,
    verify_private_main_visibility,
)
from tests.test_mlb_2027_candidate_archive import _candidate


ISSUED = datetime(2027, 1, 20, 20, tzinfo=timezone.utc)
COMPLETED = datetime(2027, 1, 20, 20, 30, tzinfo=timezone.utc)
REMOTE = datetime(2027, 1, 20, 21, tzinfo=timezone.utc)
RECEIVED = datetime(2027, 1, 20, 21, 0, 1, tzinfo=timezone.utc)


def _response(url: str, path: str, data: bytes, *, when=REMOTE, status=200, history=()):
    git_sha = hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data,
    ).hexdigest()
    body = {
        "type": "file", "path": path, "encoding": "base64",
        "content": base64.b64encode(data).decode("ascii"),
        "size": len(data), "sha": git_sha,
    }
    return SimpleNamespace(
        status_code=status, url=url, history=history,
        headers={"Date": format_datetime(when, usegmt=True)},
        content=json.dumps(body).encode(),
    )


class RemoteVisibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.source, self.calculation = _candidate()
        self.archive = archive_reference_candidate(
            Path(self.folder.name), self.source, self.calculation,
            issued_at_utc=ISSUED,
        )
        self.path = (
            f"mlb_2027_candidates/2027-01-20/"
            f"{self.source.receipt_sha256}/candidate.json"
        )
        self.url = (
            f"https://api.github.com/repos/{PRIVATE_REPOSITORY}/"
            f"contents/{self.path}?ref=main"
        )
        self.data = (self.archive.path / "candidate.json").read_bytes()

    def _check(self):
        return verify_private_main_visibility(
            self.archive, self.source, self.calculation,
            issued_at_utc=ISSUED,
            completed_at_utc=COMPLETED,
            github_token="test-only-token",
        )

    def test_exact_candidate_visible_on_private_main_before_deadline(self) -> None:
        response = _response(self.url, self.path, self.data)
        with patch(
            "src.mlb_2027_remote_visibility.requests.get", return_value=response,
        ) as get, patch(
            "src.mlb_2027_remote_visibility._now_utc", return_value=RECEIVED,
        ):
            result = self._check()
        self.assertEqual(result.candidate_sha256, hashlib.sha256(self.data).hexdigest())
        self.assertEqual(result.repository, PRIVATE_REPOSITORY)
        self.assertEqual(result.github_http_date_utc, REMOTE)
        args, kwargs = get.call_args
        self.assertEqual(args, (self.url,))
        self.assertNotIn("test-only-token", self.url)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-only-token")
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["timeout"], 10)

    def test_rejects_different_bytes_even_when_github_sha_matches_them(self) -> None:
        response = _response(self.url, self.path, b"{}\n")
        with patch(
            "src.mlb_2027_remote_visibility.requests.get", return_value=response,
        ), patch(
            "src.mlb_2027_remote_visibility._now_utc", return_value=RECEIVED,
        ):
            with self.assertRaises(MLB2027RemoteError):
                self._check()

    def test_rejects_redirect_or_wrong_branch_url(self) -> None:
        response = _response(self.url, self.path, self.data, history=(object(),))
        with patch(
            "src.mlb_2027_remote_visibility.requests.get", return_value=response,
        ), patch(
            "src.mlb_2027_remote_visibility._now_utc", return_value=RECEIVED,
        ):
            with self.assertRaises(MLB2027RemoteError):
                self._check()
        response = _response(self.url.replace("main", "other"), self.path, self.data)
        with patch(
            "src.mlb_2027_remote_visibility.requests.get", return_value=response,
        ), patch(
            "src.mlb_2027_remote_visibility._now_utc", return_value=RECEIVED,
        ):
            with self.assertRaises(MLB2027RemoteError):
                self._check()

    def test_rejects_late_http_date_or_late_receipt(self) -> None:
        late = datetime(2027, 1, 20, 22, 1, tzinfo=timezone.utc)
        response = _response(self.url, self.path, self.data, when=late)
        with patch(
            "src.mlb_2027_remote_visibility.requests.get", return_value=response,
        ), patch(
            "src.mlb_2027_remote_visibility._now_utc", return_value=late,
        ):
            with self.assertRaises(MLB2027RemoteError):
                self._check()
        response = _response(self.url, self.path, self.data)
        with patch(
            "src.mlb_2027_remote_visibility.requests.get", return_value=response,
        ), patch(
            "src.mlb_2027_remote_visibility._now_utc", return_value=late,
        ):
            with self.assertRaises(MLB2027RemoteError):
                self._check()

    def test_rejects_issued_time_mismatch_without_network(self) -> None:
        changed = datetime(2027, 1, 20, 20, 1, tzinfo=timezone.utc)
        with patch("src.mlb_2027_remote_visibility.requests.get") as get:
            with self.assertRaises(MLB2027RemoteError):
                verify_private_main_visibility(
                    self.archive, self.source, self.calculation,
                    issued_at_utc=changed, completed_at_utc=COMPLETED,
                    github_token="test-only-token",
                )
        get.assert_not_called()

    def test_rejects_missing_private_token_without_network(self) -> None:
        with patch("src.mlb_2027_remote_visibility.requests.get") as get:
            with self.assertRaises(MLB2027RemoteError):
                verify_private_main_visibility(
                    self.archive, self.source, self.calculation,
                    issued_at_utc=ISSUED, completed_at_utc=COMPLETED,
                    github_token="",
                )
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
