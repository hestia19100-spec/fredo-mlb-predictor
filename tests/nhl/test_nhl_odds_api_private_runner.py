"""Offline tests: private Git pushes precede all provider calls."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.nhl_odds_api_private_runner import run_private_capture
from src.nhl_odds_api_quota import SCORES_URL


KEY = "fixture-only-secret"
NOW = datetime(2026, 10, 10, 8, tzinfo=timezone.utc)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


class Response:
    def __init__(self):
        self.status_code = 200
        self.url = SCORES_URL + "?apiKey=" + KEY
        self.headers = {
            "Content-Type": "application/json", "x-requests-last": "2",
            "x-requests-used": "2", "x-requests-remaining": "498",
        }
        self.content = json.dumps([]).encode()


class PrivateRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.remote = self.root / "remote.git"
        self.repo = self.root / "private"
        git("init", "--bare", str(self.remote))
        git("init", "-b", "main", str(self.repo))
        git("-C", str(self.repo), "remote", "add", "origin", str(self.remote))
        (self.repo / "README.md").write_text("Private test repo\n", encoding="utf-8")
        git("-C", str(self.repo), "add", "README.md")
        git("-C", str(self.repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-m", "Initial")
        git("-C", str(self.repo), "push", "origin", "HEAD:refs/heads/main")
        self.initial = git("--git-dir", str(self.remote), "rev-parse", "refs/heads/main")
        self.calls = 0

    def transport(self, url, *, params, timeout, allow_redirects):
        self.calls += 1
        self.assertEqual(url, SCORES_URL)
        self.assertEqual(params["apiKey"], KEY)
        self.assertNotEqual(
            git("--git-dir", str(self.remote), "rev-parse", "refs/heads/main"),
            self.initial,
            "No provider call before a successful remote reservation",
        )
        self.assertIn(
            "state/nhl_daily_quota.sqlite",
            git("--git-dir", str(self.remote), "ls-tree", "-r", "--name-only", "main"),
        )
        return Response()

    def test_scores_capture_pushes_reservation_then_evidence(self):
        with patch("src.nhl_odds_api_daily._read_api_key", return_value=KEY):
            slot = run_private_capture(
                self.repo, "scores", transport=self.transport, now=lambda: NOW,
            )
        self.assertEqual(self.calls, 1)
        self.assertTrue((slot / "COMPLETED").exists())
        self.assertEqual(git("-C", str(self.repo), "status", "--porcelain"), "")
        with sqlite3.connect(self.repo / "state" / "nhl_daily_quota.sqlite") as connection:
            self.assertEqual(connection.execute(
                "SELECT reserved_cost, actual_cost FROM requests"
            ).fetchone(), (2, 2))
        self.assertIn(
            "captures/nhl/scores/",
            git("--git-dir", str(self.remote), "ls-tree", "-r", "--name-only", "main"),
        )
        self.assertNotIn(KEY, git("--git-dir", str(self.remote), "log", "--format=%B"))

    def test_repeated_day_fails_without_second_provider_call(self):
        with patch("src.nhl_odds_api_daily._read_api_key", return_value=KEY):
            run_private_capture(self.repo, "scores", transport=self.transport, now=lambda: NOW)
            fresh_runner = self.root / "fresh-runner"
            git("clone", "-b", "main", str(self.remote), str(fresh_runner))
            with self.assertRaises(Exception):
                run_private_capture(fresh_runner, "scores", transport=self.transport, now=lambda: NOW)
        self.assertEqual(self.calls, 1)

    def test_failed_push_prevents_provider_call(self):
        git("-C", str(self.repo), "remote", "set-url", "origin", str(self.root / "missing.git"))
        with patch("src.nhl_odds_api_daily._read_api_key", return_value=KEY):
            with self.assertRaises(Exception) as caught:
                run_private_capture(self.repo, "scores", transport=self.transport, now=lambda: NOW)
        self.assertEqual(self.calls, 0)
        self.assertNotIn(KEY, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
