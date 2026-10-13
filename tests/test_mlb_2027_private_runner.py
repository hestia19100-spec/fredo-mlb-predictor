"""Tests de l'exécuteur privé MLB 2027 (aucun accès réseau/Git réel)."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from src.mlb_2027_candidate_archive import archive_reference_candidate
from src.mlb_2027_observed_source import verify_2027_observed_source
from src.mlb_2027_private_runner import (
    MLB2027PrivateRunError,
    _check_private_checkout,
    _commit_private_archive,
    _periods,
    run_private_reference_day,
)
from src.mlb_2027_reference_predictor import calculate_reference_probabilities
from src.mlb_2027_remote_visibility import RemoteVisibility
from tests.test_mlb_2027_observed_source import TARGET, _observed


ISSUED = datetime(2027, 1, 20, 20, tzinfo=timezone.utc)
COMPLETED = datetime(2027, 1, 20, 20, 30, tzinfo=timezone.utc)
LATE = datetime(2027, 1, 20, 22, 1, tzinfo=timezone.utc)


def _archive(repo: Path):
    source = verify_2027_observed_source(TARGET, (_observed(),))
    calculation = calculate_reference_probabilities(
        TARGET, source.upcoming_games, source.prior_final_games,
        source_receipt_sha256=source.receipt_sha256,
    )
    archive = archive_reference_candidate(
        repo / "mlb_2027_candidates", source, calculation,
        issued_at_utc=ISSUED,
    )
    return source, calculation, archive


class PrivateRunnerTests(unittest.TestCase):
    def test_ranges_cover_2027_without_gaps_and_at_most_31_days(self) -> None:
        ranges = _periods(date(2027, 3, 2))
        self.assertEqual(ranges, (
            (date(2027, 1, 1), date(2027, 1, 31)),
            (date(2027, 2, 1), date(2027, 3, 2)),
        ))
        self.assertTrue(all((end - start).days < 31 for start, end in ranges))
        with self.assertRaises(MLB2027PrivateRunError):
            _periods(date(2026, 10, 13))

    def test_requires_clean_main_of_exact_private_repo(self) -> None:
        with TemporaryDirectory() as folder:
            repo = Path(folder).resolve()
            good = [
                str(repo), "main",
                "https://github.com/hestia19100-spec/fredo-lpf-edge-private-ops.git",
                "",
            ]
            with patch("src.mlb_2027_private_runner._git", side_effect=good):
                self.assertEqual(
                    _check_private_checkout(repo, TARGET),
                    repo / "mlb_2027_candidates",
                )
            bad = good[:2] + ["https://token@github.com/hestia19100-spec/fredo-lpf-edge-private-ops.git"]
            with patch("src.mlb_2027_private_runner._git", side_effect=bad):
                with self.assertRaises(MLB2027PrivateRunError):
                    _check_private_checkout(repo, TARGET)
            dirty = good[:3] + ["?? unrelated.txt"]
            with patch("src.mlb_2027_private_runner._git", side_effect=dirty):
                with self.assertRaises(MLB2027PrivateRunError):
                    _check_private_checkout(repo, TARGET)

    def test_commits_only_exact_sealed_slot_and_pushes_without_force(self) -> None:
        with TemporaryDirectory() as folder:
            repo = Path(folder)
            _, _, archive = _archive(repo)
            slot = archive.path.relative_to(repo).as_posix()
            expected = sorted(
                f"{slot}/{item.name}" for item in archive.path.iterdir()
            )
            calls = []

            def fake_git(root, *args):
                calls.append(args)
                if args[:3] == ("diff", "--cached", "--name-only"):
                    return "\n".join(expected)
                return ""

            with patch("src.mlb_2027_private_runner._git", side_effect=fake_git):
                _commit_private_archive(repo, archive)
            self.assertIn(("add", "--", slot), calls)
            self.assertIn(("push", "origin", "HEAD:refs/heads/main"), calls)
            self.assertFalse(any("force" in str(value) for call in calls for value in call))

    def test_staged_extra_file_blocks_commit_and_push(self) -> None:
        with TemporaryDirectory() as folder:
            repo = Path(folder)
            _, _, archive = _archive(repo)
            calls = []

            def fake_git(root, *args):
                calls.append(args)
                if args[:3] == ("diff", "--cached", "--name-only"):
                    return "unrelated.txt"
                return ""

            with patch("src.mlb_2027_private_runner._git", side_effect=fake_git):
                with self.assertRaises(MLB2027PrivateRunError):
                    _commit_private_archive(repo, archive)
            self.assertFalse(any("commit" in call or "push" in call for call in calls))

    def test_full_chain_records_real_completion_before_private_verification(self) -> None:
        with TemporaryDirectory() as folder:
            repo = Path(folder)
            receipt = _observed()
            proof = RemoteVisibility(
                repository="hestia19100-spec/fredo-lpf-edge-private-ops",
                branch="main",
                path="mlb_2027_candidates/example/candidate.json",
                candidate_sha256="a" * 64,
                github_http_date_utc=COMPLETED,
                response_received_at_utc=COMPLETED,
            )
            with (
                patch("src.mlb_2027_private_runner._check_private_checkout",
                      return_value=repo / "mlb_2027_candidates"),
                patch("src.mlb_2027_private_runner.fetch_schedule_range_observed",
                      return_value=receipt) as fetch,
                patch("src.mlb_2027_private_runner._now_utc",
                      side_effect=[ISSUED, COMPLETED]),
                patch("src.mlb_2027_private_runner._commit_private_archive") as commit,
                patch("src.mlb_2027_private_runner.verify_private_main_visibility",
                      return_value=proof) as verify,
            ):
                result = run_private_reference_day(
                    TARGET, private_repo=repo, github_token="test-token",
                )
            fetch.assert_called_once_with(date(2027, 1, 1), TARGET, ("R",))
            commit.assert_called_once()
            self.assertEqual(verify.call_args.kwargs["completed_at_utc"], COMPLETED)
            self.assertEqual(result.status, "VERIFIED_PRIVATE_ONLY")
            self.assertEqual(result.visibility, proof)

    def test_late_completion_never_pushes_or_verifies(self) -> None:
        with TemporaryDirectory() as folder:
            repo = Path(folder)
            with (
                patch("src.mlb_2027_private_runner._check_private_checkout",
                      return_value=repo / "mlb_2027_candidates"),
                patch("src.mlb_2027_private_runner.fetch_schedule_range_observed",
                      return_value=_observed()),
                patch("src.mlb_2027_private_runner._now_utc",
                      side_effect=[ISSUED, LATE]),
                patch("src.mlb_2027_private_runner._commit_private_archive") as commit,
                patch("src.mlb_2027_private_runner.verify_private_main_visibility") as verify,
            ):
                with self.assertRaises(MLB2027PrivateRunError):
                    run_private_reference_day(
                        TARGET, private_repo=repo, github_token="test-token",
                    )
            commit.assert_not_called()
            verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
