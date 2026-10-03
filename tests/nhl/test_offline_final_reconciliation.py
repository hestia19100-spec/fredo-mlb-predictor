"""NHL-16 offline final-result reconciliation and closed-gate tests."""
from __future__ import annotations

import ast
from dataclasses import replace
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.nhl.offline_final_reconciliation import (
    MARKET, SCHEMA, NHLOfflineFinalError, OfflineFinalArchive,
    OfflineFinalGame, load_offline_final_archive, reconcile_offline_finals,
)
from src.nhl.real_archive_audit import audit_reference_history
from tests.nhl.test_lagged_historical_features import _history, _regular

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "nhl_protocols/data/nhl16_offline_final_reconciliation_protocol_v1.json"
MODULE = ROOT / "src/nhl/offline_final_reconciliation.py"
DAY = date(2022, 1, 2)
GAME_ID = 2021020001
URL = f"https://www.nhl.com/gamecenter/nyr-vs-bos/2022/01/02/{GAME_ID}"


def _tied_history():
    rows = _regular(2021, 1, DAY)
    return _history(tuple(replace(row, goals_for=2, goals_against=2)
                          for row in rows))


def _record(**changes):
    values = {
        "game_id": GAME_ID, "game_date": DAY, "home_team": "BOS",
        "away_team": "NYR", "home_score": 3, "away_score": 2,
        "final_type": "SHOOTOUT", "source_url": URL,
    }
    values.update(changes)
    return OfflineFinalGame(**values)


def _archive(*games):
    return OfflineFinalArchive(
        observed_at_utc=datetime(2026, 10, 3, tzinfo=timezone.utc),
        file_sha256="a" * 64, games=tuple(games),
    )


class OfflineFinalReconciliationTests(unittest.TestCase):
    def test_tied_source_score_matches_final_but_model_stays_blocked(self) -> None:
        history = _tied_history()
        audit = audit_reference_history(history)
        result = reconcile_offline_finals(history, audit, _archive(_record()))
        self.assertEqual(result.matched_final_games, 1)
        self.assertEqual(result.matched_decisive_games, 0)
        self.assertEqual(result.resolved_tied_game_ids, (GAME_ID,))
        self.assertEqual(result.unresolved_tied_game_ids, ())
        self.assertFalse(result.independent_final_origin_verified)
        self.assertFalse(result.historical_pregame_availability_proven)
        self.assertFalse(result.training_permitted)
        self.assertFalse(result.prediction_publication_permitted)
        self.assertEqual(result, reconcile_offline_finals(history, audit,
                                                             _archive(_record())))

    def test_missing_offline_evidence_keeps_tied_match_unresolved(self) -> None:
        history = _tied_history()
        result = reconcile_offline_finals(
            history, audit_reference_history(history), _archive())
        self.assertEqual(result.resolved_tied_game_ids, ())
        self.assertEqual(result.unresolved_tied_game_ids, (GAME_ID,))
        self.assertEqual(result.matched_final_games, 0)

    def test_regulation_or_wrong_final_score_cannot_resolve_tie(self) -> None:
        history = _tied_history()
        audit = audit_reference_history(history)
        for record in (_record(final_type="REGULATION"),
                       _record(home_score=4)):
            with self.subTest(record=record), self.assertRaises(NHLOfflineFinalError):
                reconcile_offline_finals(history, audit, _archive(record))

    def test_decisive_source_needs_exact_final_and_matching_identity(self) -> None:
        history = _history(_regular(2021, 1, DAY))
        audit = audit_reference_history(history)
        final = _record(home_score=2, away_score=1, final_type="REGULATION")
        result = reconcile_offline_finals(history, audit, _archive(final))
        self.assertEqual(result.matched_decisive_games, 1)
        for record in (replace(final, home_score=3),
                       replace(final, home_team="CHI"),
                       replace(final, game_date=date(2022, 1, 3))):
            with self.subTest(record=record), self.assertRaises(NHLOfflineFinalError):
                reconcile_offline_finals(history, audit, _archive(record))
        with self.assertRaises(NHLOfflineFinalError):
            reconcile_offline_finals(history, audit,
                                     _archive(_record(game_id=2021020002,
                                                      source_url=URL[:-1] + "2")))

    def test_duplicate_or_self_certified_evidence_is_rejected(self) -> None:
        with self.assertRaises(NHLOfflineFinalError):
            _archive(_record(), _record())
        with self.assertRaises(NHLOfflineFinalError):
            replace(_archive(_record()), origin_verified=True)
        with self.assertRaises(NHLOfflineFinalError):
            _record(home_score=2)
        with self.assertRaises(NHLOfflineFinalError):
            _record(home_team=None)

    def test_local_json_is_hashed_and_three_way_market_is_rejected(self) -> None:
        document = {
            "schema": SCHEMA, "market": MARKET,
            "observed_at_utc": "2026-10-03T12:00:00Z",
            "games": [{
                "game_id": GAME_ID, "game_date": DAY.isoformat(),
                "home_team": "BOS", "away_team": "NYR",
                "home_score": 3, "away_score": 2,
                "final_type": "SHOOTOUT", "source_url": URL,
            }],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "finals.json"
            raw = json.dumps(document).encode("utf-8")
            path.write_bytes(raw)
            archive = load_offline_final_archive(path)
            self.assertEqual(archive.file_sha256, sha256(raw).hexdigest())
            self.assertEqual(archive.games[0].winner, "BOS")
            document["market"] = "REGULATION_1X2"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(NHLOfflineFinalError):
                load_offline_final_archive(path)
            duplicate = raw.replace(b'"schema":', b'"schema":"extra","schema":', 1)
            path.write_bytes(duplicate)
            with self.assertRaises(NHLOfflineFinalError):
                load_offline_final_archive(path)

    def test_protocol_and_module_keep_provider_model_and_database_off(self) -> None:
        policy = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(policy["stage"], "NHL-16")
        self.assertEqual(policy["target_market"], MARKET)
        for key in ("historical_pregame_availability_proven",
                    "independent_final_origin_verified", "training_permitted",
                    "prediction_publication_permitted", "database_mutation"):
            self.assertIs(policy[key], False, key)
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        imports = {alias.name.split(".")[0] for node in ast.walk(tree)
                   if isinstance(node, (ast.Import, ast.ImportFrom))
                   for alias in node.names}
        self.assertFalse(imports & {"requests", "httpx", "urllib", "sqlite3",
                                    "sklearn", "joblib", "subprocess"})


if __name__ == "__main__":
    unittest.main()
