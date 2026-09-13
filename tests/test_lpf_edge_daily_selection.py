"""Tests du sélecteur prospectif Moneyline LPF Edge."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.lpf_edge_daily_selection import (
    DAILY_SELECTION_ROOT,
    LPFEdgeDailySelectionError,
    POLICY_VERSION,
    create_daily_selection_publication,
    load_daily_selection,
)
from src.lpf_edge_market_snapshot import (
    COMPLETED_SCHEMA as SNAPSHOT_COMPLETED_SCHEMA,
    MARKET_SNAPSHOT_ROOT,
    SNAPSHOT_SCHEMA,
)


TARGET = date(2026, 9, 14)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


class LPFEdgeDailySelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.snapshot_slot = self.project / MARKET_SNAPSHOT_ROOT / TARGET.isoformat()
        self.snapshot_slot.mkdir(parents=True)

    def _row(
        self,
        game_id: int,
        *,
        probability: str,
        market_probability: str | None,
        gap: str | None,
        odds: str | None,
        bookmakers: int = 3,
    ) -> dict[str, object]:
        return {
            "away_team_id": game_id + 10,
            "away_team_name": f"Extérieur {game_id}",
            "best_bookmakers": ["PMU"] if odds is not None else [],
            "best_decimal_odds": odds,
            "bookmaker_count": bookmakers if odds is not None else 0,
            "bookmaker_quotes": [],
            "french_market_probability": market_probability,
            "game_id": game_id,
            "gap_percentage_points": gap,
            "home_team_id": game_id + 20,
            "home_team_name": f"Domicile {game_id}",
            "issued_at_utc": "2026-09-14T12:00:00.000000Z",
            "model_probability": probability,
            "p_away_win": str(1 - float(probability)),
            "p_home_win": probability,
            "predicted_side": "HOME",
            "predicted_team_name": f"Domicile {game_id}",
            "prediction_id": f"prediction-{game_id}",
            "scheduled_start_utc": "2026-09-14T18:00:00.000000Z",
        }

    def _write_snapshot(self, *rows: dict[str, object]) -> str:
        snapshot = {
            "batch_id": "a" * 64,
            "certification_evidence_sha256": "b" * 64,
            "certification_sha256": "c" * 64,
            "certified_at_utc": "2026-09-14T12:01:00.000000Z",
            "comparable_count": sum(row["best_decimal_odds"] is not None for row in rows),
            "odds_completed_at_utc": "2026-09-14T11:59:00.000000Z",
            "odds_region": "fr",
            "odds_run_id": 5,
            "prediction_receipt_sha256": "d" * 64,
            "prediction_results_commit": "e" * 40,
            "predictions": list(rows),
            "predictions_sha256": "f" * 64,
            "schema": SNAPSHOT_SCHEMA,
            "target_official_date": TARGET.isoformat(),
        }
        raw = canonical(snapshot)
        digest = hashlib.sha256(raw).hexdigest()
        marker = {
            "batch_id": "a" * 64,
            "comparable_count": snapshot["comparable_count"],
            "odds_run_id": 5,
            "prediction_results_commit": "e" * 40,
            "row_count": len(rows),
            "schema": SNAPSHOT_COMPLETED_SCHEMA,
            "snapshot_filename": "market_snapshot.json",
            "snapshot_sha256": digest,
            "status": "COMPLETED",
            "target_official_date": TARGET.isoformat(),
        }
        (self.snapshot_slot / "market_snapshot.json").write_bytes(raw)
        (self.snapshot_slot / "COMPLETED").write_bytes(canonical(marker))
        return digest

    def _pitchers(self, *game_ids: int):
        return {
            game_id: (f"Lanceur extérieur {game_id}", f"Lanceur domicile {game_id}")
            for game_id in game_ids
        }

    def test_selects_at_most_two_by_expected_value_then_gap(self) -> None:
        snapshot_sha256 = self._write_snapshot(
            self._row(100, probability="0.62", market_probability="0.56", gap="6", odds="1.80"),
            self._row(101, probability="0.58", market_probability="0.54", gap="4", odds="1.90"),
            self._row(102, probability="0.60", market_probability="0.56", gap="4", odds="1.95"),
        )
        publication = create_daily_selection_publication(
            TARGET,
            probable_pitchers_by_game=self._pitchers(100, 101, 102),
            project_directory=self.project,
        )

        self.assertEqual(publication.market_snapshot_sha256, snapshot_sha256)
        self.assertEqual(publication.selection_count, 2)
        self.assertEqual(publication.eligible_count, 3)
        payload = json.loads(publication.selection_path.read_bytes())
        self.assertEqual(payload["selected_game_ids"], [102, 100])
        self.assertEqual(payload["status"], "PICKS_AVAILABLE")
        self.assertEqual(payload["policy"]["version"], POLICY_VERSION)
        by_game = {row["game_id"]: row for row in payload["decisions"]}
        self.assertEqual(by_game[102]["selection_role"], "PRINCIPAL")
        self.assertEqual(by_game[100]["selection_role"], "SECONDAIRE")
        self.assertEqual(by_game[101]["decision_reasons"], ["LIMITE_DE_DEUX_SELECTIONS"])

    def test_exact_policy_boundaries_are_eligible(self) -> None:
        self._write_snapshot(
            self._row(
                100,
                probability="0.52",
                market_probability="0.50",
                gap="2.0",
                odds="1.980769230769230769230769231",
                bookmakers=2,
            )
        )
        publication = create_daily_selection_publication(
            TARGET,
            probable_pitchers_by_game=self._pitchers(100),
            project_directory=self.project,
        )
        self.assertEqual(publication.selection_count, 1)
        self.assertEqual(publication.eligible_count, 1)

    def test_missing_pitcher_produces_no_pick(self) -> None:
        self._write_snapshot(
            self._row(100, probability="0.62", market_probability="0.56", gap="6", odds="1.80")
        )
        publication = create_daily_selection_publication(
            TARGET,
            probable_pitchers_by_game={100: ("Extérieur", None)},
            project_directory=self.project,
        )
        payload = json.loads(publication.selection_path.read_bytes())
        self.assertEqual(publication.selection_count, 0)
        self.assertEqual(payload["status"], "NO_PICK")
        self.assertIn("LANCEURS_INCOMPLETS", payload["decisions"][0]["decision_reasons"])

    def test_weak_edge_or_value_produces_no_pick(self) -> None:
        self._write_snapshot(
            self._row(100, probability="0.55", market_probability="0.54", gap="1", odds="1.70")
        )
        publication = create_daily_selection_publication(
            TARGET,
            probable_pitchers_by_game=self._pitchers(100),
            project_directory=self.project,
        )
        payload = json.loads(publication.selection_path.read_bytes())
        reasons = payload["decisions"][0]["decision_reasons"]
        self.assertIn("ECART_LPF_MARCHE_INSUFFISANT", reasons)
        self.assertIn("VALEUR_THEORIQUE_INSUFFISANTE", reasons)
        self.assertEqual(payload["selection_count"], 0)
        selection = load_daily_selection(TARGET, project_directory=self.project)
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(
            dict(selection.rejection_reasons),
            {
                "ECART_LPF_MARCHE_INSUFFISANT": 1,
                "VALEUR_THEORIQUE_INSUFFISANTE": 1,
            },
        )

    def test_loader_returns_explainable_picks(self) -> None:
        self._write_snapshot(
            self._row(100, probability="0.62", market_probability="0.56", gap="6", odds="1.80")
        )
        publication = create_daily_selection_publication(
            TARGET,
            probable_pitchers_by_game=self._pitchers(100),
            project_directory=self.project,
        )
        selection = load_daily_selection(TARGET, project_directory=self.project)
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.selection_sha256, publication.selection_sha256)
        self.assertEqual(selection.selection_count, 1)
        self.assertEqual(selection.picks[0].role, "PRINCIPAL")
        self.assertEqual(selection.picks[0].predicted_team_name, "Domicile 100")
        self.assertEqual(selection.picks[0].best_decimal_odds, Decimal("1.80"))
        self.assertEqual(
            selection.picks[0].expected_value_percent,
            Decimal("11.60"),
        )

    def test_absent_selection_returns_none(self) -> None:
        self._write_snapshot(
            self._row(100, probability="0.62", market_probability="0.56", gap="6", odds="1.80")
        )
        self.assertIsNone(load_daily_selection(TARGET, project_directory=self.project))

    def test_snapshot_is_never_modified_and_slot_never_overwritten(self) -> None:
        self._write_snapshot(
            self._row(100, probability="0.62", market_probability="0.56", gap="6", odds="1.80")
        )
        snapshot_before = (self.snapshot_slot / "market_snapshot.json").read_bytes()
        publication = create_daily_selection_publication(
            TARGET,
            probable_pitchers_by_game=self._pitchers(100),
            project_directory=self.project,
        )
        selection_before = publication.selection_path.read_bytes()
        with self.assertRaisesRegex(LPFEdgeDailySelectionError, "existe déjà"):
            create_daily_selection_publication(
                TARGET,
                probable_pitchers_by_game=self._pitchers(100),
                project_directory=self.project,
            )
        self.assertEqual((self.snapshot_slot / "market_snapshot.json").read_bytes(), snapshot_before)
        self.assertEqual(publication.selection_path.read_bytes(), selection_before)

    def test_tampered_selection_is_rejected(self) -> None:
        self._write_snapshot(
            self._row(100, probability="0.62", market_probability="0.56", gap="6", odds="1.80")
        )
        publication = create_daily_selection_publication(
            TARGET,
            probable_pitchers_by_game=self._pitchers(100),
            project_directory=self.project,
        )
        payload = json.loads(publication.selection_path.read_bytes())
        payload["selected_game_ids"] = []
        publication.selection_path.write_bytes(canonical(payload))
        with self.assertRaises(LPFEdgeDailySelectionError):
            load_daily_selection(TARGET, project_directory=self.project)

    def test_semantic_tampering_is_rejected_even_with_matching_marker(self) -> None:
        self._write_snapshot(
            self._row(100, probability="0.62", market_probability="0.56", gap="6", odds="1.80")
        )
        publication = create_daily_selection_publication(
            TARGET,
            probable_pitchers_by_game=self._pitchers(100),
            project_directory=self.project,
        )
        payload = json.loads(publication.selection_path.read_bytes())
        payload["decisions"][0]["selection_role"] = "INVENTÉ"
        raw = canonical(payload)
        publication.selection_path.write_bytes(raw)
        marker = json.loads(publication.completed_path.read_bytes())
        marker["selection_sha256"] = hashlib.sha256(raw).hexdigest()
        publication.completed_path.write_bytes(canonical(marker))
        with self.assertRaisesRegex(LPFEdgeDailySelectionError, "rôle"):
            load_daily_selection(TARGET, project_directory=self.project)

    def test_pitcher_mapping_must_cover_every_prediction(self) -> None:
        self._write_snapshot(
            self._row(100, probability="0.62", market_probability="0.56", gap="6", odds="1.80")
        )
        with self.assertRaisesRegex(LPFEdgeDailySelectionError, "mêmes matchs"):
            create_daily_selection_publication(
                TARGET,
                probable_pitchers_by_game={},
                project_directory=self.project,
            )
        self.assertFalse((self.project / DAILY_SELECTION_ROOT).exists())

    def test_source_has_no_network_database_model_or_bet_execution(self) -> None:
        source = Path(__file__).parents[1].joinpath(
            "src", "lpf_edge_daily_selection.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("requests", source)
        self.assertNotIn("sqlite3", source)
        self.assertNotIn("predict_proba", source)
        self.assertNotIn("place_bet", source)
        self.assertNotIn("INSERT ", source)
        self.assertNotIn("UPDATE ", source)


if __name__ == "__main__":
    unittest.main()
