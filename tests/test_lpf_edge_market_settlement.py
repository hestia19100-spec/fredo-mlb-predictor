"""Tests du verdict prospectif immuable LPF Edge et marché français."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.lpf_edge_dashboard import DailyScoreResult, DailyScoreSummary, SCORING_ROOT
from src.lpf_edge_market_snapshot import (
    COMPLETED_SCHEMA as SNAPSHOT_COMPLETED_SCHEMA,
    MARKET_SNAPSHOT_ROOT,
    SNAPSHOT_SCHEMA,
)
from src.lpf_edge_market_settlement import (
    LPFEdgeMarketSettlementError,
    MARKET_SETTLEMENT_ROOT,
    SCORING_EVIDENCE_FILENAMES,
    create_market_settlement_publication,
    load_market_settlement,
)
from src.lpf_edge_daily_selection import create_daily_selection_publication


TARGET = date(2026, 9, 13)
CHECKPOINT = date(2026, 9, 14)
OBSERVATION = "a" * 64
PARENT_COMMIT = "b" * 40


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


def result(
    game_id: int,
    *,
    correct: bool | None,
    probability: str = "0.60",
) -> DailyScoreResult:
    return DailyScoreResult(
        prediction_id=f"prediction-{game_id}",
        game_id=game_id,
        away_team_id=10 + game_id,
        home_team_id=20 + game_id,
        predicted_side="HOME",
        predicted_probability=Decimal(probability),
        outcome_status=("SCORED_FINAL" if correct is not None else "VOID_CANCELLED"),
        away_score=2 if correct is not None else None,
        home_score=(3 if correct is True else 1) if correct is not None else None,
        actual_winner=(
            "HOME" if correct is True else "AWAY" if correct is False else None
        ),
        classification_correct=correct,
    )


def summary(*results: DailyScoreResult, pending: int = 0) -> DailyScoreSummary:
    correct = sum(item.classification_correct is True for item in results)
    incorrect = sum(item.classification_correct is False for item in results)
    void = sum(item.classification_correct is None for item in results)
    scored = correct + incorrect
    return DailyScoreSummary(
        checkpoint_date=CHECKPOINT,
        scored_count=scored,
        void_count=void,
        pending_count=pending,
        correct_count=correct,
        incorrect_count=incorrect,
        accuracy=None if scored == 0 else correct / scored,
        mean_log_loss=None,
        mean_brier_score=None,
        results=tuple(results),
    )


class LPFEdgeMarketSettlementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.snapshot_slot = self.project / MARKET_SNAPSHOT_ROOT / TARGET.isoformat()
        self.snapshot_slot.mkdir(parents=True)
        self.scoring_slot = (
            self.project
            / SCORING_ROOT
            / TARGET.isoformat()
            / "observations"
            / CHECKPOINT.isoformat()
        )
        self.scoring_slot.mkdir(parents=True)
        for filename in SCORING_EVIDENCE_FILENAMES:
            (self.scoring_slot / filename).write_bytes(
                f"preuve:{filename}\n".encode("utf-8")
            )
        self._write_snapshot(
            self._snapshot_row(100, odds="1.80", gap="4.0"),
            self._snapshot_row(101, odds="2.20", gap="-1.0"),
            self._snapshot_row(102, odds=None, gap=None),
            self._snapshot_row(103, odds="1.90", gap="2.0"),
        )

    def _snapshot_row(
        self,
        game_id: int,
        *,
        odds: str | None,
        gap: str | None,
    ) -> dict[str, object]:
        return {
            "away_team_id": 10 + game_id,
            "away_team_name": f"Extérieur {game_id}",
            "best_bookmakers": ["PMU", "ParionsSport"] if odds is not None else [],
            "best_decimal_odds": odds,
            "bookmaker_count": 2 if odds is not None else 0,
            "bookmaker_quotes": [],
            "french_market_probability": "0.56" if odds is not None else None,
            "game_id": game_id,
            "gap_percentage_points": gap,
            "home_team_id": 20 + game_id,
            "home_team_name": f"Domicile {game_id}",
            "issued_at_utc": "2026-09-13T12:00:00.000000Z",
            "model_probability": "0.60",
            "p_away_win": "0.40",
            "p_home_win": "0.60",
            "predicted_side": "HOME",
            "predicted_team_name": f"Domicile {game_id}",
            "prediction_id": f"prediction-{game_id}",
            "scheduled_start_utc": "2026-09-13T18:00:00.000000Z",
        }

    def _write_snapshot(self, *rows: dict[str, object]) -> None:
        snapshot = {
            "batch_id": "c" * 64,
            "certification_evidence_sha256": "d" * 64,
            "certification_sha256": "e" * 64,
            "certified_at_utc": "2026-09-13T12:01:00.000000Z",
            "comparable_count": sum(
                row["best_decimal_odds"] is not None for row in rows
            ),
            "odds_completed_at_utc": "2026-09-13T12:00:00.000000Z",
            "odds_region": "fr",
            "odds_run_id": 4,
            "prediction_receipt_sha256": "f" * 64,
            "prediction_results_commit": "1" * 40,
            "predictions": list(rows),
            "predictions_sha256": "2" * 64,
            "schema": SNAPSHOT_SCHEMA,
            "target_official_date": TARGET.isoformat(),
        }
        raw = canonical(snapshot)
        digest = hashlib.sha256(raw).hexdigest()
        completed = {
            "batch_id": "c" * 64,
            "comparable_count": snapshot["comparable_count"],
            "odds_run_id": 4,
            "prediction_results_commit": "1" * 40,
            "row_count": len(rows),
            "schema": SNAPSHOT_COMPLETED_SCHEMA,
            "snapshot_filename": "market_snapshot.json",
            "snapshot_sha256": digest,
            "status": "COMPLETED",
            "target_official_date": TARGET.isoformat(),
        }
        (self.snapshot_slot / "market_snapshot.json").write_bytes(raw)
        (self.snapshot_slot / "COMPLETED").write_bytes(canonical(completed))

    def _create(self):
        return create_market_settlement_publication(
            TARGET,
            summary(
                result(100, correct=True),
                result(101, correct=False),
                result(102, correct=True),
                result(103, correct=None),
            ),
            observation_id=OBSERVATION,
            publication_parent_commit=PARENT_COMMIT,
            project_directory=self.project,
        )

    def test_verdict_is_exact_canonical_and_references_frozen_proofs(self) -> None:
        snapshot_before = (self.snapshot_slot / "market_snapshot.json").read_bytes()
        marker_before = (self.snapshot_slot / "COMPLETED").read_bytes()
        publication = self._create()
        expected_slot = self.project / MARKET_SETTLEMENT_ROOT / TARGET.isoformat()
        self.assertEqual(publication.slot_path, expected_slot)
        self.assertEqual(
            sorted(path.name for path in expected_slot.iterdir()),
            ["COMPLETED", "market_evaluation.json"],
        )
        raw = publication.settlement_path.read_bytes()
        payload = json.loads(raw)
        self.assertEqual(raw, canonical(payload))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), publication.settlement_sha256)
        self.assertEqual(payload["source_market_snapshot_sha256"], publication.snapshot_sha256)
        self.assertEqual(payload["observation_id"], OBSERVATION)
        self.assertEqual(payload["publication_parent_commit"], PARENT_COMMIT)
        self.assertEqual(set(payload["scoring_evidence_sha256"]), set(SCORING_EVIDENCE_FILENAMES))
        self.assertEqual(
            (self.snapshot_slot / "market_snapshot.json").read_bytes(),
            snapshot_before,
        )
        self.assertEqual(
            (self.snapshot_slot / "COMPLETED").read_bytes(),
            marker_before,
        )

    def test_win_loss_missing_market_and_void_are_separated(self) -> None:
        publication = self._create()
        self.assertEqual(publication.evaluated_count, 2)
        self.assertEqual(publication.correct_count, 1)
        self.assertEqual(publication.missing_market_count, 1)
        self.assertEqual(publication.void_count, 1)
        self.assertEqual(publication.theoretical_net_units, Decimal("-0.20"))
        self.assertEqual(publication.theoretical_roi_percent, Decimal("-10.00"))
        payload = json.loads(publication.settlement_path.read_bytes())
        self.assertEqual(
            [row["evaluation_status"] for row in payload["results"]],
            ["EVALUATED", "EVALUATED", "MISSING_MARKET", "VOID"],
        )
        self.assertEqual(
            [row["theoretical_net_units"] for row in payload["results"]],
            ["0.80", "-1", None, None],
        )

    def test_loader_verifies_and_returns_display_summary(self) -> None:
        publication = self._create()
        loaded = load_market_settlement(TARGET, project_directory=self.project)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.settlement_sha256, publication.settlement_sha256)
        self.assertEqual(loaded.evaluated_count, 2)
        self.assertEqual(loaded.correct_count, 1)
        self.assertEqual(loaded.accuracy_percent, Decimal("50.0"))
        self.assertEqual(loaded.theoretical_net_units, Decimal("-0.20"))
        self.assertEqual(loaded.theoretical_roi_percent, Decimal("-10.00"))

    def test_daily_picks_receive_their_own_next_day_score(self) -> None:
        selection = create_daily_selection_publication(
            TARGET,
            probable_pitchers_by_game={
                game_id: (f"Extérieur {game_id}", f"Domicile {game_id}")
                for game_id in (100, 101, 102, 103)
            },
            project_directory=self.project,
        )
        self.assertEqual(selection.selection_count, 2)

        publication = self._create()
        self.assertEqual(publication.daily_selection_sha256, selection.selection_sha256)
        self.assertEqual(publication.selected_count, 2)
        self.assertEqual(publication.selected_evaluated_count, 1)
        self.assertEqual(publication.selected_correct_count, 1)
        self.assertEqual(publication.selected_void_count, 1)
        self.assertEqual(publication.selected_theoretical_net_units, Decimal("0.80"))
        self.assertEqual(publication.selected_theoretical_roi_percent, Decimal("80.0"))

        loaded = load_market_settlement(TARGET, project_directory=self.project)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.selected_count, 2)
        self.assertEqual(loaded.selected_evaluated_count, 1)
        self.assertEqual(loaded.selected_correct_count, 1)
        self.assertEqual(loaded.selected_void_count, 1)
        self.assertEqual(loaded.selected_theoretical_net_units, Decimal("0.80"))

    def test_absent_verdict_returns_none(self) -> None:
        self.assertIsNone(load_market_settlement(TARGET, project_directory=self.project))

    def test_pending_result_prevents_final_verdict(self) -> None:
        with self.assertRaisesRegex(
            LPFEdgeMarketSettlementError,
            "tous les matchs",
        ):
            create_market_settlement_publication(
                TARGET,
                summary(result(100, correct=True), pending=1),
                observation_id=OBSERVATION,
                publication_parent_commit=PARENT_COMMIT,
                project_directory=self.project,
            )
        self.assertFalse((self.project / MARKET_SETTLEMENT_ROOT).exists())

    def test_prediction_disagreement_fails_before_writing(self) -> None:
        bad = result(100, correct=True, probability="0.61")
        with self.assertRaisesRegex(
            LPFEdgeMarketSettlementError,
            "diverge",
        ):
            create_market_settlement_publication(
                TARGET,
                summary(
                    bad,
                    result(101, correct=False),
                    result(102, correct=True),
                    result(103, correct=None),
                ),
                observation_id=OBSERVATION,
                publication_parent_commit=PARENT_COMMIT,
                project_directory=self.project,
            )
        self.assertFalse((self.project / MARKET_SETTLEMENT_ROOT).exists())

    def test_tampered_snapshot_marker_is_rejected(self) -> None:
        marker_path = self.snapshot_slot / "COMPLETED"
        marker = json.loads(marker_path.read_bytes())
        marker["snapshot_sha256"] = "0" * 64
        marker_path.write_bytes(canonical(marker))
        with self.assertRaisesRegex(
            LPFEdgeMarketSettlementError,
            "ne concordent pas",
        ):
            self._create()

    def test_existing_verdict_is_never_overwritten(self) -> None:
        first = self._create()
        before = first.settlement_path.read_bytes()
        with self.assertRaisesRegex(
            LPFEdgeMarketSettlementError,
            "existe déjà",
        ):
            self._create()
        self.assertEqual(first.settlement_path.read_bytes(), before)

    def test_source_has_no_network_database_or_model_dependency(self) -> None:
        source = Path(__file__).parents[1].joinpath(
            "src", "lpf_edge_market_settlement.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("requests", source)
        self.assertNotIn("sqlite3", source)
        self.assertNotIn("predict_proba", source)
        self.assertNotIn("INSERT ", source)
        self.assertNotIn("UPDATE ", source)


if __name__ == "__main__":
    unittest.main()
