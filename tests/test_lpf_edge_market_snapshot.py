"""Tests du journal prospectif immuable LPF Edge et marché français."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.lpf_edge_dashboard import CertifiedPrediction, CertifiedPredictionDay
from src.lpf_edge_market_snapshot import (
    LPFEdgeMarketSnapshotError,
    MARKET_SNAPSHOT_ROOT,
    create_market_snapshot_publication,
)
from src.lpf_edge_odds_display import (
    MoneylineBookmakerDisplayQuote,
    MoneylineOddsDisplay,
    MoneylineOddsDisplayGame,
)


UTC = timezone.utc
TARGET = date(2026, 9, 13)
START = datetime(2026, 9, 13, 18, 10, tzinfo=UTC)
CERTIFIED_AT = datetime(2026, 9, 13, 12, 1, tzinfo=UTC)
TEAM_NAMES = {10: "Équipe extérieure", 20: "Équipe domicile"}


def prediction_day() -> CertifiedPredictionDay:
    prediction = CertifiedPrediction(
        prediction_id="prediction-1",
        batch_id="a" * 64,
        game_id=100,
        official_date=TARGET,
        away_team_id=10,
        home_team_id=20,
        scheduled_start_utc=START,
        issued_at_utc=datetime(2026, 9, 13, 12, 0, tzinfo=UTC),
        p_home_win=Decimal("0.60"),
        p_away_win=Decimal("0.40"),
    )
    return CertifiedPredictionDay(
        target_date=TARGET,
        batch_id="a" * 64,
        results_commit="b" * 40,
        certified_at_utc=CERTIFIED_AT,
        remote_lead_minutes=Decimal("300"),
        predictions_sha256="c" * 64,
        receipt_sha256="d" * 64,
        predictions=(prediction,),
    )


def quote(
    key: str,
    title: str,
    *,
    home: str,
    away: str,
) -> MoneylineBookmakerDisplayQuote:
    return MoneylineBookmakerDisplayQuote(
        key=key,
        title=title,
        last_update_utc=datetime(2026, 9, 13, 11, 59, tzinfo=UTC),
        home_decimal_odds=Decimal(home),
        away_decimal_odds=Decimal(away),
    )


def odds_display(
    *quotes: MoneylineBookmakerDisplayQuote,
    region: str | None = "fr",
) -> MoneylineOddsDisplay:
    best_home = max(
        (item.home_decimal_odds for item in quotes),
        default=None,
    )
    best_away = max(
        (item.away_decimal_odds for item in quotes),
        default=None,
    )
    game = MoneylineOddsDisplayGame(
        game_id=100,
        scheduled_start_utc=START,
        home_team_name="Équipe domicile",
        away_team_name="Équipe extérieure",
        bookmaker_count=len(quotes),
        home_best_decimal_odds=best_home,
        home_best_bookmakers=tuple(
            item.title
            for item in quotes
            if item.home_decimal_odds == best_home
        ),
        away_best_decimal_odds=best_away,
        away_best_bookmakers=tuple(
            item.title
            for item in quotes
            if item.away_decimal_odds == best_away
        ),
        bookmaker_titles=tuple(item.title for item in quotes),
        latest_bookmaker_update_utc=(
            max((item.last_update_utc for item in quotes), default=None)
        ),
        bookmaker_quotes=tuple(quotes),
    )
    return MoneylineOddsDisplay(
        target_date=TARGET,
        run_id=7 if region == "fr" else None,
        region=region,
        completed_at_utc=(
            datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
            if region == "fr"
            else None
        ),
        quote_count=len(quotes),
        games=(game,),
    )


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


class LPFEdgeMarketSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)

    def _create(
        self,
        display: MoneylineOddsDisplay | None = None,
        *,
        project: Path | None = None,
        team_names: dict[int, str] | None = None,
    ):
        return create_market_snapshot_publication(
            prediction_day(),
            display
            or odds_display(
                quote("pmu_fr", "PMU", home="1.90", away="2.00"),
                quote(
                    "betclic_fr",
                    "Betclic",
                    home="1.85",
                    away="2.05",
                ),
            ),
            team_names=team_names or TEAM_NAMES,
            certification_sha256="e" * 64,
            certification_evidence_sha256="f" * 64,
            project_directory=project or self.project,
        )

    def test_snapshot_and_completed_marker_are_exact_and_canonical(self) -> None:
        publication = self._create()

        expected_slot = self.project / MARKET_SNAPSHOT_ROOT / "2026-09-13"
        self.assertEqual(publication.slot_path, expected_slot)
        self.assertEqual(
            sorted(path.name for path in expected_slot.iterdir()),
            ["COMPLETED", "market_snapshot.json"],
        )
        snapshot_bytes = publication.snapshot_path.read_bytes()
        completed_bytes = publication.completed_path.read_bytes()
        snapshot = json.loads(snapshot_bytes)
        completed = json.loads(completed_bytes)
        self.assertEqual(snapshot_bytes, canonical_bytes(snapshot))
        self.assertEqual(completed_bytes, canonical_bytes(completed))
        self.assertEqual(
            hashlib.sha256(snapshot_bytes).hexdigest(),
            publication.snapshot_sha256,
        )
        self.assertEqual(
            completed["snapshot_sha256"],
            publication.snapshot_sha256,
        )
        self.assertEqual(completed["status"], "COMPLETED")
        self.assertEqual(snapshot["target_official_date"], "2026-09-13")
        self.assertEqual(snapshot["odds_region"], "fr")
        self.assertEqual(snapshot["odds_run_id"], 7)
        self.assertEqual(snapshot["certification_sha256"], "e" * 64)
        self.assertEqual(
            snapshot["certification_evidence_sha256"],
            "f" * 64,
        )
        self.assertEqual(snapshot["comparable_count"], 1)
        self.assertEqual(len(snapshot["predictions"]), 1)

    def test_snapshot_keeps_probabilities_gap_best_price_and_all_quotes(self) -> None:
        publication = self._create()
        row = json.loads(publication.snapshot_path.read_bytes())["predictions"][0]

        self.assertEqual(row["home_team_name"], "Équipe domicile")
        self.assertEqual(row["away_team_name"], "Équipe extérieure")
        self.assertEqual(row["predicted_side"], "HOME")
        self.assertEqual(row["model_probability"], "0.60")
        self.assertIsNotNone(row["french_market_probability"])
        self.assertIsNotNone(row["gap_percentage_points"])
        self.assertEqual(row["best_decimal_odds"], "1.90")
        self.assertEqual(row["best_bookmakers"], ["PMU"])
        self.assertEqual(
            [item["bookmaker_key"] for item in row["bookmaker_quotes"]],
            ["betclic_fr", "pmu_fr"],
        )
        self.assertEqual(row["bookmaker_count"], 2)

    def test_missing_french_odds_still_seals_every_prediction(self) -> None:
        publication = self._create(odds_display(region=None))
        snapshot = json.loads(publication.snapshot_path.read_bytes())
        row = snapshot["predictions"][0]

        self.assertIsNone(snapshot["odds_run_id"])
        self.assertEqual(snapshot["comparable_count"], 0)
        self.assertIsNone(row["french_market_probability"])
        self.assertIsNone(row["gap_percentage_points"])
        self.assertIsNone(row["best_decimal_odds"])
        self.assertEqual(row["bookmaker_quotes"], [])
        self.assertEqual(publication.row_count, 1)

    def test_existing_slot_is_never_overwritten(self) -> None:
        first = self._create()
        snapshot_before = first.snapshot_path.read_bytes()
        completed_before = first.completed_path.read_bytes()

        with self.assertRaisesRegex(
            LPFEdgeMarketSnapshotError,
            "existe déjà",
        ):
            self._create()

        self.assertEqual(first.snapshot_path.read_bytes(), snapshot_before)
        self.assertEqual(first.completed_path.read_bytes(), completed_before)

    def test_same_inputs_produce_the_same_snapshot_in_another_project(self) -> None:
        first = self._create()
        with tempfile.TemporaryDirectory() as other_directory:
            second = self._create(project=Path(other_directory))

        self.assertEqual(first.snapshot_sha256, second.snapshot_sha256)

    def test_inconsistent_team_names_fail_before_creating_a_slot(self) -> None:
        with self.assertRaises(LPFEdgeMarketSnapshotError):
            self._create(team_names={10: "Autre", 20: "Équipe domicile"})

        self.assertFalse((self.project / MARKET_SNAPSHOT_ROOT).exists())

    def test_snapshot_source_has_no_network_model_or_sqlite_write(self) -> None:
        source = Path(__file__).parents[1].joinpath(
            "src", "lpf_edge_market_snapshot.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("requests", source)
        self.assertNotIn("predict_proba", source)
        self.assertNotIn("sqlite3", source)
        self.assertNotIn("INSERT ", source)
        self.assertNotIn("UPDATE ", source)


if __name__ == "__main__":
    unittest.main()
