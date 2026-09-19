"""Contrats d'interface du choix LPF privé et de son historique."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TODAY_PAGE = ROOT / "pages" / "1_LPF_Edge.py"
STATISTICS_PAGE = ROOT / "pages" / "3_Statistiques_MLB.py"
SELECTOR_MODULE = ROOT / "src" / "lpf_edge_internal_prudence.py"


class LPFEdgeInternalPrudenceInterfaceTests(unittest.TestCase):
    def test_today_page_shows_one_private_highest_probability_choice(self) -> None:
        source = TODAY_PAGE.read_text(encoding="utf-8")
        self.assertIn("Choix ayant la plus forte probabilité", source)
        self.assertIn("select_internal_prudence", source)
        self.assertIn("Usage interne uniquement", source)
        self.assertIn("probability_percent", source)
        self.assertNotIn("Publier le choix prudent", source)
        self.assertNotIn("Exporter le choix prudent", source)

    def test_statistics_are_rebuilt_from_latest_verified_scores(self) -> None:
        source = STATISTICS_PAGE.read_text(encoding="utf-8")
        self.assertIn("bilan depuis le début", source)
        self.assertIn("build_internal_prudence_report", source)
        self.assertIn("load_latest_score_summary", source)
        self.assertIn("routine du matin", source)
        self.assertIn("simple actualisation", source)
        self.assertNotIn("Actualiser les statistiques du choix", source)

    def test_selector_cannot_read_results_or_market_data(self) -> None:
        source = SELECTOR_MODULE.read_text(encoding="utf-8")
        self.assertNotIn("DailyScoreSummary", source)
        self.assertNotIn("load_latest_score_summary", source)
        self.assertNotIn("odds_repository", source)
        self.assertNotIn("market_settlement", source)
        self.assertIn('VISIBILITY = "INTERNAL_ONLY"', source)
        self.assertIn("public_exposure_allowed=False", source)


if __name__ == "__main__":
    unittest.main()
