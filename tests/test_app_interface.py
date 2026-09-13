"""Contrats de la page d'accueil et de collecte manuelle."""

from pathlib import Path
import unittest


class AppInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).parents[1]
        cls.router = root.joinpath("app.py").read_text(
            encoding="utf-8"
        )
        cls.source = root.joinpath("pages", "0_Accueil.py").read_text(
            encoding="utf-8"
        )

    def test_navigation_uses_clear_french_page_names(self) -> None:
        self.assertIn("st.navigation", self.router)
        self.assertIn('title="Accueil"', self.router)
        self.assertIn('title="LPF Edge"', self.router)
        self.assertIn('default=True', self.router)
        self.assertIn("navigation.run()", self.router)

    def test_home_page_points_to_lpf_edge_for_daily_operations(self) -> None:
        self.assertIn("Accueil et collecte manuelle MLB", self.source)
        self.assertIn("st.page_link", self.source)
        self.assertIn('"pages/1_LPF_Edge.py"', self.source)
        self.assertIn("routine quotidienne", self.source)

    def test_technical_mvp_dashboard_is_no_longer_the_main_content(self) -> None:
        self.assertNotIn('python_column.metric("Python", "OK")', self.source)
        self.assertNotIn("Le socle technique fonctionne correctement", self.source)
        self.assertNotIn("Détails techniques du MVP", self.source)
        self.assertIn("État technique", self.source)

    def test_manual_collection_keeps_the_audited_ingestion_service(self) -> None:
        self.assertIn("run_schedule_ingestion", self.source)
        self.assertIn("Chaque collecte", self.source)
        self.assertIn("sans les dupliquer", self.source)
        self.assertIn("Cette page ne lance jamais le modèle", self.source)

    def test_calendar_uses_french_display_format(self) -> None:
        self.assertIn('format="DD/MM/YYYY"', self.source)

    def test_game_table_expands_to_show_all_rows(self) -> None:
        self.assertIn("st.dataframe", self.source)
        self.assertIn('height="content"', self.source)

    def test_home_team_is_displayed_before_away_team(self) -> None:
        self.assertIn(
            'f"{game.home_team_name} vs {game.away_team_name}"',
            self.source,
        )
        self.assertNotIn(
            'f"{game.away_team_name} @ {game.home_team_name}"',
            self.source,
        )
        self.assertIn(
            'return f"{game.home_score} - {game.away_score}"',
            self.source,
        )
        self.assertIn('"Score (dom. - ext.)"', self.source)


if __name__ == "__main__":
    unittest.main()
