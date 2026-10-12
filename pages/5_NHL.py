"""NHL development status and historical data coverage; no public picks."""

from __future__ import annotations

import streamlit as st

from src.lpf_edge_streamlit_shared import render_page_header
from src.nhl.public_dashboard import load_nhl_coverage


render_page_header(
    "NHL — pronostics et statistiques",
    "Suivi du développement NHL. Les chiffres ci-dessous décrivent les données disponibles, pas la performance d'un modèle.",
    "NHL · recherche en cours",
)

st.subheader("Pronostics NHL")
st.info(
    "Aucun pronostic NHL n'est publiable actuellement. "
    "Les observations de recherche restent privées et ne constituent pas des conseils de pari."
)

st.subheader("Couverture historique")
try:
    coverage = load_nhl_coverage()
except (OSError, ValueError, KeyError, TypeError) as error:
    st.warning("Le rapport de couverture NHL est indisponible ou nécessite une revue.")
else:
    first, second, third = st.columns(3)
    first.metric("Saisons documentées", len(coverage.seasons))
    second.metric("Matchs de saison régulière", f"{coverage.regular_games:,}".replace(",", " "))
    third.metric("Avec historique complet", f"{coverage.full_lookback_games:,}".replace(",", " "))
    st.dataframe(
        [
            {
                "Saison": f"{row.season}–{str(row.season + 1)[-2:]}",
                "Matchs de saison régulière": row.regular_games,
                "Avec historique complet": row.full_lookback_games,
            }
            for row in coverage.seasons
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Source : {coverage.attribution}. Ce rapport ne démontre ni l'accès "
        "historique aux informations d'avant-match, ni la vérification indépendante des scores."
    )

st.subheader("Statistiques des pronostics")
st.info(
    "Aucune statistique de réussite ou de rendement NHL validée n'est disponible. "
    "Elle sera affichée après des prédictions scellées avant match, des résultats vérifiés "
    "et une évaluation chronologique contrôlée."
)

st.subheader("Étapes avant publication")
st.write(
    "1. Vérifier les données d'avant-match et les résultats officiels. "
    "2. Évaluer le modèle hors échantillon et sur des journées prospectives. "
    "3. Valider le marché de cotes « vainqueur du match » et la méthode de sélection. "
    "4. Autoriser explicitement la publication après revue."
)
