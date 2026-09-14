"""Navigation principale de l'application Les Pronos de Fredo."""

import streamlit as st


navigation = st.navigation(
    [
        st.Page(
            "pages/0_Accueil.py",
            title="Accueil",
            icon="🏠",
            default=True,
        ),
        st.Page(
            "pages/1_LPF_Edge.py",
            title="Aujourd’hui",
            icon="⚾",
        ),
        st.Page(
            "pages/2_Historique_MLB.py",
            title="Historique MLB",
            icon="📅",
        ),
        st.Page(
            "pages/3_Statistiques_MLB.py",
            title="Statistiques MLB",
            icon="📊",
        ),
        st.Page(
            "pages/4_Sauvegardes.py",
            title="Sauvegardes",
            icon="💾",
        ),
    ]
)
navigation.run()
