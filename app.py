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
            title="LPF Edge",
            icon="⚾",
        ),
    ]
)
navigation.run()
