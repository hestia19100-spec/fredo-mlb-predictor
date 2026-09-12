"""Page Streamlit de consultation des prédictions LPF Edge."""

from __future__ import annotations

import html

import streamlit as st

from src.lpf_edge_dashboard import (
    LPFEdgeDashboardError,
    format_paris_datetime,
    format_paris_time,
    list_certified_prediction_dates,
    load_certified_prediction_day,
    load_latest_score_summary,
    load_team_names,
    probability_percent,
    team_label,
)


st.set_page_config(
    page_title="LPF Edge · MLB",
    page_icon="⚾",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetric"] {
        background: #f5f7fb;
        border: 1px solid #e2e7f0;
        border-radius: 14px;
        padding: 14px 16px;
    }
    .lpf-badge {
        display: inline-block;
        color: #12513c;
        background: #dcf7e9;
        border: 1px solid #a9e3c5;
        border-radius: 999px;
        padding: 0.25rem 0.65rem;
        font-size: 0.84rem;
        font-weight: 650;
        margin-bottom: 0.7rem;
    }
    .lpf-note {color: #566174; font-size: 0.95rem;}
    .lpf-results-table {
        overflow-x: auto;
        border: 1px solid #dfe5ee;
        border-radius: 14px;
        margin-top: 1rem;
    }
    .lpf-results-table table {
        width: 100%;
        border-collapse: collapse;
        min-width: 780px;
        font-size: 0.95rem;
    }
    .lpf-results-table th,
    .lpf-results-table td {
        padding: 0.8rem 0.9rem;
        text-align: left;
        border-bottom: 1px solid #dfe5ee;
        vertical-align: middle;
    }
    .lpf-results-table th {
        color: #4b5565;
        background: #f5f7fb;
        font-weight: 650;
    }
    .lpf-results-table tbody tr:last-child td {border-bottom: none;}
    .lpf-result-correct td {background: #eaf8ef; color: #185c37;}
    .lpf-result-incorrect td {background: #fdeeee; color: #8c2424;}
    .lpf-result-neutral td {background: #f4f5f7; color: #525b68;}
    .lpf-result-status {font-weight: 700; white-space: nowrap;}
    .lpf-results-table .lpf-match {color: #202735; font-weight: 550;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="lpf-badge">PRÉDICTIONS CERTIFIÉES</div>', unsafe_allow_html=True)
st.title("LPF Edge · MLB")
st.caption(
    "Les probabilités affichées sont relues depuis les fichiers immuables "
    "déjà publiés. Cette page ne relance jamais le modèle."
)

available_dates = list_certified_prediction_dates()
if not available_dates:
    st.warning("Aucune journée certifiée n'est encore disponible.")
    st.stop()

selected_date = st.selectbox(
    "Journée à consulter",
    options=list(reversed(available_dates)),
    format_func=lambda value: value.strftime("%d/%m/%Y"),
)

try:
    day = load_certified_prediction_day(selected_date)
    score = load_latest_score_summary(
        selected_date,
        certified_predictions=day.predictions,
    )
except LPFEdgeDashboardError as error:
    st.error(f"Contrôle d'intégrité impossible : {error}")
    st.stop()

team_ids = {
    team_id
    for prediction in day.predictions
    for team_id in (prediction.away_team_id, prediction.home_team_id)
}
team_names = load_team_names(team_ids)

games_column, issued_column, certification_column, results_column = st.columns(4)
games_column.metric("Matchs", len(day.predictions))
issued_column.metric(
    "Prédictions créées",
    format_paris_time(day.predictions[0].issued_at_utc),
)
certification_column.metric(
    "Certification avant matchs",
    f"{float(day.remote_lead_minutes) / 60:.1f} h",
)
results_column.metric(
    "Résultats vérifiés",
    "Disponibles" if score is not None else "En attente",
)

st.divider()
st.subheader("Prédictions de la journée")

table_rows: list[dict[str, str]] = []
for prediction in day.predictions:
    away_name = team_label(prediction.away_team_id, team_names)
    home_name = team_label(prediction.home_team_id, team_names)
    predicted_name = home_name if prediction.predicted_side == "HOME" else away_name
    table_rows.append(
        {
            "Heure de Paris": format_paris_time(prediction.scheduled_start_utc),
            "Match": f"{away_name} @ {home_name}",
            "Équipe donnée devant": predicted_name,
            "Probabilité": probability_percent(prediction.predicted_probability),
            "Domicile": probability_percent(prediction.p_home_win),
            "Extérieur": probability_percent(prediction.p_away_win),
        }
    )

st.dataframe(
    table_rows,
    width="stretch",
    hide_index=True,
    column_config={
        "Heure de Paris": st.column_config.TextColumn(width="small"),
        "Match": st.column_config.TextColumn(width="large"),
        "Équipe donnée devant": st.column_config.TextColumn(width="large"),
    },
)

st.markdown(
    '<p class="lpf-note">« Équipe donnée devant » signifie seulement que sa '
    'probabilité dépasse 50 %. Ce n’est ni une cote, ni un conseil de pari.</p>',
    unsafe_allow_html=True,
)

st.divider()
st.subheader("Résultats vérifiés")
if score is None:
    st.info(
        "Le moteur officiel de vérification n'a pas encore publié de rapport "
        "fermé pour cette journée. Les résultats apparaîtront ici dès que ce "
        "rapport immuable existera."
    )
else:
    scored_column, correct_column, pending_column, accuracy_column = st.columns(4)
    scored_column.metric("Matchs évalués", score.scored_count)
    correct_column.metric("Prédictions réussies", score.correct_count)
    pending_column.metric("Encore en attente", score.pending_count)
    accuracy_column.metric(
        "Réussite du jour",
        "—" if score.accuracy is None else f"{score.accuracy * 100:.1f} %",
    )
    st.caption(
        f"Dernier contrôle officiel : {score.checkpoint_date.strftime('%d/%m/%Y')}. "
        "Ces chiffres sont provisoires et ne constituent pas encore le verdict final du modèle."
    )
    predictions_by_id = {
        prediction.prediction_id: prediction for prediction in day.predictions
    }
    result_rows: list[str] = []
    for result in score.results:
        prediction = predictions_by_id[result.prediction_id]
        away_name = team_label(result.away_team_id, team_names)
        home_name = team_label(result.home_team_id, team_names)
        predicted_name = (
            home_name if result.predicted_side == "HOME" else away_name
        )
        if result.away_score is None or result.home_score is None:
            score_text = "—"
        else:
            score_text = f"{result.away_score} – {result.home_score}"
        status_icon = {
            "correct": "✓",
            "incorrect": "✕",
            "neutral": "•",
        }[result.display_tone]
        time_text = html.escape(format_paris_time(prediction.scheduled_start_utc))
        match_text = html.escape(f"{away_name} @ {home_name}")
        predicted_text = html.escape(predicted_name)
        probability_text = html.escape(
            probability_percent(result.predicted_probability)
        )
        score_text = html.escape(score_text)
        status_text = html.escape(f"{status_icon} {result.display_status}")
        rendered_cells = (
            f"<td>{time_text}</td>"
            f'<td class="lpf-match">{match_text}</td>'
            f"<td>{predicted_text}</td>"
            f"<td>{probability_text}</td>"
            f"<td>{score_text}</td>"
            f'<td class="lpf-result-status">{status_text}</td>'
        )
        result_rows.append(
            f'<tr class="lpf-result-{result.display_tone}">'
            f"{rendered_cells}</tr>"
        )
    st.markdown(
        '<div class="lpf-results-table"><table>'
        "<thead><tr>"
        "<th>Heure de Paris</th>"
        "<th>Match</th>"
        "<th>Équipe pronostiquée</th>"
        "<th>Probabilité</th>"
        "<th>Score final (ext. – dom.)</th>"
        "<th>Résultat</th>"
        "</tr></thead><tbody>"
        + "".join(result_rows)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )

with st.expander("Voir les preuves de cette journée"):
    st.write(f"Lot : `{day.batch_id}`")
    st.write(f"Commit des résultats : `{day.results_commit}`")
    st.write(f"SHA-256 des prédictions : `{day.predictions_sha256}`")
    st.write(f"SHA-256 du reçu : `{day.receipt_sha256}`")
    st.write(f"Certification GitHub : {format_paris_datetime(day.certified_at_utc)}")

st.warning(
    "LPF Edge estime des probabilités de victoire. Sans cotes préenregistrées, "
    "il ne mesure pas la rentabilité et ne produit aucune recommandation de pari."
)
