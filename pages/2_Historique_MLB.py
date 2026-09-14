"""Consultation d'une journée certifiée LPF Edge."""

from __future__ import annotations

import html

import streamlit as st

from src.lpf_edge_daily_selection import (
    LPFEdgeDailySelectionError,
    load_daily_selection,
)
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
from src.lpf_edge_market_comparison import (
    LPFEdgeMarketComparisonError,
    build_french_market_comparison,
)
from src.lpf_edge_market_settlement import (
    LPFEdgeMarketSettlementError,
    load_market_settlement,
)
from src.lpf_edge_odds_display import (
    LPFEdgeOddsDisplayError,
    load_latest_moneyline_odds_display,
)
from src.lpf_edge_streamlit_shared import (
    format_decimal_odds,
    format_evaluation_percent,
    format_percentage_point_gap,
    format_selection_paris_time,
    format_theoretical_units,
    render_page_header,
)


st.set_page_config(page_title="Historique MLB · LPF Edge", page_icon="📅", layout="wide")
render_page_header(
    "Historique MLB",
    "Choisis une date une seule fois, puis ouvre uniquement le détail qui "
    "t’intéresse. Cette page relit les preuves déjà publiées sans relancer le modèle.",
    "JOURNÉES CERTIFIÉES",
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

summary_columns = st.columns(4)
summary_columns[0].metric("Matchs", len(day.predictions))
summary_columns[1].metric(
    "Prédictions créées", format_paris_time(day.predictions[0].issued_at_utc)
)
summary_columns[2].metric(
    "Certification avant matchs", f"{float(day.remote_lead_minutes) / 60:.1f} h"
)
summary_columns[3].metric(
    "Résultats vérifiés", "Disponibles" if score is not None else "En attente"
)

st.divider()
st.subheader("Matchs retenus pour les pronostics")
st.caption(
    "La sélection affichée ici a été décidée et scellée avant les matchs. "
    "Elle peut contenir un choix principal, un choix secondaire, ou aucun choix."
)
try:
    sealed_daily_selection = load_daily_selection(selected_date)
except (LPFEdgeDailySelectionError, OSError, ValueError) as error:
    st.error(f"La sélection prospective est impossible à vérifier : {error}")
else:
    if sealed_daily_selection is None:
        st.info(
            "Aucune sélection prospective n’a été publiée pour cette ancienne journée."
        )
    elif not sealed_daily_selection.picks:
        st.info(
            "Aucun prono retenu aujourd’hui : aucun match ne respecte toutes "
            "les conditions prudentes de la politique prospective."
        )
        rejection_labels = {
            "LANCEURS_INCOMPLETS": "lanceurs annoncés incomplets",
            "COTES_ABSENTES": "cotes françaises absentes",
            "BOOKMAKERS_INSUFFISANTS": "moins de deux bookmakers",
            "PROBABILITE_LPF_TROP_FAIBLE": "probabilité LPF inférieure à 52 %",
            "ECART_LPF_MARCHE_INSUFFISANT": "écart LPF–marché inférieur à 2 points",
            "VALEUR_THEORIQUE_INSUFFISANTE": "valeur théorique inférieure à 3 %",
            "COTE_HORS_PLAGE": "cote hors de la plage 1,35–3,00",
            "LIMITE_DE_DEUX_SELECTIONS": "hors du duo le mieux classé",
        }
        if sealed_daily_selection.rejection_reasons:
            st.markdown("Motifs constatés :")
            for reason, count in sealed_daily_selection.rejection_reasons:
                st.markdown(
                    f"- {count} match(s) : "
                    f"{rejection_labels.get(reason, reason.lower())}."
                )
    else:
        selection_rows = []
        for pick in sealed_daily_selection.picks:
            selection_rows.append(
                {
                    "Choix": "Principal" if pick.role == "PRINCIPAL" else "Secondaire",
                    "Heure de Paris": format_selection_paris_time(
                        pick.scheduled_start_utc
                    ),
                    "Match": f"{pick.home_team_name} vs {pick.away_team_name}",
                    "Équipe retenue": pick.predicted_team_name,
                    "Probabilité LPF": probability_percent(pick.model_probability),
                    "Marché français": probability_percent(
                        pick.french_market_probability
                    ),
                    "Écart": format_percentage_point_gap(
                        pick.gap_percentage_points
                    ),
                    "Meilleure cote": format_decimal_odds(pick.best_decimal_odds),
                    "Chez": ", ".join(pick.best_bookmakers),
                    "Valeur théorique": format_evaluation_percent(
                        pick.expected_value_percent
                    ),
                }
            )
        st.dataframe(
            selection_rows,
            width="stretch",
            height="content",
            hide_index=True,
        )
        st.success(
            f"{sealed_daily_selection.selection_count} match(s) retenu(s) "
            "avant les rencontres et vérifié(s) par empreinte."
        )
    if sealed_daily_selection is not None:
        st.caption(
            f"Matchs admissibles avant la limite de deux : "
            f"{sealed_daily_selection.eligible_count} · "
            f"SHA-256 : `{sealed_daily_selection.selection_sha256}`"
        )
    st.warning(
        "Mode observation : cette sélection ne déclenche aucune mise réelle, "
        "ne garantit aucun gain et doit être évaluée sur plusieurs journées."
    )
    st.caption(
        "Conditions actuelles : deux lanceurs annoncés, au moins deux "
        "bookmakers, probabilité LPF ≥ 52 %, écart LPF–marché ≥ 2 points, "
        "valeur théorique ≥ 3 % et cote comprise entre 1,35 et 3,00."
    )

with st.expander("Voir toutes les prédictions de la journée"):
    st.subheader("Prédictions de la journée")
    rows = []
    for prediction in day.predictions:
        away_name = team_label(prediction.away_team_id, team_names)
        home_name = team_label(prediction.home_team_id, team_names)
        predicted_name = home_name if prediction.predicted_side == "HOME" else away_name
        rows.append(
            {
                "Heure de Paris": format_paris_time(prediction.scheduled_start_utc),
                "Match": f"{home_name} vs {away_name}",
                "Équipe donnée devant": predicted_name,
                "Probabilité": probability_percent(
                    prediction.predicted_probability
                ),
                "Domicile": probability_percent(prediction.p_home_win),
                "Extérieur": probability_percent(prediction.p_away_win),
            }
        )
    st.dataframe(rows, width="stretch", height="content", hide_index=True)
    st.markdown(
        '<p class="lpf-note">« Équipe donnée devant » signifie seulement que sa '
        'probabilité dépasse 50 %. Ce n’est ni une cote, ni un conseil de pari.</p>',
        unsafe_allow_html=True,
    )

with st.expander("Voir la comparaison LPF Edge / marché français"):
    st.subheader("Comparaison LPF Edge / marché français")
    st.caption(
        "Seule la dernière collecte française terminée avant la certification "
        "est utilisée ; toute collecte ultérieure est ignorée."
    )
    try:
        market_comparison = build_french_market_comparison(
            day,
            load_latest_moneyline_odds_display(
                selected_date,
                required_region="fr",
                completed_at_or_before_utc=day.certified_at_utc,
            ),
            team_names=team_names,
        )
    except (
        LPFEdgeMarketComparisonError,
        LPFEdgeOddsDisplayError,
        OSError,
        ValueError,
    ) as error:
        st.error(f"La comparaison avec le marché français est impossible : {error}")
    else:
        if market_comparison.odds_run_id is None:
            st.info(
                "Aucune collecte française terminée avant la certification n’est "
                "disponible pour cette journée."
            )
        else:
            columns = st.columns(3)
            columns[0].metric(
                "Collecte française figée", f"N° {market_comparison.odds_run_id}"
            )
            columns[1].metric(
                "Matchs comparables",
                f"{market_comparison.comparable_count} / {len(market_comparison.rows)}",
            )
            columns[2].metric(
                "Collecte terminée",
                format_paris_datetime(market_comparison.odds_completed_at_utc)
                if market_comparison.odds_completed_at_utc is not None
                else "—",
            )
            comparison_rows = []
            for row in market_comparison.rows:
                comparison_rows.append(
                    {
                        "Heure de Paris": format_paris_time(row.scheduled_start_utc),
                        "Match": f"{row.home_team_name} vs {row.away_team_name}",
                        "Équipe LPF": row.predicted_team_name,
                        "Probabilité LPF": probability_percent(row.model_probability),
                        "Marché français corrigé": (
                            probability_percent(row.french_market_probability)
                            if row.french_market_probability is not None
                            else "—"
                        ),
                        "Écart LPF – marché": format_percentage_point_gap(
                            row.gap_percentage_points
                        ),
                        "Meilleure cote française": format_decimal_odds(
                            row.best_decimal_odds
                        ),
                        "Chez": ", ".join(row.best_bookmakers) or "—",
                    }
                )
            st.dataframe(
                comparison_rows,
                width="stretch",
                height="content",
                hide_index=True,
            )
            st.caption(
                "Un écart positif signifie uniquement que LPF Edge attribue une "
                "probabilité plus élevée que le marché. "
                "Il ne garantit ni victoire ni rentabilité."
            )

with st.expander("Voir les résultats vérifiés", expanded=score is not None):
    st.subheader("Résultats vérifiés")
    if score is None:
        st.info(
            "Le moteur officiel n'a pas encore publié de rapport fermé pour "
            "cette journée."
        )
    else:
        columns = st.columns(4)
        columns[0].metric("Matchs évalués", score.scored_count)
        columns[1].metric("Prédictions réussies", score.correct_count)
        columns[2].metric("Encore en attente", score.pending_count)
        columns[3].metric(
            "Réussite du jour",
            "—" if score.accuracy is None else f"{score.accuracy * 100:.1f} %",
        )
        predictions_by_id = {
            prediction.prediction_id: prediction for prediction in day.predictions
        }
        result_rows = []
        for result in score.results:
            prediction = predictions_by_id[result.prediction_id]
            away_name = team_label(result.away_team_id, team_names)
            home_name = team_label(result.home_team_id, team_names)
            predicted_name = (
                home_name if result.predicted_side == "HOME" else away_name
            )
            score_text = (
                "—"
                if result.away_score is None or result.home_score is None
                else f"{result.home_score} – {result.away_score}"
            )
            status_icon = {"correct": "✓", "incorrect": "✕", "neutral": "•"}[
                result.display_tone
            ]
            result_rows.append(
                f'<tr class="lpf-result-{result.display_tone}">'
                f"<td>{html.escape(format_paris_time(prediction.scheduled_start_utc))}</td>"
                f'<td class="lpf-match">{html.escape(f"{home_name} vs {away_name}")}</td>'
                f"<td>{html.escape(predicted_name)}</td>"
                f"<td>{html.escape(probability_percent(result.predicted_probability))}</td>"
                f"<td>{html.escape(score_text)}</td>"
                f'<td class="lpf-result-status">{html.escape(f"{status_icon} {result.display_status}")}</td>'
                "</tr>"
            )
        st.markdown(
            '<div class="lpf-results-table"><table><thead><tr>'
            "<th>Heure de Paris</th><th>Match</th><th>Équipe pronostiquée</th>"
            "<th>Probabilité</th><th>Score final (dom. – ext.)</th><th>Résultat</th>"
            "</tr></thead><tbody>"
            + "".join(result_rows)
            + "</tbody></table></div>",
            unsafe_allow_html=True,
        )

with st.expander("Voir le verdict prospectif LPF / marché"):
    st.subheader("Verdict prospectif LPF/marché")
    st.caption(
        "Ce verdict utilise exclusivement le journal de cotes scellé avant les "
        "matchs. Le journal prospectif d’origine n’est jamais modifié."
    )
    try:
        settlement = load_market_settlement(selected_date)
    except (LPFEdgeMarketSettlementError, OSError, ValueError) as error:
        st.error(f"Le verdict prospectif est impossible à vérifier : {error}")
    else:
        if settlement is None:
            st.info(
                "Le verdict immuable sera créé automatiquement lorsque tous les "
                "matchs seront tranchés ou annulés."
            )
        else:
            columns = st.columns(4)
            columns[0].metric("Pronostics évalués", settlement.evaluated_count)
            columns[1].metric(
                "Réussite scellée",
                format_evaluation_percent(settlement.accuracy_percent),
            )
            columns[2].metric(
                "Résultat théorique scellé",
                format_theoretical_units(settlement.theoretical_net_units),
            )
            columns[3].metric(
                "Rendement théorique scellé",
                format_evaluation_percent(settlement.theoretical_roi_percent),
            )
            st.success(
                "Verdict vérifié : les cotes antérieures et les résultats "
                "officiels sont liés par des empreintes SHA-256."
            )
            if settlement.selected_count > 0:
                st.markdown("#### Verdict des matchs retenus")
                selected_columns = st.columns(4)
                selected_columns[0].metric(
                    "Sélections évaluées", settlement.selected_evaluated_count
                )
                selected_columns[1].metric(
                    "Sélections réussies", settlement.selected_correct_count
                )
                selected_columns[2].metric(
                    "Résultat théorique des sélections",
                    format_theoretical_units(
                        settlement.selected_theoretical_net_units
                    ),
                )
                selected_columns[3].metric(
                    "Rendement théorique des sélections",
                    format_evaluation_percent(
                        settlement.selected_theoretical_roi_percent
                    ),
                )

with st.expander("Voir les preuves techniques de cette journée"):
    st.write(f"Lot : `{day.batch_id}`")
    st.write(f"Commit des résultats : `{day.results_commit}`")
    st.write(f"SHA-256 des prédictions : `{day.predictions_sha256}`")
    st.write(f"SHA-256 du reçu : `{day.receipt_sha256}`")
    st.write(f"Certification GitHub : {format_paris_datetime(day.certified_at_utc)}")
