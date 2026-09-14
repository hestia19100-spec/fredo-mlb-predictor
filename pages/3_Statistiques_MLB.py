"""Statistiques globales et supervision de l'évaluation LPF Edge."""

from __future__ import annotations

from decimal import Decimal

import streamlit as st

from src.lpf_edge_daily_selection import (
    LPFEdgeDailySelectionError,
    load_daily_selection,
)
from src.lpf_edge_dashboard import (
    LPFEdgeDashboardError,
    list_certified_prediction_dates,
    load_certified_prediction_day,
    load_latest_score_summary,
    load_team_names,
)
from src.lpf_edge_evaluation_supervision import (
    EvaluationDayEvidence,
    LPFEdgeEvaluationSupervisionError,
    MINIMUM_SELECTION_OBSERVATIONS,
    MINIMUM_SHADOW_OBSERVATIONS,
    build_evaluation_supervision,
)
from src.lpf_edge_market_comparison import (
    LPFEdgeMarketComparisonError,
    build_french_market_comparison,
)
from src.lpf_edge_market_evaluation import (
    LIMITED_SAMPLE_THRESHOLD,
    LPFEdgeMarketEvaluationError,
    build_market_evaluation_history,
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
    format_evaluation_percent,
    format_theoretical_units,
    render_page_header,
)


st.set_page_config(page_title="Statistiques MLB · LPF Edge", page_icon="📊", layout="wide")
render_page_header(
    "Statistiques MLB",
    "Le recul global de Shadow v2 et celui du sélecteur sont regroupés ici. "
    "Cette page est en lecture seule et n’effectue aucune mise réelle.",
    "SUIVI EN LECTURE SEULE",
)

available_dates = list_certified_prediction_dates()
if not available_dates:
    st.warning("Aucune journée certifiée n'est encore disponible.")
    st.stop()

st.subheader("Supervision de l’évaluation")
st.caption(
    "Les performances de toutes les prédictions sont séparées de celles des "
    "matchs réellement retenus par le sélecteur."
)
try:
    supervision_evidence: list[EvaluationDayEvidence] = []
    for supervision_date in available_dates:
        prediction_day = load_certified_prediction_day(supervision_date)
        score = load_latest_score_summary(
            supervision_date,
            certified_predictions=prediction_day.predictions,
        )
        supervision_evidence.append(
            EvaluationDayEvidence(
                prediction_day=prediction_day,
                score=score,
                selection=load_daily_selection(supervision_date),
                settlement=load_market_settlement(supervision_date),
            )
        )
    evaluation_supervision = build_evaluation_supervision(supervision_evidence)
except (
    LPFEdgeDashboardError,
    LPFEdgeDailySelectionError,
    LPFEdgeEvaluationSupervisionError,
    LPFEdgeMarketSettlementError,
    OSError,
    ValueError,
) as error:
    st.error(f"La supervision de l’évaluation est impossible : {error}")
else:
    st.markdown("#### Shadow v2 — toutes les prédictions")
    shadow_columns = st.columns(4)
    shadow_columns[0].metric(
        "Journées certifiées", evaluation_supervision.certified_day_count
    )
    shadow_columns[1].metric(
        "Prédictions évaluées", evaluation_supervision.shadow_evaluated_count
    )
    shadow_columns[2].metric(
        "Réussite globale",
        format_evaluation_percent(evaluation_supervision.shadow_accuracy_percent),
    )
    shadow_columns[3].metric(
        "Résultats en attente", evaluation_supervision.shadow_pending_count
    )
    st.progress(
        float(evaluation_supervision.shadow_progress_percent / Decimal("100")),
        text=(
            "Recul statistique minimal : "
            f"{evaluation_supervision.shadow_evaluated_count} / "
            f"{MINIMUM_SHADOW_OBSERVATIONS} prédictions évaluées"
        ),
    )
    quality_columns = st.columns(3)
    quality_columns[0].metric(
        "Journées entièrement tranchées", evaluation_supervision.completed_day_count
    )
    quality_columns[1].metric(
        "Log loss moyen",
        "—"
        if evaluation_supervision.weighted_log_loss is None
        else f"{evaluation_supervision.weighted_log_loss:.4f}",
        help="Plus cette mesure est basse, meilleures sont les probabilités.",
    )
    quality_columns[2].metric(
        "Score de Brier moyen",
        "—"
        if evaluation_supervision.weighted_brier_score is None
        else f"{evaluation_supervision.weighted_brier_score:.4f}",
        help="Plus cette mesure est basse, meilleures sont les probabilités.",
    )

    st.markdown("#### Sélecteur de pronostics — mode observation")
    if evaluation_supervision.selection_publication_day_count == 0:
        st.info(
            "Aucune sélection prospective n’a encore été publiée. Le suivi "
            "commencera automatiquement avec la prochaine certification."
        )
    else:
        selector_columns = st.columns(5)
        selector_columns[0].metric(
            "Journées suivies", evaluation_supervision.selection_publication_day_count
        )
        selector_columns[1].metric(
            "Journées avec prono", evaluation_supervision.selection_day_count
        )
        selector_columns[2].metric(
            "Journées sans prono", evaluation_supervision.no_pick_day_count
        )
        selector_columns[3].metric(
            "Choix évalués", evaluation_supervision.selection.evaluated_count
        )
        selector_columns[4].metric(
            "Choix en attente", evaluation_supervision.selection.pending_count
        )
        st.progress(
            float(evaluation_supervision.selection_progress_percent / Decimal("100")),
            text=(
                "Recul statistique minimal : "
                f"{evaluation_supervision.selection.evaluated_count} / "
                f"{MINIMUM_SELECTION_OBSERVATIONS} choix évalués"
            ),
        )
        result_columns = st.columns(4)
        result_columns[0].metric(
            "Réussite des choix",
            format_evaluation_percent(evaluation_supervision.selection.accuracy_percent),
        )
        result_columns[1].metric(
            "Résultat théorique",
            format_theoretical_units(
                evaluation_supervision.selection.theoretical_net_units
            ),
        )
        result_columns[2].metric(
            "Rendement théorique",
            format_evaluation_percent(
                evaluation_supervision.selection.theoretical_roi_percent
            ),
        )
        result_columns[3].metric(
            "Baisse maximale théorique",
            f"-{evaluation_supervision.selection.maximum_drawdown_units:.2f} unité(s)",
        )
        st.caption(
            "Série de défaites actuelle : "
            f"{evaluation_supervision.selection.current_losing_streak} · "
            "plus longue série : "
            f"{evaluation_supervision.selection.maximum_losing_streak} · "
            f"choix annulés : {evaluation_supervision.selection.void_count}."
        )

        role_rows = []
        for label, performance in (
            ("Principal", evaluation_supervision.principal),
            ("Secondaire", evaluation_supervision.secondary),
        ):
            role_rows.append(
                {
                    "Type de choix": label,
                    "Publiés": performance.published_count,
                    "Évalués": performance.evaluated_count,
                    "Réussis": performance.correct_count,
                    "En attente": performance.pending_count,
                    "Réussite": format_evaluation_percent(performance.accuracy_percent),
                    "Résultat théorique": format_theoretical_units(
                        performance.theoretical_net_units
                    ),
                    "Rendement théorique": format_evaluation_percent(
                        performance.theoretical_roi_percent
                    ),
                    "Pire série de défaites": performance.maximum_losing_streak,
                }
            )
        st.dataframe(role_rows, width="stretch", height="content", hide_index=True)

    day_rows = [
        {
            "Journée": status.target_date.strftime("%d/%m/%Y"),
            "Prédictions": status.prediction_count,
            "Résultats": status.prediction_status.replace("_", " ").title(),
            "Choix": status.selection_count,
            "État du sélecteur": status.selection_status.replace("_", " ").title(),
            "Choix évalués": status.selected_evaluated_count,
            "Choix réussis": status.selected_correct_count,
            "Résultat théorique": format_theoretical_units(
                status.selected_theoretical_net_units
            ),
        }
        for status in reversed(evaluation_supervision.days)
    ]
    st.dataframe(day_rows, width="stretch", height="content", hide_index=True)
    st.warning(
        "Le seuil de 100 observations indique seulement un recul minimal. "
        "Il ne valide ni la rentabilité, ni Shadow v2, ni le sélecteur. "
        "Aucune mise réelle n’est effectuée."
    )

st.divider()
st.subheader("Évaluation historique des écarts LPF–marché")
st.caption(
    "Ce bilan associe uniquement les résultats vérifiés aux cotes françaises "
    "figées avant chaque certification."
)
try:
    evaluation_days = []
    for evaluation_date in available_dates:
        prediction_day = load_certified_prediction_day(evaluation_date)
        score = load_latest_score_summary(
            evaluation_date,
            certified_predictions=prediction_day.predictions,
        )
        if score is None:
            continue
        team_ids = {
            team_id
            for prediction in prediction_day.predictions
            for team_id in (prediction.away_team_id, prediction.home_team_id)
        }
        comparison = build_french_market_comparison(
            prediction_day,
            load_latest_moneyline_odds_display(
                evaluation_date,
                required_region="fr",
                completed_at_or_before_utc=prediction_day.certified_at_utc,
            ),
            team_names=load_team_names(team_ids),
        )
        evaluation_days.append((comparison, score))
    market_history = build_market_evaluation_history(evaluation_days)
except (
    LPFEdgeDashboardError,
    LPFEdgeMarketComparisonError,
    LPFEdgeMarketEvaluationError,
    LPFEdgeOddsDisplayError,
    OSError,
    ValueError,
) as error:
    st.error(f"L’évaluation historique est impossible : {error}")
else:
    if market_history.evaluated_count == 0:
        st.info(
            "Aucun résultat ne possède encore à la fois un verdict vérifié et "
            "une collecte française antérieure à sa certification."
        )
    else:
        columns = st.columns(4)
        columns[0].metric("Observations évaluées", market_history.evaluated_count)
        columns[1].metric(
            "Réussite observée",
            format_evaluation_percent(market_history.accuracy_percent),
        )
        columns[2].metric(
            "Résultat théorique",
            format_theoretical_units(market_history.theoretical_net_units),
        )
        columns[3].metric(
            "Rendement théorique",
            format_evaluation_percent(market_history.theoretical_roi_percent),
        )
        if market_history.sample_is_limited:
            st.warning(
                f"Seulement {market_history.evaluated_count} observation(s) : "
                f"avant {LIMITED_SAMPLE_THRESHOLD}, l’échantillon est trop "
                "limité pour tirer une conclusion fiable."
            )
        gap_rows = [
            {
                "Écart LPF–marché": band.label,
                "Observations": band.evaluated_count,
                "Réussites": band.correct_count,
                "Taux de réussite": format_evaluation_percent(band.accuracy_percent),
                "Résultat théorique": (
                    format_theoretical_units(band.theoretical_net_units)
                    if band.evaluated_count > 0
                    else "—"
                ),
                "Rendement théorique": format_evaluation_percent(
                    band.theoretical_roi_percent
                ),
            }
            for band in market_history.gap_bands
        ]
        st.dataframe(gap_rows, width="stretch", height="content", hide_index=True)
        st.caption(
            "Période évaluée : "
            f"{market_history.first_date.strftime('%d/%m/%Y')} au "
            f"{market_history.last_date.strftime('%d/%m/%Y')} · "
            f"journées avec résultats : {market_history.completed_day_count} · "
            f"sans cote antérieure : {market_history.missing_odds_count}."
        )
        st.warning(
            "Le résultat est une simulation rétrospective à mise fixe ; il ne "
            "constitue ni un gain réel ni une recommandation de pari."
        )
