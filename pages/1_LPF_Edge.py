"""Page Streamlit de consultation des prédictions LPF Edge."""

from __future__ import annotations

from datetime import datetime
import html
import sqlite3

import streamlit as st

from src.ingestion_service import ScheduleIngestionError
from src.lpf_edge_daily_operations import (
    DailyBackupAutomationError,
    DailyActionState,
    DailyOperationsError,
    DailyPredictionAutomationError,
    DailyResultsAutomationError,
    execute_daily_prediction_publication,
    execute_daily_results_publication,
    execute_verified_local_backup,
    inspect_daily_operations,
    list_local_backup_names,
    load_verified_local_backup,
    refresh_daily_mlb_data,
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
from src.mlb_api import MLBAPIError


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
    .lpf-action-state {
        display: inline-block;
        border-radius: 999px;
        padding: 0.2rem 0.55rem;
        font-size: 0.76rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .lpf-action-ready {color: #12513c; background: #dcf7e9;}
    .lpf-action-done {color: #174c76; background: #e2f0fb;}
    .lpf-action-wait {color: #76530b; background: #fff4ce;}
    .lpf-action-blocked {color: #8c2424; background: #fde7e7;}
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
    "déjà publiés. Le modèle est lancé uniquement par une action explicite "
    "et autorisée dans le centre quotidien."
)

ACTION_PRESENTATION = {
    DailyActionState.READY: ("PRÊT", "ready"),
    DailyActionState.DONE: ("TERMINÉ", "done"),
    DailyActionState.NEED_DATA: ("DONNÉES REQUISES", "wait"),
    DailyActionState.TOO_EARLY: ("TROP TÔT", "wait"),
    DailyActionState.TOO_LATE: ("TROP TARD", "blocked"),
    DailyActionState.ACTION_REQUIRED: ("ACTION REQUISE", "wait"),
    DailyActionState.BLOCKED: ("BLOQUÉ", "blocked"),
    DailyActionState.NOT_AVAILABLE: ("NON DISPONIBLE", "wait"),
}


def render_action_status(action) -> None:
    label, tone = ACTION_PRESENTATION[action.state]
    st.markdown(
        f'<span class="lpf-action-state lpf-action-{tone}">{label}</span>',
        unsafe_allow_html=True,
    )
    st.caption(action.message)


st.divider()
st.subheader("Centre d’actions quotidien")
try:
    daily = inspect_daily_operations()
except (DailyOperationsError, OSError, ValueError) as error:
    st.error(f"Le centre d’actions ne peut pas être contrôlé : {error}")
    daily = None

if daily is not None:
    st.caption(
        f"Journée MLB du {daily.target_date.strftime('%d/%m/%Y')}. "
        "Chaque bouton vérifie automatiquement si l’action est autorisée."
    )
    data_column, prediction_column, result_column, backup_column = st.columns(4)
    with data_column:
        st.markdown("#### Données MLB")
        render_action_status(daily.data_action)
        refresh_clicked = st.button(
            daily.data_action.label,
            type="primary",
            disabled=not daily.data_action.can_execute,
            use_container_width=True,
        )
    with prediction_column:
        st.markdown("#### Prédictions")
        render_action_status(daily.prediction_action)
        prediction_clicked = st.button(
            daily.prediction_action.label,
            type="primary",
            disabled=not daily.prediction_action.can_execute,
            use_container_width=True,
            help=(
                None
                if daily.prediction_action.can_execute
                else daily.prediction_action.message
            ),
        )
    with result_column:
        st.markdown("#### Résultats")
        render_action_status(daily.results_action)
        results_clicked = st.button(
            daily.results_action.label,
            type="primary",
            disabled=not daily.results_action.can_execute,
            use_container_width=True,
            help=(
                None
                if daily.results_action.can_execute
                else daily.results_action.message
            ),
        )
    with backup_column:
        st.markdown("#### Sauvegarde")
        backup_ready = daily.git.ready_for_publication
        st.markdown(
            '<span class="lpf-action-state lpf-action-'
            f'{"ready" if backup_ready else "blocked"}">'
            f'{"PRÊT" if backup_ready else "BLOQUÉ"}</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Crée une archive locale vérifiée et téléchargeable."
            if backup_ready
            else "Le dépôt doit être propre, sur main et synchronisé."
        )
        backup_clicked = st.button(
            "Créer une sauvegarde",
            type="primary",
            disabled=not backup_ready,
            use_container_width=True,
            help=(
                None
                if backup_ready
                else "Publie ou annule les changements locaux avant la sauvegarde."
            ),
        )

    if refresh_clicked:
        with st.spinner("Récupération et archivage des données MLB en cours..."):
            try:
                refresh = refresh_daily_mlb_data(daily.target_date)
            except MLBAPIError as error:
                st.error(f"MLB ne répond pas correctement : {error}")
            except sqlite3.Error as error:
                st.error(f"La base locale ne peut pas être actualisée : {error}")
            except (ScheduleIngestionError, DailyOperationsError, OSError, ValueError) as error:
                st.error(f"L’actualisation a été arrêtée en sécurité : {error}")
            else:
                st.success(
                    f"Collecte auditée n° {refresh.run_id} terminée : "
                    f"{refresh.games_received} match(s) reçu(s), "
                    f"{refresh.games_saved} enregistré(s)."
                )
                st.caption(
                    f"Archive : `{refresh.archive_relative_path}`  \n"
                    f"SHA-256 : `{refresh.response_sha256}`"
                )

    prediction_feedback = st.session_state.pop(
        "lpf_edge_prediction_success",
        None,
    )
    if prediction_feedback is not None:
        st.success(
            "Prédictions créées, publiées et certifiées sur GitHub."
        )
        st.caption(
            f"Lot : `{prediction_feedback['batch_id']}`  \n"
            f"Commit des prédictions : "
            f"`{prediction_feedback['results_commit']}`  \n"
            f"Commit de certification : "
            f"`{prediction_feedback['certification_commit']}`"
        )

    if prediction_clicked:
        with st.spinner(
            "Création, publication et certification des prédictions en cours..."
        ):
            try:
                publication = execute_daily_prediction_publication(
                    daily.target_date
                )
            except DailyPredictionAutomationError as error:
                st.error(
                    f"Opération arrêtée à l’étape {error.stage.value} : {error}"
                )
            except OSError as error:
                st.error(
                    "Opération arrêtée avant sa fin : "
                    f"{error}"
                )
            else:
                st.session_state["lpf_edge_prediction_success"] = {
                    "batch_id": publication.batch_id,
                    "results_commit": publication.results_commit,
                    "certification_commit": publication.certification_commit,
                }
                st.rerun()

    results_feedback = st.session_state.pop(
        "lpf_edge_results_success",
        None,
    )
    if results_feedback is not None:
        if results_feedback["outcome"] == "COMPLETED":
            st.success(
                "Résultats du "
                f"{results_feedback['target_date']} récupérés, vérifiés "
                "et publiés sur GitHub."
            )
        else:
            st.warning(
                "La réponse MLB a échoué de façon contrôlée. "
                "Les preuves de cette tentative ont été publiées sur GitHub."
            )
        st.caption(
            f"Observation : `{results_feedback['observation_id']}`  \n"
            f"Commit du scoring : `{results_feedback['results_commit']}`"
        )

    if results_clicked:
        with st.spinner(
            "Récupération, vérification et publication des résultats en cours..."
        ):
            try:
                publication = execute_daily_results_publication(
                    daily.target_date
                )
            except DailyResultsAutomationError as error:
                st.error(
                    f"Opération arrêtée à l’étape {error.stage.value} : {error}"
                )
            except OSError as error:
                st.error(
                    "Opération arrêtée avant sa fin : "
                    f"{error}"
                )
            else:
                st.session_state["lpf_edge_results_success"] = {
                    "target_date": publication.target_date.strftime(
                        "%d/%m/%Y"
                    ),
                    "observation_id": publication.observation_id,
                    "outcome": publication.outcome,
                    "results_commit": publication.results_commit,
                }
                st.rerun()

    if backup_clicked:
        with st.spinner("Création et vérification de la sauvegarde en cours..."):
            try:
                backup = execute_verified_local_backup()
            except DailyBackupAutomationError as error:
                st.error(
                    f"Sauvegarde arrêtée à l’étape {error.stage.value} : {error}"
                )
            except OSError as error:
                st.error(f"La sauvegarde locale a été interrompue : {error}")
            else:
                st.success("Sauvegarde créée et vérifiée.")
                st.caption(
                    f"Archive : `{backup.relative_path}`  \n"
                    f"Taille : {backup.archive_size_bytes} octets  \n"
                    f"Fichiers : {backup.file_count}, dont "
                    f"{backup.raw_archive_count} archives MLB  \n"
                    f"SHA-256 : `{backup.archive_sha256}`"
                )
                st.download_button(
                    "Télécharger la sauvegarde",
                    data=backup.archive_bytes,
                    file_name=backup.filename,
                    mime="application/gzip",
                    on_click="ignore",
                    key="download_new_backup",
                    use_container_width=True,
                )

st.markdown("#### Historique des sauvegardes")
try:
    backup_names = list_local_backup_names()
except DailyBackupAutomationError as error:
    st.error(f"L’historique des sauvegardes ne peut pas être lu : {error}")
    backup_names = ()

if not backup_names:
    st.info("Aucune sauvegarde locale n’est encore disponible.")
else:
    selected_backup_name = st.selectbox(
        "Sauvegarde à consulter",
        options=backup_names,
        format_func=lambda value: value.removeprefix(
            "fredo-mlb-backup-"
        ).removesuffix(".tar.gz"),
    )
    try:
        selected_backup = load_verified_local_backup(selected_backup_name)
        backup_created_at = datetime.fromisoformat(
            selected_backup.created_at_utc
        )
    except (DailyBackupAutomationError, OSError, ValueError) as error:
        st.error(f"La sauvegarde sélectionnée ne peut pas être vérifiée : {error}")
    else:
        backup_date_column, backup_size_column, backup_files_column = st.columns(3)
        backup_date_column.metric(
            "Créée le",
            format_paris_datetime(backup_created_at),
        )
        backup_size_column.metric(
            "Taille",
            f"{selected_backup.archive_size_bytes / (1024 * 1024):.2f} Mo",
        )
        backup_files_column.metric(
            "Fichiers protégés",
            selected_backup.file_count,
        )
        st.caption(
            f"Archive : `{selected_backup.relative_path}`  \n"
            f"Archives MLB : {selected_backup.raw_archive_count}  \n"
            f"Version du code : `{selected_backup.code_version}`  \n"
            f"SHA-256 : `{selected_backup.archive_sha256}`"
        )
        st.download_button(
            "Télécharger cette sauvegarde vérifiée",
            data=selected_backup.archive_bytes,
            file_name=selected_backup.filename,
            mime="application/gzip",
            on_click="ignore",
            key="download_existing_backup",
            use_container_width=True,
        )

st.divider()
st.subheader("Consultation des prédictions certifiées")

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
