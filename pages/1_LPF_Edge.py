"""Page Streamlit de consultation des prédictions LPF Edge."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import html

import streamlit as st

from src.lpf_edge_daily_operations import (
    DailyAfternoonAutomationError,
    DailyAfternoonOddsStatus,
    DailyBackupAutomationError,
    DailyActionState,
    DailyOperationsError,
    DailyOddsCollectionError,
    DailyResultsAutomationError,
    execute_afternoon_prediction_routine,
    execute_daily_odds_collection,
    execute_daily_results_publication,
    execute_verified_local_backup,
    inspect_daily_operations,
    inspect_daily_odds_collection,
    list_local_backup_names,
    load_prediction_preparation,
    load_verified_local_backup,
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
from src.lpf_edge_odds_display import (
    LPFEdgeOddsDisplayError,
    load_latest_moneyline_odds_display,
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


def format_remaining_minutes(value: float | None) -> str:
    if value is None:
        return "—"
    rounded = round(abs(value))
    hours, minutes = divmod(rounded, 60)
    duration = f"{hours} h {minutes:02d}" if hours else f"{minutes} min"
    if value < 0:
        return f"Dépassée de {duration}"
    return duration


def format_decimal_odds(value: Decimal | None) -> str:
    if value is None:
        return "—"
    return format(value, "f")


def odds_region_label(region: str | None) -> str:
    return {"fr": "France", "eu": "Europe (ancienne collecte)"}.get(
        region,
        "Non renseignée",
    )


def format_percentage_point_gap(value: Decimal | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.1f} pt"


def format_evaluation_percent(value: Decimal | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f} %"


def format_theoretical_units(value: Decimal) -> str:
    return f"{value:+.2f} unité(s)"


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
        "Le matin est réservé aux résultats ; l’après-midi aux nouvelles "
        "données et aux prédictions."
    )
    morning_column, afternoon_column = st.columns(2)
    with morning_column:
        st.markdown("### Matin — Résultats")
        st.caption(
            "À utiliser après 08:00, heure de Paris. Cette routine ne lance "
            "jamais le modèle de prédiction."
        )
        render_action_status(daily.morning_action)
        results_clicked = st.button(
            daily.morning_action.label,
            type="primary",
            disabled=not daily.morning_action.can_execute,
            use_container_width=True,
            help=(
                None
                if daily.morning_action.can_execute
                else daily.morning_action.message
            ),
            key="morning_results_routine",
        )
    with afternoon_column:
        st.markdown("### Après-midi — Prédictions")
        st.caption(
            "À partir de 12:00, MLB est d’abord actualisé. Les lanceurs, "
            "l’horaire du premier match et le délai minimal sont revérifiés, "
            "puis les cotes françaises sont réutilisées ou récupérées avant "
            "toute prédiction."
        )
        render_action_status(daily.afternoon_action)
        afternoon_clicked = st.button(
            daily.afternoon_action.label,
            type="primary",
            disabled=not daily.afternoon_action.can_execute,
            use_container_width=True,
            help=(
                None
                if daily.afternoon_action.can_execute
                else daily.afternoon_action.message
            ),
            key="afternoon_prediction_routine",
        )

    st.markdown("### Préparation avant les prédictions")
    st.caption(
        "Lecture locale de la dernière collecte enregistrée. Ce tableau "
        "ne contacte pas MLB et ne lance pas le modèle."
    )
    try:
        preparation = load_prediction_preparation(
            daily.target_date,
            now_utc=daily.inspected_at_utc,
        )
    except (DailyOperationsError, OSError, ValueError) as error:
        st.error(f"La préparation locale ne peut pas être affichée : {error}")
    else:
        preparation_columns = st.columns(4)
        preparation_columns[0].metric("Matchs", preparation.game_count)
        preparation_columns[1].metric(
            "Lanceurs annoncés",
            f"{preparation.announced_pitcher_count} / "
            f"{preparation.expected_pitcher_count}",
        )
        preparation_columns[2].metric(
            "Premier match (Paris)",
            (
                format_paris_time(preparation.first_start_utc)
                if preparation.first_start_utc is not None
                else "—"
            ),
        )
        preparation_columns[3].metric(
            "Limite Shadow v2 (Paris)",
            (
                format_paris_time(preparation.prediction_deadline_utc)
                if preparation.prediction_deadline_utc is not None
                else "—"
            ),
        )

        if preparation.game_count == 0:
            st.info(
                "Aucun match n’est encore enregistré pour aujourd’hui. "
                "La routine de l’après-midi commencera par actualiser MLB."
            )
        else:
            deadline_message = (
                "Temps restant avant la limite de lancement de Shadow v2 : "
                f"{format_remaining_minutes(preparation.remaining_minutes)}."
            )
            if (
                preparation.remaining_minutes is not None
                and preparation.remaining_minutes < 0
            ):
                st.error(deadline_message)
            elif (
                preparation.remaining_minutes is not None
                and preparation.remaining_minutes <= 60
            ):
                st.warning(deadline_message)
            else:
                st.caption(deadline_message)

            missing_ratio = (
                preparation.missing_pitcher_count
                / preparation.expected_pitcher_count
            )
            pitcher_message = (
                f"{preparation.missing_pitcher_count} lanceur(s) probable(s) "
                "manquant(s) sur "
                f"{preparation.expected_pitcher_count}. La routine de "
                "l’après-midi actualisera MLB avant de lancer le modèle."
            )
            if preparation.missing_pitcher_count == 0:
                st.success("Tous les lanceurs probables sont annoncés.")
            elif missing_ratio > 0.25:
                st.warning(pitcher_message)
            else:
                st.info(pitcher_message)

            preparation_rows: list[dict[str, str]] = []
            for game in preparation.games:
                missing_for_game = 2 - game.announced_pitcher_count
                if missing_for_game == 0:
                    readiness = "Complet"
                elif missing_for_game == 1:
                    readiness = "1 lanceur manquant"
                else:
                    readiness = "2 lanceurs manquants"
                preparation_rows.append(
                    {
                        "Heure de Paris": format_paris_time(
                            game.scheduled_start_utc
                        ),
                        "Match": (
                            f"{game.home_team_name} vs {game.away_team_name}"
                        ),
                        "Lanceur domicile": (
                            game.home_probable_pitcher_name or "Non annoncé"
                        ),
                        "Lanceur extérieur": (
                            game.away_probable_pitcher_name or "Non annoncé"
                        ),
                        "État": readiness,
                    }
                )
            st.dataframe(
                preparation_rows,
                width="stretch",
                height="content",
                hide_index=True,
                column_config={
                    "Heure de Paris": st.column_config.TextColumn(
                        width="small"
                    ),
                    "Match": st.column_config.TextColumn(width="large"),
                    "Lanceur domicile": st.column_config.TextColumn(
                        width="medium"
                    ),
                    "Lanceur extérieur": st.column_config.TextColumn(
                        width="medium"
                    ),
                    "État": st.column_config.TextColumn(width="medium"),
                },
            )

    st.markdown("#### Cotes françaises — Moneyline")
    st.caption(
        "Cette collecte interroge uniquement la région française de "
        "The Odds API. Elle peut notamment fournir Betclic, NetBet, PMU, "
        "Unibet France et Winamax France lorsqu’ils proposent des cotes MLB. "
        "Chaque clic autorisé consomme normalement 1 crédit. La clé secrète "
        "n’est jamais affichée et Shadow v2 reste inchangé."
    )
    try:
        odds = inspect_daily_odds_collection(
            daily.target_date,
            now_utc=daily.inspected_at_utc,
        )
    except (DailyOperationsError, OSError, ValueError) as error:
        st.error(f"L’état local des cotes ne peut pas être affiché : {error}")
    else:
        odds_columns = st.columns(4)
        odds_columns[0].metric(
            "Clé API",
            "Configurée" if odds.api_configured else "À configurer",
        )
        odds_columns[1].metric(
            "Dernière collecte",
            (
                f"N° {odds.latest_run_id}"
                if odds.latest_run_id is not None
                else "Aucune"
            ),
        )
        odds_columns[2].metric(
            "Matchs rapprochés",
            (
                f"{odds.events_matched} / {odds.events_received}"
                if odds.events_received is not None
                and odds.events_matched is not None
                else "—"
            ),
        )
        odds_columns[3].metric(
            "Crédits API restants",
            odds.quota_remaining if odds.quota_remaining is not None else "—",
        )
        render_action_status(odds.action)
        if not odds.api_configured:
            st.info(
                "Ajoute THE_ODDS_API_KEY aux secrets de ton Codespace, puis "
                "redémarre l’application. Ne colle jamais cette clé dans Git."
            )
        if odds.latest_run_id is not None:
            latest_time = (
                format_paris_datetime(odds.latest_completed_at_utc)
                if odds.latest_completed_at_utc is not None
                else "en cours"
            )
            st.caption(
                f"Dernier état : {odds.latest_status} · {latest_time} · "
                f"zone : {odds_region_label(odds.latest_region)} · "
                f"{odds.bookmaker_quotes_saved or 0} cote(s) enregistrée(s) · "
                f"coût du dernier appel : {odds.quota_last_cost or 0} crédit."
            )

        odds_feedback = st.session_state.pop(
            "lpf_edge_odds_success",
            None,
        )
        if odds_feedback is not None:
            st.success(
                "Cotes françaises Moneyline récupérées, archivées et "
                "rapprochées des matchs MLB locaux."
            )
            st.caption(
                f"Collecte auditée n° {odds_feedback['run_id']} · "
                f"{odds_feedback['matched']} match(s) rapproché(s) · "
                f"{odds_feedback['quotes']} cote(s) enregistrée(s) · "
                f"{odds_feedback['remaining']} crédit(s) restant(s)."
            )

        odds_clicked = st.button(
            odds.action.label,
            type="primary",
            disabled=not odds.action.can_execute,
            use_container_width=True,
            help=None if odds.action.can_execute else odds.action.message,
            key="daily_moneyline_odds_collection",
        )
        if odds_clicked:
            with st.spinner(
                "Récupération et archivage des cotes françaises en cours..."
            ):
                try:
                    odds_result = execute_daily_odds_collection(
                        daily.target_date
                    )
                except DailyOddsCollectionError as error:
                    st.error(
                        f"Collecte arrêtée à l’étape {error.stage.value} : "
                        f"{error}"
                    )
                else:
                    st.session_state["lpf_edge_odds_success"] = {
                        "run_id": odds_result.run_id,
                        "matched": odds_result.events_matched,
                        "quotes": odds_result.bookmaker_quotes_saved,
                        "remaining": odds_result.quota_remaining,
                    }
                    st.rerun()

        try:
            odds_display = load_latest_moneyline_odds_display(
                daily.target_date
            )
        except (LPFEdgeOddsDisplayError, OSError, ValueError) as error:
            st.error(f"Les cotes enregistrées ne peuvent pas être affichées : {error}")
        else:
            st.markdown("##### Dernières cotes enregistrées")
            if odds_display.run_id is None:
                st.info(
                    "Aucune collecte réussie n’est encore disponible pour "
                    "cette journée."
                )
            else:
                display_columns = st.columns(4)
                display_columns[0].metric(
                    "Collecte utilisée",
                    f"N° {odds_display.run_id}",
                )
                display_columns[1].metric(
                    "Matchs avec cotes",
                    f"{odds_display.quoted_game_count} / "
                    f"{odds_display.game_count}",
                )
                display_columns[2].metric(
                    "Cotes disponibles",
                    odds_display.quote_count,
                )
                display_columns[3].metric(
                    "Collecte terminée",
                    (
                        format_paris_time(odds_display.completed_at_utc)
                        if odds_display.completed_at_utc is not None
                        else "—"
                    ),
                )

                if odds_display.missing_game_count:
                    st.warning(
                        f"{odds_display.missing_game_count} match(s) ne "
                        "dispose(nt) d’aucune cote rapprochée."
                    )

                display_rows: list[dict[str, str | int]] = []
                for game in odds_display.games:
                    display_rows.append(
                        {
                            "Heure de Paris": format_paris_time(
                                game.scheduled_start_utc
                            ),
                            "Match": (
                                f"{game.home_team_name} vs "
                                f"{game.away_team_name}"
                            ),
                            "Meilleure cote domicile": format_decimal_odds(
                                game.home_best_decimal_odds
                            ),
                            "Bookmaker domicile": (
                                ", ".join(game.home_best_bookmakers) or "—"
                            ),
                            "Meilleure cote extérieur": format_decimal_odds(
                                game.away_best_decimal_odds
                            ),
                            "Bookmaker extérieur": (
                                ", ".join(game.away_best_bookmakers) or "—"
                            ),
                            "Bookmakers": game.bookmaker_count,
                            "Actualisation": (
                                format_paris_time(
                                    game.latest_bookmaker_update_utc
                                )
                                if game.latest_bookmaker_update_utc is not None
                                else "—"
                            ),
                        }
                    )
                st.dataframe(
                    display_rows,
                    width="stretch",
                    height="content",
                    hide_index=True,
                    column_config={
                        "Heure de Paris": st.column_config.TextColumn(
                            width="small"
                        ),
                        "Match": st.column_config.TextColumn(width="large"),
                        "Meilleure cote domicile": (
                            st.column_config.TextColumn(width="medium")
                        ),
                        "Bookmaker domicile": st.column_config.TextColumn(
                            width="medium"
                        ),
                        "Meilleure cote extérieur": (
                            st.column_config.TextColumn(width="medium")
                        ),
                        "Bookmaker extérieur": st.column_config.TextColumn(
                            width="medium"
                        ),
                        "Bookmakers": st.column_config.NumberColumn(
                            width="small"
                        ),
                        "Actualisation": st.column_config.TextColumn(
                            width="small"
                        ),
                    },
                )
                st.caption(
                    "Périmètre : "
                    + odds_region_label(odds_display.region)
                    + ". Bookmakers observés : "
                    + (
                        ", ".join(odds_display.bookmaker_titles)
                        if odds_display.bookmaker_titles
                        else "aucun"
                    )
                    + ". Les meilleures cotes sont descriptives et ne "
                    "constituent pas une recommandation de pari."
                )

    st.markdown("### Sauvegarde indépendante")
    backup_status_column, backup_button_column = st.columns((2, 1))
    with backup_status_column:
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
    with backup_button_column:
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

    prediction_feedback = st.session_state.pop(
        "lpf_edge_prediction_success",
        None,
    )
    if prediction_feedback is not None:
        st.success(
            "Routine de l’après-midi terminée : données MLB actualisées, "
            "cotes contrôlées, prédictions créées, publiées et certifiées."
        )
        odds_status = prediction_feedback.get(
            "odds_status",
            DailyAfternoonOddsStatus.UNAVAILABLE.value,
        )
        odds_label = {
            DailyAfternoonOddsStatus.REUSED.value: "collecte existante réutilisée",
            DailyAfternoonOddsStatus.COLLECTED.value: "nouvelle collecte réussie",
            DailyAfternoonOddsStatus.FAILED.value: "collecte échouée sans bloquer la prédiction",
            DailyAfternoonOddsStatus.UNAVAILABLE.value: "cotes indisponibles sans bloquer la prédiction",
        }[odds_status]
        odds_details = prediction_feedback.get(
            "odds_message",
            "Aucun état de cotes n’a été conservé par l’ancienne session.",
        )
        if odds_status in {
            DailyAfternoonOddsStatus.FAILED.value,
            DailyAfternoonOddsStatus.UNAVAILABLE.value,
        }:
            st.warning(f"Cotes françaises : {odds_label}. {odds_details}")
        else:
            st.info(f"Cotes françaises : {odds_label}. {odds_details}")
        odds_run = prediction_feedback.get("odds_run_id")
        odds_quotes = prediction_feedback.get("odds_quotes")
        odds_quota = prediction_feedback.get("odds_quota_remaining")
        st.caption(
            f"Collecte auditée n° {prediction_feedback['run_id']} : "
            f"{prediction_feedback['games_received']} match(s) reçu(s)  \n"
            f"Collecte de cotes : "
            f"{f'n° {odds_run}' if odds_run is not None else 'aucune'} ; "
            f"{odds_quotes if odds_quotes is not None else '—'} cote(s) ; "
            f"{odds_quota if odds_quota is not None else '—'} crédit(s) restant(s)  \n"
            f"Lot : `{prediction_feedback['batch_id']}`  \n"
            f"Commit des prédictions : "
            f"`{prediction_feedback['results_commit']}`  \n"
            f"Commit de certification : "
            f"`{prediction_feedback['certification_commit']}`"
        )

    if afternoon_clicked:
        with st.spinner(
            "Actualisation MLB, contrôle des cotes, prédictions et certification en cours..."
        ):
            try:
                publication = execute_afternoon_prediction_routine(
                    daily.target_date
                )
            except DailyAfternoonAutomationError as error:
                st.error(
                    f"Routine arrêtée à l’étape {error.stage.value} : {error}"
                )
                if error.data_refresh is not None:
                    st.info(
                        "Les données MLB ont bien été actualisées avant cet arrêt : "
                        f"collecte auditée n° {error.data_refresh.run_id}."
                    )
                if error.odds is not None:
                    st.info(f"État des cotes avant l’arrêt : {error.odds.message}")
            except OSError as error:
                st.error(
                    "Routine arrêtée avant sa fin : "
                    f"{error}"
                )
            else:
                st.session_state["lpf_edge_prediction_success"] = {
                    "run_id": publication.data_refresh.run_id,
                    "games_received": publication.data_refresh.games_received,
                    "odds_status": publication.odds.status.value,
                    "odds_message": publication.odds.message,
                    "odds_run_id": publication.odds.run_id,
                    "odds_quotes": publication.odds.bookmaker_quotes_saved,
                    "odds_quota_remaining": publication.odds.quota_remaining,
                    "batch_id": publication.prediction.batch_id,
                    "results_commit": publication.prediction.results_commit,
                    "certification_commit": (
                        publication.prediction.certification_commit
                    ),
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
            "Match": f"{home_name} vs {away_name}",
            "Équipe donnée devant": predicted_name,
            "Probabilité": probability_percent(prediction.predicted_probability),
            "Domicile": probability_percent(prediction.p_home_win),
            "Extérieur": probability_percent(prediction.p_away_win),
        }
    )

st.dataframe(
    table_rows,
    width="stretch",
    height="content",
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
st.subheader("Comparaison LPF Edge / marché français")
st.caption(
    "Seule la dernière collecte française terminée avant la certification "
    "est utilisée ; toute collecte ultérieure est ignorée. Pour chaque "
    "bookmaker, LPF Edge convertit les deux cotes en "
    "probabilités puis retire proportionnellement sa marge. La colonne "
    "« Marché français corrigé » est la moyenne de ces probabilités. "
    "L’écart reste descriptif et n’est pas un conseil de pari."
)
try:
    french_odds_display = load_latest_moneyline_odds_display(
        selected_date,
        required_region="fr",
        completed_at_or_before_utc=day.certified_at_utc,
    )
    market_comparison = build_french_market_comparison(
        day,
        french_odds_display,
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
            "disponible pour cette journée. Les collectes plus tardives ne sont "
            "volontairement pas utilisées."
        )
    else:
        comparison_columns = st.columns(3)
        comparison_columns[0].metric(
            "Collecte française figée",
            f"N° {market_comparison.odds_run_id}",
        )
        comparison_columns[1].metric(
            "Matchs comparables",
            f"{market_comparison.comparable_count} / "
            f"{len(market_comparison.rows)}",
        )
        comparison_columns[2].metric(
            "Collecte terminée",
            (
                format_paris_datetime(
                    market_comparison.odds_completed_at_utc
                )
                if market_comparison.odds_completed_at_utc is not None
                else "—"
            ),
        )

        comparison_rows: list[dict[str, str | int]] = []
        for row in market_comparison.rows:
            comparison_rows.append(
                {
                    "Heure de Paris": format_paris_time(
                        row.scheduled_start_utc
                    ),
                    "Match": (
                        f"{row.home_team_name} vs {row.away_team_name}"
                    ),
                    "Équipe LPF": row.predicted_team_name,
                    "Probabilité LPF": probability_percent(
                        row.model_probability
                    ),
                    "Marché français corrigé": probability_percent(
                        row.french_market_probability
                    )
                    if row.french_market_probability is not None
                    else "—",
                    "Écart LPF – marché": format_percentage_point_gap(
                        row.gap_percentage_points
                    ),
                    "Meilleure cote française": format_decimal_odds(
                        row.best_decimal_odds
                    ),
                    "Chez": ", ".join(row.best_bookmakers) or "—",
                    "Bookmakers": row.bookmaker_count,
                }
            )

        st.dataframe(
            comparison_rows,
            width="stretch",
            height="content",
            hide_index=True,
            column_config={
                "Heure de Paris": st.column_config.TextColumn(width="small"),
                "Match": st.column_config.TextColumn(width="large"),
                "Équipe LPF": st.column_config.TextColumn(width="large"),
                "Probabilité LPF": st.column_config.TextColumn(width="medium"),
                "Marché français corrigé": st.column_config.TextColumn(
                    width="medium"
                ),
                "Écart LPF – marché": st.column_config.TextColumn(
                    width="medium"
                ),
                "Meilleure cote française": st.column_config.TextColumn(
                    width="medium"
                ),
                "Chez": st.column_config.TextColumn(width="medium"),
                "Bookmakers": st.column_config.NumberColumn(width="small"),
            },
        )
        st.caption(
            "Un écart positif signifie uniquement que LPF Edge attribue une "
            "probabilité plus élevée que la moyenne corrigée des bookmakers. "
            "Il ne garantit ni victoire ni rentabilité."
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
            score_text = f"{result.home_score} – {result.away_score}"
        status_icon = {
            "correct": "✓",
            "incorrect": "✕",
            "neutral": "•",
        }[result.display_tone]
        time_text = html.escape(format_paris_time(prediction.scheduled_start_utc))
        match_text = html.escape(f"{home_name} vs {away_name}")
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
        "<th>Score final (dom. – ext.)</th>"
        "<th>Résultat</th>"
        "</tr></thead><tbody>"
        + "".join(result_rows)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )

st.divider()
st.subheader("Évaluation historique des écarts LPF–marché")
st.caption(
    "Ce bilan associe uniquement les résultats vérifiés aux cotes françaises "
    "figées avant chaque certification. Il simule une mise identique d’une "
    "unité sur chaque pronostic comparable ; aucune mise réelle n’est effectuée."
)
try:
    evaluation_days = []
    for evaluation_date in available_dates:
        evaluation_prediction_day = load_certified_prediction_day(
            evaluation_date
        )
        evaluation_score = load_latest_score_summary(
            evaluation_date,
            certified_predictions=evaluation_prediction_day.predictions,
        )
        if evaluation_score is None:
            continue
        evaluation_team_ids = {
            team_id
            for prediction in evaluation_prediction_day.predictions
            for team_id in (prediction.away_team_id, prediction.home_team_id)
        }
        evaluation_team_names = load_team_names(evaluation_team_ids)
        evaluation_odds = load_latest_moneyline_odds_display(
            evaluation_date,
            required_region="fr",
            completed_at_or_before_utc=(
                evaluation_prediction_day.certified_at_utc
            ),
        )
        evaluation_comparison = build_french_market_comparison(
            evaluation_prediction_day,
            evaluation_odds,
            team_names=evaluation_team_names,
        )
        evaluation_days.append((evaluation_comparison, evaluation_score))
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
            "une collecte française antérieure à sa certification. L’historique "
            "se remplira automatiquement au fil des prochaines journées."
        )
    else:
        evaluation_columns = st.columns(4)
        evaluation_columns[0].metric(
            "Observations évaluées",
            market_history.evaluated_count,
        )
        evaluation_columns[1].metric(
            "Réussite observée",
            format_evaluation_percent(market_history.accuracy_percent),
        )
        evaluation_columns[2].metric(
            "Résultat théorique",
            format_theoretical_units(market_history.theoretical_net_units),
        )
        evaluation_columns[3].metric(
            "Rendement théorique",
            format_evaluation_percent(
                market_history.theoretical_roi_percent
            ),
        )
        if market_history.sample_is_limited:
            st.warning(
                f"Seulement {market_history.evaluated_count} observation(s) : "
                f"avant {LIMITED_SAMPLE_THRESHOLD}, l’échantillon est trop "
                "limité pour tirer une conclusion fiable."
            )

        gap_band_rows = [
            {
                "Écart LPF–marché": band.label,
                "Observations": band.evaluated_count,
                "Réussites": band.correct_count,
                "Taux de réussite": format_evaluation_percent(
                    band.accuracy_percent
                ),
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
        st.dataframe(
            gap_band_rows,
            width="stretch",
            height="content",
            hide_index=True,
        )
        st.caption(
            "Période évaluée : "
            f"{market_history.first_date.strftime('%d/%m/%Y')} au "
            f"{market_history.last_date.strftime('%d/%m/%Y')}. "
            f"Journées avec résultats : {market_history.completed_day_count}. "
            f"Sans cote antérieure : {market_history.missing_odds_count}. "
            f"Non tranchées ou annulées : {market_history.unsettled_count}."
        )
        st.warning(
            "Le résultat est une simulation rétrospective à la meilleure cote "
            "archivée. Il suppose une disponibilité et une mise acceptée, sans "
            "frais ni limitation ; il ne constitue ni un gain réel ni une "
            "recommandation de pari."
        )

with st.expander("Voir les preuves de cette journée"):
    st.write(f"Lot : `{day.batch_id}`")
    st.write(f"Commit des résultats : `{day.results_commit}`")
    st.write(f"SHA-256 des prédictions : `{day.predictions_sha256}`")
    st.write(f"SHA-256 du reçu : `{day.receipt_sha256}`")
    st.write(f"Certification GitHub : {format_paris_datetime(day.certified_at_utc)}")

st.warning(
    "LPF Edge affiche des probabilités de victoire et des cotes observées. "
    "Shadow v2 n’utilise pas ces cotes, ne mesure pas la rentabilité et ne "
    "produit aucune recommandation de pari."
)
