"""Page Streamlit de consultation des prédictions LPF Edge."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import streamlit as st

from src.lpf_edge_daily_operations import (
    DailyAfternoonAutomationError,
    DailyAfternoonOddsStatus,
    DailyActionState,
    DailyOperationsError,
    DailyOddsCollectionError,
    DailyResultsAutomationError,
    execute_afternoon_prediction_routine,
    execute_daily_odds_collection,
    execute_daily_results_publication,
    inspect_daily_operations,
    inspect_daily_odds_collection,
    load_prediction_preparation,
)
from src.lpf_edge_dashboard import (
    format_paris_datetime,
    format_paris_time,
    probability_percent,
)
from src.lpf_edge_daily_selection import (
    LPFEdgeDailySelectionError,
    load_daily_selection,
)
from src.lpf_edge_odds_display import (
    LPFEdgeOddsDisplayError,
    load_latest_moneyline_odds_display,
)
st.set_page_config(
    page_title="Aujourd’hui · LPF Edge",
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

st.markdown('<div class="lpf-badge">ROUTINES DU JOUR</div>', unsafe_allow_html=True)
st.title("Aujourd’hui · LPF Edge MLB")
st.caption(
    "Les actions utiles aujourd’hui sont regroupées ici. Les anciennes "
    "journées et les statistiques sont maintenant rangées sur leurs pages dédiées."
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
            "Journal LPF/marché : "
            f"`{prediction_feedback.get('market_snapshot_sha256') or '—'}`  \n"
            "Sélection prospective : "
            f"{prediction_feedback.get('daily_selection_count', 0)} match(s) ; "
            f"`{prediction_feedback.get('daily_selection_sha256') or '—'}`  \n"
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
                    "market_snapshot_sha256": (
                        publication.prediction.market_snapshot_sha256
                    ),
                    "daily_selection_count": (
                        publication.prediction.daily_selection_count
                    ),
                    "daily_selection_sha256": (
                        publication.prediction.daily_selection_sha256
                    ),
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
            if results_feedback.get("market_settlement_sha256") is not None:
                st.success(
                    "Le verdict prospectif LPF/marché a été scellé et publié "
                    "dans le même commit que les résultats."
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
                    "market_settlement_sha256": (
                        publication.market_settlement_sha256
                    ),
                }
                st.rerun()

    st.divider()
    st.subheader("Pronostics du jour")
    st.caption(
        "Résumé de la sélection scellée cet après-midi. Le détail complet "
        "reste disponible dans Historique MLB."
    )
    try:
        today_selection = load_daily_selection(daily.target_date)
    except (LPFEdgeDailySelectionError, OSError, ValueError) as error:
        st.error(f"La sélection du jour ne peut pas être vérifiée : {error}")
    else:
        if today_selection is None:
            st.info(
                "La sélection apparaîtra ici après la certification des "
                "prédictions de l’après-midi."
            )
        elif not today_selection.picks:
            st.info(
                "Aucun prono retenu aujourd’hui : aucun match ne respecte "
                "toutes les conditions prudentes."
            )
        else:
            pick_columns = st.columns(len(today_selection.picks))
            for column, pick in zip(pick_columns, today_selection.picks):
                with column:
                    role = (
                        "Choix principal"
                        if pick.role == "PRINCIPAL"
                        else "Choix secondaire"
                    )
                    st.markdown(f"#### {role}")
                    st.success(pick.predicted_team_name)
                    start_time = datetime.fromisoformat(
                        pick.scheduled_start_utc.replace("Z", "+00:00")
                    )
                    st.caption(
                        f"{format_paris_time(start_time)} · "
                        f"{pick.home_team_name} vs {pick.away_team_name}"
                    )
                    st.write(
                        f"Probabilité LPF : "
                        f"**{probability_percent(pick.model_probability)}**  \n"
                        f"Meilleure cote : **{format_decimal_odds(pick.best_decimal_odds)}**  \n"
                        f"Bookmaker(s) : **{', '.join(pick.best_bookmakers)}**"
                    )
            st.warning(
                "Mode observation : ces choix ne déclenchent aucune mise "
                "réelle et ne garantissent aucun gain."
            )


st.divider()
st.subheader("Où trouver le reste ?")
st.caption(
    "La page Aujourd’hui s’arrête ici pour ne plus mélanger les dates. "
    "Les anciennes journées, les statistiques et les sauvegardes disposent "
    "désormais de leur propre page dans le menu de gauche."
)
link_columns = st.columns(3)
with link_columns[0]:
    st.page_link(
        "pages/2_Historique_MLB.py",
        label="Consulter une journée passée",
        icon="📅",
        use_container_width=True,
    )
with link_columns[1]:
    st.page_link(
        "pages/3_Statistiques_MLB.py",
        label="Voir les statistiques",
        icon="📊",
        use_container_width=True,
    )
with link_columns[2]:
    st.page_link(
        "pages/4_Sauvegardes.py",
        label="Gérer les sauvegardes",
        icon="💾",
        use_container_width=True,
    )
