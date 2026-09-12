from datetime import datetime
import sqlite3
from zoneinfo import ZoneInfo

import streamlit as st

from src.dashboard_data import StoredGame, load_games_for_date
from src.database import initialize_database, list_tables
from src.ingestion_service import run_schedule_ingestion
from src.mlb_api import MLBAPIError


PARIS_TIMEZONE = ZoneInfo("Europe/Paris")
UTC_TIMEZONE = ZoneInfo("UTC")

NOT_STARTED_STATUSES = {
    "scheduled",
    "pre-game",
    "preview",
    "warmup",
}


def format_game_time(game_datetime_utc: str) -> str:
    """Convertit l’heure MLB UTC en heure de Paris."""
    if not game_datetime_utc:
        return "—"

    try:
        game_datetime = datetime.fromisoformat(
            game_datetime_utc.replace("Z", "+00:00")
        )
    except ValueError:
        return "—"

    if game_datetime.tzinfo is None:
        game_datetime = game_datetime.replace(tzinfo=UTC_TIMEZONE)

    return game_datetime.astimezone(PARIS_TIMEZONE).strftime("%H:%M")


def format_score(game: StoredGame) -> str:
    """Affiche un score uniquement lorsque le match a commencé."""
    if game.status_detail.casefold() in NOT_STARTED_STATUSES:
        return "—"

    if game.away_score is None or game.home_score is None:
        return "—"

    return f"{game.away_score} - {game.home_score}"


def pitcher_name_or_missing(pitcher_name: str | None) -> str:
    """Affiche clairement un lanceur encore inconnu."""
    return pitcher_name or "Non annoncé"


def count_announced_pitchers(games: list[StoredGame]) -> int:
    """Compte les lanceurs probables disponibles."""
    return sum(
        int(game.away_probable_pitcher_name is not None)
        + int(game.home_probable_pitcher_name is not None)
        for game in games
    )


def build_table_rows(games: list[StoredGame]) -> list[dict[str, str]]:
    """Prépare les lignes du tableau Streamlit."""
    return [
        {
            "Heure de Paris": format_game_time(game.game_datetime_utc),
            "Match": (
                f"{game.away_team_name} @ {game.home_team_name}"
            ),
            "Lanceur extérieur": pitcher_name_or_missing(
                game.away_probable_pitcher_name
            ),
            "Lanceur domicile": pitcher_name_or_missing(
                game.home_probable_pitcher_name
            ),
            "Statut": game.status_detail,
            "Score": format_score(game),
            "Stade": game.venue_name or "—",
        }
        for game in games
    ]


st.set_page_config(
    page_title="Les Pronos de Fredo",
    page_icon="⚾",
    layout="wide",
)

database_path = initialize_database()
database_tables = list_tables()
today_in_paris = datetime.now(PARIS_TIMEZONE).date()

st.title("⚾ Les Pronos de Fredo")
st.subheader("Accueil et collecte manuelle MLB")

st.info(
    "Pour la routine quotidienne — données, prédictions, résultats et "
    "sauvegardes — utilise la page LPF Edge."
)
st.page_link(
    "pages/1_LPF_Edge.py",
    label="Ouvrir LPF Edge",
    icon="⚾",
    use_container_width=True,
)

with st.expander("État technique"):
    st.write(f"Base locale : `{database_path.name}`")
    st.write(f"Tables présentes : {', '.join(database_tables)}")

st.divider()
st.header("Consulter ou actualiser une date")
st.caption(
    "Cet outil manuel permet de consulter une date au choix. Chaque collecte "
    "est auditée et actualise les matchs existants sans les dupliquer."
)

selected_date = st.date_input(
    "Date des matchs",
    value=today_in_paris,
    format="DD/MM/YYYY",
)

if st.button(
    "Récupérer ou actualiser cette date depuis MLB",
    type="primary",
    help=(
        "Relit le calendrier MLB de la date choisie et met à jour les statuts, "
        "les lanceurs et les scores."
    ),
):
    with st.spinner("Collecte auditée des matchs MLB en cours..."):
        try:
            ingestion_result = run_schedule_ingestion(
                start_date=selected_date,
                end_date=selected_date,
            )
        except MLBAPIError as error:
            st.error(f"Erreur de communication avec MLB : {error}")
        except sqlite3.Error as error:
            st.error(f"Erreur SQLite : {error}")
        except (OSError, ValueError) as error:
            st.error(f"Erreur pendant l’archivage : {error}")
        else:
            if ingestion_result.games_received:
                st.success(
                    f"Collecte auditée n° {ingestion_result.run_id} "
                    f"terminée : {ingestion_result.games_received} "
                    "matchs reçus et "
                    f"{ingestion_result.games_saved} enregistrés."
                )
            else:
                st.warning(
                    f"Collecte auditée n° {ingestion_result.run_id} "
                    "terminée : aucun match renvoyé par MLB."
                )

            displayed_code_version = (
                ingestion_result.code_version or "inconnue"
            )

            st.caption(
                f"Archive : `{ingestion_result.archive_relative_path}`  \n"
                f"SHA-256 : `{ingestion_result.response_sha256}`  \n"
                f"Version du code : `{displayed_code_version}`"
            )

stored_games = load_games_for_date(selected_date)
announced_pitchers = count_announced_pitchers(stored_games)
possible_pitchers = len(stored_games) * 2

games_column, pitchers_column = st.columns(2)

games_column.metric(
    "Nombre de matchs en base",
    len(stored_games),
)
pitchers_column.metric(
    "Lanceurs probables annoncés",
    f"{announced_pitchers} / {possible_pitchers}",
)

if stored_games:
    st.dataframe(
        build_table_rows(stored_games),
        width="stretch",
        height="content",
        hide_index=True,
    )
    st.caption(
        "Chaque récupération actualise les données sans créer de "
        "doublons. Elle est également journalisée et sa réponse brute "
        "est conservée."
    )
else:
    st.warning(
        "Aucun match n’est enregistré pour cette date. "
        "Clique sur le bouton de récupération ci-dessus."
    )

st.caption(
    "Les probabilités certifiées et leurs résultats sont consultables dans "
    "LPF Edge. Cette page ne lance jamais le modèle de prédiction."
)
