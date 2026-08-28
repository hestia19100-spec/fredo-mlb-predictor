from datetime import datetime
import sqlite3
from zoneinfo import ZoneInfo

import streamlit as st

from src.dashboard_data import StoredGame, load_games_for_date
from src.database import initialize_database, list_tables
from src.game_repository import save_schedule
from src.mlb_api import MLBAPIError, fetch_schedule


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
st.subheader("Prédicteur statistique MLB")

python_column, streamlit_column, sqlite_column = st.columns(3)

python_column.metric("Python", "OK")
streamlit_column.metric("Streamlit", "OK")
sqlite_column.metric("SQLite", "OK")

st.success("Le socle technique fonctionne correctement.")

with st.expander("Détails techniques du MVP"):
    st.write(f"Base locale : `{database_path.name}`")
    st.write(f"Tables présentes : {', '.join(database_tables)}")

st.divider()
st.header("Matchs enregistrés")

selected_date = st.date_input(
    "Date des matchs",
    value=today_in_paris,
)

if st.button(
    "Récupérer ou actualiser depuis MLB",
    type="primary",
):
    with st.spinner("Récupération des matchs MLB en cours..."):
        try:
            api_games = fetch_schedule(selected_date)
            saved_games = save_schedule(api_games)
        except MLBAPIError as error:
            st.error(f"Erreur de communication avec MLB : {error}")
        except sqlite3.Error as error:
            st.error(f"Erreur SQLite : {error}")
        else:
            if saved_games:
                st.success(
                    f"{saved_games} matchs ont été enregistrés "
                    "ou actualisés."
                )
            else:
                st.warning(
                    "L’API MLB n’a renvoyé aucun match pour cette date."
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
        hide_index=True,
    )
    st.caption(
        "Une nouvelle récupération actualise les statuts, les scores "
        "et les lanceurs probables sans créer de doublons."
    )
else:
    st.warning(
        "Aucun match n’est enregistré pour cette date. "
        "Clique sur le bouton de récupération ci-dessus."
    )

st.info(
    "Aucune probabilité, cote ou recommandation de pari "
    "n’est encore calculée."
)