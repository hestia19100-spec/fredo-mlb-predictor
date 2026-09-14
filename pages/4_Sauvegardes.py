"""Création, vérification et téléchargement des sauvegardes LPF Edge."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from src.lpf_edge_daily_operations import (
    DailyBackupAutomationError,
    DailyOperationsError,
    execute_verified_local_backup,
    inspect_daily_operations,
    list_local_backup_names,
    load_verified_local_backup,
)
from src.lpf_edge_dashboard import format_paris_datetime
from src.lpf_edge_streamlit_shared import render_page_header


st.set_page_config(page_title="Sauvegardes · LPF Edge", page_icon="💾", layout="wide")
render_page_header(
    "Sauvegardes · LPF Edge",
    "Cette page regroupe uniquement les archives locales vérifiées. Elle ne "
    "récupère aucun résultat, ne contacte aucune API et ne lance jamais le modèle.",
    "ARCHIVES VÉRIFIÉES",
)

st.subheader("Créer une sauvegarde")
try:
    daily = inspect_daily_operations()
except (DailyOperationsError, OSError, ValueError) as error:
    st.error(f"L’état du dépôt ne peut pas être contrôlé : {error}")
    daily = None

backup_ready = daily is not None and daily.git.ready_for_publication
st.caption(
    "Le dépôt est propre, sur main et synchronisé : la sauvegarde peut être créée."
    if backup_ready
    else "Le dépôt doit être propre, sur main et synchronisé avant la sauvegarde."
)
backup_clicked = st.button(
    "Créer une sauvegarde",
    type="primary",
    disabled=not backup_ready,
    use_container_width=True,
)
if backup_clicked:
    with st.spinner("Création et vérification de la sauvegarde en cours..."):
        try:
            backup = execute_verified_local_backup()
        except DailyBackupAutomationError as error:
            st.error(f"Sauvegarde arrêtée à l’étape {error.stage.value} : {error}")
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

st.divider()
st.subheader("Historique des sauvegardes")
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
        format_func=lambda value: value.removeprefix("fredo-mlb-backup-").removesuffix(
            ".tar.gz"
        ),
    )
    try:
        selected_backup = load_verified_local_backup(selected_backup_name)
        backup_created_at = datetime.fromisoformat(selected_backup.created_at_utc)
    except (DailyBackupAutomationError, OSError, ValueError) as error:
        st.error(f"La sauvegarde sélectionnée ne peut pas être vérifiée : {error}")
    else:
        date_column, size_column, files_column = st.columns(3)
        date_column.metric("Créée le", format_paris_datetime(backup_created_at))
        size_column.metric(
            "Taille", f"{selected_backup.archive_size_bytes / (1024 * 1024):.2f} Mo"
        )
        files_column.metric("Fichiers protégés", selected_backup.file_count)
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
