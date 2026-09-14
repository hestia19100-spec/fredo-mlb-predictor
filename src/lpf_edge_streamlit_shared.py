"""Présentation commune aux pages Streamlit de LPF Edge."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import streamlit as st

from src.lpf_edge_dashboard import format_paris_time


def apply_lpf_edge_style() -> None:
    """Applique la présentation commune sans effectuer aucun calcul métier."""

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


def render_page_header(title: str, caption: str, badge: str) -> None:
    apply_lpf_edge_style()
    st.markdown(
        f'<div class="lpf-badge">{badge}</div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.caption(caption)


def format_decimal_odds(value: Decimal | None) -> str:
    return "—" if value is None else format(value, "f")


def format_percentage_point_gap(value: Decimal | None) -> str:
    return "—" if value is None else f"{value:+.1f} pt"


def format_evaluation_percent(value: Decimal | None) -> str:
    return "—" if value is None else f"{value:.1f} %"


def format_theoretical_units(value: Decimal) -> str:
    return f"{value:+.2f} unité(s)"


def format_selection_paris_time(value: str) -> str:
    return format_paris_time(datetime.fromisoformat(value.replace("Z", "+00:00")))
