"""Isolated 2027 MLB timing contract; not an active prediction service.

Shadow V2's 2026 engine, manifest and protocol remain frozen. A future 2027
engine must call this gate using the actual UTC completion and remote-publication
timestamps, not an intended schedule or a local commit timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

SEASON = 2027
MINIMUM_LEAD = timedelta(minutes=60)
PLANNED_START_BUFFER = timedelta(minutes=30)


class MLB2027TimingError(ValueError):
    """A proposed 2027 timing claim lacks a valid clock or chronology."""


def _require_utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise MLB2027TimingError(f"{name} must be timezone-aware UTC.")
    if value.utcoffset() != timedelta(0):
        raise MLB2027TimingError(f"{name} must be UTC.")
    return value


@dataclass(frozen=True, slots=True)
class LeadWindow:
    target_official_date: date
    earliest_start_utc: datetime
    latest_completion_utc: datetime
    latest_remote_publication_utc: datetime
    suggested_run_start_utc: datetime


def build_lead_window(
    target_official_date: date, scheduled_starts_utc: Iterable[datetime],
) -> LeadWindow:
    """Use the earliest scheduled game, never the most convenient game."""
    if type(target_official_date) is not date or target_official_date.year != SEASON:
        raise MLB2027TimingError("Only official-date 2027 is in scope.")
    try:
        starts = tuple(scheduled_starts_utc)
    except TypeError as error:
        raise MLB2027TimingError("Scheduled starts are not iterable.") from error
    if not starts:
        raise MLB2027TimingError("No scheduled game means no timing claim.")
    first = min(_require_utc(value, "scheduled start") for value in starts)
    deadline = first - MINIMUM_LEAD
    return LeadWindow(
        target_official_date=target_official_date,
        earliest_start_utc=first,
        latest_completion_utc=deadline,
        latest_remote_publication_utc=deadline,
        suggested_run_start_utc=deadline - PLANNED_START_BUFFER,
    )


def is_prospectively_on_time(
    window: LeadWindow, *, issued_at_utc: datetime,
    completed_at_utc: datetime, remote_http_date_utc: datetime,
) -> bool:
    """Require real local completion AND attested remote visibility by T-60.

    False is a late run, not permission to backfill or claim certification.
    Malformed or out-of-order evidence raises rather than silently passing.
    """
    if not isinstance(window, LeadWindow):
        raise MLB2027TimingError("A 2027 lead window is required.")
    issued = _require_utc(issued_at_utc, "issued_at_utc")
    completed = _require_utc(completed_at_utc, "completed_at_utc")
    remote = _require_utc(remote_http_date_utc, "remote_http_date_utc")
    if issued > completed or completed > remote:
        raise MLB2027TimingError("Issuance, completion and publication are out of order.")
    return (
        completed <= window.latest_completion_utc
        and remote <= window.latest_remote_publication_utc
    )
