"""Shared session schedule — single source of truth.

scheduler/daily_runner.py, api/main.py, mcp_server/nts_mcp.py and
scripts/watchdog.py import SCHEDULE (and the calendar helpers below)
from here. Adding/changing a session means updating this file ONLY; no
other file should redefine the schedule.

Times are UTC.

Calendar (2026-09-23)
---------------------
Every entry carries a ``market``:

* ``"US"``    — runs only on a US trading day (``data.market_calendar``:
                weekday, not a full-day NYSE holiday) and only if its UTC
                time falls BEFORE that day's New York close (16:00, or
                13:00 on an early-close day such as the Friday after
                Thanksgiving or December 24).  Entries flagged
                ``after_close`` (EOD) are exempt from the close test — they
                run after the close by design, on every US trading day.
* ``"XETRA"`` — weekdays only.  There is no XETRA holiday calendar yet, so
                these sessions are deliberately NOT coupled to the US
                calendar: on Thanksgiving (a XETRA trading day) they still
                fire, on Good Friday (both closed) they fire as no-ops.

``session_runs_on`` is the one function the daemon (next run / startup
run / execute guard), the watchdog (which sessions are due), the API and
the MCP server (next session) all call, so the four never disagree on
what "no session today" means.  Stdlib only — the watchdog runs under
/usr/bin/python3 without the venv.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import TypedDict

from data.market_calendar import (
    NY_TZ,
    is_us_early_close,
    is_us_trading_day,
    us_rth_close,
)


class SessionSpec(TypedDict, total=False):
    name: str
    hour: int
    minute: int
    market: str          # "US" | "XETRA"
    after_close: bool    # runs after the US close by design (EOD)


# Authoritative session schedule. Maintain in this file ONLY.
# scheduler/daily_runner.py keeps full session metadata (tickers,
# workers, session_type) elsewhere; this constant captures the
# time-of-day and calendar binding used by every consumer.
SCHEDULE: list[SessionSpec] = [
    {"name": "XETRA_PRE",      "hour": 6,  "minute": 45, "market": "XETRA"},
    {"name": "XETRA_OPEN",     "hour": 7,  "minute": 0,  "market": "XETRA"},
    {"name": "PREMARKET_SCAN", "hour": 13, "minute": 0,  "market": "US"},
    {"name": "US_PRE",         "hour": 13, "minute": 15, "market": "US"},
    {"name": "PEAD_OPEN",      "hour": 13, "minute": 45, "market": "US"},
    {"name": "US_OPEN",        "hour": 14, "minute": 30, "market": "US"},
    # 18:00 UTC is 14:00 New York in summer and 13:00 in winter — at or
    # after the 13:00 close on an early-close day, so MIDDAY is skipped
    # there (session_runs_on); it is the only session that falls behind
    # an early close.
    {"name": "MIDDAY",         "hour": 18, "minute": 0,  "market": "US"},
    # EOD runs AFTER the nightly daily_ohlc ingest (nts-ohlc-ingest.timer,
    # 22:30 UTC, ~15 s) so its indicators — and the forward signals it
    # hands to the next US_OPEN — rest on today's completed bar, not
    # yesterday's.  22:45 is past the US close in both DST (20:00 UTC)
    # and winter (21:00 UTC) and past the ingest's same-day cutoff
    # (22:00 UTC).  Moved from 22:15 on 2026-09-23.
    {"name": "EOD",            "hour": 22, "minute": 45, "market": "US",
     "after_close": True},
]


def session_time_utc(entry: SessionSpec, day: date) -> datetime:
    """Timezone-aware UTC datetime at which `entry` fires on `day`."""
    return datetime(day.year, day.month, day.day,
                    int(entry["hour"]), int(entry["minute"]), tzinfo=timezone.utc)


def session_runs_on(entry: SessionSpec, day: date) -> tuple[bool, str | None]:
    """Does scheduled session `entry` exist on calendar day `day` (UTC date)?

    Returns ``(True, None)`` or ``(False, reason)`` with a short reason such
    as ``"weekend"``, ``"US market holiday"`` or
    ``"US early close 13:00 ET"``.  Unknown/missing ``market`` is treated
    as ``"US"`` (the conservative choice: a session nobody labelled must
    not run on a closed US market).
    """
    if day.weekday() >= 5:
        return False, "weekend"
    market = (entry.get("market") or "US").upper()
    if market == "XETRA":
        return True, None
    if not is_us_trading_day(day):
        return False, "US market holiday"
    if entry.get("after_close"):
        return True, None
    at_ny = session_time_utc(entry, day).astimezone(NY_TZ)
    close = us_rth_close(day)
    if at_ny.date() == day and at_ny.time() >= close:
        if is_us_early_close(day):
            return False, f"US early close {close.strftime('%H:%M')} ET"
        return False, "after US close"
    return True, None


def sessions_on(day: date, schedule: list[SessionSpec] | None = None) -> list[SessionSpec]:
    """Schedule entries that exist on `day`, in schedule order."""
    sched = SCHEDULE if schedule is None else schedule
    return [e for e in sched if session_runs_on(e, day)[0]]


def us_calendar_note(day: date) -> str | None:
    """Human-readable calendar status for `day`, or None on a normal
    US trading day: ``"weekend"``, ``"US market holiday"`` or
    ``"US early close 13:00 ET"``."""
    if day.weekday() >= 5:
        return "weekend"
    if not is_us_trading_day(day):
        return "US market holiday"
    if is_us_early_close(day):
        return f"US early close {us_rth_close(day).strftime('%H:%M')} ET"
    return None


def next_session_run(
    after: datetime,
    schedule: list[SessionSpec] | None = None,
    *,
    lookahead_days: int = 14,
) -> tuple[SessionSpec, datetime] | None:
    """First scheduled session strictly after `after` (aware UTC) that
    exists on its calendar day.  Skips weekends, US holidays and
    early-close casualties; returns ``(entry, fire_time_utc)`` or None if
    nothing fires within `lookahead_days` (never happens with the default
    schedule — 14 days covers the longest holiday cluster)."""
    sched = SCHEDULE if schedule is None else schedule
    after = after.astimezone(timezone.utc)
    for offset in range(lookahead_days + 1):
        day = (after + timedelta(days=offset)).date()
        for entry in sched:
            t = session_time_utc(entry, day)
            if t > after and session_runs_on(entry, day)[0]:
                return entry, t
    return None


def last_session_run(
    before: datetime,
    schedule: list[SessionSpec] | None = None,
    *,
    lookback_days: int = 14,
) -> tuple[SessionSpec, datetime] | None:
    """Most recent scheduled session at or before `before` (aware UTC)
    that existed on its calendar day."""
    sched = SCHEDULE if schedule is None else schedule
    before = before.astimezone(timezone.utc)
    for offset in range(lookback_days + 1):
        day = (before - timedelta(days=offset)).date()
        for entry in reversed(sched):
            t = session_time_utc(entry, day)
            if t <= before and session_runs_on(entry, day)[0]:
                return entry, t
    return None


__all__ = [
    "SCHEDULE",
    "SessionSpec",
    "session_time_utc",
    "session_runs_on",
    "sessions_on",
    "us_calendar_note",
    "next_session_run",
    "last_session_run",
]
