"""config/sessions.py — which scheduled sessions exist on which day.

The one rule the daemon, the watchdog, the API and the MCP server share
(2026-09-23): US sessions only on US trading days and before that day's
New York close (13:00 on early-close days); XETRA sessions on weekdays,
deliberately not coupled to the US calendar.

Run: python3 -m pytest tests/test_sessions_calendar.py -v
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.sessions import (  # noqa: E402
    SCHEDULE,
    last_session_run,
    next_session_run,
    session_runs_on,
    sessions_on,
    us_calendar_note,
)

_BY_NAME = {e["name"]: e for e in SCHEDULE}
_US = [e["name"] for e in SCHEDULE if e["market"] == "US"]
_XETRA = [e["name"] for e in SCHEDULE if e["market"] == "XETRA"]


def _names(day: date) -> list[str]:
    return [e["name"] for e in sessions_on(day)]


def test_every_entry_is_bound_to_a_market():
    assert set(_US) == {"PREMARKET_SCAN", "US_PRE", "PEAD_OPEN", "US_OPEN", "MIDDAY", "EOD"}
    assert set(_XETRA) == {"XETRA_PRE", "XETRA_OPEN"}
    assert _BY_NAME["EOD"].get("after_close") is True


def test_ordinary_trading_day_runs_everything():
    assert _names(date(2026, 9, 23)) == [e["name"] for e in SCHEDULE]


@pytest.mark.parametrize("holiday", [
    date(2026, 9, 7),    # Labor Day — the 2026 incident: 8 sessions on a closed market
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
    date(2026, 7, 3),    # Independence Day observed
])
def test_us_holiday_runs_only_xetra_sessions(holiday):
    assert _names(holiday) == _XETRA
    for name in _US:
        ok, why = session_runs_on(_BY_NAME[name], holiday)
        assert ok is False and why == "US market holiday"


def test_weekend_runs_nothing():
    assert _names(date(2026, 9, 5)) == []          # Saturday
    assert _names(date(2026, 9, 6)) == []          # Sunday
    assert session_runs_on(_BY_NAME["XETRA_PRE"], date(2026, 9, 5)) == (False, "weekend")


@pytest.mark.parametrize("early", [
    date(2026, 11, 27),  # Friday after Thanksgiving (EST: 18:00 UTC = 13:00 ET, at the close)
    date(2026, 12, 24),  # Christmas Eve (EST)
    date(2025, 7, 3),    # July 3 (EDT: 18:00 UTC = 14:00 ET, after the close)
])
def test_early_close_day_skips_only_midday(early):
    names = _names(early)
    assert "MIDDAY" not in names
    assert names == [n for n in (e["name"] for e in SCHEDULE) if n != "MIDDAY"]
    ok, why = session_runs_on(_BY_NAME["MIDDAY"], early)
    assert ok is False and why == "US early close 13:00 ET"
    # EOD runs after the close by design, US_OPEN is before the 13:00 close.
    assert session_runs_on(_BY_NAME["EOD"], early) == (True, None)
    assert session_runs_on(_BY_NAME["US_OPEN"], early) == (True, None)


def test_xetra_sessions_are_not_coupled_to_the_us_calendar():
    # Thanksgiving is a XETRA trading day; Good Friday closes both, and
    # the XETRA sessions still fire (as no-ops) because there is no XETRA
    # holiday calendar — documented, not accidental.
    for day in (date(2026, 11, 26), date(2026, 4, 3)):
        for name in _XETRA:
            assert session_runs_on(_BY_NAME[name], day) == (True, None)


def test_unlabelled_entry_defaults_to_us():
    assert session_runs_on({"name": "X", "hour": 14, "minute": 30}, date(2026, 9, 7))[0] is False
    assert session_runs_on({"name": "X", "hour": 14, "minute": 30}, date(2026, 9, 8))[0] is True


def test_next_session_after_friday_eod_over_labor_day_weekend():
    after = datetime(2026, 9, 4, 23, 0, tzinfo=timezone.utc)      # Fri after EOD
    entry, at = next_session_run(after)
    assert (entry["name"], at) == ("XETRA_PRE", datetime(2026, 9, 7, 6, 45, tzinfo=timezone.utc))
    # ...and the first US session after that is Tuesday's scanner, not
    # Monday's (Labor Day).
    us_only = [e for e in SCHEDULE if e["market"] == "US"]
    entry, at = next_session_run(datetime(2026, 9, 7, 7, 0, tzinfo=timezone.utc), us_only)
    assert (entry["name"], at) == ("PREMARKET_SCAN", datetime(2026, 9, 8, 13, 0, tzinfo=timezone.utc))


def test_next_session_on_saturday_names_the_real_next_day():
    entry, at = next_session_run(datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc))
    assert at.date() == date(2026, 9, 7) and entry["name"] == "XETRA_PRE"


def test_next_session_skips_midday_on_early_close_day():
    entry, at = next_session_run(datetime(2026, 11, 27, 14, 31, tzinfo=timezone.utc))
    assert (entry["name"], at.hour) == ("EOD", 22)


def test_last_session_run_walks_back_over_holiday():
    # Labor Day 15:00 UTC: the last session that existed is XETRA_OPEN
    # that morning, not US_OPEN (which did not fire).
    entry, at = last_session_run(datetime(2026, 9, 7, 15, 0, tzinfo=timezone.utc))
    assert (entry["name"], at.date()) == ("XETRA_OPEN", date(2026, 9, 7))
    entry, at = last_session_run(datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc))   # Sunday
    assert (entry["name"], at.date()) == ("EOD", date(2026, 9, 4))


def test_us_calendar_note():
    assert us_calendar_note(date(2026, 9, 23)) is None
    assert us_calendar_note(date(2026, 9, 7)) == "US market holiday"
    assert us_calendar_note(date(2026, 11, 27)) == "US early close 13:00 ET"
    assert us_calendar_note(date(2026, 9, 5)) == "weekend"
