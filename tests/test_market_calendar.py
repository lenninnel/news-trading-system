"""Tests for data/market_calendar.py — US trading-day calendar.

Holiday dates cross-checked against the published NYSE 2026 schedule.

Run:
    python3 -m pytest tests/test_market_calendar.py -v
"""
from __future__ import annotations

import os
import pytest
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timezone

from data.market_calendar import (
    NY_TZ,
    is_us_rth,
    is_us_trading_day,
    last_us_trading_day,
    next_us_rth_open,
    next_us_trading_day,
    us_market_holidays,
    us_sessions_between,
)


def test_2026_holiday_set_exact():
    expected = {
        date(2026, 1, 1),    # New Year's Day
        date(2026, 1, 19),   # MLK Day
        date(2026, 2, 16),   # Washington's Birthday
        date(2026, 4, 3),    # Good Friday (Easter = Apr 5)
        date(2026, 5, 25),   # Memorial Day
        date(2026, 6, 19),   # Juneteenth
        date(2026, 7, 3),    # Independence Day observed (Jul 4 is a Saturday)
        date(2026, 9, 7),    # Labor Day
        date(2026, 11, 26),  # Thanksgiving
        date(2026, 12, 25),  # Christmas
    }
    assert us_market_holidays(2026) == expected


def test_2027_new_years_observed_and_good_friday():
    hol = us_market_holidays(2027)
    assert date(2027, 1, 1) in hol          # Friday, no shift
    assert date(2027, 3, 26) in hol         # Good Friday (Easter = Mar 28)
    assert date(2027, 12, 31) in hol        # New Year's 2028 falls on Saturday


def test_sunday_observed_shifts_to_monday():
    # Jul 4 2021 was a Sunday -> observed Mon Jul 5.
    assert date(2021, 7, 5) in us_market_holidays(2021)
    assert date(2021, 7, 4) not in us_market_holidays(2021)


def test_trading_day_checks():
    assert is_us_trading_day(date(2026, 8, 31))          # the incident Monday
    assert not is_us_trading_day(date(2026, 8, 30))      # Sunday
    assert not is_us_trading_day(date(2026, 9, 7))       # Labor Day


def test_last_trading_day_rolls_back_over_weekend_and_holiday():
    # Sunday 2026-08-30 -> Friday 2026-08-28
    assert last_us_trading_day(date(2026, 8, 30)) == date(2026, 8, 28)
    # Labor Day Monday 2026-09-07 -> Friday 2026-09-04
    assert last_us_trading_day(date(2026, 9, 7)) == date(2026, 9, 4)
    # A plain trading day maps to itself
    assert last_us_trading_day(date(2026, 9, 2)) == date(2026, 9, 2)


# ── session counting / RTH helpers (2026-09-08) ──────────────────────────────

def test_sessions_between_skips_holidays():
    # Thu 07-02 → Mon 07-06: Fri 07-03 is Independence Day observed
    assert us_sessions_between(date(2026, 7, 2), date(2026, 7, 6)) == 1
    # Fri 09-04 → Tue 09-08: Mon 09-07 is Labor Day
    assert us_sessions_between(date(2026, 9, 4), date(2026, 9, 8)) == 1
    assert us_sessions_between(date(2026, 9, 4), date(2026, 9, 11)) == 4
    assert us_sessions_between(date(2026, 9, 4), date(2026, 9, 15)) == 6


def test_sessions_between_same_day_and_reversed_are_zero():
    assert us_sessions_between(date(2026, 9, 8), date(2026, 9, 8)) == 0
    assert us_sessions_between(date(2026, 9, 8), date(2026, 9, 4)) == 0
    # Fri → Sat/Sun: no session yet
    assert us_sessions_between(date(2026, 9, 4), date(2026, 9, 6)) == 0


def test_next_us_trading_day():
    assert next_us_trading_day(date(2026, 9, 4)) == date(2026, 9, 8)   # over Labor Day
    assert next_us_trading_day(date(2026, 7, 2)) == date(2026, 7, 6)   # over Jul 3 + weekend
    assert next_us_trading_day(date(2026, 9, 8)) == date(2026, 9, 9)


def test_is_us_rth_holiday_and_window():
    labor_day = datetime(2026, 9, 7, 10, 0, tzinfo=NY_TZ)
    assert is_us_rth(labor_day) is False
    tue = datetime(2026, 9, 8, 10, 0, tzinfo=NY_TZ)
    assert is_us_rth(tue) is True
    assert is_us_rth(datetime(2026, 9, 8, 9, 29, tzinfo=NY_TZ)) is False
    assert is_us_rth(datetime(2026, 9, 8, 9, 30, tzinfo=NY_TZ)) is True
    assert is_us_rth(datetime(2026, 9, 8, 16, 0, tzinfo=NY_TZ)) is False
    # UTC input is converted: 14:30 UTC = 10:30 EDT
    assert is_us_rth(datetime(2026, 9, 8, 14, 30, tzinfo=timezone.utc)) is True
    assert is_us_rth(datetime(2026, 9, 5, 14, 30, tzinfo=timezone.utc)) is False  # Saturday


def test_is_us_rth_rejects_naive():
    import pytest
    with pytest.raises(ValueError):
        is_us_rth(datetime(2026, 9, 8, 10, 0))


def test_next_us_rth_open():
    fri_close = datetime(2026, 9, 4, 16, 0, tzinfo=NY_TZ)
    assert next_us_rth_open(fri_close) == datetime(2026, 9, 8, 9, 30, tzinfo=NY_TZ)
    early = datetime(2026, 9, 8, 9, 0, tzinfo=NY_TZ)
    assert next_us_rth_open(early) == datetime(2026, 9, 8, 9, 30, tzinfo=NY_TZ)
    at_open = datetime(2026, 9, 8, 9, 30, tzinfo=NY_TZ)
    assert next_us_rth_open(at_open) == datetime(2026, 9, 9, 9, 30, tzinfo=NY_TZ)
    labor_day = datetime(2026, 9, 7, 8, 0, tzinfo=NY_TZ)
    assert next_us_rth_open(labor_day) == datetime(2026, 9, 8, 9, 30, tzinfo=NY_TZ)


# ── Early closes (13:00 New York) — cross-checked against the published
# NYSE calendars 2020–2027 ─────────────────────────────────────────────

from data.market_calendar import (  # noqa: E402
    US_EARLY_CLOSE,
    US_RTH_CLOSE,
    is_us_early_close,
    us_early_closes,
    us_rth_close,
    us_rth_close_at,
)


@pytest.mark.parametrize("year, expected", [
    # Published NYSE early closings (1:00 p.m. ET).
    (2020, {date(2020, 11, 27), date(2020, 12, 24)}),                     # Jul 3 = observed holiday
    (2021, {date(2021, 11, 26)}),                                         # Jul 5 holiday, Dec 24 observed holiday
    (2022, {date(2022, 11, 25)}),                                         # Jul 4 Mon, Dec 26 observed
    (2023, {date(2023, 7, 3), date(2023, 11, 24)}),                       # Dec 24 = Sunday
    (2024, {date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24)}),
    (2025, {date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24)}),
    (2026, {date(2026, 11, 27), date(2026, 12, 24)}),                     # Jul 3 = observed holiday
    (2027, {date(2027, 11, 26)}),                                         # Jul 5 holiday, Dec 24 observed holiday
])
def test_early_closes_match_published_nyse_calendar(year, expected):
    assert set(us_early_closes(year)) == expected


def test_early_close_days_are_trading_days_never_holidays():
    for year in range(2020, 2028):
        for d in us_early_closes(year):
            assert is_us_trading_day(d)
            assert d not in us_market_holidays(year)


def test_us_rth_close_per_day():
    assert us_rth_close(date(2026, 11, 27)) == US_EARLY_CLOSE   # Fri after Thanksgiving
    assert us_rth_close(date(2026, 12, 24)) == US_EARLY_CLOSE   # Christmas Eve (Thu)
    assert us_rth_close(date(2026, 9, 23)) == US_RTH_CLOSE      # ordinary Wednesday
    assert is_us_early_close(date(2026, 11, 27)) and not is_us_early_close(date(2026, 11, 25))
    at = us_rth_close_at(date(2026, 11, 27))
    assert at.tzinfo is not None and at.hour == 13 and at.minute == 0


def test_is_us_rth_respects_early_close():
    early = date(2026, 11, 27)
    assert is_us_rth(datetime(early.year, early.month, early.day, 12, 59, tzinfo=NY_TZ))
    assert not is_us_rth(datetime(early.year, early.month, early.day, 13, 0, tzinfo=NY_TZ))
    assert not is_us_rth(datetime(early.year, early.month, early.day, 15, 0, tzinfo=NY_TZ))
    # The Wednesday before is a full session.
    assert is_us_rth(datetime(2026, 11, 25, 15, 0, tzinfo=NY_TZ))
