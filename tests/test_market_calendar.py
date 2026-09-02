"""Tests for data/market_calendar.py — US trading-day calendar.

Holiday dates cross-checked against the published NYSE 2026 schedule.

Run:
    python3 -m pytest tests/test_market_calendar.py -v
"""
from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.market_calendar import (
    is_us_trading_day,
    last_us_trading_day,
    us_market_holidays,
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
