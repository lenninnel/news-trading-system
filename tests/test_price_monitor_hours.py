"""PriceMonitor market-hours helpers are holiday-aware (2026-09-08).

The monitor is not a running service on the VPS (nts-trading's
PositionManager is the live stale guard), but it shares the same premise
and is fixed alongside it.

Run: python3 -m pytest tests/test_price_monitor_hours.py -v
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from monitoring.price_monitor import PriceMonitor  # noqa: E402

_ET = ZoneInfo("America/New_York")


def _monitor() -> PriceMonitor:
    # Bypass __init__ (config file, DB, trader) — the helpers under test only
    # read the clock and the calendar.
    return PriceMonitor.__new__(PriceMonitor)


def _frozen(at: datetime):
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return at.astimezone(tz) if tz else at
    return patch("monitoring.price_monitor.datetime", _Frozen)


def test_labor_day_is_closed_for_us():
    # 10:00 ET on Labor Day = 16:00 Berlin → XETRA (weekday-only) still counts
    # as open; so pick 17:45 Berlin = 11:45 ET to isolate the US leg.
    with _frozen(datetime(2026, 9, 7, 11, 45, tzinfo=_ET)):
        assert _monitor()._is_market_hours() is False
        assert _monitor()._market_status() == "US holiday, EU closed"


def test_regular_tuesday_is_open():
    with _frozen(datetime(2026, 9, 8, 11, 45, tzinfo=_ET)):
        assert _monitor()._is_market_hours() is True
        assert _monitor()._market_status() == "US open, EU closed"


def test_seconds_until_open_skips_holiday_weekend():
    fri_close = datetime(2026, 9, 4, 16, 5, tzinfo=_ET)
    with _frozen(fri_close):
        secs = _monitor()._seconds_until_open()
    expected = (datetime(2026, 9, 8, 9, 30, tzinfo=_ET) - fri_close).total_seconds()
    assert secs == expected


def test_early_close_friday_after_thanksgiving():
    # 14:30 ET = 20:30 Berlin → XETRA closed; the US leg alone decides.
    with _frozen(datetime(2026, 11, 27, 14, 30, tzinfo=_ET)):
        assert _monitor()._is_market_hours() is False
        assert _monitor()._market_status() == "US closed, EU closed"
    with _frozen(datetime(2026, 11, 27, 12, 30, tzinfo=_ET)):
        assert _monitor()._is_market_hours() is True
        assert _monitor()._market_status() == "US open, EU closed"
