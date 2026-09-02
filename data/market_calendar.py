"""Minimal US (NYSE/Nasdaq) trading-day calendar — pure Python, no network.

Used by the OHLC ingest freshness gate to compute the *expected* most recent
completed trading session for a given date. Full-day holidays only; early
closes (half days) still produce a daily bar and need no special handling.

Known limitation: unscheduled closures (e.g. a national day of mourning) are
not modelled. On such a day the freshness gate fails once with a clear
message — a human seeing "market was closed" can ignore that single alert.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache


def _easter_sunday(year: int) -> date:
    """Gregorian Easter via the anonymous (Meeus/Jones/Butcher) computus."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th `weekday` (Mon=0) of `month`; n=-1 means the last one."""
    if n > 0:
        d = date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + timedelta(days=offset + 7 * (n - 1))
    d = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def _observed(d: date) -> date:
    """NYSE observed-date shift: Sat -> Fri, Sun -> Mon."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=32)
def us_market_holidays(year: int) -> frozenset[date]:
    """Full-day NYSE/Nasdaq holidays for `year` (observed dates)."""
    easter = _easter_sunday(year)
    fixed = [
        date(year, 1, 1),    # New Year's Day
        date(year, 6, 19),   # Juneteenth (since 2022)
        date(year, 7, 4),    # Independence Day
        date(year, 12, 25),  # Christmas Day
    ]
    floating = [
        _nth_weekday(year, 1, 0, 3),    # MLK Day: 3rd Monday of January
        _nth_weekday(year, 2, 0, 3),    # Washington's Birthday: 3rd Mon of Feb
        easter - timedelta(days=2),     # Good Friday
        _nth_weekday(year, 5, 0, -1),   # Memorial Day: last Monday of May
        _nth_weekday(year, 9, 0, 1),    # Labor Day: 1st Monday of September
        _nth_weekday(year, 11, 3, 4),   # Thanksgiving: 4th Thursday of November
    ]
    holidays = {_observed(d) for d in fixed} | set(floating)
    # An observed New Year's shifted to Dec 31 of the *previous* year: when
    # Jan 1 of NEXT year falls on a Saturday, this year's Dec 31 is a holiday.
    if date(year + 1, 1, 1).weekday() == 5:
        holidays.add(date(year, 12, 31))
    return frozenset(h for h in holidays if h.year == year)


def is_us_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in us_market_holidays(d.year)


def last_us_trading_day(d: date) -> date:
    """Most recent US trading day on or before `d`."""
    while not is_us_trading_day(d):
        d -= timedelta(days=1)
    return d
