"""Minimal US (NYSE/Nasdaq) trading-day calendar — pure Python, no network.

Used by the OHLC ingest freshness gate to compute the *expected* most recent
completed trading session for a given date, by the PositionManager /
PriceMonitor stale-feed guards to know whether a regular session is running
at all, and by the PortfolioManager re-entry lock to count trading sessions
since a stop-loss exit. Full-day holidays only; early closes (half days)
still produce a daily bar, are treated as full sessions here, and need no
special handling for any of those callers.

Known limitation: unscheduled closures (e.g. a national day of mourning) are
not modelled. On such a day the freshness gate fails once with a clear
message — a human seeing "market was closed" can ignore that single alert.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
US_RTH_OPEN = time(9, 30)    # regular trading hours, America/New_York
US_RTH_CLOSE = time(16, 0)


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


def next_us_trading_day(d: date) -> date:
    """First US trading day strictly after `d`."""
    d += timedelta(days=1)
    while not is_us_trading_day(d):
        d += timedelta(days=1)
    return d


def us_sessions_between(start: date, end: date) -> int:
    """Number of US trading sessions in the half-open range (start, end].

    0 when `end` is on or before `start`. Same-day = 0, the next trading day
    = 1 — the counter the re-entry lock and its audit use, so a holiday
    (e.g. Fri 2026-07-03) between a stop and a re-entry does not count as a
    session the ticker sat out.
    """
    if end <= start:
        return 0
    n = 0
    d = start
    while d < end:
        d += timedelta(days=1)
        if is_us_trading_day(d):
            n += 1
    return n


def is_us_rth(now: datetime) -> bool:
    """True while the US regular session is open: a trading day (weekday,
    not a full-day holiday) between 09:30 and 16:00 America/New_York.

    `now` must be timezone-aware; it is converted to New York time here.
    """
    if now.tzinfo is None:
        raise ValueError("is_us_rth needs a timezone-aware datetime")
    local = now.astimezone(NY_TZ)
    if not is_us_trading_day(local.date()):
        return False
    return US_RTH_OPEN <= local.time() < US_RTH_CLOSE


def next_us_rth_open(now: datetime) -> datetime:
    """Next 09:30 New York open strictly after `now` on a trading day."""
    if now.tzinfo is None:
        raise ValueError("next_us_rth_open needs a timezone-aware datetime")
    local = now.astimezone(NY_TZ)
    d = local.date()
    if not (is_us_trading_day(d) and local.time() < US_RTH_OPEN):
        d = next_us_trading_day(d)
    return datetime.combine(d, US_RTH_OPEN, tzinfo=NY_TZ)
