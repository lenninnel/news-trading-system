"""Minimal US (NYSE/Nasdaq) trading-day calendar — pure Python, no network.

Used by the OHLC ingest freshness gate to compute the *expected* most recent
completed trading session for a given date, by the PositionManager /
PriceMonitor stale-feed guards to know whether a regular session is running
at all, by the PortfolioManager re-entry lock to count trading sessions
since a stop-loss exit, and — since 2026-09-23 — by ``config/sessions.py``
so the daemon, the watchdog, the API and the MCP server agree on which
scheduled sessions exist on a given day.

Two kinds of days are modelled:

* Full-day holidays (``us_market_holidays``): no session, no daily bar.
* Early closes (``us_early_closes``, 13:00 New York): a trading day with a
  daily bar, so every "is there a bar for day T" caller treats them as
  full sessions; only the *clock* callers (``is_us_rth``, ``us_rth_close``)
  see the shorter session.  NYSE early-close rule as published: the day
  after Thanksgiving; July 3 when it falls Mon–Thu (a Friday July 3 is the
  observed Independence Day holiday instead); December 24 when it falls
  Mon–Thu (a Friday December 24 is the observed Christmas holiday).

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
US_EARLY_CLOSE = time(13, 0)  # NYSE early-close days end at 13:00 New York


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


@lru_cache(maxsize=32)
def us_early_closes(year: int) -> frozenset[date]:
    """NYSE/Nasdaq early-close days (13:00 New York) for `year`.

    Published NYSE rule, cross-checked against the 2020–2027 calendars in
    tests/test_market_calendar.py: the Friday after Thanksgiving is always
    an early close; July 3 and December 24 are early closes only when they
    fall Monday–Thursday (on a Friday they are the observed holiday for
    July 4 / December 25, on a weekend there is no session at all).
    """
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    candidates = [thanksgiving + timedelta(days=1)]
    for month, day in ((7, 3), (12, 24)):
        d = date(year, month, day)
        if d.weekday() < 4:          # Mon–Thu
            candidates.append(d)
    holidays = us_market_holidays(year)
    return frozenset(d for d in candidates if d.weekday() < 5 and d not in holidays)


def is_us_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in us_market_holidays(d.year)


def is_us_early_close(d: date) -> bool:
    """True on a trading day whose regular session ends at 13:00 New York."""
    return d in us_early_closes(d.year)


def us_rth_close(d: date) -> time:
    """Regular-session close (New York wall clock) for trading day `d`:
    13:00 on an early-close day, 16:00 otherwise.  Undefined (returns the
    normal close) for non-trading days — check ``is_us_trading_day`` first.
    """
    return US_EARLY_CLOSE if is_us_early_close(d) else US_RTH_CLOSE


def us_rth_close_at(d: date) -> datetime:
    """Timezone-aware close of the regular session on trading day `d`."""
    return datetime.combine(d, us_rth_close(d), tzinfo=NY_TZ)


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
    not a full-day holiday) between 09:30 and the day's close (16:00, or
    13:00 on an early-close day) America/New_York.

    `now` must be timezone-aware; it is converted to New York time here.
    """
    if now.tzinfo is None:
        raise ValueError("is_us_rth needs a timezone-aware datetime")
    local = now.astimezone(NY_TZ)
    if not is_us_trading_day(local.date()):
        return False
    return US_RTH_OPEN <= local.time() < us_rth_close(local.date())


def next_us_rth_open(now: datetime) -> datetime:
    """Next 09:30 New York open strictly after `now` on a trading day."""
    if now.tzinfo is None:
        raise ValueError("next_us_rth_open needs a timezone-aware datetime")
    local = now.astimezone(NY_TZ)
    d = local.date()
    if not (is_us_trading_day(d) and local.time() < US_RTH_OPEN):
        d = next_us_trading_day(d)
    return datetime.combine(d, US_RTH_OPEN, tzinfo=NY_TZ)
