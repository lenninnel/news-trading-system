"""
Earnings calendar and event risk detection.

Provides utilities to check how close a ticker is to its next earnings
report date, using yfinance as the data source.  Results are cached in a
module-level dict with a 4-hour TTL to avoid repeated API calls.

Functions
---------
get_earnings_date(ticker) -> date | None
    Next earnings date from yfinance.
get_days_to_earnings(ticker) -> int | None
    Trading days until earnings (None if unknown).
is_earnings_week(ticker) -> bool
    True if earnings are within 5 trading days.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

# ── Module-level cache with 4-hour TTL ────────────────────────────────────
_CACHE_TTL_SECONDS = 4 * 60 * 60  # 4 hours

# {ticker: {"earnings_date": date | None, "fetched_at": float}}
_cache: dict[str, dict[str, Any]] = {}


def _is_cache_valid(ticker: str) -> bool:
    """Check if a cached entry exists and hasn't expired."""
    entry = _cache.get(ticker)
    if entry is None:
        return False
    age = datetime.now(timezone.utc).timestamp() - entry["fetched_at"]
    return age < _CACHE_TTL_SECONDS


def clear_cache() -> None:
    """Clear the earnings cache (useful for testing)."""
    _cache.clear()


def get_earnings_date(ticker: str) -> "date | None":
    """
    Return the next earnings report date for *ticker*, or None if unknown.

    Uses yfinance's Ticker.calendar property.  Results are cached for 4 hours.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").

    Returns:
        A datetime.date or None if yfinance has no earnings data.
    """
    ticker = ticker.upper()

    if _is_cache_valid(ticker):
        return _cache[ticker]["earnings_date"]

    try:
        t = yf.Ticker(ticker)
        cal = t.calendar

        if cal is None or (hasattr(cal, "empty") and cal.empty):
            _cache[ticker] = {
                "earnings_date": None,
                "fetched_at": datetime.now(timezone.utc).timestamp(),
            }
            return None

        # yfinance returns calendar as a dict or DataFrame depending on version
        earnings_dt = None
        if isinstance(cal, dict):
            # Keys may include 'Earnings Date', which can be a list of dates
            raw = cal.get("Earnings Date")
            if isinstance(raw, list) and raw:
                raw = raw[0]
            if raw is not None:
                if isinstance(raw, datetime):
                    earnings_dt = raw.date()
                elif isinstance(raw, date):
                    earnings_dt = raw
        else:
            # DataFrame path (older yfinance versions)
            import pandas as pd

            if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.columns:
                val = cal["Earnings Date"].iloc[0]
                if pd.notna(val):
                    if isinstance(val, (datetime, pd.Timestamp)):
                        earnings_dt = val.date()
                    elif isinstance(val, date):
                        earnings_dt = val

        _cache[ticker] = {
            "earnings_date": earnings_dt,
            "fetched_at": datetime.now(timezone.utc).timestamp(),
        }
        return earnings_dt

    except Exception as exc:
        logger.warning("Failed to fetch earnings date for %s: %s", ticker, exc)
        _cache[ticker] = {
            "earnings_date": None,
            "fetched_at": datetime.now(timezone.utc).timestamp(),
        }
        return None


def _trading_days_between(start: date, end: date) -> int:
    """
    Count trading days (weekdays) between *start* and *end*, exclusive of start.

    Returns a negative number if *end* is before *start*.
    """
    if end <= start:
        return 0

    count = 0
    current = start
    from datetime import timedelta

    delta = timedelta(days=1)
    current = current + delta
    while current <= end:
        if current.weekday() < 5:  # Mon–Fri
            count += 1
        current += delta
    return count


def get_days_to_earnings(ticker: str) -> "int | None":
    """
    Return the number of trading days until the next earnings report.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Integer trading days (0 = earnings today), or None if unknown.
    """
    earnings_dt = get_earnings_date(ticker)
    if earnings_dt is None:
        return None

    today = date.today()
    if earnings_dt < today:
        return None  # earnings already passed

    if earnings_dt == today:
        return 0

    return _trading_days_between(today, earnings_dt)


def is_earnings_week(ticker: str) -> bool:
    """
    Return True if earnings are within 5 trading days.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        True if days_to_earnings <= 5, False otherwise (including unknown).
    """
    days = get_days_to_earnings(ticker)
    if days is None:
        return False
    return days <= 5
