"""
Benzinga earnings feed (via Polygon/Massive) — Q-005.

RECORDED-ONLY observability.  This module fetches the latest *reported*
earnings record for a ticker so the Coordinator can log it into
``pead_signal_log`` in parallel to the live yfinance source.  The point
is to accrue a per-evaluation yfinance-vs-Benzinga comparison so a
future, gated source decision can be made on real disagreement data.

Benzinga data is NEVER read by the PEAD signal, sizing, threshold, or
trade logic — this is NOT a source switch.

Endpoint (same POLYGON_API_KEY / apiKey query param as the OHLC ingest)::

    GET https://api.polygon.io/benzinga/v1/earnings
        ?ticker={ticker}&sort=date.desc&limit=10&apiKey={key}

Error contract (load-bearing for the freeze-safe test):
    * Network / HTTP / parse errors MAY raise — the caller
      (Coordinator._record_benzinga_earnings) wraps every call in
      try/except → log.warning → swallow.  We deliberately do NOT
      catch-and-return-None for network errors, so the forced-exception
      test stays honest.
    * Missing data (no results, or no reported record) is a NORMAL
      ``None`` return, not an exception.

Live field-name confirmation (Q-005 STEP 0, 2026-05-29, AAPL reported
record) — the reported-earnings record carries: ``date``, ``time``,
``date_status`` (confirmed|projected), ``actual_eps``, ``estimated_eps``,
``eps_surprise_percent`` (a FRACTION — see mapping below), ``eps_method``
(raw literal e.g. "gaap"), ``importance`` (0-5).  Future/projected
records (the top ``date.desc`` rows) lack ``actual_eps`` and the surprise
fields entirely, which is why the selection skips them.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io/benzinga/v1/earnings"


def _derive_eps_time(time_str: "str | None") -> "str | None":
    """Map a Benzinga ``time`` (HH:MM:SS, US/Eastern) to a session bucket.

        time < "09:30:00"   → "BMO"  (before market open)
        time >= "16:00:00"  → "AMC"  (after market close)
        otherwise           → "DMH"  (during market hours)
        null / empty        → None

    String comparison is valid because Benzinga returns zero-padded
    HH:MM:SS (confirmed live: "16:30:00", "09:30:00").
    """
    if not time_str:
        return None
    t = time_str.strip()
    if not t:
        return None
    if t < "09:30:00":
        return "BMO"
    if t >= "16:00:00":
        return "AMC"
    return "DMH"


def fetch_latest_reported_earnings(
    ticker: str,
    *,
    api_key: str,
    timeout: float = 4.0,
) -> "dict | None":
    """Return the most recent *reported* Benzinga earnings record for
    *ticker*, mapped to the pead_signal_log capture fields, or ``None``.

    "Reported" = ``date <= today`` AND ``actual_eps`` is not None (i.e.
    already announced, not a projected future date).  When the page holds
    no such record, returns ``None`` (a normal miss — e.g. EU names that
    Benzinga, being US-focused, does not cover).

    Args:
        ticker:   Equity symbol (e.g. "AAPL").
        api_key:  POLYGON_API_KEY (Benzinga entitlement on the same key).
        timeout:  Per-request timeout in seconds.

    Returns:
        dict with keys: announce_date, surprise_pct, actual_eps,
        estimate_eps, eps_method, date_status, importance, eps_time.
        ``None`` on a miss.

    Raises:
        requests.RequestException / ValueError on network, HTTP, or parse
        failure — the caller's fail-safe wrapper handles these.  Missing
        data is NOT an error (returns None).
    """
    params = {
        "ticker": ticker,
        "sort": "date.desc",
        "limit": 10,
        "apiKey": api_key,
    }
    resp = requests.get(_BASE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    results = resp.json().get("results") or []

    today = date.today()
    best: "dict | None" = None
    best_date: "date | None" = None
    for rec in results:
        if rec.get("actual_eps") is None:
            continue  # projected/future record — not yet reported
        date_str = rec.get("date")
        if not date_str:
            continue
        try:
            rec_date = date.fromisoformat(str(date_str)[:10])
        except (ValueError, TypeError):
            continue
        if rec_date > today:
            continue  # future date — skip (defensive; should have no actual_eps)
        if best_date is None or rec_date > best_date:
            best, best_date = rec, rec_date

    if best is None:
        return None

    # Surprise unit conversion — convert once, here at the mapping site.
    # Benzinga returns a fraction (0.0361 = 3.61%); yfinance surprise_pct
    # is in percent (24.04).  ×100 to keep the comparison column
    # unit-consistent — see Q-005.  Guard the null case: some reported
    # rows may lack eps_surprise_percent → store None, never None * 100.
    raw_surprise = best.get("eps_surprise_percent")
    surprise_pct = raw_surprise * 100 if raw_surprise is not None else None

    return {
        "announce_date": best.get("date"),
        "surprise_pct":  surprise_pct,
        "actual_eps":    best.get("actual_eps"),
        "estimate_eps":  best.get("estimated_eps"),
        # eps_method captured as the raw literal string (gaap|ffo|adj) —
        # NOT normalised/folded.
        "eps_method":    best.get("eps_method"),
        "date_status":   best.get("date_status"),
        "importance":    best.get("importance"),
        "eps_time":      _derive_eps_time(best.get("time")),
    }
