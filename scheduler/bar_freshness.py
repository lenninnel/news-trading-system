"""Bar-freshness gate — no signals on a stale daily bar without an alarm.

Since the D3 fix (2026-07-01) every strategy and the technical tier compute
their indicators on the last COMPLETED daily bar from the ``daily_ohlc``
store; the live price is used for sizing and the price guard only.  That
split is deliberate.  What was missing: nothing checked, at session start,
that the store actually holds the bar the session is supposed to run on.
If the nightly ingest (22:30 UTC) failed, every session until the next
successful ingest would silently evaluate yesterday's bar as if it were
today's.

This module answers one question per session: *is the newest bar in the
store the one this session expects?*

Expected bar
------------
* Intraday sessions (US_PRE, PEAD_OPEN, US_OPEN, XETRA_*): the last US
  trading day strictly BEFORE today — T-1.  Today's bar cannot exist yet.
* EOD (runs after the ingest, 22:45 UTC): the last US trading day on or
  before today — T.  On a US holiday that is the previous trading day.

Both use ``data/market_calendar.py`` (weekends + full-day NYSE holidays),
the same calendar the ingest freshness gate and the watchdog use.

Verdict
-------
Only tickers that exist in the store are gated (``max_date`` known).
Tickers with no store rows at all (scanner tier-2 candidates outside the
US-20 universe) are reported as ``unknown`` and left alone — the technical
agent already falls back to yfinance for those with a loud warning.

* no stale store ticker            → ``ok``; session runs unchanged
* some store tickers stale         → ``skip``; those tickers are removed
                                      from the session, the rest runs
* every store ticker stale         → ``abort``; the session produces no
                                      signals at all (a global ingest
                                      failure looks exactly like this)

The caller (``DailyScheduler._execute_run``) logs the verdict at ERROR,
sends a Telegram alert and annotates the ``session_runs`` row so the
watchdog reports the aborted session instead of counting it as "ran".

Relation to the existing checks (not duplicated)
-------------------------------------------------
* ``scripts/ingest_ohlc.py`` freshness gate: fails the INGEST run when the
  store is stale after writing.  Fires at 22:30 UTC once.
* ``scripts/watchdog.py`` ``ohlc`` check: alerts every 15 min from 23:00 UTC
  while ``MAX(date)`` is behind.
* This gate: the CONSUMER side — at the moment a session is about to
  compute signals, refuse to do so on the wrong bar.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from data.market_calendar import last_us_trading_day

log = logging.getLogger(__name__)

# Sessions that compute indicators on daily bars.  "monitor" (MIDDAY),
# "scanner" (PREMARKET_SCAN) and "pre_signal" (XETRA_PRE: news + sentiment
# only, no TA) never touch the store for a decision.
GATED_SESSION_TYPES = frozenset({"signal", "execution"})

# Sessions that run AFTER the nightly ingest and therefore expect today's
# bar.  Everything else expects T-1.
POST_INGEST_SESSIONS = frozenset({"EOD"})


def expected_bar_date(session_name: str, now: datetime | None = None) -> date:
    """The bar date a session must find as MAX(date) in the store."""
    now = now or datetime.now(timezone.utc)
    today = now.date()
    if session_name in POST_INGEST_SESSIONS:
        return last_us_trading_day(today)
    return last_us_trading_day(today - timedelta(days=1))


@dataclass
class BarFreshnessResult:
    session: str
    expected: date
    checked: list[str] = field(default_factory=list)   # tickers with store rows
    stale: dict[str, str | None] = field(default_factory=dict)  # ticker → max_date
    unknown: list[str] = field(default_factory=list)   # no store rows at all

    @property
    def verdict(self) -> str:
        if not self.stale:
            return "ok"
        if len(self.stale) >= len(self.checked):
            return "abort"
        return "skip"

    @property
    def fresh(self) -> list[str]:
        return [t for t in self.checked if t not in self.stale]

    def summary(self) -> str:
        exp = self.expected.isoformat()
        if self.verdict == "ok":
            return (
                f"bar freshness ok for {self.session}: {len(self.checked)} store "
                f"ticker(s) at {exp}"
                + (f", {len(self.unknown)} not in store" if self.unknown else "")
            )
        stale_str = ", ".join(
            f"{t}={d or 'none'}" for t, d in sorted(self.stale.items())
        )
        return (
            f"STALE BARS for {self.session}: expected MAX(date)={exp}, "
            f"{len(self.stale)}/{len(self.checked)} store ticker(s) behind "
            f"({stale_str}) → {self.verdict.upper()}"
        )


def check_bar_freshness(
    db,
    session_name: str,
    tickers: list[str],
    *,
    now: datetime | None = None,
) -> BarFreshnessResult:
    """Compare each ticker's stored MAX(date) with the expected bar date.

    ``db`` needs ``get_daily_ohlc_max_dates(tickers) -> {ticker: max_date}``
    (``storage.database.Database``).  Never raises on a DB error: the
    result then has no checked tickers and verdict ``ok`` — a broken store
    read is logged here and surfaces downstream as the technical agent's
    yfinance fallback warning; it is not a reason to invent staleness.
    """
    expected = expected_bar_date(session_name, now)
    result = BarFreshnessResult(session=session_name, expected=expected)
    if not tickers:
        return result
    try:
        max_dates = db.get_daily_ohlc_max_dates([t.upper() for t in tickers])
    except Exception as exc:
        log.warning("[%s] bar freshness check skipped — store read failed: %s",
                    session_name, exc)
        return result
    exp_iso = expected.isoformat()
    for t in tickers:
        key = t.upper()
        max_date = max_dates.get(key)
        if max_date is None:
            result.unknown.append(t)
            continue
        result.checked.append(t)
        if str(max_date) < exp_iso:
            result.stale[t] = str(max_date)
    return result
