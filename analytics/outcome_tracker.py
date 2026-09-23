"""
Outcome tracker — fills 3d/5d/10d price outcomes for past signal events
from the ``daily_ohlc`` store.

Runs inside the EOD session (after the nightly ingest) and from the
``scripts/update_outcomes.py`` cron as a safety net.  Both are idempotent.

Price basis (2026-09-23)
------------------------
Outcomes come from the SAME store the strategies, the ATR stops and the
technical tier run on: ``daily_ohlc`` (Alpaca SIP, raw close, freshness-
gated nightly ingest).  There is no external fallback any more — until
2026-09-23 the tracker fetched Alpaca bars / yfinance per row, which
worked (20 EOD backfill runs in the 30 days before the switch, last fill
2026-09-22 22:20 UTC) but on a second, unaudited price path.

Horizon semantics (unchanged from the original tracker)
-------------------------------------------------------
``price_Nd`` is the store close of the last US trading day on or before
``signal_date + N calendar days`` (``data/market_calendar.py``).  The row
is filled only once the store actually holds that bar; until then it is
*pending*.  ``outcome_Nd_pct = (price_Nd - price_at_signal) / price_at_signal``.
``outcome_correct`` is rewritten on every horizon pass (so, as before, a
fully resolved row carries the sign of the 10d move) — kept identical on
purpose because RiskAgent's Kelly sizing reads it.

Row states (``outcome_status``)
-------------------------------
NULL            pending: horizon not reached or store bar not ingested yet
'filled'        all three horizons written from the store
'unevaluable'   can never be resolved; ``outcome_note`` says why:
                  sentinel        ticker is SESSION / MACRO / SCAN
                  no_store_bars   ticker has no daily_ohlc rows at all
                                  (non-US names, tier-2 scanner picks)
                  no_entry_price  price_at_signal missing and no store bar
                                  to backfill it from

Rows without ``price_at_signal`` (PreMarketScanner) on a store ticker get
their entry backfilled from the store — the close of the last completed
bar at signal time (T-1 before 22:00 UTC, T after) — and
``outcome_note='entry_backfilled_from_store'``.

Usage::

    python3 -m analytics.outcome_tracker            # fill everything due
    python3 -m analytics.outcome_tracker --dry-run  # report, write nothing
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from data.market_calendar import last_us_trading_day
from storage.database import Database

log = logging.getLogger(__name__)

# (days_offset, price_column, pct_column)
HORIZONS: list[tuple[int, str, str]] = [
    (3, "price_3d", "outcome_3d_pct"),
    (5, "price_5d", "outcome_5d_pct"),
    (10, "price_10d", "outcome_10d_pct"),
]

# Synthetic rows in signal_events that are not tradeable instruments.
SENTINEL_TICKERS: tuple[str, ...] = ("SESSION", "MACRO", "SCAN")

# Same cutoff as scripts/ingest_ohlc.py: from 22:00 UTC on, today's bar is
# the last completed one (US close 20:00 UTC in DST, 21:00 UTC in winter).
SAME_DAY_CUTOFF_UTC = 22

# Store gap tolerance when the calendar says a bar should exist but the
# store has none on that exact date (unscheduled closure): take the last
# stored bar within this many days before.
_GAP_TOLERANCE_DAYS = 4

_OUTCOME_COLUMNS: list[tuple[str, str]] = [
    ("outcome_status", "TEXT"),
    ("outcome_note", "TEXT"),
    ("outcome_source", "TEXT"),
    ("outcome_updated_at", "TEXT"),
]


@dataclass
class OutcomeSummary:
    """Per-run result; ``summary()`` is the one-line journal message."""

    filled: dict[str, int] = field(default_factory=lambda: {p: 0 for _, p, _ in HORIZONS})
    entry_backfilled: int = 0
    unevaluable: dict[str, int] = field(default_factory=dict)
    pending: int = 0
    rows_considered: int = 0
    store_tickers: int = 0
    dry_run: bool = False

    @property
    def total_filled(self) -> int:
        return sum(self.filled.values())

    @property
    def total_unevaluable(self) -> int:
        return sum(self.unevaluable.values())

    def summary(self) -> str:
        unev = ", ".join(f"{k}={v}" for k, v in sorted(self.unevaluable.items())) or "0"
        return (
            f"{'DRY-RUN ' if self.dry_run else ''}outcome tracker: "
            f"rows_considered={self.rows_considered} filled={self.total_filled} "
            f"{self.filled} entry_backfilled={self.entry_backfilled} "
            f"unevaluable={self.total_unevaluable} ({unev}) pending={self.pending} "
            f"store_tickers={self.store_tickers} source=daily_ohlc"
        )


def _is_directional(signal: str) -> int:
    """Return +1 for buy signals, -1 for sell signals, 0 otherwise."""
    s = (signal or "").upper()
    if "BUY" in s:
        return 1
    if "SELL" in s:
        return -1
    return 0


def _parse_ts(value: str) -> datetime:
    ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def entry_bar_date(signal_ts: datetime) -> date:
    """Date of the last COMPLETED daily bar at signal time.

    Mirrors what the technical tier saw: before 22:00 UTC the newest
    completed bar is the previous trading day's; from 22:00 UTC on (EOD
    after the ingest) it is today's.
    """
    d = signal_ts.date()
    if signal_ts.hour < SAME_DAY_CUTOFF_UTC:
        d -= timedelta(days=1)
    return last_us_trading_day(d)


def horizon_bar_date(signal_ts: datetime, days: int) -> date:
    """Trading day whose close is the ``days``-calendar-day outcome."""
    return last_us_trading_day(signal_ts.date() + timedelta(days=days))


def _close_on_or_before(closes: dict[str, float], target: date,
                        tolerance_days: int = _GAP_TOLERANCE_DAYS) -> float | None:
    for back in range(tolerance_days + 1):
        key = (target - timedelta(days=back)).isoformat()
        if key in closes:
            return closes[key]
    return None


def _ensure_columns(conn: sqlite3.Connection) -> None:
    for col, typedef in _OUTCOME_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE signal_events ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise


def run_outcome_tracker(
    db: Database | None = None,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
) -> OutcomeSummary:
    """Fill due outcome columns from the daily_ohlc store.

    Never raises on per-row problems (they are counted); a failure to read
    signal_events or the store propagates — the caller decides how loud.
    Always logs one summary line at INFO.
    """
    db = db or Database()
    now = now or datetime.now(timezone.utc)
    now_iso = now.isoformat()
    result = OutcomeSummary(dry_run=dry_run)
    sentinel_ph = ",".join("?" for _ in SENTINEL_TICKERS)

    with db._connect() as conn:
        _ensure_columns(conn)

        # 1. Sentinel rows: never evaluable, say so once.
        if not dry_run:
            cur = conn.execute(
                "UPDATE signal_events SET outcome_status='unevaluable', "
                "outcome_note='sentinel', outcome_updated_at=? "
                f"WHERE ticker IN ({sentinel_ph}) AND outcome_status IS NULL",
                (now_iso, *SENTINEL_TICKERS),
            )
            if cur.rowcount:
                result.unevaluable["sentinel"] = cur.rowcount
        else:
            n = conn.execute(
                f"SELECT COUNT(*) FROM signal_events WHERE ticker IN ({sentinel_ph}) "
                "AND outcome_status IS NULL", SENTINEL_TICKERS,
            ).fetchone()[0]
            if n:
                result.unevaluable["sentinel"] = n

        # 2. Candidate rows: at least the 3d horizon due, any horizon still
        #    open (a legacy row can have 5d/10d filled but 3d missing —
        #    the old yfinance path skipped holidays).
        min_days = HORIZONS[0][0]
        cutoff = (now - timedelta(days=min_days)).isoformat()
        rows = [dict(r) for r in conn.execute(
            "SELECT id, ticker, signal, price_at_signal, timestamp, "
            "price_3d, price_5d, price_10d "
            "FROM signal_events "
            "WHERE (outcome_status IS NULL OR outcome_status = 'pending') "
            "  AND (price_3d IS NULL OR price_5d IS NULL OR price_10d IS NULL) "
            f"  AND ticker NOT IN ({sentinel_ph}) "
            "  AND timestamp <= ? "
            "ORDER BY ticker, timestamp",
            (*SENTINEL_TICKERS, cutoff),
        ).fetchall()]
    result.rows_considered = len(rows)
    if not rows:
        log.info(result.summary())
        return result

    tickers = sorted({r["ticker"].upper() for r in rows})
    store_max = db.get_daily_ohlc_max_dates(tickers)
    store_tickers = [t for t in tickers if store_max.get(t)]
    result.store_tickers = len(store_tickers)

    # 3. Load store closes once per ticker (window: earliest signal → now).
    closes_by_ticker: dict[str, dict[str, float]] = {}
    for t in store_tickers:
        first_ts = min(_parse_ts(r["timestamp"]) for r in rows if r["ticker"].upper() == t)
        start = (first_ts.date() - timedelta(days=_GAP_TOLERANCE_DAYS + 3)).isoformat()
        bars = db.get_daily_ohlc(t, start, now.date().isoformat())
        closes_by_ticker[t] = {
            str(b["date"]): float(b["close"])
            for b in bars
            if b.get("close") is not None and float(b["close"]) > 0
        }

    updates: list[tuple[str, tuple]] = []   # (sql, params)

    def mark_unevaluable(row_id: int, reason: str) -> None:
        result.unevaluable[reason] = result.unevaluable.get(reason, 0) + 1
        updates.append((
            "UPDATE signal_events SET outcome_status='unevaluable', outcome_note=?, "
            "outcome_updated_at=? WHERE id=?",
            (reason, now_iso, row_id),
        ))

    for row in rows:
        ticker = row["ticker"].upper()
        if ticker not in closes_by_ticker:
            mark_unevaluable(row["id"], "no_store_bars")
            continue
        closes = closes_by_ticker[ticker]
        max_date = str(store_max[ticker])
        try:
            signal_ts = _parse_ts(row["timestamp"])
        except Exception:
            mark_unevaluable(row["id"], "bad_timestamp")
            continue

        entry = row["price_at_signal"]
        note = None
        if not entry or entry <= 0:
            entry = _close_on_or_before(closes, entry_bar_date(signal_ts))
            if entry is None:
                mark_unevaluable(row["id"], "no_entry_price")
                continue
            note = "entry_backfilled_from_store"
            result.entry_backfilled += 1

        sets: list[str] = []
        params: list = []
        direction = _is_directional(row["signal"])
        filled_now = {p: row[p] is not None for _, p, _ in HORIZONS}
        for days, price_col, pct_col in HORIZONS:
            if row[price_col] is not None:
                continue
            bar_date = horizon_bar_date(signal_ts, days)
            if max_date < bar_date.isoformat():
                break   # store not there yet → this and later horizons pending
            price = _close_on_or_before(closes, bar_date)
            if price is None:
                break   # gap in the store → keep pending, surfaces in counts
            pct = (price - entry) / entry * 100.0
            correct = (
                (1 if (direction > 0 and pct > 0) or (direction < 0 and pct < 0) else 0)
                if direction != 0 else None
            )
            sets += [f"{price_col}=?", f"{pct_col}=?", "outcome_correct=?"]
            params += [price, pct, correct]
            filled_now[price_col] = True
            result.filled[price_col] += 1

        if not sets and note is None:
            result.pending += 1
            continue

        if note is not None:
            sets += ["price_at_signal=?", "outcome_note=?"]
            params += [entry, note]
        status = "filled" if all(filled_now.values()) else "pending"
        if status == "pending":
            result.pending += 1
        sets += ["outcome_status=?", "outcome_source='daily_ohlc'", "outcome_updated_at=?"]
        params += [status, now_iso]
        updates.append((
            f"UPDATE signal_events SET {', '.join(sets)} WHERE id=?",
            (*params, row["id"]),
        ))

    if updates and not dry_run:
        with db._connect() as conn:
            for sql, params in updates:
                conn.execute(sql, params)
            conn.commit()

    log.info(result.summary())
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Fill 3d/5d/10d outcomes on signal_events from daily_ohlc",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be written; change nothing.")
    parser.add_argument("--db", default=None,
                        help="SQLite path (default: the configured DB_PATH).")
    args = parser.parse_args()

    db = Database(args.db) if args.db else Database()
    result = run_outcome_tracker(db, dry_run=args.dry_run)
    print(result.summary())


if __name__ == "__main__":
    main()
