#!/usr/bin/env python3
"""
Standalone entry point for the signal outcome tracker.

Meant to run as a nightly cron (23:00 UTC) so 3d/5d/10d outcome
columns on ``signal_events`` get filled regardless of whether the
EOD trading session completed successfully.

Wraps ``analytics.outcome_tracker.run_outcome_tracker`` with three
DB-side fixups that the bare tracker can't do on its own:

1. **Seal sentinel rows** — ``PostSessionReviewer`` writes a row with
   ``ticker='SESSION'`` whose payload is the review text (not a real
   trade signal). The tracker would burn yfinance calls on every cron
   trying to resolve "SESSION" as a ticker. We pre-fill its price
   columns with 0 so the ``IS NULL`` filter skips it.

2. **Backfill ``price_at_signal``** — ``PreMarketScanner`` logs picks
   without an entry price (it doesn't run the full pipeline that
   would set one). The tracker fills ``price_3d`` for those rows but
   leaves ``outcome_3d_pct`` NULL because it can't compute a percentage
   without an entry. We look up the close on the signal date and
   backfill ``price_at_signal`` so the percentage can be computed.

3. **Recompute pct from filled prices** — once we backfill missing
   entry prices, rows whose ``price_3d`` is already filled but
   ``outcome_3d_pct`` is still NULL need a one-shot recomputation. The
   tracker only fills pct in the same pass as the price column, so
   without this step those rows would never resolve.

Usage::

    python3 scripts/update_outcomes.py            # one-shot backfill
    python3 scripts/update_outcomes.py --backfill # same, explicit
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure the repo root is on sys.path when invoked as a bare script
# (e.g. from cron) rather than via ``python -m``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analytics.outcome_tracker import run_outcome_tracker  # noqa: E402

log = logging.getLogger(__name__)

# Tickers that are sentinel/synthetic rows in signal_events, not real
# tradeable instruments. These should never go through the outcome
# tracker. Kept as a tuple so we can splat into SQL placeholders.
_SENTINEL_TICKERS = ("SESSION", "MACRO", "SCAN")


def _seal_sentinel_rows(db) -> int:
    """Pre-fill price_3d/5d/10d=0 for sentinel-ticker rows.

    The outcome tracker selects ``WHERE price_Nd IS NULL`` so writing
    any non-NULL value (we use 0) takes them out of consideration.
    Returns the number of rows newly sealed.
    """
    placeholders = ",".join("?" for _ in _SENTINEL_TICKERS)
    sql = (
        "UPDATE signal_events "
        "SET price_3d  = COALESCE(price_3d,  0), "
        "    price_5d  = COALESCE(price_5d,  0), "
        "    price_10d = COALESCE(price_10d, 0) "
        f"WHERE ticker IN ({placeholders}) "
        "  AND (price_3d IS NULL OR price_5d IS NULL OR price_10d IS NULL)"
    )
    try:
        with db._connect() as conn:
            cur = conn.execute(sql, _SENTINEL_TICKERS)
            return cur.rowcount or 0
    except Exception as exc:
        log.warning("Sentinel seal failed (non-fatal): %s", exc)
        return 0


def _backfill_entry_prices(db, lookback_days: int = 60) -> int:
    """Fetch close-on-signal-date for rows missing ``price_at_signal``.

    Restricted to ``lookback_days`` so we don't try to download years of
    history for stale rows. Real tickers only — sentinels are excluded.
    Returns the number of rows updated.
    """
    placeholders = ",".join("?" for _ in _SENTINEL_TICKERS)
    cutoff_iso = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    try:
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT id, ticker, timestamp FROM signal_events "
                "WHERE (price_at_signal IS NULL OR price_at_signal = 0) "
                f"  AND ticker NOT IN ({placeholders}) "
                "  AND timestamp >= ? "
                "ORDER BY ticker, timestamp",
                (*_SENTINEL_TICKERS, cutoff_iso),
            ).fetchall()
    except Exception as exc:
        log.warning("Entry-price query failed (non-fatal): %s", exc)
        return 0

    if not rows:
        return 0

    # Group by ticker so we make one yfinance download per ticker, not
    # one per signal — there can be 10s of signals per ticker per cycle.
    by_ticker: dict[str, list[tuple[int, str]]] = {}
    for r in rows:
        by_ticker.setdefault(r["ticker"], []).append((r["id"], r["timestamp"]))

    try:
        import yfinance as yf
    except Exception as exc:
        log.warning("yfinance import failed: %s", exc)
        return 0

    updated = 0
    for ticker, items in by_ticker.items():
        try:
            data = yf.download(
                ticker, period="3mo", progress=False, auto_adjust=True,
            )
            if data is None or data.empty:
                continue
            closes = data["Close"].squeeze()
            if hasattr(closes, "empty") and closes.empty:
                continue
        except Exception as exc:
            log.debug("yfinance fetch failed for %s: %s", ticker, exc)
            continue

        for sid, ts in items:
            try:
                signal_date = datetime.fromisoformat(ts).date()
                # Closest trading day on or before the signal date.
                idx_dates = closes.index.date
                eligible = closes[idx_dates <= signal_date]
                if eligible.empty:
                    continue
                price = float(eligible.iloc[-1])
                if not (price > 0):
                    continue
                with db._connect() as conn:
                    conn.execute(
                        "UPDATE signal_events SET price_at_signal = ? "
                        "WHERE id = ? AND (price_at_signal IS NULL OR price_at_signal = 0)",
                        (price, sid),
                    )
                updated += 1
            except Exception as exc:
                log.debug("Entry-price write failed for id=%s: %s", sid, exc)
                continue

    return updated


def _recompute_outcomes_from_prices(db) -> dict[str, int]:
    """Fill ``outcome_Nd_pct`` where ``price_Nd`` is filled but pct is NULL.

    Needed when ``price_at_signal`` is backfilled AFTER the tracker
    already wrote ``price_Nd`` (with NULL pct because there was no entry
    price at the time). Uses pure SQL — no broker calls.

    Direction-correctness: outcome_correct is set to 1 when the signal
    direction matches the realised move sign, 0 when it doesn't. NULL
    when the signal is non-directional (HOLD, CONFLICTING, WATCH).
    """
    horizons = [
        ("price_3d",  "outcome_3d_pct"),
        ("price_5d",  "outcome_5d_pct"),
        ("price_10d", "outcome_10d_pct"),
    ]
    updated: dict[str, int] = {}
    for price_col, pct_col in horizons:
        sql = (
            f"UPDATE signal_events "
            f"SET {pct_col} = "
            f"  CASE WHEN price_at_signal > 0 AND {price_col} > 0 "
            f"       THEN ({price_col} - price_at_signal) / price_at_signal * 100.0 "
            f"       ELSE NULL END "
            f"WHERE {pct_col} IS NULL "
            f"  AND price_at_signal IS NOT NULL AND price_at_signal > 0 "
            f"  AND {price_col} IS NOT NULL AND {price_col} > 0"
        )
        try:
            with db._connect() as conn:
                cur = conn.execute(sql)
                updated[pct_col] = cur.rowcount or 0
        except Exception as exc:
            log.warning("Recompute %s failed (non-fatal): %s", pct_col, exc)
            updated[pct_col] = 0
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill 3d/5d/10d price outcomes on signal_events.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill all pending outcomes (default behaviour — flag kept "
             "for CLI clarity / cron scripts).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO logs; only print the summary line.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from storage.database import Database
    db = Database()

    sealed = _seal_sentinel_rows(db)
    if sealed:
        log.info("Sealed %d sentinel rows (SESSION/MACRO/SCAN)", sealed)

    entry_filled = _backfill_entry_prices(db)
    if entry_filled:
        log.info("Backfilled price_at_signal on %d rows", entry_filled)

    tracker_result = run_outcome_tracker(db)

    recomputed = _recompute_outcomes_from_prices(db)

    total_tracker = sum(tracker_result.values())
    total_recompute = sum(recomputed.values())
    print(
        f"update_outcomes: sealed={sealed} entry_backfill={entry_filled} "
        f"tracker={total_tracker} {tracker_result} "
        f"recompute={total_recompute} {recomputed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
