"""
Outcome tracker — backfills price outcomes for past signal events.

Runs separately from the trading daemon.  For each signal older than N days
whose price_Nd column is still NULL, it fetches the historical price from
Alpaca (or yfinance as fallback) and fills in the outcome columns.

Usage::

    python3 -m analytics.outcome_tracker          # backfill all pending
    python3 -m analytics.outcome_tracker --days 5  # only 5-day outcomes
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone

from storage.database import Database

log = logging.getLogger(__name__)

# (column_name, days_offset, pct_column, price_column)
_HORIZONS = [
    ("price_3d",  3,  "outcome_3d_pct",  "price_3d"),
    ("price_5d",  5,  "outcome_5d_pct",  "price_5d"),
    ("price_10d", 10, "outcome_10d_pct", "price_10d"),
]


def _fetch_price(ticker: str, target_date: datetime) -> float | None:
    """Fetch close price for *ticker* on or near *target_date*.

    Tries Alpaca bars first, falls back to yfinance.
    """
    try:
        from data.alpaca_data import AlpacaDataClient
        client = AlpacaDataClient()
        start = target_date - timedelta(days=3)
        bars = client.get_bars(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(target_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            timeframe="1Day",
        )
        if bars is not None and not bars.empty:
            return float(bars["Close"].squeeze().iloc[-1])
    except Exception as exc:
        log.debug("Alpaca price fetch failed for %s: %s", ticker, exc)

    try:
        import yfinance as yf
        start = target_date - timedelta(days=3)
        end = target_date + timedelta(days=1)
        data = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
        )
        if data is not None and not data.empty:
            return float(data["Close"].squeeze().iloc[-1])
    except Exception as exc:
        log.debug("yfinance price fetch failed for %s: %s", ticker, exc)

    return None


def _is_directional(signal: str) -> int:
    """Return +1 for buy signals, -1 for sell signals, 0 otherwise."""
    s = (signal or "").upper()
    if "BUY" in s:
        return 1
    if "SELL" in s:
        return -1
    return 0


def run_outcome_tracker(db: Database | None = None) -> dict:
    """Backfill outcome columns for eligible signal events.

    Returns a summary dict with counts of updated rows per horizon.
    """
    db = db or Database()
    now = datetime.now(timezone.utc)
    updated: dict[str, int] = {}

    for price_col, days_offset, pct_col, _ in _HORIZONS:
        cutoff = (now - timedelta(days=days_offset)).isoformat()
        try:
            with db._connect() as conn:
                rows = conn.execute(
                    f"SELECT id, ticker, signal, price_at_signal, timestamp "
                    f"FROM signal_events "
                    f"WHERE {price_col} IS NULL AND timestamp <= ?",
                    (cutoff,),
                ).fetchall()
        except Exception as exc:
            log.warning("Failed to query signal_events for %s: %s", price_col, exc)
            continue

        count = 0
        for row in rows:
            row = dict(row)
            ticker = row["ticker"]
            signal_ts = datetime.fromisoformat(row["timestamp"])
            target_date = signal_ts + timedelta(days=days_offset)
            price = _fetch_price(ticker, target_date)

            if price is None:
                continue

            entry_price = row["price_at_signal"]
            if entry_price and entry_price > 0:
                pct = ((price - entry_price) / entry_price) * 100
                direction = _is_directional(row["signal"])
                if direction != 0:
                    correct = 1 if (direction > 0 and pct > 0) or (direction < 0 and pct < 0) else 0
                else:
                    correct = None
            else:
                pct = None
                correct = None

            try:
                with db._connect() as conn:
                    conn.execute(
                        f"UPDATE signal_events "
                        f"SET {price_col} = ?, {pct_col} = ?, outcome_correct = ? "
                        f"WHERE id = ?",
                        (price, pct, correct, row["id"]),
                    )
                count += 1
            except Exception as exc:
                log.warning("Failed to update signal %d: %s", row["id"], exc)

        updated[price_col] = count
        if count:
            log.info("Backfilled %d rows for %s", count, price_col)

    return updated


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Backfill outcome data for signal events",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Only backfill a specific horizon (3, 5, or 10).",
    )
    args = parser.parse_args()

    print("Signal outcome tracker — backfilling price outcomes...")
    result = run_outcome_tracker()
    total = sum(result.values())
    print(f"Done. Updated {total} rows: {result}")


if __name__ == "__main__":
    main()
