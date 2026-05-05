#!/usr/bin/env python3
"""
One-time recovery for the 2026-05-05 ghost-cleanup incident.

The startup ghost-cleanup (with the now-removed $200 threshold) deleted:
  - 6 portfolio_positions rows (broken avg_price values from ibkr_trader.py:373)
  - 2 trade_history rows: TOL @ $138.33, XOM @ $153.28

portfolio_positions self-heals from IBKR on the next get_portfolio() call.
trade_history rows are gone forever unless rehydrated — PositionManager
needs them to look up stop_loss / take_profit during stop-loss enforcement.

This script:
  1. Reinserts the TOL and XOM BUY rows with ATR-based SL/TP computed
     against current ATR (best approximation of fill-time values).
  2. Backfills position_metadata for AAPL, MSFT, XOM (the cached US_OPEN
     execution path bypassed register_position pre-fix).

Idempotent: skips rows that already exist. Prints inserted row IDs so
the actions are reversible by ID if the values turn out wrong.

Usage::

    python3 scripts/recover_ghost_cleanup_2026_05_05.py --apply
    python3 scripts/recover_ghost_cleanup_2026_05_05.py            # dry run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.risk_agent import RiskAgent
from execution.portfolio_manager import _fetch_sector
from strategies.router import strategy_label

# ── Trade rehydration data ────────────────────────────────────────────────
# Source: IBKR position log + user instruction (2026-05-05).
# avgCost values from 11:35:51 UTC startup sync are the broker truth;
# user-provided fill prices match within $0.01.
_TRADES_TO_RESTORE = [
    {
        "ticker":     "TOL",
        "shares":     213,
        "price":      138.33,
        "created_at": "2026-05-04T13:48:00+00:00",  # original BUY timestamp (~13:48 UTC)
    },
    {
        "ticker":     "XOM",
        "shares":     192,
        "price":      153.28,
        "created_at": "2026-05-04T14:30:00+00:00",  # original BUY timestamp (~14:30 UTC)
    },
]

# ── position_metadata backfill ────────────────────────────────────────────
# Strategy via router.strategy_label() (PULLBACK_TICKERS for all three).
# Sector via execution.portfolio_manager._fetch_sector (yfinance).
# entry_price from trade_history.price column.
_METADATA_TO_BACKFILL = ["AAPL", "MSFT", "XOM"]


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def restore_trade(
    conn: sqlite3.Connection,
    risk: RiskAgent,
    ticker: str,
    shares: int,
    price: float,
    created_at: str,
    apply: bool,
) -> dict:
    """Reinsert a trade_history row with ATR-based SL/TP. Idempotent."""
    existing = conn.execute(
        "SELECT id, stop_loss, take_profit FROM trade_history "
        "WHERE ticker = ? AND action = 'BUY' "
        "AND date(created_at) = date(?)",
        (ticker, created_at),
    ).fetchone()
    if existing:
        return {
            "ticker":   ticker,
            "skipped":  True,
            "reason":   f"trade_history row already exists (id={existing['id']})",
            "row_id":   existing["id"],
        }

    atr = risk.calculate_atr_stops(
        ticker=ticker, entry_price=price, direction="BUY",
    )
    if not atr.get("atr_available"):
        return {
            "ticker":   ticker,
            "skipped":  True,
            "reason":   "ATR unavailable — refusing to insert without SL/TP",
        }

    sl = atr["stop_loss"]
    tp = atr["take_profit"]

    if not apply:
        return {
            "ticker":   ticker,
            "would_insert": True,
            "shares":   shares,
            "price":    price,
            "stop_loss":   sl,
            "take_profit": tp,
            "atr":      atr["atr"],
        }

    cur = conn.execute(
        """
        INSERT INTO trade_history
            (ticker, action, shares, price, stop_loss, take_profit, pnl, created_at)
        VALUES (?, 'BUY', ?, ?, ?, ?, 0.0, ?)
        """,
        (ticker, shares, price, sl, tp, created_at),
    )
    conn.commit()
    return {
        "ticker":   ticker,
        "inserted": True,
        "row_id":   cur.lastrowid,
        "shares":   shares,
        "price":    price,
        "stop_loss":   sl,
        "take_profit": tp,
        "atr":      atr["atr"],
    }


def backfill_metadata(
    conn: sqlite3.Connection,
    ticker: str,
    apply: bool,
) -> dict:
    """Backfill a position_metadata row using router strategy + yf sector."""
    existing = conn.execute(
        "SELECT ticker FROM position_metadata WHERE ticker = ?",
        (ticker,),
    ).fetchone()
    if existing:
        return {
            "ticker":  ticker,
            "skipped": True,
            "reason":  "position_metadata row already exists",
        }

    fill = conn.execute(
        "SELECT price, created_at FROM trade_history "
        "WHERE ticker = ? AND action = 'BUY' "
        "ORDER BY created_at DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    if not fill:
        return {
            "ticker":  ticker,
            "skipped": True,
            "reason":  "no BUY in trade_history — cannot infer entry_price",
        }

    strategy = strategy_label(ticker)        # "Pullback" / "Momentum"
    sector = _fetch_sector(ticker)           # "Tech" / "Energy" / etc
    entry_price = fill["price"]
    entry_date = fill["created_at"] or datetime.now(timezone.utc).isoformat()

    if not apply:
        return {
            "ticker":   ticker,
            "would_insert": True,
            "strategy": strategy,
            "sector":   sector,
            "entry_price": entry_price,
            "entry_date":  entry_date,
        }

    conn.execute(
        """
        INSERT INTO position_metadata
            (ticker, strategy, sector, entry_date, entry_price)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ticker, strategy, sector, entry_date, entry_price),
    )
    conn.commit()
    return {
        "ticker":   ticker,
        "inserted": True,
        "strategy": strategy,
        "sector":   sector,
        "entry_price": entry_price,
        "entry_date":  entry_date,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Path to news_trading.db")
    parser.add_argument("--apply", action="store_true",
                        help="Actually insert rows (default: dry-run)")
    args = parser.parse_args()

    conn = _connect(args.db)
    risk = RiskAgent()

    print(f"\n{'═' * 60}")
    print(f"  Recovery — {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"  DB: {args.db}")
    print(f"{'═' * 60}\n")

    print("─── trade_history reinsert ────────────────────────────────")
    for t in _TRADES_TO_RESTORE:
        r = restore_trade(conn, risk, apply=args.apply, **t)
        print(f"  {r['ticker']}: {r}")

    print("\n─── position_metadata backfill ────────────────────────────")
    for ticker in _METADATA_TO_BACKFILL:
        r = backfill_metadata(conn, ticker, apply=args.apply)
        print(f"  {r['ticker']}: {r}")

    print(f"\n{'═' * 60}\n")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
