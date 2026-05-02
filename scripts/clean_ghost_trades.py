#!/usr/bin/env python3
"""
One-time (or startup) cleanup of ghost positions and trades.

Ghost data are rows with unrealistically low prices that were inserted by
a bug (cached/stale price data).  This script deletes them and prints a
clear summary.

Usage::

    python scripts/clean_ghost_trades.py          # dry-run (default)
    python scripts/clean_ghost_trades.py --apply   # actually delete

It is also called from entrypoint.sh on every Railway container restart
(with --apply) so ghost data cannot accumulate.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

# Threshold: any price below this is considered a ghost/test fixture price.
# AAPL trades at ~$240+, even the cheapest watchlist stock is above $25.
# Using $200 catches the $150 fixture prices while leaving real trades alone.
GHOST_PRICE_THRESHOLD = 200.0

# ISO 4217 cash currency codes that can appear in portfolio_positions /
# trade_history as cash holdings (e.g. after a FX conversion that moves
# EUR -> USD, the EUR balance shows up with price=0.00). These rows
# must never be touched by ghost cleanup. "BASE" is IBKR's aggregate
# row for the account base currency.
# Why an explicit allowlist instead of a length heuristic: real equity
# tickers also fit a 3-letter pattern (JPM, BAC, WMT, XOM), so we have
# to enumerate currencies — we can't infer "is this cash?" from shape.
_CASH_CURRENCY_CODES = frozenset({
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
    "HKD", "SGD", "CNH", "CNY", "SEK", "NOK", "DKK", "MXN",
    "ZAR", "KRW", "INR", "BRL", "ILS", "PLN", "HUF", "CZK",
    "TRY", "AED", "SAR", "THB", "IDR", "MYR", "PHP", "VND",
    "RUB", "BASE",
})


def _resolve_db_path() -> str:
    """Find the database — Railway /data volume first, then local default."""
    railway_dir = "/data"
    if os.path.isdir(railway_dir) and os.access(railway_dir, os.W_OK):
        return os.path.join(railway_dir, "news_trading.db")
    from config.settings import DB_PATH
    return DB_PATH


def clean_ghost_data(db_path: str, apply: bool = False) -> dict:
    """
    Find and optionally delete ghost positions and trades.

    Returns a summary dict with counts and details.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Build "AND UPPER(ticker) NOT IN (?, ?, ...)" so cash holdings
    # (EUR / USD / etc.) are skipped by both the SELECT and the DELETE.
    placeholders = ",".join("?" * len(_CASH_CURRENCY_CODES))
    cash_excl_sql = f" AND UPPER(ticker) NOT IN ({placeholders})"
    cash_excl_params: tuple = tuple(_CASH_CURRENCY_CODES)

    # --- Ghost positions (avg_price < threshold, excluding cash) ---
    ghost_positions = conn.execute(
        f"SELECT * FROM portfolio_positions WHERE avg_price < ?{cash_excl_sql}",
        (GHOST_PRICE_THRESHOLD, *cash_excl_params),
    ).fetchall()

    # --- Ghost trades (price < threshold, excluding cash) ---
    ghost_trades = conn.execute(
        f"SELECT * FROM trade_history WHERE price < ?{cash_excl_sql}",
        (GHOST_PRICE_THRESHOLD, *cash_excl_params),
    ).fetchall()

    deleted_positions = 0
    deleted_trades = 0

    if apply and (ghost_positions or ghost_trades):
        if ghost_positions:
            cursor = conn.execute(
                f"DELETE FROM portfolio_positions WHERE avg_price < ?{cash_excl_sql}",
                (GHOST_PRICE_THRESHOLD, *cash_excl_params),
            )
            deleted_positions = cursor.rowcount

        if ghost_trades:
            cursor = conn.execute(
                f"DELETE FROM trade_history WHERE price < ?{cash_excl_sql}",
                (GHOST_PRICE_THRESHOLD, *cash_excl_params),
            )
            deleted_trades = cursor.rowcount

        conn.commit()

    conn.close()

    return {
        "db_path": db_path,
        "ghost_positions": [dict(r) for r in ghost_positions],
        "ghost_trades": [dict(r) for r in ghost_trades],
        "deleted_positions": deleted_positions,
        "deleted_trades": deleted_trades,
        "applied": apply,
    }


def print_summary(result: dict) -> None:
    """Print a human-readable cleanup summary."""
    print(f"\n{'=' * 60}")
    print(f"  Ghost Trade Cleanup")
    print(f"  DB: {result['db_path']}")
    print(f"{'=' * 60}")

    gp = result["ghost_positions"]
    gt = result["ghost_trades"]

    if not gp and not gt:
        print("\n  No ghost data found. Database is clean.\n")
        return

    if gp:
        print(f"\n  GHOST POSITIONS ({len(gp)}):")
        for p in gp:
            print(f"    {p.get('ticker', '?')}: {p.get('shares', '?')} shares "
                  f"@ ${p.get('avg_price', 0):.2f}")

    if gt:
        print(f"\n  GHOST TRADES ({len(gt)}):")
        for t in gt[:20]:  # limit display
            print(f"    #{t.get('id', '?')} {t.get('action', '?')} "
                  f"{t.get('shares', '?')} {t.get('ticker', '?')} "
                  f"@ ${t.get('price', 0):.2f} "
                  f"({t.get('created_at', '?')})")
        if len(gt) > 20:
            print(f"    ... and {len(gt) - 20} more")

    if result["applied"]:
        print(f"\n  DELETED: {result['deleted_positions']} positions, "
              f"{result['deleted_trades']} trades")
    else:
        print(f"\n  DRY RUN — would delete {len(gp)} positions, {len(gt)} trades")
        print(f"  Re-run with --apply to delete.")

    print(f"\n{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean ghost positions and trades from the database",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually delete ghost data (default is dry-run)",
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Path to SQLite database (auto-detected if omitted)",
    )
    args = parser.parse_args()

    db_path = args.db or _resolve_db_path()
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path} — nothing to clean.")
        return

    result = clean_ghost_data(db_path, apply=args.apply)
    print_summary(result)


if __name__ == "__main__":
    main()
