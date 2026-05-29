#!/usr/bin/env python3
"""One-shot, idempotent maintenance: flag the META ticker-recycle window.

WHY THIS EXISTS — IDENTITY BREAK, NOT A CORPORATE ACTION
--------------------------------------------------------
In `daily_ohlc`, the "META" series carries two DIFFERENT issuers:

  * date <  2022-06-09 : a prior, unrelated holder of the "META" symbol
                         (~$12-15 prints, ~0.2-2M volume), then a ~4.5-month
                         gap (last bar 2022-01-28 -> next bar 2022-06-09).
  * date >= 2022-06-09 : Meta Platforms (FB -> META renaming day; opens ~$184,
                         ~23-31M volume). 1,143 valid Meta Platforms bars.

The day-over-day adj_close ratio across the boundary is ~14.95 — adj_close
CANNOT bridge it because this is a symbol reassignment, NOT a split/dividend.
These pre-2022-06-09 rows must therefore NOT be treated as Meta Platforms
history, and must NOT be "cleaned up" as a stale EXTREME_MOVE later.

This script sets quality_flag='TICKER_RECYCLE' on those pre-rename rows.
Flag-don't-drop: nothing is deleted. Idempotent: it only promotes currently
UNFLAGGED rows in the window, so a re-run affects 0 rows and never alters any
existing flag (e.g. the EXTREME_MOVE on the 2022-06-09 boundary, which is
>= the cutoff and out of scope here). Only `daily_ohlc` is written.

Usage:
    python3 scripts/flag_meta_recycle.py            # apply
    python3 scripts/flag_meta_recycle.py --dry-run  # report only, no write
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

TICKER = "META"
CUTOFF = "2022-06-09"  # first Meta Platforms bar; flag everything strictly before
FLAG = "TICKER_RECYCLE"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the affected count without writing.")
    parser.add_argument("--db", default=os.environ.get("DB_PATH", "news_trading.db"),
                        help="SQLite path (defaults to $DB_PATH).")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA busy_timeout=30000")  # wait out the live daemon's writes

    # Only currently-unflagged pre-rename rows: idempotent + never clobbers
    # an existing flag.
    where = (
        "ticker = ? AND date < ? AND quality_flag IS NULL"
    )
    to_flag = conn.execute(
        f"SELECT COUNT(*) FROM daily_ohlc WHERE {where}", (TICKER, CUTOFF)
    ).fetchone()[0]
    already = conn.execute(
        "SELECT COUNT(*) FROM daily_ohlc WHERE ticker=? AND date<? AND quality_flag=?",
        (TICKER, CUTOFF, FLAG),
    ).fetchone()[0]
    print(f"pre-{CUTOFF} {TICKER} rows: to-flag(NULL)={to_flag} "
          f"already-{FLAG}={already}")

    if args.dry_run:
        print("dry-run: no write performed")
        return 0

    with conn:
        cur = conn.execute(
            f"UPDATE daily_ohlc SET quality_flag=? WHERE {where}",
            (FLAG, TICKER, CUTOFF),
        )
    print(f"rows affected: {cur.rowcount}")

    # Post-state: full flag distribution + the META window split.
    dist = conn.execute(
        "SELECT quality_flag, COUNT(*) FROM daily_ohlc GROUP BY quality_flag"
    ).fetchall()
    print("post-state flag distribution:", dist)
    print(f"{TICKER} < {CUTOFF} flagged {FLAG}:",
          conn.execute(
              "SELECT COUNT(*) FROM daily_ohlc WHERE ticker=? AND date<? AND quality_flag=?",
              (TICKER, CUTOFF, FLAG)).fetchone()[0])
    print(f"{TICKER} >= {CUTOFF} touched (must be 0 non-original):",
          conn.execute(
              "SELECT COUNT(*) FROM daily_ohlc WHERE ticker=? AND date>=? AND quality_flag=?",
              (TICKER, CUTOFF, FLAG)).fetchone()[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
