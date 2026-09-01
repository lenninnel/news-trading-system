"""
Post-deploy verification for the Wilder-ATR change (2026-09-01).

Read-only. Compares the stop-distance distribution and share sizes of
risk calculations before vs. after the deploy cut-off.

Expected effect (KW35 stop-origin report): the close-to-close proxy ran
at ~0.556 of true ATR, so stop distances should roughly double, share
sizes roughly halve, and the 10% portfolio cap should bind far less
often.

Usage (on the VPS, against the production DB):

    python3 scripts/verify_atr_deploy.py \
        --db /home/trading/trading-data/news_trading.db \
        --deploy 2026-09-01T18:00:00

Run after the first 2-3 post-deploy sessions; the "before" window is
the 7 days preceding the cut-off.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timedelta


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(int(round(q * (len(s) - 1))), len(s) - 1)
    return s[idx]


def _summarise(label: str, rows: list[sqlite3.Row]) -> None:
    sized = [r for r in rows if not r["skipped"] and r["stop_pct"]]
    stop_pcts = [r["stop_pct"] * 100 for r in sized]
    shares = [r["shares"] for r in sized]
    # 10% portfolio cap binds when position_size_usd is pinned at the cap
    capped = [
        r for r in sized
        if r["account_balance"]
        and r["position_size_usd"] >= 0.995 * 0.10 * r["account_balance"]
    ]
    print(f"\n── {label}: {len(sized)} sized calcs "
          f"({len(rows)} total, {len(rows) - len(sized)} skipped/unsized)")
    if not sized:
        return
    print(f"   stop distance %  min {min(stop_pcts):.2f}  "
          f"p25 {_pct(stop_pcts, .25):.2f}  med {_pct(stop_pcts, .50):.2f}  "
          f"p75 {_pct(stop_pcts, .75):.2f}  max {max(stop_pcts):.2f}")
    print(f"   shares           min {min(shares)}  "
          f"med {int(_pct(shares, .50))}  max {max(shares)}")
    print(f"   10% cap binding  {len(capped)}/{len(sized)} "
          f"({len(capped) / len(sized) * 100:.0f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="news_trading.db")
    ap.add_argument("--deploy", required=True,
                    help="deploy cut-off, ISO timestamp (UTC)")
    ap.add_argument("--before-days", type=int, default=7)
    args = ap.parse_args()

    cutoff = datetime.fromisoformat(args.deploy)
    before_start = cutoff - timedelta(days=args.before_days)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    q = ("SELECT ticker, stop_pct, shares, position_size_usd, account_balance, "
         "skipped, created_at FROM risk_calculations "
         "WHERE created_at >= ? AND created_at < ? ORDER BY created_at")

    before = conn.execute(q, (before_start.isoformat(), cutoff.isoformat())).fetchall()
    after = conn.execute(q, (cutoff.isoformat(), "9999")).fetchall()

    print(f"Deploy cut-off: {cutoff.isoformat()}  "
          f"(before window: {args.before_days}d)")
    _summarise(f"BEFORE (proxy ATR, {before_start.date()}..{cutoff.date()})", before)
    _summarise(f"AFTER  (Wilder ATR, {cutoff.date()}..)", after)

    print("\nExpectation: median stop distance ~2x the before window, "
          "median shares ~0.5x, cap-binding share clearly down.")


if __name__ == "__main__":
    main()
