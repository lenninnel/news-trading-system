#!/usr/bin/env python3
"""
Replay the cluster agreement gate on an existing signal_events table.

Read-only (``mode=ro`` URI) — never writes.  Groups every Combined row
with the Momentum / Pullback / NewsCatalyst rows of the same run (same
ticker + session, logged within 20 minutes before it), re-applies the
detector's own rules (MIN_CONFIDENCE 0.35, direction buckets) and counts
how many strategies agreed.  Then attributes every executed BUY in
trade_history to its originating Combined row (same ticker, latest
directional row within 26 h before the fill, 5 min tolerance because the
Combined row is logged just after the fill) and reports how many of
those would have passed the gate.

Usage:
    python3 scripts/cluster_gate_backtest.py                 # live VPS path
    python3 scripts/cluster_gate_backtest.py path/to/news_trading.db
    python3 scripts/cluster_gate_backtest.py --min 3         # threshold sweep
    python3 scripts/cluster_gate_backtest.py --fold Momentum NewsCatalyst

docs/CLUSTER_AGREEMENT_GATE_2026-09-07.md §1 / §2 / §4 were produced
with this script (defaults, and ``--fold Momentum NewsCatalyst``).
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import sqlite3

DEFAULT_DB = "/home/trading/trading-data/news_trading.db"
MIN_CONF = 0.35
BUY = {"STRONG BUY", "BUY", "WEAK BUY"}
SELL = {"SELL", "STRONG SELL", "WEAK SELL"}
STRATS = ("Momentum", "Pullback", "NewsCatalyst")
RUN_WINDOW = dt.timedelta(minutes=20)
FILL_LOOKBACK = dt.timedelta(hours=26)
FILL_TOLERANCE = dt.timedelta(minutes=-5)


def _ts(s: str) -> dt.datetime:
    d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def _week(t: dt.datetime) -> str:
    y, w, _ = t.isocalendar()
    return f"{y}-W{w:02d}"


def load_runs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, timestamp, session, ticker, strategy, signal, confidence, "
        "trade_executed, trade_id, signal_path FROM signal_events "
        "ORDER BY ticker, timestamp, id"
    ).fetchall()
    runs: list[dict] = []
    buf: list[sqlite3.Row] = []
    cur = None
    for r in rows:
        if r["ticker"] != cur:
            buf, cur = [], r["ticker"]
        if r["strategy"] in STRATS:
            buf.append(r)
            continue
        if r["strategy"] == "Combined":
            t = _ts(r["timestamp"])
            votes = {
                v["strategy"]: v for v in buf
                if t - _ts(v["timestamp"]) <= RUN_WINDOW
                and v["session"] == r["session"]
            }
            buf = []
            runs.append({"row": r, "votes": votes, "t": t})
    return runs


def agreeing(votes: dict) -> list[str]:
    buy = [n for n, v in votes.items()
           if v["signal"].upper() in BUY and (v["confidence"] or 0) >= MIN_CONF]
    sell = [n for n, v in votes.items()
            if v["signal"].upper() in SELL and (v["confidence"] or 0) >= MIN_CONF]
    return buy or sell


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("db", nargs="?", default=DEFAULT_DB)
    ap.add_argument("--min", type=int, default=2,
                    help="minimum distinct sources (default 2)")
    ap.add_argument("--fold", nargs="+", default=[], metavar="STRATEGY",
                    help="strategies to count as ONE vote source")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    fold = set(args.fold)

    def n_sources(names: list[str]) -> int:
        return len({"<folded>" if n in fold else n for n in names})

    runs = load_runs(conn)
    cluster_dir = []
    for x in runs:
        if x["row"]["signal"].upper() in BUY | SELL and x["votes"] \
                and (x["row"]["signal_path"] or "").startswith("CLUSTER"):
            x["names"] = agreeing(x["votes"])
            x["n"] = len(x["names"])
            x["src"] = n_sources(x["names"])
            cluster_dir.append(x)

    print(f"signal_events Combined rows: {len(runs)}  "
          f"directional on CLUSTER path: {len(cluster_dir)}  "
          f"gate: >= {args.min} sources, folded={sorted(fold) or '-'}")
    print("agreeing votes:", dict(sorted(collections.Counter(x["n"] for x in cluster_dir).items())))
    passed = [x for x in cluster_dir if x["src"] >= args.min]
    print(f"would pass: {len(passed)} / {len(cluster_dir)} "
          f"({100 * len(passed) / max(1, len(cluster_dir)):.1f} %)")
    print("composition of passing sets:",
          collections.Counter(tuple(sorted(x["names"])) for x in passed).most_common())

    by_ticker: dict[str, list[dict]] = collections.defaultdict(list)
    for x in cluster_dir:
        by_ticker[x["row"]["ticker"]].append(x)

    buys = conn.execute(
        "SELECT id, ticker, created_at FROM trade_history "
        "WHERE upper(action)='BUY' ORDER BY created_at"
    ).fetchall()
    weekly: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    dist: collections.Counter = collections.Counter()
    unmatched = 0
    for b in buys:
        tb = _ts(b["created_at"])
        cands = [x for x in by_ticker[b["ticker"]]
                 if FILL_TOLERANCE <= tb - x["t"] <= FILL_LOOKBACK]
        w = _week(tb)
        weekly[w][0] += 1
        if not cands:
            unmatched += 1
            continue
        x = max(cands, key=lambda y: y["t"])
        dist[x["n"]] += 1
        if x["src"] >= args.min:
            weekly[w][1] += 1

    print(f"\nexecuted BUYs: {len(buys)}  unmatched: {unmatched}  "
          f"by agreeing votes: {dict(sorted(dist.items()))}")
    print("week        all  pass")
    for w in sorted(weekly):
        print(f"  {w}  {weekly[w][0]:4d}  {weekly[w][1]:4d}")
    tot_all = sum(v[0] for v in weekly.values())
    tot_pass = sum(v[1] for v in weekly.values())
    n_w = max(1, len(weekly))
    print(f"totals      {tot_all:4d}  {tot_pass:4d}   "
          f"per week {tot_all / n_w:.1f} -> {tot_pass / n_w:.1f}")


if __name__ == "__main__":
    main()
