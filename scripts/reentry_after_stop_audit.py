#!/usr/bin/env python3
"""Re-entry-after-stop audit on the production trade book (read-only).

Question: how often did the live system buy a ticker back after it had just
stopped that ticker out, how long after, what did those re-entries cost, and
how much of the shadow book's selection effect (B − A, −1.02 % per trade in
docs/SHADOW_STRATEGY_BOOK_2026-09-07.md) do they carry?

Method (all from the DB, nothing inferred beyond what is stated here):

* Round trips: FIFO pairing of trade_history BUY/SELL rows per ticker
  (orphan SELLs without an open BUY are dropped — the partial-fill finding of
  2026-08-25).
* Exit class per round trip, from the PositionManager row in signal_events
  logged at the SELL (same ticker, within the match window):
      ENTRY_STOP  "stop_loss_triggered at $L" with L <= opening stop_loss * (1 + tol)
      TRAIL_STOP  "stop_loss_triggered at $L" with L above that (a trailed stop)
      TP          "take_profit_triggered"
      OTHER       no PositionManager row (manual / reconciliation / signal exit)
* Re-entry: a BUY whose ticker's previous round trip closed before it. The
  gap is measured in US trading sessions (data.market_calendar, holiday-aware;
  0 = same session) and in calendar hours.
* Cost: realised P&L of the SELL that closed the re-entry (trade_history.pnl)
  and fill-to-fill return. Open re-entries are counted but carry no P&L.
* Selection effect: per-round-trip shadow returns from
  scripts/shadow_strategy_book.py --csv (real_round_trips.csv), joined on
  (ticker, buy_at). Line A (all Combined signals) is taken from that report.

Usage:
    python3 scripts/reentry_after_stop_audit.py --db snapshot.db \
        --shadow-csv /tmp/shadow/real_round_trips.csv --line-a 0.0011
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sqlite3
import statistics
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.market_calendar import us_sessions_between  # noqa: E402

DEFAULT_DB = "/home/trading/trading-data/news_trading.db"
STOP_TOL = 0.003                 # same tolerance the Q-016 gate uses
PM_MATCH_BEFORE = timedelta(minutes=15)
PM_MATCH_AFTER = timedelta(minutes=2)
GATE_DEPLOYED = date(2026, 7, 1)  # Q-016 1-session cool-down live since
_LEVEL_RE = re.compile(r"(stop_loss|take_profit)_triggered at \$?([\d.]+)")


def _ts(s: str) -> datetime:
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def pct(x: float) -> str:
    return f"{100 * x:+.2f} %"


def load(db: str):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    trades = [dict(r) for r in conn.execute(
        "SELECT id, ticker, action, shares, price, executed_price, stop_loss, "
        "take_profit, pnl, created_at, strategy FROM trade_history "
        "ORDER BY created_at, id")]
    pm = [dict(r) for r in conn.execute(
        "SELECT timestamp, ticker, bull_case FROM signal_events "
        "WHERE strategy = 'PositionManager' AND signal = 'SELL'")]
    conn.close()
    return trades, pm


def build_round_trips(trades: list[dict]):
    open_buys: dict[str, list[dict]] = collections.defaultdict(list)
    rts: list[dict] = []
    orphans = 0
    for t in trades:
        tk = t["ticker"]
        if t["action"] == "BUY":
            open_buys[tk].append(t)
        elif t["action"] == "SELL":
            if open_buys[tk]:
                b = open_buys[tk].pop(0)
                px_b = b["price"]
                px_s = t["executed_price"] if t["executed_price"] else t["price"]
                rts.append({"ticker": tk, "buy": b, "sell": t,
                            "ret": (px_s - px_b) / px_b, "pnl": t["pnl"],
                            "buy_at": _ts(b["created_at"]), "sell_at": _ts(t["created_at"])})
            else:
                orphans += 1
    still_open = [{"ticker": tk, "buy": b, "sell": None, "buy_at": _ts(b["created_at"])}
                  for tk, bs in open_buys.items() for b in bs]
    return rts, still_open, orphans


def classify_exits(rts: list[dict], pm_rows: list[dict]) -> None:
    by_tk: dict[str, list[tuple[datetime, str]]] = collections.defaultdict(list)
    for r in pm_rows:
        by_tk[r["ticker"]].append((_ts(r["timestamp"]), r["bull_case"] or ""))
    for rt in rts:
        rt["exit"] = "OTHER"
        rt["pm_level"] = None
        cands = [(abs(ts - rt["sell_at"]), ts, bc) for ts, bc in by_tk[rt["ticker"]]
                 if rt["sell_at"] - PM_MATCH_BEFORE <= ts <= rt["sell_at"] + PM_MATCH_AFTER]
        if not cands:
            continue
        _, _, bc = min(cands)
        m = _LEVEL_RE.search(bc)
        if not m:
            continue
        kind, level = m.group(1), float(m.group(2))
        rt["pm_level"] = level
        if kind == "take_profit":
            rt["exit"] = "TP"
        else:
            stop = rt["buy"]["stop_loss"]
            if stop and level <= float(stop) * (1 + STOP_TOL):
                rt["exit"] = "ENTRY_STOP"
            else:
                rt["exit"] = "TRAIL_STOP"
        # what the live Q-016 classifier would say about this exit
        px = rt["sell"]["executed_price"] or rt["sell"]["price"]
        stop = rt["buy"]["stop_loss"]
        rt["q016_stop"] = bool(stop) and float(px) <= float(stop) * (1 + STOP_TOL)


def mark_reentries(rts: list[dict], still_open: list[dict]) -> list[dict]:
    """Attach the previous round trip (same ticker) to every BUY."""
    entries = sorted(rts + still_open, key=lambda e: e["buy_at"])
    closed_by_tk: dict[str, list[dict]] = collections.defaultdict(list)
    for rt in rts:
        closed_by_tk[rt["ticker"]].append(rt)
    for e in entries:
        prev = [p for p in closed_by_tk[e["ticker"]] if p["sell_at"] < e["buy_at"]]
        e["prev"] = max(prev, key=lambda p: p["sell_at"]) if prev else None
        if e["prev"]:
            d_sell = e["prev"]["sell_at"].astimezone(timezone.utc).date()
            d_buy = e["buy_at"].astimezone(timezone.utc).date()
            e["gap_sessions"] = us_sessions_between(d_sell, d_buy)
            e["gap_hours"] = (e["buy_at"] - e["prev"]["sell_at"]).total_seconds() / 3600
    return entries


def load_shadow(path: str) -> dict[tuple[str, str], float]:
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["shadow_ret"]:
                out[(row["ticker"], row["buy_at"])] = float(row["shadow_ret"])
    return out


def summarise(v: list[float]) -> str:
    if not v:
        return "n/a"
    wins = sum(1 for x in v if x > 0)
    return f"N={len(v)}, mean {pct(statistics.fmean(v))}, median {pct(statistics.median(v))}, win {100 * wins / len(v):.0f} %"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--shadow-csv", help="real_round_trips.csv from shadow_strategy_book.py --csv")
    ap.add_argument("--line-a", type=float, default=None,
                    help="shadow line A expectancy as a fraction (e.g. 0.0011 for +0.11 %%)")
    args = ap.parse_args()

    trades, pm_rows = load(args.db)
    rts, still_open, orphans = build_round_trips(trades)
    classify_exits(rts, pm_rows)
    entries = mark_reentries(rts, still_open)
    shadow = load_shadow(args.shadow_csv) if args.shadow_csv else {}

    n_buy = sum(1 for t in trades if t["action"] == "BUY")
    n_sell = sum(1 for t in trades if t["action"] == "SELL")
    print(f"# Re-entry after stop — audit of the production trade book\n")
    print(f"DB: `{args.db}`; trade_history {n_buy} BUY / {n_sell} SELL "
          f"({trades[0]['created_at'][:10]} … {trades[-1]['created_at'][:10]}); "
          f"{len(rts)} FIFO round trips, {orphans} orphan SELLs dropped, {len(still_open)} BUYs open.\n")

    # ── 1. exit classes ──────────────────────────────────────────────────
    print("## 1. Exit classes of all closed round trips\n")
    print("| exit | N | realised P&L | fill-to-fill return | Q-016 classifier says STOP |")
    print("|---|---|---|---|---|")
    for k in ("ENTRY_STOP", "TRAIL_STOP", "TP", "OTHER"):
        sub = [rt for rt in rts if rt["exit"] == k]
        if not sub:
            continue
        q = sum(1 for rt in sub if rt.get("q016_stop"))
        print(f"| {k} | {len(sub)} | {sum(rt['pnl'] for rt in sub):+,.0f} USD | "
              f"{summarise([rt['ret'] for rt in sub])} | {q}/{len(sub)} |")
    print()

    # ── 2. re-entries ────────────────────────────────────────────────────
    with_prev = [e for e in entries if e["prev"]]
    print("## 2. Entries by what the ticker's previous round trip ended in\n")
    print(f"{len(entries)} BUYs; {len(entries) - len(with_prev)} first entries of a ticker, "
          f"{len(with_prev)} re-entries into a ticker traded before.\n")
    print("| previous exit | N re-entries | closed | realised P&L of the re-entry | fill-to-fill return | "
          "gap sessions (min / median / p90 / max) | gap hours median |")
    print("|---|---|---|---|---|---|---|")
    for k in ("ENTRY_STOP", "TRAIL_STOP", "TP", "OTHER"):
        sub = [e for e in with_prev if e["prev"]["exit"] == k]
        if not sub:
            continue
        closed = [e for e in sub if e["sell"]]
        gaps = sorted(e["gap_sessions"] for e in sub)
        p90 = gaps[min(len(gaps) - 1, int(round(0.9 * (len(gaps) - 1))))]
        print(f"| {k} | {len(sub)} | {len(closed)} | {sum(e['pnl'] for e in closed):+,.0f} USD | "
              f"{summarise([e['ret'] for e in closed])} | {gaps[0]} / {statistics.median(gaps):g} / {p90} / {gaps[-1]} | "
              f"{statistics.median(e['gap_hours'] for e in sub):.0f} h |")
    print()

    # ── 3. gap distribution after an ENTRY_STOP ─────────────────────────
    after_stop = [e for e in with_prev if e["prev"]["exit"] == "ENTRY_STOP"]
    print("## 3. Gap distribution of re-entries after an ENTRY_STOP\n")
    print("| gap (sessions) | N | of which closed | realised P&L | return | pre-gate (< 2026-07-01) | post-gate |")
    print("|---|---|---|---|---|---|---|")
    max_gap = max((e["gap_sessions"] for e in after_stop), default=0)
    for g in range(0, max_gap + 1):
        sub = [e for e in after_stop if e["gap_sessions"] == g]
        if not sub:
            continue
        closed = [e for e in sub if e["sell"]]
        pre = sum(1 for e in sub if e["buy_at"].date() < GATE_DEPLOYED)
        print(f"| {g} | {len(sub)} | {len(closed)} | {sum(e['pnl'] for e in closed):+,.0f} USD | "
              f"{summarise([e['ret'] for e in closed])} | {pre} | {len(sub) - pre} |")
    print()
    # cumulative: what a lock of N sessions would have caught
    print("### 3b. Cumulative: a lock of N sessions after an ENTRY_STOP would have blocked\n")
    print("| lock N (sessions) | blocked re-entries | share of all after-stop re-entries | "
          "realised P&L of blocked (closed ones) | mean return of blocked | mean return of the rest (after-stop, not blocked) | "
          "blocked in the 1-session-gate era (>= 2026-07-01) | their realised P&L |")
    print("|---|---|---|---|---|---|---|---|")
    shown = 0
    for n in range(0, max_gap + 1):
        blocked = [e for e in after_stop if e["gap_sessions"] <= n]
        rest = [e for e in after_stop if e["gap_sessions"] > n and e["sell"]]
        bc = [e for e in blocked if e["sell"]]
        post = [e for e in blocked if e["buy_at"].date() >= GATE_DEPLOYED]
        post_c = [e for e in post if e["sell"]]
        if not blocked or (n > 12 and len(blocked) == shown):
            continue
        shown = len(blocked)
        print(f"| {n} | {len(blocked)} | {100 * len(blocked) / len(after_stop):.0f} % | "
              f"{sum(e['pnl'] for e in bc):+,.0f} USD | "
              f"{pct(statistics.fmean(e['ret'] for e in bc)) if bc else 'n/a'} | "
              f"{pct(statistics.fmean(e['ret'] for e in rest)) if rest else 'n/a'} | "
              f"{len(post)} | {sum(e['pnl'] for e in post_c):+,.0f} USD |")
    print()

    # ── 4. by ticker ─────────────────────────────────────────────────────
    print("## 4. After-stop re-entries by ticker\n")
    print("| ticker | BUYs total | re-entries after ENTRY_STOP | of which gap <= 1 | realised P&L of after-stop re-entries |")
    print("|---|---|---|---|---|")
    by_tk = collections.Counter(e["ticker"] for e in entries)
    rows = []
    for tk in by_tk:
        sub = [e for e in after_stop if e["ticker"] == tk]
        if not sub:
            continue
        rows.append((tk, by_tk[tk], len(sub), sum(1 for e in sub if e["gap_sessions"] <= 1),
                     sum(e["pnl"] for e in sub if e["sell"])))
    for tk, nb, nr, n1, p in sorted(rows, key=lambda r: -r[2]):
        print(f"| {tk} | {nb} | {nr} | {n1} | {p:+,.0f} USD |")
    print()

    # ── 5. selection effect attribution ─────────────────────────────────
    if shadow:
        print("## 5. Shadow-model expectancy (line B) split by re-entry status\n")
        def sh(e):
            return shadow.get((e["ticker"], e["buy"]["created_at"]))
        matched = [e for e in rts if sh(e) is not None]
        groups = [
            ("all filled Combined signals with a shadow return (= line B)", matched),
            ("re-entry after ENTRY_STOP", [e for e in matched if e["prev"] and e["prev"]["exit"] == "ENTRY_STOP"]),
            ("re-entry after ENTRY_STOP, gap <= 1", [e for e in matched if e["prev"] and e["prev"]["exit"] == "ENTRY_STOP" and e["gap_sessions"] <= 1]),
            ("re-entry after ENTRY_STOP, gap <= 3", [e for e in matched if e["prev"] and e["prev"]["exit"] == "ENTRY_STOP" and e["gap_sessions"] <= 3]),
            ("re-entry after TRAIL_STOP", [e for e in matched if e["prev"] and e["prev"]["exit"] == "TRAIL_STOP"]),
            ("re-entry after TP / OTHER", [e for e in matched if e["prev"] and e["prev"]["exit"] in ("TP", "OTHER")]),
            ("first entry of a ticker", [e for e in matched if not e["prev"]]),
            ("everything except re-entry after ENTRY_STOP", [e for e in matched if not (e["prev"] and e["prev"]["exit"] == "ENTRY_STOP")]),
        ]
        print("| subset | N | shadow expectancy | real fill-to-fill |")
        print("|---|---|---|---|")
        for name, sub in groups:
            if sub:
                print(f"| {name} | {len(sub)} | {pct(statistics.fmean(sh(e) for e in sub))} | "
                      f"{pct(statistics.fmean(e['ret'] for e in sub))} |")
        if args.line_a is not None and matched:
            b_all = statistics.fmean(sh(e) for e in matched)
            rest = [e for e in matched if not (e["prev"] and e["prev"]["exit"] == "ENTRY_STOP")]
            b_rest = statistics.fmean(sh(e) for e in rest)
            print(f"\nselection effect B − A: {pct(b_all - args.line_a)} per trade; "
                  f"without after-stop re-entries: {pct(b_rest - args.line_a)} per trade "
                  f"→ after-stop re-entries carry {100 * (b_rest - b_all) / (args.line_a - b_all):.0f} % of it "
                  f"(they are {100 * (len(matched) - len(rest)) / len(matched):.0f} % of the filled signals).")
            for n in (1, 3):
                rest_n = [e for e in matched if not (e["prev"] and e["prev"]["exit"] == "ENTRY_STOP" and e["gap_sessions"] <= n)]
                b_n = statistics.fmean(sh(e) for e in rest_n)
                print(f"without after-stop re-entries at gap <= {n}: {pct(b_n - args.line_a)} per trade "
                      f"→ {100 * (b_n - b_all) / (args.line_a - b_all):.0f} % of the effect.")
        print()

    # ── 6. listing ───────────────────────────────────────────────────────
    print("## 6. Every re-entry after an ENTRY_STOP\n")
    print("| ticker | stop exit (UTC) | stop P&L | re-entry (UTC) | gap sessions | gap h | re-entry exit | re-entry return | re-entry P&L | shadow ret |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for e in after_stop:
        p = e["prev"]
        s = shadow.get((e["ticker"], e["buy"]["created_at"]))
        print(f"| {e['ticker']} | {p['sell_at']:%Y-%m-%d %H:%M} | {p['pnl']:+,.0f} | {e['buy_at']:%Y-%m-%d %H:%M} | "
              f"{e['gap_sessions']} | {e['gap_hours']:.0f} | {e['exit'] if e['sell'] else 'open'} | "
              f"{pct(e['ret']) if e['sell'] else ''} | {e['pnl'] if e['sell'] else ''} | "
              f"{pct(s) if s is not None else ''} |")


if __name__ == "__main__":
    main()
