#!/usr/bin/env python3
"""
Shadow book per strategy — what would have become of every directional
vote, evaluated on daily_ohlc with the live RiskAgent stop model.

READ-ONLY.  Opens the DB with ``mode=ro`` (URI) and never writes to it.
Stdlib only, so it runs unchanged on the VPS (python3 -m not needed).

    python3 scripts/shadow_strategy_book.py                    # live VPS path
    python3 scripts/shadow_strategy_book.py --db path/to.db    # snapshot
    python3 scripts/shadow_strategy_book.py --csv /tmp/shadow  # dump trades

═══════════════════════════════════════════════════════════════════════════
PARAMETER SET — fixed before the first run, no sweeps, no variants.
═══════════════════════════════════════════════════════════════════════════
Given by the live model (agents/risk_agent.py, orchestrator/cluster_detector.py,
execution/portfolio_manager.py):

  ATR_PERIOD          = 14      Wilder ATR(14) on H/L/raw-close (D3 convention)
  ATR_LOOKBACK_DAYS   = 90      calendar-day window read from daily_ohlc
  ATR_MIN_BARS        = 15      fewer → live falls back to fixed stops; here: skipped, counted
  STOP_MULT           = 1.5     stop_distance = ATR × 1.5
  TP_MULT             = 3.0     tp_distance   = ATR × 3.0
  STOP_FLOOR_PCT      = 0.01    stop_distance ≥ 1 % of entry (Q-012 F1)
  MIN_RR              = 2.0     tp_distance ≥ 2 × stop_distance
  ONE_POSITION_PER_TICKER       mirrors the live "Already holding" gate
  BUY side only                 the live path never opens shorts (15 WEAK SELL
                                Combined rows are excluded and counted)

NOT given by the live model — chosen ONCE, before looking at any result:

  HOLD_DAYS           = 10      trading days incl. entry day; the live path has
                                no time exit, 10 d is the system's own longest
                                logged outcome horizon (signal_events.outcome_10d_pct)
  ENTRY               = first daily bar whose regular-session open (09:30
                                America/New_York) lies strictly AFTER the signal
                                timestamp; entry price = that bar's raw open
  SAME_BAR_RULE       = SL      if a bar touches both SL and TP, count the stop
                                (conservative; daily bars carry no path)
  GAP_RULE            = open    on bars after entry day, an open beyond SL/TP
                                fills at the open, not at the level
  TIME_EXIT           = close of the HOLD_DAYS-th bar
  NOTIONAL_USD        = 10 000  flat per trade for the $ column (live sizing is
                                risk/stop capped at 10 % of ~$270 k; the cap bound
                                almost always, so flat notional is the closer proxy)
  MIN_N_FOR_SHARPE    = 10      below this no Sharpe is printed at all
  PSR / MinTRL        at 95 %   deflated variant: with exactly ONE pre-registered
                                rule set the DSR deflation term is zero, DSR ≡ PSR

Linking rules (only place where anything is inferred):
  RUN_WINDOW          = 20 min  strategy rows belong to the Combined row of the
                                same ticker+session logged ≤ 20 min later
  FILL_LOOKBACK       = 26 h    a trade_history BUY belongs to the latest directional
                                Combined row of that ticker within 26 h before the fill
  ATTRIBUTION_CUTOFF  = 2026-08-28  B3 signal_attribution recorded from here on
  STOP_TOL            = 0.003   real exit ≤ stop×(1+tol) → STOP (as PM cool-down gate)
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import math
import os
import sqlite3
import statistics
from zoneinfo import ZoneInfo

# ── parameters (see module docstring) ─────────────────────────────────────
DEFAULT_DB = "/home/trading/trading-data/news_trading.db"

ATR_PERIOD = 14
ATR_LOOKBACK_DAYS = 90
ATR_MIN_BARS = ATR_PERIOD + 1
STOP_MULT = 1.5
TP_MULT = 3.0
STOP_FLOOR_PCT = 0.01
MIN_RR = 2.0

HOLD_DAYS = 10
NOTIONAL_USD = 10_000.0
MIN_N_FOR_SHARPE = 10
PSR_CONFIDENCE = 0.95

RUN_WINDOW = dt.timedelta(minutes=20)
FILL_LOOKBACK = dt.timedelta(hours=26)
FILL_TOLERANCE = dt.timedelta(minutes=-5)
ATTRIBUTION_CUTOFF = "2026-08-28"
STOP_TOL = 0.003

NY = ZoneInfo("America/New_York")
MARKET_OPEN = dt.time(9, 30)

BUY = {"STRONG BUY", "BUY", "WEAK BUY"}
SELL = {"SELL", "STRONG SELL", "WEAK SELL"}
STRATS = ("Momentum", "Pullback", "NewsCatalyst")
BOOKS = STRATS + ("Combined",)


# ── helpers ───────────────────────────────────────────────────────────────
def _ts(s: str) -> dt.datetime:
    d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def _open_dt(date_str: str) -> dt.datetime:
    d = dt.date.fromisoformat(date_str)
    return dt.datetime.combine(d, MARKET_OPEN, tzinfo=NY)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


_Z = {0.95: 1.6449}


# ── data ──────────────────────────────────────────────────────────────────
class Store:
    """daily_ohlc in memory: ticker → list of bar dicts ascending by date."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.bars: dict[str, list[dict]] = collections.defaultdict(list)
        for r in conn.execute(
            "SELECT ticker, date, open, high, low, close FROM daily_ohlc "
            "ORDER BY ticker, date"
        ):
            self.bars[r["ticker"]].append(dict(r))
        self.index: dict[str, dict[str, int]] = {
            t: {b["date"]: i for i, b in enumerate(bs)} for t, bs in self.bars.items()
        }
        self.max_date = max((bs[-1]["date"] for bs in self.bars.values()), default="")

    def entry_index(self, ticker: str, signal_ts: dt.datetime) -> int | None:
        """Index of the first bar whose open is strictly after signal_ts."""
        bs = self.bars.get(ticker)
        if not bs:
            return None
        day = signal_ts.astimezone(NY).date().isoformat()
        # bars are sorted; scan forward from the first bar on/after `day`
        lo, hi = 0, len(bs)
        while lo < hi:
            mid = (lo + hi) // 2
            if bs[mid]["date"] < day:
                lo = mid + 1
            else:
                hi = mid
        for i in range(lo, len(bs)):
            if _open_dt(bs[i]["date"]) > signal_ts:
                return i
        return None

    def wilder_atr(self, ticker: str, entry_idx: int) -> float | None:
        """Wilder ATR(14) from bars strictly before entry, 90-calendar-day window.

        Same arithmetic as RiskAgent._fetch_atr: TR = max(H−L, |H−C_prev|,
        |L−C_prev|), seed = mean of first 14 TR, then (ATR·13 + TR) / 14.
        """
        bs = self.bars[ticker]
        entry_date = dt.date.fromisoformat(bs[entry_idx]["date"])
        start = (entry_date - dt.timedelta(days=ATR_LOOKBACK_DAYS)).isoformat()
        window = [b for b in bs[:entry_idx] if b["date"] >= start]
        if len(window) < ATR_MIN_BARS:
            return None
        trs = []
        for prev, cur in zip(window, window[1:]):
            trs.append(max(
                cur["high"] - cur["low"],
                abs(cur["high"] - prev["close"]),
                abs(cur["low"] - prev["close"]),
            ))
        atr = sum(trs[:ATR_PERIOD]) / ATR_PERIOD
        for tr in trs[ATR_PERIOD:]:
            atr = (atr * (ATR_PERIOD - 1) + tr) / ATR_PERIOD
        return atr


def evaluate(store: Store, ticker: str, signal_ts: dt.datetime) -> dict:
    """Hypothetical path of one BUY signal.  Returns a dict with status."""
    i0 = store.entry_index(ticker, signal_ts)
    if i0 is None:
        return {"status": "no_entry_bar"}
    atr = store.wilder_atr(ticker, i0)
    if atr is None or atr <= 0:
        return {"status": "no_atr"}
    bs = store.bars[ticker]
    entry = bs[i0]["open"]
    stop_d = max(atr * STOP_MULT, STOP_FLOOR_PCT * entry)
    tp_d = max(atr * TP_MULT, MIN_RR * stop_d)
    sl = round(entry - stop_d, 4)
    tp = round(entry + tp_d, 4)

    exit_px = exit_reason = exit_date = None
    hold = 0
    for k in range(HOLD_DAYS):
        i = i0 + k
        if i >= len(bs):
            break
        b = bs[i]
        hold = k + 1
        if k > 0 and b["open"] <= sl:
            exit_px, exit_reason = b["open"], "SL"
        elif k > 0 and b["open"] >= tp:
            exit_px, exit_reason = b["open"], "TP"
        elif b["low"] <= sl:
            exit_px, exit_reason = sl, "SL"          # SAME_BAR_RULE = SL
        elif b["high"] >= tp:
            exit_px, exit_reason = tp, "TP"
        elif k == HOLD_DAYS - 1:
            exit_px, exit_reason = b["close"], "TIME"
        if exit_px is not None:
            exit_date = b["date"]
            break

    out = {
        "status": "closed" if exit_px is not None else "open",
        "entry_date": bs[i0]["date"], "entry": entry, "atr": atr,
        "sl": sl, "tp": tp, "stop_pct": stop_d / entry, "hold": hold,
    }
    if exit_px is None:
        last = bs[-1]
        out.update(exit_date=last["date"], exit=last["close"], reason="OPEN",
                   ret=last["close"] / entry - 1.0)
    else:
        out.update(exit_date=exit_date, exit=exit_px, reason=exit_reason,
                   ret=exit_px / entry - 1.0)
    out["r_mult"] = (out["exit"] - entry) / stop_d
    return out


# ── statistics ────────────────────────────────────────────────────────────
def describe(rets: list[float]) -> dict:
    n = len(rets)
    if n == 0:
        return {"n": 0}
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    mean = statistics.fmean(rets)
    sd = statistics.stdev(rets) if n > 1 else float("nan")
    out = {
        "n": n, "win_rate": len(wins) / n,
        "avg_win": statistics.fmean(wins) if wins else float("nan"),
        "avg_loss": statistics.fmean(losses) if losses else float("nan"),
        "mean": mean, "sd": sd,
        "ci_lo": mean - 1.96 * sd / math.sqrt(n) if n > 1 else float("nan"),
        "ci_hi": mean + 1.96 * sd / math.sqrt(n) if n > 1 else float("nan"),
        "sum": sum(rets), "sum_usd": sum(rets) * NOTIONAL_USD,
        "median": statistics.median(rets),
    }
    if n >= MIN_N_FOR_SHARPE and sd > 0:
        sr = mean / sd
        m3 = sum((r - mean) ** 3 for r in rets) / n
        m4 = sum((r - mean) ** 4 for r in rets) / n
        var = sum((r - mean) ** 2 for r in rets) / n
        skew = m3 / var ** 1.5 if var > 0 else 0.0
        kurt = m4 / var ** 2 if var > 0 else 3.0
        denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
        denom = max(denom, 1e-9)
        psr = _phi(sr * math.sqrt(n - 1) / math.sqrt(denom))
        z = _Z[PSR_CONFIDENCE]
        min_trl = 1.0 + denom * (z / sr) ** 2 if sr > 0 else float("inf")
        out.update(sr=sr, skew=skew, kurt=kurt, psr=psr, min_trl=min_trl)
    return out


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:+.2f} %"


def fmt_stats(name: str, d: dict, holds: list[int] | None = None,
              reasons: collections.Counter | None = None,
              reason_ret: dict | None = None) -> str:
    if d["n"] == 0:
        return f"### {name}\n\nN = 0 — nothing to evaluate.\n"
    lines = [f"### {name}", "",
             "| metric | value |", "|---|---|",
             f"| closed trades N | {d['n']} |",
             f"| win rate | {100 * d['win_rate']:.1f} % |",
             f"| avg win / avg loss | {pct(d['avg_win'])} / {pct(d['avg_loss'])} |",
             f"| expectancy per trade | {pct(d['mean'])} (95 % CI {pct(d['ci_lo'])} … {pct(d['ci_hi'])}) |",
             f"| median per trade | {pct(d['median'])} |",
             f"| sum of returns | {pct(d['sum'])} = {d['sum_usd']:+,.0f} USD @ {NOTIONAL_USD:,.0f} USD flat |"]
    if holds:
        hs = sorted(holds)
        q = lambda p: hs[min(len(hs) - 1, int(p * len(hs)))]
        lines.append(f"| holding days (min / p25 / median / p75 / max / mean) | "
                     f"{hs[0]} / {q(.25)} / {q(.5)} / {q(.75)} / {hs[-1]} / {statistics.fmean(hs):.1f} |")
    if reasons:
        parts = []
        for k in ("SL", "TP", "TIME"):
            c = reasons.get(k, 0)
            rr = reason_ret.get(k) if reason_ret else None
            m = f", mean {pct(statistics.fmean(rr))}" if rr else ""
            parts.append(f"{k} {c} ({100 * c / d['n']:.0f} %{m})")
        lines.append(f"| exit reason | {' · '.join(parts)} |")
    if "sr" in d:
        if d["sr"] <= 0:
            verdict, trl = "n/a (SR ≤ 0, nothing to confirm)", "n/a"
        else:
            trl = f"{d['min_trl']:.0f}"
            verdict = ("sample supports it" if d["n"] >= d["min_trl"]
                       else f"sample does NOT support it (need ≥ {trl})")
        lines.append(f"| Sharpe per trade (not annualised) | {d['sr']:.3f} (skew {d['skew']:.2f}, kurt {d['kurt']:.2f}) |")
        lines.append(f"| PSR = DSR (1 trial): P(true SR > 0) | {100 * d['psr']:.1f} % |")
        lines.append(f"| MinTRL @95 % vs N | {trl} vs {d['n']} → {verdict} |")
    else:
        lines.append(f"| Sharpe | not reported (N < {MIN_N_FOR_SHARPE} or zero variance) |")
    return "\n".join(lines) + "\n"


# ── main ──────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--csv", metavar="DIR", help="dump per-trade books as CSV")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    store = Store(conn)
    covered = set(store.bars)
    print(f"# Shadow book — {args.db}\n")
    print(f"daily_ohlc: {len(covered)} tickers, last bar {store.max_date}; "
          f"parameters see script header.\n")

    rows = conn.execute(
        "SELECT id, timestamp, session, ticker, strategy, signal, confidence, "
        "price_at_signal, trade_executed, trade_id, signal_path FROM signal_events "
        "WHERE strategy IN ('Momentum','Pullback','NewsCatalyst','Combined') "
        "ORDER BY timestamp, id"
    ).fetchall()

    # ── run grouping (votes ↔ Combined row), as in cluster_gate_backtest ──
    by_ticker: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_ticker[r["ticker"]].append(r)
    combined_votes: dict[int, dict] = {}          # combined id → {strategy: row}
    grouping = collections.Counter()
    for t, rs in by_ticker.items():
        buf: list = []
        for r in rs:
            if r["strategy"] in STRATS:
                buf.append(r)
                continue
            tc = _ts(r["timestamp"])
            votes = {}
            for v in buf:
                if v["session"] == r["session"] and dt.timedelta(0) <= tc - _ts(v["timestamp"]) <= RUN_WINDOW:
                    votes[v["strategy"]] = v
            buf = []
            combined_votes[r["id"]] = votes
            if r["signal"].upper() in BUY:
                grouping[len(votes)] += 1

    # ── per-signal evaluation ─────────────────────────────────────────────
    excl = collections.Counter()
    evals: dict[int, dict] = {}
    signals: dict[str, list] = {b: [] for b in BOOKS}
    for r in rows:
        sig = r["signal"].upper()
        if sig == "HOLD":
            continue
        book = r["strategy"]
        if sig in SELL:
            excl[(book, "sell_side")] += 1
            continue
        if r["ticker"] not in covered:
            excl[(book, "ticker_not_in_daily_ohlc")] += 1
            continue
        e = evaluate(store, r["ticker"], _ts(r["timestamp"]))
        if e["status"] in ("no_entry_bar", "no_atr"):
            excl[(book, e["status"])] += 1
            continue
        e.update(id=r["id"], ticker=r["ticker"], ts=r["timestamp"], session=r["session"],
                 confidence=r["confidence"], signal=sig, book=book)
        evals[r["id"]] = e
        signals[book].append(e)

    # ── books with one-position-per-ticker ───────────────────────────────
    books: dict[str, list[dict]] = {}
    dedup = collections.Counter()
    for b in BOOKS:
        open_until: dict[str, str] = {}
        taken = []
        for e in signals[b]:          # already in timestamp order
            t = e["ticker"]
            if t in open_until and e["entry_date"] <= open_until[t]:
                dedup[b] += 1
                continue
            open_until[t] = e["exit_date"]
            taken.append(e)
        books[b] = taken

    print("## 1. Population and coverage\n")
    print("| book | directional rows | excluded: non-OHLC ticker | excluded: SELL side | "
          "excluded: no entry bar / no ATR | evaluable signals | folded into open position | "
          "trades | of which still open |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b in BOOKS:
        n_dir = sum(1 for r in rows if r["strategy"] == b and r["signal"].upper() != "HOLD")
        closed = [e for e in books[b] if e["status"] == "closed"]
        print(f"| {b} | {n_dir} | {excl[(b, 'ticker_not_in_daily_ohlc')]} | {excl[(b, 'sell_side')]} | "
              f"{excl[(b, 'no_entry_bar')]} / {excl[(b, 'no_atr')]} | {len(signals[b])} | "
              f"{dedup[b]} | {len(books[b])} | {len(books[b]) - len(closed)} |")
    print()
    print(f"Run grouping (directional Combined rows → number of strategy rows found within "
          f"{RUN_WINDOW.seconds // 60} min, same ticker+session): "
          f"{dict(sorted(grouping.items()))}\n")

    # ── per-book stats ────────────────────────────────────────────────────
    print("## 2. Shadow books (closed trades only)\n")
    for b in BOOKS:
        closed = [e for e in books[b] if e["status"] == "closed"]
        rets = [e["ret"] for e in closed]
        reasons = collections.Counter(e["reason"] for e in closed)
        reason_ret = collections.defaultdict(list)
        for e in closed:
            reason_ret[e["reason"]].append(e["ret"])
        print(fmt_stats(b, describe(rets), [e["hold"] for e in closed], reasons, reason_ret))
        opens = [e for e in books[b] if e["status"] == "open"]
        if opens:
            print(f"Still open at {store.max_date}: {len(opens)} trades, "
                  f"unrealised {pct(statistics.fmean(e['ret'] for e in opens))} mean.\n")
        # attribution-era split
        pre = [e["ret"] for e in closed if e["ts"] < ATTRIBUTION_CUTOFF]
        post = [e["ret"] for e in closed if e["ts"] >= ATTRIBUTION_CUTOFF]
        print(f"Era split at {ATTRIBUTION_CUTOFF}: before N={len(pre)} "
              f"expectancy {pct(statistics.fmean(pre)) if pre else 'n/a'}; "
              f"from cutoff N={len(post)} expectancy "
              f"{pct(statistics.fmean(post)) if post else 'n/a'}.\n")
        # first / last trade
        if books[b]:
            print(f"Trade window: {books[b][0]['entry_date']} … {books[b][-1]['entry_date']}; "
                  f"median stop distance {100 * statistics.median(e['stop_pct'] for e in books[b]):.2f} % "
                  f"of entry.\n")

    # ── null reference: same exit model, unconditional entry ────────────
    if books["Combined"]:
        d0 = min(e["entry_date"] for b in BOOKS for e in books[b])
        d1 = max(e["entry_date"] for b in BOOKS for e in books[b])
        null_rets = []
        for t in sorted(covered):
            for b in store.bars[t]:
                if d0 <= b["date"] <= d1:
                    e = evaluate(store, t, _open_dt(b["date"]) - dt.timedelta(minutes=1))
                    if e["status"] == "closed":
                        null_rets.append(e["ret"])
        dn = describe(null_rets)
        print(f"### Null reference (not a strategy): entry at EVERY bar open of the {len(covered)} "
              f"covered tickers, {d0} … {d1}, same exit model\n")
        print(f"N = {dn['n']} overlapping paths, expectancy {pct(dn['mean'])} "
              f"(95 % CI {pct(dn['ci_lo'])} … {pct(dn['ci_hi'])}), win rate {100 * dn['win_rate']:.1f} %, "
              f"median {pct(dn['median'])}. Paths overlap heavily (one per ticker and day), so the CI "
              f"is far too narrow — read the mean as the drift of the universe under this exit model, "
              f"nothing more.\n")

    # ── Combined vs components: overlap by (ticker, entry_date) ──────────
    print("## 3. Combined vs. its components\n")
    keyset = {b: {(e["ticker"], e["entry_date"]) for e in books[b]} for b in BOOKS}
    comb = [e for e in books["Combined"] if e["status"] == "closed"]
    print("| Combined trade also opened by … | N | expectancy |")
    print("|---|---|---|")
    for s in STRATS:
        sub = [e for e in comb if (e["ticker"], e["entry_date"]) in keyset[s]]
        print(f"| {s} | {len(sub)} | {pct(statistics.fmean(e['ret'] for e in sub)) if sub else 'n/a'} |")
    none = [e for e in comb if not any((e["ticker"], e["entry_date"]) in keyset[s] for s in STRATS)]
    print(f"| none of the three (same ticker+entry day) | {len(none)} | "
          f"{pct(statistics.fmean(e['ret'] for e in none)) if none else 'n/a'} |")
    # by number of directional votes in the run (re-derived, MIN_CONFIDENCE not applied)
    def n_votes(e: dict) -> int:
        return sum(1 for v in combined_votes.get(e["id"], {}).values()
                   if v["signal"].upper() in BUY)
    by_votes = collections.defaultdict(list)
    for e in comb:
        by_votes[n_votes(e)].append(e["ret"])
    print("\n| directional strategy votes behind the Combined trade | N | expectancy | win rate |")
    print("|---|---|---|---|")
    for k in sorted(by_votes):
        v = by_votes[k]
        print(f"| {k} | {len(v)} | {pct(statistics.fmean(v))} | "
              f"{100 * sum(1 for x in v if x > 0) / len(v):.0f} % |")
    print()

    # ── real book ─────────────────────────────────────────────────────────
    print("## 4. Real book vs. shadow (selection vs. execution)\n")
    th = conn.execute(
        "SELECT id, ticker, action, shares, price, stop_loss, take_profit, pnl, created_at "
        "FROM trade_history ORDER BY created_at, id"
    ).fetchall()
    open_buys: dict[str, list] = collections.defaultdict(list)
    round_trips = []
    orphan_sells = 0
    for r in th:
        if r["action"].upper() == "BUY":
            open_buys[r["ticker"]].append(r)
        else:
            if not open_buys[r["ticker"]]:
                orphan_sells += 1
                continue
            b = open_buys[r["ticker"]].pop(0)
            ret = r["price"] / b["price"] - 1.0
            if b["stop_loss"] and r["price"] <= b["stop_loss"] * (1 + STOP_TOL):
                reason = "SL"
            elif b["take_profit"] and r["price"] >= b["take_profit"] * (1 - STOP_TOL):
                reason = "TP"
            else:
                reason = "OTHER"
            days = (_ts(r["created_at"]).date() - _ts(b["created_at"]).date()).days
            round_trips.append({"buy": b, "sell": r, "ret": ret, "reason": reason,
                                "pnl": r["pnl"], "cal_days": days})
    still_open = sum(len(v) for v in open_buys.values())
    first_fill = _ts(th[0]["created_at"]) if th else None

    # map BUY → Combined row
    comb_dir_by_ticker: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        if r["strategy"] == "Combined" and r["signal"].upper() in BUY:
            comb_dir_by_ticker[r["ticker"]].append((r["id"], _ts(r["timestamp"])))
    matched, unmatched = [], 0
    for rt in round_trips:
        tb = _ts(rt["buy"]["created_at"])
        cands = [(i, t) for i, t in comb_dir_by_ticker[rt["buy"]["ticker"]]
                 if FILL_TOLERANCE <= tb - t <= FILL_LOOKBACK]
        if not cands:
            unmatched += 1
            continue
        cid = max(cands, key=lambda x: x[1])[0]
        rt["combined_id"] = cid
        rt["shadow"] = evals.get(cid)
        matched.append(rt)

    real_rets = [rt["ret"] for rt in round_trips]
    print(f"trade_history: {sum(1 for r in th if r['action'].upper() == 'BUY')} BUY / "
          f"{sum(1 for r in th if r['action'].upper() != 'BUY')} SELL rows → {len(round_trips)} FIFO "
          f"round trips, {orphan_sells} orphan SELLs dropped, {still_open} BUYs still open. "
          f"{len(matched)} round trips link to a Combined signal, {unmatched} do not.\n")
    print(f"Recorded realised P&L over the round trips: "
          f"{sum(rt['pnl'] for rt in round_trips):+,.0f} USD.\n")

    reasons = collections.Counter(rt["reason"] for rt in round_trips)
    rr = collections.defaultdict(list)
    for rt in round_trips:
        rr[rt["reason"]].append(rt["ret"])
    d_real = describe(real_rets)
    print(fmt_stats("Real round trips (all, fill-to-fill return)", d_real,
                    [rt["cal_days"] for rt in round_trips]))
    print("Real exit reasons (reconstructed from stop/TP levels, OTHER = trailing / signal / manual): "
          + " · ".join(f"{k} {reasons[k]} (mean {pct(statistics.fmean(rr[k]))})" for k in ("SL", "TP", "OTHER") if reasons[k])
          + "\n")

    # decomposition
    paper_start = first_fill.isoformat() if first_fill else "0000"
    a = [e["ret"] for e in signals["Combined"]
         if e["status"] == "closed" and e["ts"] >= paper_start]
    a_book = [e["ret"] for e in books["Combined"]
              if e["status"] == "closed" and e["ts"] >= paper_start]
    b_ = [rt["shadow"]["ret"] for rt in matched
          if rt["shadow"] and rt["shadow"]["status"] == "closed"]
    c = [rt["ret"] for rt in matched if rt["shadow"] and rt["shadow"]["status"] == "closed"]
    print("| line | N | expectancy | win rate | sum |")
    print("|---|---|---|---|---|")
    for name, v in (("A  shadow, every directional Combined signal in the paper era (per signal, overlapping)", a),
                    ("A' shadow, same but one position per ticker (the book of §2, paper era)", a_book),
                    ("B  shadow, the Combined signals that were actually filled (per signal)", b_),
                    ("C  real, the same filled signals (fill-to-fill)", c)):
        if v:
            print(f"| {name} | {len(v)} | {pct(statistics.fmean(v))} | "
                  f"{100 * sum(1 for x in v if x > 0) / len(v):.0f} % | {pct(sum(v))} |")
    if a and b_ and c:
        print(f"\nselection effect (B − A): {pct(statistics.fmean(b_) - statistics.fmean(a))} per trade; "
              f"execution effect (C − B): {pct(statistics.fmean(c) - statistics.fmean(b_))} per trade.\n")

    # execution detail on matched trades
    if matched:
        slip = [rt["buy"]["price"] / rt["shadow"]["entry"] - 1.0 for rt in matched if rt["shadow"]]
        same_day = sum(1 for rt in matched if rt["shadow"]
                       and rt["shadow"]["entry_date"] == _ts(rt["buy"]["created_at"]).astimezone(NY).date().isoformat())
        stop_real = [(rt["buy"]["price"] - rt["buy"]["stop_loss"]) / rt["buy"]["price"]
                     for rt in matched if rt["buy"]["stop_loss"]]
        stop_shadow = [rt["shadow"]["stop_pct"] for rt in matched if rt["shadow"]]
        print("| execution detail (matched round trips) | value |")
        print("|---|---|")
        print(f"| real fill vs shadow open, mean / median | {pct(statistics.fmean(slip))} / {pct(statistics.median(slip))} |")
        print(f"| shadow entry on the same trading day as the fill | {same_day} / {len(matched)} |")
        print(f"| stop distance real (median) vs shadow (median) | "
              f"{100 * statistics.median(stop_real):.2f} % vs {100 * statistics.median(stop_shadow):.2f} % |")
        print(f"| holding, real calendar days median vs shadow bars median | "
              f"{statistics.median(rt['cal_days'] for rt in matched)} vs "
              f"{statistics.median(rt['shadow']['hold'] for rt in matched if rt['shadow'])} |")
        cross = collections.Counter((rt["reason"], rt["shadow"]["reason"]) for rt in matched if rt["shadow"])
        print("\n| real exit → shadow exit | N |")
        print("|---|---|")
        for (r1, r2), n in sorted(cross.items(), key=lambda x: -x[1]):
            print(f"| {r1} → {r2} | {n} |")
        print()

    # ── per-strategy view of the real book (votes behind filled trades) ──
    print("## 5. Real trades by voting strategy (run-grouped, inferred before "
          f"{ATTRIBUTION_CUTOFF})\n")
    print("| strategy voted BUY in the filled run | N round trips | real expectancy | shadow expectancy of same signals |")
    print("|---|---|---|---|")
    for s in STRATS:
        sub = [rt for rt in matched
               if s in combined_votes.get(rt["combined_id"], {})
               and combined_votes[rt["combined_id"]][s]["signal"].upper() in BUY]
        sh = [rt["shadow"]["ret"] for rt in sub if rt["shadow"] and rt["shadow"]["status"] == "closed"]
        print(f"| {s} | {len(sub)} | {pct(statistics.fmean(rt['ret'] for rt in sub)) if sub else 'n/a'} | "
              f"{pct(statistics.fmean(sh)) if sh else 'n/a'} |")
    print()

    # ── optional CSV dump ─────────────────────────────────────────────────
    if args.csv:
        os.makedirs(args.csv, exist_ok=True)
        cols = ["book", "id", "ts", "session", "ticker", "signal", "confidence", "entry_date",
                "entry", "atr", "sl", "tp", "stop_pct", "exit_date", "exit", "reason", "hold",
                "ret", "r_mult", "status"]
        for b in BOOKS:
            with open(os.path.join(args.csv, f"shadow_{b}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(books[b])
        with open(os.path.join(args.csv, "real_round_trips.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ticker", "buy_at", "buy_px", "sell_at", "sell_px", "ret", "pnl",
                        "reason", "combined_id", "shadow_ret", "shadow_reason"])
            for rt in round_trips:
                sh = rt.get("shadow") or {}
                w.writerow([rt["buy"]["ticker"], rt["buy"]["created_at"], rt["buy"]["price"],
                            rt["sell"]["created_at"], rt["sell"]["price"], f"{rt['ret']:.5f}",
                            rt["pnl"], rt["reason"], rt.get("combined_id"),
                            f"{sh['ret']:.5f}" if sh else "", sh.get("reason", "")])
        print(f"CSV written to {args.csv}/")


if __name__ == "__main__":
    main()
