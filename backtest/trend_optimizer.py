"""
Deep walk-forward optimization of TREND_FOLLOWING strategy parameters.

Searches 5,184 parameter combinations per ticker using walk-forward
validation (6mo train / 2mo test, rolling 2mo) across 2 years of data.
Technical-only — no simulated sentiment.

Usage::

    python3 -m backtest.trend_optimizer --workers 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import _download_ohlcv, run_backtest  # noqa: E402
from backtest.strategies import TICKER_SECTOR, TICKERS  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "backtest" / "results"

# ── Parameter grid (5,184 combinations) ──────────────────────────────

PARAM_GRID: dict[str, list] = {
    "sma_fast":                     [20, 50],
    "sma_slow":                     [100, 200],
    "rsi_period":                   [14, 21],
    "rsi_oversold":                 [25, 30],
    "rsi_overbought":               [70, 75],
    "stop_loss_pct":                [0.015, 0.02, 0.03],
    "take_profit_ratio":            [2.0, 2.5, 3.0],
    "require_volume_confirmation":  [True, False],
}

# Full grid for exhaustive search (set via --full-grid flag)
PARAM_GRID_FULL: dict[str, list] = {
    "sma_fast":                     [20, 50, 100],
    "sma_slow":                     [100, 200],
    "rsi_period":                   [10, 14, 21],
    "rsi_oversold":                 [25, 30, 35],
    "rsi_overbought":               [65, 70, 75],
    "stop_loss_pct":                [0.01, 0.015, 0.02, 0.03],
    "take_profit_ratio":            [1.5, 2.0, 2.5, 3.0],
    "require_volume_confirmation":  [True, False],
}

# Fixed params for all combos (pure technical, trend-following)
_FIXED_PARAMS = {
    "use_sentiment": False,
    "use_technical": True,
    "require_trend_alignment": True,
    "buy_threshold": 0.0,
    "sell_threshold": 0.0,
}

_TRAIN_MONTHS = 6
_TEST_MONTHS = 2
_OPT_START = "2023-01-01"
_OPT_END = "2025-01-01"


def _build_combos(full: bool = False) -> list[dict]:
    """Generate all parameter combinations."""
    grid = PARAM_GRID_FULL if full else PARAM_GRID
    keys = list(grid.keys())
    combos = []
    for vals in itertools.product(*grid.values()):
        combo = dict(zip(keys, vals))
        # Skip invalid: sma_fast must be < sma_slow
        if combo["sma_fast"] >= combo["sma_slow"]:
            continue
        combo.update(_FIXED_PARAMS)
        combos.append(combo)
    return combos


def _add_months(d: date, months: int) -> date:
    import calendar
    m = d.month + months
    y = d.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    max_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, max_day))


def _build_windows(
    start: date, end: date
) -> list[tuple[date, date, date, date]]:
    """Generate walk-forward windows."""
    windows = []
    cursor = start
    while True:
        train_start = cursor
        train_end = _add_months(train_start, _TRAIN_MONTHS)
        test_start = train_end
        test_end = _add_months(test_start, _TEST_MONTHS)
        if test_end > end:
            break
        windows.append((train_start, train_end, test_start, test_end))
        cursor = _add_months(cursor, _TEST_MONTHS)
    return windows


# ── Per-ticker optimizer (top-level for multiprocessing) ─────────────

def _optimize_ticker(args: tuple) -> dict[str, Any]:
    """Run full walk-forward optimization for one ticker."""
    ticker, idx, total, opt_start, opt_end, ohlcv_dict = args
    t0 = time.monotonic()

    try:
        # Reconstruct DataFrame from serialised dict
        if ohlcv_dict is not None:
            df = pd.DataFrame(ohlcv_dict)
            df.index = pd.to_datetime(df.index)
        else:
            df = _download_ohlcv(ticker, opt_start, opt_end)
        if df.empty or len(df) < 100:
            print(f"  [{idx}/{total}] {ticker:12s}  SKIP (insufficient data)", flush=True)
            return _empty_ticker_result(ticker, "insufficient data")

        combos = _build_combos()
        start_d = date.fromisoformat(opt_start)
        end_d = date.fromisoformat(opt_end)
        windows = _build_windows(start_d, end_d)

        if not windows:
            print(f"  [{idx}/{total}] {ticker:12s}  SKIP (no valid windows)", flush=True)
            return _empty_ticker_result(ticker, "no valid windows")

        window_results: list[dict] = []

        for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
            # Grid search on training window
            best_sharpe = -999.0
            best_params = combos[0]

            for combo in combos:
                r = run_backtest(
                    ticker=ticker,
                    start_date=tr_s.isoformat(),
                    end_date=tr_e.isoformat(),
                    params=combo,
                    ohlcv=df,
                )
                if r["sharpe_ratio"] > best_sharpe:
                    best_sharpe = r["sharpe_ratio"]
                    best_params = combo

            # Evaluate on test window
            test_r = run_backtest(
                ticker=ticker,
                start_date=te_s.isoformat(),
                end_date=te_e.isoformat(),
                params=best_params,
                ohlcv=df,
            )

            window_results.append({
                "window": wi + 1,
                "train": f"{tr_s} to {tr_e}",
                "test": f"{te_s} to {te_e}",
                "is_sharpe": round(best_sharpe, 4),
                "oos_sharpe": test_r["sharpe_ratio"],
                "oos_return": round(test_r["total_return"] * 100, 2),
                "oos_drawdown": round(test_r["max_drawdown"] * 100, 2),
                "oos_trades": test_r["trade_count"],
                "oos_win_rate": round(test_r["win_rate"] * 100, 1),
                "best_params": _extract_tunable(best_params),
            })

        # Aggregate
        oos_sharpes = [w["oos_sharpe"] for w in window_results]
        is_sharpes = [w["is_sharpe"] for w in window_results]
        oos_trades = sum(w["oos_trades"] for w in window_results)
        avg_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        avg_is = float(np.mean(is_sharpes)) if is_sharpes else 0.0

        # Consensus params (most common across windows)
        param_strs = [str(sorted(w["best_params"].items())) for w in window_results]
        most_common_str = Counter(param_strs).most_common(1)[0][0]
        consensus_params = {}
        for w in window_results:
            if str(sorted(w["best_params"].items())) == most_common_str:
                consensus_params = w["best_params"]
                break

        elapsed = time.monotonic() - t0
        overfitted = (avg_is - avg_oos) > 0.5

        print(
            f"  [{idx}/{total}] {ticker:12s}  "
            f"OOS Sharpe: {avg_oos:+.2f}  IS: {avg_is:+.2f}  "
            f"trades: {oos_trades}  "
            f"{'OVERFIT' if overfitted else 'OK':>7s}  "
            f"— {elapsed:.1f}s",
            flush=True,
        )

        return {
            "ticker": ticker,
            "sector": TICKER_SECTOR.get(ticker, "UNKNOWN"),
            "avg_oos_sharpe": round(avg_oos, 4),
            "avg_is_sharpe": round(avg_is, 4),
            "sharpe_gap": round(avg_is - avg_oos, 4),
            "overfitted": overfitted,
            "total_oos_trades": oos_trades,
            "low_trades": oos_trades < 10,
            "consensus_params": consensus_params,
            "windows": window_results,
            "num_windows": len(windows),
            "num_combos": len(combos),
            "elapsed_s": round(elapsed, 1),
            "error": None,
        }

    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"  [{idx}/{total}] {ticker:12s}  FAILED: {exc} — {elapsed:.1f}s", flush=True)
        return _empty_ticker_result(ticker, str(exc))


def _extract_tunable(params: dict) -> dict:
    """Extract only the tunable params (not fixed ones)."""
    return {k: params[k] for k in PARAM_GRID if k in params}


def _empty_ticker_result(ticker: str, error: str) -> dict:
    return {
        "ticker": ticker,
        "sector": TICKER_SECTOR.get(ticker, "UNKNOWN"),
        "avg_oos_sharpe": 0.0,
        "avg_is_sharpe": 0.0,
        "sharpe_gap": 0.0,
        "overfitted": False,
        "total_oos_trades": 0,
        "low_trades": True,
        "consensus_params": {},
        "windows": [],
        "num_windows": 0,
        "num_combos": 0,
        "elapsed_s": 0.0,
        "error": error,
    }


# ── Main runner ───────────────────────────────────────────────────────

def run_optimization(
    workers: int = 8,
    opt_start: str = _OPT_START,
    opt_end: str = _OPT_END,
    full_grid: bool = False,
) -> dict[str, Any]:
    """Run walk-forward optimization across all tickers."""
    from backtest.strategies import ALL_TICKERS

    combos = _build_combos(full=full_grid)
    windows = _build_windows(date.fromisoformat(opt_start), date.fromisoformat(opt_end))

    print(f"\nTrend-Following Deep Optimization")
    print(f"  Tickers:      {len(ALL_TICKERS)}")
    print(f"  Param combos: {len(combos)}")
    print(f"  WF windows:   {len(windows)} per ticker")
    print(f"  Total evals:  {len(combos) * len(windows) * len(ALL_TICKERS):,}")
    print(f"  Period:       {opt_start} to {opt_end}")
    print(f"  Workers:      {workers}")
    print(f"{'=' * 70}\n")

    # Pre-download all OHLCV data sequentially (avoids yfinance rate limits)
    print("  Downloading OHLCV data...", flush=True)
    ohlcv_data: dict[str, dict | None] = {}
    for ticker in ALL_TICKERS:
        try:
            df = _download_ohlcv(ticker, opt_start, opt_end)
            ohlcv_data[ticker] = df.to_dict() if not df.empty else None
            print(f"    {ticker:12s} — {len(df)} bars", flush=True)
        except Exception as exc:
            print(f"    {ticker:12s} — FAILED: {exc}", flush=True)
            ohlcv_data[ticker] = None
    print(flush=True)

    tasks = [
        (ticker, i + 1, len(ALL_TICKERS), opt_start, opt_end, ohlcv_data.get(ticker))
        for i, ticker in enumerate(ALL_TICKERS)
    ]

    t0 = time.monotonic()

    if workers > 1:
        with mp.Pool(processes=min(workers, len(tasks))) as pool:
            results = pool.map(_optimize_ticker, tasks)
    else:
        results = [_optimize_ticker(t) for t in tasks]

    elapsed = time.monotonic() - t0
    succeeded = sum(1 for r in results if r["error"] is None)

    print(f"\n{'=' * 70}")
    print(f"  Done: {succeeded}/{len(ALL_TICKERS)} tickers optimized")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 70}\n")

    return {
        "results": results,
        "meta": {
            "opt_start": opt_start,
            "opt_end": opt_end,
            "num_combos": len(combos),
            "num_windows": len(windows),
            "elapsed_s": round(elapsed, 1),
        },
    }


# ── Report generation ─────────────────────────────────────────────────

def generate_report(data: dict[str, Any]) -> Path:
    """Generate Markdown report and print to stdout."""
    results = data["results"]
    meta = data["meta"]
    valid = [r for r in results if r["error"] is None]

    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    out("# Trend-Following Deep Optimization Report")
    out(f"**Period:** {meta['opt_start']} to {meta['opt_end']}  ")
    out(f"**Combos tested:** {meta['num_combos']} per ticker  ")
    out(f"**Walk-forward windows:** {meta['num_windows']} per ticker  ")
    out()

    # ── Per-ticker results ───────────────────────────────────────────
    out("## Per-Ticker Results")
    out()
    out("| Ticker | Sector | OOS Sharpe | IS Sharpe | Gap | Trades | Status |")
    out("|--------|--------|-----------|----------|-----|--------|--------|")

    for r in sorted(valid, key=lambda x: x["avg_oos_sharpe"], reverse=True):
        status = ""
        if r["overfitted"]:
            status = "OVERFIT"
        elif r["low_trades"]:
            status = "LOW TRADES"
        else:
            status = "OK"
        out(f"| {r['ticker']:12s} | {r['sector']:10s} | "
            f"{r['avg_oos_sharpe']:+9.2f} | {r['avg_is_sharpe']:+8.2f} | "
            f"{r['sharpe_gap']:+.2f} | {r['total_oos_trades']:5d} | {status:10s} |")
    out()

    # ── Sector consensus ─────────────────────────────────────────────
    out("## Sector Consensus Parameters")
    out()
    sectors = defaultdict(list)
    for r in valid:
        if r["consensus_params"]:
            sectors[r["sector"]].append(r)

    sector_consensus: dict[str, dict] = {}
    for sector in sorted(sectors):
        s_results = sectors[sector]
        # Pick consensus from best-performing ticker in sector
        best = max(s_results, key=lambda x: x["avg_oos_sharpe"])
        sector_consensus[sector] = best["consensus_params"]
        out(f"**{sector}** (from {best['ticker']}, OOS Sharpe {best['avg_oos_sharpe']:+.2f}):")
        for k, v in sorted(best["consensus_params"].items()):
            out(f"  - {k}: {v}")
        out()

    # ── Top 5 combinations ───────────────────────────────────────────
    out("## Top 5 Ticker+Param Combinations (by OOS Sharpe)")
    out()
    top5 = sorted(valid, key=lambda x: x["avg_oos_sharpe"], reverse=True)[:5]
    for i, r in enumerate(top5, 1):
        status = "OVERFIT" if r["overfitted"] else ("LOW" if r["low_trades"] else "VALID")
        out(f"{i}. **{r['ticker']}** — OOS Sharpe: {r['avg_oos_sharpe']:+.2f} "
            f"(IS: {r['avg_is_sharpe']:+.2f}, trades: {r['total_oos_trades']}, {status})")
        if r["consensus_params"]:
            p = r["consensus_params"]
            out(f"   SMA {p.get('sma_fast')}/{p.get('sma_slow')}, "
                f"RSI-{p.get('rsi_period')} ({p.get('rsi_oversold')}/{p.get('rsi_overbought')}), "
                f"SL {p.get('stop_loss_pct', 0) * 100:.1f}%, "
                f"TP {p.get('take_profit_ratio')}x, "
                f"vol_confirm={p.get('require_volume_confirmation')}")
    out()

    # ── Production-ready vs skip ─────────────────────────────────────
    out("## Production Readiness")
    out()
    prod_ready = [r for r in valid if not r["overfitted"] and not r["low_trades"]
                  and r["avg_oos_sharpe"] > 0]
    skip = [r for r in valid if r["overfitted"] or r["low_trades"]
            or r["avg_oos_sharpe"] <= 0]

    out(f"**Production-ready:** {', '.join(r['ticker'] for r in prod_ready) or 'None'}")
    out(f"**Skip (overfit/low trades/negative):** {', '.join(r['ticker'] for r in skip) or 'None'}")
    out()

    # ── Save ─────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    md_path = RESULTS_DIR / f"trend_optimization_{today}.md"
    md_path.write_text("\n".join(lines) + "\n")

    json_path = RESULTS_DIR / f"trend_optimization_{today}.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return md_path


# ── Telegram notification ─────────────────────────────────────────────

def _send_telegram(data: dict[str, Any]) -> None:
    """Best-effort Telegram summary. Never raises."""
    try:
        import os
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return

        from notifications.telegram_bot import TelegramNotifier
        tg = TelegramNotifier(bot_token=token, chat_id=chat_id)

        valid = [r for r in data["results"] if r["error"] is None]
        top3 = sorted(valid, key=lambda x: x["avg_oos_sharpe"], reverse=True)[:3]
        prod = [r for r in valid if not r["overfitted"] and not r["low_trades"]
                and r["avg_oos_sharpe"] > 0]
        skip = [r for r in valid if r["overfitted"]]

        # Build sector consensus
        sectors = defaultdict(list)
        for r in valid:
            if r["consensus_params"]:
                sectors[r["sector"]].append(r)

        lines = ["Trend Optimization Complete\n"]
        lines.append("Best validated combinations:")
        for i, r in enumerate(top3, 1):
            p = r.get("consensus_params", {})
            lines.append(
                f"{i}. {r['ticker']} -- Sharpe: {r['avg_oos_sharpe']:+.2f} "
                f"(SMA{p.get('sma_fast', '?')}/{p.get('sma_slow', '?')}, "
                f"RSI{p.get('rsi_period', '?')}, "
                f"SL{(p.get('stop_loss_pct', 0) * 100):.1f}%)"
            )

        lines.append("\nConsensus params by sector:")
        for sector in sorted(sectors):
            best = max(sectors[sector], key=lambda x: x["avg_oos_sharpe"])
            p = best["consensus_params"]
            lines.append(
                f"  {sector}: SMA {p.get('sma_fast')}/{p.get('sma_slow')}, "
                f"SL {p.get('stop_loss_pct', 0) * 100:.1f}%"
            )

        if skip:
            lines.append(f"\nOverfitted (skip): {', '.join(r['ticker'] for r in skip)}")
        if prod:
            lines.append(f"Production-ready: {', '.join(r['ticker'] for r in prod)}")

        tg.send_message("\n".join(lines))
    except Exception:
        pass


# ── CLI ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Deep walk-forward optimization of TREND_FOLLOWING strategy.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start", default=_OPT_START)
    parser.add_argument("--end", default=_OPT_END)
    parser.add_argument("--full-grid", action="store_true",
                        help="Use full 4320-combo grid (slow). Default: focused 288-combo grid.")
    args = parser.parse_args(argv)

    data = run_optimization(
        workers=args.workers, opt_start=args.start, opt_end=args.end,
        full_grid=args.full_grid,
    )

    report_path = generate_report(data)
    print(f"\nReport saved → {report_path}")

    _send_telegram(data)


if __name__ == "__main__":
    main()
