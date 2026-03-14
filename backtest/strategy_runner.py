"""
Strategy comparison runner — backtests multiple strategies across multiple
tickers in parallel and produces a ranked comparison report.

Usage::

    python3 -m backtest.strategy_runner
    python3 -m backtest.strategy_runner --tickers NVDA,AAPL,BTC --strategies MOMENTUM,BASELINE
    python3 -m backtest.strategy_runner --workers 8 --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import run_backtest  # noqa: E402
from backtest.strategies import (  # noqa: E402
    ALL_TICKERS,
    DEFAULT_BALANCE,
    DEFAULT_END,
    DEFAULT_START,
    STRATEGIES,
    TICKER_SECTOR,
)

# ── Results directory ─────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "backtest" / "results"


# ── Worker function (top-level for pickling) ──────────────────────────

def _run_one(args: tuple) -> dict[str, Any]:
    """Run a single backtest.  Returns a result dict or error dict."""
    ticker, strategy_name, params, start, end, balance, task_idx, total = args
    t0 = time.monotonic()
    try:
        result = run_backtest(
            ticker=ticker,
            start_date=start,
            end_date=end,
            params=params,
            account_balance=balance,
        )
        elapsed = time.monotonic() - t0
        sharpe = result["sharpe_ratio"]
        print(
            f"  [{task_idx}/{total}] {ticker:12s} x {strategy_name:20s} "
            f"Sharpe: {sharpe:+.2f}  — {elapsed:.1f}s",
            flush=True,
        )
        return {
            "ticker": ticker,
            "strategy": strategy_name,
            "sector": TICKER_SECTOR.get(ticker, "UNKNOWN"),
            "sharpe": result["sharpe_ratio"],
            "total_return_pct": round(result["total_return"] * 100, 2),
            "max_drawdown_pct": round(result["max_drawdown"] * 100, 2),
            "win_rate": round(result["win_rate"] * 100, 1),
            "total_trades": result["trade_count"],
            "elapsed_s": round(elapsed, 2),
            "error": None,
        }
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(
            f"  [{task_idx}/{total}] {ticker:12s} x {strategy_name:20s} "
            f"FAILED: {exc}  — {elapsed:.1f}s",
            flush=True,
        )
        return {
            "ticker": ticker,
            "strategy": strategy_name,
            "sector": TICKER_SECTOR.get(ticker, "UNKNOWN"),
            "sharpe": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "elapsed_s": round(elapsed, 2),
            "error": str(exc),
        }


# ── Main runner ───────────────────────────────────────────────────────

def run_comparison(
    tickers: list[str] | None = None,
    strategies: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    balance: float = DEFAULT_BALANCE,
    workers: int = 8,
) -> dict[str, Any]:
    """
    Run strategy comparison across tickers and strategies.

    Returns:
        dict with keys: results (list[dict]), meta (dict).
    """
    tickers = tickers or ALL_TICKERS
    strategy_names = strategies or list(STRATEGIES.keys())

    # Validate strategy names
    for s in strategy_names:
        if s not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {s}. Valid: {list(STRATEGIES.keys())}")

    # Build task list
    tasks = []
    idx = 0
    total = len(tickers) * len(strategy_names)
    for ticker in tickers:
        for sname in strategy_names:
            idx += 1
            tasks.append((
                ticker, sname, STRATEGIES[sname],
                start, end, balance, idx, total,
            ))

    print(f"\nStrategy Comparison — {total} backtests")
    print(f"  Tickers:    {len(tickers)}")
    print(f"  Strategies: {len(strategy_names)}")
    print(f"  Period:     {start} to {end}")
    print(f"  Workers:    {workers}")
    print(f"  Balance:    ${balance:,.0f}")
    print(f"{'=' * 70}\n")

    t0 = time.monotonic()

    if workers > 1 and len(tasks) > 1:
        with mp.Pool(processes=min(workers, len(tasks))) as pool:
            results = pool.map(_run_one, tasks)
    else:
        results = [_run_one(t) for t in tasks]

    elapsed = time.monotonic() - t0

    succeeded = sum(1 for r in results if r["error"] is None)
    failed = sum(1 for r in results if r["error"] is not None)

    print(f"\n{'=' * 70}")
    print(f"  Done: {succeeded}/{total} succeeded, {failed} failed")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 70}\n")

    meta = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "start_date": start,
        "end_date": end,
        "balance": balance,
        "tickers": tickers,
        "strategies": strategy_names,
        "total_backtests": total,
        "succeeded": succeeded,
        "failed": failed,
        "elapsed_s": round(elapsed, 2),
    }

    return {"results": results, "meta": meta}


def save_results(data: dict[str, Any]) -> Path:
    """Save results to JSON file. Returns the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    path = RESULTS_DIR / f"strategy_comparison_{today}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strategy comparison backtests in parallel.",
    )
    parser.add_argument(
        "--tickers", default="all",
        help="Comma-separated tickers or 'all' (default: all).",
    )
    parser.add_argument(
        "--strategies", default="all",
        help="Comma-separated strategy names or 'all' (default: all).",
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date.")
    parser.add_argument("--end", default=DEFAULT_END, help="End date.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers.")
    parser.add_argument(
        "--balance", type=float, default=DEFAULT_BALANCE, help="Starting capital.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    tickers = None if args.tickers.lower() == "all" else [
        t.strip().upper() for t in args.tickers.split(",")
    ]
    strategies = None if args.strategies.lower() == "all" else [
        s.strip().upper() for s in args.strategies.split(",")
    ]

    data = run_comparison(
        tickers=tickers,
        strategies=strategies,
        start=args.start,
        end=args.end,
        balance=args.balance,
        workers=args.workers,
    )

    # Save JSON
    json_path = save_results(data)
    print(f"Results saved → {json_path}")

    # Generate report
    from backtest.strategy_report import generate_report
    report_path = generate_report(data)
    print(f"Report saved  → {report_path}")


if __name__ == "__main__":
    main()
