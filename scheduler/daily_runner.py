"""
Async batch runner for the News Trading System.

Processes a watchlist of tickers concurrently using ``asyncio.gather()``
with configurable concurrency limits.  Prints progress after each ticker.

Usage::

    # Run now with default watchlist (5 concurrent tickers)
    python -m scheduler.daily_runner --now

    # Custom watchlist + balance
    python -m scheduler.daily_runner --now --watchlist AAPL,MSFT,NVDA --balance 25000

    # More concurrent workers
    python -m scheduler.daily_runner --now --workers 8

    # Dry run (no trade execution)
    python -m scheduler.daily_runner --now --no-execute

    # Benchmark: sync vs async on 10 tickers
    python -m scheduler.daily_runner --benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from orchestrator.coordinator import Coordinator

# Default watchlist — popular US large-caps across sectors
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "JPM", "V", "UNH",
    "JNJ", "WMT", "PG", "MA", "HD",
    "BAC", "XOM", "PFE", "COST", "ABBV",
]


# ── Progress tracker ──────────────────────────────────────────────────

class _ProgressTracker:
    """Thread-safe progress counter for async batch runs."""

    def __init__(self, total: int) -> None:
        self._total = total
        self._completed = 0
        self._lock = asyncio.Lock()
        self._results: list[str] = []

    async def record(self, ticker: str, result: dict | None, error: str | None) -> None:
        async with self._lock:
            self._completed += 1
            idx = self._completed
        if error:
            line = f"[{idx}/{self._total}] {ticker} x FAILED — {error}"
        else:
            sig = result.get("combined_signal", "?")
            conf = result.get("confidence", 0)
            elapsed = result.get("elapsed_s", 0)
            line = f"[{idx}/{self._total}] {ticker} \u2713 {sig} ({conf:.0%}) — {elapsed:.1f}s"
        print(line, flush=True)
        async with self._lock:
            self._results.append(line)

    @property
    def lines(self) -> list[str]:
        return list(self._results)


# ── Single-ticker wrapper ─────────────────────────────────────────────

async def _process_ticker(
    coordinator: Coordinator,
    ticker: str,
    *,
    account_balance: float,
    execute: bool,
    api_semaphore: asyncio.Semaphore,
    data_semaphore: asyncio.Semaphore,
    db_lock: asyncio.Lock,
    worker_semaphore: asyncio.Semaphore,
    tracker: _ProgressTracker,
) -> dict | None:
    """Process one ticker under the worker semaphore, report progress."""
    async with worker_semaphore:
        try:
            result = await coordinator.analyse_ticker_async(
                ticker,
                account_balance=account_balance,
                execute=execute,
                api_semaphore=api_semaphore,
                data_semaphore=data_semaphore,
                db_lock=db_lock,
            )
            await tracker.record(ticker, result, None)
            return result
        except Exception as exc:
            await tracker.record(ticker, None, str(exc))
            return None


# ── Batch runner ──────────────────────────────────────────────────────

async def run_batch(
    tickers: list[str],
    *,
    workers: int = 5,
    account_balance: float = 10_000.0,
    execute: bool = False,
) -> dict:
    """
    Analyse a list of tickers concurrently.

    Args:
        tickers:         List of ticker symbols.
        workers:         Max concurrent tickers.
        account_balance: Account size in USD.
        execute:         When True, execute trades via broker.

    Returns:
        dict with keys: results, elapsed_s, success_count, fail_count.
    """
    api_semaphore = asyncio.Semaphore(5)
    data_semaphore = asyncio.Semaphore(10)
    db_lock = asyncio.Lock()
    worker_semaphore = asyncio.Semaphore(workers)
    tracker = _ProgressTracker(len(tickers))

    coordinator = Coordinator()

    t0 = time.monotonic()

    tasks = [
        _process_ticker(
            coordinator,
            ticker,
            account_balance=account_balance,
            execute=execute,
            api_semaphore=api_semaphore,
            data_semaphore=data_semaphore,
            db_lock=db_lock,
            worker_semaphore=worker_semaphore,
            tracker=tracker,
        )
        for ticker in tickers
    ]

    results = await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t0
    success = sum(1 for r in results if r is not None)

    return {
        "results": results,
        "elapsed_s": round(elapsed, 2),
        "success_count": success,
        "fail_count": len(tickers) - success,
    }


# ── Sync runner (for comparison) ──────────────────────────────────────

def run_batch_sync(
    tickers: list[str],
    *,
    account_balance: float = 10_000.0,
    execute: bool = False,
) -> dict:
    """
    Analyse tickers sequentially (for benchmarking).

    Returns:
        dict with keys: results, elapsed_s, success_count, fail_count.
    """
    coordinator = Coordinator()
    t0 = time.monotonic()
    results = []
    for i, ticker in enumerate(tickers, 1):
        try:
            r = coordinator.run_combined(
                ticker,
                verbose=False,
                account_balance=account_balance,
                execute=execute,
            )
            sig = r.get("combined_signal", "?")
            conf = r.get("confidence", 0)
            elapsed_tick = time.monotonic() - t0
            print(f"[{i}/{len(tickers)}] {ticker} \u2713 {sig} ({conf:.0%}) — {elapsed_tick:.1f}s (sync)")
            results.append(r)
        except Exception as exc:
            print(f"[{i}/{len(tickers)}] {ticker} x FAILED — {exc} (sync)")
            results.append(None)

    elapsed = time.monotonic() - t0
    success = sum(1 for r in results if r is not None)
    return {
        "results": results,
        "elapsed_s": round(elapsed, 2),
        "success_count": success,
        "fail_count": len(tickers) - success,
    }


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="News Trading System — async batch runner"
    )
    parser.add_argument(
        "--now",
        action="store_true",
        default=False,
        help="Run the pipeline immediately (required to start).",
    )
    parser.add_argument(
        "--watchlist",
        type=str,
        default=None,
        help="Comma-separated list of tickers (default: built-in 20 stocks).",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10_000.0,
        metavar="USD",
        help="Account balance in USD (default: 10000).",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        default=False,
        help="Skip trade execution (analysis only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        metavar="N",
        help="Max concurrent tickers (default: 3).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run sync vs async benchmark on 10 tickers.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.benchmark:
        _run_benchmark()
        return

    if not args.now:
        print("Use --now to run the pipeline. See --help for options.")
        return

    tickers = (
        [t.strip().upper() for t in args.watchlist.split(",")]
        if args.watchlist
        else DEFAULT_WATCHLIST
    )
    execute = not args.no_execute

    print(f"\nAsync batch runner")
    print(f"  Tickers:  {len(tickers)}")
    print(f"  Workers:  {args.workers}")
    print(f"  Balance:  ${args.balance:,.2f}")
    print(f"  Execute:  {execute}")
    print(f"{'=' * 50}\n")

    batch = asyncio.run(
        run_batch(
            tickers,
            workers=args.workers,
            account_balance=args.balance,
            execute=execute,
        )
    )

    print(f"\n{'=' * 50}")
    print(f"  Done: {batch['success_count']}/{len(tickers)} succeeded")
    print(f"  Time: {batch['elapsed_s']:.1f}s")
    print(f"{'=' * 50}\n")


def _run_benchmark() -> None:
    """Run sync vs async comparison on 10 tickers."""
    tickers = DEFAULT_WATCHLIST[:10]

    print(f"\nBenchmark: {len(tickers)} tickers")
    print("=" * 50)

    # Sync
    print("\n--- Sync ---")
    sync_result = run_batch_sync(tickers, execute=False)
    print(f"\nSync: {len(tickers)} tickers in {sync_result['elapsed_s']:.1f}s")

    # Async
    print("\n--- Async (5 workers) ---")
    async_result = asyncio.run(
        run_batch(tickers, workers=5, execute=False)
    )
    print(f"\nAsync: {len(tickers)} tickers in {async_result['elapsed_s']:.1f}s")

    # Comparison
    speedup = sync_result["elapsed_s"] / async_result["elapsed_s"] if async_result["elapsed_s"] > 0 else 0
    print(f"\n{'=' * 50}")
    print(f"  Sync:    {sync_result['elapsed_s']:.1f}s")
    print(f"  Async:   {async_result['elapsed_s']:.1f}s")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
