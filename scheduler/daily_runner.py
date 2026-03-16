"""
Async batch runner and daemon scheduler for the News Trading System.

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

    # Run as daemon (4 scheduled runs/day, weekdays only)
    python -m scheduler.daily_runner --daemon

    # Benchmark: sync vs async on 10 tickers
    python -m scheduler.daily_runner --benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from orchestrator.coordinator import Coordinator

log = logging.getLogger(__name__)

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


# ── Daemon scheduler ─────────────────────────────────────────────────

# Session-specific watchlists
_XETRA_TICKERS = ["SAP.XETRA", "SIE.XETRA"]
_US_TICKERS = ["MSFT", "META", "AAPL", "NVDA", "GOOGL", "DELL", "VST", "CEG"]

# Schedule: 4 runs per weekday, all times UTC
SCHEDULE = [
    {"name": "XETRA_OPEN", "hour": 7,  "minute": 0,  "tickers": _XETRA_TICKERS, "workers": 2, "eod": False},
    {"name": "US_OPEN",    "hour": 14, "minute": 30, "tickers": _US_TICKERS,     "workers": 3, "eod": False},
    {"name": "MIDDAY",     "hour": 18, "minute": 0,  "tickers": None,            "workers": 3, "eod": False},
    {"name": "EOD",        "hour": 22, "minute": 15, "tickers": None,            "workers": 3, "eod": True},
]

# Trading window (UTC)
_WINDOW_START = (7, 0)    # 07:00
_WINDOW_END   = (22, 30)  # 22:30


class DailyScheduler:
    """
    Daemon that sleeps between 4 daily trading runs (weekdays, UTC).

    Runs: XETRA_OPEN (07:00), US_OPEN (14:30), MIDDAY (18:00), EOD (22:15).
    """

    def __init__(self, full_watchlist: list[str] | None = None) -> None:
        self._full_watchlist = full_watchlist or self._load_watchlist()
        self._tg = self._build_telegram()

    @staticmethod
    def _load_watchlist() -> list[str]:
        path = Path(__file__).resolve().parent.parent / "config" / "watchlist.yaml"
        try:
            with open(path) as fh:
                cfg = yaml.safe_load(fh) or {}
            return cfg.get("watchlist", DEFAULT_WATCHLIST)
        except Exception:
            return DEFAULT_WATCHLIST

    @staticmethod
    def _build_telegram():
        try:
            from notifications.telegram_bot import TelegramNotifier
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if token and chat_id:
                return TelegramNotifier(bot_token=token, chat_id=chat_id)
        except Exception:
            pass
        return None

    # ── Helper methods ────────────────────────────────────────────────

    def next_run_time(self, after: datetime | None = None) -> datetime:
        """Return the next scheduled run time (UTC) after *after*."""
        now = after or datetime.now(timezone.utc)

        # Check remaining runs today if it's a weekday
        if now.weekday() < 5:
            for run in SCHEDULE:
                run_time = now.replace(
                    hour=run["hour"], minute=run["minute"],
                    second=0, microsecond=0,
                )
                if run_time > now:
                    return run_time

        # Advance to next weekday's first run
        next_day = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)

        first = SCHEDULE[0]
        return next_day.replace(hour=first["hour"], minute=first["minute"])

    def seconds_until_next_run(self) -> int:
        """Seconds from now until the next scheduled run."""
        delta = self.next_run_time() - datetime.now(timezone.utc)
        return max(0, int(delta.total_seconds()))

    def current_session(self) -> str:
        """Return ``'XETRA_OPEN'``, ``'US_OPEN'``, ``'MIDDAY'``, ``'EOD'``, or ``'CLOSED'``."""
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:
            return "CLOSED"

        now_min = now.hour * 60 + now.minute
        start_min = _WINDOW_START[0] * 60 + _WINDOW_START[1]
        end_min   = _WINDOW_END[0] * 60 + _WINDOW_END[1]

        if now_min < start_min or now_min >= end_min:
            return "CLOSED"

        session = "CLOSED"
        for run in SCHEDULE:
            if now_min >= run["hour"] * 60 + run["minute"]:
                session = run["name"]
        return session

    # ── Main loop ─────────────────────────────────────────────────────

    def run_forever(self) -> None:
        """Sleep-loop daemon: find next run, sleep, execute, repeat."""
        log.info("DailyScheduler daemon started — %d tickers in full watchlist",
                 len(self._full_watchlist))
        print(f"[scheduler] Daemon started — full watchlist: "
              f"{', '.join(self._full_watchlist)}", flush=True)

        while True:
            nrt = self.next_run_time()
            wait_s = max(0, int((nrt - datetime.now(timezone.utc)).total_seconds()))
            run_info = self._run_for_time(nrt)
            run_name = run_info["name"] if run_info else "UNKNOWN"

            print(f"[scheduler] Next: {run_name} at "
                  f"{nrt.strftime('%Y-%m-%d %H:%M')} UTC ({wait_s}s away)",
                  flush=True)
            log.info("Sleeping %ds until %s at %s", wait_s, run_name,
                     nrt.strftime("%H:%M"))

            time.sleep(wait_s)

            if run_info:
                self._execute_run(run_info)

    # ── Execution ─────────────────────────────────────────────────────

    def _run_for_time(self, dt: datetime) -> dict | None:
        for run in SCHEDULE:
            if dt.hour == run["hour"] and dt.minute == run["minute"]:
                return run
        return None

    def _execute_run(self, run: dict) -> None:
        run_name = run["name"]
        tickers = run["tickers"] or self._full_watchlist
        workers = run["workers"]
        now_str = datetime.now(timezone.utc).strftime("%H:%M")

        # Telegram: starting
        if self._tg:
            try:
                self._tg._send(
                    f"\U0001f558 *{run_name}* run starting \u2014 {now_str} UTC\n"
                    f"Tickers: {', '.join(tickers)}"
                )
            except Exception:
                pass

        log.info("Starting %s: %d tickers, %d workers",
                 run_name, len(tickers), workers)
        print(f"[scheduler] Running {run_name}: {', '.join(tickers)} "
              f"({workers} workers)", flush=True)

        try:
            batch = asyncio.run(
                run_batch(
                    tickers,
                    workers=workers,
                    account_balance=10_000.0,
                    execute=True,
                )
            )

            ok = batch["success_count"]
            total = len(tickers)
            elapsed = batch["elapsed_s"]
            log.info("%s complete: %d/%d in %.1fs", run_name, ok, total, elapsed)
            print(f"[scheduler] {run_name} done: {ok}/{total} in "
                  f"{elapsed:.1f}s", flush=True)

            # Telegram: run summary
            if self._tg:
                try:
                    self._send_run_summary(run_name, batch, tickers)
                except Exception:
                    pass

            # EOD P&L summary
            if run["eod"]:
                try:
                    send_eod_summary(self._tg, batch, tickers)
                except Exception as exc:
                    log.warning("EOD summary failed: %s", exc)

        except Exception as exc:
            log.error("Scheduler error in %s: %s", run_name, exc)
            print(f"[scheduler] ERROR in {run_name}: {exc}", flush=True)
            if self._tg:
                try:
                    self._tg._send(
                        f"\U0001f6a8 *Scheduler error in {run_name}:*\n"
                        f"{str(exc)[:300]}"
                    )
                except Exception:
                    pass

    def _send_run_summary(self, run_name: str, batch: dict,
                          tickers: list[str]) -> None:
        results = batch["results"]
        signals = []
        trades = 0
        for r in results:
            if r is None:
                continue
            sig = r.get("combined_signal", "HOLD")
            conf = r.get("confidence", 0)
            ticker = r.get("ticker", "?")
            signals.append(f"  {ticker}: {sig} ({conf:.0%})")
            if r.get("execution", {}).get("trade_id"):
                trades += 1

        lines = [
            f"\u2705 *{run_name}* complete",
            f"{batch['success_count']}/{len(tickers)} succeeded | "
            f"{batch['elapsed_s']:.0f}s",
        ]
        if trades:
            lines.append(f"Trades executed: {trades}")
        if signals:
            lines.append("")
            lines.extend(signals[:12])

        self._tg._send("\n".join(lines))


# ── EOD summary (shared by daemon + track.py --eod) ──────────────────

def send_eod_summary(tg, batch: dict, tickers: list[str]) -> None:
    """
    Send an end-of-day P&L summary via Telegram.

    Attempts to pull live data from Alpaca; falls back to batch results.
    """
    if tg is None:
        return

    results = batch["results"]
    day_name = datetime.now(timezone.utc).strftime("%a %b %d")

    # Count trades
    trades_executed = sum(
        1 for r in results
        if r and r.get("execution", {}).get("trade_id")
    )
    trades_failed = batch.get("fail_count", 0)

    # Try Alpaca for live portfolio data
    portfolio_value = 0.0
    daily_pnl = 0.0
    daily_pct = 0.0
    pos_lines: list[str] = []

    try:
        from execution.alpaca_trader import AlpacaTrader
        trader = AlpacaTrader()
        account = trader._api.get_account()
        positions = trader._api.list_positions()

        portfolio_value = float(account.portfolio_value)
        prev_close = float(account.last_equity)
        daily_pnl = portfolio_value - prev_close
        daily_pct = (daily_pnl / prev_close * 100) if prev_close else 0.0

        for p in positions:
            pct = float(p.unrealized_plpc) * 100
            sign = "+" if pct >= 0 else ""
            pos_lines.append(f"{p.symbol} {sign}{pct:.1f}%")
    except Exception as exc:
        log.debug("Alpaca data unavailable for EOD summary: %s", exc)
        # Fallback: use batch result balance
        for r in results:
            if r is not None:
                portfolio_value = r.get("account_balance", 0)
                break

    # Top signals for next session
    strong: list[str] = []
    for r in results:
        if r is None:
            continue
        sig = r.get("combined_signal", "HOLD")
        if sig not in ("HOLD", "CONFLICTING"):
            strong.append(f"{r['ticker']} {sig}")

    sign = "+" if daily_pnl >= 0 else ""
    lines = [
        f"\U0001f4ca *EOD Summary \u2014 {day_name}*",
        f"\U0001f4bc Portfolio: ${portfolio_value:,.0f} "
        f"({sign}${daily_pnl:,.0f} today, {sign}{daily_pct:.2f}%)",
        f"\U0001f4c8 Trades today: {trades_executed} executed, "
        f"{trades_failed} failed",
    ]
    if pos_lines:
        lines.append(f"\U0001f513 Open positions: {', '.join(pos_lines)}")
    if strong:
        lines.append(
            f"\U0001f52e Top signals for tomorrow: {', '.join(strong[:5])}"
        )

    try:
        tg._send("\n".join(lines))
    except Exception as exc:
        log.warning("Failed to send EOD summary: %s", exc)


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
        default=2,
        metavar="N",
        help="Max concurrent tickers (default: 2).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run sync vs async benchmark on 10 tickers.",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        default=False,
        help="Run as a daemon with 4 scheduled runs/day (weekdays, UTC).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.benchmark:
        _run_benchmark()
        return

    if args.daemon:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        scheduler = DailyScheduler()
        scheduler.run_forever()
        return  # unreachable, but explicit

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
