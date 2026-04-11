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
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from orchestrator.coordinator import Coordinator

log = logging.getLogger(__name__)

# Default watchlist — 18 core US tickers (synced with config/watchlist.yaml)
DEFAULT_WATCHLIST = [
    "META", "JPM", "AMZN", "XOM", "CVX",
    "BAC", "PFE", "TSLA", "NVDA", "COIN",
    "MSTR", "SMCI", "VRT", "AXON", "UNH",
    "COST", "BE", "NBIS",
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
    debate_semaphore: asyncio.Semaphore | None = None,
    session: str | None = None,
    session_type: str = "signal",
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
                debate_semaphore=debate_semaphore,
                session=session,
                session_type=session_type,
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
    session: str | None = None,
    session_type: str = "signal",
    macro_context: str = "",
) -> dict:
    """
    Analyse a list of tickers concurrently.

    Args:
        tickers:         List of ticker symbols.
        workers:         Max concurrent tickers.
        account_balance: Account size in USD.
        execute:         When True, execute trades via broker.
        session:         Trading session name (e.g. "US_OPEN").
        session_type:    "signal" | "execution" | "monitor".
        macro_context:   Session-level macro block from MacroContextAgent,
                         prepended to every bull/bear debate prompt. Empty
                         string when the feature is off.

    Returns:
        dict with keys: results, elapsed_s, success_count, fail_count.
    """
    api_semaphore = asyncio.Semaphore(5)
    data_semaphore = asyncio.Semaphore(10)
    debate_semaphore = asyncio.Semaphore(4)  # cap concurrent debates (rate limit safety)
    db_lock = asyncio.Lock()
    worker_semaphore = asyncio.Semaphore(workers)
    tracker = _ProgressTracker(len(tickers))

    coordinator = Coordinator(macro_context=macro_context)

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
            debate_semaphore=debate_semaphore,
            worker_semaphore=worker_semaphore,
            tracker=tracker,
            session=session,
            session_type=session_type,
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

# Session-specific watchlists (fallback — prefer YAML config).
# XETRA tickers are dropped from the live watchlist as of 2026-04-09; the
# scheduler entries below stay in place but resolve to an empty ticker list,
# making the XETRA sessions effective no-ops until tickers are re-added.
_XETRA_TICKERS: list[str] = []

# Session types:
#   "signal"    — full signal generation, store forward signals for next session
#   "pre_signal"— lightweight signal refresh (news + sentiment + debate, no TA)
#   "execution" — validate yesterday's EOD signals, execute if conditions hold
#   "monitor"   — lightweight position check only, no new signals
#
# Schedule: 7 runs per weekday, all times UTC
SCHEDULE = [
    {"name": "XETRA_PRE",     "hour": 6,  "minute": 45, "tickers": _XETRA_TICKERS, "workers": 2, "eod": False, "session_type": "pre_signal"},
    {"name": "XETRA_OPEN",    "hour": 7,  "minute": 0,  "tickers": _XETRA_TICKERS, "workers": 2, "eod": False, "session_type": "signal"},
    {"name": "PREMARKET_SCAN","hour": 13, "minute": 0,  "tickers": None,            "workers": 1, "eod": False, "session_type": "scanner"},
    {"name": "US_PRE",        "hour": 13, "minute": 15, "tickers": None,            "workers": 3, "eod": False, "session_type": "signal"},
    {"name": "PEAD_OPEN",     "hour": 13, "minute": 45, "tickers": None,            "workers": 3, "eod": False, "session_type": "signal"},
    {"name": "US_OPEN",       "hour": 14, "minute": 30, "tickers": None,            "workers": 3, "eod": False, "session_type": "execution"},
    {"name": "MIDDAY",        "hour": 18, "minute": 0,  "tickers": None,            "workers": 3, "eod": False, "session_type": "monitor"},
    {"name": "EOD",           "hour": 22, "minute": 15, "tickers": None,            "workers": 3, "eod": True,  "session_type": "signal"},
]

# ── Weekly jobs ──────────────────────────────────────────────────────────────
# Cron-style entries that fire on a specific weekday rather than every weekday.
# Entries here are dispatched by ``_execute_weekly_run`` (not _execute_run) and
# DO NOT participate in the ticker pipeline.
WEEKLY_JOBS: list[dict] = [
    {
        "name": "SECTOR_CORRELATION",
        "weekday": 6,   # Sunday (Mon=0 .. Sun=6)
        "hour": 6,
        "minute": 0,
        "kind": "weekly",
    },
]

# Trading window (UTC)
_WINDOW_START = (6, 45)   # 06:45 (XETRA_PRE)
_WINDOW_END   = (22, 30)  # 22:30


def _runner_id() -> str:
    """Stable identifier for *this* daemon process.

    Used in logs / Telegram so that two daemons firing the same session
    after a Railway rolling deploy can be told apart at a glance.
    """
    return (
        os.environ.get("RAILWAY_REPLICA_ID")
        or os.environ.get("RAILWAY_DEPLOYMENT_ID")
        or os.environ.get("RAILWAY_GIT_COMMIT_SHA", "")[:7]
        or os.environ.get("HOSTNAME")
        or f"pid-{os.getpid()}"
    )


def _is_execution_allowed() -> tuple[bool, str]:
    """
    Check whether trade execution is allowed based on environment.

    Returns (allowed, reason).

    Rules:
      1. EXECUTE_TRADES=false  → blocked (explicit toggle)
      2. Not on Railway AND EXECUTE_TRADES not explicitly "true"
         → blocked (prevents accidental local execution)
      3. Otherwise → allowed
    """
    env_val = os.environ.get("EXECUTE_TRADES", "").strip().lower()

    # Explicit disable
    if env_val == "false":
        return False, "EXECUTE_TRADES=false"

    # Local safety: block unless explicitly opted-in
    on_railway = bool(os.environ.get("RAILWAY_ENVIRONMENT"))
    if not on_railway and env_val != "true":
        return False, "not running on Railway and EXECUTE_TRADES != true"

    return True, "ok"


class DailyScheduler:
    """
    Daemon that sleeps between 7 daily trading runs (weekdays, UTC).

    Runs: XETRA_PRE (06:45), XETRA_OPEN (07:00), US_PRE (13:15),
          PEAD_OPEN (13:45), US_OPEN (14:30), MIDDAY (18:00), EOD (22:15).
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

    def _run_post_session_review(
        self,
        session: str,
        tickers: list[str],
        batch: dict,
    ) -> None:
        """Run PostSessionReviewer and persist / deliver the result.

        Fire-and-forget helper: the reviewer's own flag gating + fallback
        to "" handles the disabled case. On any unexpected failure we log
        and return — the session has already completed, nothing to roll
        back.

        Responsibilities split from the agent:
            1. Extract a lightweight signals list from the batch results
               (the agent doesn't touch the full Coordinator output).
            2. Call the agent (async; wrap in asyncio.run since we're
               on the sync _execute_run hot path).
            3. Send the text via Telegram (self._tg).
            4. Log the review to signal_events under strategy
               "PostSessionReviewer" so the text is retrievable later.
        """
        # 1. Build the signals list the agent expects: one dict per
        #    ticker with the fields the prompt / flags need.
        signals: list[dict] = []
        for r in (batch.get("results") or []):
            if not r:
                continue
            execution = r.get("execution") or {}
            signals.append({
                "ticker": r.get("ticker"),
                "signal": r.get("combined_signal") or r.get("signal"),
                "confidence": r.get("confidence"),
                "debate_outcome": (
                    (r.get("debate") or {}).get("summary")
                    if isinstance(r.get("debate"), dict)
                    else None
                ),
                "trade_executed": bool(execution.get("trade_id")),
            })

        # 2. Run the agent. The reviewer has its own ENABLE flag check —
        #    returns "" immediately if disabled or on any failure.
        try:
            from agents.post_session_reviewer import PostSessionReviewer
            review_text = asyncio.run(
                PostSessionReviewer().review(session, tickers, signals)
            )
        except Exception as exc:
            log.warning("PostSessionReviewer: agent call failed: %s", exc)
            review_text = ""

        if not review_text:
            return  # disabled or failed; nothing more to do

        # 3. Telegram
        if self._tg:
            try:
                self._tg._send(review_text)
            except Exception as exc:
                log.warning("PostSessionReviewer: Telegram send failed: %s", exc)

        # 4. signal_events row for later retrieval
        try:
            from analytics.signal_logger import SignalLogger
            SignalLogger().log({
                "session": session,
                "ticker": "SESSION",  # sentinel — not a real ticker
                "strategy": "PostSessionReviewer",
                "signal": "REVIEW",
                "bull_case": review_text,
            })
        except Exception as exc:
            log.warning("PostSessionReviewer: signal_events log failed: %s", exc)

    @staticmethod
    def _fetch_macro_context(session: str) -> str:
        """Run ``MacroContextAgent.get_context`` in a fresh event loop.

        This is called from the sync ``_execute_run`` path before
        ``asyncio.run(run_batch(...))``. MacroContextAgent itself handles
        the ENABLE_MACRO_CONTEXT feature flag, US-only session filter, and
        timeout — returning an empty string on any failure. We just route
        the call through a dedicated ``asyncio.run`` so the subsequent
        ``run_batch`` call still starts with a clean loop.
        """
        try:
            from agents.macro_context_agent import MacroContextAgent
            agent = MacroContextAgent()
            return asyncio.run(agent.get_context(session))
        except Exception as exc:
            log.warning("MacroContext fetch failed (non-fatal): %s", exc)
            return ""

    @staticmethod
    def _load_us_tickers_core() -> list[str]:
        """Load raw US core watchlist — no scanner layering.

        Used by the scanner itself (to avoid infinite recursion) and as the
        safety-floor fallback when the scanner is disabled or has no fresh
        output.
        """
        path = Path(__file__).resolve().parent.parent / "config" / "watchlist.yaml"
        try:
            with open(path) as fh:
                cfg = yaml.safe_load(fh) or {}
            us = cfg.get("us_tickers")
            if us:
                return us
        except Exception:
            pass
        # Fallback: full watchlist minus XETRA
        return [t for t in DEFAULT_WATCHLIST if not t.endswith(".XETRA")]

    @staticmethod
    def _load_us_tickers() -> list[str]:
        """Load US tickers for today's sessions.

        When ENABLE_PREMARKET_SCANNER=true and a fresh scanner_output.json
        exists, returns the scanner's expanded list (union with the core
        watchlist). Otherwise returns the raw core watchlist — preserving
        legacy behaviour.
        """
        core = DailyScheduler._load_us_tickers_core()
        try:
            from scripts.premarket_scanner import resolve_us_tickers
            return resolve_us_tickers(core)
        except Exception as exc:
            log.warning("Scanner resolve failed (falling back to core): %s", exc)
            return core

    @staticmethod
    def _load_xetra_tickers() -> list[str]:
        """Load XETRA tickers from watchlist.yaml (xetra_tickers key)."""
        path = Path(__file__).resolve().parent.parent / "config" / "watchlist.yaml"
        try:
            with open(path) as fh:
                cfg = yaml.safe_load(fh) or {}
            xetra = cfg.get("xetra_tickers")
            if xetra:
                return xetra
        except Exception:
            pass
        return _XETRA_TICKERS

    @staticmethod
    def _claim_session_slot(session_name: str) -> bool:
        """Atomically claim today's slot for *session_name*.

        Returns True if this caller is the first to run the session today
        (and should proceed), False if another runner already claimed it.

        Uses an ``INSERT OR IGNORE`` against a tiny ``session_runs`` table
        in the shared SQLite database. SQLite serialises writes with file
        locks, so two daemons sharing /data on Railway will see exactly
        one INSERT succeed — the loser's INSERT no-ops with rowcount 0.

        On any DB error this fails OPEN (returns True) so we never block
        a real run because of a transient connectivity issue.
        """
        try:
            from storage.database import _resolve_db_path
            db_path = _resolve_db_path()
        except Exception as exc:
            log.warning("Could not resolve DB path for slot claim: %s", exc)
            return True

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        started_at = datetime.now(timezone.utc).isoformat()
        runner_id = _runner_id()

        try:
            with sqlite3.connect(db_path, timeout=10) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS session_runs ("
                    " session TEXT NOT NULL,"
                    " run_date TEXT NOT NULL,"
                    " started_at TEXT NOT NULL,"
                    " runner_id TEXT,"
                    " PRIMARY KEY (session, run_date)"
                    ")"
                )
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO session_runs "
                    "(session, run_date, started_at, runner_id) VALUES (?, ?, ?, ?)",
                    (session_name, today, started_at, runner_id),
                )
                conn.commit()
                claimed = cursor.rowcount > 0
            if claimed:
                log.info(
                    "[%s] Claimed session slot for %s (runner=%s)",
                    session_name, today, runner_id,
                )
            else:
                log.info(
                    "[%s] Slot for %s already claimed (this runner=%s)",
                    session_name, today, runner_id,
                )
            return claimed
        except Exception as exc:
            log.warning(
                "[%s] Slot claim failed (%s) — failing open",
                session_name, exc,
            )
            return True

    @staticmethod
    def _build_telegram():
        try:
            from notifications.telegram_bot import TelegramNotifier
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if token and chat_id:
                return TelegramNotifier(bot_token=token, chat_id=chat_id)
            log.warning(
                "Telegram disabled — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set"
            )
        except Exception as exc:
            log.warning("Failed to build TelegramNotifier: %s", exc)
        return None

    # ── Helper methods ────────────────────────────────────────────────

    def next_run_time(self, after: datetime | None = None) -> datetime:
        """Return the next scheduled run time (UTC) strictly after *after*.

        Considers both the weekday SCHEDULE and any WEEKLY_JOBS (e.g. the
        Sunday sector-correlation refresh) and returns whichever fires first.
        """
        now = after or datetime.now(timezone.utc)
        candidates: list[datetime] = []

        # Look at the next 8 days — enough to cover any weekend skip + first
        # Sunday weekly job after a Friday EOD.
        for offset in range(8):
            day = (now + timedelta(days=offset)).replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            if day.weekday() < 5:
                for run in SCHEDULE:
                    t = day.replace(hour=run["hour"], minute=run["minute"])
                    if t > now:
                        candidates.append(t)
            for job in WEEKLY_JOBS:
                if day.weekday() == job["weekday"]:
                    t = day.replace(hour=job["hour"], minute=job["minute"])
                    if t > now:
                        candidates.append(t)

        return min(candidates)

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

        # Startup sanity: purge ghost positions AND trades with stale prices.
        # The $200 threshold catches test-fixture prices ($150) while leaving
        # all real trades intact (cheapest watchlist stock is well above $200).
        try:
            from scripts.clean_ghost_trades import clean_ghost_data, _resolve_db_path
            db_path = _resolve_db_path()
            result = clean_ghost_data(db_path, apply=True)
            n_pos = result["deleted_positions"]
            n_trades = result["deleted_trades"]
            if n_pos or n_trades:
                log.warning(
                    "[startup] Ghost cleanup: deleted %d positions, %d trades (price < $200)",
                    n_pos, n_trades,
                )
                print(f"[scheduler] Startup cleanup: removed {n_pos} ghost positions, "
                      f"{n_trades} ghost trades", flush=True)
                if self._tg:
                    try:
                        details = ", ".join(
                            f"{p['ticker']} @${p['avg_price']:.2f}"
                            for p in result["ghost_positions"]
                        )
                        self._tg._send(
                            f"\u26a0\ufe0f *Startup ghost cleanup*\n"
                            f"Deleted {n_pos} positions, {n_trades} trades (price < $200)\n"
                            f"Positions: {details or 'none'}"
                        )
                    except Exception:
                        pass
            else:
                log.info("[startup] Ghost cleanup: database clean")
        except Exception as exc:
            log.warning("Startup ghost cleanup failed: %s", exc)

        # Startup notification so we know the daemon is alive
        if self._tg:
            nrt = self.next_run_time()
            try:
                self._tg._send(
                    "\U0001f7e2 *News Trading Daemon started*\n"
                    f"Watchlist: {len(self._full_watchlist)} tickers\n"
                    f"Next run: {nrt.strftime('%Y-%m-%d %H:%M')} UTC"
                )
            except Exception as exc:
                log.warning("Telegram startup notification failed: %s", exc)

        # Start the intraday position manager (stop-loss / trailing-stop monitor)
        position_manager = None
        try:
            from execution.broker_factory import create_trader
            from monitoring.position_manager import PositionManager
            trader = create_trader()
            position_manager = PositionManager(
                trader=trader, notifier=self._tg,
            )
            position_manager.start()
            log.info("PositionManager background thread started")
            print("[scheduler] PositionManager started (60s interval)", flush=True)
        except Exception as exc:
            log.warning("PositionManager startup failed (non-fatal): %s", exc)

        # Run once immediately on startup if within trading hours
        last_executed: str | None = None
        session = self.current_session()
        if session != "CLOSED":
            startup_run = self._run_for_session(session)
            if startup_run:
                print(f"[scheduler] Immediate startup run: {startup_run['name']}",
                      flush=True)
                self._execute_run(startup_run)
                last_executed = startup_run["name"]

        try:
            while True:
                try:
                    nrt = self.next_run_time()
                    wait_s = max(0, int((nrt - datetime.now(timezone.utc)).total_seconds()))
                    run_info = self._run_for_time(nrt)
                    run_name = run_info["name"] if run_info else "UNKNOWN"

                    # Skip if this is the same session we just ran at startup
                    if run_info and run_info["name"] == last_executed and wait_s == 0:
                        log.info("Skipping %s — already executed at startup", run_name)
                        print(f"[scheduler] Skipping {run_name} — already ran at startup",
                              flush=True)
                        last_executed = None  # only skip once
                        continue

                    print(f"[scheduler] Next: {run_name} at "
                          f"{nrt.strftime('%Y-%m-%d %H:%M')} UTC ({wait_s}s away)",
                          flush=True)
                    log.info("Sleeping %ds until %s at %s", wait_s, run_name,
                             nrt.strftime("%H:%M"))

                    time.sleep(wait_s)

                    if run_info:
                        if run_info.get("kind") == "weekly":
                            self._execute_weekly_run(run_info)
                        else:
                            self._execute_run(run_info)
                        last_executed = run_info["name"]

                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    log.error("Unhandled exception in scheduler loop: %s", exc,
                              exc_info=True)
                    print(f"[scheduler] LOOP ERROR (restarting in 60s): {exc}", flush=True)
                    if self._tg:
                        try:
                            self._tg._send(
                                f"\U0001f6a8 *Scheduler loop crashed:*\n"
                                f"{str(exc)[:300]}\nRestarting in 60s\u2026"
                            )
                        except Exception:
                            pass
                    time.sleep(60)
        finally:
            if position_manager is not None:
                position_manager.stop()
                log.info("PositionManager stopped on shutdown")

    # ── Execution ─────────────────────────────────────────────────────

    def _run_for_time(self, dt: datetime) -> dict | None:
        # Daily runs only fire on weekdays
        if dt.weekday() < 5:
            for run in SCHEDULE:
                if dt.hour == run["hour"] and dt.minute == run["minute"]:
                    return run
        # Weekly jobs are weekday-specific
        for job in WEEKLY_JOBS:
            if (
                dt.weekday() == job["weekday"]
                and dt.hour == job["hour"]
                and dt.minute == job["minute"]
            ):
                return job
        return None

    def _run_for_session(self, session_name: str) -> dict | None:
        """Return the schedule entry matching the given session name."""
        for run in SCHEDULE:
            if run["name"] == session_name:
                return run
        return None

    def _execute_run(self, run: dict) -> None:
        run_name = run["name"]
        session_type = run.get("session_type", "signal")

        # Skip PRE sessions when disabled
        if session_type == "pre_signal":
            from config.settings import ENABLE_PRE_SESSIONS
            if not ENABLE_PRE_SESSIONS:
                log.info("Skipping %s — ENABLE_PRE_SESSIONS=false", run_name)
                print(f"[scheduler] Skipping {run_name} — PRE sessions disabled",
                      flush=True)
                return

        # Idempotency: prevent the same session from firing twice on the
        # same UTC day. This protects against deployment overlap (Railway
        # rolling deploys leave the old container alive briefly, so both
        # the old and new daemon would otherwise fire US_OPEN at 14:30,
        # each loading whichever watchlist.yaml is baked into its image).
        # The claim is atomic via SQLite INSERT OR IGNORE so any second
        # caller — same process or different container sharing /data —
        # sees the row already exists and bails out cleanly.
        if not self._claim_session_slot(run_name):
            log.warning(
                "[%s] Skipping — session already claimed today by another runner",
                run_name,
            )
            print(
                f"[scheduler] Skipping {run_name} — already ran today "
                f"(duplicate runner / deploy overlap)",
                flush=True,
            )
            # Only send Telegram for non-XETRA duplicates (XETRA overlaps
            # are common during deploys and just create noise).
            if self._tg and run_name not in ("XETRA_OPEN", "XETRA_PRE"):
                try:
                    self._tg._send(
                        f"\u26a0\ufe0f Skipped duplicate *{run_name}* run "
                        f"(another daemon already executed it today)."
                    )
                except Exception:
                    pass
            return

        # D9 pre-market scanner — no ticker batch, just builds today's
        # scanner_output.json. Subsequent US_PRE/US_OPEN sessions read it via
        # _load_us_tickers() when ENABLE_PREMARKET_SCANNER=true.
        if session_type == "scanner":
            try:
                from scripts.premarket_scanner import main as run_scanner_main
                out = run_scanner_main(core_watchlist=self._load_us_tickers_core())
                stats = out.get("stats", {})
                log.info(
                    "[%s] scanner complete — final=%d (universe=%d, liquid=%d, "
                    "earnings=%d, momentum=%d, sentiment=%d)",
                    run_name, stats.get("final_count", 0),
                    stats.get("universe_size", 0), stats.get("post_liquidity", 0),
                    stats.get("earnings_flagged", 0), stats.get("momentum_flagged", 0),
                    stats.get("sentiment_flagged", 0),
                )
                if self._tg:
                    try:
                        self._tg._send(
                            f"\U0001f50d *{run_name}* complete — {stats.get('final_count', 0)} "
                            f"tickers picked (universe={stats.get('universe_size', 0)}, "
                            f"earnings={stats.get('earnings_flagged', 0)}, "
                            f"momentum={stats.get('momentum_flagged', 0)}, "
                            f"sentiment={stats.get('sentiment_flagged', 0)})"
                        )
                    except Exception as exc:
                        log.warning("Telegram scanner summary failed: %s", exc)
            except Exception as exc:
                log.warning("[%s] scanner failed (non-fatal, falls back to core watchlist): %s",
                            run_name, exc)
                print(f"[scheduler] {run_name} ERROR (fallback to core): {exc}",
                      flush=True)
            return

        # Resolve ticker list based on session
        if run_name in ("XETRA_OPEN", "XETRA_PRE"):
            tickers = run["tickers"] or self._load_xetra_tickers()
            if not tickers:
                log.info("Skipping %s — no XETRA tickers configured", run_name)
                print(f"[scheduler] Skipping {run_name} — no XETRA tickers",
                      flush=True)
                return
        elif run_name == "PEAD_OPEN":
            from config.settings import PEAD_ENABLED, PEAD_TICKERS
            if not PEAD_ENABLED:
                log.info("Skipping %s — PEAD_ENABLED=false", run_name)
                print(f"[scheduler] Skipping {run_name} — PEAD disabled", flush=True)
                return
            tickers = PEAD_TICKERS
        else:
            tickers = run["tickers"] or self._load_us_tickers()
            # Safety belt: also strip any XETRA tickers that snuck in
            tickers = [t for t in tickers if not t.endswith(".XETRA")]
            log.info("Excluded XETRA tickers from %s session", run["name"])

        workers = run["workers"]
        now_str = datetime.now(timezone.utc).strftime("%H:%M")
        runner_id = _runner_id()

        # Fetch session-level macro context BEFORE the Telegram start
        # message so it can be included in the preview. The agent returns
        # "" when ENABLE_MACRO_CONTEXT=false, the session is non-US, or
        # anything goes wrong — so this call is always safe to make.
        macro_context = self._fetch_macro_context(run_name)
        macro_preview = ""
        if macro_context:
            # Collapse whitespace so Telegram doesn't render multiple lines
            flat = " ".join(macro_context.split())
            macro_preview = flat[:100] + ("…" if len(flat) > 100 else "")

        # Telegram: starting. Include runner_id + ticker count + first three
        # tickers so a stale daemon firing an old watchlist is obvious in
        # real-time (two messages with two different runner ids = deploy
        # overlap, two messages with different ticker counts = stale yaml).
        # MIDDAY ("monitor") session is suppressed — too noisy.
        if self._tg and session_type != "monitor":
            try:
                preview = ", ".join(tickers[:3])
                if len(tickers) > 3:
                    preview += f", \u2026 (+{len(tickers) - 3})"
                message = (
                    f"\U0001f558 *{run_name}* ({session_type}) starting \u2014 {now_str} UTC\n"
                    f"Runner: `{runner_id}`\n"
                    f"Tickers ({len(tickers)}): {preview}"
                )
                if macro_preview:
                    message += f"\nMacro: {macro_preview}"
                self._tg._send(message)
            except Exception as exc:
                log.warning("Telegram session-start notification failed: %s", exc)

        log.info("Starting %s [%s]: %d tickers, %d workers (runner=%s)",
                 run_name, session_type, len(tickers), workers, runner_id)
        print(f"[scheduler] Running {run_name} [{session_type}] (runner={runner_id}): "
              f"{', '.join(tickers)} ({workers} workers)", flush=True)

        execute, reason = _is_execution_allowed()
        if not execute:
            log.warning("Trade execution DISABLED: %s", reason)
            print(f"[scheduler] Execution disabled ({reason}) — analysis only",
                  flush=True)

        # Expire stale forward signals before execution sessions
        if session_type == "execution":
            try:
                from analytics.signal_logger import SignalLogger
                expired = SignalLogger().expire_stale_forward_signals()
                if expired:
                    log.info("Expired %d stale forward signals", expired)
            except Exception as exc:
                log.warning("Forward signal expiry failed (non-fatal): %s", exc)

        try:
            batch = asyncio.run(
                run_batch(
                    tickers,
                    workers=workers,
                    account_balance=10_000.0,
                    execute=execute,
                    session=run_name,
                    session_type=session_type,
                    macro_context=macro_context,
                )
            )

            ok = batch["success_count"]
            total = len(tickers)
            elapsed = batch["elapsed_s"]
            log.info("%s complete: %d/%d in %.1fs", run_name, ok, total, elapsed)
            print(f"[scheduler] {run_name} done: {ok}/{total} in "
                  f"{elapsed:.1f}s", flush=True)

            # Telegram: run summary (suppress MIDDAY — too noisy)
            if self._tg and session_type != "monitor":
                try:
                    self._send_run_summary(run_name, batch, tickers)
                except Exception as exc:
                    log.warning("Telegram run-summary notification failed: %s", exc)

            # EOD P&L summary
            if run["eod"]:
                try:
                    send_eod_summary(self._tg, batch, tickers)
                except Exception as exc:
                    log.warning("EOD summary failed: %s", exc)

                # Backfill signal outcomes (3d/5d/10d price changes)
                try:
                    from analytics.outcome_tracker import run_outcome_tracker
                    outcome_result = run_outcome_tracker()
                    total_updated = sum(outcome_result.values())
                    if total_updated:
                        log.info("EOD outcome backfill: %d rows updated %s",
                                 total_updated, outcome_result)
                except Exception as exc:
                    log.warning("EOD outcome tracker failed (non-fatal): %s", exc)

                # Nightly per-strategy performance tracker
                try:
                    from analytics.strategy_performance import StrategyPerformanceTracker
                    perf = StrategyPerformanceTracker()
                    metrics = perf.compute()
                    parts = []
                    for name, m in sorted(metrics.items()):
                        sr = f"SR={m.sharpe_30d:.2f}" if m.sharpe_30d is not None else "SR=N/A"
                        wr = f"WR={m.win_rate_30d:.0%}" if m.win_rate_30d is not None else "WR=N/A"
                        parts.append(f"{name} {sr}, {wr}")
                    log.info("Strategy performance: %s", " | ".join(parts))
                    print(f"[scheduler] Strategy performance: {' | '.join(parts)}",
                          flush=True)
                except Exception as exc:
                    log.warning("Strategy performance tracker failed (non-fatal): %s", exc)

            # Post-session review (EOD always, US_OPEN only when a trade
            # was actually executed during the session). Fire-and-forget —
            # the reviewer has its own feature flag and never blocks if
            # disabled or failing.
            try:
                any_trade = any(
                    ((r or {}).get("execution") or {}).get("trade_id")
                    for r in (batch.get("results") or [])
                )
                should_review = run["eod"] or (
                    run_name == "US_OPEN" and any_trade
                )
                if should_review:
                    self._run_post_session_review(run_name, tickers, batch)
            except Exception as exc:
                log.warning("PostSessionReviewer dispatch failed (non-fatal): %s", exc)

        except Exception as exc:
            log.error("Scheduler error in %s: %s", run_name, exc)
            print(f"[scheduler] ERROR in {run_name}: {exc}", flush=True)
            if self._tg:
                try:
                    self._tg._send(
                        f"\U0001f6a8 *Scheduler error in {run_name}:*\n"
                        f"{str(exc)[:300]}"
                    )
                except Exception as tg_exc:
                    log.warning("Telegram error notification failed: %s", tg_exc)

    # ── Weekly job dispatch ───────────────────────────────────────────

    def _execute_weekly_run(self, job: dict) -> None:
        """Run a weekly maintenance job (e.g. sector correlation refresh)."""
        name = job["name"]
        log.info("Starting weekly job %s", name)
        print(f"[scheduler] Running weekly job {name}", flush=True)

        if name == "SECTOR_CORRELATION":
            try:
                from scripts.sector_correlation import run as run_sector_correlation
                summary = run_sector_correlation()
                log.info("%s complete: %s", name, summary)
                print(f"[scheduler] {name} done: {summary}", flush=True)
                if self._tg:
                    try:
                        self._tg._send(
                            f"\U0001f4ca *Sector correlation refresh*\n"
                            f"Universe: {summary.get('universe_size')} tickers\n"
                            f"Static refreshed: {summary.get('static_refreshed')}\n"
                            f"Matrix refreshed: {summary.get('matrix_refreshed')}"
                        )
                    except Exception as exc:
                        log.warning("Telegram weekly notification failed: %s", exc)
            except Exception as exc:
                log.error("Weekly job %s failed: %s", name, exc, exc_info=True)
                print(f"[scheduler] ERROR in weekly job {name}: {exc}", flush=True)
                if self._tg:
                    try:
                        self._tg._send(
                            f"\U0001f6a8 *Weekly job {name} crashed:*\n{str(exc)[:300]}"
                        )
                    except Exception:
                        pass
        else:
            log.warning("Unknown weekly job: %s", name)

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
            if (r.get("execution") or {}).get("trade_id"):
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

        if self._tg:
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
        if r and (r.get("execution") or {}).get("trade_id")
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
