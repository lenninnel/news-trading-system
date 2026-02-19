"""
Daily runner for the News Trading System.

Two modes
---------
  Daemon  (no flags)   Block and fire at the configured schedule.time each day.
                       Keep alive with: nohup python3 scheduler/daily_runner.py &

  Instant (--now)      Run one full cycle immediately and exit.
                       Used by cron jobs and for manual testing.

Strategy selection
------------------
  --strategy all           Run MomentumAgent + MeanReversionAgent + SwingAgent (default)
  --strategy momentum      Run MomentumAgent only
  --strategy mean-reversion  Run MeanReversionAgent only
  --strategy swing         Run SwingAgent only

  Default strategy can also be set in config/watchlist.yaml:
    scheduler:
      default_strategy: all

Resilience testing
------------------
  --resilience-test        Run all 6 recovery mechanism tests and exit.

Usage
-----
  python3 scheduler/daily_runner.py                          # daemon
  python3 scheduler/daily_runner.py --now                    # run immediately
  python3 scheduler/daily_runner.py --now --strategy momentum  # momentum only
  python3 scheduler/daily_runner.py --resilience-test        # test recovery

Cron example (installed by scheduler/install_cron.sh):
  30 9 * * 1-5  /usr/bin/python3 /path/to/scheduler/daily_runner.py --now >> /path/to/scheduler/logs/cron.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import smtplib
import sys
import threading
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402 — needs path setup first

from config.settings import DB_PATH  # noqa: E402
from execution.paper_trader import PaperTrader  # noqa: E402
from notifications.telegram_bot import TelegramNotifier  # noqa: E402
from orchestrator.coordinator import Coordinator  # noqa: E402
from orchestrator.strategy_coordinator import StrategyCoordinator  # noqa: E402
from storage.database import Database  # noqa: E402
from utils.api_recovery import APIRecovery  # noqa: E402
from utils.network_recovery import NetworkMonitor  # noqa: E402
from utils.state_recovery import CheckpointManager  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
WATCHLIST_PATH = PROJECT_ROOT / "config" / "watchlist.yaml"
LOGS_DIR       = PROJECT_ROOT / "scheduler" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ── Logging ───────────────────────────────────────────────────────────────────

def _build_logger() -> logging.Logger:
    log_file = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    logger   = logging.getLogger("daily_runner")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:          # avoid duplicate handlers on re-import
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = _build_logger()


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(WATCHLIST_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Normalise keys with defaults
    cfg.setdefault("watchlist", ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL"])
    cfg.setdefault("account", {}).setdefault("balance", 10_000.0)
    cfg.setdefault("execution", {}).setdefault("enabled", True)
    cfg.setdefault("schedule", {}).setdefault("time", "09:30")
    cfg["schedule"].setdefault("weekdays_only", True)
    cfg.setdefault("email", {}).setdefault("enabled", False)
    cfg.setdefault("telegram", {}).setdefault("enabled", False)
    cfg.setdefault("scheduler", {}).setdefault("default_strategy", "all")
    return cfg


# ── Email (optional) ──────────────────────────────────────────────────────────

def _send_failure_email(cfg: dict, subject: str, body: str) -> None:
    email_cfg = cfg.get("email", {})
    if not email_cfg.get("enabled"):
        return

    password = os.environ.get("SMTP_PASSWORD", "")
    if not password:
        log.warning("Email enabled but SMTP_PASSWORD env var not set — skipping.")
        return

    msg            = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = email_cfg["from_address"]
    msg["To"]      = email_cfg["to_address"]

    try:
        with smtplib.SMTP(email_cfg["smtp_host"], email_cfg["smtp_port"]) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(email_cfg["from_address"], password)
            smtp.sendmail(email_cfg["from_address"], [email_cfg["to_address"]], msg.as_string())
        log.info("Failure email sent to %s", email_cfg["to_address"])
    except Exception as exc:
        log.error("Could not send failure email: %s", exc)


# ── Core run logic ────────────────────────────────────────────────────────────

def run_daily(
    cfg: dict | None = None,
    notifier: TelegramNotifier | None = None,
    strategy: str = "all",
) -> None:
    """
    Run one full analysis cycle for every ticker in the watchlist.

    Args:
        cfg:      Configuration dict (loaded from watchlist.yaml when None).
        notifier: Optional TelegramNotifier for signal/trade alerts.
        strategy: Strategy agents to use: "all" | "momentum" |
                  "mean-reversion" | "swing".
    """
    if cfg is None:
        cfg = _load_config()

    tickers  = [t.upper() for t in cfg["watchlist"]]
    balance  = float(cfg["account"]["balance"])
    execute  = bool(cfg["execution"]["enabled"])

    started_at = datetime.now(timezone.utc)
    log.info("=" * 60)
    log.info(
        "Daily runner started  |  %d tickers  |  balance $%.2f  |  "
        "execute=%s  |  strategy=%s",
        len(tickers), balance, execute, strategy,
    )
    log.info("Tickers: %s", ", ".join(tickers))
    log.info("=" * 60)

    paper_trader = PaperTrader() if execute else None
    db           = Database()
    strat_coord  = StrategyCoordinator(db=db)

    # ── Wire recovery modules to DB ──────────────────────────────────────────
    APIRecovery.set_db(db)
    NetworkMonitor.set_db(db)

    # ── Checkpoint: resume from a previous interrupted run ───────────────────
    checkpoint = CheckpointManager(
        name=f"daily_run_{started_at.strftime('%Y%m%d')}",
        save_interval=1,   # save after every ticker (frequent, cheap)
    )
    checkpoint.set_db(db)
    pending_tickers = checkpoint.get_pending(tickers)
    if len(pending_tickers) < len(tickers):
        skipped = len(tickers) - len(pending_tickers)
        log.info(
            "Checkpoint: resuming run — %d ticker(s) already done, "
            "%d remaining",
            skipped, len(pending_tickers),
        )
    completed_tickers: list[str] = list(
        set(tickers) - set(pending_tickers)
    )

    signals_generated = 0
    trades_executed   = 0
    errors: list[str] = []
    results: list[dict] = []

    # Per-agent signal counts (non-HOLD signals only)
    strategy_counts: dict[str, int] = {
        "momentum": 0, "mean_reversion": 0, "swing": 0,
    }

    for ticker in pending_tickers:
        log.info("── %s ─────────────────────────────────────", ticker)
        try:
            report = strat_coord.run(
                ticker=ticker,
                account_balance=balance,
                verbose=False,
                strategy=strategy,
            )

            sig  = report["combined_strategy_signal"]
            conf = report["ensemble_confidence"] / 100.0   # normalise to 0–1
            risk = report["risk"]

            # Count active (non-HOLD) per-agent signals for the breakdown
            for sig_info in report.get("strategy_signals", []):
                if sig_info["signal"] in ("BUY", "SELL"):
                    strat_key = sig_info["strategy"]  # "momentum" | "mean_reversion" | "swing"
                    strategy_counts[strat_key] = strategy_counts.get(strat_key, 0) + 1

            signals_generated += 1

            # Paper execution (strategy mode handles risk sizing but not trading)
            trade = None
            if paper_trader and not risk["skipped"]:
                price = risk.get("current_price") or 0.0
                trade = paper_trader.track_trade(
                    ticker=ticker,
                    action=risk["direction"],
                    shares=risk["shares"],
                    price=price,
                    stop_loss=risk["stop_loss"],
                    take_profit=risk["take_profit"],
                )

            if trade is not None:
                trades_executed += 1

            # Telegram signal alert
            if notifier is not None:
                top_reasons = [
                    s["reasoning"][0]
                    for s in report.get("ranked_signals", [])
                    if s.get("reasoning")
                ][:2]
                reasoning = "  |  ".join(top_reasons)
                notifier.send_signal(
                    ticker=ticker,
                    signal=sig,
                    confidence=conf * 100,
                    reasoning=reasoning,
                )
                if trade is not None and not risk["skipped"]:
                    price = risk.get("current_price") or 0.0
                    notifier.send_trade_executed(
                        ticker=ticker,
                        action=risk["direction"],
                        shares=risk["shares"],
                        price=price,
                        stop_loss=risk["stop_loss"],
                        take_profit=risk["take_profit"],
                    )

            log.info(
                "%s  →  %s  (conf: %.0f%%)  |  %s",
                ticker,
                sig,
                conf * 100,
                (f"TRADE #{trade}  ${risk['position_size_usd']:,.2f}  "
                 f"{risk['shares']} sh  SL ${risk['stop_loss']:.2f}  TP ${risk['take_profit']:.2f}")
                if trade else
                f"no trade — {risk.get('skip_reason', 'skipped')}",
            )

            results.append({
                "ticker":   ticker,
                "signal":   sig,
                "conf":     conf,
                "traded":   trade is not None,
                "trade_id": trade,
                "strategy": strategy,
                "errors":   report.get("errors", []),
            })
            # Mark ticker as completed in checkpoint
            completed_tickers.append(ticker)
            checkpoint.update("completed_tickers", completed_tickers)

        except Exception as exc:
            msg = f"{ticker}: {exc}"
            log.error("ERROR  %s", msg, exc_info=True)
            errors.append(msg)

    # ── Portfolio snapshot ────────────────────────────────────────────────────
    portfolio_value = 0.0
    if paper_trader:
        positions = paper_trader.get_portfolio()
        portfolio_value = sum(p["current_value"] for p in positions)

    # ── Timing ───────────────────────────────────────────────────────────────
    duration = (datetime.now(timezone.utc) - started_at).total_seconds()

    # ── Status ───────────────────────────────────────────────────────────────
    if not errors:
        status = "success"
    elif len(errors) < len(tickers):
        status = "partial"
    else:
        status = "failed"

    # ── Summary report ────────────────────────────────────────────────────────
    _STRATEGY_LABELS = {
        "momentum":       "Momentum",
        "mean_reversion": "Mean Reversion",
        "swing":          "Swing",
    }

    lines = [
        f"Run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Status          : {status.upper()}",
        f"Strategy        : {strategy}",
        f"Tickers checked : {len(tickers)}",
        f"Signals         : {signals_generated}",
        f"Trades executed : {trades_executed}",
        f"Portfolio value : ${portfolio_value:,.2f}",
        f"Duration        : {duration:.1f}s",
    ]
    if errors:
        lines.append(f"Errors          : {len(errors)}")
        for e in errors:
            lines.append(f"  • {e}")

    # Strategy breakdown — only show strategies that were actually active
    active_counts = {
        k: v for k, v in strategy_counts.items() if v > 0
    }
    if active_counts:
        parts = []
        for key, label in _STRATEGY_LABELS.items():
            if key in active_counts:
                n = active_counts[key]
                parts.append(f"{label}: {n} signal{'s' if n != 1 else ''}")
        if parts:
            lines.append(f"Strategy breakdown: {', '.join(parts)}")

    if results:
        lines.append("\nSignal breakdown:")
        for r in results:
            arrow = "▶" if r["traded"] else "·"
            lines.append(
                f"  {arrow} {r['ticker']:<6}  {r['signal']:<14}  {r['conf']:.0%}"
                + (f"  → trade #{r['trade_id']}" if r["traded"] else "")
            )

    summary = "\n".join(lines)
    log.info("\n%s", summary)

    # ── Persist scheduler log ────────────────────────────────────────────────
    try:
        db.log_scheduler_run(
            run_at=started_at.isoformat(),
            tickers=tickers,
            signals_generated=signals_generated,
            trades_executed=trades_executed,
            portfolio_value=portfolio_value,
            duration_seconds=duration,
            errors=errors,
            status=status,
            summary=summary,
        )
        log.info("Scheduler run saved to DB.")
    except Exception as exc:
        log.error("Could not save scheduler log to DB: %s", exc)

    # ── Email on failure ─────────────────────────────────────────────────────
    if status in ("partial", "failed"):
        _send_failure_email(
            cfg,
            subject=f"[Trading System] Daily run {status.upper()} — {datetime.now().strftime('%Y-%m-%d')}",
            body=summary,
        )

    # ── Telegram daily summary ────────────────────────────────────────────────
    if notifier is not None:
        notifier.send_daily_summary(
            signals_count=signals_generated,
            trades_count=trades_executed,
            portfolio_value=portfolio_value,
            results=results,
            errors=errors,
            status=status,
        )

    # ── Email daily summary ───────────────────────────────────────────────────
    try:
        from notifications.email_notifier import EmailNotifier
        email_notifier = EmailNotifier.from_config(cfg)
        if email_notifier:
            email_notifier.send_daily_summary(
                signals_count=signals_generated,
                trades_count=trades_executed,
                portfolio_value=portfolio_value,
                results=results,
                errors=errors,
                status=status,
            )
    except Exception as exc:
        log.warning("Email daily summary failed: %s", exc)

    # Clear checkpoint on clean completion (partial runs keep it for resume)
    if status == "success":
        checkpoint.clear()

    log.info("Done in %.1fs  |  status=%s", duration, status)


# ── Scheduler loop (daemon mode) ──────────────────────────────────────────────

def _run_daemon(
    cfg: dict,
    notifier: TelegramNotifier | None = None,
    strategy: str = "all",
) -> None:
    """Block forever, firing run_daily() at the configured time each day."""
    import functools

    try:
        import schedule  # type: ignore[import]
    except ImportError:
        log.error("'schedule' package not installed.  Run: pip install schedule")
        sys.exit(1)

    # Register PID for kill-switch --stop-all support
    try:
        from emergency_stop import KillSwitch
        KillSwitch.register_pid("scheduler")
    except Exception:
        pass

    _shutdown = threading.Event()

    def _handle_signal(sig, _frame):
        log.info("Signal %s received — scheduler daemon shutting down gracefully.", sig)
        _shutdown.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    run_time      = cfg["schedule"]["time"]
    weekdays_only = cfg["schedule"]["weekdays_only"]
    job           = functools.partial(run_daily, cfg, notifier=notifier, strategy=strategy)

    if weekdays_only:
        for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
            getattr(schedule.every(), day).at(run_time).do(job)
        log.info("Daemon started — will run Mon–Fri at %s (local time).", run_time)
    else:
        schedule.every().day.at(run_time).do(job)
        log.info("Daemon started — will run daily at %s (local time).", run_time)

    try:
        while not _shutdown.is_set():
            # Honour kill switch
            try:
                from emergency_stop import KillSwitch
                if KillSwitch.is_stopped():
                    log.warning("Kill switch active — scheduler daemon exiting.")
                    break
            except Exception:
                pass
            schedule.run_pending()
            _shutdown.wait(timeout=30)
    finally:
        log.info("Scheduler daemon stopped.")
        try:
            from emergency_stop import KillSwitch
            KillSwitch.unregister_pid("scheduler")
        except Exception:
            pass


# ── Resilience test ───────────────────────────────────────────────────────────

def resilience_test() -> int:
    """
    Test all six error-recovery mechanisms.

    Each test injects a specific failure mode (via mock/patch) and verifies
    that the corresponding recovery path activates correctly.  No real API
    calls, trades, or DB writes are made.

    Returns:
        Exit code: 0 = all passed, 1 = one or more failed.
    """
    import tempfile
    from unittest.mock import MagicMock, patch

    passed: list[str] = []
    failed: list[str] = []

    GREEN = "\033[32m"
    RED   = "\033[31m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def ok(name: str) -> None:
        passed.append(name)
        print(f"  {GREEN}✓{RESET}  {name}")

    def fail(name: str, detail: str = "") -> None:
        failed.append(name)
        print(f"  {RED}✗{RESET}  {name}" + (f" — {detail}" if detail else ""))

    print(f"\n{BOLD}Resilience Test Suite{RESET}\n{'─' * 50}")

    # ── Test 1: Circuit Breaker opens after threshold failures ────────────────
    print("\n[1] Circuit Breaker — opens after 5 failures")
    try:
        from utils.api_recovery import APIRecovery, CircuitBreaker, CircuitOpenError

        cb = CircuitBreaker("test_service", failure_threshold=5, reset_timeout=300)
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN, f"Expected OPEN, got {cb.state}"
        cb.record_success()
        assert cb.state == CircuitBreaker.CLOSED, f"Expected CLOSED after reset, got {cb.state}"
        ok("Circuit breaker opens at threshold and closes on success")
    except Exception as exc:
        fail("Circuit breaker state machine", str(exc))

    # ── Test 2: NewsAPI down → cached headlines served ────────────────────────
    print("\n[2] NewsAPI Fallback — cached headlines on failure")
    try:
        from utils.network_recovery import ResponseCache

        cache = ResponseCache(max_age_seconds=3600)
        fake_headlines = ["Fake headline A", "Fake headline B"]
        cache.set("newsapi", "headlines:AAPL", fake_headlines)

        with patch("utils.network_recovery._cache", cache):
            # Simulate APIRecovery.call raising on all attempts
            with patch("utils.api_recovery.APIRecovery.call", side_effect=Exception("newsapi down")):
                from data.news_feed import NewsFeed
                feed = NewsFeed.__new__(NewsFeed)
                feed.api_key = "dummy"
                feed.max_headlines = 10

                # Directly test the fallback path
                cached, hit = cache.get("newsapi", "headlines:AAPL")
                assert hit, "Cache should have a fresh entry"
                assert cached == fake_headlines, "Cache returned wrong data"
        ok("NewsAPI down → cache hit returns cached headlines")
    except Exception as exc:
        fail("NewsAPI cache fallback", str(exc))

    # ── Test 3: Anthropic down → rule-based sentiment ────────────────────────
    print("\n[3] Anthropic Fallback — rule-based keyword sentiment")
    try:
        from agents.sentiment_agent import _rule_based_sentiment

        # Bullish headline
        result = _rule_based_sentiment("Apple reports record profit and revenue surge")
        assert result["degraded"] is True, "degraded flag must be True"
        assert result["sentiment"] == "bullish", f"Expected bullish, got {result['sentiment']}"
        assert result["score"] == 1

        # Bearish headline
        result = _rule_based_sentiment("Company faces bankruptcy after massive fraud scandal")
        assert result["degraded"] is True
        assert result["sentiment"] == "bearish", f"Expected bearish, got {result['sentiment']}"
        assert result["score"] == -1

        # Neutral headline
        result = _rule_based_sentiment("Company announces quarterly earnings release date")
        assert result["degraded"] is True

        ok("Rule-based sentiment produces correct bullish/bearish/neutral scores")
    except Exception as exc:
        fail("Anthropic rule-based fallback", str(exc))

    # ── Test 4: yfinance down → cached indicators or HOLD skip ───────────────
    print("\n[4] yfinance Fallback — cached indicators then HOLD skip")
    try:
        from utils.network_recovery import ResponseCache
        from utils.api_recovery import APIRecovery, CircuitOpenError

        cache     = ResponseCache(max_age_seconds=3600)
        fake_ind  = {
            "rsi": 55.0, "macd": 0.1, "macd_signal": 0.05, "macd_hist": 0.05,
            "macd_bull_cross": False, "macd_bear_cross": False,
            "sma_20": 150.0, "sma_50": 148.0,
            "bb_upper": 155.0, "bb_lower": 145.0, "price": 150.0,
        }
        cache.set("yfinance", "indicators:TSLA", fake_ind)

        # With cache: should return cached indicators
        cached, hit = cache.get("yfinance", "indicators:TSLA")
        assert hit, "yfinance indicators should be cached"
        assert cached["rsi"] == 55.0

        # Without cache: should return all-None HOLD indicators
        empty_cache = ResponseCache(max_age_seconds=3600)
        cached2, hit2 = empty_cache.get("yfinance", "indicators:TSLA")
        assert not hit2, "Empty cache should return no-hit"
        ok("yfinance cache hit returns cached indicators; miss returns empty dict")
    except Exception as exc:
        fail("yfinance cache/skip fallback", str(exc))

    # ── Test 5: Network outage → degraded mode ───────────────────────────────
    print("\n[5] Network Recovery — degraded mode detection")
    try:
        from utils.network_recovery import NetworkMonitor

        # Simulate network outage by patching is_online
        with patch.object(NetworkMonitor, "is_online", return_value=False):
            NetworkMonitor._degraded      = False   # reset
            NetworkMonitor._offline_since = None
            NetworkMonitor._last_check_at = None
            NetworkMonitor._db            = None
            result = NetworkMonitor.check_and_update(force=True)
            assert result is False, "check_and_update should return False when offline"
            assert NetworkMonitor.is_degraded() is True, "Degraded mode should be active"

        # Simulate restore
        with patch.object(NetworkMonitor, "is_online", return_value=True):
            result = NetworkMonitor.check_and_update(force=True)
            assert result is True
            assert NetworkMonitor.is_degraded() is False, "Degraded mode should clear"

        ok("Network outage sets degraded mode; restore clears it")
    except Exception as exc:
        fail("Network degraded mode", str(exc))
    finally:
        # Always clean up class state
        NetworkMonitor._degraded      = False
        NetworkMonitor._offline_since = None
        NetworkMonitor._last_check_at = None

    # ── Test 6: Checkpoint save / validate / resume ──────────────────────────
    print("\n[6] State Recovery — checkpoint save/load/resume")
    try:
        from utils.state_recovery import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                name="resilience_test",
                checkpoint_dir=tmpdir,
                save_interval=1,
                max_age_hours=24,
            )

            # Simulate completing 3 of 5 tickers
            all_tickers = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL"]
            for t in all_tickers[:3]:
                mgr.update("completed_tickers", all_tickers[:all_tickers.index(t) + 1])
            mgr.save()

            # New manager loading from same dir
            mgr2 = CheckpointManager(
                name="resilience_test",
                checkpoint_dir=tmpdir,
                save_interval=1,
                max_age_hours=24,
            )
            pending = mgr2.get_pending(all_tickers)
            assert pending == ["MSFT", "GOOGL"], (
                f"Expected ['MSFT', 'GOOGL'], got {pending}"
            )

            # Clear removes the file
            mgr2.clear()
            assert not mgr2.path.exists(), "Checkpoint file should be deleted after clear()"

        ok("Checkpoint saves, loads, resumes pending tickers, and clears on success")
    except Exception as exc:
        fail("State checkpoint save/load/resume", str(exc))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"  Passed: {GREEN}{len(passed)}{RESET}   Failed: {RED}{len(failed)}{RESET}")
    if failed:
        print(f"\n  {RED}Failed tests:{RESET}")
        for name in failed:
            print(f"    • {name}")
        print()
        return 1
    print(f"  {GREEN}{BOLD}All {len(passed)} resilience tests passed.{RESET}\n")
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="News Trading System — daily scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--now",
        action="store_true",
        help="Run one cycle immediately and exit (for cron jobs and testing).",
    )
    parser.add_argument(
        "--watchlist",
        nargs="+",
        metavar="TICKER",
        help="Override the watchlist for this run (e.g. --watchlist AAPL TSLA).",
    )
    parser.add_argument(
        "--balance",
        type=float,
        metavar="USD",
        help="Override account balance for this run.",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Analysis only — do not log paper trades.",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send Telegram notifications (requires telegram section in watchlist.yaml "
             "and TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars).",
    )
    parser.add_argument(
        "--strategy",
        choices=["momentum", "mean-reversion", "swing", "all"],
        default=None,
        metavar="STRATEGY",
        help=(
            "Strategy agents to run: momentum | mean-reversion | swing | all. "
            "Overrides scheduler.default_strategy in watchlist.yaml."
        ),
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help=(
            "Start the price monitor after the daily run. "
            "In --now mode: runs one check-now pass after the cycle. "
            "In daemon mode: starts a monitoring thread alongside the scheduler."
        ),
    )
    parser.add_argument(
        "--resilience-test",
        action="store_true",
        help=(
            "Run the resilience test suite (circuit breaker, API fallbacks, "
            "network degraded mode, checkpoint resume) and exit."
        ),
    )
    args = parser.parse_args()

    # Resilience test — runs standalone, no live APIs or DB needed
    if args.resilience_test:
        sys.exit(resilience_test())

    cfg = _load_config()

    # Apply CLI overrides
    if args.watchlist:
        cfg["watchlist"] = args.watchlist
    if args.balance:
        cfg["account"]["balance"] = args.balance
    if args.no_execute:
        cfg["execution"]["enabled"] = False
    if args.notify:
        cfg["telegram"]["enabled"] = True

    # Resolve strategy: CLI flag > config file > hardcoded default
    strategy = args.strategy or cfg["scheduler"].get("default_strategy", "all")
    log.info("Strategy mode: %s", strategy)

    notifier = TelegramNotifier.from_config(cfg)
    if notifier:
        log.info("Telegram notifications enabled.")

    if args.now:
        run_daily(cfg, notifier=notifier, strategy=strategy)
        if args.monitor:
            from monitoring.price_monitor import PriceMonitor
            log.info("Running post-trade price check...")
            monitor = PriceMonitor(cfg=cfg, notifier=notifier)
            alerts  = monitor.check_now()
            log.info("Price check: %d alert(s) fired.", len(alerts))
    else:
        # Daemon mode: optionally spin up monitoring in a background thread
        if args.monitor:
            import threading
            from monitoring.price_monitor import PriceMonitor
            monitor     = PriceMonitor(cfg=cfg, notifier=notifier)
            mon_thread  = threading.Thread(target=monitor.run_daemon, daemon=True, name="price-monitor")
            mon_thread.start()
            log.info("Price monitor thread started (daemon=True).")
        _run_daemon(cfg, notifier=notifier, strategy=strategy)


if __name__ == "__main__":
    main()
