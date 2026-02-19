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

Usage
-----
  python3 scheduler/daily_runner.py                          # daemon
  python3 scheduler/daily_runner.py --now                    # run immediately
  python3 scheduler/daily_runner.py --now --strategy momentum  # momentum only

Cron example (installed by scheduler/install_cron.sh):
  30 9 * * 1-5  /usr/bin/python3 /path/to/scheduler/daily_runner.py --now >> /path/to/scheduler/logs/cron.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
import sys
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

    signals_generated = 0
    trades_executed   = 0
    errors: list[str] = []
    results: list[dict] = []

    # Per-agent signal counts (non-HOLD signals only)
    strategy_counts: dict[str, int] = {
        "momentum": 0, "mean_reversion": 0, "swing": 0,
    }

    for ticker in tickers:
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

    run_time      = cfg["schedule"]["time"]      # e.g. "09:30"
    weekdays_only = cfg["schedule"]["weekdays_only"]
    job           = functools.partial(run_daily, cfg, notifier=notifier, strategy=strategy)

    if weekdays_only:
        for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
            getattr(schedule.every(), day).at(run_time).do(job)
        log.info("Daemon started — will run Mon–Fri at %s (local time).", run_time)
    else:
        schedule.every().day.at(run_time).do(job)
        log.info("Daemon started — will run daily at %s (local time).", run_time)

    while True:
        schedule.run_pending()
        time.sleep(30)


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
    args = parser.parse_args()

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
