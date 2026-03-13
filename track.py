"""
News Trading System — CLI entry point with optional Telegram notifications.

Usage::

    # Run the pipeline (same as scheduler/daily_runner.py --now)
    python track.py

    # Run with Telegram notifications
    python track.py --notify

    # Custom watchlist + balance
    python track.py --watchlist AAPL,MSFT,NVDA --balance 25000 --notify

    # Dry run (no trade execution)
    python track.py --no-execute
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notifications.telegram_bot import TelegramNotifier  # noqa: E402
from scheduler.daily_runner import DEFAULT_WATCHLIST, run_batch  # noqa: E402


def _load_watchlist_config() -> dict:
    """Load config/watchlist.yaml, returning an empty dict on failure."""
    path = PROJECT_ROOT / "config" / "watchlist.yaml"
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _build_telegram(cfg: dict) -> TelegramNotifier | None:
    """Build a TelegramNotifier from env vars (preferred) or config."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if token and chat_id:
        dashboard_url = cfg.get("telegram", {}).get("dashboard_url", "")
        return TelegramNotifier(
            bot_token=token,
            chat_id=chat_id,
            dashboard_url=dashboard_url,
        )
    # Fall back to config-based factory
    return TelegramNotifier.from_config(cfg)


def _send_notifications(
    tg: TelegramNotifier,
    batch: dict,
    tickers: list[str],
) -> None:
    """Send per-signal alerts and a daily summary via Telegram. Never raises."""
    results = batch["results"]

    # Per-signal alerts for actionable signals
    for r in results:
        if r is None:
            continue
        signal = r.get("combined_signal", "HOLD")
        if signal in ("HOLD", "CONFLICTING"):
            continue
        try:
            tg.send_signal(
                ticker=r["ticker"],
                signal=signal,
                confidence=r.get("confidence", 0) * 100,
                reasoning=r.get("sentiment", {}).get("signal", ""),
            )
        except Exception:
            pass

    # Trade execution alerts
    for r in results:
        if r is None:
            continue
        execution = r.get("execution")
        if not execution or not execution.get("trade_id"):
            continue
        try:
            risk = r.get("risk", {})
            tg.send_trade_executed(
                ticker=r["ticker"],
                action=risk.get("direction", "BUY"),
                shares=risk.get("shares", 0),
                price=risk.get("current_price", 0),
                stop_loss=risk.get("stop_loss", 0),
                take_profit=risk.get("take_profit", 0),
            )
        except Exception:
            pass

    # Daily summary
    try:
        summary_results = []
        errors: list[str] = []
        for i, r in enumerate(results):
            if r is None:
                errors.append(f"{tickers[i]}: pipeline failed")
                continue
            summary_results.append({
                "ticker": r["ticker"],
                "signal": r.get("combined_signal", "?"),
                "conf": r.get("confidence", 0),
                "traded": bool(r.get("execution", {}).get("trade_id")),
            })

        portfolio_value = sum(
            r.get("account_balance", 0)
            for r in results
            if r is not None
        )
        # Use the first non-None result's balance as the portfolio value
        for r in results:
            if r is not None:
                portfolio_value = r.get("account_balance", 0)
                break

        trades_count = sum(
            1 for r in results
            if r and r.get("execution", {}).get("trade_id")
        )
        status = (
            "success" if batch["fail_count"] == 0
            else "failed" if batch["success_count"] == 0
            else "partial"
        )

        tg.send_daily_summary(
            signals_count=batch["success_count"],
            trades_count=trades_count,
            portfolio_value=portfolio_value,
            results=summary_results,
            errors=errors,
            status=status,
        )
    except Exception:
        pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="News Trading System — run pipeline with optional Telegram alerts",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        default=False,
        help="Send results via Telegram (requires TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID).",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    tickers = (
        [t.strip().upper() for t in args.watchlist.split(",")]
        if args.watchlist
        else DEFAULT_WATCHLIST
    )
    execute = not args.no_execute

    # Build Telegram notifier if --notify
    tg: TelegramNotifier | None = None
    if args.notify:
        cfg = _load_watchlist_config()
        tg = _build_telegram(cfg)
        if tg is None:
            print(
                "WARNING: --notify given but Telegram credentials not found. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.",
                file=sys.stderr,
            )

    print(f"\nNews Trading System — track.py")
    print(f"  Tickers:  {len(tickers)}")
    print(f"  Workers:  {args.workers}")
    print(f"  Balance:  ${args.balance:,.2f}")
    print(f"  Execute:  {execute}")
    print(f"  Notify:   {'Telegram' if tg else 'off'}")
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

    # Send Telegram notifications (wrapped in try/except — never crashes pipeline)
    if tg:
        try:
            _send_notifications(tg, batch, tickers)
            print("Telegram notifications sent.")
        except Exception as exc:
            print(f"Telegram notification error (non-fatal): {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
