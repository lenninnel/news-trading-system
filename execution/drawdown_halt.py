"""
Drawdown halt — BUY-only block triggered by peak-to-current NetLiq drawdown.

Designed 2026-05-01, shipped 2026-05-12. Tracks an all-time-high NetLiq peak
in the ``portfolio_peak`` table and flips ``halted=1`` when the live
NetLiquidation drops more than ``DRAWDOWN_HALT_THRESHOLD`` (default 10%)
below that peak. PortfolioManager.can_add_position reads the flag and
refuses new BUYs while halted. SELLs / stops / TPs go through PositionManager
and bypass this gate entirely.

Manual unlock only — the peak does NOT reset itself when NetLiq recovers.
After unlocking, the peak remains where it was; a new peak is reached only
when NetLiq sets a fresh high.

Commands
--------
  python3 -m execution.drawdown_halt --status
        Show peak, last NetLiq seen, drawdown %, halt state.

  python3 -m execution.drawdown_halt --unlock --reason "manual review ok"
        Clear an active halt. New BUYs resume immediately.

  python3 -m execution.drawdown_halt --reset-peak --value 95000 --reason "deposit $5k"
        Set the peak to a new value. Use after deposits / withdrawals
        that legitimately change the baseline. Also clears any halt.

  python3 -m execution.drawdown_halt --snapshot
        Query the live broker NetLiq once and feed it through the peak
        update (ratchet + halt-check). Useful for forcing an out-of-cycle
        check without waiting for the next session balance fetch.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DRAWDOWN_HALT_THRESHOLD
from storage.database import Database


def _send_telegram(msg: str) -> None:
    """Best-effort Telegram alert mirroring emergency_stop.py's pattern."""
    try:
        import yaml
        watchlist_path = PROJECT_ROOT / "config" / "watchlist.yaml"
        if not watchlist_path.exists():
            return
        with open(watchlist_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        from notifications.telegram_bot import TelegramNotifier
        notifier = TelegramNotifier.from_config(cfg)
        if notifier:
            notifier.send_message(msg)
    except Exception as exc:
        print(f"  [Telegram] alert failed: {exc}")


def _fmt_money(v) -> str:
    if v is None:
        return "—"
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def cmd_status(db: Database) -> int:
    state = db.get_drawdown_state()
    W = 64
    print(f"\n{'═' * W}")
    print("  Drawdown halt status")
    print(f"{'─' * W}")
    if state.get("peak_value") is None:
        print("  Peak: not yet initialized")
        print("  Halt: INACTIVE")
        print(f"\n  Threshold: {DRAWDOWN_HALT_THRESHOLD:.0%}")
        print(f"{'═' * W}\n")
        return 0

    halted = bool(state.get("halted"))
    print(f"  Peak NetLiq      : {_fmt_money(state['peak_value'])}")
    print(f"  Peak observed    : {state.get('peak_observed_at') or '—'}")
    print(f"  Last updated     : {state.get('updated_at') or '—'}")
    print(f"  Threshold        : {DRAWDOWN_HALT_THRESHOLD:.0%}")
    print()
    if halted:
        print(f"  STATUS           : 🛑 HALTED")
        print(f"  Halted at        : {state.get('halted_at') or '—'}")
        print(f"  NetLiq at halt   : {_fmt_money(state.get('halted_value'))}")
        print(f"  Drawdown at halt : {_fmt_pct(state.get('halted_drawdown_pct'))}")
        print(f"  Reason           : {state.get('halt_reason') or '—'}")
        print()
        print("  To unlock: python3 -m execution.drawdown_halt --unlock --reason \"...\"")
    else:
        print(f"  STATUS           : ✅ active (no halt)")
        if state.get("unlocked_at"):
            print(f"  Last unlocked    : {state['unlocked_at']} by {state.get('unlocked_by') or '—'}")
            if state.get("unlock_reason"):
                print(f"  Unlock reason    : {state['unlock_reason']}")
    print(f"{'═' * W}\n")
    return 0


def cmd_unlock(db: Database, reason: str | None) -> int:
    who = f"CLI (PID {os.getpid()}, user {os.environ.get('USER', '?')})"
    cleared = db.unlock_drawdown_halt(who, reason)
    if not cleared:
        print("  No active halt — nothing to unlock.")
        return 0
    print("  ✅ Drawdown halt cleared. New BUYs will be allowed on next signal.")
    _send_telegram(
        "✅ *DRAWDOWN HALT CLEARED*\n"
        f"By: {who}\n"
        + (f"Reason: {reason}" if reason else "")
    )
    return 0


def cmd_reset_peak(db: Database, value: float, reason: str | None) -> int:
    who = f"CLI (PID {os.getpid()}, user {os.environ.get('USER', '?')})"
    state = db.reset_drawdown_peak(value, who, reason)
    print(f"  Peak reset to {_fmt_money(state['peak_value'])} at {state['peak_observed_at']}")
    _send_telegram(
        "🔄 *DRAWDOWN PEAK RESET*\n"
        f"New peak: {_fmt_money(value)}\n"
        f"By: {who}\n"
        + (f"Reason: {reason}" if reason else "")
    )
    return 0


def cmd_snapshot(db: Database) -> int:
    """Force a peak-update cycle using a fresh broker NetLiq."""
    try:
        from execution.ibkr_trader import IBKRTrader
    except Exception as exc:
        print(f"  Cannot import IBKRTrader: {exc}")
        return 2
    try:
        trader = IBKRTrader()
        account = trader.get_account() or {}
        value = float(account.get("portfolio_value") or 0.0)
        try:
            trader.disconnect()
        except Exception:
            pass
    except Exception as exc:
        print(f"  Broker NetLiq fetch failed: {exc}")
        return 2

    if value <= 0:
        print(f"  Broker returned non-positive NetLiq ({value}); aborting.")
        return 2

    state = db.update_portfolio_peak(value, DRAWDOWN_HALT_THRESHOLD)
    dd = state.get("drawdown_pct") or 0.0
    print(
        f"  Snapshot: NetLiq={_fmt_money(value)} "
        f"peak={_fmt_money(state['peak_value'])} "
        f"dd={dd:.2%} halted={state.get('halted')}"
    )
    if state.get("newly_halted"):
        _send_telegram(
            "🛑 *DRAWDOWN HALT TRIGGERED (snapshot)*\n"
            f"NetLiq: {_fmt_money(value)}\n"
            f"Peak: {_fmt_money(state['peak_value'])}\n"
            f"Drawdown: {dd:.2%}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drawdown halt — BUY-only block from peak NetLiq drawdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status",     action="store_true", help="Show current halt state.")
    group.add_argument("--unlock",     action="store_true", help="Clear an active halt.")
    group.add_argument("--reset-peak", action="store_true", help="Reset the peak (use --value).")
    group.add_argument("--snapshot",   action="store_true", help="Fetch live NetLiq and update peak/halt.")
    parser.add_argument("--reason", default=None, metavar="TEXT", help="Reason logged with the event.")
    parser.add_argument("--value",  type=float, default=None, metavar="USD", help="New peak value (with --reset-peak).")
    args = parser.parse_args()

    db = Database()

    if args.status:
        return cmd_status(db)
    if args.unlock:
        return cmd_unlock(db, args.reason)
    if args.reset_peak:
        if args.value is None or args.value <= 0:
            parser.error("--reset-peak requires --value <positive USD>")
        return cmd_reset_peak(db, float(args.value), args.reason)
    if args.snapshot:
        return cmd_snapshot(db)
    return 2


if __name__ == "__main__":
    sys.exit(main())
