"""
Emergency Kill Switch for the News Trading System.

Creates / removes an emergency_stop.flag file that all trading agents
check before executing trades or starting a new cycle.

Commands
--------
  python3 emergency_stop.py --stop-trading   Block new trades (flag created)
  python3 emergency_stop.py --stop-all       Block trades AND send SIGTERM to daemons
  python3 emergency_stop.py --resume         Remove flag — resume normal operation
  python3 emergency_stop.py --status         Show current kill-switch state

Flag file
---------
  {PROJECT_ROOT}/emergency_stop.flag
  Content: JSON { "action": "...", "activated_at": "...", "reason": "..." }

Integration
-----------
  Agents and the paper-trader call KillSwitch.is_trading_blocked() before
  executing any trade.  When the flag exists they raise TradingBlocked.

Daemon shutdown (--stop-all)
----------------------------
  Sends SIGTERM to PIDs stored in {PROJECT_ROOT}/.pids/*.pid.
  Each daemon writes its own PID file on startup via KillSwitch.register_pid().
"""

from __future__ import annotations

import json
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

FLAG_FILE = PROJECT_ROOT / "emergency_stop.flag"
PIDS_DIR  = PROJECT_ROOT / ".pids"


# ══════════════════════════════════════════════════════════════════════════════
# KillSwitch — thin utility class (import-friendly, no heavy deps)
# ══════════════════════════════════════════════════════════════════════════════

class TradingBlocked(RuntimeError):
    """Raised when a trade is attempted while the kill switch is active."""


class KillSwitch:
    """
    Utility class for checking / managing the emergency-stop flag.

    All methods are class-level so no instance is needed:
        KillSwitch.is_trading_blocked()
        KillSwitch.register_pid()
    """

    # ------------------------------------------------------------------
    # Flag queries
    # ------------------------------------------------------------------

    @classmethod
    def is_stopped(cls) -> bool:
        """Return True if the kill switch is active (flag file exists)."""
        return FLAG_FILE.exists()

    @classmethod
    def is_trading_blocked(cls) -> bool:
        """Return True if trading should be blocked right now."""
        return FLAG_FILE.exists()

    @classmethod
    def get_state(cls) -> dict | None:
        """Return the flag contents as a dict, or None if not active."""
        if not FLAG_FILE.exists():
            return None
        try:
            return json.loads(FLAG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"action": "unknown", "activated_at": "unknown"}

    @classmethod
    def assert_trading_allowed(cls) -> None:
        """Raise TradingBlocked if the kill switch is active."""
        if cls.is_trading_blocked():
            state = cls.get_state() or {}
            raise TradingBlocked(
                f"Kill switch active: {state.get('action', 'stop')} "
                f"(activated {state.get('activated_at', 'unknown')}). "
                f"Run: python3 emergency_stop.py --resume"
            )

    # ------------------------------------------------------------------
    # Flag management
    # ------------------------------------------------------------------

    @classmethod
    def activate(cls, action: str, reason: str | None = None) -> None:
        """Write the flag file."""
        PIDS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "action":       action,
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "reason":       reason or "",
        }
        FLAG_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def deactivate(cls) -> None:
        """Remove the flag file."""
        if FLAG_FILE.exists():
            FLAG_FILE.unlink()

    # ------------------------------------------------------------------
    # PID file management (for --stop-all daemon termination)
    # ------------------------------------------------------------------

    @classmethod
    def register_pid(cls, name: str) -> None:
        """Write current process PID to .pids/{name}.pid."""
        PIDS_DIR.mkdir(parents=True, exist_ok=True)
        pid_file = PIDS_DIR / f"{name}.pid"
        pid_file.write_text(str(os.getpid()), encoding="utf-8")

    @classmethod
    def unregister_pid(cls, name: str) -> None:
        """Remove the PID file for this process on clean exit."""
        pid_file = PIDS_DIR / f"{name}.pid"
        if pid_file.exists():
            pid_file.unlink(missing_ok=True)

    @classmethod
    def _stop_all_daemons(cls) -> list[str]:
        """Send SIGTERM to all registered daemons. Returns names of signalled processes."""
        if not PIDS_DIR.exists():
            return []
        signalled = []
        for pid_file in PIDS_DIR.glob("*.pid"):
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, signal.SIGTERM)
                signalled.append(f"{pid_file.stem} (PID {pid})")
            except (ProcessLookupError, ValueError):
                pid_file.unlink(missing_ok=True)  # stale
            except PermissionError as exc:
                print(f"  WARNING: could not signal {pid_file.stem}: {exc}")
        return signalled


# ══════════════════════════════════════════════════════════════════════════════
# CLI actions
# ══════════════════════════════════════════════════════════════════════════════

def _log_to_db(action: str, reason: str | None) -> None:
    """Persist the kill-switch event to the database (best-effort)."""
    try:
        from storage.database import Database
        db = Database()
        db.log_emergency_stop(
            action=action,
            reason=reason,
            activated_by=f"CLI (PID {os.getpid()})",
        )
    except Exception as exc:
        print(f"  [DB] Could not persist event: {exc}")


def _send_telegram(action: str, reason: str | None) -> None:
    """Send a Telegram alert about the kill-switch event (best-effort)."""
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
            icon = "🛑" if action != "resume" else "✅"
            msg  = (
                f"{icon} KILL SWITCH: {action.upper()}\n"
                + (f"Reason: {reason}" if reason else "")
            )
            notifier.send_message(msg)
    except Exception as exc:
        print(f"  [Telegram] Alert failed: {exc}")


def cmd_stop_trading(reason: str | None) -> None:
    print("Activating kill switch: stop-trading …")
    KillSwitch.activate("stop_trading", reason)
    print(f"  Flag written: {FLAG_FILE}")
    _log_to_db("stop_trading", reason)
    _send_telegram("stop_trading", reason)
    print("  New trades are now blocked. Existing positions are unaffected.")
    print(f"  To resume: python3 emergency_stop.py --resume")


def cmd_stop_all(reason: str | None) -> None:
    print("Activating kill switch: stop-all …")
    KillSwitch.activate("stop_all", reason)
    print(f"  Flag written: {FLAG_FILE}")

    signalled = KillSwitch._stop_all_daemons()
    if signalled:
        print(f"  SIGTERM sent to: {', '.join(signalled)}")
    else:
        print("  No registered daemons found (use KillSwitch.register_pid() in each daemon).")

    _log_to_db("stop_all", reason)
    _send_telegram("stop_all", reason)
    print("  All trading blocked. Daemons will stop at next cycle check.")


def cmd_resume(reason: str | None) -> None:
    if not KillSwitch.is_stopped():
        print("Kill switch is not active — nothing to resume.")
        return
    print("Deactivating kill switch …")
    KillSwitch.deactivate()
    print(f"  Flag removed: {FLAG_FILE}")
    _log_to_db("resume", reason)
    _send_telegram("resume", reason)
    print("  Trading resumed. Restart any stopped daemons manually.")


def cmd_status() -> None:
    state = KillSwitch.get_state()
    W = 60
    print(f"\n{'═' * W}")
    if state is None:
        print("  Kill switch: INACTIVE — trading is running normally")
    else:
        print(f"  Kill switch: ACTIVE")
        print(f"  Action      : {state.get('action', 'unknown')}")
        print(f"  Activated   : {state.get('activated_at', 'unknown')}")
        if state.get("reason"):
            print(f"  Reason      : {state['reason']}")

    print(f"\n  Flag file: {FLAG_FILE}")
    print(f"  Exists   : {FLAG_FILE.exists()}")

    # Registered daemons
    if PIDS_DIR.exists():
        pids = list(PIDS_DIR.glob("*.pid"))
        if pids:
            print(f"\n  Registered daemons ({len(pids)}):")
            for p in pids:
                pid_val = p.read_text().strip()
                alive = _pid_alive(int(pid_val)) if pid_val.isdigit() else False
                print(f"    {p.stem:<20} PID {pid_val}  {'running' if alive else 'stopped'}")
    print(f"{'═' * W}\n")


def _pid_alive(pid: int) -> bool:
    """Return True if a process with the given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Emergency Kill Switch for the News Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--stop-trading",
        action="store_true",
        help="Block new trades immediately (does not stop running processes).",
    )
    group.add_argument(
        "--stop-all",
        action="store_true",
        help="Block trades AND send SIGTERM to all registered daemon processes.",
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help="Remove the kill switch — resume normal operation.",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Show the current kill-switch state and registered daemons.",
    )
    parser.add_argument(
        "--reason",
        default=None,
        metavar="TEXT",
        help="Optional reason string logged with the activation event.",
    )
    args = parser.parse_args()

    if args.stop_trading:
        cmd_stop_trading(args.reason)
    elif args.stop_all:
        cmd_stop_all(args.reason)
    elif args.resume:
        cmd_resume(args.reason)
    elif args.status:
        cmd_status()


if __name__ == "__main__":
    main()
