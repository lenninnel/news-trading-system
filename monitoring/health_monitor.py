"""
System Health Monitor for the News Trading System.

Runs five checks every N minutes (default: 5):
  1. database    — can we query the DB?
  2. api_keys    — are ANTHROPIC_API_KEY and NEWSAPI_KEY set?
  3. scheduler   — was the last scheduler run within the past 24 hours?
  4. disk_space  — is at least 1 GB free on the data volume?
  5. memory      — is memory usage below 80 %?

On failure, a Telegram alert is sent and the result is logged to the
health_checks table.

The monitor also embeds a lightweight HTTP server (stdlib http.server) that
serves the last check result as JSON on /health (default port 9090).

Modes
-----
  Daemon      python3 monitoring/health_monitor.py --daemon
              Loops forever, checks every --interval seconds.

  Once        python3 monitoring/health_monitor.py --check-now
              Single pass, print results, exit.

Config (watchlist.yaml → health_monitor section)
-------------------------------------------------
  health_monitor:
    enabled: true
    interval: 300          # seconds between checks
    http_port: 9090        # 0 = disable HTTP server
    disk_min_gb: 1.0       # minimum free disk space
    memory_max_pct: 80.0   # maximum memory usage %
    scheduler_max_age_h: 24  # max hours since last scheduler run
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH                            # noqa: E402
from storage.database import Database                          # noqa: E402
from utils.retry import with_retry                             # noqa: E402

log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INTERVAL         = 300    # 5 minutes
DEFAULT_HTTP_PORT        = 9090
DEFAULT_DISK_MIN_GB      = 1.0
DEFAULT_MEMORY_MAX_PCT   = 80.0
DEFAULT_SCHED_MAX_AGE_H  = 24


# ══════════════════════════════════════════════════════════════════════════════
# HealthMonitor
# ══════════════════════════════════════════════════════════════════════════════

class HealthMonitor:
    """
    Periodic system health checker.

    Args:
        cfg:       Full config dict (from watchlist.yaml).
        notifier:  Optional TelegramNotifier for alert messages.
        db:        Optional Database instance (creates one if None).
    """

    def __init__(
        self,
        cfg: dict | None = None,
        notifier: Any | None = None,
        db: Database | None = None,
    ) -> None:
        cfg = cfg or {}
        hm  = cfg.get("health_monitor", {})

        self._notifier      = notifier
        self._db            = db or Database()
        self._interval      = int(hm.get("interval",             DEFAULT_INTERVAL))
        self._http_port     = int(hm.get("http_port",            DEFAULT_HTTP_PORT))
        self._disk_min_gb   = float(hm.get("disk_min_gb",        DEFAULT_DISK_MIN_GB))
        self._mem_max_pct   = float(hm.get("memory_max_pct",     DEFAULT_MEMORY_MAX_PCT))
        self._sched_max_age = float(hm.get("scheduler_max_age_h", DEFAULT_SCHED_MAX_AGE_H))

        self._shutdown      = threading.Event()
        self._last_result: dict = {}

    # ------------------------------------------------------------------
    # Public modes
    # ------------------------------------------------------------------

    def run_daemon(self) -> None:
        """Block forever, checking every self._interval seconds."""
        log.info(
            "HealthMonitor daemon started (interval=%ds, http_port=%s)",
            self._interval,
            self._http_port if self._http_port else "disabled",
        )

        def _handle_signal(sig, _frame):
            log.info("Signal %s received — shutting down health monitor.", sig)
            self._shutdown.set()

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT,  _handle_signal)

        # Start embedded HTTP server
        if self._http_port:
            self._start_http_server()

        try:
            while not self._shutdown.is_set():
                self._run_checks(notify=True)
                self._shutdown.wait(timeout=self._interval)
        finally:
            log.info("HealthMonitor daemon stopped. Logging shutdown.")
            self._log_shutdown()

    def check_now(self) -> dict:
        """Run one check pass, print results, return the result dict."""
        return self._run_checks(notify=False)

    # ------------------------------------------------------------------
    # Check execution
    # ------------------------------------------------------------------

    def _run_checks(self, notify: bool = True) -> dict:
        """Execute all checks, log to DB, alert on failure. Returns result dict."""
        checked_at = datetime.now(timezone.utc).isoformat()
        checks: dict[str, bool] = {}
        details: dict[str, str] = {}

        # 1. Database
        db_ok, db_msg = self._check_database()
        checks["database"]   = db_ok
        details["database"]  = db_msg

        # 2. API keys
        api_ok, api_msg = self._check_api_keys()
        checks["api_keys"]   = api_ok
        details["api_keys"]  = api_msg

        # 3. Scheduler recency
        sched_ok, sched_msg = self._check_scheduler()
        checks["scheduler"]  = sched_ok
        details["scheduler"] = sched_msg

        # 4. Disk space
        disk_ok, disk_msg = self._check_disk()
        checks["disk_space"] = disk_ok
        details["disk_space"] = disk_msg

        # 5. Memory
        mem_ok, mem_msg = self._check_memory()
        checks["memory"]     = mem_ok
        details["memory"]    = mem_msg

        overall_ok = all(checks.values())

        result = {
            "status":     "healthy" if overall_ok else "degraded",
            "checked_at": checked_at,
            "overall_ok": overall_ok,
            "checks":     {k: {"ok": v, "detail": details[k]} for k, v in checks.items()},
        }
        self._last_result = result

        # Persist to DB
        try:
            self._db.log_health_check(
                checked_at  = checked_at,
                database_ok = checks["database"],
                api_keys_ok = checks["api_keys"],
                scheduler_ok= checks["scheduler"],
                disk_ok     = checks["disk_space"],
                memory_ok   = checks["memory"],
                overall_ok  = overall_ok,
                details_json= json.dumps(details),
            )
        except Exception as exc:
            log.warning("Could not persist health check result: %s", exc)

        # Telegram alert on failure
        if not overall_ok and notify and self._notifier is not None:
            failed = [k for k, v in checks.items() if not v]
            msg = (
                f"🚨 SYSTEM HEALTH ALERT\n"
                f"Failed checks: {', '.join(failed)}\n"
                + "\n".join(f"  • {k}: {details[k]}" for k in failed)
            )
            try:
                self._notifier.send_message(msg)
            except Exception as exc:
                log.warning("Telegram alert failed: %s", exc)

        if not overall_ok:
            failed = [k for k, v in checks.items() if not v]
            log.warning("Health check FAILED: %s", ", ".join(failed))
        else:
            log.debug("Health check OK.")

        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_database(self) -> tuple[bool, str]:
        """Verify DB is accessible by fetching one recent run."""
        try:
            @with_retry(max_attempts=2, base_delay=1.0)
            def _query():
                self._db.get_recent_runs(limit=1)

            _query()
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    def _check_api_keys(self) -> tuple[bool, str]:
        """Check that required API keys are set (no live call)."""
        missing = []
        if not os.environ.get("ANTHROPIC_API_KEY"):
            missing.append("ANTHROPIC_API_KEY")
        if not os.environ.get("NEWSAPI_KEY"):
            missing.append("NEWSAPI_KEY")
        if missing:
            return False, f"Missing: {', '.join(missing)}"
        return True, "all keys present"

    def _check_scheduler(self) -> tuple[bool, str]:
        """Check that the scheduler ran within the past self._sched_max_age hours."""
        try:
            rows = self._db._select(
                "SELECT run_at FROM scheduler_logs ORDER BY id DESC LIMIT 1"
            )
            if not rows:
                return False, "no scheduler runs found"

            run_at_str = rows[0]["run_at"]
            # Parse ISO-8601 (with or without timezone)
            try:
                run_at = datetime.fromisoformat(run_at_str)
                if run_at.tzinfo is None:
                    run_at = run_at.replace(tzinfo=timezone.utc)
            except ValueError:
                return False, f"unparseable timestamp: {run_at_str}"

            now = datetime.now(timezone.utc)
            age_h = (now - run_at).total_seconds() / 3600
            if age_h > self._sched_max_age:
                return False, f"last run {age_h:.1f}h ago (limit: {self._sched_max_age}h)"
            return True, f"last run {age_h:.1f}h ago"

        except Exception as exc:
            return False, str(exc)

    def _check_disk(self) -> tuple[bool, str]:
        """Check free disk space on the volume containing the DB."""
        try:
            path = Path(self._db._db_path).resolve().parent
            usage = shutil.disk_usage(str(path))
            free_gb = usage.free / (1024 ** 3)
            if free_gb < self._disk_min_gb:
                return False, f"{free_gb:.2f} GB free (min: {self._disk_min_gb} GB)"
            return True, f"{free_gb:.1f} GB free"
        except Exception as exc:
            return False, str(exc)

    def _check_memory(self) -> tuple[bool, str]:
        """Check system memory usage via psutil (graceful fallback if missing)."""
        try:
            import psutil
            pct = psutil.virtual_memory().percent
            if pct > self._mem_max_pct:
                return False, f"{pct:.1f}% used (max: {self._mem_max_pct:.0f}%)"
            return True, f"{pct:.1f}% used"
        except ImportError:
            return True, "psutil not installed — check skipped"
        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------------
    # Embedded HTTP server
    # ------------------------------------------------------------------

    def _start_http_server(self) -> None:
        """Start a background thread serving /health as JSON."""
        monitor = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in ("/health", "/"):
                    body = json.dumps(monitor._last_result, indent=2).encode()
                    self.send_response(200 if monitor._last_result.get("overall_ok", True) else 503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *args):  # type: ignore[override]
                pass  # silence access log

        def _serve() -> None:
            try:
                srv = HTTPServer(("0.0.0.0", self._http_port), _Handler)
                log.info("Health HTTP server listening on port %d", self._http_port)
                while not self._shutdown.is_set():
                    srv.handle_request()
                srv.server_close()
            except Exception as exc:
                log.error("Health HTTP server error: %s", exc)

        t = threading.Thread(target=_serve, daemon=True, name="health-http")
        t.start()

    # ------------------------------------------------------------------
    # Graceful shutdown helpers
    # ------------------------------------------------------------------

    def _log_shutdown(self) -> None:
        """Log a final health entry marking the monitor as shut down."""
        try:
            self._db.log_health_check(
                checked_at  = datetime.now(timezone.utc).isoformat(),
                database_ok = False,
                api_keys_ok = False,
                scheduler_ok= False,
                disk_ok     = False,
                memory_ok   = False,
                overall_ok  = False,
                details_json= json.dumps({"shutdown": "health monitor stopped"}),
            )
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Config loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_config() -> dict:
    import yaml
    path = PROJECT_ROOT / "config" / "watchlist.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    hm = cfg.setdefault("health_monitor", {})
    hm.setdefault("enabled",              True)
    hm.setdefault("interval",             DEFAULT_INTERVAL)
    hm.setdefault("http_port",            DEFAULT_HTTP_PORT)
    hm.setdefault("disk_min_gb",          DEFAULT_DISK_MIN_GB)
    hm.setdefault("memory_max_pct",       DEFAULT_MEMORY_MAX_PCT)
    hm.setdefault("scheduler_max_age_h",  DEFAULT_SCHED_MAX_AGE_H)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _print_result(result: dict) -> None:
    status  = result.get("status", "unknown").upper()
    checked = result.get("checked_at", "")
    W = 60
    print(f"\n{'═' * W}")
    print(f"  Health Monitor — {status}  ({checked})")
    print(f"{'═' * W}")
    for name, info in result.get("checks", {}).items():
        icon = "✓" if info["ok"] else "✗"
        print(f"  {icon}  {name:<15}  {info['detail']}")
    print(f"{'═' * W}\n")


def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="System Health Monitor")
    mode   = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--daemon",     action="store_true", help="Run continuously.")
    mode.add_argument("--check-now",  action="store_true", help="Single check and exit.")
    parser.add_argument("--notify",   action="store_true", help="Enable Telegram alerts.")
    parser.add_argument("--interval", type=int, default=None,
                        help="Override check interval in seconds.")
    args = parser.parse_args()

    cfg = _load_config()
    if args.interval:
        cfg["health_monitor"]["interval"] = args.interval

    notifier = None
    if args.notify:
        from notifications.telegram_bot import TelegramNotifier
        notifier = TelegramNotifier.from_config(cfg)

    monitor = HealthMonitor(cfg=cfg, notifier=notifier)

    if args.check_now:
        result = monitor.check_now()
        _print_result(result)
        sys.exit(0 if result["overall_ok"] else 1)
    else:
        monitor.run_daemon()


if __name__ == "__main__":
    main()
