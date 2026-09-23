#!/usr/bin/env python3
"""
NTS watchdog — alerts on the *absence* of expected events.

Everything else in the system alerts from inside a running process
(daemon, PositionManager, ingest). This script is the outside view: it
runs as an independent systemd user timer (deployment/systemd/
nts-watchdog.timer, every 15 minutes) and checks that what should have
happened did happen. It depends on nothing that could be the thing that
failed — standard library only, plain-text Telegram, its own state file.

Checks (kind → behaviour)
-------------------------
state  daemon         nts-trading.service is active
state  session:D:NAME every session that exists on today's calendar
                      (config/sessions.py: US sessions only on US trading
                      days and before the NY close, XETRA on weekdays) has
                      claimed its session_runs slot once its time + grace
                      has passed (and, since 2026-09-23, was not ABORTED by
                      the bar-freshness gate — session_runs.note).  On a
                      NYSE holiday no US session is expected; the status
                      block says so ("US calendar" info line).
state  ohlc           daily_ohlc holds the last completed US trading day
                      (ingest runs 22:30 UTC, expected from 23:00 UTC on)
state  timer:*        nts-ohlc-ingest.timer / nts-backup.timer are active
state  backup         newest file in the backup dir is younger than 26 h
state  gateway        IB Gateway API port accepts TCP (after 2 misses;
                      a NEW outage inside the nightly Gateway maintenance
                      window, default 00:00–06:15 UTC, is shown but not
                      alerted — see check_gateway)
state  disk           ≥ 1 GB free on the DB volume
state  db             the SQLite DB opens and answers
event  restarts       nts-trading auto-restarted since the last check
info   kill switch, drawdown halt, open positions (status block only)

Messaging policy (few messages that mean something)
----------------------------------------------------
* A state check that flips to failing → one alert. While it stays failing
  → one reminder every REALERT_HOURS. When it recovers → one recovery line.
* Events are reported each time they occur.
* One status block per day at HEARTBEAT_UTC_HOUR ("the watchdog lives").
  Its absence is the signal that the watchdog, its timer, or Telegram
  itself is broken.
* All of the above go out as ONE message per run.

Exit codes: 0 ok · 2 a message was due but could not be delivered ·
1 internal error (traceback on stderr). Both non-zero states trip
OnFailure=nts-alert@%n.service on the unit.

Usage:
    watchdog.py              # normal timer run
    watchdog.py --dry-run    # print what would be sent, touch nothing
    watchdog.py --heartbeat  # force the status block (install check)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from notifications.telegram_plain import credentials_from_env, send_plain_text  # noqa: E402

log = logging.getLogger("nts.watchdog")

# --- schedule / calendar: repo modules are stdlib-only; fall back if absent ---
try:  # pragma: no cover - exercised implicitly
    from config.sessions import SCHEDULE as _SCHEDULE
except Exception:  # pragma: no cover
    _SCHEDULE = [
        {"name": "XETRA_PRE", "hour": 6, "minute": 45, "market": "XETRA"},
        {"name": "XETRA_OPEN", "hour": 7, "minute": 0, "market": "XETRA"},
        {"name": "PREMARKET_SCAN", "hour": 13, "minute": 0, "market": "US"},
        {"name": "US_PRE", "hour": 13, "minute": 15, "market": "US"},
        {"name": "PEAD_OPEN", "hour": 13, "minute": 45, "market": "US"},
        {"name": "US_OPEN", "hour": 14, "minute": 30, "market": "US"},
        {"name": "MIDDAY", "hour": 18, "minute": 0, "market": "US"},
        {"name": "EOD", "hour": 22, "minute": 45, "market": "US", "after_close": True},
    ]

try:  # pragma: no cover
    from data.market_calendar import last_us_trading_day
except Exception:  # pragma: no cover
    def last_us_trading_day(d: date) -> date:
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d

# Which sessions exist on a given day — the SAME rule the daemon uses to
# decide what to fire (config/sessions.py, 2026-09-23): US sessions only on
# US trading days and before that day's NY close, XETRA sessions on
# weekdays.  So on Labor Day the watchdog expects the two XETRA rows and
# nothing else, and on the Friday after Thanksgiving it does not expect
# MIDDAY.  Fallback (repo module missing): weekdays only, as before.
try:  # pragma: no cover
    from config.sessions import session_runs_on, us_calendar_note
except Exception:  # pragma: no cover
    def session_runs_on(entry: dict, day: date) -> tuple[bool, "str | None"]:
        return (day.weekday() < 5), ("weekend" if day.weekday() >= 5 else None)

    def us_calendar_note(day: date) -> "str | None":
        return "weekend" if day.weekday() >= 5 else None

# Sessions the daemon skips when ENABLE_PRE_SESSIONS=false (daily_runner
# session_type == "pre_signal").
_PRE_SESSIONS = {"XETRA_PRE"}

STATE_VERSION = 1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


@dataclass
class Config:
    db_path: Path
    state_path: Path
    repo_dir: Path = REPO_DIR
    daemon_unit: str = "nts-trading.service"
    ingest_timer: str = "nts-ohlc-ingest.timer"
    backup_timer: str = "nts-backup.timer"
    backup_dir: Path = Path("/home/trading/backups")
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002
    session_grace_min: int = 20
    ohlc_ready_utc_hour: int = 23
    backup_max_age_h: float = 26.0
    realert_hours: float = 6.0
    heartbeat_utc_hour: int = 6
    gateway_fail_cycles: int = 2
    # Nightly IB Gateway maintenance window (UTC, [start, end), may wrap
    # midnight). Determined from the data 2026-09-07 (watchdog journal +
    # state, ibc-restart.timer on claw): the API port closes at 00:00 UTC
    # (Gateway auto-logoff) and comes back with the root-side
    # ibc-restart.timer at 06:00 UTC (listening again within ~40 s). The
    # 15-min tail past 06:00 covers the restart itself. None = no window.
    gateway_maint_window: tuple[time, time] | None = (time(0, 0), time(6, 15))
    disk_min_gb: float = 1.0
    pre_sessions_enabled: bool = True
    schedule: list[dict] = field(default_factory=lambda: list(_SCHEDULE))

    @classmethod
    def from_env(cls) -> "Config":
        db = (
            os.environ.get("NTS_WATCHDOG_DB")
            or os.environ.get("DB_PATH")
            or "/home/trading/trading-data/news_trading.db"
        )
        db_path = Path(db)
        if not db_path.is_absolute():
            db_path = (REPO_DIR / db_path).resolve()
        state = os.environ.get("NTS_WATCHDOG_STATE") or str(db_path.parent / "watchdog_state.json")
        paper = _env_bool("IBKR_PAPER", True)
        return cls(
            db_path=db_path,
            state_path=Path(state),
            backup_dir=Path(os.environ.get("NTS_BACKUP_DST_DIR", "/home/trading/backups")),
            ibkr_host=os.environ.get("IBKR_HOST", "127.0.0.1"),
            ibkr_port=int(os.environ.get("IBKR_PORT", "4002" if paper else "4001")),
            session_grace_min=int(os.environ.get("NTS_WATCHDOG_SESSION_GRACE_MIN", "20")),
            realert_hours=float(os.environ.get("NTS_WATCHDOG_REALERT_HOURS", "6")),
            heartbeat_utc_hour=int(os.environ.get("NTS_WATCHDOG_HEARTBEAT_UTC_HOUR", "6")),
            backup_max_age_h=float(os.environ.get("NTS_WATCHDOG_BACKUP_MAX_AGE_H", "26")),
            disk_min_gb=float(os.environ.get("NTS_WATCHDOG_DISK_MIN_GB", "1")),
            pre_sessions_enabled=_env_bool("ENABLE_PRE_SESSIONS", True),
            gateway_maint_window=parse_window(
                os.environ.get("NTS_WATCHDOG_GATEWAY_MAINT_UTC", "00:00-06:15")
            ),
        )


def parse_window(spec: str | None) -> tuple[time, time] | None:
    """'HH:MM-HH:MM' (UTC) → (start, end); '' / None / 'off' → None.

    Raises ValueError on malformed input so a typo in .env fails the unit
    loudly (OnFailure → nts-alert@) instead of silently disabling the
    window or the check.
    """
    if spec is None:
        return None
    spec = spec.strip()
    if not spec or spec.lower() in ("off", "none", "0"):
        return None
    try:
        start_s, end_s = spec.split("-")
        start = time.fromisoformat(start_s.strip())
        end = time.fromisoformat(end_s.strip())
    except ValueError as exc:
        raise ValueError(
            f"NTS_WATCHDOG_GATEWAY_MAINT_UTC={spec!r}: expected 'HH:MM-HH:MM'"
        ) from exc
    if start == end:
        raise ValueError(
            f"NTS_WATCHDOG_GATEWAY_MAINT_UTC={spec!r}: start and end must differ"
        )
    return (start, end)


def in_window(now: datetime, window: tuple[time, time] | None) -> bool:
    """True if ``now`` (UTC) falls in [start, end); handles midnight wrap."""
    if window is None:
        return False
    start, end = window
    t = now.astimezone(timezone.utc).time().replace(tzinfo=None)
    if start < end:
        return start <= t < end
    return t >= start or t < end


def _fmt_window(window: tuple[time, time]) -> str:
    return f"{window[0].strftime('%H:%M')}–{window[1].strftime('%H:%M')} UTC"


# ---------------------------------------------------------------------------
# Probes — the only places that touch the outside world (injectable in tests)
# ---------------------------------------------------------------------------

_SHOW_PROPS = (
    "LoadState,ActiveState,SubState,Result,NRestarts,"
    "ActiveEnterTimestamp,InactiveEnterTimestamp,ExecMainStatus"
)


def systemctl_show(unit: str) -> dict[str, str]:
    """``systemctl --user show <unit>`` → {prop: value}. Raises on failure."""
    proc = subprocess.run(
        ["systemctl", "--user", "show", unit, "--no-pager", "-p", _SHOW_PROPS],
        capture_output=True, text=True, timeout=20, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"systemctl show {unit} rc={proc.returncode}: {proc.stderr.strip()[:200]}"
        )
    props: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            props[k.strip()] = v.strip()
    return props


def tcp_open(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def disk_free_gb(path: Path) -> float:
    target = path if path.exists() else path.parent
    return shutil.disk_usage(target).free / (1024 ** 3)


@dataclass
class Probes:
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    systemctl_show: Callable[[str], dict[str, str]] = systemctl_show
    tcp_open: Callable[[str, int], bool] = tcp_open
    disk_free_gb: Callable[[Path], float] = disk_free_gb
    hostname: Callable[[], str] = socket.gethostname


# ---------------------------------------------------------------------------
# Check results
# ---------------------------------------------------------------------------

@dataclass
class Check:
    key: str            # stable identifier (state key in the JSON file)
    label: str          # human name in messages
    kind: str           # "state" | "event" | "info"
    ok: bool | None     # None = informational / could not evaluate
    detail: str
    muted: bool = False  # condition present but alert deliberately suppressed
                         # (planned maintenance); shown honestly in the
                         # status block, never counted as failing

    @property
    def failing(self) -> bool:
        return self.kind == "state" and self.ok is False

    @property
    def mark(self) -> str:
        if self.kind == "info":
            return "·"
        if self.muted:
            return "⏸"
        return "✓" if self.ok else "✗"


def _fmt_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _parse_systemd_ts(value: str) -> datetime | None:
    """'Mon 2026-08-31 14:02:11 UTC' → aware datetime (None if unparsable)."""
    parts = value.split()
    if len(parts) < 3:
        return None
    try:
        naive = datetime.strptime(f"{parts[1]} {parts[2]}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return naive.replace(tzinfo=timezone.utc)


def _age_h(now: datetime, then: datetime) -> float:
    return (now - then).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_daemon(cfg: Config, probes: Probes, state: dict) -> list[Check]:
    """Daemon active + auto-restart event."""
    now = probes.now()
    try:
        props = probes.systemctl_show(cfg.daemon_unit)
    except Exception as exc:
        return [Check("daemon", "daemon", "state", False,
                      f"cannot query systemd for {cfg.daemon_unit}: {exc}")]

    if props.get("LoadState") not in ("loaded", None):
        return [Check("daemon", "daemon", "state", False,
                      f"{cfg.daemon_unit} LoadState={props.get('LoadState')} (unit missing?)")]

    active = props.get("ActiveState") == "active"
    n_restarts = int(props.get("NRestarts") or 0)
    since = _parse_systemd_ts(props.get("ActiveEnterTimestamp", ""))
    if active:
        up = f"up {_age_h(now, since):.1f} h since {_fmt_ts(since)}" if since else "active"
        detail = f"{cfg.daemon_unit} {up}, {n_restarts} auto-restart(s) total"
    else:
        detail = (
            f"{cfg.daemon_unit} ActiveState={props.get('ActiveState')} "
            f"SubState={props.get('SubState')} Result={props.get('Result')} "
            f"exit={props.get('ExecMainStatus')} NRestarts={n_restarts}"
        )
    results = [Check("daemon", "daemon", "state", active, detail)]

    prev = state.get("nrestarts")
    if isinstance(prev, int) and n_restarts > prev:
        results.append(Check(
            "restarts", "daemon restart", "event", False,
            f"{cfg.daemon_unit} auto-restarted {n_restarts - prev}× since the last "
            f"check (NRestarts={n_restarts}); currently "
            f"{'active' if active else props.get('ActiveState')}",
        ))
    state["nrestarts"] = n_restarts
    return results


def check_timer(unit: str, probes: Probes) -> Check:
    key = f"timer:{unit}"
    try:
        props = probes.systemctl_show(unit)
    except Exception as exc:
        return Check(key, unit, "state", False, f"cannot query systemd: {exc}")
    if props.get("LoadState") not in ("loaded", None):
        return Check(key, unit, "state", False, f"LoadState={props.get('LoadState')} (not installed?)")
    active = props.get("ActiveState") == "active"
    return Check(key, unit, "state", active,
                 "active" if active else f"ActiveState={props.get('ActiveState')} — timer not running")


def _due_sessions(cfg: Config, now: datetime) -> list[tuple[str, datetime]]:
    """Sessions whose time + grace has passed today and that exist on
    today's calendar (weekday; for US sessions also a US trading day and
    before the NY close — ``config.sessions.session_runs_on``)."""
    if now.weekday() >= 5:
        return []
    due: list[tuple[str, datetime]] = []
    for entry in cfg.schedule:
        name = entry["name"]
        if name in _PRE_SESSIONS and not cfg.pre_sessions_enabled:
            continue
        if not session_runs_on(entry, now.date())[0]:
            continue
        at = now.replace(hour=int(entry["hour"]), minute=int(entry["minute"]),
                         second=0, microsecond=0)
        if now >= at + timedelta(minutes=cfg.session_grace_min):
            due.append((name, at))
    return due


def check_sessions(cfg: Config, conn: sqlite3.Connection, probes: Probes) -> list[Check]:
    now = probes.now()
    today = now.strftime("%Y-%m-%d")
    due = _due_sessions(cfg, now)
    if not due:
        return []
    try:
        # `note` (additive column, 2026-09-23) carries the bar-freshness
        # verdict: "ABORTED: …" when the session refused to run on a stale
        # daily bar, "SKIPPED n ticker(s): …" for a partial skip.  Older
        # DBs have no such column — read it only when present.
        has_note = any(
            r[1] == "note"
            for r in conn.execute("PRAGMA table_info(session_runs)").fetchall()
        )
        cols = "session, started_at" + (", note" if has_note else ", NULL")
        rows = conn.execute(
            f"SELECT {cols} FROM session_runs WHERE run_date = ?",
            (today,),
        ).fetchall()
        ran = {r[0]: (r[1], r[2]) for r in rows}
        table_missing = None
    except sqlite3.Error as exc:
        ran = {}
        table_missing = str(exc)

    checks: list[Check] = []
    for name, at in due:
        key = f"session:{today}:{name}"
        if name in ran:
            started, note = ran[name]
            if note and str(note).upper().startswith("ABORTED"):
                checks.append(Check(key, f"session {name}", "state", False,
                                    f"claimed {started[:16]} but {note}"))
            else:
                detail = f"ran (claimed {started[:16]})"
                if note:
                    detail += f" — {note}"
                checks.append(Check(key, f"session {name}", "state", True, detail))
        else:
            why = f"session_runs unreadable: {table_missing}" if table_missing else (
                f"no session_runs row by {now.strftime('%H:%M')} UTC "
                f"(scheduled {at.strftime('%H:%M')} UTC + {cfg.session_grace_min} min grace)"
            )
            checks.append(Check(key, f"session {name}", "state", False, why))
    return checks


def expected_ohlc_date(cfg: Config, now: datetime) -> date:
    """Last US trading day whose ingest (22:30 UTC) should be complete."""
    anchor = now.date() if now.hour >= cfg.ohlc_ready_utc_hour else now.date() - timedelta(days=1)
    return last_us_trading_day(anchor)


def check_ohlc(cfg: Config, conn: sqlite3.Connection, probes: Probes) -> Check:
    now = probes.now()
    expected = expected_ohlc_date(cfg, now)
    try:
        row = conn.execute("SELECT MAX(date) FROM daily_ohlc").fetchone()
        max_date = row[0] if row else None
        n_tickers = 0
        if max_date:
            n_tickers = conn.execute(
                "SELECT COUNT(DISTINCT ticker) FROM daily_ohlc WHERE date = ?",
                (max_date,),
            ).fetchone()[0]
    except sqlite3.Error as exc:
        return Check("ohlc", "OHLC ingest", "state", False, f"daily_ohlc unreadable: {exc}")

    if not max_date:
        return Check("ohlc", "OHLC ingest", "state", False,
                     f"daily_ohlc is empty (expected data through {expected})")
    ok = str(max_date) >= expected.isoformat()
    detail = f"MAX(date)={max_date} ({n_tickers} tickers), expected ≥ {expected}"
    if not ok:
        detail = f"STALE — {detail}; nts-ohlc-ingest did not deliver"
    return Check("ohlc", "OHLC ingest", "state", ok, detail)


def check_backup(cfg: Config, probes: Probes) -> Check:
    now = probes.now()
    if not cfg.backup_dir.is_dir():
        return Check("backup", "DB backup", "state", False,
                     f"backup dir {cfg.backup_dir} does not exist")
    newest: tuple[float, Path] | None = None
    for p in cfg.backup_dir.iterdir():
        if p.is_file():
            mtime = p.stat().st_mtime
            if newest is None or mtime > newest[0]:
                newest = (mtime, p)
    if newest is None:
        return Check("backup", "DB backup", "state", False,
                     f"no files in {cfg.backup_dir}")
    then = datetime.fromtimestamp(newest[0], tz=timezone.utc)
    age = _age_h(now, then)
    ok = age <= cfg.backup_max_age_h
    detail = f"{newest[1].name} written {_fmt_ts(then)} ({age:.1f} h ago)"
    if not ok:
        detail = f"STALE — {detail}, limit {cfg.backup_max_age_h:.0f} h"
    return Check("backup", "DB backup", "state", ok, detail)


def check_gateway(cfg: Config, probes: Probes, state: dict) -> Check:
    """IB Gateway API port reachable.

    Alert after ``gateway_fail_cycles`` consecutive misses — except when
    the outage BEGAN inside the nightly maintenance window
    (``gateway_maint_window``): the Gateway logs off at 00:00 UTC and is
    restarted by the root-side ibc-restart.timer at 06:00 UTC every day,
    so unreachability in that window is the plan, not an incident. Inside
    the window a fresh outage is reported as a muted check (visible in
    the status block as ⏸, never a 🚨). The moment the window ends and
    the port is still closed, the check fails and alerts as usual, naming
    the time of the first miss. An outage that was already failing when
    the window opened stays failing through it (no bogus recovery, no
    second alert) and reports its true recovery time.
    """
    now = probes.now()
    reachable = bool(probes.tcp_open(cfg.ibkr_host, cfg.ibkr_port))
    prev_streak = int(state.get("gateway_fail_streak", 0))
    streak = 0 if reachable else prev_streak + 1
    state["gateway_fail_streak"] = streak
    if reachable:
        state.pop("gateway_fail_since", None)
    elif prev_streak == 0 or not state.get("gateway_fail_since"):
        state["gateway_fail_since"] = now.isoformat(timespec="seconds")
    target = f"{cfg.ibkr_host}:{cfg.ibkr_port}"

    if reachable:
        return Check("gateway", "IB Gateway", "state", True, f"{target} accepts connections")
    if streak < cfg.gateway_fail_cycles:
        # A single miss is tolerated (Gateway restart, transient) — wait one more cycle.
        return Check("gateway", "IB Gateway", "state", True,
                     f"{target} unreachable ({streak}/{cfg.gateway_fail_cycles} misses, tolerated)")

    since = state.get("gateway_fail_since") or now.isoformat(timespec="seconds")
    since_txt = since[:16].replace("T", " ") + " UTC"
    already_failing = "gateway" in state.get("failing", {})
    window = cfg.gateway_maint_window
    if window is not None and in_window(now, window) and not already_failing:
        return Check(
            "gateway", "IB Gateway", "state", True,
            f"{target} unreachable for {streak} consecutive checks (since {since_txt}) — "
            f"inside the nightly Gateway maintenance window {_fmt_window(window)}, "
            f"alert suppressed; alerts if still down after the window",
            muted=True,
        )
    tail = ""
    if window is not None and not already_failing and in_window(
        datetime.fromisoformat(since), window
    ):
        tail = f" — did not come back after the maintenance window {_fmt_window(window)}"
    return Check("gateway", "IB Gateway", "state", False,
                 f"{target} unreachable for {streak} consecutive checks (since {since_txt})"
                 f"{tail} — no execution, no stop enforcement possible")


def check_disk(cfg: Config, probes: Probes) -> Check:
    try:
        free = float(probes.disk_free_gb(cfg.db_path))
    except Exception as exc:
        return Check("disk", "disk", "state", False, f"cannot stat {cfg.db_path}: {exc}")
    ok = free >= cfg.disk_min_gb
    return Check("disk", "disk", "state", ok,
                 f"{free:.1f} GB free on DB volume" + ("" if ok else f" (min {cfg.disk_min_gb} GB)"))


def info_checks(cfg: Config, conn: sqlite3.Connection | None,
                now: "datetime | None" = None) -> list[Check]:
    out: list[Check] = []
    # Calendar context so a quiet holiday reads as "expected", not "odd".
    if now is not None:
        note = us_calendar_note(now.date())
        if note and note != "weekend":
            skipped = [
                e["name"] for e in cfg.schedule
                if (e.get("market") or "US").upper() == "US"
                and not session_runs_on(e, now.date())[0]
            ]
            detail = note + (f" — not expected today: {', '.join(skipped)}" if skipped else "")
            out.append(Check("calendar", "US calendar", "info", None, detail))
    flag = cfg.repo_dir / "emergency_stop.flag"
    if flag.exists():
        try:
            data = json.loads(flag.read_text(encoding="utf-8"))
            detail = f"ACTIVE — {data.get('action', '?')} since {data.get('activated_at', '?')}"
        except Exception:
            detail = "ACTIVE (flag unreadable)"
        out.append(Check("killswitch", "kill switch", "info", None, detail))
    else:
        out.append(Check("killswitch", "kill switch", "info", None, "inactive"))

    if conn is not None:
        try:
            row = conn.execute(
                "SELECT halted, halted_at, halted_drawdown_pct FROM portfolio_peak "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row and row[0]:
                pct = f" ({float(row[2]) * 100:.1f}% dd)" if row[2] is not None else ""
                out.append(Check("drawdown", "drawdown halt", "info", None,
                                 f"HALTED since {row[1]}{pct} — BUYs blocked"))
            else:
                out.append(Check("drawdown", "drawdown halt", "info", None, "no"))
        except sqlite3.Error as exc:
            out.append(Check("drawdown", "drawdown halt", "info", None, f"unreadable: {exc}"))
        try:
            n = conn.execute(
                "SELECT COUNT(*) FROM portfolio_positions WHERE shares > 0"
            ).fetchone()[0]
            out.append(Check("positions", "open positions", "info", None, str(n)))
        except sqlite3.Error as exc:
            out.append(Check("positions", "open positions", "info", None, f"unreadable: {exc}"))
    return out


def run_checks(cfg: Config, probes: Probes, state: dict) -> list[Check]:
    checks: list[Check] = []
    checks.extend(check_daemon(cfg, probes, state))
    checks.append(check_gateway(cfg, probes, state))

    conn: sqlite3.Connection | None = None
    if not cfg.db_path.exists():
        checks.append(Check("db", "database", "state", False, f"{cfg.db_path} does not exist"))
    else:
        try:
            conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True, timeout=10)
            conn.execute("SELECT 1").fetchone()
            checks.append(Check("db", "database", "state", True, str(cfg.db_path)))
        except sqlite3.Error as exc:
            conn = None
            checks.append(Check("db", "database", "state", False, f"{cfg.db_path}: {exc}"))

    if conn is not None:
        checks.extend(check_sessions(cfg, conn, probes))
        checks.append(check_ohlc(cfg, conn, probes))
    checks.append(check_timer(cfg.ingest_timer, probes))
    checks.append(check_backup(cfg, probes))
    checks.append(check_timer(cfg.backup_timer, probes))
    checks.append(check_disk(cfg, probes))
    checks.extend(info_checks(cfg, conn, probes.now()))
    if conn is not None:
        conn.close()
    return checks


# ---------------------------------------------------------------------------
# State + message composition
# ---------------------------------------------------------------------------

def load_state(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("failing", {})
            return data
    except FileNotFoundError:
        pass
    except Exception as exc:
        log.warning("state file %s unreadable (%s) — starting fresh", path, exc)
    return {"version": STATE_VERSION, "failing": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".watchdog_state.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@dataclass
class Outcome:
    message: str | None
    new_failures: list[Check]
    reminders: list[Check]
    recoveries: list[tuple[Check, str]]   # (check, since-iso)
    events: list[Check]
    heartbeat: bool


def compose(cfg: Config, probes: Probes, state: dict, checks: list[Check],
            *, force_heartbeat: bool = False) -> Outcome:
    """Diff checks against state, update state, build the message (or None).

    ``state['failing']`` and ``state['last_heartbeat_date']`` are only
    advanced here; the caller must roll back ``last_alert``/heartbeat
    bookkeeping if the send fails (see :func:`main`).
    """
    now = probes.now()
    now_iso = now.isoformat(timespec="seconds")
    failing: dict = state.setdefault("failing", {})
    seen_keys = {c.key for c in checks if c.kind == "state"}

    new_failures: list[Check] = []
    reminders: list[Check] = []
    recoveries: list[tuple[Check, str]] = []
    events = [c for c in checks if c.kind == "event"]

    for c in checks:
        if c.kind != "state":
            continue
        entry = failing.get(c.key)
        if c.failing:
            if entry is None:
                failing[c.key] = {"since": now_iso, "last_alert": now_iso, "detail": c.detail}
                new_failures.append(c)
            else:
                entry["detail"] = c.detail
                last = datetime.fromisoformat(entry["last_alert"])
                if _age_h(now, last) >= cfg.realert_hours:
                    entry["last_alert"] = now_iso
                    reminders.append(c)
        elif entry is not None:
            recoveries.append((c, entry["since"]))
            del failing[c.key]

    # Per-day keys (sessions) that are no longer evaluated expire silently.
    for key in list(failing):
        if key not in seen_keys:
            del failing[key]

    heartbeat_due = force_heartbeat or (
        now.hour >= cfg.heartbeat_utc_hour
        and state.get("last_heartbeat_date") != now.strftime("%Y-%m-%d")
    )
    if heartbeat_due:
        state["last_heartbeat_date"] = now.strftime("%Y-%m-%d")

    if not (new_failures or reminders or recoveries or events or heartbeat_due):
        return Outcome(None, [], [], [], [], False)

    host = probes.hostname()
    lines: list[str] = []
    if new_failures or reminders or events:
        lines.append(f"🚨 NTS watchdog — {host} — {_fmt_ts(now)}")
        for c in new_failures:
            lines.append(f"NEW  {c.label}: {c.detail}")
        for c in reminders:
            since = failing[c.key]["since"]
            lines.append(f"STILL {c.label} (since {since[:16].replace('T', ' ')}): {c.detail}")
        for c in events:
            lines.append(f"EVENT {c.label}: {c.detail}")
        other = [k for k in failing if k not in {c.key for c in new_failures + reminders}]
        if other:
            lines.append(f"(still failing, reminder pending: {', '.join(sorted(other))})")
    if recoveries:
        if not lines:
            lines.append(f"✅ NTS watchdog — {host} — {_fmt_ts(now)}")
        for c, since in recoveries:
            down = _age_h(now, datetime.fromisoformat(since))
            lines.append(f"OK   {c.label} recovered after {down:.1f} h: {c.detail}")
    if heartbeat_due:
        if lines:
            lines.append("")
        lines.append(f"🫀 NTS watchdog status — {host} — {_fmt_ts(now)}")
        for c in checks:
            if c.kind == "event":
                continue
            lines.append(f"{c.mark} {c.label}: {c.detail}")
        n_fail = sum(1 for c in checks if c.failing)
        n_muted = sum(1 for c in checks if c.muted)
        summary = f"{n_fail} failing check(s)." if n_fail else "All checks passing."
        if n_muted:
            summary += f" {n_muted} muted (planned maintenance)."
        lines.append(summary)

    return Outcome("\n".join(lines), new_failures, reminders, recoveries, events, heartbeat_due)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None, *, probes: Probes | None = None,
         cfg: Config | None = None, sender=None) -> int:
    parser = argparse.ArgumentParser(description="NTS watchdog (absence alerting)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the would-be message; do not send or persist state")
    parser.add_argument("--heartbeat", action="store_true",
                        help="force the daily status block")
    parser.add_argument("--json", action="store_true", help="print check results as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)
    cfg = cfg or Config.from_env()
    probes = probes or Probes()
    sender = sender or send_plain_text

    state = load_state(cfg.state_path)
    checks = run_checks(cfg, probes, state)

    if args.json:
        print(json.dumps([c.__dict__ for c in checks], indent=1, default=str))

    if args.dry_run:
        # Compose against a throwaway copy so nothing is persisted.
        scratch = json.loads(json.dumps(state))
        outcome = compose(cfg, probes, scratch, checks, force_heartbeat=args.heartbeat)
        print(outcome.message or "(nothing to send)")
        return 0

    snapshot = json.loads(json.dumps(state))   # for rollback of alert bookkeeping
    outcome = compose(cfg, probes, state, checks, force_heartbeat=args.heartbeat)

    rc = 0
    if outcome.message:
        creds = credentials_from_env()
        if creds is None:
            log.error("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — cannot deliver:\n%s",
                      outcome.message)
            ok, detail = False, "no credentials"
        else:
            ok, detail = sender(creds[0], creds[1], outcome.message)
        if ok:
            log.info("sent %d line(s): new=%d reminders=%d recovered=%d events=%d heartbeat=%s",
                     outcome.message.count("\n") + 1, len(outcome.new_failures),
                     len(outcome.reminders), len(outcome.recoveries), len(outcome.events),
                     outcome.heartbeat)
        else:
            log.error("Telegram delivery failed (%s); message was:\n%s", detail, outcome.message)
            # Keep the probe bookkeeping (restart counter, gateway streak)
            # but undo alert/heartbeat bookkeeping so the next run retries.
            state["failing"] = snapshot.get("failing", {})
            state["last_heartbeat_date"] = snapshot.get("last_heartbeat_date")
            # Failures that are new this run must stay "new" next run.
            for c in outcome.new_failures:
                state["failing"].pop(c.key, None)
            rc = 2
    else:
        log.info("nothing to report (%d checks, %d failing)",
                 len(checks), sum(1 for c in checks if c.failing))

    state["version"] = STATE_VERSION
    state["last_run"] = probes.now().isoformat(timespec="seconds")
    save_state(cfg.state_path, state)
    return rc


if __name__ == "__main__":
    sys.exit(main())
