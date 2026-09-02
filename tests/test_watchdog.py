"""
Tests for scripts/watchdog.py — absence alerting, dedupe, recovery, heartbeat.

All probes (systemd, TCP, disk, clock, hostname) are injected; the DB is
a temp SQLite file; the sender is a recording stub. No network, no systemd.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts import watchdog as wd  # noqa: E402


# ── helpers ─────────────────────────────────────────────────────────────────

TUE_1030 = datetime(2026, 9, 1, 10, 30, tzinfo=timezone.utc)   # Tue, after XETRA sessions


def _make_db(path: Path, *, sessions=(), max_ohlc="2026-08-31", halted=0, positions=2):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE session_runs(session TEXT NOT NULL, run_date TEXT NOT NULL,
            started_at TEXT NOT NULL, runner_id TEXT, PRIMARY KEY(session, run_date));
        CREATE TABLE daily_ohlc(ticker TEXT, date TEXT);
        CREATE TABLE portfolio_peak(id INTEGER PRIMARY KEY, peak_value REAL,
            halted INTEGER NOT NULL DEFAULT 0, halted_at TEXT, halted_drawdown_pct REAL);
        CREATE TABLE portfolio_positions(ticker TEXT, shares REAL);
        """
    )
    for name, run_date in sessions:
        conn.execute("INSERT INTO session_runs VALUES (?,?,?,?)",
                     (name, run_date, f"{run_date}T07:00:05+00:00", "r1"))
    if max_ohlc:
        for t in ("AAPL", "MSFT"):
            conn.execute("INSERT INTO daily_ohlc VALUES (?,?)", (t, max_ohlc))
    conn.execute("INSERT INTO portfolio_peak(peak_value, halted, halted_at, halted_drawdown_pct) "
                 "VALUES (100000, ?, ?, ?)", (halted, "2026-08-20T15:00:00" if halted else None,
                                             0.12 if halted else None))
    for i in range(positions):
        conn.execute("INSERT INTO portfolio_positions VALUES (?, 10)", (f"T{i}",))
    conn.commit()
    conn.close()


class FakeSystemd:
    def __init__(self):
        self.units = {
            "nts-trading.service": {
                "LoadState": "loaded", "ActiveState": "active", "SubState": "running",
                "Result": "success", "NRestarts": "0",
                "ActiveEnterTimestamp": "Mon 2026-08-31 14:02:11 UTC", "ExecMainStatus": "0",
            },
            "nts-ohlc-ingest.timer": {"LoadState": "loaded", "ActiveState": "active"},
            "nts-backup.timer": {"LoadState": "loaded", "ActiveState": "active"},
        }
        self.raise_for: set[str] = set()

    def __call__(self, unit):
        if unit in self.raise_for:
            raise RuntimeError("systemctl unavailable")
        return dict(self.units[unit])


class Harness:
    """One watchdog environment: temp DB, temp state, fake probes, recording sender."""

    def __init__(self, tmp_path: Path, *, now=TUE_1030, **db_kwargs):
        self.now = now
        self.systemd = FakeSystemd()
        self.tcp_ok = True
        self.free_gb = 40.0
        self.sent: list[str] = []
        self.send_ok = True
        self.db = tmp_path / "t.db"
        _make_db(self.db, **db_kwargs)
        self.backup_dir = tmp_path / "backups"
        self.backup_dir.mkdir()
        self.touch_backup(age_h=5)
        self.cfg = wd.Config(
            db_path=self.db, state_path=tmp_path / "state.json", repo_dir=tmp_path,
            backup_dir=self.backup_dir, realert_hours=6, heartbeat_utc_hour=6,
        )
        self.probes = wd.Probes(
            now=lambda: self.now,
            systemctl_show=self.systemd,
            tcp_open=lambda h, p: self.tcp_ok,
            disk_free_gb=lambda p: self.free_gb,
            hostname=lambda: "claw",
        )

    def touch_backup(self, *, age_h: float):
        f = self.backup_dir / f"news_trading_{int(age_h)}.db"
        f.write_bytes(b"x")
        ts = (self.now - timedelta(hours=age_h)).timestamp()
        os.utime(f, (ts, ts))

    def sender(self, token, chat_id, text):
        self.sent.append(text)
        return (True, "HTTP 200") if self.send_ok else (False, "HTTP 502")

    def run(self, *args) -> int:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("TELEGRAM_BOT_TOKEN", "t")
            mp.setenv("TELEGRAM_CHAT_ID", "c")
            return wd.main(list(args), probes=self.probes, cfg=self.cfg, sender=self.sender)

    def state(self) -> dict:
        return json.loads(self.cfg.state_path.read_text())

    def advance(self, **kw):
        self.now = self.now + timedelta(**kw)


# ── all green ───────────────────────────────────────────────────────────────

class TestQuiet:
    def test_healthy_system_after_heartbeat_sends_nothing(self, tmp_path):
        h = Harness(tmp_path, sessions=[("XETRA_PRE", "2026-09-01"), ("XETRA_OPEN", "2026-09-01")])
        # first run of the day → heartbeat only
        assert h.run() == 0
        assert len(h.sent) == 1
        assert h.sent[0].startswith("🫀 NTS watchdog status — claw")
        assert "All checks passing." in h.sent[0]
        assert "🚨" not in h.sent[0]
        # second run same day → silence
        h.advance(minutes=15)
        assert h.run() == 0
        assert len(h.sent) == 1
        assert h.state()["failing"] == {}
        assert h.state()["last_heartbeat_date"] == "2026-09-01"

    def test_heartbeat_not_before_configured_hour(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 5, 45, tzinfo=timezone.utc))
        assert h.run() == 0
        assert h.sent == []
        h.now = datetime(2026, 9, 1, 6, 0, tzinfo=timezone.utc)
        assert h.run() == 0
        assert len(h.sent) == 1 and "🫀" in h.sent[0]

    def test_forced_heartbeat_lists_info_lines(self, tmp_path):
        h = Harness(tmp_path, halted=1, positions=3)
        (tmp_path / "emergency_stop.flag").write_text(
            json.dumps({"action": "stop_trading", "activated_at": "2026-08-30T10:00:00+00:00"}))
        h.now = datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc)
        assert h.run("--heartbeat") == 0
        text = h.sent[0]
        assert "· kill switch: ACTIVE — stop_trading since 2026-08-30" in text
        assert "· drawdown halt: HALTED since 2026-08-20" in text
        assert "· open positions: 3" in text
        assert "✓ daemon: nts-trading.service up" in text
        assert "✓ IB Gateway: 127.0.0.1:4002 accepts connections" in text


# ── daemon ──────────────────────────────────────────────────────────────────

class TestDaemon:
    def test_inactive_daemon_alerts_once_then_reminds_after_realert(self, tmp_path):
        h = Harness(tmp_path, sessions=[("XETRA_PRE", "2026-09-01"), ("XETRA_OPEN", "2026-09-01")])
        h.run()  # heartbeat
        h.systemd.units["nts-trading.service"].update(
            ActiveState="failed", SubState="failed", Result="exit-code", ExecMainStatus="1")
        h.advance(minutes=15)
        assert h.run() == 0
        assert len(h.sent) == 2
        assert "🚨 NTS watchdog — claw" in h.sent[1]
        assert "NEW  daemon: nts-trading.service ActiveState=failed" in h.sent[1]
        # still failing 15 min later → no repeat
        h.advance(minutes=15)
        h.run()
        assert len(h.sent) == 2
        # ... but after the re-alert interval → reminder
        h.advance(hours=6)
        h.run()
        assert len(h.sent) == 3
        assert "STILL daemon (since 2026-09-01 10:45)" in h.sent[2]

    def test_recovery_message_once(self, tmp_path):
        h = Harness(tmp_path, sessions=[("XETRA_PRE", "2026-09-01"), ("XETRA_OPEN", "2026-09-01")])
        h.run()
        h.systemd.units["nts-trading.service"]["ActiveState"] = "inactive"
        h.advance(minutes=15); h.run()
        h.systemd.units["nts-trading.service"]["ActiveState"] = "active"
        h.advance(minutes=30); h.run()
        assert h.sent[-1].startswith("✅ NTS watchdog — claw")
        assert "OK   daemon recovered after 0.5 h" in h.sent[-1]
        h.advance(minutes=15); h.run()
        assert len(h.sent) == 3
        assert h.state()["failing"] == {}

    def test_auto_restart_is_reported_as_event_every_time(self, tmp_path):
        h = Harness(tmp_path, sessions=[("XETRA_PRE", "2026-09-01"), ("XETRA_OPEN", "2026-09-01")])
        h.run()
        assert h.state()["nrestarts"] == 0
        h.systemd.units["nts-trading.service"]["NRestarts"] = "2"
        h.advance(minutes=15); h.run()
        assert "EVENT daemon restart: nts-trading.service auto-restarted 2× since the last check" in h.sent[-1]
        h.advance(minutes=15); h.run()
        assert len(h.sent) == 2          # no restart since → nothing
        h.systemd.units["nts-trading.service"]["NRestarts"] = "3"
        h.advance(minutes=15); h.run()
        assert "auto-restarted 1×" in h.sent[-1]

    def test_systemctl_failure_is_loud_not_silent(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc))
        h.systemd.raise_for = {"nts-trading.service"}
        assert h.run() == 0
        assert "NEW  daemon: cannot query systemd for nts-trading.service" in h.sent[0]

    def test_missing_unit_fails(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc))
        h.systemd.units["nts-trading.service"]["LoadState"] = "not-found"
        h.run()
        assert "LoadState=not-found" in h.sent[0]


# ── sessions ────────────────────────────────────────────────────────────────

class TestSessions:
    def test_missing_session_after_grace(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 7, 25, tzinfo=timezone.utc),
                    sessions=[("XETRA_PRE", "2026-09-01")])
        h.run()
        text = h.sent[0]
        assert "NEW  session XETRA_OPEN: no session_runs row by 07:25 UTC (scheduled 07:00 UTC + 20 min grace)" in text
        assert "session XETRA_PRE: ran" in text
        assert "session:2026-09-01:XETRA_OPEN" in h.state()["failing"]

    def test_not_due_before_grace(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 7, 10, tzinfo=timezone.utc),
                    sessions=[("XETRA_PRE", "2026-09-01")])
        h.run()
        assert "XETRA_OPEN" not in h.sent[0]

    def test_late_session_recovers(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 7, 25, tzinfo=timezone.utc),
                    sessions=[("XETRA_PRE", "2026-09-01")])
        h.run()
        conn = sqlite3.connect(h.db)
        conn.execute("INSERT INTO session_runs VALUES ('XETRA_OPEN','2026-09-01','2026-09-01T07:31:00','r1')")
        conn.commit(); conn.close()
        h.advance(minutes=15); h.run()
        assert "OK   session XETRA_OPEN recovered" in h.sent[-1]

    def test_per_day_keys_expire_silently_at_midnight(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 23, 0, tzinfo=timezone.utc), sessions=[])
        h.run()  # heartbeat + 8 missing sessions
        assert sum(1 for k in h.state()["failing"] if k.startswith("session:")) == 8
        h.now = datetime(2026, 9, 2, 0, 5, tzinfo=timezone.utc)
        h.run()
        assert not any(k.startswith("session:") for k in h.state()["failing"])
        assert "recovered" not in h.sent[-1]     # expiry is not a recovery
        assert len(h.sent) == 1                  # nothing sent at 00:05

    def test_no_session_expectations_on_weekend(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc), sessions=[])
        h.run()
        assert "session " not in h.sent[0]
        assert not any(k.startswith("session:") for k in h.state()["failing"])

    def test_pre_session_skipped_when_disabled(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 7, 25, tzinfo=timezone.utc),
                    sessions=[("XETRA_OPEN", "2026-09-01")])
        h.cfg.pre_sessions_enabled = False
        h.run()
        assert "XETRA_PRE" not in h.sent[0]
        assert "0 failing" not in h.sent[0] or "All checks passing." in h.sent[0]

    def test_missing_table_is_reported(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 7, 25, tzinfo=timezone.utc))
        conn = sqlite3.connect(h.db); conn.execute("DROP TABLE session_runs"); conn.commit(); conn.close()
        h.run()
        assert "session_runs unreadable" in h.sent[0]


# ── OHLC ingest ─────────────────────────────────────────────────────────────

class TestOhlc:
    @pytest.mark.parametrize("now, expected", [
        (datetime(2026, 9, 1, 23, 30, tzinfo=timezone.utc), date(2026, 9, 1)),   # after ingest same day
        (datetime(2026, 9, 1, 22, 45, tzinfo=timezone.utc), date(2026, 8, 31)),  # ingest running
        (datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc), date(2026, 9, 1)),    # next morning
        (datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc), date(2026, 9, 4)),    # Saturday → Friday
        (datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc), date(2026, 9, 4)),    # Tue after Labor Day
    ])
    def test_expected_date(self, tmp_path, now, expected):
        cfg = wd.Config(db_path=tmp_path / "x", state_path=tmp_path / "s")
        assert wd.expected_ohlc_date(cfg, now) == expected

    def test_stale_store_alerts(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc),
                    max_ohlc="2026-08-31", sessions=[("XETRA_PRE", "2026-09-02"), ("XETRA_OPEN", "2026-09-02")])
        h.run()
        assert "NEW  OHLC ingest: STALE — MAX(date)=2026-08-31 (2 tickers), expected ≥ 2026-09-01" in h.sent[0]

    def test_fresh_store_passes(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc),
                    max_ohlc="2026-09-01", sessions=[("XETRA_PRE", "2026-09-02"), ("XETRA_OPEN", "2026-09-02")])
        h.run()
        assert "✓ OHLC ingest: MAX(date)=2026-09-01" in h.sent[0]
        assert "All checks passing." in h.sent[0]

    def test_empty_store_alerts(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc), max_ohlc=None)
        h.run()
        assert "OHLC ingest: daily_ohlc is empty" in h.sent[0]

    def test_disabled_timer_alerts(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc))
        h.systemd.units["nts-ohlc-ingest.timer"]["ActiveState"] = "inactive"
        h.run()
        assert "NEW  nts-ohlc-ingest.timer: ActiveState=inactive — timer not running" in h.sent[0]


# ── backup / gateway / disk / db ────────────────────────────────────────────

class TestOtherChecks:
    def test_old_backup_alerts(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc))
        for f in h.backup_dir.iterdir():
            f.unlink()
        h.touch_backup(age_h=30)
        h.run()
        assert "NEW  DB backup: STALE — news_trading_30.db written" in h.sent[0]
        assert "(30.0 h ago), limit 26 h" in h.sent[0]

    def test_missing_backup_dir_alerts(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc))
        h.cfg.backup_dir = tmp_path / "nope"
        h.run()
        assert "backup dir" in h.sent[0] and "does not exist" in h.sent[0]

    def test_gateway_needs_two_misses(self, tmp_path):
        h = Harness(tmp_path, sessions=[("XETRA_PRE", "2026-09-01"), ("XETRA_OPEN", "2026-09-01")])
        h.run()
        h.tcp_ok = False
        h.advance(minutes=15); h.run()
        assert len(h.sent) == 1                       # one miss tolerated
        assert h.state()["gateway_fail_streak"] == 1
        h.advance(minutes=15); h.run()
        assert "NEW  IB Gateway: 127.0.0.1:4002 unreachable for 2 consecutive checks" in h.sent[-1]
        h.tcp_ok = True
        h.advance(minutes=15); h.run()
        assert "OK   IB Gateway recovered" in h.sent[-1]
        assert h.state()["gateway_fail_streak"] == 0

    def test_low_disk_alerts(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc))
        h.free_gb = 0.4
        h.run()
        assert "NEW  disk: 0.4 GB free on DB volume (min 1.0 GB)" in h.sent[0]

    def test_missing_db_alerts_and_skips_db_checks(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 7, 25, tzinfo=timezone.utc))
        h.db.unlink()
        h.run()
        assert "NEW  database:" in h.sent[0] and "does not exist" in h.sent[0]
        assert "session" not in h.sent[0]
        assert "OHLC" not in h.sent[0]


# ── delivery + state robustness ─────────────────────────────────────────────

class TestDelivery:
    def test_failed_send_returns_2_and_retries_next_run(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc))
        h.systemd.units["nts-trading.service"]["ActiveState"] = "inactive"
        h.send_ok = False
        assert h.run("--heartbeat") == 2
        # alert bookkeeping rolled back: failure is still "new", heartbeat still due
        assert "daemon" not in h.state()["failing"]
        assert h.state().get("last_heartbeat_date") is None
        h.send_ok = True
        h.advance(minutes=15)
        assert h.run("--heartbeat") == 0
        assert "NEW  daemon" in h.sent[-1]
        assert "daemon" in h.state()["failing"]

    def test_missing_credentials_returns_2(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc))
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("TELEGRAM_BOT_TOKEN", raising=False)
            mp.delenv("TELEGRAM_CHAT_ID", raising=False)
            rc = wd.main(["--heartbeat"], probes=h.probes, cfg=h.cfg, sender=h.sender)
        assert rc == 2
        assert h.sent == []

    def test_dry_run_sends_nothing_and_writes_no_state(self, tmp_path, capsys):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc))
        assert h.run("--dry-run", "--heartbeat") == 0
        assert h.sent == []
        assert not h.cfg.state_path.exists()
        assert "🫀 NTS watchdog status" in capsys.readouterr().out

    def test_corrupt_state_file_starts_fresh(self, tmp_path):
        h = Harness(tmp_path, now=datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc))
        h.cfg.state_path.write_text("{not json")
        assert h.run() == 0
        assert h.state()["version"] == wd.STATE_VERSION

    def test_one_message_per_run_combines_sections(self, tmp_path):
        h = Harness(tmp_path, sessions=[("XETRA_PRE", "2026-09-01"), ("XETRA_OPEN", "2026-09-01")])
        h.run()
        h.free_gb = 0.2
        h.systemd.units["nts-trading.service"]["NRestarts"] = "1"
        h.advance(minutes=15); h.run()
        msg = h.sent[-1]
        assert msg.count("🚨 NTS watchdog") == 1
        assert "NEW  disk" in msg and "EVENT daemon restart" in msg


class TestConfigFromEnv:
    def test_defaults_and_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DB_PATH", str(tmp_path / "db.sqlite"))
        monkeypatch.delenv("NTS_WATCHDOG_DB", raising=False)
        monkeypatch.delenv("NTS_WATCHDOG_STATE", raising=False)
        monkeypatch.setenv("IBKR_PAPER", "false")
        monkeypatch.setenv("ENABLE_PRE_SESSIONS", "false")
        monkeypatch.setenv("NTS_WATCHDOG_REALERT_HOURS", "2.5")
        cfg = wd.Config.from_env()
        assert cfg.db_path == tmp_path / "db.sqlite"
        assert cfg.state_path == tmp_path / "watchdog_state.json"
        assert cfg.ibkr_port == 4001
        assert cfg.pre_sessions_enabled is False
        assert cfg.realert_hours == 2.5

    def test_relative_db_path_resolves_against_repo(self, monkeypatch):
        monkeypatch.setenv("DB_PATH", "news_trading.db")
        monkeypatch.delenv("NTS_WATCHDOG_DB", raising=False)
        cfg = wd.Config.from_env()
        assert cfg.db_path == (wd.REPO_DIR / "news_trading.db").resolve()

    def test_systemd_timestamp_parsing(self):
        ts = wd._parse_systemd_ts("Mon 2026-08-31 14:02:11 UTC")
        assert ts == datetime(2026, 8, 31, 14, 2, 11, tzinfo=timezone.utc)
        assert wd._parse_systemd_ts("") is None
        assert wd._parse_systemd_ts("n/a") is None
