"""Bar-freshness gate (2026-09-23): no signals on a stale daily bar.

Covers scheduler/bar_freshness.py (expected date, verdicts), the
DailyScheduler hook (skip / abort / Telegram / session_runs.note / fail-open)
and the watchdog's reading of session_runs.note.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scheduler import bar_freshness as bf  # noqa: E402
from scheduler.daily_runner import DailyScheduler  # noqa: E402


# ── expected_bar_date ──────────────────────────────────────────────────────

class TestExpectedBarDate:
    @pytest.mark.parametrize("session, now, expected", [
        # Tue 2026-09-08 intraday → Fri 09-04 (Mon 09-07 is Labor Day)
        ("US_PRE", datetime(2026, 9, 8, 13, 15, tzinfo=timezone.utc), date(2026, 9, 4)),
        ("US_OPEN", datetime(2026, 9, 8, 14, 30, tzinfo=timezone.utc), date(2026, 9, 4)),
        # Wed intraday → Tue
        ("US_OPEN", datetime(2026, 9, 9, 14, 30, tzinfo=timezone.utc), date(2026, 9, 8)),
        # EOD after the ingest → today
        ("EOD", datetime(2026, 9, 9, 22, 45, tzinfo=timezone.utc), date(2026, 9, 9)),
        # EOD on Labor Day (daemon runs on holidays) → previous Friday
        ("EOD", datetime(2026, 9, 7, 22, 45, tzinfo=timezone.utc), date(2026, 9, 4)),
        # Monday intraday → Friday
        ("PEAD_OPEN", datetime(2026, 9, 14, 13, 45, tzinfo=timezone.utc), date(2026, 9, 11)),
    ])
    def test_expected(self, session, now, expected):
        assert bf.expected_bar_date(session, now) == expected


# ── check_bar_freshness verdicts ───────────────────────────────────────────

class _FakeDB:
    def __init__(self, max_dates: dict, *, fail: bool = False):
        self._max = max_dates
        self._fail = fail

    def get_daily_ohlc_max_dates(self, tickers):
        if self._fail:
            raise sqlite3.OperationalError("locked")
        return {t: self._max.get(t) for t in tickers}


_NOW = datetime(2026, 9, 9, 14, 30, tzinfo=timezone.utc)   # expects 2026-09-08


class TestVerdicts:
    def test_ok_when_every_store_ticker_is_current(self):
        db = _FakeDB({"AAPL": "2026-09-08", "MSFT": "2026-09-08", "ZZZ": None})
        r = bf.check_bar_freshness(db, "US_OPEN", ["AAPL", "MSFT", "ZZZ"], now=_NOW)
        assert r.verdict == "ok"
        assert r.checked == ["AAPL", "MSFT"]
        assert r.unknown == ["ZZZ"]           # not in store → not gated
        assert "ok" in r.summary() and "1 not in store" in r.summary()

    def test_skip_when_some_are_behind(self):
        db = _FakeDB({"AAPL": "2026-09-08", "MSFT": "2026-09-04"})
        r = bf.check_bar_freshness(db, "US_OPEN", ["AAPL", "MSFT"], now=_NOW)
        assert r.verdict == "skip"
        assert r.stale == {"MSFT": "2026-09-04"}
        assert r.fresh == ["AAPL"]
        assert "STALE BARS" in r.summary() and "SKIP" in r.summary()

    def test_abort_when_all_are_behind(self):
        db = _FakeDB({"AAPL": "2026-09-04", "MSFT": "2026-09-04", "ZZZ": None})
        r = bf.check_bar_freshness(db, "US_OPEN", ["AAPL", "MSFT", "ZZZ"], now=_NOW)
        assert r.verdict == "abort"
        assert "ABORT" in r.summary()

    def test_eod_expects_todays_bar(self):
        eod_now = datetime(2026, 9, 9, 22, 45, tzinfo=timezone.utc)
        db = _FakeDB({"AAPL": "2026-09-08"})     # ingest did not land
        r = bf.check_bar_freshness(db, "EOD", ["AAPL"], now=eod_now)
        assert r.expected == date(2026, 9, 9)
        assert r.verdict == "abort"

    def test_store_read_failure_fails_open(self):
        r = bf.check_bar_freshness(_FakeDB({}, fail=True), "US_OPEN", ["AAPL"], now=_NOW)
        assert r.verdict == "ok" and r.checked == []

    def test_no_tickers(self):
        assert bf.check_bar_freshness(_FakeDB({}), "US_OPEN", [], now=_NOW).verdict == "ok"


# ── DailyScheduler hook ────────────────────────────────────────────────────

def _run(name="US_OPEN", session_type="execution"):
    return {"name": name, "hour": 14, "minute": 30, "tickers": None,
            "workers": 3, "eod": False, "session_type": session_type}


class TestSchedulerHook:
    def _execute(self, sched, run, tickers, max_dates, *, gate_enabled=True):
        captured = {}

        async def fake_run_batch(tickers, **kwargs):
            captured["tickers"] = list(tickers)
            return {"results": [], "success_count": 0, "fail_count": 0, "elapsed_s": 0.0}

        fake_db = _FakeDB(max_dates)
        with patch("scheduler.daily_runner.run_batch", side_effect=fake_run_batch), \
             patch("scheduler.daily_runner._is_execution_allowed", return_value=(False, "test")), \
             patch.object(DailyScheduler, "_load_us_tickers", return_value=list(tickers)), \
             patch.object(DailyScheduler, "_load_scanner_candidate_tickers", return_value=set()), \
             patch.object(DailyScheduler, "_claim_session_slot", return_value=True), \
             patch.object(DailyScheduler, "_annotate_session_slot") as annotate, \
             patch.object(DailyScheduler, "_fetch_macro_context", return_value=""), \
             patch.object(DailyScheduler, "_fetch_session_account_balance", return_value=10_000.0), \
             patch("storage.database.Database", return_value=fake_db), \
             patch("config.settings.BAR_FRESHNESS_GATE_ENABLED", gate_enabled), \
             patch("config.settings.BAR_FRESHNESS_EOD_WAIT_S", 0), \
             patch("scheduler.bar_freshness.datetime") as fake_dt:
            fake_dt.now.return_value = _NOW
            sched._execute_run(run)
        return captured.get("tickers"), annotate

    @pytest.fixture
    def sched(self):
        s = DailyScheduler(full_watchlist=["AAPL", "MSFT"])
        s._tg = MagicMock()
        s._position_manager_trader = None
        return s

    def test_fresh_store_runs_everything(self, sched):
        ran, annotate = self._execute(sched, _run(), ["AAPL", "MSFT"],
                                      {"AAPL": "2026-09-08", "MSFT": "2026-09-08"})
        assert ran == ["AAPL", "MSFT"]
        annotate.assert_not_called()

    def test_partial_staleness_skips_those_tickers_and_alerts(self, sched):
        ran, annotate = self._execute(sched, _run(), ["AAPL", "MSFT"],
                                      {"AAPL": "2026-09-08", "MSFT": "2026-09-04"})
        assert ran == ["AAPL"]
        annotate.assert_called_once()
        note = annotate.call_args[0][1]
        assert note.startswith("SKIPPED 1 ticker(s)") and "MSFT" in note
        sent = " ".join(str(c) for c in sched._tg._send.call_args_list)
        assert "stale daily bars" in sent and "MSFT=2026-09-04" in sent

    def test_fully_stale_store_aborts_session(self, sched):
        ran, annotate = self._execute(sched, _run(), ["AAPL", "MSFT"],
                                      {"AAPL": "2026-09-04", "MSFT": "2026-09-04"})
        assert ran is None                       # run_batch never called
        note = annotate.call_args[0][1]
        assert note.startswith("ABORTED: stale daily bars")
        sent = " ".join(str(c) for c in sched._tg._send.call_args_list)
        assert "ABORTED" in sent and "2026-09-08" in sent

    def test_monitor_session_is_not_gated(self, sched):
        ran, annotate = self._execute(sched, _run("MIDDAY", "monitor"), ["AAPL"],
                                      {"AAPL": "2026-01-01"})
        assert ran == ["AAPL"]
        annotate.assert_not_called()

    def test_gate_can_be_disabled(self, sched):
        ran, _ = self._execute(sched, _run(), ["AAPL"], {"AAPL": "2026-01-01"},
                               gate_enabled=False)
        assert ran == ["AAPL"]

    def test_gate_crash_fails_open(self, sched):
        with patch("scheduler.bar_freshness.check_bar_freshness", side_effect=RuntimeError("boom")), \
             patch("config.settings.BAR_FRESHNESS_GATE_ENABLED", True), \
             patch("storage.database.Database"):
            assert sched._apply_bar_freshness_gate("US_OPEN", "execution", ["AAPL"]) == ["AAPL"]


# ── session_runs.note round trip + watchdog ────────────────────────────────

class TestSessionNote:
    def test_annotate_adds_column_and_is_idempotent(self, tmp_path):
        db_path = tmp_path / "t.db"
        with patch("storage.database._resolve_db_path", return_value=str(db_path)):
            assert DailyScheduler._claim_session_slot("US_OPEN") is True
            DailyScheduler._annotate_session_slot("US_OPEN", "ABORTED: stale daily bars")
            DailyScheduler._annotate_session_slot("US_OPEN", "ABORTED: stale daily bars (again)")
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT session, note FROM session_runs").fetchone()
        assert row == ("US_OPEN", "ABORTED: stale daily bars (again)")

    def test_watchdog_reports_aborted_session_as_failing(self, tmp_path):
        from scripts import watchdog as wd
        db_path = tmp_path / "w.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            "CREATE TABLE session_runs(session TEXT NOT NULL, run_date TEXT NOT NULL,"
            " started_at TEXT NOT NULL, runner_id TEXT, note TEXT, PRIMARY KEY(session, run_date));"
        )
        conn.execute("INSERT INTO session_runs VALUES (?,?,?,?,?)",
                     ("US_OPEN", "2026-09-09", "2026-09-09T14:30:01+00:00", "r1",
                      "ABORTED: stale daily bars (expected 2026-09-08)"))
        conn.execute("INSERT INTO session_runs VALUES (?,?,?,?,?)",
                     ("US_PRE", "2026-09-09", "2026-09-09T13:15:01+00:00", "r1",
                      "SKIPPED 1 ticker(s) on stale bars: MSFT"))
        conn.commit()
        cfg = wd.Config(db_path=db_path, state_path=tmp_path / "s.json")
        probes = wd.Probes(
            now=lambda: datetime(2026, 9, 9, 15, 0, tzinfo=timezone.utc),
            systemctl_show=lambda u: {}, tcp_open=lambda h, p: True,
            disk_free_gb=lambda p: 40.0, hostname=lambda: "claw",
        )
        checks = {c.key: c for c in wd.check_sessions(cfg, conn, probes)}
        assert checks["session:2026-09-09:US_OPEN"].ok is False
        assert "ABORTED" in checks["session:2026-09-09:US_OPEN"].detail
        assert checks["session:2026-09-09:US_PRE"].ok is True
        assert "SKIPPED" in checks["session:2026-09-09:US_PRE"].detail

    def test_watchdog_without_note_column_still_works(self, tmp_path):
        from scripts import watchdog as wd
        db_path = tmp_path / "w.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            "CREATE TABLE session_runs(session TEXT NOT NULL, run_date TEXT NOT NULL,"
            " started_at TEXT NOT NULL, runner_id TEXT, PRIMARY KEY(session, run_date));"
        )
        conn.execute("INSERT INTO session_runs VALUES (?,?,?,?)",
                     ("US_OPEN", "2026-09-09", "2026-09-09T14:30:01+00:00", "r1"))
        conn.commit()
        cfg = wd.Config(db_path=db_path, state_path=tmp_path / "s.json")
        probes = wd.Probes(
            now=lambda: datetime(2026, 9, 9, 15, 0, tzinfo=timezone.utc),
            systemctl_show=lambda u: {}, tcp_open=lambda h, p: True,
            disk_free_gb=lambda p: 40.0, hostname=lambda: "claw",
        )
        checks = {c.key: c for c in wd.check_sessions(cfg, conn, probes)}
        assert checks["session:2026-09-09:US_OPEN"].ok is True

    def test_watchdog_schedule_knows_the_new_eod_time(self):
        from scripts import watchdog as wd
        eod = [e for e in wd._SCHEDULE if e["name"] == "EOD"][0]
        assert (eod["hour"], eod["minute"]) == (22, 45)
        cfg = wd.Config(db_path=Path("x"), state_path=Path("s"))
        # 23:00 UTC: EOD (22:45 + 20 min grace) is NOT due yet → no false alarm
        due = {n for n, _ in wd._due_sessions(cfg, datetime(2026, 9, 9, 23, 0, tzinfo=timezone.utc))}
        assert "EOD" not in due
        due = {n for n, _ in wd._due_sessions(cfg, datetime(2026, 9, 9, 23, 6, tzinfo=timezone.utc))}
        assert "EOD" in due
