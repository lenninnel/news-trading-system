"""
Tests for the DailyScheduler daemon.

Covers: run time calculation, weekend skipping, session detection,
and EOD summary formatting.
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scheduler.daily_runner import (
    SCHEDULE,
    DailyScheduler,
    send_eod_summary,
)


@pytest.fixture
def scheduler():
    return DailyScheduler(full_watchlist=["AAPL", "MSFT", "META"])


# ── next_run_time ─────────────────────────────────────────────────────


class TestNextRunTime:
    def test_before_first_run_returns_xetra_pre(self, scheduler):
        """Monday 05:00 UTC → next run is XETRA_PRE at 06:45."""
        monday_5am = datetime(2026, 3, 16, 5, 0, 0, tzinfo=timezone.utc)
        nrt = scheduler.next_run_time(after=monday_5am)
        assert nrt.hour == 6
        assert nrt.minute == 45
        assert nrt.day == 16

    def test_between_runs_returns_next(self, scheduler):
        """Monday 10:00 UTC → next run is PREMARKET_SCAN at 13:00."""
        monday_10am = datetime(2026, 3, 16, 10, 0, 0, tzinfo=timezone.utc)
        nrt = scheduler.next_run_time(after=monday_10am)
        assert nrt.hour == 13
        assert nrt.minute == 0

    def test_after_midday_returns_eod(self, scheduler):
        """Monday 20:00 UTC → next run is EOD at 22:15."""
        monday_8pm = datetime(2026, 3, 16, 20, 0, 0, tzinfo=timezone.utc)
        nrt = scheduler.next_run_time(after=monday_8pm)
        assert nrt.hour == 22
        assert nrt.minute == 15

    def test_after_last_run_returns_next_day(self, scheduler):
        """Monday 22:30 UTC → next run is Tuesday XETRA_PRE 06:45."""
        monday_late = datetime(2026, 3, 16, 22, 30, 0, tzinfo=timezone.utc)
        nrt = scheduler.next_run_time(after=monday_late)
        assert nrt.weekday() == 1  # Tuesday
        assert nrt.hour == 6
        assert nrt.minute == 45

    def test_all_eight_runs_fire_on_weekday(self, scheduler):
        """Starting from Monday 00:00, iterating should hit all 8 run times
        (7 session runs + the D9 PREMARKET_SCAN at 13:00)."""
        t = datetime(2026, 3, 16, 0, 0, 0, tzinfo=timezone.utc)
        run_hours = []
        for _ in range(8):
            t = scheduler.next_run_time(after=t)
            run_hours.append((t.hour, t.minute))
            t = t.replace(second=1)
        assert run_hours == [
            (6, 45), (7, 0), (13, 0), (13, 15), (13, 45),
            (14, 30), (18, 0), (22, 15),
        ]


# ── Weekend skip ──────────────────────────────────────────────────────


class TestWeekendSkip:
    def test_friday_after_eod_runs_sunday_sector_correlation(self, scheduler):
        """Friday 23:00 UTC → next run is the Sunday sector-correlation job at 06:00."""
        # 2026-03-20 is a Friday
        friday_late = datetime(2026, 3, 20, 23, 0, 0, tzinfo=timezone.utc)
        nrt = scheduler.next_run_time(after=friday_late)
        assert nrt.weekday() == 6  # Sunday
        assert nrt.day == 22
        assert nrt.hour == 6
        assert nrt.minute == 0

    def test_saturday_runs_sunday_sector_correlation(self, scheduler):
        """Saturday 12:00 UTC → next run is the Sunday sector-correlation job at 06:00."""
        saturday = datetime(2026, 3, 21, 12, 0, 0, tzinfo=timezone.utc)
        nrt = scheduler.next_run_time(after=saturday)
        assert nrt.weekday() == 6  # Sunday
        assert nrt.hour == 6
        assert nrt.minute == 0

    def test_sunday_after_weekly_skips_to_monday(self, scheduler):
        """Sunday 18:00 UTC → next run is Monday XETRA_PRE at 06:45."""
        sunday = datetime(2026, 3, 22, 18, 0, 0, tzinfo=timezone.utc)
        nrt = scheduler.next_run_time(after=sunday)
        assert nrt.weekday() == 0
        assert nrt.hour == 6
        assert nrt.minute == 45


# ── current_session ───────────────────────────────────────────────────


class TestCurrentSession:
    def _session_at(self, scheduler, dt):
        with patch("scheduler.daily_runner.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            return scheduler.current_session()

    def test_before_trading_is_closed(self, scheduler):
        dt = datetime(2026, 3, 16, 5, 0, 0, tzinfo=timezone.utc)
        assert self._session_at(scheduler, dt) == "CLOSED"

    def test_xetra_pre_session(self, scheduler):
        dt = datetime(2026, 3, 16, 6, 50, 0, tzinfo=timezone.utc)
        assert self._session_at(scheduler, dt) == "XETRA_PRE"

    def test_xetra_session(self, scheduler):
        dt = datetime(2026, 3, 16, 8, 0, 0, tzinfo=timezone.utc)
        assert self._session_at(scheduler, dt) == "XETRA_OPEN"

    def test_us_pre_session(self, scheduler):
        dt = datetime(2026, 3, 16, 13, 30, 0, tzinfo=timezone.utc)
        assert self._session_at(scheduler, dt) == "US_PRE"

    def test_us_open_session(self, scheduler):
        dt = datetime(2026, 3, 16, 15, 0, 0, tzinfo=timezone.utc)
        assert self._session_at(scheduler, dt) == "US_OPEN"

    def test_midday_session(self, scheduler):
        dt = datetime(2026, 3, 16, 19, 0, 0, tzinfo=timezone.utc)
        assert self._session_at(scheduler, dt) == "MIDDAY"

    def test_eod_session(self, scheduler):
        dt = datetime(2026, 3, 16, 22, 20, 0, tzinfo=timezone.utc)
        assert self._session_at(scheduler, dt) == "EOD"

    def test_after_close_is_closed(self, scheduler):
        dt = datetime(2026, 3, 16, 22, 35, 0, tzinfo=timezone.utc)
        assert self._session_at(scheduler, dt) == "CLOSED"

    def test_weekend_is_closed(self, scheduler):
        dt = datetime(2026, 3, 21, 14, 0, 0, tzinfo=timezone.utc)  # Saturday
        assert self._session_at(scheduler, dt) == "CLOSED"


# ── seconds_until_next_run ────────────────────────────────────────────


class TestSecondsUntilNextRun:
    def test_returns_positive(self, scheduler):
        assert scheduler.seconds_until_next_run() >= 0

    def test_consistent_with_next_run_time(self, scheduler):
        secs = scheduler.seconds_until_next_run()
        nrt = scheduler.next_run_time()
        delta = (nrt - datetime.now(timezone.utc)).total_seconds()
        assert abs(secs - delta) < 2  # allow 2s tolerance


# ── EOD summary formatting ────────────────────────────────────────────


class TestEodSummary:
    def test_sends_formatted_message(self):
        """send_eod_summary calls tg._send with the expected structure."""
        tg = MagicMock()
        batch = {
            "results": [
                {
                    "ticker": "AAPL",
                    "combined_signal": "STRONG BUY",
                    "confidence": 0.72,
                    "execution": {"trade_id": 1},
                    "account_balance": 10000,
                },
                {
                    "ticker": "MSFT",
                    "combined_signal": "HOLD",
                    "confidence": 0.25,
                    "execution": {},
                    "account_balance": 10000,
                },
                None,
            ],
            "success_count": 2,
            "fail_count": 1,
            "elapsed_s": 120.0,
        }

        # Mock AlpacaTrader import to fail → falls back to batch data
        with patch.dict("sys.modules", {"execution.alpaca_trader": None}):
            send_eod_summary(tg, batch, ["AAPL", "MSFT", "NVDA"])

        tg._send.assert_called_once()
        msg = tg._send.call_args[0][0]
        assert "EOD Summary" in msg
        assert "Portfolio" in msg
        assert "Trades today" in msg
        assert "AAPL STRONG BUY" in msg

    def test_no_crash_without_telegram(self):
        """send_eod_summary(None, ...) does nothing."""
        send_eod_summary(None, {"results": [], "fail_count": 0}, [])


# ── Schedule constants ────────────────────────────────────────────────


class TestScheduleConstants:
    def test_eight_runs_defined(self):
        # 7 ticker-batch sessions + D9 PREMARKET_SCAN (scanner)
        assert len(SCHEDULE) == 8

    def test_run_names(self):
        names = [r["name"] for r in SCHEDULE]
        assert names == [
            "XETRA_PRE", "XETRA_OPEN", "PREMARKET_SCAN", "US_PRE",
            "PEAD_OPEN", "US_OPEN", "MIDDAY", "EOD",
        ]

    def test_runs_are_chronological(self):
        minutes = [r["hour"] * 60 + r["minute"] for r in SCHEDULE]
        assert minutes == sorted(minutes)

    def test_eod_is_last_run(self):
        assert SCHEDULE[-1]["eod"] is True
        assert all(not r["eod"] for r in SCHEDULE[:-1])

    def test_pre_sessions_types(self):
        """US_PRE runs full pipeline (signal); XETRA_PRE is lightweight (pre_signal)."""
        pre_sessions = {r["name"]: r for r in SCHEDULE if r["name"].endswith("_PRE")}
        assert len(pre_sessions) == 2
        assert pre_sessions["XETRA_PRE"]["session_type"] == "pre_signal"
        assert pre_sessions["US_PRE"]["session_type"] == "signal"


# ── XETRA ticker filtering ──────────────────────────────────────────


class TestXetraTickerFiltering:
    """XETRA tickers must only appear in XETRA_OPEN, never in US sessions.

    XETRA tickers were removed from the live watchlist on 2026-04-09. The
    scheduler routing logic still needs to work in case they are re-added,
    so these tests stub the loaders to inject XETRA + US tickers explicitly.
    """

    _STUB_US = ["NVDA", "META", "JPM"]
    _STUB_XETRA = ["SAP.XETRA", "SIE.XETRA"]

    @pytest.fixture
    def mixed_scheduler(self):
        """Scheduler whose full watchlist includes both US and XETRA tickers."""
        return DailyScheduler(
            full_watchlist=self._STUB_US + self._STUB_XETRA
        )

    def _captured_tickers(self, scheduler, session_name):
        """Execute a session and return the ticker list passed to run_batch."""
        run = scheduler._run_for_session(session_name)
        assert run is not None, f"No schedule entry for {session_name}"
        captured = {}

        async def fake_run_batch(tickers, **kwargs):
            captured["tickers"] = list(tickers)
            return {
                "results": [],
                "success_count": 0,
                "fail_count": 0,
                "elapsed_s": 0.0,
            }

        with patch("scheduler.daily_runner.run_batch", side_effect=fake_run_batch), \
             patch("scheduler.daily_runner._is_execution_allowed", return_value=(False, "test")), \
             patch.object(DailyScheduler, "_load_us_tickers", return_value=list(self._STUB_US)), \
             patch.object(DailyScheduler, "_load_xetra_tickers", return_value=list(self._STUB_XETRA)), \
             patch.object(DailyScheduler, "_claim_session_slot", return_value=True):
            scheduler._execute_run(run)

        return captured.get("tickers")

    def test_xetra_pre_includes_xetra_tickers(self, mixed_scheduler):
        tickers = self._captured_tickers(mixed_scheduler, "XETRA_PRE")
        assert "SAP.XETRA" in tickers
        assert "SIE.XETRA" in tickers

    def test_xetra_open_includes_xetra_tickers(self, mixed_scheduler):
        tickers = self._captured_tickers(mixed_scheduler, "XETRA_OPEN")
        assert "SAP.XETRA" in tickers
        assert "SIE.XETRA" in tickers

    def test_us_pre_excludes_xetra_tickers(self, mixed_scheduler):
        tickers = self._captured_tickers(mixed_scheduler, "US_PRE")
        assert "NVDA" in tickers
        assert "SAP.XETRA" not in tickers

    def test_us_open_excludes_xetra_tickers(self, mixed_scheduler):
        tickers = self._captured_tickers(mixed_scheduler, "US_OPEN")
        assert "NVDA" in tickers
        assert "META" in tickers
        assert "SAP.XETRA" not in tickers
        assert "SIE.XETRA" not in tickers

    def test_midday_excludes_xetra_tickers(self, mixed_scheduler):
        tickers = self._captured_tickers(mixed_scheduler, "MIDDAY")
        assert "NVDA" in tickers
        assert "SAP.XETRA" not in tickers
        assert "SIE.XETRA" not in tickers

    def test_eod_excludes_xetra_tickers(self, mixed_scheduler):
        tickers = self._captured_tickers(mixed_scheduler, "EOD")
        assert "NVDA" in tickers
        assert "SAP.XETRA" not in tickers
        assert "SIE.XETRA" not in tickers


# ── Slot-claim idempotency (deploy-overlap protection) ───────────────


class TestSessionSlotClaim:
    """The daemon must not let the same session fire twice on the same UTC day.

    On Railway a rolling deploy can leave the old container alive while the
    new one starts; both daemons would otherwise call _execute_run for the
    same session and fire two batches with two different watchlist.yaml
    contents (each baked into its own image). The slot claim sits in the
    shared SQLite DB so the second caller sees the row already exists.
    """

    @pytest.fixture
    def isolated_db(self, tmp_path, monkeypatch):
        """Point _resolve_db_path at a throwaway file."""
        db_file = tmp_path / "slot_claim.db"
        monkeypatch.setattr(
            "storage.database._resolve_db_path",
            lambda default=None: str(db_file),
        )
        return db_file

    def test_first_claim_returns_true(self, isolated_db):
        assert DailyScheduler._claim_session_slot("US_OPEN") is True

    def test_second_claim_same_day_returns_false(self, isolated_db):
        assert DailyScheduler._claim_session_slot("US_OPEN") is True
        assert DailyScheduler._claim_session_slot("US_OPEN") is False

    def test_different_sessions_independent(self, isolated_db):
        assert DailyScheduler._claim_session_slot("US_OPEN") is True
        assert DailyScheduler._claim_session_slot("MIDDAY") is True
        assert DailyScheduler._claim_session_slot("EOD") is True

    def test_failure_to_open_db_fails_open(self, monkeypatch):
        """If we cannot reach the DB, the slot claim must NOT block real runs."""
        def boom(*a, **kw):
            raise RuntimeError("simulated DB outage")
        monkeypatch.setattr(
            "storage.database._resolve_db_path",
            lambda default=None: "/no/such/path/forbidden.db",
        )
        monkeypatch.setattr("sqlite3.connect", boom)
        assert DailyScheduler._claim_session_slot("US_OPEN") is True

    def test_execute_run_skips_when_claim_lost(self):
        """_execute_run must early-return when the claim returns False."""
        sched = DailyScheduler(full_watchlist=["NVDA", "META"])
        run = {
            "name": "US_OPEN",
            "hour": 14,
            "minute": 30,
            "tickers": None,
            "workers": 3,
            "eod": False,
            "session_type": "execution",
        }
        with patch.object(DailyScheduler, "_claim_session_slot", return_value=False), \
             patch("scheduler.daily_runner.run_batch") as mock_run_batch:
            sched._execute_run(run)
        mock_run_batch.assert_not_called()
