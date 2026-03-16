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
    def test_before_first_run_returns_xetra(self, scheduler):
        """Monday 05:00 UTC → next run is XETRA_OPEN at 07:00."""
        monday_5am = datetime(2026, 3, 16, 5, 0, 0, tzinfo=timezone.utc)
        nrt = scheduler.next_run_time(after=monday_5am)
        assert nrt.hour == 7
        assert nrt.minute == 0
        assert nrt.day == 16

    def test_between_runs_returns_next(self, scheduler):
        """Monday 10:00 UTC → next run is US_OPEN at 14:30."""
        monday_10am = datetime(2026, 3, 16, 10, 0, 0, tzinfo=timezone.utc)
        nrt = scheduler.next_run_time(after=monday_10am)
        assert nrt.hour == 14
        assert nrt.minute == 30

    def test_after_midday_returns_eod(self, scheduler):
        """Monday 20:00 UTC → next run is EOD at 22:15."""
        monday_8pm = datetime(2026, 3, 16, 20, 0, 0, tzinfo=timezone.utc)
        nrt = scheduler.next_run_time(after=monday_8pm)
        assert nrt.hour == 22
        assert nrt.minute == 15

    def test_after_last_run_returns_next_day(self, scheduler):
        """Monday 22:30 UTC → next run is Tuesday XETRA_OPEN 07:00."""
        monday_late = datetime(2026, 3, 16, 22, 30, 0, tzinfo=timezone.utc)
        nrt = scheduler.next_run_time(after=monday_late)
        assert nrt.weekday() == 1  # Tuesday
        assert nrt.hour == 7
        assert nrt.minute == 0

    def test_all_four_runs_fire_on_weekday(self, scheduler):
        """Starting from Monday 00:00, iterating should hit all 4 run times."""
        t = datetime(2026, 3, 16, 0, 0, 0, tzinfo=timezone.utc)
        run_hours = []
        for _ in range(4):
            t = scheduler.next_run_time(after=t)
            run_hours.append((t.hour, t.minute))
            # Advance 1 minute past the run time
            t = t.replace(second=1)
        assert run_hours == [(7, 0), (14, 30), (18, 0), (22, 15)]


# ── Weekend skip ──────────────────────────────────────────────────────


class TestWeekendSkip:
    def test_friday_after_eod_skips_to_monday(self, scheduler):
        """Friday 23:00 UTC → next run is Monday 07:00."""
        # 2026-03-20 is a Friday
        friday_late = datetime(2026, 3, 20, 23, 0, 0, tzinfo=timezone.utc)
        nrt = scheduler.next_run_time(after=friday_late)
        assert nrt.weekday() == 0  # Monday
        assert nrt.day == 23
        assert nrt.hour == 7

    def test_saturday_skips_to_monday(self, scheduler):
        """Saturday 12:00 UTC → next run is Monday 07:00."""
        saturday = datetime(2026, 3, 21, 12, 0, 0, tzinfo=timezone.utc)
        nrt = scheduler.next_run_time(after=saturday)
        assert nrt.weekday() == 0
        assert nrt.hour == 7

    def test_sunday_skips_to_monday(self, scheduler):
        """Sunday 18:00 UTC → next run is Monday 07:00."""
        sunday = datetime(2026, 3, 22, 18, 0, 0, tzinfo=timezone.utc)
        nrt = scheduler.next_run_time(after=sunday)
        assert nrt.weekday() == 0
        assert nrt.hour == 7


# ── current_session ───────────────────────────────────────────────────


class TestCurrentSession:
    def _session_at(self, scheduler, dt):
        with patch("scheduler.daily_runner.datetime") as mock_dt:
            mock_dt.now.return_value = dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            return scheduler.current_session()

    def test_before_trading_is_closed(self, scheduler):
        dt = datetime(2026, 3, 16, 6, 0, 0, tzinfo=timezone.utc)
        assert self._session_at(scheduler, dt) == "CLOSED"

    def test_xetra_session(self, scheduler):
        dt = datetime(2026, 3, 16, 8, 0, 0, tzinfo=timezone.utc)
        assert self._session_at(scheduler, dt) == "XETRA_OPEN"

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
    def test_four_runs_defined(self):
        assert len(SCHEDULE) == 4

    def test_run_names(self):
        names = [r["name"] for r in SCHEDULE]
        assert names == ["XETRA_OPEN", "US_OPEN", "MIDDAY", "EOD"]

    def test_runs_are_chronological(self):
        minutes = [r["hour"] * 60 + r["minute"] for r in SCHEDULE]
        assert minutes == sorted(minutes)

    def test_eod_is_last_run(self):
        assert SCHEDULE[-1]["eod"] is True
        assert all(not r["eod"] for r in SCHEDULE[:-1])
