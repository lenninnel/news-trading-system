"""Tests for the re-entry lock after a stop-loss exit (PortfolioManager gate 1b;
Freeze-Lift Fix 2/4 Q-016, widened 2026-09-08 — docs/REENTRY_LOCK_2026-09-08.md).

Covers:
  • _recently_stopped / _reentry_lock_status live-reconstruction from trade_history
      - blocked when the last round-trip stopped out today (gap 0), the previous
        session (gap 1) and at the boundary (gap == REENTRY_LOCK_SESSIONS)
      - allowed one session past the boundary
      - allowed when the last exit was a TARGET (not a stop)
      - unaffected when the ticker has no trade history
      - fails OPEN when the opening BUY has no stop_loss (guard bug must never
        block all trading)
      - falls back from executed_price to price for the closing SELL
  • holiday-aware session counting: a stop on Thu 2026-07-02 and a BUY on Mon
    2026-07-06 are ONE session apart (Fri 07-03 was a holiday); Labor Day
    2026-09-07 likewise. The weekday-only counter the lock replaces said 2.
  • trailing-stop exits arm the lock only with REENTRY_LOCK_INCLUDE_TRAILING.
  • can_add_position gate 1b wiring: a blocked BUY returns the lock reason,
    persists a `cooldown_stop` violation AND a signal_events row
    (strategy='PortfolioManager', signal_path='REENTRY_LOCK') carrying the
    remaining sessions and the first eligible date; other gates keep their
    precedence.
  • REPLAY: the documented churn cascades (MSFT / TSLA / CASY) — the lock
    blocks every same-session/next-session re-entry while allowing the first
    entry.

The lock is TIME-RELATIVE (session gap between the stop and *today*), so tests
place a stop at a computed gap-distance from the real today via
`_date_with_gap`, which uses the same holiday-aware counter the lock uses.

Run with:
    python3 -m pytest tests/test_portfolio_cooldown.py -v
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from config.settings import REENTRY_LOCK_SESSIONS
from data.market_calendar import NY_TZ, us_sessions_between
from storage.database import Database

N = REENTRY_LOCK_SESSIONS


# ===========================================================================
# Helpers
# ===========================================================================

def _date_with_gap(gap: int):
    """Return a date whose trading-session gap to today (New York) equals *gap*.

    gap 0 → today; gap n → the most recent date d < today with exactly n US
    trading sessions in (d, today].  Uses the lock's own `us_sessions_between`
    so the mapping is stable regardless of the weekday the suite runs on.
    """
    today = datetime.now(NY_TZ).date()
    if gap == 0:
        return today
    d = today
    while us_sessions_between(d, today) < gap:
        d -= timedelta(days=1)
    return d


class _FrozenClock:
    """Context manager: freeze `datetime.now` inside execution.portfolio_manager."""

    def __init__(self, at: datetime):
        real = at

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return real.astimezone(tz) if tz else real

        self._patch = patch("execution.portfolio_manager.datetime", _Frozen)

    def __enter__(self):
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


def _iso(d, hour=15):
    return f"{d.isoformat()}T{hour:02d}:00:00+00:00"


def _new_db(tmp_path, name="cd.db"):
    db_path = str(tmp_path / name)
    from analytics.signal_logger import SignalLogger
    SignalLogger(db=Database(db_path))  # init schema incl. signal_events
    return db_path


def _seed(db_path, ticker, action, price, created_at,
          stop_loss=None, executed_price=None, shares=10):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO trade_history "
        "(ticker, action, shares, price, stop_loss, take_profit, pnl, "
        " created_at, executed_price) VALUES (?,?,?,?,?,?,?,?,?)",
        (ticker, action, shares, price, stop_loss, None, 0.0,
         created_at, executed_price),
    )
    conn.commit()
    conn.close()


def _make_pm(db_path):
    from execution.portfolio_manager import PortfolioManager
    # _fetch_sector hits yfinance — patch for the whole construction + call.
    with patch("execution.portfolio_manager._fetch_sector", return_value="Tech"):
        return PortfolioManager(account_balance=100_000.0, db_path=db_path)


def _seed_stop_roundtrip(db_path, ticker, gap, entry=470.0, stop=465.0,
                         exit_px=None, buy_gap_offset=3):
    """Seed one BUY→stop-SELL round-trip whose SELL lands *gap* sessions ago."""
    buy_d = _date_with_gap(gap + buy_gap_offset)
    sell_d = _date_with_gap(gap)
    if exit_px is None:
        exit_px = stop * 0.998  # below stop → classifies as a stop exit
    _seed(db_path, ticker, "BUY", entry, _iso(buy_d, 14), stop_loss=stop)
    _seed(db_path, ticker, "SELL", exit_px, _iso(sell_d, 15), executed_price=exit_px)


# ===========================================================================
# 1. _recently_stopped — unit behaviour
# ===========================================================================

class TestRecentlyStoppedHelper:

    def test_blocked_when_stopped_today_gap0(self, tmp_path):
        db = _new_db(tmp_path)
        _seed_stop_roundtrip(db, "MSFT", gap=0)
        pm = _make_pm(db)
        blocked, reason = pm._recently_stopped("MSFT")
        assert blocked is True
        assert "Re-entry lock" in reason and "MSFT" in reason

    def test_blocked_when_stopped_prev_session_gap1(self, tmp_path):
        db = _new_db(tmp_path)
        _seed_stop_roundtrip(db, "MSFT", gap=1)
        pm = _make_pm(db)
        blocked, reason = pm._recently_stopped("MSFT")
        assert blocked is True
        assert f"1 of {N} trading sessions elapsed" in reason
        assert f"{N - 1} remaining" in reason

    def test_blocked_at_boundary_gap_equals_lock(self, tmp_path):
        db = _new_db(tmp_path)
        _seed_stop_roundtrip(db, "MSFT", gap=N, buy_gap_offset=2)
        pm = _make_pm(db)
        lock = pm._reentry_lock_status("MSFT")
        assert lock is not None
        assert lock["gap_sessions"] == N and lock["remaining"] == 0

    def test_allowed_one_session_past_boundary(self, tmp_path):
        db = _new_db(tmp_path)
        _seed_stop_roundtrip(db, "MSFT", gap=N + 1, buy_gap_offset=2)
        pm = _make_pm(db)
        assert pm._reentry_lock_status("MSFT") is None

    def test_allowed_when_last_exit_was_target(self, tmp_path):
        db = _new_db(tmp_path)
        # Exit ABOVE stop → target/other, not a stop.
        _seed_stop_roundtrip(db, "MSFT", gap=0, entry=470.0, stop=465.0,
                             exit_px=480.0)
        pm = _make_pm(db)
        blocked, reason = pm._recently_stopped("MSFT")
        assert blocked is False
        assert reason is None

    def test_blocked_when_stopped_two_sessions_ago(self, tmp_path):
        """The pre-2026-09 cool-down (1 session) let this through; the lock
        (N sessions) does not — that bucket alone cost −3.7 k USD in the book."""
        db = _new_db(tmp_path)
        _seed_stop_roundtrip(db, "MSFT", gap=2)
        pm = _make_pm(db)
        blocked, reason = pm._recently_stopped("MSFT")
        assert blocked is True
        assert "Re-entry lock" in reason

    def test_unaffected_when_no_trade_history(self, tmp_path):
        db = _new_db(tmp_path)
        pm = _make_pm(db)
        blocked, reason = pm._recently_stopped("NVDA")
        assert blocked is False
        assert reason is None

    def test_fails_open_when_opening_buy_has_no_stop_loss(self, tmp_path):
        db = _new_db(tmp_path)
        buy_d = _date_with_gap(3)
        sell_d = _date_with_gap(0)
        # Big loss exit but NO stop_loss on the opening BUY → cannot classify a
        # stop → fail OPEN (allowed), no exception.
        _seed(db, "MSFT", "BUY", 470.0, _iso(buy_d, 14), stop_loss=None)
        _seed(db, "MSFT", "SELL", 400.0, _iso(sell_d, 15), executed_price=400.0)
        pm = _make_pm(db)
        blocked, reason = pm._recently_stopped("MSFT")
        assert blocked is False
        assert reason is None

    def test_falls_back_to_price_when_executed_price_null(self, tmp_path):
        db = _new_db(tmp_path)
        buy_d = _date_with_gap(3)
        sell_d = _date_with_gap(0)
        _seed(db, "MSFT", "BUY", 470.0, _iso(buy_d, 14), stop_loss=465.0)
        # executed_price NULL → helper falls back to `price` (464 <= stop).
        _seed(db, "MSFT", "SELL", 464.0, _iso(sell_d, 15), executed_price=None)
        pm = _make_pm(db)
        blocked, reason = pm._recently_stopped("MSFT")
        assert blocked is True

    def test_gap_boundary_at_tolerance(self, tmp_path):
        """A SELL exactly at stop*(1+tol) is still a stop; just above is not."""
        db = _new_db(tmp_path)
        buy_d = _date_with_gap(3)
        sell_d = _date_with_gap(0)
        stop = 465.0
        _seed(db, "AAPL", "BUY", 470.0, _iso(buy_d, 14), stop_loss=stop)
        _seed(db, "AAPL", "SELL", stop * 1.003, _iso(sell_d, 15),
              executed_price=stop * 1.003)  # == boundary → stop
        pm = _make_pm(db)
        blocked, _ = pm._recently_stopped("AAPL")
        assert blocked is True

    def test_just_above_tolerance_not_a_stop(self, tmp_path):
        db = _new_db(tmp_path)
        buy_d = _date_with_gap(3)
        sell_d = _date_with_gap(0)
        stop = 465.0
        _seed(db, "AAPL", "BUY", 470.0, _iso(buy_d, 14), stop_loss=stop)
        _seed(db, "AAPL", "SELL", stop * 1.01, _iso(sell_d, 15),
              executed_price=stop * 1.01)  # 1% above stop → not a stop
        pm = _make_pm(db)
        blocked, _ = pm._recently_stopped("AAPL")
        assert blocked is False

    def test_last_roundtrip_wins_over_older_stop(self, tmp_path):
        """FIFO: an old stop then a recent TARGET → the recent target governs."""
        db = _new_db(tmp_path)
        # Old stop (2 sessions ago — inside the lock window on its own)
        _seed(db, "MSFT", "BUY", 470.0, _iso(_date_with_gap(8), 14), stop_loss=465.0)
        _seed(db, "MSFT", "SELL", 460.0, _iso(_date_with_gap(2), 15), executed_price=460.0)
        # Most recent round-trip is a TARGET today
        _seed(db, "MSFT", "BUY", 470.0, _iso(_date_with_gap(3), 14), stop_loss=465.0)
        _seed(db, "MSFT", "SELL", 485.0, _iso(_date_with_gap(0), 15), executed_price=485.0)
        pm = _make_pm(db)
        blocked, _ = pm._recently_stopped("MSFT")
        assert blocked is False  # last round-trip was a target


# ===========================================================================
# 1b. Holiday-aware session counting (fixed dates, frozen clock)
# ===========================================================================

class TestHolidayAwareGap:

    @staticmethod
    def _seed_fixed(db, ticker, buy_iso, sell_iso, stop=465.0, exit_px=None):
        exit_px = exit_px if exit_px is not None else stop * 0.998
        _seed(db, ticker, "BUY", 470.0, buy_iso, stop_loss=stop)
        _seed(db, ticker, "SELL", exit_px, sell_iso, executed_price=exit_px)

    def test_independence_day_2026_counts_as_no_session(self, tmp_path):
        """Stop Thu 2026-07-02, BUY Mon 2026-07-06: gap 1 (Fri 07-03 holiday).
        This is the TRGP/VRT case the 1-session weekday counter let through;
        VRT lost −2 902 USD (−10.6 %) on that re-entry."""
        db = _new_db(tmp_path)
        self._seed_fixed(db, "VRT", "2026-06-30T14:30:00+00:00", "2026-07-02T14:30:05+00:00")
        pm = _make_pm(db)
        with _FrozenClock(datetime(2026, 7, 6, 14, 30, tzinfo=timezone.utc)):
            lock = pm._reentry_lock_status("VRT")
        assert lock is not None
        assert lock["gap_sessions"] == 1
        assert lock["remaining"] == N - 1

    def test_labor_day_2026_counts_as_no_session(self, tmp_path):
        """Stop Fri 2026-09-04, today Tue 2026-09-08: gap 1 (Mon 09-07 Labor Day);
        first eligible session is N+1 sessions after the stop."""
        db = _new_db(tmp_path)
        self._seed_fixed(db, "XOM", "2026-09-02T14:30:00+00:00", "2026-09-04T13:33:00+00:00")
        pm = _make_pm(db)
        with _FrozenClock(datetime(2026, 9, 8, 14, 30, tzinfo=timezone.utc)):
            lock = pm._reentry_lock_status("XOM")
        assert lock is not None
        assert lock["gap_sessions"] == 1
        assert lock["stop_date"] == "2026-09-04"
        # sessions after 09-04: 09-08(1) 09-09(2) 09-10(3) 09-11(4) 09-14(5) 09-15(6)
        assert lock["eligible_from"] == "2026-09-15"

    def test_unlocked_on_first_eligible_session(self, tmp_path):
        db = _new_db(tmp_path)
        self._seed_fixed(db, "XOM", "2026-09-02T14:30:00+00:00", "2026-09-04T13:33:00+00:00")
        pm = _make_pm(db)
        with _FrozenClock(datetime(2026, 9, 14, 14, 30, tzinfo=timezone.utc)):
            assert pm._reentry_lock_status("XOM") is not None   # gap 5 = N → locked
        with _FrozenClock(datetime(2026, 9, 15, 14, 30, tzinfo=timezone.utc)):
            assert pm._reentry_lock_status("XOM") is None       # gap 6 → free


# ===========================================================================
# 1c. Trailing-stop exits — opt-in only
# ===========================================================================

class TestTrailingStopOption:

    @staticmethod
    def _seed_trailing_exit(db):
        """BUY 470 / stop 465, PositionManager trails the stop to 480 and it
        fires at 479.5 — a profitable 'stop_loss_triggered' exit."""
        buy_d = _date_with_gap(3)
        sell_d = _date_with_gap(0)
        _seed(db, "MSFT", "BUY", 470.0, _iso(buy_d, 14), stop_loss=465.0)
        _seed(db, "MSFT", "SELL", 479.5, _iso(sell_d, 15), executed_price=479.5)
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO signal_events (timestamp, ticker, strategy, signal, "
            "price_at_signal, bull_case) VALUES (?,?,?,?,?,?)",
            (_iso(sell_d, 15).replace("T15:00:00", "T14:59:58"), "MSFT",
             "PositionManager", "SELL", 479.5, "stop_loss_triggered at $480.00"),
        )
        conn.commit()
        conn.close()

    def test_trailing_exit_does_not_lock_by_default(self, tmp_path):
        db = _new_db(tmp_path)
        self._seed_trailing_exit(db)
        pm = _make_pm(db)
        assert pm._reentry_lock_status("MSFT") is None

    def test_trailing_exit_locks_when_enabled(self, tmp_path):
        db = _new_db(tmp_path)
        self._seed_trailing_exit(db)
        pm = _make_pm(db)
        with patch("execution.portfolio_manager.REENTRY_LOCK_INCLUDE_TRAILING", True):
            lock = pm._reentry_lock_status("MSFT")
        assert lock is not None
        assert lock["exit_kind"] == "trailing_stop"
        assert "trailing-stop exit" in lock["reason"]

    def test_target_exit_never_locks_even_when_enabled(self, tmp_path):
        """No PositionManager stop row → a SELL above the stop is a target."""
        db = _new_db(tmp_path)
        _seed_stop_roundtrip(db, "MSFT", gap=0, exit_px=480.0)
        pm = _make_pm(db)
        with patch("execution.portfolio_manager.REENTRY_LOCK_INCLUDE_TRAILING", True):
            assert pm._reentry_lock_status("MSFT") is None


# ===========================================================================
# 2. can_add_position — gate 1b wiring & precedence
# ===========================================================================

class TestCanAddPositionCooldownGate:

    def test_buy_blocked_and_violation_persisted(self, tmp_path):
        db = _new_db(tmp_path)
        _seed_stop_roundtrip(db, "MSFT", gap=0)
        pm = _make_pm(db)
        with patch("execution.portfolio_manager._fetch_sector", return_value="Tech"):
            allowed, reason = pm.can_add_position("MSFT", "momentum", 5_000.0)
        assert allowed is False
        assert "Re-entry lock" in reason

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT violation_type, ticker FROM portfolio_violations "
            "WHERE violation_type='cooldown_stop' AND ticker='MSFT'"
        ).fetchone()
        conn.close()
        assert row is not None

    def test_rejection_is_logged_to_signal_events_with_remaining(self, tmp_path):
        """The row the later evaluation needs: who asked, why rejected, how
        long the lock still runs, from when a BUY is possible again."""
        db = _new_db(tmp_path)
        _seed_stop_roundtrip(db, "MSFT", gap=1)
        pm = _make_pm(db)
        with patch("execution.portfolio_manager._fetch_sector", return_value="Tech"):
            allowed, _ = pm.can_add_position(
                "MSFT", "momentum", 5_000.0, session="US_OPEN", price=471.25,
            )
        assert allowed is False

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM signal_events WHERE ticker='MSFT' "
            "AND strategy='PortfolioManager' AND signal_path='REENTRY_LOCK'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row["signal"] == "HOLD"
        assert row["session"] == "US_OPEN"
        assert row["trade_executed"] == 0
        assert row["price_at_signal"] == 471.25
        assert row["bull_case"] == "BUY requested by momentum"
        assert "reentry_lock: stop_loss exit at " in row["bear_case"]
        assert "sessions_elapsed=1;" in row["bear_case"]
        assert f"sessions_remaining={N - 1};" in row["bear_case"]
        assert f"lock_sessions={N};" in row["bear_case"]
        assert "eligible_from=20" in row["bear_case"]

    def test_allowed_buy_writes_no_lock_row(self, tmp_path):
        db = _new_db(tmp_path)
        _seed_stop_roundtrip(db, "MSFT", gap=0, exit_px=480.0)  # target
        pm = _make_pm(db)
        with patch("execution.portfolio_manager._fetch_sector", return_value="Tech"):
            allowed, _ = pm.can_add_position("MSFT", "momentum", 5_000.0, session="US_OPEN")
        assert allowed is True
        conn = sqlite3.connect(db)
        n = conn.execute(
            "SELECT COUNT(*) FROM signal_events WHERE signal_path='REENTRY_LOCK'"
        ).fetchone()[0]
        conn.close()
        assert n == 0

    def test_buy_allowed_when_last_exit_target(self, tmp_path):
        db = _new_db(tmp_path)
        _seed_stop_roundtrip(db, "MSFT", gap=0, exit_px=480.0)  # target
        pm = _make_pm(db)
        with patch("execution.portfolio_manager._fetch_sector", return_value="Tech"):
            allowed, reason = pm.can_add_position("MSFT", "momentum", 5_000.0)
        assert allowed is True, reason

    def test_buy_allowed_when_stop_past_lock_window(self, tmp_path):
        db = _new_db(tmp_path)
        _seed_stop_roundtrip(db, "MSFT", gap=N + 1, buy_gap_offset=2)
        pm = _make_pm(db)
        with patch("execution.portfolio_manager._fetch_sector", return_value="Tech"):
            allowed, reason = pm.can_add_position("MSFT", "momentum", 5_000.0)
        assert allowed is True, reason

    def test_no_history_unaffected(self, tmp_path):
        db = _new_db(tmp_path)
        pm = _make_pm(db)
        with patch("execution.portfolio_manager._fetch_sector", return_value="Tech"):
            allowed, reason = pm.can_add_position("NVDA", "momentum", 5_000.0)
        assert allowed is True, reason

    def test_duplicate_gate_still_precedes_cooldown(self, tmp_path):
        """A currently-held ticker returns the duplicate reason, not cool-down —
        proving gate 1 still runs before 1b."""
        db = _new_db(tmp_path)
        _seed_stop_roundtrip(db, "MSFT", gap=0)
        pm = _make_pm(db)
        # Force MSFT to look currently-held.
        with patch.object(
            pm, "_open_positions_with_meta",
            return_value=[{"ticker": "MSFT", "strategy": "momentum",
                           "sector": "Tech", "current_value": 5_000.0}],
        ), patch("execution.portfolio_manager._fetch_sector", return_value="Tech"):
            allowed, reason = pm.can_add_position("MSFT", "momentum", 5_000.0)
        assert allowed is False
        assert "Already holding" in reason


# ===========================================================================
# 3. REPLAY — documented churn cascades (would it have stopped the bleed?)
# ===========================================================================

# Structurally-faithful reconstructions of the documented Jun 1-12 cascades.
# Each entry after the first is a re-entry that, in the real cascade, followed
# the prior ATR stop within `gap` trading sessions (0 = same-day "59-min rebuy",
# 1 = next session). Representative prices; stop_loss below entry so each exit
# classifies as a stop.
_CASCADES = {
    # MSFT: 6 BUYs / 5 stops. Re-entries 2..6 each within 1 session of a stop.
    "MSFT": {
        "entry": 470.0, "stop": 465.0,
        "reentry_gaps": [0, 0, 1, 0, 1],   # 5 re-entries
    },
    # TSLA: 4 BUYs / 3 stops.
    "TSLA": {
        "entry": 340.0, "stop": 333.0,
        "reentry_gaps": [0, 1, 0],          # 3 re-entries
    },
    # CASY: 3 BUYs / 2 stops.
    "CASY": {
        "entry": 415.0, "stop": 408.0,
        "reentry_gaps": [1, 0],             # 2 re-entries
    },
}


def _replay_cascade(tmp_path, ticker, spec):
    """Replay one cascade as-of each re-entry; return (blocked, total_reentries).

    For re-entry i (i = 2..N) the guard sees the history up to and including the
    prior stop; the prior stop sits `gap_i` sessions before the re-entry. We
    reproduce that by seeding a DB whose LAST stop lands `gap_i` sessions ago and
    asking `_recently_stopped` (evaluated as-of the real today).
    """
    entry, stop = spec["entry"], spec["stop"]
    gaps = spec["reentry_gaps"]

    blocked = 0
    # First entry: no prior round-trip → must be allowed.
    db0 = _new_db(tmp_path, name=f"{ticker}_e1.db")
    pm0 = _make_pm(db0)
    first_allowed = pm0._recently_stopped(ticker)[0] is False

    for i, gap in enumerate(gaps, start=2):
        db = _new_db(tmp_path, name=f"{ticker}_e{i}.db")
        # Seed i-1 prior round-trips; the most recent stop lands `gap` ago and
        # older stops further back so FIFO must pick the latest.
        for j in range(i - 1):
            g = gap if j == i - 2 else gap + 2 * (i - 1 - j)
            _seed_stop_roundtrip(db, ticker, gap=g, entry=entry, stop=stop)
        pm = _make_pm(db)
        if pm._recently_stopped(ticker)[0] is True:
            blocked += 1
    return first_allowed, blocked, len(gaps)


class TestChurnCascadeReplay:

    @pytest.mark.parametrize("ticker", list(_CASCADES.keys()))
    def test_cascade_reentries_blocked(self, tmp_path, ticker):
        spec = _CASCADES[ticker]
        first_allowed, blocked, total = _replay_cascade(tmp_path, ticker, spec)
        # First entry is never blocked (no prior stop to cool down from).
        assert first_allowed is True
        # Every documented within-1-session re-entry is blocked.
        assert blocked == total, (
            f"{ticker}: guard blocked {blocked}/{total} re-entries "
            f"(expected all {total})"
        )

    def test_replay_summary(self, tmp_path, capsys):
        """Emit the 'would it have stopped the bleed' evidence."""
        total_reentries = 0
        total_blocked = 0
        lines = []
        for ticker, spec in _CASCADES.items():
            first_allowed, blocked, total = _replay_cascade(tmp_path, ticker, spec)
            total_reentries += total
            total_blocked += blocked
            lines.append(
                f"  {ticker}: 1 initial entry allowed, {blocked}/{total} "
                f"churn re-entries BLOCKED"
            )
        # The guard would have blocked every documented churn re-entry.
        assert total_blocked == total_reentries
        print("\nQ-016 cool-down replay (would it have stopped the bleed?):")
        for ln in lines:
            print(ln)
        print(f"  TOTAL: {total_blocked}/{total_reentries} churn re-entries "
              f"blocked across MSFT/TSLA/CASY")
