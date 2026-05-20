"""
Tests for monitoring.position_manager.PositionManager.

Covers:
  - Stop-loss triggers position close
  - Take-profit triggers position close
  - Trailing stop updates when position is profitable >2%
  - No action outside market hours
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from monitoring.position_manager import (
    PositionManager,
    _TRAILING_ACTIVATION_PCT,
    _TRAILING_LOCK_PCT,
)


@pytest.fixture(autouse=True)
def _isolate_portfolio_positions():
    """Wipe portfolio_positions rows before each test.

    The pre-existing PM tests construct PM without passing db=, which
    causes PM to instantiate a real Database() against the shared
    /tmp/pytest_trading.db. _check_all_positions then writes test
    tickers (AAPL, NVDA, …) into portfolio_positions via _mark_to_market,
    plus (since 2026-05-12) trailing_stop values. Those persist across
    tests and contaminate the rehydration path. Clearing the table
    before each test isolates state without touching the existing tests.
    """
    from storage.database import Database
    db = Database()
    with db._connect() as conn:
        conn.execute("DELETE FROM portfolio_positions")
    yield


# ── helpers ──────────────────────────────────────────────────────────

def _make_trader(portfolio=None, trade_history=None, fill_outcome="filled"):
    """Return a mock trader with configurable portfolio and trade history.

    ``track_trade`` returns a dict matching the IBKRTrader contract.
    ``fill_outcome`` controls what kind of result the broker returns:
      "filled"    — trade_id set, no skipped marker (normal close path)
      "stuck"     — extended SELL timed out after STOP_MAX_WAIT
      "cancelled" — IBKR cancelled the order
      "timeout"   — fast-poll ceiling reached (BUY only; legacy)
    """
    trader = MagicMock()
    trader.get_portfolio.return_value = portfolio or []
    trader.get_trade_history.return_value = trade_history or []

    def _track(*, ticker, action, shares, price, **_kwargs):
        if fill_outcome == "filled":
            return {
                "trade_id": 1, "ticker": ticker, "action": action,
                "shares": shares, "price": price, "pnl": 0.0,
                "total_value": shares * price,
            }
        return {
            "trade_id": None, "ticker": ticker, "action": action,
            "shares": shares, "price": price, "pnl": 0.0, "total_value": 0.0,
            "skipped": True,
            "skip_reason": f"IBKR order {fill_outcome}: test",
            "outcome": fill_outcome,
        }

    trader.track_trade.side_effect = _track
    return trader


def _make_position(ticker="AAPL", shares=10, avg_price=200.0,
                   stop_loss=190.0, take_profit=220.0):
    """Build a portfolio position + matching trade history entry."""
    portfolio = [{
        "ticker": ticker,
        "shares": shares,
        "avg_price": avg_price,
    }]
    trade_history = [{
        "action": "BUY",
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }]
    return portfolio, trade_history


# ── test: stop-loss triggers close ───────────────────────────────────

class TestStopLoss:
    def test_stop_loss_triggers_close(self):
        """When price <= stop_loss, position should be closed via SELL."""
        portfolio, history = _make_position(
            ticker="AAPL", shares=10, avg_price=200.0,
            stop_loss=190.0, take_profit=220.0,
        )
        trader = _make_trader(portfolio=portfolio, trade_history=history)
        notifier = MagicMock()

        pm = PositionManager(trader=trader, notifier=notifier)

        # Mock price at stop-loss level
        with patch.object(pm, "_fetch_current_price", return_value=189.0):
            results = pm._check_all_positions()

        # Should have sold. _close_position also threads attribution +
        # intended_price onto the SELL — see DEV-Q-001-INFRA P1.
        trader.track_trade.assert_called_once_with(
            ticker="AAPL", action="SELL", shares=10, price=189.0,
            strategy=None, intended_price=189.0,
        )
        # Telegram alert sent
        notifier.send_price_alert.assert_called_once()
        alert_msg = notifier.send_price_alert.call_args[0][0]
        assert "Stop-loss" in alert_msg
        assert "AAPL" in alert_msg

        assert len(results) == 1
        assert results[0]["action"] == "stop_loss"

    def test_stop_loss_at_exact_boundary(self):
        """Price exactly equal to stop_loss should trigger close."""
        portfolio, history = _make_position(
            stop_loss=190.0, take_profit=220.0,
        )
        trader = _make_trader(portfolio=portfolio, trade_history=history)
        pm = PositionManager(trader=trader)

        with patch.object(pm, "_fetch_current_price", return_value=190.0):
            results = pm._check_all_positions()

        trader.track_trade.assert_called_once()
        assert results[0]["action"] == "stop_loss"

    def test_stop_loss_stuck_alerts_and_skips_next_cycle(self):
        """When the SELL hits IBKR's STOP_MAX_WAIT and comes back ``stuck``:
        - PM sends an "ORDER STUCK" Telegram alert (NOT "Stop-loss hit"),
        - the next 60s cycle is skipped for that ticker so we don't pile
          a duplicate order on top of the in-flight one. Regression for
          VRT 2026-05-15 (false "Closed position" alert + retry slippage).
        """
        portfolio, history = _make_position(
            ticker="VRT", shares=10, avg_price=400.0,
            stop_loss=380.0, take_profit=440.0,
        )
        trader = _make_trader(
            portfolio=portfolio, trade_history=history,
            fill_outcome="stuck",
        )
        notifier = MagicMock()
        pm = PositionManager(trader=trader, notifier=notifier)

        with patch.object(pm, "_fetch_current_price", return_value=370.0):
            results = pm._check_all_positions()

        alert_msg = notifier.send_price_alert.call_args[0][0]
        assert "STUCK" in alert_msg.upper()
        assert "VRT" in alert_msg
        # Critical: the false "Stop-loss hit / Closed position" wording
        # must NOT be in the alert when the order hasn't actually filled.
        assert "Stop-loss hit" not in alert_msg
        assert results[0]["action"] == "stop_loss_stuck"

        # Second cycle: trader should NOT be asked to close again — the
        # cooldown suppresses duplicate orders against the in-flight SELL.
        trader.track_trade.reset_mock()
        with patch.object(pm, "_fetch_current_price", return_value=370.0):
            second = pm._check_all_positions()
        trader.track_trade.assert_not_called()
        assert second == []  # cooldown skip yields no actions

    def test_stop_loss_cancelled_does_not_alert(self):
        """Cancelled / failed SELL should not send a 'Stop-loss hit' alert;
        the next 60s cycle will re-evaluate and retry naturally if the
        position is still in violation. Regression for false-closure alerts.
        """
        portfolio, history = _make_position(
            ticker="AAPL", shares=10, avg_price=200.0,
            stop_loss=190.0, take_profit=220.0,
        )
        trader = _make_trader(
            portfolio=portfolio, trade_history=history,
            fill_outcome="cancelled",
        )
        notifier = MagicMock()
        pm = PositionManager(trader=trader, notifier=notifier)

        with patch.object(pm, "_fetch_current_price", return_value=189.0):
            results = pm._check_all_positions()

        notifier.send_price_alert.assert_not_called()
        assert results == []


# ── test: take-profit triggers close ─────────────────────────────────

class TestTakeProfit:
    def test_take_profit_triggers_close(self):
        """When price >= take_profit, position should be closed."""
        portfolio, history = _make_position(
            ticker="MSFT", shares=5, avg_price=400.0,
            stop_loss=380.0, take_profit=440.0,
        )
        trader = _make_trader(portfolio=portfolio, trade_history=history)
        notifier = MagicMock()

        pm = PositionManager(trader=trader, notifier=notifier)

        with patch.object(pm, "_fetch_current_price", return_value=445.0):
            results = pm._check_all_positions()

        trader.track_trade.assert_called_once_with(
            ticker="MSFT", action="SELL", shares=5, price=445.0,
            strategy=None, intended_price=445.0,
        )
        alert_msg = notifier.send_price_alert.call_args[0][0]
        assert "Take-profit" in alert_msg
        assert "MSFT" in alert_msg

        assert len(results) == 1
        assert results[0]["action"] == "take_profit"

    def test_take_profit_at_exact_boundary(self):
        """Price exactly at take_profit should trigger close."""
        portfolio, history = _make_position(
            take_profit=220.0, stop_loss=190.0,
        )
        trader = _make_trader(portfolio=portfolio, trade_history=history)
        pm = PositionManager(trader=trader)

        with patch.object(pm, "_fetch_current_price", return_value=220.0):
            results = pm._check_all_positions()

        trader.track_trade.assert_called_once()
        assert results[0]["action"] == "take_profit"


# ── test: trailing stop updates ──────────────────────────────────────

class TestTrailingStop:
    def test_trailing_stop_updates(self):
        """When position is >2% profitable, trailing stop should be set."""
        portfolio, history = _make_position(
            ticker="NVDA", shares=8, avg_price=200.0,
            stop_loss=190.0, take_profit=250.0,
        )
        trader = _make_trader(portfolio=portfolio, trade_history=history)
        notifier = MagicMock()

        pm = PositionManager(trader=trader, notifier=notifier)

        # Price up 5% → should trigger trailing stop
        with patch.object(pm, "_fetch_current_price", return_value=210.0):
            results = pm._check_all_positions()

        # No trade executed (not closing)
        trader.track_trade.assert_not_called()

        # Trailing stop update is logged but NOT sent to Telegram (noise reduction)
        notifier.send_price_alert.assert_not_called()

        assert len(results) == 1
        assert results[0]["action"] == "trailing_update"

        # Trailing stop should be stored
        assert "NVDA" in pm._trailing_stops
        trailing = pm._trailing_stops["NVDA"]
        # Must lock in at least 1% gain
        min_lock = 200.0 * (1 + _TRAILING_LOCK_PCT / 100)
        assert trailing >= min_lock

    def test_trailing_stop_only_moves_up(self):
        """Trailing stop should never decrease."""
        portfolio, history = _make_position(
            ticker="AAPL", shares=10, avg_price=200.0,
            stop_loss=190.0, take_profit=250.0,
        )
        trader = _make_trader(portfolio=portfolio, trade_history=history)
        pm = PositionManager(trader=trader)

        # First check: price at 210 (up 5%)
        with patch.object(pm, "_fetch_current_price", return_value=210.0):
            pm._check_all_positions()
        stop_1 = pm._trailing_stops["AAPL"]

        # Second check: price drops to 206 (still up 3%, above activation)
        with patch.object(pm, "_fetch_current_price", return_value=206.0):
            pm._check_all_positions()
        stop_2 = pm._trailing_stops["AAPL"]

        # Trailing stop must not have decreased
        assert stop_2 >= stop_1

    def test_no_trailing_below_activation(self):
        """No trailing stop when gain is below 2%."""
        portfolio, history = _make_position(
            ticker="AAPL", shares=10, avg_price=200.0,
            stop_loss=190.0, take_profit=250.0,
        )
        trader = _make_trader(portfolio=portfolio, trade_history=history)
        pm = PositionManager(trader=trader)

        # Price up only 1% → below activation threshold
        with patch.object(pm, "_fetch_current_price", return_value=202.0):
            results = pm._check_all_positions()

        assert len(results) == 0
        assert "AAPL" not in pm._trailing_stops

    def test_trailing_stop_triggers_close(self):
        """When price drops below trailing stop, position is closed."""
        portfolio, history = _make_position(
            ticker="AAPL", shares=10, avg_price=200.0,
            stop_loss=190.0, take_profit=250.0,
        )
        trader = _make_trader(portfolio=portfolio, trade_history=history)
        pm = PositionManager(trader=trader)

        # Set trailing stop by pushing price up
        with patch.object(pm, "_fetch_current_price", return_value=210.0):
            pm._check_all_positions()
        trailing = pm._trailing_stops["AAPL"]

        # Now price drops below the trailing stop
        with patch.object(pm, "_fetch_current_price", return_value=trailing - 0.50):
            results = pm._check_all_positions()

        trader.track_trade.assert_called_once()
        assert results[0]["action"] == "stop_loss"
        # Trailing stop should be cleaned up
        assert "AAPL" not in pm._trailing_stops


# ── test: trailing-stop persistence (DB rehydration) ─────────────────

class TestTrailingStopPersistence:
    def test_rehydrates_from_db_on_init(self):
        """PM should preload _trailing_stops from db.get_trailing_stops()."""
        trader = _make_trader()
        db = MagicMock()
        db.get_trailing_stops.return_value = {"AAPL": 210.50, "MSFT": 405.25}

        pm = PositionManager(trader=trader, db=db)

        db.get_trailing_stops.assert_called_once()
        assert pm._trailing_stops == {"AAPL": 210.50, "MSFT": 405.25}

    def test_empty_db_yields_empty_dict(self):
        """No persisted trailing stops → in-memory dict stays empty."""
        trader = _make_trader()
        db = MagicMock()
        db.get_trailing_stops.return_value = {}

        pm = PositionManager(trader=trader, db=db)

        assert pm._trailing_stops == {}

    def test_rehydrate_db_error_is_non_fatal(self):
        """A DB failure during rehydrate must not crash PM construction."""
        trader = _make_trader()
        db = MagicMock()
        db.get_trailing_stops.side_effect = RuntimeError("disk I/O error")

        # Should not raise
        pm = PositionManager(trader=trader, db=db)
        assert pm._trailing_stops == {}

    def test_trailing_update_persists_to_db(self):
        """When trailing stop activates, set_trailing_stop must be called."""
        portfolio, history = _make_position(
            ticker="AAPL", shares=10, avg_price=200.0,
            stop_loss=190.0, take_profit=250.0,
        )
        trader = _make_trader(portfolio=portfolio, trade_history=history)
        db = MagicMock()
        db.get_trailing_stops.return_value = {}

        pm = PositionManager(trader=trader, db=db)
        with patch.object(pm, "_fetch_current_price", return_value=210.0):
            pm._check_all_positions()

        # AAPL is +5% from $200 → trail activated; DB write must occur.
        assert "AAPL" in pm._trailing_stops
        db.set_trailing_stop.assert_called_once()
        ticker_arg, stop_arg = db.set_trailing_stop.call_args[0]
        assert ticker_arg == "AAPL"
        assert stop_arg == pytest.approx(pm._trailing_stops["AAPL"])

    def test_stop_loss_close_clears_persisted_trailing(self):
        """On SL fire, the DB column must be NULLed (set_trailing_stop(None))."""
        portfolio, history = _make_position(
            ticker="AAPL", shares=10, avg_price=200.0,
            stop_loss=195.0, take_profit=220.0,
        )
        trader = _make_trader(portfolio=portfolio, trade_history=history)
        db = MagicMock()
        db.get_trailing_stops.return_value = {}

        pm = PositionManager(trader=trader, db=db)
        with patch.object(pm, "_fetch_current_price", return_value=190.0):
            pm._check_all_positions()

        # Verify clear was called with None for AAPL
        clear_calls = [
            c for c in db.set_trailing_stop.call_args_list
            if c.args[0] == "AAPL" and c.args[1] is None
        ]
        assert clear_calls, "expected set_trailing_stop('AAPL', None) on SL"


# ── test: outside market hours ───────────────────────────────────────

class TestMarketHours:
    def test_outside_market_hours_does_nothing(self):
        """_run_loop should skip checks when market is closed."""
        portfolio, history = _make_position()
        trader = _make_trader(portfolio=portfolio, trade_history=history)
        pm = PositionManager(trader=trader, interval=1)

        # Mock market hours to return False
        with patch.object(PositionManager, "_is_market_hours", return_value=False):
            # Run the loop briefly — it should not call check
            pm._shutdown = threading.Event()
            # Set shutdown after a short delay so the loop exits
            timer = threading.Timer(0.2, pm._shutdown.set)
            timer.start()
            pm._run_loop()
            timer.join()

        # Should never have fetched portfolio
        trader.get_portfolio.assert_not_called()

    def test_during_market_hours_checks_positions(self):
        """Positions should be checked during market hours."""
        portfolio, history = _make_position()
        trader = _make_trader(portfolio=portfolio, trade_history=history)
        pm = PositionManager(trader=trader, interval=1)

        with patch.object(PositionManager, "_is_market_hours", return_value=True), \
             patch.object(pm, "_fetch_current_price", return_value=205.0):
            pm._shutdown = threading.Event()
            timer = threading.Timer(0.2, pm._shutdown.set)
            timer.start()
            pm._run_loop()
            timer.join()

        # Should have checked portfolio at least once
        trader.get_portfolio.assert_called()

    @pytest.mark.parametrize("weekday_date", [
        "2026-03-28",  # Saturday
        "2026-03-29",  # Sunday
    ])
    def test_weekends_are_closed(self, weekday_date):
        """Saturday and Sunday should be outside market hours."""
        # Use a real datetime for the weekend at 16:00 UTC (would be open on weekday)
        from datetime import datetime as _dt
        weekend = _dt.fromisoformat(f"{weekday_date}T16:00:00+00:00")
        assert weekend.weekday() >= 5  # confirm it's actually a weekend

        with patch("monitoring.position_manager.datetime") as mock_dt_cls:
            mock_dt_cls.now.return_value = weekend
            assert PositionManager._is_market_hours() is False


# ── test: signal logging ─────────────────────────────────────────────

class TestSignalLogging:
    def test_stop_loss_logs_to_signal_events(self):
        """Stop-loss should log with strategy='PositionManager'."""
        portfolio, history = _make_position()
        trader = _make_trader(portfolio=portfolio, trade_history=history)
        pm = PositionManager(trader=trader)

        with patch.object(pm, "_fetch_current_price", return_value=185.0), \
             patch.object(pm._signal_logger, "log") as mock_log:
            pm._check_all_positions()

        mock_log.assert_called_once()
        logged = mock_log.call_args[0][0]
        assert logged["strategy"] == "PositionManager"
        assert logged["ticker"] == "AAPL"
        assert logged["signal"] == "SELL"

    def test_take_profit_logs_to_signal_events(self):
        """Take-profit should log with strategy='PositionManager'."""
        portfolio, history = _make_position(take_profit=220.0)
        trader = _make_trader(portfolio=portfolio, trade_history=history)
        pm = PositionManager(trader=trader)

        with patch.object(pm, "_fetch_current_price", return_value=225.0), \
             patch.object(pm._signal_logger, "log") as mock_log:
            pm._check_all_positions()

        mock_log.assert_called_once()
        logged = mock_log.call_args[0][0]
        assert logged["strategy"] == "PositionManager"
        assert logged["signal"] == "SELL"


# ── test: thread lifecycle ───────────────────────────────────────────

class TestLifecycle:
    def test_start_and_stop(self):
        """PositionManager starts and stops cleanly."""
        pm = PositionManager(trader=MagicMock(), interval=1)

        with patch.object(PositionManager, "_is_market_hours", return_value=False):
            pm.start()
            assert pm.is_running
            pm.stop()
            assert not pm.is_running

    def test_double_start_is_safe(self):
        """Starting twice should not create a second thread."""
        pm = PositionManager(trader=MagicMock(), interval=1)

        with patch.object(PositionManager, "_is_market_hours", return_value=False):
            pm.start()
            thread1 = pm._thread
            pm.start()  # should warn and return
            assert pm._thread is thread1
            pm.stop()


# ── test: no positions ───────────────────────────────────────────────

class TestNoPositions:
    def test_empty_portfolio_returns_empty(self):
        """No positions → no actions, no errors."""
        trader = _make_trader(portfolio=[])
        pm = PositionManager(trader=trader)
        results = pm._check_all_positions()
        assert results == []
        trader.track_trade.assert_not_called()
