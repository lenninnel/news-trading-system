"""
Alerts for the absence/failure classes an unattended system needs:

* PositionManager: IBKR connection cannot be re-established (once per
  outage + recovery) and broker-rejected exit orders (once per ticker).
* DailyScheduler run summary: orders that reached the broker but did
  not fill are listed, not silently dropped.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from monitoring.position_manager import PositionManager, _RECONNECT_ALERT_AFTER  # noqa: E402
from scheduler.daily_runner import DailyScheduler  # noqa: E402


def _pm(trader=None, notifier=None):
    db = MagicMock()
    db.get_trailing_stops.return_value = {}
    return PositionManager(trader=trader or MagicMock(), notifier=notifier or MagicMock(),
                           db=db, interval=60)


class TestReconnectOutageAlert:
    def test_alert_after_threshold_once_then_recovery(self):
        notifier = MagicMock()
        trader = MagicMock()
        trader.ensure_connected.return_value = False
        trader.is_connected.return_value = False
        pm = _pm(trader, notifier)

        for _ in range(_RECONNECT_ALERT_AFTER - 1):
            assert pm._get_open_positions() == []
        notifier.send_price_alert.assert_not_called()

        pm._get_open_positions()
        notifier.send_price_alert.assert_called_once()
        msg = notifier.send_price_alert.call_args[0][0]
        assert msg.startswith("🚨 IBKR CONNECTION LOST")
        assert f"{_RECONNECT_ALERT_AFTER} consecutive cycles" in msg
        assert "NOT enforced" in msg

        # Persisting outage → no repeat
        for _ in range(5):
            pm._get_open_positions()
        assert notifier.send_price_alert.call_count == 1

        # Recovery → exactly one restore message, counters reset
        trader.ensure_connected.return_value = True
        trader.is_connected.return_value = True
        trader.get_portfolio.return_value = []
        pm._get_open_positions()
        assert notifier.send_price_alert.call_count == 2
        assert notifier.send_price_alert.call_args[0][0].startswith("✅ IBKR connection restored")
        assert pm._reconnect_failures == 0 and pm._reconnect_alerted is False

        pm._get_open_positions()
        assert notifier.send_price_alert.call_count == 2

    def test_single_hiccup_is_silent(self):
        notifier = MagicMock()
        trader = MagicMock()
        trader.ensure_connected.side_effect = [False, True]
        trader.get_portfolio.return_value = []
        pm = _pm(trader, notifier)
        pm._get_open_positions()
        pm._get_open_positions()
        notifier.send_price_alert.assert_not_called()

    def test_reconnect_raising_counts_as_failure(self):
        notifier = MagicMock()
        trader = MagicMock()
        trader.ensure_connected.side_effect = RuntimeError("socket")
        pm = _pm(trader, notifier)
        for _ in range(_RECONNECT_ALERT_AFTER):
            pm._get_open_positions()
        assert notifier.send_price_alert.call_count == 1


class TestRejectedExitAlert:
    def _close(self, pm, result):
        return pm._handle_close_result(
            result, "AAPL", 150.0, -3.0, trigger="stop_loss", level=151.0,
            alert_emoji="🛑", alert_label="Stop-loss",
        )

    def test_cancelled_sell_alerts_once_per_ticker(self):
        notifier = MagicMock()
        pm = _pm(notifier=notifier)
        rejected = {"trade_id": None, "skipped": True, "outcome": "cancelled",
                    "skip_reason": "IBKR order cancelled: Order rejected - reason: no permission"}
        assert self._close(pm, rejected) is None
        notifier.send_price_alert.assert_called_once()
        msg = notifier.send_price_alert.call_args[0][0]
        assert msg.startswith("🚨 EXIT ORDER REJECTED: AAPL Stop-loss SELL")
        assert "no permission" in msg

        self._close(pm, rejected)                       # next cycle retry
        assert notifier.send_price_alert.call_count == 1

    def test_timeout_and_none_stay_quiet(self):
        notifier = MagicMock()
        pm = _pm(notifier=notifier)
        self._close(pm, {"trade_id": None, "skipped": True, "outcome": "timeout",
                         "skip_reason": "IBKR order timeout: Submitted"})
        self._close(pm, None)
        notifier.send_price_alert.assert_not_called()

    def test_fill_after_rejection_resets_dedupe(self):
        notifier = MagicMock()
        pm = _pm(notifier=notifier)
        rejected = {"trade_id": None, "skipped": True, "outcome": "cancelled", "skip_reason": "x"}
        self._close(pm, rejected)
        self._close(pm, {"trade_id": 7, "skipped": False})
        assert "AAPL" not in pm._exit_rejected_alerted
        self._close(pm, rejected)
        assert notifier.send_price_alert.call_count == 3   # rejected, filled, rejected again


class TestRunSummaryUnfilled:
    def _scheduler(self):
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}):
            s = DailyScheduler(full_watchlist=["AAPL"])
        s._tg = MagicMock()
        return s

    def test_unfilled_orders_listed(self):
        s = self._scheduler()
        batch = {
            "results": [
                {"ticker": "AAPL", "combined_signal": "STRONG BUY", "confidence": 0.8,
                 "execution": {"trade_id": None, "skipped": True, "outcome": "cancelled",
                               "action": "BUY",
                               "skip_reason": "IBKR order cancelled: Order rejected - reason: margin"}},
                {"ticker": "MSFT", "combined_signal": "WEAK BUY", "confidence": 0.6,
                 "execution": {"trade_id": None, "skipped": True, "outcome": "timeout",
                               "action": "BUY", "skip_reason": "IBKR order timeout: Submitted"}},
                {"ticker": "NVDA", "combined_signal": "HOLD", "confidence": 0.2,
                 "execution": {"trade_id": None, "skipped": True,
                               "skip_reason": "NVDA not supported on IBKR"}},
                {"ticker": "TSLA", "combined_signal": "STRONG BUY", "confidence": 0.9,
                 "execution": {"trade_id": 42}},
            ],
            "success_count": 4, "elapsed_s": 12.0,
        }
        s._send_run_summary("US_OPEN", batch, ["AAPL", "MSFT", "NVDA", "TSLA"])
        msg = s._tg._send.call_args[0][0]
        assert "Trades executed: 1" in msg
        assert "🚨 Orders NOT filled (2):" in msg
        assert "⚠️ AAPL BUY IBKR order cancelled: Order rejected - reason: margin" in msg
        assert "⚠️ MSFT BUY IBKR order timeout: Submitted" in msg
        assert "NVDA not supported" not in msg     # pre-broker skips stay quiet

    def test_no_unfilled_section_when_all_filled(self):
        s = self._scheduler()
        batch = {"results": [{"ticker": "AAPL", "combined_signal": "HOLD", "confidence": 0.1,
                              "execution": None}],
                 "success_count": 1, "elapsed_s": 1.0}
        s._send_run_summary("EOD", batch, ["AAPL"])
        assert "NOT filled" not in s._tg._send.call_args[0][0]
