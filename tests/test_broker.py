"""
Tests for the broker integration layer.

Covers:
  - AlpacaTrader paper order placement (mocked API)
  - Live trading gate (missing LIVE_TRADING_CONFIRMED raises RuntimeError)
  - broker_factory mode switching
"""

from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from execution.broker_factory import create_trader, get_trading_mode
from storage.database import Database


# ── helpers ──────────────────────────────────────────────────────────

def _make_order(status="filled", filled_avg_price="243.75"):
    """Return a mock Alpaca order object."""
    return SimpleNamespace(
        id="order-abc-123",
        status=status,
        filled_avg_price=filled_avg_price,
    )


def _make_position(symbol="AAPL", qty="10", avg_entry_price="241.50",
                   market_value="2437.50"):
    return SimpleNamespace(
        symbol=symbol,
        qty=qty,
        avg_entry_price=avg_entry_price,
        market_value=market_value,
    )


def _mock_api():
    """Return a MagicMock that behaves like tradeapi.REST."""
    api = MagicMock()
    order = _make_order()
    api.submit_order.return_value = order
    api.get_order.return_value = order
    api.list_positions.return_value = [_make_position()]
    # SELL pre-check (track_trade) calls get_position(ticker) — return a
    # plausible non-zero position so existing happy-path SELL tests don't
    # trip the ghost-cleanup branch.
    api.get_position.return_value = _make_position()
    # Live-price divergence check expects a realistic quote
    api.get_latest_trade.return_value = SimpleNamespace(price=243.75)
    return api


def _alpaca_not_found_error():
    """Build the exception alpaca_trade_api raises when a position is missing."""
    class _APIError(Exception):
        pass
    return _APIError("position does not exist")


# Force-inject a fake alpaca_trade_api module to prevent real API connections.
# CRITICAL: use direct assignment, NOT setdefault — setdefault silently fails
# if the real module was already imported by another test.
_fake_tradeapi = MagicMock()
sys.modules["alpaca_trade_api"] = _fake_tradeapi

from execution.alpaca_trader import AlpacaTrader  # noqa: E402


# ── broker_factory tests ────────────────────────────────────────────

class TestBrokerFactory:

    def test_default_is_paper_local(self):
        env = {"TRADING_MODE": "paper_local"}
        with patch.dict(os.environ, env, clear=False):
            trader = create_trader()
        from execution.paper_trader import PaperTrader
        assert isinstance(trader, PaperTrader)

    def test_missing_env_defaults_to_paper_local(self):
        with patch.dict(os.environ, {}, clear=True):
            mode = get_trading_mode()
        assert mode == "paper_local"

    def test_invalid_mode_raises(self):
        with patch.dict(os.environ, {"TRADING_MODE": "yolo"}, clear=False):
            with pytest.raises(ValueError, match="TRADING_MODE must be one of"):
                get_trading_mode()

    @patch("execution.alpaca_trader._is_pytest_running", return_value=False)
    def test_alpaca_paper_returns_alpaca_trader(self, _mock_guard):
        env = {
            "TRADING_MODE": "alpaca_paper",
            "ALPACA_API_KEY": "test-key",
            "ALPACA_SECRET_KEY": "test-secret",
        }
        _fake_tradeapi.REST.return_value = _mock_api()
        with patch.dict(os.environ, env, clear=False):
            trader = create_trader()
        assert isinstance(trader, AlpacaTrader)

    def test_alpaca_live_blocked_without_confirmation(self):
        env = {
            "TRADING_MODE": "alpaca_live",
            "ALPACA_API_KEY": "test-key",
            "ALPACA_SECRET_KEY": "test-secret",
            "LIVE_TRADING_CONFIRMED": "",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(RuntimeError, match="LIVE TRADING BLOCKED"):
                create_trader()

    def test_alpaca_live_blocked_with_wrong_value(self):
        env = {
            "TRADING_MODE": "alpaca_live",
            "ALPACA_API_KEY": "test-key",
            "ALPACA_SECRET_KEY": "test-secret",
            "LIVE_TRADING_CONFIRMED": "yes",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(RuntimeError, match="LIVE TRADING BLOCKED"):
                create_trader()

    @patch("execution.alpaca_trader._is_pytest_running", return_value=False)
    def test_alpaca_live_allowed_with_confirmation(self, _mock_guard):
        env = {
            "TRADING_MODE": "alpaca_live",
            "ALPACA_API_KEY": "test-key",
            "ALPACA_SECRET_KEY": "test-secret",
            "LIVE_TRADING_CONFIRMED": "true",
        }
        _fake_tradeapi.REST.return_value = _mock_api()
        with patch.dict(os.environ, env, clear=False):
            trader = create_trader()
        assert isinstance(trader, AlpacaTrader)


# ── AlpacaTrader tests ──────────────────────────────────────────────

class TestAlpacaTrader:

    def _make_trader(self):
        db = Database(db_path=tempfile.mktemp(suffix=".db"))
        return AlpacaTrader(db=db, api=_mock_api())

    def test_buy_order_placement(self):
        trader = self._make_trader()
        result = trader.track_trade(
            "AAPL", "BUY", 10, 243.00,
            stop_loss=238.00, take_profit=253.00,
        )

        assert result["ticker"] == "AAPL"
        assert result["action"] == "BUY"
        assert result["shares"] == 10
        assert result["price"] == 243.75  # filled price from mock
        assert result["pnl"] == 0.0
        assert result["total_value"] == round(10 * 243.75, 2)

        # Verify bracket order was submitted
        api = trader._api
        call_kwargs = api.submit_order.call_args
        assert call_kwargs[1]["order_class"] == "bracket"
        assert call_kwargs[1]["stop_loss"] == {"stop_price": "238.0"}
        assert call_kwargs[1]["take_profit"] == {"limit_price": "253.0"}

    def test_sell_order_with_pnl(self):
        trader = self._make_trader()
        # Buy at filled price 243.75
        trader.track_trade("AAPL", "BUY", 10, 243.00,
                           stop_loss=238.0, take_profit=253.0)

        # Change mock fill price for the sell
        sell_order = _make_order(filled_avg_price="248.50")
        trader._api.submit_order.return_value = sell_order
        trader._api.get_order.return_value = sell_order

        result = trader.track_trade("AAPL", "SELL", 10, 248.50)
        assert result["action"] == "SELL"
        # PnL = (248.50 - 243.75) * 10 = 47.50
        assert result["pnl"] == 47.50

    def test_market_order_without_bracket(self):
        """BUY without stop_loss/take_profit is now rejected."""
        trader = self._make_trader()
        with pytest.raises(ValueError, match="requires valid stop_loss"):
            trader.track_trade("AAPL", "BUY", 5, 243.00)

    def test_invalid_action_raises(self):
        trader = self._make_trader()
        with pytest.raises(ValueError, match="action must be BUY or SELL"):
            trader.track_trade("AAPL", "HOLD", 10, 243.00)

    def test_invalid_shares_raises(self):
        trader = self._make_trader()
        with pytest.raises(ValueError, match="shares must be >= 1"):
            trader.track_trade("AAPL", "BUY", 0, 243.00)

    def test_get_portfolio_syncs_from_alpaca(self):
        trader = self._make_trader()
        positions = trader.get_portfolio()

        assert len(positions) == 1
        assert positions[0]["ticker"] == "AAPL"
        assert positions[0]["shares"] == 10
        assert positions[0]["avg_price"] == 241.50

    def test_get_trade_history_delegates_to_db(self):
        trader = self._make_trader()
        trader.track_trade("AAPL", "BUY", 5, 243.00,
                           stop_loss=238.0, take_profit=253.0)
        history = trader.get_trade_history(ticker="AAPL")
        assert len(history) >= 1
        assert history[0]["ticker"] == "AAPL"

    def test_order_timeout_uses_fallback_price(self):
        """When the order never fills, fall back to the quoted price."""
        api = _mock_api()
        api.get_order.return_value = SimpleNamespace(
            status="new", filled_avg_price=None,
        )
        db = Database(db_path=tempfile.mktemp(suffix=".db"))
        trader = AlpacaTrader(db=db, api=api)

        with patch("execution.alpaca_trader._FILL_TIMEOUT", 0):
            result = trader.track_trade("AAPL", "BUY", 5, 242.99,
                                        stop_loss=238.0, take_profit=253.0)

        assert result["price"] == 242.99

    def test_rejected_order_uses_fallback_price(self):
        """When the order is rejected, fall back to the quoted price."""
        api = _mock_api()
        api.get_order.return_value = SimpleNamespace(
            status="rejected", filled_avg_price=None,
        )
        db = Database(db_path=tempfile.mktemp(suffix=".db"))
        trader = AlpacaTrader(db=db, api=api)
        result = trader.track_trade("AAPL", "BUY", 5, 242.99,
                                    stop_loss=238.0, take_profit=253.0)
        assert result["price"] == 242.99


# ── unsupported ticker tests ────────────────────────────────────────

class TestUnsupportedTickers:

    def _make_trader(self):
        api = _mock_api()
        db = Database(db_path=tempfile.mktemp(suffix=".db"))
        return AlpacaTrader(db=db, api=api), api

    def test_xetra_ticker_skipped_without_error(self):
        trader, api = self._make_trader()
        result = trader.track_trade("SAP.XETRA", "BUY", 5, 250.00,
                                    stop_loss=240.0, take_profit=270.0)
        assert result["skipped"] is True
        assert "not supported" in result["skip_reason"]
        # No order should have been submitted
        api.submit_order.assert_not_called()

    def test_unsupported_returns_correct_dict_shape(self):
        trader, _ = self._make_trader()
        result = trader.track_trade("SIE.XETRA", "BUY", 3, 180.00,
                                    stop_loss=170.0, take_profit=195.0)
        assert result["trade_id"] is None
        assert result["ticker"] == "SIE.XETRA"
        assert result["action"] == "BUY"
        assert result["shares"] == 3
        assert result["pnl"] == 0.0
        assert result["total_value"] == 0.0

    def test_unsupported_no_telegram_error(self):
        trader, api = self._make_trader()
        trader.track_trade("SAP.XETRA", "BUY", 5, 250.00,
                           stop_loss=240.0, take_profit=270.0)
        # _notify_trade_failed should not be called
        # (it's only called on real order submission errors)
        api.submit_order.assert_not_called()

    def test_supported_ticker_still_executes(self):
        trader, api = self._make_trader()
        result = trader.track_trade("AAPL", "BUY", 10, 243.00,
                                    stop_loss=238.0, take_profit=253.0)
        assert result.get("skipped") is not True
        api.submit_order.assert_called_once()


# ── Ghost-position reconciliation (BUG: NBIS sell hit Alpaca with 0 shares) ──


class TestGhostPositionReconciliation:
    """SELL must be aborted if Alpaca shows zero shares for the ticker.

    Local DB can drift out of sync with Alpaca after a bracket exit
    (Alpaca's stop-loss closed the position but the local position row
    is still around). The next stop-loss tick on our side would then ask
    Alpaca to sell shares it no longer holds, which fails with
    "insufficient qty available for order". The fix: pre-check via
    get_position(); if Alpaca says 0, clean the local row and skip.
    """

    def _make_trader(self, *, position_qty: "str | None" = "8"):
        api = _mock_api()
        if position_qty is None:
            api.get_position.side_effect = _alpaca_not_found_error()
        else:
            api.get_position.return_value = _make_position(
                symbol="NBIS", qty=position_qty, avg_entry_price="55.00",
                market_value=str(float(position_qty) * 55.00),
            )
        db = Database(db_path=tempfile.mktemp(suffix=".db"))
        # Pre-seed the local DB with a fake NBIS position so we can prove
        # the ghost cleaner deletes it.
        db.set_portfolio_position(
            ticker="NBIS", shares=8, avg_price=55.00, current_value=440.00,
        )
        return AlpacaTrader(db=db, api=api), api, db

    def test_sell_with_zero_alpaca_qty_skips_and_cleans_local(self):
        trader, api, db = self._make_trader(position_qty=None)

        result = trader.track_trade("NBIS", "SELL", 8, 55.00)

        assert result["skipped"] is True
        assert "ghost position" in result["skip_reason"].lower()
        assert result["trade_id"] is None
        assert result["pnl"] == 0.0
        # No Alpaca order should have been sent
        api.submit_order.assert_not_called()
        # Local row should be gone
        assert db.get_portfolio_position("NBIS") is None

    def test_sell_with_nonzero_alpaca_qty_proceeds(self):
        trader, api, db = self._make_trader(position_qty="8")

        # Configure submit_order / get_order for the SELL fill
        sell_order = _make_order(filled_avg_price="55.50")
        api.submit_order.return_value = sell_order
        api.get_order.return_value = sell_order
        api.get_latest_trade.return_value = SimpleNamespace(price=55.50)

        result = trader.track_trade("NBIS", "SELL", 8, 55.50)

        assert result.get("skipped") is not True
        api.submit_order.assert_called_once()
        # Local row should be cleaned by the normal sync (shares=0)
        assert db.get_portfolio_position("NBIS") is None

    def test_get_alpaca_position_qty_returns_zero_on_not_found(self):
        api = _mock_api()
        api.get_position.side_effect = Exception("position does not exist")
        db = Database(db_path=tempfile.mktemp(suffix=".db"))
        trader = AlpacaTrader(db=db, api=api)
        assert trader._get_alpaca_position_qty("NBIS") == 0

    def test_get_alpaca_position_qty_fails_open_on_other_errors(self):
        """Network / auth errors must NOT be treated as 'no position'."""
        api = _mock_api()
        api.get_position.side_effect = Exception("connection refused")
        db = Database(db_path=tempfile.mktemp(suffix=".db"))
        trader = AlpacaTrader(db=db, api=api)
        # None means "unknown" — caller should fall through to normal sell
        assert trader._get_alpaca_position_qty("NBIS") is None

    def test_ghost_cleanup_does_not_block_other_tickers(self):
        """A ghost on NBIS must not affect a BUY/SELL on a different ticker."""
        trader, api, _ = self._make_trader(position_qty="10")  # AAPL has 10
        api.get_position.return_value = _make_position()  # AAPL position OK

        result = trader.track_trade(
            "AAPL", "BUY", 5, 243.00,
            stop_loss=238.0, take_profit=253.0,
        )
        assert result.get("skipped") is not True
        api.submit_order.assert_called_once()


# ── get_trading_mode tests ──────────────────────────────────────────

class TestGetTradingMode:

    def test_paper_local(self):
        with patch.dict(os.environ, {"TRADING_MODE": "paper_local"}):
            assert get_trading_mode() == "paper_local"

    def test_alpaca_paper(self):
        with patch.dict(os.environ, {"TRADING_MODE": "alpaca_paper"}):
            assert get_trading_mode() == "alpaca_paper"

    def test_alpaca_live(self):
        with patch.dict(os.environ, {"TRADING_MODE": "alpaca_live"}):
            assert get_trading_mode() == "alpaca_live"

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"TRADING_MODE": "ALPACA_PAPER"}):
            assert get_trading_mode() == "alpaca_paper"
