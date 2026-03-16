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

def _make_order(status="filled", filled_avg_price="150.25"):
    """Return a mock Alpaca order object."""
    return SimpleNamespace(
        id="order-abc-123",
        status=status,
        filled_avg_price=filled_avg_price,
    )


def _make_position(symbol="AAPL", qty="10", avg_entry_price="148.50",
                   market_value="1502.50"):
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
    return api


# Install a fake alpaca_trade_api module so imports succeed without pip
_fake_tradeapi = MagicMock()
sys.modules.setdefault("alpaca_trade_api", _fake_tradeapi)

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

    def test_alpaca_paper_returns_alpaca_trader(self):
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

    def test_alpaca_live_allowed_with_confirmation(self):
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
            "AAPL", "BUY", 10, 150.00,
            stop_loss=145.00, take_profit=160.00,
        )

        assert result["ticker"] == "AAPL"
        assert result["action"] == "BUY"
        assert result["shares"] == 10
        assert result["price"] == 150.25  # filled price from mock
        assert result["pnl"] == 0.0
        assert result["total_value"] == round(10 * 150.25, 2)

        # Verify bracket order was submitted
        api = trader._api
        call_kwargs = api.submit_order.call_args
        assert call_kwargs[1]["order_class"] == "bracket"
        assert call_kwargs[1]["stop_loss"] == {"stop_price": "145.0"}
        assert call_kwargs[1]["take_profit"] == {"limit_price": "160.0"}

    def test_sell_order_with_pnl(self):
        trader = self._make_trader()
        # Buy at filled price 150.25
        trader.track_trade("AAPL", "BUY", 10, 150.00,
                           stop_loss=145.0, take_profit=160.0)

        # Change mock fill price for the sell
        sell_order = _make_order(filled_avg_price="155.00")
        trader._api.submit_order.return_value = sell_order
        trader._api.get_order.return_value = sell_order

        result = trader.track_trade("AAPL", "SELL", 10, 155.00)
        assert result["action"] == "SELL"
        # PnL = (155.00 - 150.25) * 10 = 47.50
        assert result["pnl"] == 47.50

    def test_market_order_without_bracket(self):
        """BUY without stop_loss/take_profit is now rejected."""
        trader = self._make_trader()
        with pytest.raises(ValueError, match="requires valid stop_loss"):
            trader.track_trade("AAPL", "BUY", 5, 150.00)

    def test_invalid_action_raises(self):
        trader = self._make_trader()
        with pytest.raises(ValueError, match="action must be BUY or SELL"):
            trader.track_trade("AAPL", "HOLD", 10, 150.00)

    def test_invalid_shares_raises(self):
        trader = self._make_trader()
        with pytest.raises(ValueError, match="shares must be >= 1"):
            trader.track_trade("AAPL", "BUY", 0, 150.00)

    def test_get_portfolio_syncs_from_alpaca(self):
        trader = self._make_trader()
        positions = trader.get_portfolio()

        assert len(positions) == 1
        assert positions[0]["ticker"] == "AAPL"
        assert positions[0]["shares"] == 10
        assert positions[0]["avg_price"] == 148.50

    def test_get_trade_history_delegates_to_db(self):
        trader = self._make_trader()
        trader.track_trade("AAPL", "BUY", 5, 150.00,
                           stop_loss=145.0, take_profit=160.0)
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
            result = trader.track_trade("AAPL", "BUY", 5, 149.99,
                                        stop_loss=145.0, take_profit=160.0)

        assert result["price"] == 149.99

    def test_rejected_order_uses_fallback_price(self):
        """When the order is rejected, fall back to the quoted price."""
        api = _mock_api()
        api.get_order.return_value = SimpleNamespace(
            status="rejected", filled_avg_price=None,
        )
        db = Database(db_path=tempfile.mktemp(suffix=".db"))
        trader = AlpacaTrader(db=db, api=api)
        result = trader.track_trade("AAPL", "BUY", 5, 149.99,
                                    stop_loss=145.0, take_profit=160.0)
        assert result["price"] == 149.99


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
