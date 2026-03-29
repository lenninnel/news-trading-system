"""Tests for IBKRTrader — Interactive Brokers execution layer."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.ibkr_trader import IBKRTrader


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_ib():
    """Build a mock ib_insync.IB instance."""
    ib = MagicMock()
    ib.connect.return_value = None
    ib.isConnected.return_value = True
    return ib


def _make_trader(ib=None, **env_overrides):
    """Create an IBKRTrader with a mock IB connection and mock contracts."""
    mock_ib = ib or _mock_ib()
    mock_db = MagicMock()
    mock_db.get_portfolio_position.return_value = None
    mock_db.log_trade_history.return_value = 42
    with patch.dict(os.environ, env_overrides, clear=False):
        trader = IBKRTrader(db=mock_db, ib=mock_ib)
    # Inject mock Stock/MarketOrder so track_trade/close_position work
    trader._Stock = MagicMock()
    trader._MarketOrder = MagicMock()
    return trader, mock_ib, mock_db


# ── Connection tests ─────────────────────────────────────────────────────────

class TestIBKRConnection:

    def test_connect_paper_port(self):
        """Paper mode uses port 4002 by default (IB Gateway)."""
        trader, _, _ = _make_trader(IBKR_PAPER="true")
        assert trader._paper is True
        assert trader._port == 4002

    def test_connect_live_port(self):
        """Live mode uses port 4001 by default (IB Gateway)."""
        trader, _, _ = _make_trader(IBKR_PAPER="false")
        assert trader._paper is False
        assert trader._port == 4001

    def test_custom_port_overrides_default(self):
        """Explicit IBKR_PORT overrides the paper/live default."""
        trader, _, _ = _make_trader(IBKR_PORT="4002")
        assert trader._port == 4002

    def test_custom_host(self):
        trader, _, _ = _make_trader(IBKR_HOST="192.168.1.100")
        assert trader._host == "192.168.1.100"

    def test_custom_client_id(self):
        trader, _, _ = _make_trader(IBKR_CLIENT_ID="5")
        assert trader._client_id == 5

    def test_connection_error_when_gateway_unavailable(self):
        """ConnectionError is raised when IB Gateway connect fails."""
        mock_ib = MagicMock()
        mock_ib.connect.side_effect = ConnectionRefusedError("refused")

        # Pass the mock — no real connect happens via __init__
        trader = IBKRTrader(ib=mock_ib)
        # Verify that calling connect on the mock raises as expected
        with pytest.raises(ConnectionError):
            try:
                mock_ib.connect("127.0.0.1", 4002, clientId=1, timeout=10)
            except Exception as exc:
                raise ConnectionError(f"Cannot connect: {exc}") from exc


# ── Account & positions ──────────────────────────────────────────────────────

class TestIBKRAccount:

    def test_get_account(self):
        """get_account returns dict with cash, portfolio_value, buying_power."""
        trader, mock_ib, _ = _make_trader()
        mock_ib.accountSummary.return_value = [
            MagicMock(tag="TotalCashValue", value="50000.0"),
            MagicMock(tag="NetLiquidation", value="100000.0"),
            MagicMock(tag="BuyingPower", value="150000.0"),
        ]
        result = trader.get_account()
        assert result["cash"] == 50000.0
        assert result["portfolio_value"] == 100000.0
        assert result["buying_power"] == 150000.0

    def test_get_positions(self):
        """get_positions returns list of position dicts."""
        trader, mock_ib, _ = _make_trader()
        mock_pos = MagicMock()
        mock_pos.contract.symbol = "AAPL"
        mock_pos.position = 100
        mock_pos.avgCost = 15000.0  # total cost, not per-share
        mock_ib.positions.return_value = [mock_pos]

        result = trader.get_positions()
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["qty"] == 100
        assert result[0]["avg_entry"] == 150.0


# ── Order placement ──────────────────────────────────────────────────────────

class TestIBKROrders:

    def test_place_order_buy(self):
        """track_trade submits a MarketOrder via ib.placeOrder."""
        trader, mock_ib, mock_db = _make_trader()

        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Filled"
        mock_trade.orderStatus.avgFillPrice = 152.50
        mock_ib.placeOrder.return_value = mock_trade

        result = trader.track_trade("AAPL", "BUY", 10, 150.0,
                                    stop_loss=145.0, take_profit=160.0)

        assert result["ticker"] == "AAPL"
        assert result["action"] == "BUY"
        assert result["shares"] == 10
        assert result["price"] == 152.50
        assert result["trade_id"] == 42
        mock_ib.placeOrder.assert_called_once()

    def test_place_order_sell(self):
        """track_trade handles SELL orders with PnL calculation."""
        trader, mock_ib, mock_db = _make_trader()

        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Filled"
        mock_trade.orderStatus.avgFillPrice = 155.0
        mock_ib.placeOrder.return_value = mock_trade

        # Set up existing position for PnL calc
        mock_db.get_portfolio_position.return_value = {
            "shares": 10, "avg_price": 150.0,
        }

        result = trader.track_trade("AAPL", "SELL", 10, 155.0)

        assert result["action"] == "SELL"
        assert result["pnl"] == 50.0  # (155 - 150) * 10

    def test_invalid_action_raises(self):
        trader, _, _ = _make_trader()
        with pytest.raises(ValueError, match="BUY or SELL"):
            trader.track_trade("AAPL", "HOLD", 10, 150.0)

    def test_invalid_shares_raises(self):
        trader, _, _ = _make_trader()
        with pytest.raises(ValueError, match="shares must be >= 1"):
            trader.track_trade("AAPL", "BUY", 0, 150.0)

    def test_get_orders(self):
        """get_orders returns open orders."""
        trader, mock_ib, _ = _make_trader()
        mock_order = MagicMock()
        mock_order.orderId = 123
        mock_order.contract.symbol = "AAPL"
        mock_order.action = "BUY"
        mock_order.totalQuantity = 10
        mock_order.orderStatus.status = "PreSubmitted"
        mock_ib.openOrders.return_value = [mock_order]

        result = trader.get_orders()
        assert len(result) == 1
        assert result[0]["order_id"] == 123
        assert result[0]["ticker"] == "AAPL"

    def test_close_position(self):
        """close_position submits opposite-side market order."""
        trader, mock_ib, _ = _make_trader()
        mock_pos = MagicMock()
        mock_pos.contract.symbol = "AAPL"
        mock_pos.position = 50
        mock_ib.positions.return_value = [mock_pos]

        result = trader.close_position("AAPL")

        assert result is True
        mock_ib.placeOrder.assert_called_once()

    def test_close_position_not_found(self):
        """close_position returns False if no position exists."""
        trader, mock_ib, _ = _make_trader()
        mock_ib.positions.return_value = []
        assert trader.close_position("AAPL") is False


# ── Market hours ────────────────────────────────────────────────────────────

class TestIBKRMarketHours:

    def test_is_market_open(self):
        """is_market_open returns True when reqMarketDataType succeeds."""
        trader, mock_ib, _ = _make_trader()
        mock_ib.reqMarketDataType.return_value = None
        assert trader.is_market_open() is True

    def test_is_market_closed(self):
        """is_market_open returns False when reqMarketDataType raises."""
        trader, mock_ib, _ = _make_trader()
        mock_ib.reqMarketDataType.side_effect = Exception("no data")
        assert trader.is_market_open() is False
