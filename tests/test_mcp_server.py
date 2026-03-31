"""Tests for the NTS MCP server (mcp_server.nts_mcp)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from mcp_server.nts_mcp import (
    get_signals,
    get_portfolio,
    get_status,
    get_signal_detail,
    get_performance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SIGNALS = [
    {
        "id": 1,
        "timestamp": "2026-03-30T14:30:00",
        "session": "US_OPEN",
        "ticker": "AAPL",
        "strategy": "Combined",
        "signal": "STRONG BUY",
        "confidence": 85,
        "rsi": 42.5,
        "sma_ratio": 1.02,
        "volume_ratio": 1.5,
        "sentiment_score": 0.8,
        "news_score": 0.75,
        "social_score": 0.6,
        "bull_case": "Strong earnings beat, raised guidance.",
        "bear_case": "Valuation stretched at 30x forward PE.",
        "debate_outcome": "BULLISH",
        "price_at_signal": 195.50,
        "outcome_3d_pct": 2.1,
        "outcome_5d_pct": 3.5,
        "outcome_10d_pct": None,
        "trade_executed": True,
    },
    {
        "id": 2,
        "timestamp": "2026-03-29T07:00:00",
        "session": "XETRA_OPEN",
        "ticker": "MSFT",
        "strategy": "NewsCatalyst",
        "signal": "BUY",
        "confidence": 72,
        "rsi": 55.0,
        "sma_ratio": 1.01,
        "volume_ratio": 1.2,
        "sentiment_score": 0.6,
        "news_score": 0.7,
        "social_score": None,
        "bull_case": "Azure growth accelerating.",
        "bear_case": "AI capex spend may not pay off.",
        "debate_outcome": "NEUTRAL",
        "price_at_signal": 420.00,
        "outcome_3d_pct": None,
        "outcome_5d_pct": None,
        "outcome_10d_pct": None,
        "trade_executed": False,
    },
]

SAMPLE_PORTFOLIO = {
    "value": 10250.50,
    "daily_pnl": 125.30,
    "daily_pnl_pct": 1.24,
    "positions": [
        {
            "ticker": "AAPL",
            "shares": 10,
            "entry": 190.00,
            "current": 195.50,
            "pnl_pct": 2.9,
        },
    ],
    "cash": 8295.50,
}

SAMPLE_STATUS = {
    "running": True,
    "uptime_seconds": 7200,
    "last_session": "US_OPEN",
    "last_run_at": "2026-03-30T14:30:00",
    "next_session": "MIDDAY",
    "next_run_at": "2026-03-30T18:00:00+00:00",
    "watchlist": ["AAPL", "MSFT", "NVDA"],
    "mode": "paper_local",
}

SAMPLE_PERFORMANCE = {
    "total_trades": 42,
    "win_rate": 64.3,
    "sharpe": None,
    "max_drawdown": None,
    "signals_today": 8,
    "strong_buy_today": 2,
    "sessions_today": 3,
    "sessions_total": 4,
}


def _mock_api_get(responses: dict):
    """Return an AsyncMock that dispatches on the path argument."""
    async def _side_effect(path: str, params=None):
        return responses.get(path, [])
    return _side_effect


# ---------------------------------------------------------------------------
# get_signals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_signals_returns_formatted_data():
    with patch("mcp_server.nts_mcp._api_get", new=_mock_api_get({
        "/api/signals": SAMPLE_SIGNALS,
    })):
        result = await get_signals(days=7)

    assert "Found 2 signal(s)" in result
    assert "AAPL" in result
    assert "MSFT" in result
    assert "STRONG BUY" in result


@pytest.mark.asyncio
async def test_get_signals_strategy_filter():
    with patch("mcp_server.nts_mcp._api_get", new=_mock_api_get({
        "/api/signals": SAMPLE_SIGNALS,
    })):
        result = await get_signals(days=7, strategy="Combined")

    assert "Found 1 signal(s)" in result
    assert "AAPL" in result
    assert "MSFT" not in result.split("Full JSON")[0]


@pytest.mark.asyncio
async def test_get_signals_empty():
    with patch("mcp_server.nts_mcp._api_get", new=_mock_api_get({
        "/api/signals": [],
    })):
        result = await get_signals(days=7)

    assert "No signals found" in result


# ---------------------------------------------------------------------------
# get_portfolio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_portfolio_with_positions():
    with patch("mcp_server.nts_mcp._api_get", new=_mock_api_get({
        "/api/portfolio": SAMPLE_PORTFOLIO,
    })):
        result = await get_portfolio()

    assert "Portfolio Summary" in result
    assert "$10,250.50" in result
    assert "AAPL" in result
    assert "+2.9%" in result


@pytest.mark.asyncio
async def test_get_portfolio_handles_empty():
    empty_portfolio = {
        "value": 0,
        "daily_pnl": 0,
        "daily_pnl_pct": 0,
        "positions": [],
        "cash": 10000.0,
    }
    with patch("mcp_server.nts_mcp._api_get", new=_mock_api_get({
        "/api/portfolio": empty_portfolio,
    })):
        result = await get_portfolio()

    assert "No open positions" in result
    assert "$10,000.00" in result


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_status_shows_next_session():
    with patch("mcp_server.nts_mcp._api_get", new=_mock_api_get({
        "/api/status": SAMPLE_STATUS,
    })):
        result = await get_status()

    assert "NTS System Status" in result
    assert "MIDDAY" in result
    assert "US_OPEN" in result
    assert "AAPL" in result
    assert "2h 0m" in result


@pytest.mark.asyncio
async def test_get_status_connection_error():
    async def _raise_connect(*args, **kwargs):
        raise __import__("httpx").ConnectError("Connection refused")

    with patch("mcp_server.nts_mcp._api_get", side_effect=_raise_connect):
        result = await get_status()

    assert "Cannot reach NTS API" in result


# ---------------------------------------------------------------------------
# get_performance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_performance_metrics():
    with patch("mcp_server.nts_mcp._api_get", new=_mock_api_get({
        "/api/performance": SAMPLE_PERFORMANCE,
    })):
        result = await get_performance()

    assert "Trading Performance" in result
    assert "42" in result
    assert "64.3%" in result
    assert "3 / 4" in result


# ---------------------------------------------------------------------------
# get_signal_detail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_signal_detail_with_debate():
    with patch("mcp_server.nts_mcp._api_get", new=_mock_api_get({
        "/api/signals": [SAMPLE_SIGNALS[0]],
    })):
        result = await get_signal_detail(ticker="AAPL")

    assert "Signal Detail: AAPL" in result
    assert "Bull/Bear Debate" in result
    assert "Strong earnings beat" in result
    assert "Valuation stretched" in result
    assert "BULLISH" in result
    assert "3d=+2.10%" in result


@pytest.mark.asyncio
async def test_get_signal_detail_no_signals():
    with patch("mcp_server.nts_mcp._api_get", new=_mock_api_get({
        "/api/signals": [],
    })):
        result = await get_signal_detail(ticker="XYZ")

    assert "No recent signals found for XYZ" in result
