"""
Shared pytest fixtures for the News Trading System test suite.

Provides mocked versions of all external dependencies so that every
unit test runs fully offline without real API keys.

Fixture overview
----------------
mock_anthropic_client   Patches anthropic.Anthropic; returns a fixed bullish response.
mock_newsapi            Patches requests.get for NewsAPI calls; returns fake headlines.
mock_yfinance           Patches yf.Ticker and yf.download; returns synthetic OHLCV.
tmp_db                  Temporary SQLite Database for each test function.
reset_api_recovery      Autouse: clears circuit-breaker and fallback-coordinator state.

Usage
-----
    def test_sentiment_run(mock_anthropic_client, tmp_db):
        agent = SentimentAgent()
        result = agent.run("Apple hits record high", "AAPL")
        assert result["sentiment"] == "bullish"
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Ensure project root is importable ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment variables before any project module imports
os.environ.setdefault("DB_PATH",           "/tmp/pytest_trading.db")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("NEWSAPI_KEY",       "test-key-not-real")
os.environ.setdefault("ALPHA_VANTAGE_KEY", "")


# ── Synthetic data builders ────────────────────────────────────────────────────

def make_ohlcv(
    ticker: str = "AAPL",
    days: int = 60,
    base_price: float = 150.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a deterministic synthetic OHLCV DataFrame (daily frequency)."""
    rng = np.random.default_rng(seed)
    dates  = pd.date_range(end="2025-01-15", periods=days, freq="B")
    close  = base_price + np.cumsum(rng.standard_normal(days) * 1.5)
    volume = np.full(days, 50_000_000, dtype=float)
    return pd.DataFrame(
        {
            "Open":   close * 0.99,
            "High":   close * 1.01,
            "Low":    close * 0.98,
            "Close":  close,
            "Volume": volume,
        },
        index=dates,
    )


def make_multiindex_download(tickers: list[str], days: int = 60) -> pd.DataFrame:
    """
    Simulate a yfinance multi-ticker download that returns a MultiIndex DataFrame.

    The returned DataFrame has column MultiIndex: (ticker, field).
    """
    frames = {}
    for i, t in enumerate(tickers):
        df = make_ohlcv(t, days=days, base_price=100.0 + i * 10)
        frames[t] = df

    if len(tickers) == 1:
        return frames[tickers[0]]

    # Build MultiIndex: level-0 = ticker, level-1 = field
    cols = pd.MultiIndex.from_product(
        [tickers, ["Open", "High", "Low", "Close", "Volume"]],
        names=["Ticker", "Price"],
    )
    combined = pd.DataFrame(index=frames[tickers[0]].index, columns=cols)
    for t in tickers:
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            combined[(t, col)] = frames[t][col]
    return combined


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_anthropic_client():
    """
    Patch anthropic.Anthropic so no real API call is made.

    The mock returns a fixed bullish JSON response for every call to
    ``client.messages.create()``.

    Yields the mock client so tests can inspect call args:
        assert mock_anthropic_client.messages.create.call_count == 1
    """
    canned_response = json.dumps({
        "sentiment": "bullish",
        "reason":    "Strong earnings beat reported by the company.",
    })
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=canned_response)]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg

    with patch("agents.sentiment_agent.anthropic.Anthropic", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_newsapi():
    """
    Patch requests.get for NewsAPI calls.

    Returns 3 fake bullish headlines.  Tests can override by replacing
    the ``side_effect`` on the yielded mock.
    """
    def _fake_response(url="", **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "status":       "ok",
            "totalResults": 3,
            "articles": [
                {"title": "AAPL beats earnings expectations by wide margin"},
                {"title": "AAPL shares surge on strong forward guidance"},
                {"title": "AAPL announces record quarterly revenue"},
            ],
        }
        return resp

    with patch("data.news_aggregator.requests.get", side_effect=_fake_response) as m:
        yield m


@pytest.fixture
def mock_yfinance():
    """
    Patch yfinance.Ticker and yfinance.download with synthetic data.

    yf.Ticker(ticker).info         → dict with currentPrice=150.0
    yf.Ticker(ticker).history()    → 60-row OHLCV DataFrame
    yf.download(tickers, ...)      → per-ticker OHLCV dict (single) or
                                     MultiIndex DataFrame (multiple)

    Yields the mock Ticker instance.
    """
    ohlcv = make_ohlcv()

    mock_ticker = MagicMock()
    mock_ticker.info = {
        "currentPrice":                150.0,
        "regularMarketPrice":          150.0,
        "currency":                    "USD",
        "longName":                    "Test Corp",
        "marketCap":                   2_000_000_000,
        "regularMarketChangePercent":  1.5,
    }
    mock_ticker.fast_info          = MagicMock(market_cap=2_000_000_000)
    mock_ticker.history.return_value = ohlcv

    with patch("yfinance.Ticker",   return_value=mock_ticker), \
         patch("yfinance.download", return_value=ohlcv):
        yield mock_ticker


@pytest.fixture
def tmp_db(tmp_path):
    """
    Isolated SQLite Database instance backed by a temporary file.

    The file is automatically deleted when the test ends.
    Yields the Database instance.
    """
    from storage.database import Database
    db = Database(str(tmp_path / "test.db"))
    yield db


@pytest.fixture(autouse=True)
def reset_shared_state():
    """
    Autouse fixture: resets all shared class-level state between tests.

    Clears:
    • APIRecovery circuit breakers
    • FallbackCoordinator registry
    • NetworkMonitor degraded-mode flag
    """
    from utils.api_recovery import APIRecovery
    from data.fallback_coordinator import FallbackCoordinator
    from utils.network_recovery import NetworkMonitor

    APIRecovery._circuits.clear()
    APIRecovery._db = None
    FallbackCoordinator.reset()
    NetworkMonitor._degraded      = False
    NetworkMonitor._offline_since = None
    NetworkMonitor._last_check_at = None

    yield

    # teardown
    APIRecovery._circuits.clear()
    FallbackCoordinator.reset()
