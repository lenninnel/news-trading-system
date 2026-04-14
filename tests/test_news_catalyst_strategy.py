"""Tests for the NewsCatalyst strategy."""

import pandas as pd
import numpy as np
from strategies.news_catalyst import NewsCatalystStrategy


def _make_bars(days=30, base_price=100.0, last_volume_mult=1.5, last_pct_change=1.0):
    """Build synthetic OHLCV bars for testing."""
    dates = pd.date_range(end="2026-03-25", periods=days, freq="B")
    np.random.seed(42)
    prices = base_price + np.random.randn(days).cumsum() * 0.5
    # Ensure last bar has the desired % change
    if days >= 2:
        prices[-1] = prices[-2] * (1 + last_pct_change / 100)
    volumes = np.full(days, 1_000_000)
    volumes[-1] = int(volumes[-1] * last_volume_mult)
    return pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.01,
        "Low": prices * 0.99,
        "Close": prices,
        "Volume": volumes,
    }, index=dates)


class TestNewsCatalystStrategy:
    def setup_method(self):
        self.strategy = NewsCatalystStrategy()

    def test_name(self):
        assert self.strategy.name == "NewsCatalyst"

    def test_all_conditions_met_buy(self):
        bars = _make_bars(last_volume_mult=1.5, last_pct_change=1.0)
        news = {"news_score": 0.80, "headline_count": 3, "sentiment_direction": "BUY"}
        result = self.strategy.analyze("AAPL", bars, "BUY", news_data=news)
        assert result.signal == "BUY"
        assert 55.0 <= result.confidence <= 75.0
        assert result.strategy_name == "NewsCatalyst"

    def test_news_only_no_price_reaction(self):
        bars = _make_bars(last_volume_mult=0.8, last_pct_change=0.1)
        news = {"news_score": 0.80, "headline_count": 3, "sentiment_direction": "BUY"}
        result = self.strategy.analyze("AAPL", bars, "BUY", news_data=news)
        assert result.signal == "WEAK BUY"
        assert 35.0 <= result.confidence <= 55.0

    def test_no_news_data(self):
        bars = _make_bars()
        result = self.strategy.analyze("AAPL", bars, "HOLD")
        assert result.signal == "HOLD"
        assert result.confidence == 25.0

    def test_negative_news_sell(self):
        bars = _make_bars(last_volume_mult=1.5, last_pct_change=-1.5)
        news = {"news_score": 0.80, "headline_count": 2, "sentiment_direction": "SELL"}
        result = self.strategy.analyze("AAPL", bars, "SELL", news_data=news)
        assert result.signal == "SELL"
        assert 55.0 <= result.confidence <= 75.0

    def test_missing_news_data_none(self):
        bars = _make_bars()
        result = self.strategy.analyze("AAPL", bars, "BUY", news_data=None)
        assert result.signal == "HOLD"
        assert result.confidence == 25.0

    def test_insufficient_bars(self):
        bars = pd.DataFrame({"Close": [100], "Volume": [1000], "Open": [99], "High": [101], "Low": [99]})
        news = {"news_score": 0.80, "headline_count": 3, "sentiment_direction": "BUY"}
        result = self.strategy.analyze("AAPL", bars, "BUY", news_data=news)
        assert result.signal == "HOLD"

    def test_reasoning_populated(self):
        bars = _make_bars(last_volume_mult=1.5, last_pct_change=1.0)
        news = {"news_score": 0.80, "headline_count": 3, "sentiment_direction": "BUY"}
        result = self.strategy.analyze("AAPL", bars, "BUY", news_data=news)
        assert len(result.reasoning) >= 3
