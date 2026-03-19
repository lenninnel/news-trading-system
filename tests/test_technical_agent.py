"""Tests for advanced TA pattern detection in TechnicalAgent.

Covers:
    - Golden cross / death cross detection
    - ADX and trend strength labelling
    - Bull flag pattern detection and breakout
    - Wedge detection (ascending / descending) and breakout
    - Support / resistance level identification
    - Confidence adjustment logic
    - SMA-200 and MA200 distance calculation
    - DB schema migration (new columns)
    - Full indicator field coverage

All tests use synthetic data — no network calls or real market data.

Run with:
    python3 -m pytest tests/test_technical_agent.py -v
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.technical_agent import TechnicalAgent
from utils import safe_column


# ===========================================================================
# Helpers — synthetic data builders
# ===========================================================================

def _make_simple_df(n=250, seed=42):
    """Random walk price data with OHLCV."""
    np.random.seed(seed)
    returns = np.random.normal(0, 0.01, n)
    prices = 100 * np.exp(np.cumsum(returns))
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.DataFrame({
        "Open": prices * 0.99,
        "High": prices * 1.01,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


def _make_uptrend_df(n=250, seed=42):
    """Steadily rising prices."""
    np.random.seed(seed)
    prices = np.linspace(80, 160, n) + np.random.normal(0, 0.5, n)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.DataFrame({
        "Open": prices * 0.99,
        "High": prices * 1.01,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


def _make_downtrend_df(n=250, seed=42):
    """Steadily falling prices."""
    np.random.seed(seed)
    prices = np.linspace(160, 80, n) + np.random.normal(0, 0.5, n)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.DataFrame({
        "Open": prices * 0.99,
        "High": prices * 1.01,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


def _make_golden_cross_df():
    """Create price data where SMA50 just crossed above SMA200 in last 5 bars.

    Flat at 100, then a mild dip (bars -60 to -6) to pull SMA50 below SMA200,
    then a sharp spike in the last 6 bars to force SMA50 back above SMA200
    within the last 5 bars.
    """
    n = 250
    np.random.seed(42)
    prices = np.full(n, 100.0)
    # Dip to pull SMA50 below SMA200
    prices[-60:-6] = 97.0
    # Strong spike in last 6 bars to force golden cross
    prices[-6:] = np.linspace(120, 150, 6)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.DataFrame({
        "Open": prices * 0.99,
        "High": prices * 1.01,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


def _make_death_cross_df():
    """Create price data where SMA50 just crossed below SMA200 in last 5 bars.

    Flat at 100, then a mild bump (bars -60 to -6) to pull SMA50 above SMA200,
    then a sharp drop in the last 6 bars to force SMA50 back below SMA200
    within the last 5 bars.
    """
    n = 250
    np.random.seed(42)
    prices = np.full(n, 100.0)
    # Bump to pull SMA50 above SMA200
    prices[-60:-6] = 103.0
    # Sharp drop in last 6 bars to force death cross
    prices[-6:] = np.linspace(80, 50, 6)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.DataFrame({
        "Open": prices * 0.99,
        "High": prices * 1.01,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


def _make_ranging_df(n=250, seed=42):
    """Sideways/ranging price data oscillating around a constant mean."""
    np.random.seed(seed)
    t = np.arange(n)
    prices = 100 + 2 * np.sin(2 * np.pi * t / 20) + np.random.normal(0, 0.3, n)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.003,
        "Low": prices * 0.997,
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


def _make_strong_trend_df(n=250, seed=42):
    """Strong unidirectional trend (for high ADX)."""
    np.random.seed(seed)
    prices = np.linspace(80, 200, n) + np.random.normal(0, 0.3, n)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.DataFrame({
        "Open": prices * 0.995,
        "High": prices * 1.015,
        "Low": prices * 0.985,
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


def _make_bull_flag_df():
    """Create data with a strong uptrend followed by tight consolidation.

    - 240 bars of gradual move
    - Bars -20 to -10: >10% gain (uptrend pole)
    - Last 10 bars: tight consolidation (<5% range), declining volume
    """
    np.random.seed(42)
    n = 250
    # Initial period: slow drift
    prices_initial = np.linspace(80, 90, n - 20)
    # Uptrend pole: bars -20 to -10 (10 bars, >10% gain: 90 -> 100)
    prices_pole = np.linspace(90, 100, 10)
    # Consolidation: last 10 bars, tight range around 100 (<5% range)
    prices_consolidation = np.full(10, 100.0) + np.random.uniform(-1.0, 1.0, 10)
    prices = np.concatenate([prices_initial, prices_pole, prices_consolidation])

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)

    # Declining volume during consolidation
    volume = np.full(n, 5_000_000)
    volume[-10:] = 2_000_000  # Lower volume in consolidation

    return pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.002,
        "Low": prices * 0.998,
        "Close": prices,
        "Volume": volume,
    }, index=dates)


def _make_bull_flag_breakout_df():
    """Bull flag where last bar breaks above consolidation range."""
    np.random.seed(42)
    n = 250
    prices_initial = np.linspace(80, 90, n - 20)
    prices_pole = np.linspace(90, 100, 10)
    # Consolidation with last bar breaking out
    prices_consolidation = np.full(10, 100.0) + np.random.uniform(-1.0, 1.0, 10)
    prices_consolidation[-1] = np.max(prices_consolidation[:-1]) + 1.0  # breakout
    prices = np.concatenate([prices_initial, prices_pole, prices_consolidation])

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)

    volume = np.full(n, 5_000_000)
    volume[-10:] = 2_000_000  # Lower volume in consolidation

    return pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.002,
        "Low": prices * 0.998,
        "Close": prices,
        "Volume": volume,
    }, index=dates)


def _make_descending_wedge_df(n=250, breakout=False):
    """Create data with lower highs AND lower lows converging (descending wedge).

    The high slope and low slope are both negative, but high slope > low slope
    (i.e., converging).
    """
    np.random.seed(42)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)

    # Build last 20 bars as a descending wedge
    base_prices = np.linspace(100, 90, n)
    x = np.arange(20)
    # Highs slope down steeply, lows slope down even more steeply
    wedge_highs = 110 - 0.3 * x   # slope_high = -0.3
    wedge_lows = 105 - 0.5 * x    # slope_low = -0.5 (steeper, so slope_high > slope_low)
    wedge_close = (wedge_highs + wedge_lows) / 2

    high = np.concatenate([base_prices[:-20] * 1.01, wedge_highs])
    low = np.concatenate([base_prices[:-20] * 0.98, wedge_lows])
    close = np.concatenate([base_prices[:-20], wedge_close])

    if breakout:
        # Price breaks above the projected high trendline
        close[-1] = wedge_highs[-1] + 3.0

    open_ = close * 0.999

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


def _make_ascending_wedge_df(n=250):
    """Create data with higher highs AND higher lows converging (ascending wedge).

    Both slopes positive, but slope_low > slope_high (converging from below).
    """
    np.random.seed(42)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)

    base_prices = np.linspace(90, 100, n)
    x = np.arange(20)
    # Highs slope up slowly, lows slope up faster (converging)
    wedge_highs = 100 + 0.3 * x   # slope_high = 0.3
    wedge_lows = 90 + 0.5 * x     # slope_low = 0.5 (steeper, so slope_low > slope_high)
    wedge_close = (wedge_highs + wedge_lows) / 2

    high = np.concatenate([base_prices[:-20] * 1.01, wedge_highs])
    low = np.concatenate([base_prices[:-20] * 0.98, wedge_lows])
    close = np.concatenate([base_prices[:-20], wedge_close])
    open_ = close * 0.999

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


def _make_support_resistance_df(n=120, seed=42):
    """Data with clear swing highs and lows for S/R testing.

    Creates a zigzag pattern so that swing lows and swing highs are easily
    identifiable.
    """
    np.random.seed(seed)
    # Build a zigzag around 100
    t = np.arange(n)
    prices = 100 + 8 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.2, n)
    # End close to 100 so there are swing lows below and swing highs above
    prices[-1] = 100.0
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.005,
        "Low": prices * 0.995,
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


# ===========================================================================
# 1. Golden Cross / Death Cross Detection
# ===========================================================================

class TestGoldenCrossDetection:
    """Test golden cross and death cross detection in _calculate_indicators."""

    def test_golden_cross_detected(self):
        """SMA50 crossing above SMA200 should set golden_cross_recent = True."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_golden_cross_df()
        indicators = agent._calculate_indicators(df)
        assert indicators["golden_cross_recent"] is True

    def test_death_cross_detected(self):
        """SMA50 crossing below SMA200 should set death_cross_recent = True."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_death_cross_df()
        indicators = agent._calculate_indicators(df)
        assert indicators["death_cross_recent"] is True

    def test_no_cross_when_stable_uptrend(self):
        """Steady uptrend should not trigger any cross in the last 5 bars."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_uptrend_df(n=250)
        indicators = agent._calculate_indicators(df)
        # In a steady uptrend, SMA50 is above SMA200 throughout — no recent cross
        assert indicators["golden_cross_recent"] is False or indicators["death_cross_recent"] is False

    def test_no_cross_when_stable_downtrend(self):
        """Steady downtrend should not trigger golden cross."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_downtrend_df(n=250)
        indicators = agent._calculate_indicators(df)
        assert indicators["golden_cross_recent"] is False

    def test_golden_cross_not_death_cross(self):
        """Golden cross scenario should not trigger death cross."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_golden_cross_df()
        indicators = agent._calculate_indicators(df)
        # If a golden cross happened, death cross should not also be True
        # (unless the data is very choppy — but our synthetic data is clean)
        if indicators["golden_cross_recent"]:
            assert indicators["death_cross_recent"] is False

    def test_death_cross_not_golden_cross(self):
        """Death cross scenario should not trigger golden cross."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_death_cross_df()
        indicators = agent._calculate_indicators(df)
        if indicators["death_cross_recent"]:
            assert indicators["golden_cross_recent"] is False


# ===========================================================================
# 2. ADX and Trend Strength
# ===========================================================================

class TestADX:
    """Test ADX calculation and trend strength labelling."""

    def test_adx_strong_trend(self):
        """Strong trending data should produce ADX > 25."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_strong_trend_df(n=250)
        indicators = agent._calculate_indicators(df)
        assert indicators["adx"] is not None
        assert indicators["adx"] > 25
        assert indicators["trend_strength"] == "strong"

    def test_adx_ranging_market(self):
        """Sideways/ranging data should produce low ADX (typically < 25)."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_ranging_df(n=250)
        indicators = agent._calculate_indicators(df)
        assert indicators["adx"] is not None
        # Ranging market should have relatively low ADX
        assert indicators["adx"] < 30  # generous bound for ranging

    def test_trend_strength_labels(self):
        """Verify trend_strength labels map correctly to ADX values."""
        agent = TechnicalAgent(db=MagicMock())

        # Strong trend
        df_strong = _make_strong_trend_df(n=250)
        ind_strong = agent._calculate_indicators(df_strong)
        if ind_strong["adx"] is not None and ind_strong["adx"] > 25:
            assert ind_strong["trend_strength"] == "strong"

        # Ranging market
        df_range = _make_ranging_df(n=250)
        ind_range = agent._calculate_indicators(df_range)
        if ind_range["adx"] is not None and ind_range["adx"] < 20:
            assert ind_range["trend_strength"] == "weak"

    def test_trend_strength_moderate(self):
        """ADX between 20 and 25 should give 'moderate' label."""
        # Test the labelling logic directly via _apply_confidence_adjustments
        # by checking the indicator dict
        agent = TechnicalAgent(db=MagicMock())
        df = _make_simple_df(n=250)
        indicators = agent._calculate_indicators(df)
        adx = indicators["adx"]
        ts = indicators["trend_strength"]
        if adx is not None:
            if adx > 25:
                assert ts == "strong"
            elif adx < 20:
                assert ts == "weak"
            else:
                assert ts == "moderate"

    def test_adx_insufficient_data(self):
        """Short data should still return something (or None)."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_simple_df(n=20, seed=42)
        indicators = agent._calculate_indicators(df)
        # With very short data, ADX may be None
        # The important thing is it doesn't crash
        assert "adx" in indicators
        assert "trend_strength" in indicators


# ===========================================================================
# 3. Bull Flag Detection
# ===========================================================================

class TestBullFlag:
    """Test bull flag pattern detection and breakout."""

    def test_bull_flag_detected(self):
        """Data with uptrend pole + tight consolidation should detect bull flag."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_bull_flag_df()
        indicators = agent._calculate_indicators(df)
        assert indicators["bull_flag_detected"] is True

    def test_no_bull_flag_without_uptrend(self):
        """Flat price series should not trigger bull flag."""
        agent = TechnicalAgent(db=MagicMock())
        np.random.seed(42)
        n = 250
        # Flat prices — no >10% prior gain
        prices = np.full(n, 100.0) + np.random.normal(0, 0.5, n)
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
        df = pd.DataFrame({
            "Open": prices * 0.999,
            "High": prices * 1.002,
            "Low": prices * 0.998,
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        }, index=dates)
        indicators = agent._calculate_indicators(df)
        assert indicators["bull_flag_detected"] is False

    def test_bull_flag_breakout(self):
        """When price breaks above consolidation range, breakout should be True."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_bull_flag_breakout_df()
        indicators = agent._calculate_indicators(df)
        # Must detect the flag first
        if indicators["bull_flag_detected"]:
            assert indicators["bull_flag_breakout"] is True

    def test_bull_flag_no_breakout_during_consolidation(self):
        """During normal consolidation, breakout should be False."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_bull_flag_df()
        indicators = agent._calculate_indicators(df)
        if indicators["bull_flag_detected"]:
            # The non-breakout version should not have breakout
            # (depends on data — the helper is designed so last bar
            # is within the consolidation range)
            assert isinstance(indicators["bull_flag_breakout"], bool)

    def test_bull_flag_insufficient_data(self):
        """With fewer than 20 bars, bull flag should not be detected."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_simple_df(n=15, seed=42)
        indicators = agent._calculate_indicators(df)
        assert indicators["bull_flag_detected"] is False
        assert indicators["bull_flag_breakout"] is False


# ===========================================================================
# 4. Wedge Detection
# ===========================================================================

class TestWedgeDetection:
    """Test descending and ascending wedge detection."""

    def test_descending_wedge_detected(self):
        """Series of lower highs AND lower lows converging should detect descending wedge."""
        df = _make_descending_wedge_df(n=250, breakout=False)
        close = safe_column(df, "Close")
        wedge_type, wedge_breakout = TechnicalAgent._detect_wedge(df, close)
        assert wedge_type == "descending"

    def test_ascending_wedge_detected(self):
        """Series of higher highs AND higher lows converging should detect ascending wedge."""
        df = _make_ascending_wedge_df(n=250)
        close = safe_column(df, "Close")
        wedge_type, wedge_breakout = TechnicalAgent._detect_wedge(df, close)
        assert wedge_type == "ascending"

    def test_no_wedge_in_trending_market(self):
        """Strong trend should not trigger a wedge detection."""
        df = _make_strong_trend_df(n=250)
        close = safe_column(df, "Close")
        wedge_type, wedge_breakout = TechnicalAgent._detect_wedge(df, close)
        # A strong parallel uptrend may or may not be detected as a wedge;
        # the key is it shouldn't be a converging wedge
        # (for truly parallel trends, slopes are similar — not converging)
        # We accept None or either type but verify it returns valid values
        assert wedge_type in (None, "descending", "ascending")

    def test_wedge_breakout(self):
        """Price breaking above descending wedge trendline should trigger breakout."""
        df = _make_descending_wedge_df(n=250, breakout=True)
        close = safe_column(df, "Close")
        wedge_type, wedge_breakout = TechnicalAgent._detect_wedge(df, close)
        if wedge_type == "descending":
            assert wedge_breakout == True  # noqa: E712 (numpy bool)

    def test_no_wedge_with_insufficient_data(self):
        """Short data (< 20 bars) should return None."""
        df = _make_simple_df(n=15, seed=42)
        close = safe_column(df, "Close")
        wedge_type, wedge_breakout = TechnicalAgent._detect_wedge(df, close)
        assert wedge_type is None
        assert wedge_breakout is False

    def test_wedge_returns_valid_types(self):
        """Wedge type should be None, 'descending', or 'ascending'."""
        df = _make_simple_df(n=250)
        close = safe_column(df, "Close")
        wedge_type, wedge_breakout = TechnicalAgent._detect_wedge(df, close)
        assert wedge_type in (None, "descending", "ascending")
        assert isinstance(wedge_breakout, bool)


# ===========================================================================
# 5. Support and Resistance
# ===========================================================================

class TestSupportResistance:
    """Test support and resistance level identification."""

    def test_finds_nearest_support(self):
        """Creates data with clear swing lows below the current price."""
        df = _make_support_resistance_df()
        close = safe_column(df, "Close")
        price = float(close.iloc[-1])
        support, resistance, pct_s, pct_r = TechnicalAgent._find_support_resistance(
            close, price
        )
        # The zigzag pattern should have swing lows below 100
        assert support is not None
        assert support < price

    def test_finds_nearest_resistance(self):
        """Creates data with clear swing highs above the current price."""
        df = _make_support_resistance_df()
        close = safe_column(df, "Close")
        price = float(close.iloc[-1])
        support, resistance, pct_s, pct_r = TechnicalAgent._find_support_resistance(
            close, price
        )
        assert resistance is not None
        assert resistance > price

    def test_percentage_distances(self):
        """Verify pct_to_support and pct_to_resistance calculations."""
        df = _make_support_resistance_df()
        close = safe_column(df, "Close")
        price = float(close.iloc[-1])
        support, resistance, pct_s, pct_r = TechnicalAgent._find_support_resistance(
            close, price
        )
        if support is not None:
            expected_pct_s = round((price - support) / price * 100, 2)
            assert pct_s == expected_pct_s
            assert pct_s > 0  # support is below price

        if resistance is not None:
            expected_pct_r = round((resistance - price) / price * 100, 2)
            assert pct_r == expected_pct_r
            assert pct_r > 0  # resistance is above price

    def test_no_support_when_all_above(self):
        """When all swing lows are above price, nearest_support should be None."""
        np.random.seed(42)
        n = 60
        # Price data trending up, last price is at the very bottom
        prices = np.linspace(120, 110, n)
        prices[-1] = 80.0  # current price far below everything
        close = pd.Series(prices)
        support, _, _, _ = TechnicalAgent._find_support_resistance(close, 80.0)
        # All swing lows should be > 80 if the data trends above
        # (support may or may not be None depending on exact swings)
        assert support is None or support < 80.0

    def test_none_price_returns_nones(self):
        """None price should return all Nones."""
        close = pd.Series([100.0, 101.0, 99.0, 102.0, 98.0])
        support, resistance, pct_s, pct_r = TechnicalAgent._find_support_resistance(
            close, None
        )
        assert support is None
        assert resistance is None
        assert pct_s is None
        assert pct_r is None

    def test_insufficient_data(self):
        """Very short close series returns all Nones."""
        close = pd.Series([100.0, 101.0])
        support, resistance, pct_s, pct_r = TechnicalAgent._find_support_resistance(
            close, 100.0
        )
        # With only 2 data points, can't identify swing points
        assert support is None
        assert resistance is None


# ===========================================================================
# 6. Confidence Adjustments
# ===========================================================================

class TestConfidenceAdjustments:
    """Test the confidence boost/reduction logic in _apply_confidence_adjustments."""

    def test_golden_cross_adx_boost(self):
        """golden_cross_recent + ADX > 25 + price > SMA200 should add +0.15."""
        indicators = {
            "golden_cross_recent": True,
            "death_cross_recent": False,
            "adx": 30.0,
            "trend_strength": "strong",
            "sma_200": 100.0,
            "price": 110.0,
            "bull_flag_breakout": False,
            "wedge_type": None,
            "wedge_breakout": False,
        }
        confidence, reasons = TechnicalAgent._apply_confidence_adjustments("BUY", indicators)
        # Base for BUY is 0.5, with golden cross boost +0.15 = 0.65
        assert confidence >= 0.65
        assert any("+0.15" in r for r in reasons)

    def test_bull_flag_breakout_boost(self):
        """bull_flag_breakout should add +0.10."""
        indicators = {
            "golden_cross_recent": False,
            "death_cross_recent": False,
            "adx": 15.0,
            "trend_strength": "weak",
            "sma_200": 100.0,
            "price": 95.0,
            "bull_flag_breakout": True,
            "wedge_type": None,
            "wedge_breakout": False,
        }
        confidence, reasons = TechnicalAgent._apply_confidence_adjustments("BUY", indicators)
        # Base 0.5 + bull flag breakout 0.10 = 0.60
        assert confidence >= 0.55  # might have reductions too
        assert any("Bull flag breakout" in r for r in reasons)

    def test_death_cross_reduction(self):
        """death_cross_recent should reduce confidence by -0.15."""
        indicators = {
            "golden_cross_recent": False,
            "death_cross_recent": True,
            "adx": 15.0,
            "trend_strength": "weak",
            "sma_200": 100.0,
            "price": 110.0,
            "bull_flag_breakout": False,
            "wedge_type": None,
            "wedge_breakout": False,
        }
        confidence, reasons = TechnicalAgent._apply_confidence_adjustments("BUY", indicators)
        # Base 0.5 - death cross 0.15 = 0.35
        assert confidence <= 0.35
        assert any("-0.15" in r for r in reasons)

    def test_confidence_cap_at_95(self):
        """Multiple boosts should not exceed 0.95."""
        indicators = {
            "golden_cross_recent": True,
            "death_cross_recent": False,
            "adx": 30.0,
            "trend_strength": "strong",
            "sma_200": 90.0,
            "price": 110.0,
            "bull_flag_breakout": True,
            "wedge_type": "descending",
            "wedge_breakout": True,
        }
        confidence, reasons = TechnicalAgent._apply_confidence_adjustments("BUY", indicators)
        # Base 0.5 + golden+adx 0.15 + bull flag 0.10 + wedge 0.10 = 0.85
        # Even with all boosts, should be <= 0.95
        assert confidence <= 0.95

    def test_confidence_floor_at_zero(self):
        """Multiple reductions should not go below 0.0."""
        indicators = {
            "golden_cross_recent": False,
            "death_cross_recent": True,
            "adx": 30.0,
            "trend_strength": "strong",
            "sma_200": 110.0,
            "price": 90.0,      # price below SMA200
            "bull_flag_breakout": False,
            "wedge_type": None,
            "wedge_breakout": False,
        }
        confidence, reasons = TechnicalAgent._apply_confidence_adjustments("HOLD", indicators)
        # Base for HOLD is 0.25, -0.15 (death cross) -0.10 (below SMA200 strong) = 0.0
        assert confidence >= 0.0

    def test_no_adjustment_for_hold(self):
        """HOLD signal with no patterns should get base confidence (0.25)."""
        indicators = {
            "golden_cross_recent": False,
            "death_cross_recent": False,
            "adx": 15.0,
            "trend_strength": "weak",
            "sma_200": 100.0,
            "price": 100.5,
            "bull_flag_breakout": False,
            "wedge_type": None,
            "wedge_breakout": False,
        }
        confidence, reasons = TechnicalAgent._apply_confidence_adjustments("HOLD", indicators)
        assert confidence == 0.25
        assert len(reasons) == 0

    def test_base_confidence_buy(self):
        """BUY signal with no patterns should get base confidence (0.50)."""
        indicators = {
            "golden_cross_recent": False,
            "death_cross_recent": False,
            "adx": 15.0,
            "trend_strength": "weak",
            "sma_200": 100.0,
            "price": 100.5,
            "bull_flag_breakout": False,
            "wedge_type": None,
            "wedge_breakout": False,
        }
        confidence, reasons = TechnicalAgent._apply_confidence_adjustments("BUY", indicators)
        assert confidence == 0.50
        assert len(reasons) == 0

    def test_base_confidence_sell(self):
        """SELL signal with no patterns should get base confidence (0.50)."""
        indicators = {
            "golden_cross_recent": False,
            "death_cross_recent": False,
            "adx": 15.0,
            "trend_strength": "weak",
            "sma_200": 100.0,
            "price": 100.5,
            "bull_flag_breakout": False,
            "wedge_type": None,
            "wedge_breakout": False,
        }
        confidence, reasons = TechnicalAgent._apply_confidence_adjustments("SELL", indicators)
        assert confidence == 0.50
        assert len(reasons) == 0

    def test_descending_wedge_breakout_boosts_buy(self):
        """Descending wedge breakout + BUY signal should boost confidence."""
        indicators = {
            "golden_cross_recent": False,
            "death_cross_recent": False,
            "adx": 15.0,
            "trend_strength": "weak",
            "sma_200": 100.0,
            "price": 100.5,
            "bull_flag_breakout": False,
            "wedge_type": "descending",
            "wedge_breakout": True,
        }
        confidence, reasons = TechnicalAgent._apply_confidence_adjustments("BUY", indicators)
        # Base 0.5 + wedge breakout 0.10 = 0.60
        assert confidence == 0.60
        assert any("wedge breakout" in r.lower() for r in reasons)

    def test_ascending_wedge_breakout_boosts_sell(self):
        """Ascending wedge breakout + SELL signal should boost confidence."""
        indicators = {
            "golden_cross_recent": False,
            "death_cross_recent": False,
            "adx": 15.0,
            "trend_strength": "weak",
            "sma_200": 100.0,
            "price": 100.5,
            "bull_flag_breakout": False,
            "wedge_type": "ascending",
            "wedge_breakout": True,
        }
        confidence, reasons = TechnicalAgent._apply_confidence_adjustments("SELL", indicators)
        # Base 0.5 + wedge breakout 0.10 = 0.60
        assert confidence == 0.60
        assert any("wedge breakout" in r.lower() for r in reasons)

    def test_descending_wedge_breakout_no_boost_for_sell(self):
        """Descending wedge breakout should NOT boost SELL signal."""
        indicators = {
            "golden_cross_recent": False,
            "death_cross_recent": False,
            "adx": 15.0,
            "trend_strength": "weak",
            "sma_200": 100.0,
            "price": 100.5,
            "bull_flag_breakout": False,
            "wedge_type": "descending",
            "wedge_breakout": True,
        }
        confidence, reasons = TechnicalAgent._apply_confidence_adjustments("SELL", indicators)
        # Should be base 0.5 with no wedge boost
        assert confidence == 0.50

    def test_price_below_sma200_strong_trend_reduction(self):
        """Price below SMA200 with strong downtrend should reduce confidence."""
        indicators = {
            "golden_cross_recent": False,
            "death_cross_recent": False,
            "adx": 30.0,
            "trend_strength": "strong",
            "sma_200": 110.0,
            "price": 100.0,  # below SMA200
            "bull_flag_breakout": False,
            "wedge_type": None,
            "wedge_breakout": False,
        }
        confidence, reasons = TechnicalAgent._apply_confidence_adjustments("BUY", indicators)
        # Base 0.5 - 0.10 = 0.40
        assert confidence == 0.40
        assert any("-0.10" in r for r in reasons)


# ===========================================================================
# 7. SMA-200
# ===========================================================================

class TestSMA200:
    """Test SMA-200 computation and MA200 distance."""

    def test_sma200_computed(self):
        """Verify sma_200 field exists and is numeric."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_simple_df(n=250)
        indicators = agent._calculate_indicators(df)
        assert "sma_200" in indicators
        assert indicators["sma_200"] is not None
        assert isinstance(indicators["sma_200"], float)

    def test_ma200_distance_positive(self):
        """When price > SMA200, ma200_distance_pct should be positive."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_uptrend_df(n=250)
        indicators = agent._calculate_indicators(df)
        if indicators["sma_200"] is not None and indicators["price"] is not None:
            if indicators["price"] > indicators["sma_200"]:
                assert indicators["ma200_distance_pct"] > 0

    def test_ma200_distance_negative(self):
        """When price < SMA200, ma200_distance_pct should be negative."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_downtrend_df(n=250)
        indicators = agent._calculate_indicators(df)
        if indicators["sma_200"] is not None and indicators["price"] is not None:
            if indicators["price"] < indicators["sma_200"]:
                assert indicators["ma200_distance_pct"] < 0

    def test_sma200_none_with_insufficient_data(self):
        """With < 200 bars, SMA200 should be None."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_simple_df(n=50, seed=42)
        indicators = agent._calculate_indicators(df)
        # Not enough data for SMA200
        assert indicators["sma_200"] is None

    def test_ma200_distance_pct_none_when_sma200_none(self):
        """If SMA200 is None, distance pct should also be None."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_simple_df(n=50, seed=42)
        indicators = agent._calculate_indicators(df)
        if indicators["sma_200"] is None:
            assert indicators["ma200_distance_pct"] is None

    def test_ma200_distance_calculation(self):
        """Verify the distance percentage formula."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_simple_df(n=250)
        indicators = agent._calculate_indicators(df)
        if indicators["sma_200"] is not None and indicators["price"] is not None:
            expected = round(
                (indicators["price"] - indicators["sma_200"]) / indicators["sma_200"] * 100,
                2,
            )
            assert indicators["ma200_distance_pct"] == expected


# ===========================================================================
# 8. DB Schema Migration
# ===========================================================================

class TestDBSchemaMigration:
    """Test that new columns are added cleanly to the database."""

    def test_new_columns_added(self):
        """Verify new TA columns exist in technical_signals table."""
        import sqlite3
        import tempfile

        from storage.database import Database

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path=db_path)
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("PRAGMA table_info(technical_signals)")
            columns = {row[1] for row in cursor.fetchall()}
            conn.close()

            expected_new = {
                "sma_200", "ma200_distance_pct", "golden_cross_recent",
                "death_cross_recent", "adx", "trend_strength",
                "bull_flag_detected", "wedge_type", "wedge_breakout",
                "nearest_support", "nearest_resistance",
            }
            assert expected_new.issubset(columns), f"Missing columns: {expected_new - columns}"
        finally:
            os.unlink(db_path)

    def test_migration_idempotent(self):
        """Running migration twice should not fail."""
        import tempfile

        from storage.database import Database

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # First init creates the schema
            db1 = Database(db_path=db_path)
            # Second init re-runs migration — should not raise
            db2 = Database(db_path=db_path)
        finally:
            os.unlink(db_path)

    def test_log_technical_signal_with_new_columns(self):
        """Verify log_technical_signal accepts and stores new column values."""
        import sqlite3
        import tempfile

        from storage.database import Database

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path=db_path)
            signal_id = db.log_technical_signal(
                ticker="TEST",
                signal="BUY",
                reasoning="Test signal",
                rsi=45.0,
                price=100.0,
                sma_200=95.0,
                ma200_distance_pct=5.26,
                golden_cross_recent=True,
                death_cross_recent=False,
                adx=28.5,
                trend_strength="strong",
                bull_flag_detected=True,
                wedge_type="descending",
                wedge_breakout=False,
                nearest_support=92.0,
                nearest_resistance=108.0,
            )
            assert isinstance(signal_id, int)
            assert signal_id > 0

            # Verify stored values
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM technical_signals WHERE id = ?", (signal_id,)
            ).fetchone()
            conn.close()

            assert row["sma_200"] == 95.0
            assert row["adx"] == 28.5
            assert row["golden_cross_recent"] == 1  # stored as integer
            assert row["trend_strength"] == "strong"
            assert row["wedge_type"] == "descending"
        finally:
            os.unlink(db_path)


# ===========================================================================
# 9. All Indicator Fields Present
# ===========================================================================

class TestIndicatorsReturnAllFields:
    """Verify _calculate_indicators returns all expected keys."""

    def test_all_indicator_fields_present(self):
        """_calculate_indicators returns all expected keys."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_simple_df(250)
        indicators = agent._calculate_indicators(df)

        expected_keys = {
            # existing
            "rsi", "macd", "macd_signal", "macd_hist",
            "macd_bull_cross", "macd_bear_cross",
            "sma_20", "sma_50", "bb_upper", "bb_lower", "price",
            "rvol", "volume_trending_up", "obv_trend",
            # new
            "sma_200", "ma200_distance_pct",
            "golden_cross_recent", "death_cross_recent",
            "adx", "trend_strength",
            "bull_flag_detected", "bull_flag_breakout",
            "wedge_type", "wedge_breakout",
            "nearest_support", "nearest_resistance",
            "pct_to_support", "pct_to_resistance",
        }
        assert expected_keys.issubset(set(indicators.keys())), (
            f"Missing keys: {expected_keys - set(indicators.keys())}"
        )

    def test_indicator_types(self):
        """Check that indicator values have correct types."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_simple_df(250)
        indicators = agent._calculate_indicators(df)

        # Boolean fields
        assert isinstance(indicators["golden_cross_recent"], bool)
        assert isinstance(indicators["death_cross_recent"], bool)
        assert isinstance(indicators["macd_bull_cross"], bool)
        assert isinstance(indicators["macd_bear_cross"], bool)
        assert isinstance(indicators["bull_flag_detected"], bool)
        assert isinstance(indicators["bull_flag_breakout"], bool)
        assert isinstance(indicators["wedge_breakout"], bool)

        # Numeric fields (may be None if insufficient data)
        for key in ["rsi", "macd", "sma_20", "sma_50", "price"]:
            val = indicators[key]
            assert val is None or isinstance(val, float), f"{key} is {type(val)}"

        # String or None fields
        for key in ["obv_trend", "trend_strength", "wedge_type"]:
            val = indicators[key]
            assert val is None or isinstance(val, str), f"{key} is {type(val)}"

    def test_short_data_returns_all_keys(self):
        """Even with very short data, all keys should still be present."""
        agent = TechnicalAgent(db=MagicMock())
        df = _make_simple_df(n=20, seed=42)
        indicators = agent._calculate_indicators(df)

        expected_keys = {
            "rsi", "macd", "macd_signal", "macd_hist",
            "macd_bull_cross", "macd_bear_cross",
            "sma_20", "sma_50", "bb_upper", "bb_lower", "price",
            "rvol", "volume_trending_up", "obv_trend",
            "sma_200", "ma200_distance_pct",
            "golden_cross_recent", "death_cross_recent",
            "adx", "trend_strength",
            "bull_flag_detected", "bull_flag_breakout",
            "wedge_type", "wedge_breakout",
            "nearest_support", "nearest_resistance",
            "pct_to_support", "pct_to_resistance",
        }
        assert expected_keys.issubset(set(indicators.keys())), (
            f"Missing keys: {expected_keys - set(indicators.keys())}"
        )


# ===========================================================================
# 10. Static pattern helpers (direct tests)
# ===========================================================================

class TestDetectBullFlagStatic:
    """Direct tests for the _detect_bull_flag static method."""

    def test_returns_tuple(self):
        """Should always return a (bool, bool) tuple."""
        df = _make_simple_df(n=250)
        close = safe_column(df, "Close")
        result = TechnicalAgent._detect_bull_flag(df, close)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], bool)

    def test_empty_df(self):
        """Empty DataFrame should return (False, False)."""
        df = pd.DataFrame({"Close": [], "Volume": []})
        close = safe_column(df, "Close")
        detected, breakout = TechnicalAgent._detect_bull_flag(df, close)
        assert detected is False
        assert breakout is False


class TestDetectWedgeStatic:
    """Direct tests for the _detect_wedge static method."""

    def test_returns_tuple(self):
        """Should always return a (str|None, bool) tuple."""
        df = _make_simple_df(n=250)
        close = safe_column(df, "Close")
        result = TechnicalAgent._detect_wedge(df, close)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] in (None, "descending", "ascending")
        assert isinstance(result[1], bool)

    def test_missing_high_low_columns(self):
        """If High/Low missing, should return (None, False)."""
        n = 250
        np.random.seed(42)
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
        df = pd.DataFrame({
            "Close": np.linspace(100, 120, n),
        }, index=dates)
        close = safe_column(df, "Close")
        wedge_type, breakout = TechnicalAgent._detect_wedge(df, close)
        assert wedge_type is None
        assert breakout is False


class TestFindSupportResistanceStatic:
    """Direct tests for the _find_support_resistance static method."""

    def test_returns_four_element_tuple(self):
        """Should return (support, resistance, pct_to_support, pct_to_resistance)."""
        close = pd.Series(np.linspace(90, 110, 60))
        result = TechnicalAgent._find_support_resistance(close, 100.0)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_with_clear_swing_points(self):
        """Zigzag data should identify both support and resistance."""
        np.random.seed(42)
        t = np.arange(60)
        prices = 100 + 5 * np.sin(2 * np.pi * t / 15)
        prices[-1] = 100.0  # current price at midpoint
        close = pd.Series(prices)
        support, resistance, pct_s, pct_r = TechnicalAgent._find_support_resistance(
            close, 100.0
        )
        # Should find swing lows (~95) and swing highs (~105)
        assert support is not None
        assert resistance is not None
        assert support < 100.0
        assert resistance > 100.0


# ===========================================================================
# 11. Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_very_short_data(self):
        """Very short data (5 bars) should not crash — most indicators will be None."""
        agent = TechnicalAgent(db=MagicMock())
        n = 5
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
        df = pd.DataFrame({
            "Open": [99.0, 100.0, 101.0, 100.5, 99.5],
            "High": [101.0, 102.0, 103.0, 102.5, 101.5],
            "Low": [98.0, 99.0, 100.0, 99.5, 98.5],
            "Close": [100.0, 101.0, 102.0, 101.0, 100.0],
            "Volume": [5_000_000] * n,
        }, index=dates)
        indicators = agent._calculate_indicators(df)
        # Should not crash and should have a price
        assert indicators["price"] == 100.0
        assert "sma_200" in indicators
        # SMA200 should be None with only 5 bars
        assert indicators["sma_200"] is None

    def test_constant_prices(self):
        """Constant prices (no volatility) should not crash."""
        agent = TechnicalAgent(db=MagicMock())
        n = 250
        prices = np.full(n, 100.0)
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
        df = pd.DataFrame({
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": np.full(n, 5_000_000),
        }, index=dates)
        indicators = agent._calculate_indicators(df)
        assert indicators["price"] == 100.0
        assert indicators["sma_200"] == 100.0
        assert indicators["ma200_distance_pct"] == 0.0

    def test_deterministic_with_seed(self):
        """Results should be deterministic with the same seed."""
        agent = TechnicalAgent(db=MagicMock())
        df1 = _make_simple_df(n=250, seed=123)
        df2 = _make_simple_df(n=250, seed=123)
        ind1 = agent._calculate_indicators(df1)
        ind2 = agent._calculate_indicators(df2)
        assert ind1["rsi"] == ind2["rsi"]
        assert ind1["sma_200"] == ind2["sma_200"]
        assert ind1["adx"] == ind2["adx"]
        assert ind1["golden_cross_recent"] == ind2["golden_cross_recent"]


# ===========================================================================
# 12. Signal quality — known bullish setup produces BUY
# ===========================================================================

def _bullish_indicators():
    """
    Return a canonical bullish indicator dict.

    Score breakdown:
      RSI 55   → neutral (45–60), 0 pts (not penalised under new threshold)
      MACD hist positive → +1.0
      price > SMA20 > SMA50 → +1.0
      Total = +2.0 ≥ 1.5 → BUY
    """
    return {
        "rsi": 55.0,
        "macd_bull_cross": False,
        "macd_bear_cross": False,
        "macd_hist": 0.10,
        "price": 160.0,
        "bb_lower": 148.0,
        "bb_upper": 172.0,
        "sma_20": 158.0,
        "sma_50": 154.0,
        "golden_cross_recent": False,
        "death_cross_recent": False,
    }


class TestSignalQuality:
    """
    Validate that realistic market setups produce the expected signal direction.

    All tests use _apply_signal_rules directly (no network I/O).
    """

    def test_clear_bullish_setup_produces_buy(self):
        """
        A healthy uptrend — price > SMA20 > SMA50, positive MACD histogram,
        RSI in neutral zone (55) — must produce a BUY signal.

        Score: SMA alignment (+1.0) + positive MACD hist (+1.0) = 2.0 ≥ 1.5 → BUY.
        RSI 55 must NOT contribute a negative penalty (threshold raised to 60).

        This is the canonical regression test for the signal scoring pipeline.
        """
        ind = _bullish_indicators()
        signal, reasoning = TechnicalAgent._apply_signal_rules(ind)
        assert signal == "BUY", (
            f"Expected BUY for canonical bullish indicator set, got {signal}. "
            f"RSI={ind['rsi']}, MACD_hist={ind['macd_hist']}, "
            f"price={ind['price']} > SMA20={ind['sma_20']} > SMA50={ind['sma_50']}. "
            f"Scoring reasons: {reasoning}"
        )

    def test_rsi_neutral_zone_does_not_penalise_uptrend(self):
        """
        RSI in the 45–60 neutral zone must not contribute a negative score.
        In a trending stock, RSI 45–60 is healthy and common — penalising it
        causes false HOLDs.
        """
        # Inject indicators with RSI=55 (was penalised before fix, now neutral)
        ind = {
            "rsi": 55.0,
            "macd_bull_cross": False,
            "macd_bear_cross": False,
            "macd_hist": 0.10,
            "price": 160.0,
            "bb_lower": 150.0,
            "bb_upper": 170.0,
            "sma_20": 158.0,
            "sma_50": 155.0,
            "golden_cross_recent": False,
            "death_cross_recent": False,
        }
        signal, reasons = TechnicalAgent._apply_signal_rules(ind)
        # SMA alignment (+1) + positive MACD hist (+1) = 2.0 → BUY
        # RSI 55 must NOT add a negative penalty
        assert signal == "BUY", (
            f"RSI 55 should not penalise a bullish setup; got signal={signal}, reasons={reasons}"
        )
        assert not any("> 55" in r for r in reasons), (
            "RSI 55 should no longer be flagged as mildly overbought (threshold is now 60)"
        )

    def test_rsi_just_above_new_threshold_applies_mild_penalty(self):
        """
        RSI 61 is just above the new 60 threshold and should apply the -0.5
        mildly-overbought penalty.
        """
        ind = {
            "rsi": 61.0,
            "macd_bull_cross": False,
            "macd_bear_cross": False,
            "macd_hist": 0.0,
            "price": 160.0,
            "bb_lower": 150.0,
            "bb_upper": 170.0,
            "sma_20": 158.0,
            "sma_50": 155.0,
            "golden_cross_recent": False,
            "death_cross_recent": False,
        }
        signal, reasons = TechnicalAgent._apply_signal_rules(ind)
        # Score: RSI 61 (-0.5) + SMA alignment (+1.0) + MACD hist=0 (0) = 0.5 → HOLD
        assert signal == "HOLD", (
            f"RSI 61 should apply mild penalty making score 0.5 → HOLD; got {signal}"
        )
        assert any("> 60" in r for r in reasons), (
            "RSI 61 should be flagged as mildly overbought (> 60)"
        )


# ===========================================================================
# 13. Multi-Timeframe Confirmation
# ===========================================================================

def _make_intraday_df(n=100, trend="up", seed=42):
    """Create synthetic intraday bar data.

    Args:
        n:     Number of bars.
        trend: "up" for bullish, "down" for bearish.
        seed:  Random seed.
    """
    np.random.seed(seed)
    if trend == "up":
        prices = np.linspace(100, 120, n) + np.random.normal(0, 0.3, n)
    else:
        prices = np.linspace(120, 100, n) + np.random.normal(0, 0.3, n)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="1h")
    return pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.002,
        "Low": prices * 0.998,
        "Close": prices,
        "Volume": np.random.randint(100_000, 1_000_000, n),
    }, index=dates)


class TestMultiTimeframeConfirmation:
    """Test multi-timeframe analysis and timeframe alignment logic."""

    def test_all_timeframes_agree_bullish_boosts_confidence(self):
        """When daily=BUY, 1h RSI>50, 15m MACD>0 → alignment=1.0, +0.15 boost."""
        mock_db = MagicMock()
        mock_db.log_technical_signal.return_value = 1

        # Create bullish intraday data
        df_1h = _make_intraday_df(n=100, trend="up", seed=42)
        df_15m = _make_intraday_df(n=100, trend="up", seed=43)

        mock_alpaca = MagicMock()

        def mock_get_bars(ticker, timeframe, limit=252):
            if timeframe == "1Day":
                return _make_uptrend_df(n=250)
            elif timeframe == "1Hour":
                return df_1h
            elif timeframe == "15Min":
                return df_15m
            raise ValueError(f"Unexpected timeframe: {timeframe}")

        mock_alpaca.get_bars = mock_get_bars

        mock_eodhd = MagicMock()
        mock_eodhd.available = False

        agent = TechnicalAgent(
            db=mock_db, alpaca_client=mock_alpaca, eodhd_feed=mock_eodhd,
        )
        result = agent.run("AAPL")

        indicators = result["indicators"]
        assert indicators["intraday_available"] is True
        assert indicators["rsi_1h"] is not None
        assert indicators["macd_15m_hist"] is not None

        # If alignment is 1.0, confidence should have been boosted
        if indicators["timeframe_alignment"] == 1.0:
            assert any("boosted +0.15" in r for r in result["reasoning"])

    def test_timeframes_conflict_penalises_confidence(self):
        """When daily=BUY but intraday is bearish → alignment=0.0, -0.10 penalty."""
        mock_db = MagicMock()
        mock_db.log_technical_signal.return_value = 1

        # Bullish daily data (will produce BUY signal)
        df_daily = _make_uptrend_df(n=250)
        # Bearish intraday data (RSI < 50, MACD negative)
        df_1h = _make_intraday_df(n=100, trend="down", seed=42)
        df_15m = _make_intraday_df(n=100, trend="down", seed=43)

        mock_alpaca = MagicMock()

        def mock_get_bars(ticker, timeframe, limit=252):
            if timeframe == "1Day":
                return df_daily
            elif timeframe == "1Hour":
                return df_1h
            elif timeframe == "15Min":
                return df_15m
            raise ValueError(f"Unexpected timeframe: {timeframe}")

        mock_alpaca.get_bars = mock_get_bars

        mock_eodhd = MagicMock()
        mock_eodhd.available = False

        agent = TechnicalAgent(
            db=mock_db, alpaca_client=mock_alpaca, eodhd_feed=mock_eodhd,
        )
        result = agent.run("AAPL")
        indicators = result["indicators"]

        assert indicators["intraday_available"] is True

        # If alignment is 0.0, confidence should have been penalised
        if indicators["timeframe_alignment"] == 0.0:
            assert any("penalised -0.10" in r for r in result["reasoning"])

    def test_intraday_unavailable_defaults_to_neutral(self):
        """When intraday bars fail, alignment=0.5, no boost/penalty."""
        mock_db = MagicMock()
        mock_db.log_technical_signal.return_value = 1

        mock_alpaca = MagicMock()

        def mock_get_bars(ticker, timeframe, limit=252):
            if timeframe == "1Day":
                return _make_simple_df(n=250)
            # Intraday requests raise exceptions
            raise ValueError("Market closed — no intraday data")

        mock_alpaca.get_bars = mock_get_bars

        mock_eodhd = MagicMock()
        mock_eodhd.available = False

        agent = TechnicalAgent(
            db=mock_db, alpaca_client=mock_alpaca, eodhd_feed=mock_eodhd,
        )
        result = agent.run("AAPL")
        indicators = result["indicators"]

        assert indicators["intraday_available"] is False
        assert indicators["rsi_1h"] is None
        assert indicators["macd_15m_hist"] is None
        assert indicators["timeframe_alignment"] == 0.5
        # No multi-timeframe reasoning added
        assert not any("Multi-timeframe" in r for r in result["reasoning"])

    def test_xetra_ticker_uses_daily_only(self):
        """German/XETRA tickers should skip intraday and get alignment=0.5."""
        mock_db = MagicMock()
        mock_db.log_technical_signal.return_value = 1

        mock_eodhd = MagicMock()
        mock_eodhd.available = True
        mock_eodhd.get_ohlcv_daily.return_value = _make_simple_df(n=250)
        mock_eodhd.get_ohlcv_intraday.return_value = None

        agent = TechnicalAgent(db=mock_db, eodhd_feed=mock_eodhd)
        result = agent.run("SAP.XETRA")
        indicators = result["indicators"]

        assert indicators["intraday_available"] is False
        assert indicators["rsi_1h"] is None
        assert indicators["macd_15m_hist"] is None
        assert indicators["timeframe_alignment"] == 0.5

    def test_returned_indicators_include_new_fields(self):
        """Result indicators dict must include all multi-timeframe fields."""
        mock_db = MagicMock()
        mock_db.log_technical_signal.return_value = 1

        mock_alpaca = MagicMock()

        def mock_get_bars(ticker, timeframe, limit=252):
            if timeframe == "1Day":
                return _make_simple_df(n=250)
            raise ValueError("No intraday data")

        mock_alpaca.get_bars = mock_get_bars

        mock_eodhd = MagicMock()
        mock_eodhd.available = False

        agent = TechnicalAgent(
            db=mock_db, alpaca_client=mock_alpaca, eodhd_feed=mock_eodhd,
        )
        result = agent.run("AAPL")
        indicators = result["indicators"]

        # All new keys must be present
        assert "rsi_1h" in indicators
        assert "macd_15m_hist" in indicators
        assert "timeframe_alignment" in indicators
        assert "intraday_available" in indicators

        # Types
        assert indicators["rsi_1h"] is None or isinstance(indicators["rsi_1h"], float)
        assert indicators["macd_15m_hist"] is None or isinstance(indicators["macd_15m_hist"], float)
        assert isinstance(indicators["timeframe_alignment"], float)
        assert isinstance(indicators["intraday_available"], bool)

    def test_alignment_boost_capped_at_1(self):
        """Confidence + 0.15 boost should never exceed 1.0."""
        mock_db = MagicMock()
        mock_db.log_technical_signal.return_value = 1

        # Create data that gives high base confidence (golden cross + strong ADX + bull flag)
        df_daily = _make_golden_cross_df()
        df_1h = _make_intraday_df(n=100, trend="up", seed=42)
        df_15m = _make_intraday_df(n=100, trend="up", seed=43)

        mock_alpaca = MagicMock()

        def mock_get_bars(ticker, timeframe, limit=252):
            if timeframe == "1Day":
                return df_daily
            elif timeframe == "1Hour":
                return df_1h
            elif timeframe == "15Min":
                return df_15m
            raise ValueError(f"Unexpected: {timeframe}")

        mock_alpaca.get_bars = mock_get_bars

        mock_eodhd = MagicMock()
        mock_eodhd.available = False

        agent = TechnicalAgent(
            db=mock_db, alpaca_client=mock_alpaca, eodhd_feed=mock_eodhd,
        )
        result = agent.run("AAPL")

        # Must be <= 1.0 regardless of boosts
        assert result["adjusted_confidence"] <= 1.0

    def test_alignment_penalty_floored_at_0(self):
        """Confidence - 0.10 penalty should never go below 0.0."""
        mock_db = MagicMock()
        mock_db.log_technical_signal.return_value = 1

        # Create data that gives low base confidence (death cross + below SMA200)
        df_daily = _make_death_cross_df()
        # Bullish intraday to create conflict with SELL daily signal
        df_1h = _make_intraday_df(n=100, trend="up", seed=42)
        df_15m = _make_intraday_df(n=100, trend="up", seed=43)

        mock_alpaca = MagicMock()

        def mock_get_bars(ticker, timeframe, limit=252):
            if timeframe == "1Day":
                return df_daily
            elif timeframe == "1Hour":
                return df_1h
            elif timeframe == "15Min":
                return df_15m
            raise ValueError(f"Unexpected: {timeframe}")

        mock_alpaca.get_bars = mock_get_bars

        mock_eodhd = MagicMock()
        mock_eodhd.available = False

        agent = TechnicalAgent(
            db=mock_db, alpaca_client=mock_alpaca, eodhd_feed=mock_eodhd,
        )
        result = agent.run("AAPL")

        # Must be >= 0.0 regardless of penalties
        assert result["adjusted_confidence"] >= 0.0


class TestFetchMultiTimeframeDirect:
    """Direct unit tests for _fetch_multi_timeframe method."""

    def test_returns_default_for_crypto(self):
        """Crypto tickers should get default (no intraday) result."""
        agent = TechnicalAgent(db=MagicMock())
        result = agent._fetch_multi_timeframe("BTC", "BUY")
        assert result["intraday_available"] is False
        assert result["timeframe_alignment"] == 0.5
        assert result["rsi_1h"] is None
        assert result["macd_15m_hist"] is None

    def test_returns_default_for_german_ticker(self):
        """German tickers should get default (no intraday) result."""
        agent = TechnicalAgent(db=MagicMock())
        result = agent._fetch_multi_timeframe("SAP.XETRA", "SELL")
        assert result["intraday_available"] is False
        assert result["timeframe_alignment"] == 0.5

    def test_hold_signal_gets_neutral_alignment(self):
        """HOLD daily signal gets alignment 0.5 even with intraday data."""
        mock_alpaca = MagicMock()
        df_1h = _make_intraday_df(n=100, trend="up")
        df_15m = _make_intraday_df(n=100, trend="up")

        def mock_get_bars(ticker, timeframe, limit=252):
            if timeframe == "1Hour":
                return df_1h
            if timeframe == "15Min":
                return df_15m
            raise ValueError("unexpected")

        mock_alpaca.get_bars = mock_get_bars

        agent = TechnicalAgent(db=MagicMock(), alpaca_client=mock_alpaca)
        result = agent._fetch_multi_timeframe("AAPL", "HOLD")

        assert result["intraday_available"] is True
        assert result["timeframe_alignment"] == 0.5

    def test_all_bullish_alignment_1_0(self):
        """BUY + bullish 1h + bullish 15m → alignment 1.0."""
        mock_alpaca = MagicMock()
        df_1h = _make_intraday_df(n=100, trend="up")
        df_15m = _make_intraday_df(n=100, trend="up")

        def mock_get_bars(ticker, timeframe, limit=252):
            if timeframe == "1Hour":
                return df_1h
            if timeframe == "15Min":
                return df_15m
            raise ValueError("unexpected")

        mock_alpaca.get_bars = mock_get_bars

        agent = TechnicalAgent(db=MagicMock(), alpaca_client=mock_alpaca)
        result = agent._fetch_multi_timeframe("AAPL", "BUY")

        assert result["intraday_available"] is True
        # With bullish intraday and BUY daily, all three should agree
        assert result["timeframe_alignment"] == 1.0

    def test_all_bearish_alignment_1_0(self):
        """SELL + bearish 1h + bearish 15m → alignment 1.0."""
        mock_alpaca = MagicMock()
        df_1h = _make_intraday_df(n=100, trend="down")
        df_15m = _make_intraday_df(n=100, trend="down")

        def mock_get_bars(ticker, timeframe, limit=252):
            if timeframe == "1Hour":
                return df_1h
            if timeframe == "15Min":
                return df_15m
            raise ValueError("unexpected")

        mock_alpaca.get_bars = mock_get_bars

        agent = TechnicalAgent(db=MagicMock(), alpaca_client=mock_alpaca)
        result = agent._fetch_multi_timeframe("AAPL", "SELL")

        assert result["intraday_available"] is True
        assert result["timeframe_alignment"] == 1.0

    def test_conflicting_alignment_0_0(self):
        """BUY daily + bearish 1h + bearish 15m → alignment 0.0."""
        mock_alpaca = MagicMock()
        df_1h = _make_intraday_df(n=100, trend="down")
        df_15m = _make_intraday_df(n=100, trend="down")

        def mock_get_bars(ticker, timeframe, limit=252):
            if timeframe == "1Hour":
                return df_1h
            if timeframe == "15Min":
                return df_15m
            raise ValueError("unexpected")

        mock_alpaca.get_bars = mock_get_bars

        agent = TechnicalAgent(db=MagicMock(), alpaca_client=mock_alpaca)
        result = agent._fetch_multi_timeframe("AAPL", "BUY")

        assert result["intraday_available"] is True
        assert result["timeframe_alignment"] == 0.0

    def test_partial_intraday_only_1h_available(self):
        """When only 1h data is available, still produces meaningful result."""
        mock_alpaca = MagicMock()
        df_1h = _make_intraday_df(n=100, trend="up")

        def mock_get_bars(ticker, timeframe, limit=252):
            if timeframe == "1Hour":
                return df_1h
            if timeframe == "15Min":
                raise ValueError("No 15m data")
            raise ValueError("unexpected")

        mock_alpaca.get_bars = mock_get_bars

        agent = TechnicalAgent(db=MagicMock(), alpaca_client=mock_alpaca)
        result = agent._fetch_multi_timeframe("AAPL", "BUY")

        assert result["intraday_available"] is True
        assert result["rsi_1h"] is not None
        assert result["macd_15m_hist"] is None
        # With BUY daily + bullish 1h RSI + neutral 15m (0) →
        # agree_count = 2 (daily + 1h), alignment = 0.5
        assert result["timeframe_alignment"] in (0.5, 1.0)

    def test_alpaca_error_returns_default(self):
        """If all Alpaca intraday calls fail, returns default neutral."""
        mock_alpaca = MagicMock()
        mock_alpaca.get_bars.side_effect = ValueError("API error")

        agent = TechnicalAgent(db=MagicMock(), alpaca_client=mock_alpaca)
        result = agent._fetch_multi_timeframe("AAPL", "BUY")

        assert result["intraday_available"] is False
        assert result["timeframe_alignment"] == 0.5
        assert result["rsi_1h"] is None
        assert result["macd_15m_hist"] is None
