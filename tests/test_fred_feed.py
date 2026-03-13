"""Tests for the FRED macro data feed module."""

from unittest.mock import MagicMock, patch

import pandas as pd

from data.fred_feed import FredFeed, clear_cache


class TestFredFeed:
    """Tests for data.fred_feed.FredFeed."""

    def setup_method(self) -> None:
        clear_cache()

    # ── Happy path ────────────────────────────────────────────────────────

    @patch("data.fred_feed.fredapi.Fred")
    def test_happy_path(self, mock_fred_cls: MagicMock) -> None:
        """Successful fetch returns dict with vix, yield_curve, sp500."""
        mock_fred = MagicMock()
        mock_fred_cls.return_value = mock_fred

        # Each get_series call returns a pandas Series with one float
        def fake_series(series_id: str, observation_start: str = "") -> pd.Series:
            data = {
                "VIXCLS": pd.Series([18.5]),
                "T10Y2Y": pd.Series([0.42]),
                "SP500": pd.Series([5120.0]),
            }
            return data[series_id]

        mock_fred.get_series.side_effect = fake_series

        feed = FredFeed(api_key="test-key")
        result = feed.get_macro_regime()

        assert result is not None
        assert result["vix"] == 18.5
        assert result["yield_curve"] == 0.42
        assert result["sp500"] == 5120.0
        mock_fred_cls.assert_called_once_with(api_key="test-key")

    # ── No API key ────────────────────────────────────────────────────────

    def test_no_api_key_returns_none(self) -> None:
        """Empty API key should return None without making any requests."""
        feed = FredFeed(api_key="")
        result = feed.get_macro_regime()
        assert result is None

    # ── API error ─────────────────────────────────────────────────────────

    @patch("data.fred_feed.fredapi.Fred")
    def test_api_error_returns_none(self, mock_fred_cls: MagicMock) -> None:
        """Exception during FRED instantiation returns None, does not raise."""
        mock_fred_cls.side_effect = Exception("FRED API unavailable")

        feed = FredFeed(api_key="test-key")
        result = feed.get_macro_regime()

        assert result is None

    # ── Caching ───────────────────────────────────────────────────────────

    @patch("data.fred_feed.fredapi.Fred")
    def test_caching_prevents_second_call(self, mock_fred_cls: MagicMock) -> None:
        """Second call within TTL should use cache, not hit FRED again."""
        mock_fred = MagicMock()
        mock_fred_cls.return_value = mock_fred

        def fake_series(series_id: str, observation_start: str = "") -> pd.Series:
            data = {
                "VIXCLS": pd.Series([20.0]),
                "T10Y2Y": pd.Series([-0.1]),
                "SP500": pd.Series([4900.0]),
            }
            return data[series_id]

        mock_fred.get_series.side_effect = fake_series

        feed = FredFeed(api_key="test-key")
        first = feed.get_macro_regime()
        second = feed.get_macro_regime()

        assert first == second
        # Fred class should only be instantiated once (first call)
        assert mock_fred_cls.call_count == 1

    # ── Partial failure ───────────────────────────────────────────────────

    @patch("data.fred_feed.fredapi.Fred")
    def test_partial_series_failure(self, mock_fred_cls: MagicMock) -> None:
        """If one series fails, its value is None but others still populate."""
        mock_fred = MagicMock()
        mock_fred_cls.return_value = mock_fred

        def fake_series(series_id: str, observation_start: str = "") -> pd.Series:
            if series_id == "T10Y2Y":
                raise Exception("series not found")
            data = {
                "VIXCLS": pd.Series([15.0]),
                "SP500": pd.Series([5000.0]),
            }
            return data[series_id]

        mock_fred.get_series.side_effect = fake_series

        feed = FredFeed(api_key="test-key")
        result = feed.get_macro_regime()

        assert result is not None
        assert result["vix"] == 15.0
        assert result["yield_curve"] is None
        assert result["sp500"] == 5000.0

    # ── NaN handling ──────────────────────────────────────────────────────

    @patch("data.fred_feed.fredapi.Fred")
    def test_nan_values_skipped(self, mock_fred_cls: MagicMock) -> None:
        """NaN values in a series are dropped; last valid value is used."""
        mock_fred = MagicMock()
        mock_fred_cls.return_value = mock_fred

        def fake_series(series_id: str, observation_start: str = "") -> pd.Series:
            import numpy as np

            data = {
                "VIXCLS": pd.Series([18.0, np.nan]),
                "T10Y2Y": pd.Series([0.5, np.nan]),
                "SP500": pd.Series([5100.0, np.nan]),
            }
            return data[series_id]

        mock_fred.get_series.side_effect = fake_series

        feed = FredFeed(api_key="test-key")
        result = feed.get_macro_regime()

        assert result is not None
        assert result["vix"] == 18.0
        assert result["yield_curve"] == 0.5
        assert result["sp500"] == 5100.0
