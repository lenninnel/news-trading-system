"""yfinance-based OHLCV data feed.

Drop-in replacement for AlpacaDataClient.get_bars() — same column
format (Open, High, Low, Close, Volume) and validation rules.

Used for all bar data so Alpaca is reserved for trade execution only.
"""

from __future__ import annotations

import logging
import math

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# Alpaca timeframe → yfinance (interval, period-per-bar-in-days)
_TIMEFRAME_MAP: dict[str, tuple[str, float]] = {
    "1Day":  ("1d",  1.0),
    "1Hour": ("1h",  1.0 / 6.5),   # ~6.5 trading hours per day
    "15Min": ("15m", 1.0 / 26.0),  # ~26 × 15-min bars per day
}


def _yf_ticker(ticker: str) -> str:
    """Convert internal ticker symbols to yfinance format.

    SAP.XETRA → SAP.DE, SIE.XETRA → SIE.DE, etc.
    """
    t = ticker.upper()
    if t.endswith(".XETRA"):
        return t.rsplit(".", 1)[0] + ".DE"
    return t


def _period_for(interval: str, limit: int, days_per_bar: float) -> str:
    """Pick the smallest yfinance period string that covers *limit* bars."""
    raw_days = math.ceil(limit * days_per_bar * 1.5)  # 50% padding

    if interval in ("1h", "15m", "5m", "1m"):
        # yfinance caps intraday history at ~60 days
        capped = min(raw_days, 59)
        return f"{capped}d"

    if raw_days <= 5:
        return "5d"
    if raw_days <= 30:
        return "1mo"
    if raw_days <= 90:
        return "3mo"
    if raw_days <= 180:
        return "6mo"
    return "1y"


class YFinanceFeed:
    """Fetch OHLCV bars via yfinance with the same interface as AlpacaDataClient."""

    def get_bars(
        self,
        ticker: str,
        timeframe: str = "1Day",
        limit: int = 252,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars using yfinance.

        Args:
            ticker:    Stock symbol. XETRA tickers (.XETRA suffix)
                       are converted: SAP.XETRA → SAP.DE.
            timeframe: ``"1Day"``, ``"1Hour"``, or ``"15Min"``
                       (Alpaca convention).
            limit:     Approximate number of bars to fetch.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            (title-case, timezone-naive index).

        Raises:
            ValueError: If *timeframe* is unknown, data is empty,
                        or fewer than 20 bars are returned.
        """
        if timeframe not in _TIMEFRAME_MAP:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {', '.join(_TIMEFRAME_MAP)}"
            )

        yf_interval, days_per_bar = _TIMEFRAME_MAP[timeframe]
        yf_symbol = _yf_ticker(ticker)
        period = _period_for(yf_interval, limit, days_per_bar)

        log.debug(
            "yfinance: %s interval=%s period=%s (limit=%d)",
            yf_symbol, yf_interval, period, limit,
        )

        df: pd.DataFrame = yf.download(
            yf_symbol,
            period=period,
            interval=yf_interval,
            progress=False,
            auto_adjust=True,
        )

        # yf.download returns MultiIndex columns for single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            raise ValueError(f"yfinance returned no data for {ticker} ({yf_symbol})")

        # Ensure title-case columns
        col_map = {
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        }
        df.rename(columns=col_map, inplace=True)

        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col not in df.columns:
                raise ValueError(
                    f"yfinance data for {ticker} missing '{col}' column"
                )

        # Drop timezone info if present (strategies expect tz-naive)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Trim to requested limit (keep most recent)
        if len(df) > limit:
            df = df.iloc[-limit:]

        if len(df) < 20:
            raise ValueError(
                f"yfinance returned only {len(df)} bars for {ticker} (need >= 20)"
            )

        log.debug("yfinance: %s → %d bars (%s)", ticker, len(df), timeframe)
        return df
