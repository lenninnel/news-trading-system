"""
Binance crypto OHLCV feed.

BinanceFeed fetches daily OHLCV candlestick data from the Binance public
klines endpoint.  No API key is required for public market data.

Results are cached per symbol for 4 hours to avoid redundant requests.

Requires:
    pip install pandas requests
"""

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Module-level cache: {symbol: {"df": DataFrame, "fetched_at": float}}
_cache: dict[str, dict] = {}
_CACHE_TTL = 4 * 3600  # 4 hours

_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def clear_cache() -> None:
    """Clear the module-level OHLCV cache."""
    _cache.clear()


class BinanceFeed:
    """
    Fetches daily OHLCV candlestick data for crypto symbols from Binance.

    Example::

        feed = BinanceFeed()
        df = feed.get_ohlcv("BTC")
        # DataFrame with columns: Open, High, Low, Close, Volume
    """

    def get_ohlcv(self, symbol: str, limit: int = 100) -> pd.DataFrame | None:
        """
        Fetch daily OHLCV for a crypto symbol from Binance.

        Args:
            symbol: Crypto symbol WITHOUT suffix (e.g. "BTC", "ETH").
            limit: Number of days (default 100).

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            (same format as yfinance output) with DatetimeIndex.
            Returns None on failure.
        """
        symbol = symbol.upper()

        # Check cache
        cached = _cache.get(symbol)
        if cached is not None:
            age = time.time() - cached["fetched_at"]
            if age < _CACHE_TTL:
                logger.debug("Cache hit for %s (age %.0fs)", symbol, age)
                return cached["df"]

        try:
            pair = f"{symbol}USDT"
            resp = requests.get(
                _BINANCE_KLINES_URL,
                params={"symbol": pair, "interval": "1d", "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            klines = resp.json()

            if not isinstance(klines, list) or len(klines) == 0:
                logger.warning("Empty klines response for %s", pair)
                return None

            # Parse klines into DataFrame
            # Each kline: [open_time, open, high, low, close, volume, close_time, ...]
            rows = []
            for k in klines:
                rows.append(
                    {
                        "Date": pd.to_datetime(k[0], unit="ms"),
                        "Open": float(k[1]),
                        "High": float(k[2]),
                        "Low": float(k[3]),
                        "Close": float(k[4]),
                        "Volume": float(k[5]),
                    }
                )

            df = pd.DataFrame(rows)
            df.set_index("Date", inplace=True)

            # Store in cache
            _cache[symbol] = {"df": df, "fetched_at": time.time()}
            logger.info("Fetched %d candles for %s", len(df), pair)
            return df

        except Exception as exc:
            logger.error("Failed to fetch OHLCV for %s: %s", symbol, exc)
            return None

    def get_price(self, symbol: str) -> float | None:
        """
        Get latest close price for a crypto symbol.

        Args:
            symbol: Crypto symbol WITHOUT suffix (e.g. "BTC", "ETH").

        Returns:
            Latest close price as float, or None on failure.
        """
        df = self.get_ohlcv(symbol)
        if df is None or df.empty:
            return None
        try:
            return float(df["Close"].iloc[-1])
        except Exception as exc:
            logger.error("Failed to extract price for %s: %s", symbol, exc)
            return None
