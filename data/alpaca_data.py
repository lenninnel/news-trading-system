"""
Alpaca Market Data API client — thin wrapper for reliable price data.

Replaces yfinance as the primary data source for US stocks.  Provides
current prices, historical OHLCV bars, and full snapshots via the
Alpaca Data API (https://data.alpaca.markets).

Alpaca free-tier delivers 15-minute delayed quotes, but the prices are
reliable (no phantom $150 AAPL quotes that yfinance occasionally returns).

For European (XETRA/DE) tickers Alpaca has no coverage.  Callers should
fall back to yfinance or EODHD for those.

Environment variables
---------------------
ALPACA_API_KEY      Alpaca API key ID
ALPACA_SECRET_KEY   Alpaca API secret key

Usage
-----
    from data.alpaca_data import AlpacaDataClient

    client = AlpacaDataClient()
    price  = client.get_current_price("AAPL")           # float
    bars   = client.get_bars("AAPL", "1Day", limit=100) # pd.DataFrame
    snap   = client.get_snapshot("AAPL")                 # dict
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# Alpaca paper-trading and data URLs
_PAPER_URL = "https://paper-api.alpaca.markets"
_DATA_URL = "https://data.alpaca.markets"


def _build_rest_client() -> Any:
    """
    Build an ``alpaca_trade_api.REST`` instance configured for market data.

    Raises RuntimeError if API credentials are missing.
    """
    import alpaca_trade_api as tradeapi

    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")

    if not key or not secret:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set "
            "to use the Alpaca data client"
        )

    return tradeapi.REST(
        key_id=key,
        secret_key=secret,
        base_url=_PAPER_URL,
        api_version="v2",
    )


class AlpacaDataClient:
    """
    Thin wrapper around the Alpaca Data API.

    Args:
        api: Optional pre-built ``tradeapi.REST`` instance (for testing).
    """

    def __init__(self, api: Any | None = None) -> None:
        self._api = api

    @property
    def api(self) -> Any:
        """Lazy-initialise the REST client on first use."""
        if self._api is None:
            self._api = _build_rest_client()
        return self._api

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_price(self, ticker: str) -> float:
        """
        Fetch the latest quote price for *ticker*.

        Tries latest trade first (most reliable), then latest quote mid-price.

        Args:
            ticker: US stock ticker symbol (e.g. "AAPL").

        Returns:
            Current price as float.

        Raises:
            ValueError: If Alpaca returns no data or a zero price.
        """
        ticker = ticker.upper()
        try:
            trade = self.api.get_latest_trade(ticker)
            price = float(trade.price)
            if price > 0:
                log.debug("Alpaca latest trade for %s: $%.2f", ticker, price)
                return price
        except Exception as exc:
            log.debug("Alpaca latest trade failed for %s: %s", ticker, exc)

        # Fallback: latest quote mid-price
        try:
            quote = self.api.get_latest_quote(ticker)
            bid = float(quote.bid_price) if quote.bid_price else 0.0
            ask = float(quote.ask_price) if quote.ask_price else 0.0
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                log.debug("Alpaca quote mid for %s: $%.2f", ticker, mid)
                return mid
            if ask > 0:
                return ask
            if bid > 0:
                return bid
        except Exception as exc:
            log.debug("Alpaca latest quote failed for %s: %s", ticker, exc)

        raise ValueError(f"Alpaca returned no price data for {ticker}")

    def get_bars(
        self,
        ticker: str,
        timeframe: str = "1Day",
        limit: int = 252,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars for *ticker*.

        Args:
            ticker:    US stock ticker symbol.
            timeframe: Bar timeframe — "1Min", "5Min", "15Min", "1Hour", "1Day".
            limit:     Maximum number of bars to return (default 252 ~ 1 year).
            start:     ISO-8601 start date (optional).
            end:       ISO-8601 end date (optional).

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            indexed by timestamp.

        Raises:
            ValueError: If Alpaca returns empty data.
        """
        ticker = ticker.upper()
        try:
            kwargs: dict[str, Any] = {"limit": limit}
            if start:
                kwargs["start"] = start
            if end:
                kwargs["end"] = end

            bars = self.api.get_bars(ticker, timeframe, **kwargs)
            df = bars.df if hasattr(bars, "df") else pd.DataFrame(bars)

            if df.empty:
                raise ValueError(f"Alpaca returned no bars for {ticker}")

            # Normalise column names to title-case (match yfinance convention)
            col_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "trade_count": "TradeCount",
                "vwap": "VWAP",
            }
            df.rename(columns=col_map, inplace=True)

            # Ensure standard OHLCV columns exist
            for col in ("Open", "High", "Low", "Close", "Volume"):
                if col not in df.columns:
                    raise ValueError(
                        f"Alpaca bars for {ticker} missing '{col}' column"
                    )

            if len(df) < 20:
                raise ValueError(
                    f"Alpaca returned only {len(df)} bars for {ticker} (need >= 20)"
                )

            log.debug(
                "Alpaca bars for %s: %d rows (%s)", ticker, len(df), timeframe,
            )
            return df

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Alpaca bar fetch failed for {ticker}: {exc}"
            ) from exc

    def get_intraday_bars(
        self,
        ticker: str,
        timeframe: str = "5Min",
        limit: int = 100,
    ) -> pd.DataFrame | None:
        """
        Fetch intraday OHLCV bars for *ticker*.

        Args:
            ticker:    US stock ticker symbol.
            timeframe: Bar timeframe — "5Min", "15Min", or "1Hour".
            limit:     Maximum number of bars to return (default 100).

        Returns:
            DataFrame with columns Open, High, Low, Close, Volume
            indexed by timestamp, or None if data is unavailable.
        """
        ticker = ticker.upper()

        # XETRA tickers are not available on Alpaca
        if ".XETRA" in ticker or ticker.endswith(".DE"):
            log.warning("Intraday bars not available for XETRA ticker %s", ticker)
            return None

        valid_timeframes = {"5Min", "15Min", "1Hour"}
        if timeframe not in valid_timeframes:
            log.warning(
                "Invalid intraday timeframe '%s'; expected one of %s",
                timeframe,
                valid_timeframes,
            )
            return None

        try:
            bars = self.api.get_bars(ticker, timeframe, limit=limit)
            df = bars.df if hasattr(bars, "df") else pd.DataFrame(bars)

            if df is None or df.empty:
                log.warning(
                    "Alpaca returned no intraday bars for %s (%s)", ticker, timeframe,
                )
                return None

            # Normalise column names to title-case (match existing get_bars format)
            col_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "trade_count": "TradeCount",
                "vwap": "VWAP",
            }
            df.rename(columns=col_map, inplace=True)

            # Verify standard OHLCV columns are present
            for col in ("Open", "High", "Low", "Close", "Volume"):
                if col not in df.columns:
                    log.warning(
                        "Alpaca intraday bars for %s missing '%s' column",
                        ticker,
                        col,
                    )
                    return None

            log.debug(
                "Alpaca intraday bars for %s: %d rows (%s)",
                ticker,
                len(df),
                timeframe,
            )
            return df

        except Exception as exc:
            log.warning(
                "Alpaca intraday bar fetch failed for %s (%s): %s",
                ticker,
                timeframe,
                exc,
            )
            return None

    def get_multi_timeframe_bars(self, ticker: str) -> dict[str, pd.DataFrame | None]:
        """
        Fetch bars at multiple timeframes for *ticker*.

        Returns:
            dict with keys '5min', '15min', '1hour', '1day', each mapping
            to a DataFrame or None if data is unavailable.
        """
        result: dict[str, pd.DataFrame | None] = {}

        # Intraday timeframes via get_intraday_bars (returns None on failure)
        result["5min"] = self.get_intraday_bars(ticker, timeframe="5Min", limit=100)
        result["15min"] = self.get_intraday_bars(ticker, timeframe="15Min", limit=100)
        result["1hour"] = self.get_intraday_bars(ticker, timeframe="1Hour", limit=50)

        # Daily bars via existing get_bars (raises on failure, so catch)
        try:
            result["1day"] = self.get_bars(ticker, timeframe="1Day", limit=100)
        except (ValueError, Exception) as exc:
            log.warning("Daily bars unavailable for %s: %s", ticker, exc)
            result["1day"] = None

        return result

    def get_snapshot(self, ticker: str) -> dict:
        """
        Fetch the latest snapshot for *ticker* (price, volume, daily bar).

        Args:
            ticker: US stock ticker symbol.

        Returns:
            dict with keys: ticker, price, volume, prev_close, change_pct,
            name, currency, market_cap (None — Alpaca does not provide this).

        Raises:
            ValueError: If Alpaca returns no snapshot.
        """
        ticker = ticker.upper()
        try:
            snap = self.api.get_snapshot(ticker)
            price = float(snap.latest_trade.price) if snap.latest_trade else None
            volume = int(snap.daily_bar.volume) if snap.daily_bar else 0
            prev_close = float(snap.prev_daily_bar.close) if snap.prev_daily_bar else None

            if price is None or price <= 0:
                raise ValueError(f"Alpaca snapshot has no valid price for {ticker}")

            change_pct = None
            if prev_close and prev_close > 0:
                change_pct = round((price - prev_close) / prev_close * 100, 2)

            return {
                "ticker": ticker,
                "price": price,
                "volume": volume,
                "prev_close": prev_close,
                "change_pct": change_pct,
                "currency": "USD",
                "name": "",            # Alpaca does not provide company name
                "market_cap": None,    # Alpaca does not provide market cap
            }

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Alpaca snapshot failed for {ticker}: {exc}"
            ) from exc


# ── Module-level singleton for convenience ────────────────────────────────

_default_client: AlpacaDataClient | None = None


def get_alpaca_client(api: Any | None = None) -> AlpacaDataClient:
    """Return a module-level AlpacaDataClient singleton."""
    global _default_client
    if _default_client is None or api is not None:
        _default_client = AlpacaDataClient(api=api)
    return _default_client
