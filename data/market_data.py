"""
Market data source with multi-level price fallback.

MarketData fetches current price and basic fundamental information for a
ticker symbol.  It delegates to PriceFallback which implements a 4-level
chain (yfinance → Alpha Vantage → Yahoo Finance JSON → cached last price)
so the coordinator always gets a price dict, even when primary sources fail.

Requires:
    pip install yfinance
"""

from __future__ import annotations

from typing import Any

from data.price_fallback import PriceFallback


class MarketData:
    """
    Fetches current market data for a given ticker using PriceFallback.

    Args:
        db:        Optional Database for recovery event logging.
        alpha_key: Alpha Vantage API key (reads ALPHA_VANTAGE_KEY env if None).

    Example::

        md = MarketData()
        data = md.fetch("AAPL")
        # {
        #   "ticker": "AAPL",
        #   "name": "Apple Inc.",
        #   "price": 189.30,
        #   "currency": "USD",
        #   "market_cap": 2_950_000_000_000,
        # }
    """

    def __init__(
        self,
        db: "Any | None" = None,
        alpha_key: "str | None" = None,
    ) -> None:
        self._fallback = PriceFallback(db=db, alpha_key=alpha_key)

    def fetch(self, ticker: str) -> dict:
        """
        Retrieve current market data for *ticker*.

        Uses the PriceFallback 4-level chain so the call never raises on
        transient network issues.  When all live sources fail the returned
        price is None and ``degraded=True`` in the result.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").

        Returns:
            dict with keys:
                ticker      (str):        The requested ticker symbol.
                name        (str):        Full company name, or "".
                price       (float|None): Current / last market price.
                currency    (str):        Currency code (e.g. "USD").
                market_cap  (int|None):   Market capitalisation in base currency.
                source      (str):        Which fallback level supplied the data.
                degraded    (bool):       True when data is not from the primary source.

        Raises:
            Exception: Only when an unexpected programming error occurs; all
                       network / data errors are handled internally.
        """
        result = self._fallback.get_price(ticker)
        return {
            "ticker":     result.ticker,
            "name":       result.name,
            "price":      result.price,
            "currency":   result.currency,
            "market_cap": result.market_cap,
            "source":     result.source,
            "degraded":   result.degraded,
        }
