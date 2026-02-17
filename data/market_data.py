"""
Market data source.

MarketData fetches current price and basic fundamental information for a
ticker symbol using the yfinance library, which wraps the Yahoo Finance API.

This module is intentionally independent of the news/sentiment pipeline so
that market context can be attached to reports without coupling data sources.

Requires:
    pip install yfinance
"""

import yfinance as yf


class MarketData:
    """
    Fetches current market data for a given ticker.

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

    def fetch(self, ticker: str) -> dict:
        """
        Retrieve current market data for *ticker*.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").

        Returns:
            dict with keys:
                ticker      (str):        The requested ticker symbol.
                name        (str):        Full company name, or "N/A".
                price       (float|None): Current / last market price.
                currency    (str):        Currency code (e.g. "USD").
                market_cap  (int|None):   Market capitalisation in base currency.

        Raises:
            Exception: Propagates any yfinance or network errors to the caller.
        """
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        return {
            "ticker": ticker,
            "name": info.get("longName", "N/A"),
            "price": price,
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap"),
        }
