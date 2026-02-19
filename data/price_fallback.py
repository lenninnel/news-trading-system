"""
Multi-source price data with 4-level fallback chain.

Fallback chain
--------------
Level 0 — yfinance            Primary (yf.Ticker.info)
Level 1 — Alpha Vantage       Requires ALPHA_VANTAGE_KEY env var
Level 2 — Yahoo Finance JSON  Direct Yahoo Finance chart API (no auth)
Level 3 — Last known price    From ResponseCache or DB; is_estimated=True

For critical operations pass ``require_fresh=True`` to raise
``FreshDataRequired`` when only Level 3 (estimated) data is available.

FallbackCoordinator integration
--------------------------------
After each call the source level is registered with FallbackCoordinator.

Usage
-----
    from data.price_fallback import PriceFallback

    pf     = PriceFallback(db=db)
    result = pf.get_price("AAPL")
    print(result.price, result.source, result.is_estimated)

    # Require fresh data (raises FreshDataRequired if only cache available)
    result = pf.get_price("AAPL", require_fresh=True)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import requests
import yfinance as yf

from utils.api_recovery import APIRecovery, CircuitOpenError
from utils.network_recovery import get_cache

log = logging.getLogger(__name__)

_CACHE_SERVICE = "yfinance_price"
_YF_TIMEOUT    = 10
_AV_URL        = "https://www.alphavantage.co/query"
_YF_CHART_URL  = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


# ── Exceptions ─────────────────────────────────────────────────────────────────

class FreshDataRequired(RuntimeError):
    """Raised when require_fresh=True but only estimated/cached data is available."""

    def __init__(self, ticker: str) -> None:
        super().__init__(
            f"Fresh price data required for {ticker} but all live sources failed. "
            "Cannot proceed with estimated price for this critical operation."
        )


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PriceResult:
    """Price data with provenance metadata."""
    ticker:       str
    price:        "float | None"
    source:       str       # "yfinance" | "alpha_vantage" | "yahoo_json" | "cache" | "none"
    level:        int       # 0-3
    is_fresh:     bool      # False when using cached/estimated data
    is_estimated: bool      # True when no live source succeeded
    currency:     str       = "USD"
    name:         str       = ""
    market_cap:   "int | None" = None
    change_pct:   "float | None" = None   # vs previous close (when available)
    degraded:     bool      = False


# ── PriceFallback ─────────────────────────────────────────────────────────────

class PriceFallback:
    """
    Multi-source price data retrieval with automatic fallback.

    Args:
        db:            Optional Database for recovery event logging.
        alpha_key:     Alpha Vantage API key.  Reads ALPHA_VANTAGE_KEY env if None.
    """

    def __init__(
        self,
        db:        "Any | None" = None,
        alpha_key: "str | None" = None,
    ) -> None:
        self._db        = db
        self._alpha_key = alpha_key or os.environ.get("ALPHA_VANTAGE_KEY", "")
        if db is not None:
            APIRecovery.set_db(db)

    # -- Public API -----------------------------------------------------------

    def get_price(self, ticker: str, require_fresh: bool = False) -> PriceResult:
        """
        Fetch the current price for *ticker* using all available sources.

        Args:
            ticker:        Stock ticker symbol (e.g. "AAPL").
            require_fresh: If True, raises FreshDataRequired when only
                           Level 3 (cached/estimated) data is available.

        Returns:
            PriceResult with price and provenance metadata.

        Raises:
            FreshDataRequired: Only when require_fresh=True and all live
                               sources failed.
        """
        ticker = ticker.upper()

        for level, (name, fn) in enumerate([
            ("yfinance",      lambda t: self._level0_yfinance(t)),
            ("alpha_vantage", lambda t: self._level1_alpha_vantage(t)),
            ("yahoo_json",    lambda t: self._level2_yahoo_json(t)),
            ("cache",         lambda t: self._level3_cache(t)),
        ]):
            # Skip Alpha Vantage if no key configured
            if level == 1 and not self._alpha_key:
                log.debug("Alpha Vantage key not set — skipping Level 1 for %s", ticker)
                continue

            try:
                result = fn(ticker)
                if result is not None and result.price is not None:
                    # Cache the price for Level 3 reuse
                    get_cache().set(
                        _CACHE_SERVICE, ticker,
                        {"price": result.price, "currency": result.currency,
                         "name": result.name, "market_cap": result.market_cap},
                    )

                    if level > 0:
                        log.warning(
                            "[DEGRADED L%d] Price for %s from %s: %.2f",
                            level, ticker, name, result.price,
                        )
                    self._register(ticker, level, name)
                    return result

            except FreshDataRequired:
                raise
            except Exception as exc:
                log.warning("Price source '%s' failed for %s: %s", name, ticker, exc)

        # All sources failed
        if require_fresh:
            raise FreshDataRequired(ticker)

        log.error("All price sources failed for %s — returning None price", ticker)
        return PriceResult(
            ticker=ticker, price=None, source="none", level=4,
            is_fresh=False, is_estimated=True, degraded=True,
        )

    # -- Level 0: yfinance ---------------------------------------------------

    def _level0_yfinance(self, ticker: str) -> PriceResult:
        """Primary: yfinance Ticker.info — current/regular market price."""
        def _fetch():
            info  = yf.Ticker(ticker).info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if price is None:
                raise ValueError(f"yfinance returned no price for {ticker}")
            return {
                "price":      float(price),
                "currency":   info.get("currency", "USD"),
                "name":       info.get("longName", ""),
                "market_cap": info.get("marketCap"),
                "change_pct": info.get("regularMarketChangePercent"),
            }

        data = APIRecovery.call("yfinance", _fetch, ticker=ticker)
        return PriceResult(
            ticker=ticker,
            price=data["price"],
            source="yfinance",
            level=0,
            is_fresh=True,
            is_estimated=False,
            currency=data["currency"],
            name=data["name"],
            market_cap=data["market_cap"],
            change_pct=data["change_pct"],
        )

    # -- Level 1: Alpha Vantage ----------------------------------------------

    def _level1_alpha_vantage(self, ticker: str) -> PriceResult:
        """Fallback 1: Alpha Vantage GLOBAL_QUOTE (free tier: 25 req/day)."""
        def _fetch():
            resp = requests.get(
                _AV_URL,
                params={
                    "function": "GLOBAL_QUOTE",
                    "symbol":   ticker,
                    "apikey":   self._alpha_key,
                },
                timeout=_YF_TIMEOUT,
            )
            resp.raise_for_status()
            data  = resp.json()
            quote = data.get("Global Quote", {})
            price_str = quote.get("05. price")
            if not price_str:
                raise ValueError(
                    f"Alpha Vantage returned no price for {ticker}: {data}"
                )
            change_str = quote.get("10. change percent", "").rstrip("%")
            return {
                "price":      float(price_str),
                "change_pct": float(change_str) if change_str else None,
            }

        data = _fetch()
        return PriceResult(
            ticker=ticker,
            price=data["price"],
            source="alpha_vantage",
            level=1,
            is_fresh=True,
            is_estimated=False,
            change_pct=data.get("change_pct"),
            degraded=True,
        )

    # -- Level 2: Yahoo Finance JSON API -------------------------------------

    def _level2_yahoo_json(self, ticker: str) -> PriceResult:
        """
        Fallback 2: Yahoo Finance chart API — no authentication required.

        Endpoint: https://query1.finance.yahoo.com/v8/finance/chart/{ticker}
        Returns OHLCV; we take the most recent close price.
        """
        def _fetch():
            url  = _YF_CHART_URL.format(ticker=ticker)
            resp = requests.get(
                url,
                params={"interval": "1d", "range": "5d"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=_YF_TIMEOUT,
            )
            resp.raise_for_status()
            data  = resp.json()
            chart = data.get("chart", {})
            result_list = chart.get("result")
            if not result_list:
                raise ValueError(f"Yahoo Finance JSON returned no data for {ticker}")

            meta      = result_list[0].get("meta", {})
            price_raw = (
                meta.get("regularMarketPrice")
                or meta.get("chartPreviousClose")
            )
            if price_raw is None:
                # Try last close from indicators
                closes = (
                    result_list[0]
                    .get("indicators", {})
                    .get("quote", [{}])[0]
                    .get("close", [])
                )
                price_raw = next((c for c in reversed(closes) if c is not None), None)
            if price_raw is None:
                raise ValueError(f"No price in Yahoo Finance JSON response for {ticker}")
            return {
                "price":    float(price_raw),
                "currency": meta.get("currency", "USD"),
                "name":     meta.get("longName", ""),
            }

        data = _fetch()
        return PriceResult(
            ticker=ticker,
            price=data["price"],
            source="yahoo_json",
            level=2,
            is_fresh=True,
            is_estimated=False,
            currency=data["currency"],
            name=data["name"],
            degraded=True,
        )

    # -- Level 3: Last known price -------------------------------------------

    def _level3_cache(self, ticker: str) -> "PriceResult | None":
        """
        Fallback 3: Return the last price we cached from any live source.

        The stored entry is served as-is (no drift estimate applied — callers
        can decide whether to use it based on is_estimated=True).
        """
        cached, hit = get_cache().get(_CACHE_SERVICE, ticker)
        if not hit or cached is None:
            # Try DB as secondary cache
            if self._db is not None:
                try:
                    rows = self._db._select(
                        "SELECT price FROM technical_signals "
                        "WHERE ticker = ? AND price IS NOT NULL "
                        "ORDER BY created_at DESC LIMIT 1",
                        (ticker,),
                    )
                    if rows and rows[0].get("price"):
                        cached = {"price": float(rows[0]["price"]), "currency": "USD",
                                  "name": "", "market_cap": None}
                        hit = True
                except Exception:
                    pass

        if not hit or cached is None:
            return None

        log.warning(
            "[CACHE FALLBACK] Last known price for %s: %.2f (is_estimated=True)",
            ticker, cached["price"],
        )
        return PriceResult(
            ticker=ticker,
            price=cached["price"],
            source="cache",
            level=3,
            is_fresh=False,
            is_estimated=True,
            currency=cached.get("currency", "USD"),
            name=cached.get("name", ""),
            market_cap=cached.get("market_cap"),
            degraded=True,
        )

    # -- FallbackCoordinator registration ------------------------------------

    @staticmethod
    def _register(ticker: str, level: int, source: str) -> None:
        try:
            from data.fallback_coordinator import FallbackCoordinator
            FallbackCoordinator.register("price", level, source, ticker=ticker)
        except Exception:
            pass
