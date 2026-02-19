"""
Technical analysis agent using classic price indicators.

TechnicalAgent fetches recent OHLCV data for a ticker, computes RSI, MACD,
SMA-20, SMA-50, and Bollinger Bands via the ``ta`` library, then applies
deterministic signal rules to produce BUY / SELL / HOLD.

Signal rules (any matching condition triggers the signal):
    BUY  — RSI < 30 (oversold)
           OR MACD line crosses above signal line (bullish crossover)
           OR latest close below lower Bollinger Band
    SELL — RSI > 70 (overbought)
           OR MACD line crosses below signal line (bearish crossover)
           OR latest close above upper Bollinger Band
    HOLD — none of the above conditions met

Recovery behaviour
------------------
  1. APIRecovery.call("yfinance", …) wraps every yfinance download call
     with per-service retry logic (max 5 attempts, 5 s backoff) and a
     circuit breaker (opens after 5 consecutive failures).

  2. On each successful download the resulting indicators dict is cached in
     the module-level ResponseCache (max age: 1 hour).

  3. On yfinance failure the recovery path is:
       a. Check the ResponseCache for a recent indicators snapshot.
          If found, re-run signal rules on cached indicators and return a
          result with ``"degraded": True``.
       b. If no cache, skip technical analysis and return a HOLD signal
          with ``"degraded": True`` and ``"skip_reason"`` explaining why.

  4. All fallback activations are logged to recovery_log when a Database
     instance has been attached to APIRecovery.

Results are persisted to the ``technical_signals`` table.

Requires:
    pip install yfinance ta
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import ta
import yfinance as yf

from agents.base_agent import BaseAgent
from storage.database import Database
from utils.api_recovery import APIRecovery, CircuitOpenError
from utils.network_recovery import get_cache

log = logging.getLogger(__name__)

_INDICATORS_CACHE_SERVICE = "yfinance"
_INDICATORS_CACHE_PREFIX  = "indicators"


class TechnicalAgent(BaseAgent):
    """
    Computes technical indicators and derives a trading signal.

    Attributes:
        _db: Database instance used to persist results.

    Example::

        agent = TechnicalAgent()
        result = agent.run("AAPL")
        # {
        #   "ticker": "AAPL",
        #   "signal": "BUY",
        #   "reasoning": ["RSI 28.4 is below 30 (oversold)"],
        #   "indicators": {"rsi": 28.4, "price": 189.3, ...},
        #   "signal_id": 1,
        #   "degraded": False,
        # }
    """

    # Enough history for SMA-50 and indicator warm-up
    _DOWNLOAD_PERIOD = "3mo"

    def __init__(self, db: "Database | None" = None) -> None:
        """
        Initialise the agent.

        Args:
            db: Optional Database instance for dependency injection.
                A new instance is created automatically when omitted.
        """
        self._db = db or Database()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "TechnicalAgent"

    def run(self, ticker: str, **kwargs: Any) -> dict:
        """
        Fetch price history, calculate indicators, and return a signal.

        Falls back to cached indicators or a HOLD skip when yfinance is down.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").

        Returns:
            dict with keys:
                ticker     (str):   The analysed ticker symbol.
                signal     (str):   BUY / SELL / HOLD.
                reasoning  (list):  Human-readable triggered conditions.
                indicators (dict):  Latest numeric indicator values.
                signal_id  (int):   DB primary key of the persisted row.
                degraded   (bool):  True when cached or fallback data was used.
        """
        ticker = ticker.upper()
        cache_key = f"{_INDICATORS_CACHE_PREFIX}:{ticker}"
        degraded  = False

        # -- 1. Fetch price history ----------------------------------------
        try:
            df = APIRecovery.call(
                "yfinance",
                self._download,
                ticker,
                ticker=ticker,
            )
            # Cache fresh indicators after computing them
            indicators = self._calculate_indicators(df)
            get_cache().set(_INDICATORS_CACHE_SERVICE, cache_key, indicators)

        except CircuitOpenError as exc:
            log.warning(
                "[DEGRADED MODE] yfinance circuit OPEN for %s — trying cache", ticker
            )
            indicators, degraded = self._fallback_indicators(ticker, cache_key, str(exc))

        except Exception as exc:
            log.warning(
                "[DEGRADED MODE] yfinance unavailable for %s (%s) — trying cache",
                ticker, exc,
            )
            indicators, degraded = self._fallback_indicators(ticker, cache_key, str(exc))

        # -- 2. Apply signal rules ------------------------------------------
        signal, reasoning = self._apply_signal_rules(indicators)

        if degraded:
            reasoning = [f"[CACHED DATA] {r}" for r in reasoning]

        # -- 3. Persist to database -----------------------------------------
        signal_id = self._db.log_technical_signal(
            ticker=ticker,
            signal=signal,
            reasoning="; ".join(reasoning) if reasoning else "No conditions triggered",
            **{k: indicators.get(k) for k in (
                "rsi", "macd", "macd_signal", "macd_hist",
                "sma_20", "sma_50", "bb_upper", "bb_lower", "price",
            )},
        )

        return {
            "ticker":     ticker,
            "signal":     signal,
            "reasoning":  reasoning,
            "indicators": indicators,
            "signal_id":  signal_id,
            "degraded":   degraded,
        }

    # ------------------------------------------------------------------
    # Fallback helpers
    # ------------------------------------------------------------------

    def _fallback_indicators(
        self, ticker: str, cache_key: str, error: str
    ) -> "tuple[dict, bool]":
        """
        Try the indicator cache; return empty HOLD indicators on miss.

        Returns:
            (indicators_dict, degraded=True)
        """
        cached, hit = get_cache().get(_INDICATORS_CACHE_SERVICE, cache_key)
        if hit:
            log.warning(
                "[DEGRADED MODE] yfinance: using cached indicators for %s", ticker
            )
            self._log_cache_hit(ticker, error)
            return cached, True  # type: ignore[return-value]

        log.warning(
            "[DEGRADED MODE] yfinance: no cache for %s — returning HOLD (skip)", ticker
        )
        self._log_skip(ticker, error)
        # Empty indicators → _apply_signal_rules will return HOLD
        return {
            "rsi": None, "macd": None, "macd_signal": None, "macd_hist": None,
            "macd_bull_cross": False, "macd_bear_cross": False,
            "sma_20": None, "sma_50": None,
            "bb_upper": None, "bb_lower": None, "price": None,
        }, True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download(self, ticker: str) -> pd.DataFrame:
        """Thin yfinance wrapper; raises on empty result."""
        df: pd.DataFrame = yf.download(
            ticker,
            period=self._DOWNLOAD_PERIOD,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            raise ValueError(f"No price data returned for ticker '{ticker}'")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    # Kept for backward-compat with tests that call _fetch_history directly
    def _fetch_history(self, ticker: str) -> pd.DataFrame:
        return self._download(ticker)

    def _calculate_indicators(self, df: pd.DataFrame) -> dict:
        """
        Compute RSI, MACD, SMA-20, SMA-50, and Bollinger Bands.

        Args:
            df: DataFrame with at least a 'Close' column.

        Returns:
            dict of indicator names → latest float values (or None).
        """
        close: pd.Series = df["Close"].squeeze()

        def latest(series: pd.Series) -> "float | None":
            clean = series.dropna()
            return float(clean.iloc[-1]) if not clean.empty else None

        def prev(series: pd.Series) -> "float | None":
            clean = series.dropna()
            return float(clean.iloc[-2]) if len(clean) >= 2 else None

        rsi_series  = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        macd_obj    = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
        macd_series   = macd_obj.macd()
        signal_series = macd_obj.macd_signal()
        hist_series   = macd_obj.macd_diff()

        macd_bull_cross = False
        macd_bear_cross = False
        cur_macd, cur_sig = latest(macd_series), latest(signal_series)
        prv_macd, prv_sig = prev(macd_series),   prev(signal_series)
        if all(v is not None for v in (cur_macd, cur_sig, prv_macd, prv_sig)):
            macd_bull_cross = (prv_macd <= prv_sig) and (cur_macd > cur_sig)  # type: ignore[operator]
            macd_bear_cross = (prv_macd >= prv_sig) and (cur_macd < cur_sig)  # type: ignore[operator]

        sma20 = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        sma50 = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
        bb    = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)

        return {
            "rsi":             latest(rsi_series),
            "macd":            cur_macd,
            "macd_signal":     cur_sig,
            "macd_hist":       latest(hist_series),
            "macd_bull_cross": macd_bull_cross,
            "macd_bear_cross": macd_bear_cross,
            "sma_20":          latest(sma20),
            "sma_50":          latest(sma50),
            "bb_upper":        latest(bb.bollinger_hband()),
            "bb_lower":        latest(bb.bollinger_lband()),
            "price":           float(close.iloc[-1]) if not close.empty else None,
        }

    @staticmethod
    def _apply_signal_rules(ind: dict) -> "tuple[str, list[str]]":
        """
        Evaluate BUY / SELL / HOLD conditions against computed indicators.

        BUY conditions (any one triggers BUY):
            - RSI < 30 (oversold)
            - MACD bullish crossover
            - Price below lower Bollinger Band

        SELL conditions (evaluated only when no BUY):
            - RSI > 70 (overbought)
            - MACD bearish crossover
            - Price above upper Bollinger Band
        """
        buy_reasons:  list[str] = []
        sell_reasons: list[str] = []

        rsi    = ind.get("rsi")
        price  = ind.get("price")
        bb_low = ind.get("bb_lower")
        bb_up  = ind.get("bb_upper")

        if rsi is not None and rsi < 30:
            buy_reasons.append(f"RSI {rsi:.1f} is below 30 (oversold)")
        if ind.get("macd_bull_cross"):
            buy_reasons.append("MACD bullish crossover (MACD crossed above signal line)")
        if price is not None and bb_low is not None and price < bb_low:
            buy_reasons.append(
                f"Price {price:.2f} is below lower Bollinger Band {bb_low:.2f}"
            )

        if rsi is not None and rsi > 70:
            sell_reasons.append(f"RSI {rsi:.1f} is above 70 (overbought)")
        if ind.get("macd_bear_cross"):
            sell_reasons.append("MACD bearish crossover (MACD crossed below signal line)")
        if price is not None and bb_up is not None and price > bb_up:
            sell_reasons.append(
                f"Price {price:.2f} is above upper Bollinger Band {bb_up:.2f}"
            )

        if buy_reasons:
            return "BUY", buy_reasons
        if sell_reasons:
            return "SELL", sell_reasons
        return "HOLD", ["No extreme conditions detected — staying neutral"]

    # -- Recovery logging ------------------------------------------------------

    def _log_cache_hit(self, ticker: str, error: str) -> None:
        db = APIRecovery._db
        if db is None:
            return
        try:
            db.log_recovery_event(
                service="yfinance",
                event_type="cache_hit",
                ticker=ticker,
                error_msg=error,
                recovery_action="using_cached_indicators",
                success=True,
            )
        except Exception:
            pass

    def _log_skip(self, ticker: str, error: str) -> None:
        db = APIRecovery._db
        if db is None:
            return
        try:
            db.log_recovery_event(
                service="yfinance",
                event_type="degraded_mode",
                ticker=ticker,
                error_msg=error,
                recovery_action="skip_technical_analysis_hold",
                success=False,
            )
        except Exception:
            pass
