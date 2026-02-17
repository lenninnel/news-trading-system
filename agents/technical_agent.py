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

Results are persisted to the ``technical_signals`` table.

Requires:
    pip install yfinance ta
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import ta
import yfinance as yf

from agents.base_agent import BaseAgent
from storage.database import Database


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
        # }
    """

    # Enough history for SMA-50 and indicator warm-up
    _DOWNLOAD_PERIOD = "3mo"

    def __init__(self, db: Database | None = None) -> None:
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

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").

        Returns:
            dict with keys:
                ticker     (str):   The analysed ticker symbol.
                signal     (str):   BUY / SELL / HOLD.
                reasoning  (list):  Human-readable list of triggered conditions.
                indicators (dict):  Latest numeric values for each indicator.
                signal_id  (int):   Database primary key of the persisted row.

        Raises:
            ValueError: If no price data can be fetched for the ticker.
            Exception:  Propagates unexpected yfinance or pandas errors.
        """
        ticker = ticker.upper()

        # -- 1. Fetch price history ----------------------------------------
        df = self._fetch_history(ticker)

        # -- 2. Calculate indicators ----------------------------------------
        indicators = self._calculate_indicators(df)

        # -- 3. Apply signal rules ------------------------------------------
        signal, reasoning = self._apply_signal_rules(indicators)

        # -- 4. Persist to database -----------------------------------------
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
            "ticker": ticker,
            "signal": signal,
            "reasoning": reasoning,
            "indicators": indicators,
            "signal_id": signal_id,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_history(self, ticker: str) -> pd.DataFrame:
        """Download OHLCV data and validate the result."""
        df: pd.DataFrame = yf.download(
            ticker,
            period=self._DOWNLOAD_PERIOD,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            raise ValueError(f"No price data returned for ticker '{ticker}'")
        # yfinance may return MultiIndex columns — flatten to simple names.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> dict:
        """
        Compute RSI, MACD, SMA-20, SMA-50, and Bollinger Bands.

        Uses the ``ta`` library (pure-Python, no C dependencies).
        NaN values are stored as None.

        Args:
            df: DataFrame with at least a 'Close' column.

        Returns:
            dict of indicator names → latest float values (or None).
        """
        close: pd.Series = df["Close"].squeeze()

        def latest(series: pd.Series) -> "float | None":
            """Return the last non-NaN value, or None."""
            clean = series.dropna()
            return float(clean.iloc[-1]) if not clean.empty else None

        def prev(series: pd.Series) -> "float | None":
            """Return the second-to-last non-NaN value, or None."""
            clean = series.dropna()
            return float(clean.iloc[-2]) if len(clean) >= 2 else None

        # RSI (14-period)
        rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        # MACD (12, 26, 9)
        macd_obj = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
        macd_series   = macd_obj.macd()
        signal_series = macd_obj.macd_signal()
        hist_series   = macd_obj.macd_diff()

        # MACD crossover detection: compare last two rows
        macd_bull_cross = False
        macd_bear_cross = False
        cur_macd, cur_sig = latest(macd_series), latest(signal_series)
        prv_macd, prv_sig = prev(macd_series),   prev(signal_series)
        if all(v is not None for v in (cur_macd, cur_sig, prv_macd, prv_sig)):
            macd_bull_cross = (prv_macd <= prv_sig) and (cur_macd > cur_sig)
            macd_bear_cross = (prv_macd >= prv_sig) and (cur_macd < cur_sig)

        # SMA 20 and SMA 50
        sma20 = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        sma50 = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()

        # Bollinger Bands (20-period, 2 std dev)
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)

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
    def _apply_signal_rules(ind: dict) -> tuple[str, list[str]]:
        """
        Evaluate BUY / SELL / HOLD conditions against computed indicators.

        BUY conditions (any one triggers BUY):
            - RSI < 30 (oversold)
            - MACD bullish crossover (MACD crosses above signal line)
            - Price below lower Bollinger Band

        SELL conditions (any one triggers SELL; evaluated only when no BUY):
            - RSI > 70 (overbought)
            - MACD bearish crossover (MACD crosses below signal line)
            - Price above upper Bollinger Band

        Args:
            ind: Indicator dict from ``_calculate_indicators``.

        Returns:
            Tuple of (signal string, list of triggered condition descriptions).
        """
        buy_reasons:  list[str] = []
        sell_reasons: list[str] = []

        rsi    = ind.get("rsi")
        price  = ind.get("price")
        bb_low = ind.get("bb_lower")
        bb_up  = ind.get("bb_upper")

        # --- BUY conditions ---
        if rsi is not None and rsi < 30:
            buy_reasons.append(f"RSI {rsi:.1f} is below 30 (oversold)")

        if ind.get("macd_bull_cross"):
            buy_reasons.append("MACD bullish crossover (MACD crossed above signal line)")

        if price is not None and bb_low is not None and price < bb_low:
            buy_reasons.append(
                f"Price {price:.2f} is below lower Bollinger Band {bb_low:.2f}"
            )

        # --- SELL conditions ---
        if rsi is not None and rsi > 70:
            sell_reasons.append(f"RSI {rsi:.1f} is above 70 (overbought)")

        if ind.get("macd_bear_cross"):
            sell_reasons.append("MACD bearish crossover (MACD crossed below signal line)")

        if price is not None and bb_up is not None and price > bb_up:
            sell_reasons.append(
                f"Price {price:.2f} is above upper Bollinger Band {bb_up:.2f}"
            )

        # BUY takes priority when both conditions appear on the same bar
        if buy_reasons:
            return "BUY", buy_reasons
        if sell_reasons:
            return "SELL", sell_reasons
        return "HOLD", ["No extreme conditions detected — staying neutral"]
