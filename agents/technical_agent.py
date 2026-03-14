"""
Technical analysis agent using classic price indicators.

TechnicalAgent fetches recent OHLCV data for a ticker, computes RSI, MACD,
SMA-20, SMA-50, Bollinger Bands, and volume/momentum indicators via the
``ta`` library, then applies deterministic signal rules to produce
BUY / SELL / HOLD.

Data source routing:
    - Crypto tickers → Binance
    - German/EU tickers (.XETRA, .DE) → EODHD (fallback: yfinance)
    - All others → yfinance

Signal rules (any matching condition triggers the signal):
    BUY  — RSI < 30 (oversold)
           OR MACD line crosses above signal line (bullish crossover)
           OR latest close below lower Bollinger Band
    SELL — RSI > 70 (overbought)
           OR MACD line crosses below signal line (bearish crossover)
           OR latest close above upper Bollinger Band
    HOLD — none of the above conditions met

Volume confirmation (adjusts confidence, does not change signal):
    RVOL (Relative Volume) = current volume / 20-day average volume
    OBV trend              = rising or falling (last 5 bars)
    volume_confirmed       = True when RVOL > 1.5 and OBV supports direction

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
from config.settings import CRYPTO_TICKERS, is_german_ticker
from data.binance_feed import BinanceFeed
from data.eodhd_feed import EODHDFeed
from storage.database import Database

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        db: Database | None = None,
        binance_feed: BinanceFeed | None = None,
        eodhd_feed: EODHDFeed | None = None,
    ) -> None:
        self._db = db or Database()
        self._binance = binance_feed or BinanceFeed()
        self._eodhd = eodhd_feed or EODHDFeed()

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

        # -- 4. Volume confirmation -----------------------------------------
        volume_confirmed = self._check_volume_confirmation(signal, indicators)

        # -- 5. Intraday supplement (optional) ------------------------------
        intraday_note = self._intraday_supplement(ticker)
        if intraday_note:
            reasoning.append(intraday_note)

        # -- 6. Persist to database -----------------------------------------
        signal_id = self._db.log_technical_signal(
            ticker=ticker,
            signal=signal,
            reasoning="; ".join(reasoning) if reasoning else "No conditions triggered",
            rvol=indicators.get("rvol"),
            obv_trend=indicators.get("obv_trend"),
            volume_confirmed=volume_confirmed,
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
            "volume_confirmed": volume_confirmed,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_history(self, ticker: str) -> pd.DataFrame:
        """Download OHLCV data and validate the result.

        Crypto tickers are routed to Binance.  German/EU tickers are
        routed to EODHD with yfinance fallback.  All others use yfinance.
        """
        if ticker.upper() in CRYPTO_TICKERS:
            df = self._binance.get_ohlcv(ticker)
            if df is not None and not df.empty:
                return df
            raise ValueError(f"No Binance data returned for crypto ticker '{ticker}'")

        if is_german_ticker(ticker):
            return self._fetch_german_history(ticker)

        df = self._download_once(ticker)
        if df.empty:
            logger.warning(
                "yfinance returned empty data for %s — "
                "resetting session cache and retrying",
                ticker,
            )
            from data.market_data import _clear_yf_cache
            _clear_yf_cache()
            df = self._download_once(ticker)
            if df.empty:
                raise ValueError(f"No price data returned for ticker '{ticker}'")
        return df

    def _fetch_german_history(self, ticker: str) -> pd.DataFrame:
        """Fetch OHLCV for a German ticker via EODHD, falling back to yfinance."""
        if self._eodhd.available:
            df = self._eodhd.get_ohlcv_daily(ticker)
            if df is not None and not df.empty:
                logger.info("Using EODHD daily data for %s (%d bars)", ticker, len(df))
                return df
            logger.warning("EODHD returned no data for %s — falling back to yfinance", ticker)

        # Fallback: convert .XETRA → .DE for yfinance
        yf_ticker = ticker
        if ticker.upper().endswith(".XETRA"):
            yf_ticker = ticker.rsplit(".", 1)[0] + ".DE"

        df = self._download_once(yf_ticker)
        if df.empty:
            raise ValueError(f"No price data for German ticker '{ticker}' (tried EODHD + yfinance)")
        return df

    def _download_once(self, ticker: str) -> pd.DataFrame:
        """Single yfinance download attempt."""
        df: pd.DataFrame = yf.download(
            ticker,
            period=self._DOWNLOAD_PERIOD,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def _intraday_supplement(self, ticker: str) -> str | None:
        """Compute short-term RSI/MACD on 5min bars as supplementary signal."""
        if ticker in CRYPTO_TICKERS:
            return None

        if not (is_german_ticker(ticker) and self._eodhd.available):
            return None

        try:
            df = self._eodhd.get_ohlcv_intraday(ticker, interval="5m")
            if df is None or len(df) < 30:
                return None

            close = df["Close"].squeeze()

            # Short-term RSI-14 on 5min bars
            rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            rsi_clean = rsi_series.dropna()
            rsi_5m = float(rsi_clean.iloc[-1]) if not rsi_clean.empty else None

            # MACD on 5min bars
            macd_obj = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
            macd_val = macd_obj.macd().dropna()
            sig_val = macd_obj.macd_signal().dropna()

            macd_5m = float(macd_val.iloc[-1]) if not macd_val.empty else None
            sig_5m = float(sig_val.iloc[-1]) if not sig_val.empty else None

            parts = []
            if rsi_5m is not None:
                parts.append(f"5m-RSI={rsi_5m:.1f}")
            if macd_5m is not None and sig_5m is not None:
                direction = "above" if macd_5m > sig_5m else "below"
                parts.append(f"5m-MACD {direction} signal")

            if parts:
                return f"Intraday supplement: {', '.join(parts)}"

        except Exception as exc:
            logger.debug("Intraday supplement failed for %s: %s", ticker, exc)

        return None

    def _fallback_indicators(
        self, ticker: str, cache_key: str, error: str
    ) -> tuple[dict, bool]:
        """Return cached indicators or an all-None placeholder on cache miss."""
        from utils.network_recovery import get_cache
        cache = get_cache()
        cached, hit = cache.get("yfinance", cache_key)
        if hit and isinstance(cached, dict):
            logger.warning(
                "[DEGRADED] Using cached indicators for %s: %s", ticker, error,
            )
            return cached, True

        logger.warning(
            "[DEGRADED] No cached indicators for %s — returning None-filled: %s",
            ticker, error,
        )
        none_ind = {
            "rsi": None, "macd": None, "macd_signal": None, "macd_hist": None,
            "macd_bull_cross": False, "macd_bear_cross": False,
            "sma_20": None, "sma_50": None,
            "bb_upper": None, "bb_lower": None, "price": None,
            "rvol": None, "volume_trending_up": None, "obv_trend": None,
        }
        return none_ind, True

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

        # -- Volume & momentum indicators ---------------------------------
        volume: pd.Series = df["Volume"].squeeze() if "Volume" in df.columns else pd.Series(dtype=float)

        # RVOL: current volume / 20-day average volume
        rvol: "float | None" = None
        if not volume.empty and len(volume) >= 20:
            avg_vol_20 = float(volume.iloc[-20:].mean())
            if avg_vol_20 > 0:
                rvol = round(float(volume.iloc[-1]) / avg_vol_20, 2)

        # Volume trend: is volume increasing over last 5 bars?
        volume_trending_up: "bool | None" = None
        if not volume.empty and len(volume) >= 5:
            last5 = volume.iloc[-5:]
            volume_trending_up = bool(float(last5.iloc[-1]) > float(last5.iloc[0]))

        # OBV direction: rising or falling (compare last vs 5-bars-ago)
        obv_trend: "str | None" = None
        if not volume.empty and len(close) == len(volume) and len(close) >= 5:
            obv_series = ta.volume.OnBalanceVolumeIndicator(
                close=close, volume=volume
            ).on_balance_volume()
            obv_clean = obv_series.dropna()
            if len(obv_clean) >= 5:
                obv_trend = "rising" if float(obv_clean.iloc[-1]) > float(obv_clean.iloc[-5]) else "falling"

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
            "rvol":            rvol,
            "volume_trending_up": volume_trending_up,
            "obv_trend":       obv_trend,
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

    @staticmethod
    def _check_volume_confirmation(signal: str, ind: dict) -> bool:
        """
        Check whether volume supports the signal direction.

        BUY is confirmed when RVOL > 1.5 and OBV is rising.
        SELL is confirmed when RVOL > 1.5 and OBV is falling.
        HOLD is never volume-confirmed.

        Args:
            signal: "BUY", "SELL", or "HOLD".
            ind:    Indicator dict from ``_calculate_indicators``.

        Returns:
            True if volume confirms the signal direction.
        """
        rvol = ind.get("rvol")
        obv_trend = ind.get("obv_trend")
        if rvol is None or obv_trend is None or signal == "HOLD":
            return False
        if signal == "BUY":
            return rvol > 1.5 and obv_trend == "rising"
        if signal == "SELL":
            return rvol > 1.5 and obv_trend == "falling"
        return False
