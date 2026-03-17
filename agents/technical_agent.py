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

Signal rules (weighted scoring system — score >= +2 → BUY, <= -2 → SELL):
    RSI           — graduated: <30 (+2), <35 (+1.5), <45 (+0.5),
                     >70 (-2), >65 (-1.5), >55 (-0.5)
    MACD crossover — bullish (+2), bearish (-2)
    MACD histogram — positive (+1), negative (-1)
    Bollinger Band — below lower (+1.5), above upper (-1.5)
    SMA alignment  — price > SMA20 > SMA50 (+1), price < SMA20 < SMA50 (-1)
    Golden/Death X — golden cross (+1), death cross (-1)
    HOLD           — net score between -2 and +2

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

import numpy as np
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

    # Enough history for SMA-200 and indicator warm-up
    _DOWNLOAD_PERIOD = "1y"

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

        # -- 4. Confidence adjustments from advanced patterns ---------------
        adjusted_confidence, boost_reasons = self._apply_confidence_adjustments(
            signal, indicators,
        )
        reasoning.extend(boost_reasons)

        # -- 5. Volume confirmation -----------------------------------------
        volume_confirmed = self._check_volume_confirmation(signal, indicators)

        # -- 6. Intraday supplement (optional) ------------------------------
        intraday_note = self._intraday_supplement(ticker)
        if intraday_note:
            reasoning.append(intraday_note)

        # -- 7. Persist to database -----------------------------------------
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
            **{
                "sma_200": indicators.get("sma_200"),
                "ma200_distance_pct": indicators.get("ma200_distance_pct"),
                "golden_cross_recent": indicators.get("golden_cross_recent", False),
                "death_cross_recent": indicators.get("death_cross_recent", False),
                "adx": indicators.get("adx"),
                "trend_strength": indicators.get("trend_strength"),
                "bull_flag_detected": indicators.get("bull_flag_detected", False),
                "wedge_type": indicators.get("wedge_type"),
                "wedge_breakout": indicators.get("wedge_breakout", False),
                "nearest_support": indicators.get("nearest_support"),
                "nearest_resistance": indicators.get("nearest_resistance"),
            },
        )

        return {
            "ticker": ticker,
            "signal": signal,
            "reasoning": reasoning,
            "indicators": indicators,
            "signal_id": signal_id,
            "volume_confirmed": volume_confirmed,
            "adjusted_confidence": adjusted_confidence,
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
            "sma_200": None, "ma200_distance_pct": None,
            "golden_cross_recent": False, "death_cross_recent": False,
            "adx": None, "trend_strength": None,
            "bull_flag_detected": False, "bull_flag_breakout": False,
            "wedge_type": None, "wedge_breakout": False,
            "nearest_support": None, "nearest_resistance": None,
            "pct_to_support": None, "pct_to_resistance": None,
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

        # -- SMA 200 -------------------------------------------------------
        sma200 = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
        sma_200_val = latest(sma200)
        price_val = float(close.iloc[-1]) if not close.empty else None
        ma200_distance_pct: "float | None" = None
        if sma_200_val and price_val:
            ma200_distance_pct = round((price_val - sma_200_val) / sma_200_val * 100, 2)

        # -- Golden Cross / Death Cross (last 5 bars) ---------------------
        golden_cross_recent = False
        death_cross_recent = False
        sma50_series = sma50   # already computed
        sma200_series = sma200
        if len(sma50_series.dropna()) >= 5 and len(sma200_series.dropna()) >= 5:
            for i in range(-5, 0):
                try:
                    s50_cur = float(sma50_series.iloc[i])
                    s200_cur = float(sma200_series.iloc[i])
                    s50_prev = float(sma50_series.iloc[i - 1])
                    s200_prev = float(sma200_series.iloc[i - 1])
                    if s50_prev <= s200_prev and s50_cur > s200_cur:
                        golden_cross_recent = True
                    if s50_prev >= s200_prev and s50_cur < s200_cur:
                        death_cross_recent = True
                except (IndexError, ValueError):
                    pass

        # -- ADX (Average Directional Index) -------------------------------
        adx_val: "float | None" = None
        trend_strength: "str | None" = None
        try:
            high = df["High"].squeeze()
            low = df["Low"].squeeze()
            adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
            adx_series = adx_indicator.adx()
            adx_val = latest(adx_series)
            if adx_val is not None:
                if adx_val > 25:
                    trend_strength = "strong"
                elif adx_val < 20:
                    trend_strength = "weak"
                else:
                    trend_strength = "moderate"
        except Exception:
            pass

        # -- Pattern detection ---------------------------------------------
        bull_flag_detected, bull_flag_breakout = TechnicalAgent._detect_bull_flag(df, close)
        wedge_type, wedge_breakout = TechnicalAgent._detect_wedge(df, close)
        nearest_support, nearest_resistance, pct_to_support, pct_to_resistance = (
            TechnicalAgent._find_support_resistance(close, price_val)
        )

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
            "price":           price_val,
            "rvol":            rvol,
            "volume_trending_up": volume_trending_up,
            "obv_trend":       obv_trend,
            "sma_200":         sma_200_val,
            "ma200_distance_pct": ma200_distance_pct,
            "golden_cross_recent": golden_cross_recent,
            "death_cross_recent": death_cross_recent,
            "adx":             adx_val,
            "trend_strength":  trend_strength,
            "bull_flag_detected": bull_flag_detected,
            "bull_flag_breakout": bull_flag_breakout,
            "wedge_type":      wedge_type,
            "wedge_breakout":  wedge_breakout,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "pct_to_support":  pct_to_support,
            "pct_to_resistance": pct_to_resistance,
        }

    @staticmethod
    def _apply_signal_rules(ind: dict) -> tuple[str, list[str]]:
        """
        Evaluate BUY / SELL / HOLD using a weighted scoring system.

        Each indicator contributes a score: positive for bullish, negative for
        bearish.  The final signal is determined by the total score crossing a
        threshold.  This avoids the previous all-or-nothing approach where
        every single indicator had to be at an extreme before any signal fired.

        Scoring weights:
            RSI           — up to +/- 2 points (strong oversold/overbought)
            MACD crossover — 2 points (exact crossover bar)
            MACD histogram — 1 point  (sustained momentum direction)
            Bollinger Band — 1.5 points (price outside band)
            SMA trend      — 1 point  (price vs SMA-20/SMA-50 alignment)
            Golden/Death X — 1 point  (recent major cross)

        Thresholds:
            score >= +1.5  → BUY
            score <= -1.5  → SELL
            otherwise      → HOLD

        Args:
            ind: Indicator dict from ``_calculate_indicators``.

        Returns:
            Tuple of (signal string, list of triggered condition descriptions).
        """
        score = 0.0
        reasons: list[str] = []

        rsi        = ind.get("rsi")
        price      = ind.get("price")
        bb_low     = ind.get("bb_lower")
        bb_up      = ind.get("bb_upper")
        sma_20     = ind.get("sma_20")
        sma_50     = ind.get("sma_50")
        macd_hist  = ind.get("macd_hist")

        # --- RSI (relaxed thresholds with graduated scoring) ---
        if rsi is not None:
            if rsi < 30:
                score += 2.0
                reasons.append(f"RSI {rsi:.1f} < 30 (strongly oversold, +2)")
            elif rsi < 35:
                score += 1.5
                reasons.append(f"RSI {rsi:.1f} < 35 (oversold, +1.5)")
            elif rsi < 45:
                score += 0.5
                reasons.append(f"RSI {rsi:.1f} < 45 (mildly oversold, +0.5)")
            elif rsi > 70:
                score -= 2.0
                reasons.append(f"RSI {rsi:.1f} > 70 (strongly overbought, -2)")
            elif rsi > 65:
                score -= 1.5
                reasons.append(f"RSI {rsi:.1f} > 65 (overbought, -1.5)")
            elif rsi > 55:
                score -= 0.5
                reasons.append(f"RSI {rsi:.1f} > 55 (mildly overbought, -0.5)")

        # --- MACD crossover (exact bar — strong signal) ---
        if ind.get("macd_bull_cross"):
            score += 2.0
            reasons.append("MACD bullish crossover (+2)")

        if ind.get("macd_bear_cross"):
            score -= 2.0
            reasons.append("MACD bearish crossover (-2)")

        # --- MACD histogram momentum (sustained direction) ---
        if macd_hist is not None:
            if macd_hist > 0:
                score += 1.0
                reasons.append(f"MACD histogram {macd_hist:.3f} positive (bullish momentum, +1)")
            elif macd_hist < 0:
                score -= 1.0
                reasons.append(f"MACD histogram {macd_hist:.3f} negative (bearish momentum, -1)")

        # --- Bollinger Band breach ---
        if price is not None and bb_low is not None and price < bb_low:
            score += 1.5
            reasons.append(
                f"Price {price:.2f} below lower BB {bb_low:.2f} (+1.5)"
            )

        if price is not None and bb_up is not None and price > bb_up:
            score -= 1.5
            reasons.append(
                f"Price {price:.2f} above upper BB {bb_up:.2f} (-1.5)"
            )

        # --- SMA trend alignment ---
        if price is not None and sma_20 is not None and sma_50 is not None:
            if price > sma_20 > sma_50:
                score += 1.0
                reasons.append(
                    f"Bullish SMA alignment: price {price:.2f} > SMA20 {sma_20:.2f} > SMA50 {sma_50:.2f} (+1)"
                )
            elif price < sma_20 < sma_50:
                score -= 1.0
                reasons.append(
                    f"Bearish SMA alignment: price {price:.2f} < SMA20 {sma_20:.2f} < SMA50 {sma_50:.2f} (-1)"
                )

        # --- Golden / Death cross ---
        if ind.get("golden_cross_recent"):
            score += 1.0
            reasons.append("Recent golden cross (SMA50 crossed above SMA200, +1)")

        if ind.get("death_cross_recent"):
            score -= 1.0
            reasons.append("Recent death cross (SMA50 crossed below SMA200, -1)")

        # --- Determine final signal ---
        if score >= 1.5:
            return "BUY", reasons
        if score <= -1.5:
            return "SELL", reasons
        reasons.append(f"Net score {score:.1f} is between -1.5 and +1.5 — staying neutral")
        return "HOLD", reasons

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

    # ------------------------------------------------------------------
    # Pattern detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_bull_flag(
        df: pd.DataFrame, close: pd.Series
    ) -> tuple[bool, bool]:
        """Detect a bull-flag pattern and possible breakout.

        A bull flag consists of:
        1. A prior uptrend (price up >10% in last 20 bars).
        2. A tight consolidation in the last 5-10 bars (range < 5%).
        3. Declining volume during consolidation.
        4. Breakout: current bar closes above the consolidation high.

        Returns:
            (bull_flag_detected, bull_flag_breakout)
        """
        try:
            if len(close.dropna()) < 20:
                return False, False

            # Prior uptrend: price change over bars -20 to -10
            price_start = float(close.iloc[-20])
            price_mid = float(close.iloc[-10])
            if price_start <= 0:
                return False, False
            prior_gain_pct = (price_mid - price_start) / price_start * 100
            if prior_gain_pct < 10:
                return False, False

            # Consolidation: last 10 bars
            consolidation = close.iloc[-10:]
            cons_high = float(consolidation.max())
            cons_low = float(consolidation.min())
            if cons_low <= 0:
                return False, False
            cons_range_pct = (cons_high - cons_low) / cons_low * 100
            if cons_range_pct >= 5:
                return False, False

            # Volume declining during consolidation
            if "Volume" in df.columns:
                vol = df["Volume"].squeeze()
                if len(vol) >= 20:
                    vol_prior = float(vol.iloc[-20:-10].mean())
                    vol_cons = float(vol.iloc[-10:].mean())
                    if vol_prior > 0 and vol_cons >= vol_prior:
                        return False, False

            bull_flag_detected = True

            # Breakout: current close > consolidation high (excluding last bar)
            cons_high_excl = float(consolidation.iloc[:-1].max())
            bull_flag_breakout = float(close.iloc[-1]) > cons_high_excl

            return bull_flag_detected, bull_flag_breakout
        except Exception:
            return False, False

    @staticmethod
    def _detect_wedge(
        df: pd.DataFrame, close: pd.Series
    ) -> tuple["str | None", bool]:
        """Detect descending or ascending wedge via linear regression on highs/lows.

        Uses numpy polyfit on the last 20 bars of highs and lows.

        Returns:
            (wedge_type, wedge_breakout) where wedge_type is
            "descending", "ascending", or None.
        """
        try:
            n = 20
            if "High" not in df.columns or "Low" not in df.columns:
                return None, False
            high = df["High"].squeeze()
            low = df["Low"].squeeze()
            if len(high.dropna()) < n or len(low.dropna()) < n or len(close.dropna()) < n:
                return None, False

            highs = high.iloc[-n:].values.astype(float)
            lows = low.iloc[-n:].values.astype(float)
            closes = close.iloc[-n:].values.astype(float)
            x = np.arange(n)

            slope_high, intercept_high = np.polyfit(x, highs, 1)
            slope_low, intercept_low = np.polyfit(x, lows, 1)

            # Descending wedge: both slopes negative, converging (slope_high > slope_low)
            if slope_high < 0 and slope_low < 0 and slope_high > slope_low:
                projected_high = intercept_high + slope_high * (n - 1)
                breakout = closes[-1] > projected_high
                return "descending", breakout

            # Ascending wedge: both slopes positive, converging (slope_low > slope_high)
            if slope_high > 0 and slope_low > 0 and slope_low > slope_high:
                projected_low = intercept_low + slope_low * (n - 1)
                breakout = closes[-1] < projected_low
                return "ascending", breakout

            return None, False
        except Exception:
            return None, False

    @staticmethod
    def _find_support_resistance(
        close: pd.Series, price: "float | None"
    ) -> tuple["float | None", "float | None", "float | None", "float | None"]:
        """Find nearest support and resistance from swing highs/lows.

        Examines the last 60 bars for local minima (swing lows) and local
        maxima (swing highs), then identifies the nearest levels relative
        to the current price.

        Returns:
            (nearest_support, nearest_resistance, pct_to_support, pct_to_resistance)
        """
        try:
            n = 60
            if price is None or len(close.dropna()) < 5:
                return None, None, None, None

            data = close.iloc[-n:].values.astype(float)
            if len(data) < 3:
                return None, None, None, None

            swing_lows: list[float] = []
            swing_highs: list[float] = []

            for i in range(1, len(data) - 1):
                if data[i] < data[i - 1] and data[i] < data[i + 1]:
                    swing_lows.append(data[i])
                if data[i] > data[i - 1] and data[i] > data[i + 1]:
                    swing_highs.append(data[i])

            # Nearest support: highest swing low below current price
            supports_below = [s for s in swing_lows if s < price]
            nearest_support = max(supports_below) if supports_below else None

            # Nearest resistance: lowest swing high above current price
            resistances_above = [r for r in swing_highs if r > price]
            nearest_resistance = min(resistances_above) if resistances_above else None

            pct_to_support: "float | None" = None
            pct_to_resistance: "float | None" = None
            if nearest_support is not None and price > 0:
                pct_to_support = round((price - nearest_support) / price * 100, 2)
            if nearest_resistance is not None and price > 0:
                pct_to_resistance = round((nearest_resistance - price) / price * 100, 2)

            return nearest_support, nearest_resistance, pct_to_support, pct_to_resistance
        except Exception:
            return None, None, None, None

    @staticmethod
    def _apply_confidence_adjustments(
        signal: str, ind: dict
    ) -> tuple[float, list[str]]:
        """Compute a confidence score with boosts/reductions from advanced patterns.

        Returns:
            (adjusted_confidence, boost_reasons) where confidence is 0.0 - 0.95.
        """
        # Base confidence
        if signal in ("BUY", "SELL"):
            confidence = 0.5
        else:
            confidence = 0.25

        boost_reasons: list[str] = []

        # Golden cross + strong trend + price above SMA200 → boost
        golden = ind.get("golden_cross_recent", False)
        adx = ind.get("adx")
        sma_200 = ind.get("sma_200")
        price = ind.get("price")
        trend_str = ind.get("trend_strength")

        if golden and adx is not None and adx > 25 and price is not None and sma_200 is not None and price > sma_200:
            confidence += 0.15
            boost_reasons.append(
                f"Golden cross with strong ADX ({adx:.1f}) and price above SMA200 (+0.15)"
            )

        # Bull flag breakout → boost
        if ind.get("bull_flag_breakout", False):
            confidence += 0.10
            boost_reasons.append("Bull flag breakout detected (+0.10)")

        # Wedge breakout in correct direction → boost
        wedge_type = ind.get("wedge_type")
        wedge_breakout = ind.get("wedge_breakout", False)
        if wedge_breakout and wedge_type:
            if (wedge_type == "descending" and signal == "BUY") or (
                wedge_type == "ascending" and signal == "SELL"
            ):
                confidence += 0.10
                boost_reasons.append(
                    f"{wedge_type.capitalize()} wedge breakout confirms {signal} (+0.10)"
                )

        # Death cross → reduce
        if ind.get("death_cross_recent", False):
            confidence -= 0.15
            boost_reasons.append("Death cross recent (-0.15)")

        # Price below SMA200 with strong downtrend → reduce
        if (
            price is not None
            and sma_200 is not None
            and price < sma_200
            and trend_str == "strong"
        ):
            confidence -= 0.10
            boost_reasons.append(
                "Price below SMA200 with strong downtrend (-0.10)"
            )

        # Clamp
        confidence = max(0.0, min(0.95, confidence))
        confidence = round(confidence, 2)

        return confidence, boost_reasons
