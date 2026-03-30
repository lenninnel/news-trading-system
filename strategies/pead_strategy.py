"""
PEADStrategy — Post-Earnings Announcement Drift.

Pure data-driven strategy: no Claude API calls.  Checks the IBKR
earnings cache for recent earnings beats and generates BUY signals
with confidence scaled by surprise magnitude.

Validated parameters (walk-forward backtest, 104 tickers, DSR=1.00):
    Entry:    Day+2 after announcement
    Hold:     20 days
    Stop:     TIME_ONLY (no price stop)
    Sizing:   ATR-based (target 2% risk per trade)

Earnings data:
    Primary:  IBKR earnings cache (ibkr_earnings_cache.json)
    Fallback: yfinance earnings_dates
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from strategies.base import BaseStrategy, StrategyResult

log = logging.getLogger(__name__)

# Default cache path — overridden by config.settings.PEAD_EARNINGS_CACHE_PATH
_DEFAULT_CACHE = Path(__file__).resolve().parent.parent.parent / "walk-forward-backtest" / "data" / "ibkr_earnings_cache.json"


class PEADStrategy(BaseStrategy):
    """Post-Earnings Announcement Drift — earnings beat detection."""

    PREFERRED_TICKERS: list[str] = []  # set from config.PEAD_TICKERS

    def __init__(self, cache_path: str | Path | None = None) -> None:
        self._cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE
        self._cache: dict[str, list[dict]] | None = None

    @property
    def name(self) -> str:
        return "PEAD"

    # ------------------------------------------------------------------
    # Cache loading
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict[str, list[dict]]:
        """Load the IBKR earnings cache from disk (lazy, once)."""
        if self._cache is not None:
            return self._cache
        try:
            with open(self._cache_path) as f:
                self._cache = json.load(f)
            log.info("PEAD: loaded earnings cache (%d tickers) from %s",
                     len(self._cache), self._cache_path)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            log.warning("PEAD: earnings cache unavailable (%s), trying yfinance fallback", exc)
            self._cache = {}
        return self._cache

    # ------------------------------------------------------------------
    # Earnings lookup
    # ------------------------------------------------------------------

    def _recent_beat(
        self,
        ticker: str,
        current_date: date,
        lookback_days: int = 2,
    ) -> dict | None:
        """
        Check if *ticker* had an earnings beat within *lookback_days*.

        Returns the matching earnings record or ``None``.
        """
        cache = self._load_cache()
        records = cache.get(ticker, [])

        if not records:
            # Fallback: try yfinance
            return self._yfinance_fallback(ticker, current_date, lookback_days)

        for rec in records:
            # Use announce_date if available, else period_end
            date_str = rec.get("announce_date") or rec.get("period_end")
            if not date_str:
                continue
            try:
                earn_date = date.fromisoformat(date_str[:10])
            except (ValueError, TypeError):
                continue

            days_ago = (current_date - earn_date).days
            if 0 <= days_ago <= lookback_days:
                surprise = rec.get("surprise_pct")
                if surprise is not None and surprise > 5.0:
                    return rec

        return None

    @staticmethod
    def _yfinance_fallback(
        ticker: str,
        current_date: date,
        lookback_days: int,
    ) -> dict | None:
        """Try yfinance as a fallback for earnings data."""
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            eds = t.earnings_dates
            if eds is None or eds.empty:
                return None
            # Normalize index
            idx = eds.index.tz_localize(None) if eds.index.tz else eds.index
            cutoff = pd.Timestamp(current_date - timedelta(days=lookback_days))
            current_ts = pd.Timestamp(current_date)
            mask = (idx >= cutoff) & (idx <= current_ts)
            recent = eds[mask]
            if recent.empty:
                return None
            for dt, row in recent.iterrows():
                surprise = row.get("Surprise(%)")
                if surprise is not None and not pd.isna(surprise) and surprise > 5.0:
                    return {
                        "announce_date": str(dt.date()),
                        "surprise_pct": float(surprise),
                        "source": "yfinance",
                    }
        except Exception as exc:
            log.debug("PEAD yfinance fallback failed for %s: %s", ticker, exc)
        return None

    # ------------------------------------------------------------------
    # Confidence scaling
    # ------------------------------------------------------------------

    @staticmethod
    def _scale_confidence(surprise_pct: float) -> float:
        """
        Map surprise magnitude to confidence (0-100 scale).

        5-10%:  55.0
        10-20%: 65.0
        >20%:   75.0
        """
        if surprise_pct > 20.0:
            return 75.0
        if surprise_pct > 10.0:
            return 65.0
        return 55.0

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(self, ticker: str, current_date: date) -> StrategyResult | None:
        """
        Check for a recent earnings beat and return a BUY signal if found.

        Returns ``None`` if no qualifying beat occurred.
        """
        ticker = ticker.upper()
        beat = self._recent_beat(ticker, current_date)
        if beat is None:
            return None

        surprise = beat.get("surprise_pct", 0)
        confidence = self._scale_confidence(surprise)
        announce = beat.get("announce_date") or beat.get("period_end", "?")

        reasoning = [
            f"Earnings beat: {surprise:+.1f}% surprise on {announce}",
            "PEAD drift expected over next 20 trading days",
            f"Confidence {confidence:.0f}% based on surprise magnitude",
        ]
        log.info("[%s] PEAD signal: surprise=%+.1f%% conf=%.0f%%",
                 ticker, surprise, confidence)

        return StrategyResult(
            signal="BUY",
            confidence=confidence,
            indicators={
                "surprise_pct": surprise,
                "announce_date": announce,
                "hold_days": 20,
                "stop_mode": "TIME_ONLY",
            },
            strategy_name=self.name,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def analyze(
        self,
        ticker: str,
        bars: pd.DataFrame,
        sentiment_signal: str = "HOLD",
    ) -> StrategyResult:
        """
        Implement BaseStrategy.analyze().

        PEAD ignores bars and sentiment — it's purely earnings-driven.
        The bars parameter is accepted for interface compatibility but
        the signal is generated from the earnings cache.
        """
        today = date.today()
        result = self.generate_signal(ticker, today)
        if result is not None:
            return result

        return StrategyResult(
            signal="HOLD",
            confidence=0.0,
            strategy_name=self.name,
            reasoning=["No recent earnings beat (>5% surprise in last 2 days)"],
        )
