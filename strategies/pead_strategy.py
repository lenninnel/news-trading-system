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

Signal-attribution sidecar (Q-004 Fork B, 2026-05-27)
-----------------------------------------------------
``generate_signal`` writes ONE row per evaluation to ``pead_signal_log``
via an injected callable.  This captures the load-bearing inputs that
were previously dropped (``surprise_pct``, ``announce_date``,
``earnings_source``) so PEAD trades can be attributed after the fact.

The writer is observability only:

* Full-universe — every evaluation produces a row, including
  ``threshold_met=0`` near-misses and the "no earnings in window" case.
  Only ``threshold_met=1`` rows turn into BUY signals.
* Forward-only — the table is empty on deploy and fills from there;
  historical surprises cannot be reconstructed.
* Freeze-safe — the writer call is wrapped in try/except here so any
  failure is logged and swallowed.  Signal generation must never depend
  on this write succeeding.

On a BUY, the returned ``StrategyResult.indicators`` includes
``pead_log_id`` so execution can later stamp the trade id onto the row
(phase-2 write in coordinator).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from strategies.base import BaseStrategy, StrategyResult

log = logging.getLogger(__name__)

# Default cache path — overridden by config.settings.PEAD_EARNINGS_CACHE_PATH
_DEFAULT_CACHE = Path(__file__).resolve().parent.parent.parent / "walk-forward-backtest" / "data" / "ibkr_earnings_cache.json"

# Earnings-source enum.  These exact strings land in pead_signal_log.earnings_source
# so the dashboard / attribution queries can filter on them.
_SOURCE_IBKR_CACHE = "ibkr_cache"
_SOURCE_YFINANCE   = "yfinance_fallback"

# Beat threshold (matches validated PEAD parameters — see module docstring).
_BEAT_THRESHOLD_PCT = 5.0


class PEADStrategy(BaseStrategy):
    """Post-Earnings Announcement Drift — earnings beat detection."""

    PREFERRED_TICKERS: list[str] = []  # set from config.PEAD_TICKERS

    def __init__(
        self,
        cache_path: "str | Path | None" = None,
        signal_log_writer: "Callable[[dict], int | None] | None" = None,
    ) -> None:
        """
        Args:
            cache_path:        Optional override for the IBKR earnings
                               cache JSON.  Defaults to the walk-forward-
                               backtest data dir bundled with the repo.
            signal_log_writer: Optional callable that persists a
                               pead_signal_log row.  Receives a dict and
                               returns the new row id (or None).  When
                               omitted, generate_signal does NOT write
                               (matches the test default — the writer
                               only fires in production once Coordinator
                               injects the DB-backed wrapper).
        """
        self._cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE
        self._cache: "dict[str, list[dict]] | None" = None
        self._signal_log_writer = signal_log_writer

    @property
    def name(self) -> str:
        return "PEAD"

    # ------------------------------------------------------------------
    # Cache loading
    # ------------------------------------------------------------------

    def _load_cache(self) -> "dict[str, list[dict]]":
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
    # Earnings evaluation
    # ------------------------------------------------------------------
    #
    # _evaluate_earnings returns a dict with ALL the attribution fields
    # the sidecar needs — including the case where no earnings landed in
    # the lookback window.  Routing matches the legacy _recent_beat:
    # cache wins if the ticker has any records; yfinance is consulted
    # only when the cache has nothing for that ticker.

    _NO_EARNINGS_EVAL: dict = {
        "matched_record":  None,
        "surprise_pct":    None,
        "announce_date":   None,
        "earnings_source": None,
        "threshold_met":   0,
    }

    def _evaluate_earnings(
        self,
        ticker: str,
        current_date: date,
        lookback_days: int = 2,
    ) -> dict:
        """Return an evaluation dict regardless of whether a beat was found.

        Keys (always present):
            matched_record   — the underlying earnings record or None.
            surprise_pct     — float | None.  None when no record in window.
            announce_date    — str | None.    Announce date of the matched record.
            earnings_source  — "ibkr_cache" | "yfinance_fallback" | None.
                               None only when neither source was consulted
                               with a result (handled by the no-earnings path).
            threshold_met    — 1 if surprise_pct > 5.0, else 0.
        """
        cache = self._load_cache()
        records = cache.get(ticker, [])

        if records:
            # Cache wins — yfinance is NOT consulted for tickers that
            # already have IBKR cache entries.  Preserves legacy routing.
            return self._evaluate_from_cache(records, current_date, lookback_days)
        return self._evaluate_from_yfinance(ticker, current_date, lookback_days)

    @staticmethod
    def _evaluate_from_cache(
        records: list[dict],
        current_date: date,
        lookback_days: int,
    ) -> dict:
        """Find the earnings record (if any) in *records* whose announce
        date lies in ``[current_date - lookback_days, current_date]``.

        Returns the full evaluation dict.  When no record falls in the
        window, earnings_source is still ``ibkr_cache`` — we consulted
        it; the lack of a hit is itself a fact worth logging.
        """
        for rec in records:
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
                announce = rec.get("announce_date") or rec.get("period_end")
                if surprise is None:
                    return {
                        "matched_record":  rec,
                        "surprise_pct":    None,
                        "announce_date":   announce,
                        "earnings_source": _SOURCE_IBKR_CACHE,
                        "threshold_met":   0,
                    }
                surprise = float(surprise)
                return {
                    "matched_record":  rec,
                    "surprise_pct":    surprise,
                    "announce_date":   announce,
                    "earnings_source": _SOURCE_IBKR_CACHE,
                    "threshold_met":   1 if surprise > _BEAT_THRESHOLD_PCT else 0,
                }

        # Ticker is in the cache, but nothing in the lookback window.
        return {
            "matched_record":  None,
            "surprise_pct":    None,
            "announce_date":   None,
            "earnings_source": _SOURCE_IBKR_CACHE,
            "threshold_met":   0,
        }

    @staticmethod
    def _evaluate_from_yfinance(
        ticker: str,
        current_date: date,
        lookback_days: int,
    ) -> dict:
        """Try yfinance as the earnings data source.  Mirrors the legacy
        fallback semantics: any failure (network, schema drift, missing
        column) silently drops to the no-earnings result with
        ``earnings_source = yfinance_fallback`` so the row still records
        which source we consulted.
        """
        try:
            import yfinance as yf

            t = yf.Ticker(ticker)
            eds = t.earnings_dates
            if eds is None or eds.empty:
                return {
                    "matched_record":  None,
                    "surprise_pct":    None,
                    "announce_date":   None,
                    "earnings_source": _SOURCE_YFINANCE,
                    "threshold_met":   0,
                }
            idx = eds.index.tz_localize(None) if eds.index.tz else eds.index
            cutoff = pd.Timestamp(current_date - timedelta(days=lookback_days))
            current_ts = pd.Timestamp(current_date)
            mask = (idx >= cutoff) & (idx <= current_ts)
            recent = eds[mask]
            if recent.empty:
                return {
                    "matched_record":  None,
                    "surprise_pct":    None,
                    "announce_date":   None,
                    "earnings_source": _SOURCE_YFINANCE,
                    "threshold_met":   0,
                }
            # Take the latest match — earnings are at most one per ~90d
            # so a 2-day window typically contains zero or one, but
            # iloc[-1] is defensive in case the index isn't sorted.
            recent_sorted = recent.sort_index()
            latest_ts = recent_sorted.index[-1]
            row = recent_sorted.iloc[-1]
            surprise = row.get("Surprise(%)")
            announce_date = (
                str(latest_ts.date())
                if hasattr(latest_ts, "date") else None
            )
            if surprise is None or pd.isna(surprise):
                return {
                    "matched_record":  None,
                    "surprise_pct":    None,
                    "announce_date":   announce_date,
                    "earnings_source": _SOURCE_YFINANCE,
                    "threshold_met":   0,
                }
            surprise = float(surprise)
            return {
                "matched_record": {
                    "announce_date": announce_date,
                    "surprise_pct":  surprise,
                    "source":        _SOURCE_YFINANCE,
                },
                "surprise_pct":    surprise,
                "announce_date":   announce_date,
                "earnings_source": _SOURCE_YFINANCE,
                "threshold_met":   1 if surprise > _BEAT_THRESHOLD_PCT else 0,
            }
        except Exception as exc:
            log.debug("PEAD yfinance fallback failed for %s: %s", ticker, exc)
            return {
                "matched_record":  None,
                "surprise_pct":    None,
                "announce_date":   None,
                "earnings_source": _SOURCE_YFINANCE,
                "threshold_met":   0,
            }

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
    # Signal-log writer (never raises — observability only)
    # ------------------------------------------------------------------

    def _write_signal_log(
        self,
        *,
        ticker: str,
        session: "str | None",
        threshold_met: int,
        surprise_pct: "float | None",
        announce_date: "str | None",
        confidence: "float | None",
        earnings_source: "str | None",
    ) -> "int | None":
        """Persist one pead_signal_log row via the injected writer.

        Never raises — every failure mode (writer is None, callable
        raises, returns garbage) is logged and turned into ``None``.
        Returning None means callers cannot stamp a trade_id later, but
        the BUY signal itself is unaffected.
        """
        if self._signal_log_writer is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        row = {
            "timestamp":       now,
            "ticker":          ticker,
            "session":         session,
            "signal":          "BUY" if threshold_met else "HOLD",
            "threshold_met":   1 if threshold_met else 0,
            "surprise_pct":    surprise_pct,
            "announce_date":   announce_date,
            "confidence":      confidence,
            "hold_days":       20 if threshold_met else None,
            "stop_mode":       "TIME_ONLY" if threshold_met else None,
            "earnings_source": earnings_source,
            "trade_id":        None,
            "created_at":      now,
        }
        try:
            return self._signal_log_writer(row)
        except Exception as exc:
            log.warning(
                "[%s] PEAD signal log write failed (non-fatal): %s",
                ticker, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        ticker: str,
        current_date: date,
        session: "str | None" = None,
    ) -> "StrategyResult | None":
        """
        Evaluate earnings for *ticker* on *current_date*, write a phase-1
        log row regardless of outcome, and return a BUY ``StrategyResult``
        only when the surprise cleared the >5% threshold.

        Returns ``None`` when no BUY fires — preserves the legacy
        contract used by :meth:`Coordinator._run_pead` and the existing
        unit tests.  The phase-1 log row is still written for the
        no-fire case (full-universe logging).
        """
        ticker = ticker.upper()
        eval_data = self._evaluate_earnings(ticker, current_date)

        threshold_met   = eval_data["threshold_met"]
        surprise        = eval_data["surprise_pct"]
        announce        = eval_data["announce_date"]
        earnings_source = eval_data["earnings_source"]

        result: "StrategyResult | None" = None
        confidence: "float | None" = None
        if threshold_met:
            confidence = self._scale_confidence(surprise)
            reasoning = [
                f"Earnings beat: {surprise:+.1f}% surprise on {announce}",
                "PEAD drift expected over next 20 trading days",
                f"Confidence {confidence:.0f}% based on surprise magnitude",
            ]
            log.info("[%s] PEAD signal: surprise=%+.1f%% conf=%.0f%%",
                     ticker, surprise, confidence)

            result = StrategyResult(
                signal="BUY",
                confidence=confidence,
                indicators={
                    "surprise_pct":    surprise,
                    "announce_date":   announce,
                    "hold_days":       20,
                    "stop_mode":       "TIME_ONLY",
                    "earnings_source": earnings_source,
                },
                strategy_name=self.name,
                reasoning=reasoning,
            )

        # Phase-1 write — full universe, including threshold_met=0.
        # Never raises; failures are logged and swallowed inside
        # _write_signal_log so the BUY signal (if any) is unaffected.
        log_id = self._write_signal_log(
            ticker=ticker,
            session=session,
            threshold_met=threshold_met,
            surprise_pct=surprise,
            announce_date=announce,
            confidence=confidence,
            earnings_source=earnings_source,
        )

        # Stash the log id on the BUY result so the execution-time path
        # can stamp trade_id onto the same row (phase 2).  No effect on
        # threshold_met=0 rows — they have no result to attach to and
        # will keep trade_id NULL forever (correct).
        if result is not None and log_id is not None:
            result.indicators["pead_log_id"] = log_id

        return result

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
