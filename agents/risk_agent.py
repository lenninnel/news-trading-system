"""
Risk management agent — position sizing and stop-loss calculation.

RiskAgent takes a combined trading signal and account parameters, then
computes a safe position size using a half-Kelly Criterion adjusted by
confidence, with hard caps and a 2 % per-trade risk budget.

Kelly Criterion
---------------
Win probability is estimated as:
    p = 0.5 + (confidence / 100) * 0.30   →  range [0.50, 0.80]

This maps a 0 % confident signal to a coin flip and a 100 % confident
signal to a 80 % win probability.  The reward-to-risk ratio ``b`` is
fixed at 2.0 (matching the 2:1 take-profit target).

    kelly*  = (p * b - q) / b     where q = 1 - p
    kelly   = kelly* / 2          (half-Kelly — conservative)

Position size constraints (applied in order, tightest wins):
    1. Kelly fraction of account balance
    2. 10 % portfolio cap per trade
    3. 2 % max risk budget  →  position ≤ (0.02 * balance) / stop_pct

Stop-loss / take-profit
-----------------------
    Signal strength   stop_pct   take-profit multiplier (2:1 RR)
    ──────────────    ────────   ──────────────────────────────
    STRONG            2.00 %     4.00 %
    WEAK              1.00 %     2.00 %

Safety guards
-------------
- confidence < 30  → no position (signal too uncertain)
- HOLD / CONFLICTING → no position
- shares == 0 (price > position budget) → no position
- Transaction cost of 0.10 % deducted from position before share calculation

Earnings event risk
-------------------
- earnings_week (≤5 trading days): position capped at 50 % of Kelly result
- earnings_imminent (≤2 trading days): position capped at 25 % of Kelly result;
  if confidence < 50, position is skipped entirely

Market regime adjustment (applied before earnings cap)
------------------------------------------------------
- TRENDING_BULL:  1.0× (no change)
- TRENDING_BEAR:  0.7× position reduction
- RANGING:        0.8× position reduction; WEAK signals are skipped entirely
- HIGH_VOL:       0.5× position reduction; only STRONG signals with conf > 70%

Requires:
    No additional dependencies (uses only stdlib + storage layer)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import yfinance as yf

from agents.base_agent import BaseAgent
from config.settings import (
    ACCOUNT_RISK_PCT,
    ATR_STOP_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    DRAWDOWN_HALT_THRESHOLD,
    USE_ATR_STOPS,
)
from data.events_feed import get_days_to_earnings
from storage.database import Database

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
_MAX_PORTFOLIO_FRACTION = 0.10   # 10 % portfolio cap
_MAX_RISK_FRACTION      = 0.02   # 2 % account risk per trade
_TRANSACTION_COST       = 0.001  # 0.1 % per trade
_REWARD_RISK_RATIO      = 2.0    # 2:1 take-profit
_MIN_CONFIDENCE         = 30.0   # below this → no position
_WIN_PROB_BASE          = 0.50   # base win probability (random)
_WIN_PROB_RANGE         = 0.30   # additional range driven by confidence

# Historical Kelly constants
_HISTORICAL_KELLY_MIN_TRADES = 10  # need ≥10 outcomes for historical Kelly
_HISTORICAL_KELLY_CAP        = 0.05  # 5 % hard cap per position
_HISTORICAL_KELLY_FALLBACK   = 0.02  # fixed 2 % when insufficient data

# Portfolio VaR z-scores (avoids scipy dependency)
_Z_SCORES: dict[float, float] = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}

# Regime-based position multipliers
_REGIME_MULTIPLIER: dict[str, float] = {
    "TRENDING_BULL": 1.0,
    "TRENDING_BEAR": 0.7,
    "RANGING":       0.8,
    "HIGH_VOL":      0.5,
}

# Stop-loss percentages keyed by signal strength
_STOP_PCT: dict[str, float] = {
    "STRONG": 0.02,
    "WEAK":   0.01,
}


def _parse_signal(signal: str) -> tuple[str, str]:
    """
    Extract direction and strength from a signal string.

    Args:
        signal: One of STRONG BUY, STRONG SELL, WEAK BUY, WEAK SELL,
                BUY, SELL, HOLD, CONFLICTING.

    Returns:
        Tuple of (direction, strength):
            direction — "BUY" | "SELL" | "HOLD"
            strength  — "STRONG" | "WEAK"
    """
    s = signal.upper().strip()
    if s == "STRONG BUY":   return "BUY",  "STRONG"
    if s == "WEAK BUY":     return "BUY",  "WEAK"
    if s == "BUY":          return "BUY",  "STRONG"   # raw BUY → STRONG
    if s == "STRONG SELL":  return "SELL", "STRONG"
    if s == "WEAK SELL":    return "SELL", "WEAK"
    if s == "SELL":         return "SELL", "STRONG"   # raw SELL → STRONG
    return "HOLD", "NONE"


def _kelly_fraction(confidence: float) -> float:
    """
    Compute the half-Kelly fraction for a given confidence score.

    Args:
        confidence: 0–100 signal confidence.

    Returns:
        Half-Kelly fraction in [0.0, 1.0].
    """
    p = _WIN_PROB_BASE + (confidence / 100.0) * _WIN_PROB_RANGE
    q = 1.0 - p
    b = _REWARD_RISK_RATIO
    kelly_star = (p * b - q) / b
    return max(0.0, kelly_star / 2.0)   # half-Kelly, floor at 0


class RiskAgent(BaseAgent):
    """
    Calculates position sizing and stop-loss levels for a trading signal.

    Attributes:
        _db: Database instance used to persist results.

    Example::

        agent = RiskAgent()
        result = agent.run(
            ticker="AAPL",
            signal="STRONG BUY",
            confidence=75,
            current_price=263.88,
            account_balance=10_000,
        )
        # {
        #   "position_size_usd": 792.0,
        #   "shares": 3,
        #   "stop_loss": 258.60,
        #   "take_profit": 279.15,
        #   "risk_amount": 15.84,
        #   "kelly_fraction": 0.10,
        #   "stop_pct": 0.02,
        #   "skipped": False,
        #   "skip_reason": None,
        #   "calc_id": 1,
        # }
    """

    def __init__(self, db: Database | None = None) -> None:
        """
        Initialise the agent.

        Args:
            db: Optional Database instance for dependency injection.
        """
        self._db = db or Database()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "RiskAgent"

    def run(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        current_price: float,
        account_balance: float,
        **kwargs: Any,
    ) -> dict:
        """
        Calculate position sizing and risk parameters for a trade.

        Args:
            ticker:          Stock ticker symbol.
            signal:          Combined signal string (e.g. "STRONG BUY").
            confidence:      Signal confidence 0–100.
            current_price:   Latest price per share.
            account_balance: Total account value in USD.

        Returns:
            dict with keys:
                position_size_usd (float): Dollar value to deploy.
                shares            (int):   Whole shares to buy/sell.
                stop_loss         (float|None): Stop-loss price.
                take_profit       (float|None): Take-profit price.
                risk_amount       (float): Max dollar loss if stop hit.
                kelly_fraction    (float): Raw Kelly fraction used.
                stop_pct          (float|None): Stop-loss percentage used.
                skipped           (bool):  True when no position is taken.
                skip_reason       (str|None): Explanation when skipped.
                calc_id           (int):  Database primary key.
        """
        ticker = ticker.upper()
        regime = kwargs.get("regime")

        # -- 1. Safety gates -----------------------------------------------
        skip_reason: str | None = None

        direction, strength = _parse_signal(signal)

        if confidence < _MIN_CONFIDENCE:
            skip_reason = f"Confidence {confidence:.0f}% is below minimum {_MIN_CONFIDENCE:.0f}%"
        elif direction == "HOLD":
            skip_reason = f"Signal '{signal}' carries no directional conviction"

        # -- 1c. Regime-based signal gates ---------------------------------
        if not skip_reason and regime:
            if regime == "RANGING" and strength == "WEAK":
                skip_reason = f"WEAK signal skipped in RANGING regime"
            elif regime == "HIGH_VOL":
                if strength != "STRONG":
                    skip_reason = f"Only STRONG signals allowed in HIGH_VOL regime"
                elif confidence <= 70:
                    skip_reason = (
                        f"HIGH_VOL regime requires confidence > 70% "
                        f"(got {confidence:.0f}%)"
                    )

        # -- 1b. Earnings event risk check ---------------------------------
        days_to_earn = get_days_to_earnings(ticker)
        if days_to_earn is not None and days_to_earn <= 2:
            event_risk_flag = "earnings_imminent"
        elif days_to_earn is not None and days_to_earn <= 5:
            event_risk_flag = "earnings_week"
        else:
            event_risk_flag = "none"

        # Earnings-imminent + low confidence → skip
        if event_risk_flag == "earnings_imminent" and confidence < 50 and not skip_reason:
            skip_reason = (
                f"Earnings imminent ({days_to_earn} trading days) "
                f"with low confidence ({confidence:.0f}%)"
            )

        if skip_reason:
            result = self._no_position(ticker, signal, confidence, current_price,
                                       account_balance, skip_reason)
            result["event_risk_flag"] = event_risk_flag
            result["days_to_earnings"] = days_to_earn
            result["regime"] = regime
            return result

        # -- 2. ATR-based or fixed stops ------------------------------------
        atr_result = None
        if USE_ATR_STOPS:
            atr_result = self.calculate_atr_stops(
                ticker, current_price, direction=direction,
                account_balance=account_balance,
            )

        if atr_result and atr_result.get("atr_available") and atr_result.get("shares", 0) > 0:
            # ATR-based sizing and stops
            stop_loss = atr_result["stop_loss"]
            take_profit = atr_result["take_profit"]
            shares = atr_result["shares"]
            stop_pct = atr_result["stop_distance"] / current_price if current_price > 0 else 0.02
            kelly = atr_result.get("position_size_pct", 0.0)

            # Cap at portfolio max (10%)
            max_shares_by_portfolio = int((account_balance * _MAX_PORTFOLIO_FRACTION) / current_price)
            shares = min(shares, max_shares_by_portfolio)

            # Apply regime and earnings adjustments to ATR-based position
            multiplier = 1.0
            if regime and regime in _REGIME_MULTIPLIER:
                multiplier *= _REGIME_MULTIPLIER[regime]
            if event_risk_flag == "earnings_imminent":
                multiplier *= 0.25
            elif event_risk_flag == "earnings_week":
                multiplier *= 0.50
            if multiplier < 1.0:
                shares = max(0, int(shares * multiplier))

            position_size_usd = shares * current_price
            risk_amount = atr_result["stop_distance"] * shares
        else:
            # Fixed percentage stops (fallback)
            stop_pct = _STOP_PCT[strength]

            # -- 3. Kelly Criterion sizing ----------------------------------
            historical_kelly = self.calculate_kelly_position(
                ticker, signal, confidence, account_balance,
            )
            kelly = historical_kelly if historical_kelly is not None else _kelly_fraction(confidence)

            max_by_kelly     = account_balance * kelly
            max_by_portfolio = account_balance * _MAX_PORTFOLIO_FRACTION
            max_by_risk      = (account_balance * _MAX_RISK_FRACTION) / stop_pct

            position_raw = min(max_by_kelly, max_by_portfolio, max_by_risk)

            # Regime adjustment
            if regime and regime in _REGIME_MULTIPLIER:
                position_raw *= _REGIME_MULTIPLIER[regime]

            # Earnings cap
            if event_risk_flag == "earnings_imminent":
                position_raw *= 0.25
            elif event_risk_flag == "earnings_week":
                position_raw *= 0.50

            position_after_cost = position_raw * (1.0 - _TRANSACTION_COST)
            shares = int(position_after_cost / current_price)

            if shares > 0:
                position_size_usd = shares * current_price
                risk_amount = position_size_usd * stop_pct
                if direction == "BUY":
                    stop_loss   = round(current_price * (1.0 - stop_pct), 4)
                    take_profit = round(current_price * (1.0 + stop_pct * _REWARD_RISK_RATIO), 4)
                else:
                    stop_loss   = round(current_price * (1.0 + stop_pct), 4)
                    take_profit = round(current_price * (1.0 - stop_pct * _REWARD_RISK_RATIO), 4)

        if shares == 0:
            skip_reason = (
                f"Price ${current_price:.2f} exceeds affordable position"
            )
            result = self._no_position(ticker, signal, confidence, current_price,
                                       account_balance, skip_reason)
            result["event_risk_flag"] = event_risk_flag
            result["days_to_earnings"] = days_to_earn
            result["regime"] = regime
            return result

        result = {
            "ticker":           ticker,
            "signal":           signal,
            "direction":        direction,
            "position_size_usd": round(position_size_usd, 2),
            "shares":           shares,
            "stop_loss":        stop_loss,
            "take_profit":      take_profit,
            "risk_amount":      round(risk_amount, 2),
            "kelly_fraction":   round(kelly, 4),
            "stop_pct":         stop_pct,
            "skipped":          False,
            "skip_reason":      None,
            "event_risk_flag":  event_risk_flag,
            "days_to_earnings": days_to_earn,
            "regime":           regime,
        }

        # -- 7. Persist ----------------------------------------------------
        result["calc_id"] = self._db.log_risk_calculation(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            current_price=current_price,
            account_balance=account_balance,
            **{k: result[k] for k in (
                "position_size_usd", "shares", "stop_loss", "take_profit",
                "risk_amount", "kelly_fraction", "stop_pct",
            )},
            skipped=False,
            skip_reason=None,
            event_risk_flag=event_risk_flag,
            days_to_earnings=days_to_earn,
            regime=regime,
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _no_position(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        current_price: float,
        account_balance: float,
        skip_reason: str,
        event_risk_flag: str = "none",
        days_to_earnings: "int | None" = None,
        regime: "str | None" = None,
    ) -> dict:
        """Persist and return a zero-position result."""
        calc_id = self._db.log_risk_calculation(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            current_price=current_price,
            account_balance=account_balance,
            position_size_usd=0.0,
            shares=0,
            stop_loss=None,
            take_profit=None,
            risk_amount=0.0,
            kelly_fraction=0.0,
            stop_pct=None,
            skipped=True,
            skip_reason=skip_reason,
            event_risk_flag=event_risk_flag,
            days_to_earnings=days_to_earnings,
            regime=regime,
        )
        return {
            "ticker":           ticker,
            "signal":           signal,
            "direction":        "HOLD",
            "position_size_usd": 0.0,
            "shares":           0,
            "stop_loss":        None,
            "take_profit":      None,
            "risk_amount":      0.0,
            "kelly_fraction":   0.0,
            "stop_pct":         None,
            "skipped":          True,
            "skip_reason":      skip_reason,
            "calc_id":          calc_id,
            "event_risk_flag":  event_risk_flag,
            "days_to_earnings": days_to_earnings,
            "regime":           regime,
        }

    # ------------------------------------------------------------------
    # ATR-based dynamic stops
    # ------------------------------------------------------------------

    def calculate_atr_stops(
        self,
        ticker: str,
        entry_price: float,
        direction: str = "BUY",
        atr_stop_multiplier: float = ATR_STOP_MULTIPLIER,
        atr_tp_multiplier: float = ATR_TP_MULTIPLIER,
        account_risk_pct: float = ACCOUNT_RISK_PCT,
        account_balance: float = 10_000.0,
        prices: "pd.Series | None" = None,
    ) -> dict:
        """
        ATR-based stop-loss and take-profit calculation.

        Args:
            ticker:              Stock ticker symbol.
            entry_price:         Entry price per share.
            direction:           "BUY" or "SELL".
            atr_stop_multiplier: ATR multiplier for stop distance (default 1.5).
            atr_tp_multiplier:   ATR multiplier for take-profit (default 3.0).
            account_risk_pct:    Fraction of account risked per trade (default 0.01).
            account_balance:     Account value in USD.
            prices:              Optional pre-fetched close prices (avoids API call).

        Returns:
            dict with stop_loss, take_profit, atr, stop_distance, rr_ratio,
            position_size_pct, shares.  Returns None values on failure.
        """
        import pandas as pd

        atr = self._fetch_atr(ticker, prices)
        if atr is None or atr <= 0:
            log.debug("[%s] ATR unavailable — falling back to fixed stops", ticker)
            return {"atr": None, "atr_available": False}

        stop_distance = atr * atr_stop_multiplier
        tp_distance = atr * atr_tp_multiplier

        # Enforce minimum 2:1 reward/risk
        if tp_distance < 2.0 * stop_distance:
            tp_distance = 2.0 * stop_distance

        rr_ratio = tp_distance / stop_distance if stop_distance > 0 else 0.0

        if direction == "BUY":
            stop_loss = round(entry_price - stop_distance, 4)
            take_profit = round(entry_price + tp_distance, 4)
        else:
            stop_loss = round(entry_price + stop_distance, 4)
            take_profit = round(entry_price - tp_distance, 4)

        # Position sizing: risk budget / stop distance
        risk_budget = account_balance * account_risk_pct
        shares = int(risk_budget / stop_distance) if stop_distance > 0 else 0
        position_size_usd = shares * entry_price
        position_size_pct = position_size_usd / account_balance if account_balance > 0 else 0

        log.info(
            "[%s] ATR stop: $%.2f (%.1f× ATR $%.2f), TP: $%.2f (%.1f× ATR), R:R %.1f:1",
            ticker, stop_loss, atr_stop_multiplier, atr,
            take_profit, atr_tp_multiplier, rr_ratio,
        )

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": round(atr, 4),
            "stop_distance": round(stop_distance, 4),
            "tp_distance": round(tp_distance, 4),
            "rr_ratio": round(rr_ratio, 2),
            "position_size_pct": round(position_size_pct, 4),
            "shares": shares,
            "position_size_usd": round(position_size_usd, 2),
            "atr_available": True,
        }

    def _fetch_atr(
        self, ticker: str, prices: "pd.Series | None" = None,
    ) -> float | None:
        """Compute ATR(14) from price data. Fetches via yfinance if needed."""
        import pandas as pd

        try:
            if prices is not None and len(prices) >= 15:
                close = prices
            else:
                yf_ticker = self._yf_ticker(ticker)
                data = yf.download(yf_ticker, period="1mo", interval="1d", progress=False)
                if data.empty or len(data) < 15:
                    return None
                close = data["Close"].squeeze()

            # ATR approximation using close-to-close (no High/Low needed)
            daily_range = close.pct_change().abs().dropna()
            if len(daily_range) < 14:
                return None
            atr_pct = float(daily_range.rolling(14).mean().iloc[-1])
            return atr_pct * float(close.iloc[-1])
        except Exception as exc:
            log.debug("[%s] ATR fetch failed: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Historical Kelly Criterion
    # ------------------------------------------------------------------

    def calculate_kelly_position(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        account_balance: float,
    ) -> float | None:
        """
        Kelly criterion using historical signal outcomes.

        ``f* = (b·p − q) / b`` where:
        - *p* = win rate from signal_events history
        - *q* = 1 − p
        - *b* = avg win / avg loss ratio (from outcome_5d_pct)

        Falls back to ``None`` when fewer than 10 outcomes exist
        (caller should use fixed 2 %).  Uses half-Kelly (0.5×) and
        caps at 5 % regardless of Kelly output.
        """
        try:
            rows = self._query_outcomes(ticker)
            if len(rows) < _HISTORICAL_KELLY_MIN_TRADES:
                # Try portfolio-wide
                rows = self._query_outcomes(None)
            if len(rows) < _HISTORICAL_KELLY_MIN_TRADES:
                return None

            wins = [r for r in rows if r["outcome_correct"] == 1]
            losses = [r for r in rows if r["outcome_correct"] == 0]
            p = len(wins) / len(rows)
            q = 1.0 - p

            avg_win = (
                sum(abs(r["outcome_5d_pct"]) for r in wins) / len(wins)
                if wins else 0.0
            )
            avg_loss = (
                sum(abs(r["outcome_5d_pct"]) for r in losses) / len(losses)
                if losses else 0.0
            )

            b = avg_win / avg_loss if avg_loss > 0 else 10.0
            kelly_star = (b * p - q) / b
            kelly = max(0.0, kelly_star * 0.5)          # half-Kelly
            kelly = min(kelly, _HISTORICAL_KELLY_CAP)    # 5 % cap
            return kelly

        except Exception as exc:
            log.debug("Historical Kelly failed (non-fatal): %s", exc)
            return None

    def _query_outcomes(self, ticker: str | None) -> list[dict]:
        """Read signal_events rows that have outcome data."""
        try:
            with self._db._connect() as conn:
                if ticker:
                    rows = conn.execute(
                        "SELECT outcome_correct, outcome_5d_pct "
                        "FROM signal_events "
                        "WHERE ticker = ? AND outcome_correct IS NOT NULL "
                        "AND outcome_5d_pct IS NOT NULL",
                        (ticker.upper(),),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT outcome_correct, outcome_5d_pct "
                        "FROM signal_events "
                        "WHERE outcome_correct IS NOT NULL "
                        "AND outcome_5d_pct IS NOT NULL",
                    ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Portfolio VaR
    # ------------------------------------------------------------------

    def calculate_portfolio_var(
        self,
        positions: list[dict],
        confidence_level: float = 0.95,
        lookback_days: int = 252,
    ) -> dict:
        """
        Parametric Value-at-Risk for the current portfolio.

        Args:
            positions:        List of dicts with ``ticker`` and ``current_value``.
            confidence_level: Confidence level (0.90, 0.95, or 0.99).
            lookback_days:    Trading days of history for covariance estimation.

        Returns:
            dict with var_1day, var_1day_pct, var_5day, should_halt, reasoning.
        """
        if not positions:
            return {
                "var_1day": 0.0, "var_1day_pct": 0.0, "var_5day": 0.0,
                "should_halt": False, "reasoning": "No positions.",
            }

        total_value = sum(p.get("current_value", 0) for p in positions)
        if total_value <= 0:
            return {
                "var_1day": 0.0, "var_1day_pct": 0.0, "var_5day": 0.0,
                "should_halt": False, "reasoning": "Portfolio value is zero.",
            }

        # Convert XETRA tickers for yfinance
        raw_tickers = [p["ticker"] for p in positions]
        yf_tickers = [self._yf_ticker(t) for t in raw_tickers]
        weights = np.array([p.get("current_value", 0) / total_value for p in positions])

        # Fetch historical prices
        try:
            period = "1y" if lookback_days <= 252 else f"{lookback_days}d"
            data = yf.download(
                yf_tickers, period=period, interval="1d",
                progress=False, auto_adjust=True,
            )
            if data.empty:
                return {
                    "var_1day": 0.0, "var_1day_pct": 0.0, "var_5day": 0.0,
                    "should_halt": True,
                    "reasoning": "No price data available — halting conservatively.",
                }

            # Extract Close prices
            if len(yf_tickers) == 1:
                closes = data[["Close"]].rename(columns={"Close": yf_tickers[0]})
            else:
                closes = data["Close"]

            # Daily log returns
            returns = np.log(closes / closes.shift(1)).dropna()
            if returns.empty or len(returns) < 20:
                return {
                    "var_1day": 0.0, "var_1day_pct": 0.0, "var_5day": 0.0,
                    "should_halt": True,
                    "reasoning": "Insufficient return data — halting conservatively.",
                }

            # Covariance matrix
            cov_matrix = returns.cov().values

            # Portfolio variance and std
            port_variance = float(weights @ cov_matrix @ weights)
            port_daily_std = math.sqrt(port_variance)

            # z-score
            z = _Z_SCORES.get(confidence_level, 1.645)

            var_1day = z * port_daily_std * total_value
            var_1day_pct = z * port_daily_std
            var_5day = var_1day * math.sqrt(5)
            should_halt = var_1day_pct > 0.03

            reasoning = (
                f"1-day VaR at {confidence_level:.0%}: "
                f"${var_1day:,.2f} ({var_1day_pct:.2%} of portfolio). "
            )
            if should_halt:
                reasoning += "HALT: exceeds 3% threshold."
            else:
                reasoning += "Within acceptable limits."

            return {
                "var_1day": round(var_1day, 2),
                "var_1day_pct": round(var_1day_pct, 4),
                "var_5day": round(var_5day, 2),
                "should_halt": should_halt,
                "reasoning": reasoning,
            }

        except Exception as exc:
            log.warning("Portfolio VaR calculation failed: %s", exc)
            return {
                "var_1day": 0.0, "var_1day_pct": 0.0, "var_5day": 0.0,
                "should_halt": True,
                "reasoning": f"VaR calculation failed ({exc}) — halting conservatively.",
            }

    @staticmethod
    def _yf_ticker(ticker: str) -> str:
        """Convert internal ticker format to yfinance format."""
        if ticker.endswith(".XETRA"):
            return ticker.replace(".XETRA", ".DE")
        return ticker

    # ------------------------------------------------------------------
    # Drawdown halt
    # ------------------------------------------------------------------

    def check_drawdown_halt(
        self,
        current_portfolio_value: float,
        peak_portfolio_value: float,
    ) -> bool:
        """
        Return True if the portfolio has drawn down beyond the halt threshold.

        The threshold defaults to 10 % (configurable via DRAWDOWN_HALT_THRESHOLD
        in config/settings.py or the environment).
        """
        if peak_portfolio_value <= 0:
            return False
        drawdown = (peak_portfolio_value - current_portfolio_value) / peak_portfolio_value
        return drawdown > DRAWDOWN_HALT_THRESHOLD
