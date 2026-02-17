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

Requires:
    No additional dependencies (uses only stdlib + storage layer)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agents.base_agent import BaseAgent
from storage.database import Database

# ── Constants ──────────────────────────────────────────────────────────────
_MAX_PORTFOLIO_FRACTION = 0.10   # 10 % portfolio cap
_MAX_RISK_FRACTION      = 0.02   # 2 % account risk per trade
_TRANSACTION_COST       = 0.001  # 0.1 % per trade
_REWARD_RISK_RATIO      = 2.0    # 2:1 take-profit
_MIN_CONFIDENCE         = 30.0   # below this → no position
_WIN_PROB_BASE          = 0.50   # base win probability (random)
_WIN_PROB_RANGE         = 0.30   # additional range driven by confidence

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

        # -- 1. Safety gates -----------------------------------------------
        skip_reason: str | None = None

        direction, strength = _parse_signal(signal)

        if confidence < _MIN_CONFIDENCE:
            skip_reason = f"Confidence {confidence:.0f}% is below minimum {_MIN_CONFIDENCE:.0f}%"
        elif direction == "HOLD":
            skip_reason = f"Signal '{signal}' carries no directional conviction"

        if skip_reason:
            return self._no_position(ticker, signal, confidence, current_price,
                                     account_balance, skip_reason)

        # -- 2. Stop-loss percentage ----------------------------------------
        stop_pct = _STOP_PCT[strength]

        # -- 3. Kelly Criterion sizing --------------------------------------
        kelly = _kelly_fraction(confidence)

        max_by_kelly     = account_balance * kelly
        max_by_portfolio = account_balance * _MAX_PORTFOLIO_FRACTION
        max_by_risk      = (account_balance * _MAX_RISK_FRACTION) / stop_pct

        position_raw = min(max_by_kelly, max_by_portfolio, max_by_risk)

        # -- 4. Deduct transaction cost ------------------------------------
        position_after_cost = position_raw * (1.0 - _TRANSACTION_COST)

        # -- 5. Round down to whole shares ---------------------------------
        shares = int(position_after_cost / current_price)
        if shares == 0:
            skip_reason = (
                f"Price ${current_price:.2f} exceeds affordable position "
                f"(budget: ${position_after_cost:.2f})"
            )
            return self._no_position(ticker, signal, confidence, current_price,
                                     account_balance, skip_reason)

        # Recalculate on whole-share basis
        position_size_usd = shares * current_price
        risk_amount       = position_size_usd * stop_pct

        # -- 6. Price targets ----------------------------------------------
        if direction == "BUY":
            stop_loss   = round(current_price * (1.0 - stop_pct), 4)
            take_profit = round(current_price * (1.0 + stop_pct * _REWARD_RISK_RATIO), 4)
        else:  # SELL / short
            stop_loss   = round(current_price * (1.0 + stop_pct), 4)
            take_profit = round(current_price * (1.0 - stop_pct * _REWARD_RISK_RATIO), 4)

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
        }
