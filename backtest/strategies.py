"""
Strategy definitions and ticker universe for the comparison framework.

Each strategy is a dict of parameters passed to ``backtest.engine.run_backtest()``.
"""

from __future__ import annotations

# ── Strategy definitions ─────────────────────────────────────────────

STRATEGIES: dict[str, dict] = {
    "BASELINE": {
        "buy_threshold": 0.3,
        "sell_threshold": -0.3,
        "stop_loss_pct": 0.02,
        "take_profit_ratio": 2.0,
        "use_sentiment": True,
        "use_technical": True,
    },
    "TECHNICAL_ONLY": {
        "buy_threshold": 0.0,
        "sell_threshold": 0.0,
        "stop_loss_pct": 0.02,
        "take_profit_ratio": 2.0,
        "use_sentiment": False,
        "use_technical": True,
    },
    "SENTIMENT_ONLY": {
        "buy_threshold": 0.3,
        "sell_threshold": -0.3,
        "stop_loss_pct": 0.02,
        "take_profit_ratio": 2.0,
        "use_sentiment": True,
        "use_technical": False,
    },
    "MOMENTUM": {
        "buy_threshold": 0.5,
        "sell_threshold": -0.5,
        "stop_loss_pct": 0.015,
        "take_profit_ratio": 3.0,
        "use_sentiment": True,
        "use_technical": True,
    },
    "MEAN_REVERSION": {
        "buy_threshold": 0.2,
        "sell_threshold": -0.2,
        "stop_loss_pct": 0.03,
        "take_profit_ratio": 1.5,
        "use_sentiment": True,
        "use_technical": True,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
    },
    "TREND_FOLLOWING": {
        "buy_threshold": 0.25,
        "sell_threshold": -0.25,
        "stop_loss_pct": 0.025,
        "take_profit_ratio": 2.5,
        "use_sentiment": True,
        "use_technical": True,
        "require_trend_alignment": True,
    },
    "NEWS_EVENT_DRIVEN": {
        "buy_threshold": 0.6,
        "sell_threshold": -0.6,
        "stop_loss_pct": 0.02,
        "take_profit_ratio": 2.0,
        "use_sentiment": True,
        "use_technical": True,
    },
}

# Required keys every strategy must have
_REQUIRED_KEYS = {
    "buy_threshold", "sell_threshold", "stop_loss_pct",
    "take_profit_ratio", "use_sentiment", "use_technical",
}

# ── Ticker universe ──────────────────────────────────────────────────

TICKERS: dict[str, list[str]] = {
    "AI_CHIPS":    ["NVDA", "AMD", "TSLA", "MSFT", "META", "GOOGL"],
    "DATACENTER":  ["VST", "CEG", "SMCI", "DELL", "AAPL"],
    "GERMAN_TECH": ["SAP.XETRA", "SIE.XETRA"],
    "CRYPTO":      ["BTC", "ETH", "SOL"],
}

ALL_TICKERS: list[str] = [t for group in TICKERS.values() for t in group]

# Reverse lookup: ticker → sector
TICKER_SECTOR: dict[str, str] = {}
for _sector, _tickers in TICKERS.items():
    for _t in _tickers:
        TICKER_SECTOR[_t] = _sector

# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_START = "2024-01-01"
DEFAULT_END = "2025-01-01"
DEFAULT_BALANCE = 10_000.0
