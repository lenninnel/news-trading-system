"""
Configuration settings for the News Trading System.

Loads environment variables via python-dotenv and exposes system-wide
constants so that every module imports from a single source of truth.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------

NEWSAPI_KEY: str = os.environ.get("NEWSAPI_KEY", "")
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
REDDIT_CLIENT_ID: str = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT: str = os.environ.get("REDDIT_USER_AGENT", "news-trading-bot/1.0")
MARKETAUX_API_TOKEN: str = os.environ.get("MARKETAUX_API_TOKEN", "")
ADANOS_API_KEY: str = os.environ.get("ADANOS_API_KEY", "")
FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")

# ---------------------------------------------------------------------------
# Claude model
# ---------------------------------------------------------------------------

CLAUDE_MODEL: str = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------------------

NEWSAPI_URL: str = "https://newsapi.org/v2/everything"
MAX_HEADLINES: int = 10

# ---------------------------------------------------------------------------
# Sentiment → numeric score mapping
# ---------------------------------------------------------------------------

SCORE_MAP: dict[str, int] = {"bullish": 1, "neutral": 0, "bearish": -1}

# ---------------------------------------------------------------------------
# Trading signal thresholds
# ---------------------------------------------------------------------------

BUY_THRESHOLD: float = 0.3
SELL_THRESHOLD: float = -0.3

# ---------------------------------------------------------------------------
# Source weights for multi-source sentiment
# ---------------------------------------------------------------------------

SOURCE_WEIGHTS: dict[str, float] = {
    "newsapi": 1.0,
    "marketaux": 0.9,
    "stocktwits": 0.8,
    "reddit": 0.6,
    "adanos": 0.5,
    "apewisdom": 0.4,
}

# Known crypto tickers (used to route to Binance instead of yfinance)
CRYPTO_TICKERS: set[str] = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX",
    "MATIC", "LINK", "UNI", "ATOM", "LTC", "FIL", "NEAR", "APT", "ARB",
}

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

DB_PATH: str = os.environ.get("DB_PATH", "news_trading.db")


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def validate_api_keys() -> None:
    """Raise RuntimeError if required API keys are missing."""
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not NEWSAPI_KEY:
        missing.append("NEWSAPI_KEY")
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them in your .env file or export them in your shell."
        )
