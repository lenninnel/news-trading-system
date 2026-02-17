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
# Storage
# ---------------------------------------------------------------------------

DB_PATH: str = os.environ.get("DB_PATH", "news_trading.db")
