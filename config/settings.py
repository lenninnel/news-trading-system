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
ALPHA_VANTAGE_KEY: str = os.environ.get("ALPHA_VANTAGE_KEY", "")

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

# ---------------------------------------------------------------------------
# Deployment / runtime
# ---------------------------------------------------------------------------

ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "development")
DATABASE_URL: str = os.environ.get("DATABASE_URL", "")  # PostgreSQL DSN in production
HEALTH_PORT: int = int(os.environ.get("HEALTH_PORT", "8080"))
ACCOUNT_BALANCE: float = float(os.environ.get("ACCOUNT_BALANCE", "10000.0"))

# ---------------------------------------------------------------------------
# Telegram notifications (optional)
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.environ.get("TELEGRAM_CHAT_ID", "")
