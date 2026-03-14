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
EODHD_API_TOKEN: str = os.environ.get("EODHD_API_TOKEN", "")

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
# German / EU ticker lists (used to route to EODHD)
# ---------------------------------------------------------------------------

DAX_TICKERS: list[str] = [
    "SAP.XETRA", "SIE.XETRA", "ALV.XETRA", "MUV2.XETRA", "BMW.XETRA",
    "VOW3.XETRA", "MBG.XETRA", "DTE.XETRA", "BAYN.XETRA", "BAS.XETRA",
    "ADS.XETRA", "RWE.XETRA", "EOAN.XETRA", "DBK.XETRA", "IFX.XETRA",
    "DHL.XETRA", "DB1.XETRA", "LIN.XETRA", "MRK.XETRA", "HEI.XETRA",
    "HEN3.XETRA", "FRE.XETRA", "ZAL.XETRA", "CON.XETRA", "VNA.XETRA",
    "RHM.XETRA", "AIR.XETRA", "PAH3.XETRA", "P911.XETRA", "BNR.XETRA",
    "MTX.XETRA", "SRT3.XETRA", "DHER.XETRA", "FME.XETRA", "CBK.XETRA",
    "HNR1.XETRA", "ENR.XETRA", "SHL.XETRA", "EVK.XETRA", "SY1.XETRA",
]

MDAX_TICKERS: list[str] = [
    "AFX.XETRA", "AIXA.XETRA", "BC8.XETRA", "BOSS.XETRA", "DWS.XETRA",
    "EVD.XETRA", "FNTN.XETRA", "GBF.XETRA", "HOT.XETRA", "LEG.XETRA",
    "MLP.XETRA", "NDX1.XETRA", "PSM.XETRA", "PUM.XETRA", "QIA.XETRA",
    "TAG.XETRA", "TUI1.XETRA", "WCHA.XETRA", "O2D.XETRA", "SDF.XETRA",
    "RDC.XETRA", "VBK.XETRA", "LHA.XETRA", "TKA.XETRA", "UTDI.XETRA",
]

# All known German tickers (union of DAX + MDAX)
_GERMAN_TICKERS_SET: set[str] = set(DAX_TICKERS) | set(MDAX_TICKERS)


def is_german_ticker(ticker: str) -> bool:
    """Return True if *ticker* should be routed to EODHD (German/EU stock)."""
    t = ticker.upper()
    if t.endswith(".XETRA") or t.endswith(".DE"):
        return True
    return t in _GERMAN_TICKERS_SET


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
