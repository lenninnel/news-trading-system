"""
Railway.app / Render.com deployment configuration.

Validates that all required environment variables are present and provides
a single Config object used across the application in production.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Required variables — the app refuses to start if any of these is missing
# ---------------------------------------------------------------------------

_REQUIRED_PROD = [
    "ANTHROPIC_API_KEY",
    "NEWSAPI_KEY",
    "DATABASE_URL",
]

_OPTIONAL = {
    "TELEGRAM_BOT_TOKEN": "",
    "TELEGRAM_CHAT_ID": "",
    "HEALTH_PORT": "8080",
    "ACCOUNT_BALANCE": "10000.0",
    "ENVIRONMENT": "development",
}


@dataclass
class DeploymentConfig:
    """Typed container for all deployment settings."""

    environment: str
    anthropic_api_key: str
    newsapi_key: str
    database_url: str
    telegram_bot_token: str
    telegram_chat_id: str
    health_port: int
    account_balance: float
    is_production: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_production = self.environment == "production"


def load_config(strict: bool = True) -> DeploymentConfig:
    """
    Load and validate the deployment configuration from environment variables.

    Args:
        strict: When True (default), raises SystemExit if required production
                variables are missing and ENVIRONMENT == "production".

    Returns:
        A populated DeploymentConfig instance.
    """
    env = os.environ.get("ENVIRONMENT", "development")

    if strict and env == "production":
        missing = [k for k in _REQUIRED_PROD if not os.environ.get(k)]
        if missing:
            print(
                f"[deployment] FATAL: missing required environment variables: "
                f"{', '.join(missing)}",
                file=sys.stderr,
            )
            sys.exit(1)

    return DeploymentConfig(
        environment=env,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        newsapi_key=os.environ.get("NEWSAPI_KEY", ""),
        database_url=os.environ.get("DATABASE_URL", ""),
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        health_port=int(os.environ.get("HEALTH_PORT", "8080")),
        account_balance=float(os.environ.get("ACCOUNT_BALANCE", "10000.0")),
    )


def print_config_summary(cfg: DeploymentConfig) -> None:
    """Print a sanitised (no secrets) configuration summary."""
    print("=" * 60)
    print("  Deployment Configuration")
    print("=" * 60)
    print(f"  Environment    : {cfg.environment}")
    print(f"  Backend DB     : {'PostgreSQL' if cfg.database_url else 'SQLite'}")
    print(f"  Anthropic key  : {'✓ set' if cfg.anthropic_api_key else '✗ missing'}")
    print(f"  NewsAPI key    : {'✓ set' if cfg.newsapi_key else '✗ missing'}")
    print(f"  Telegram       : {'✓ set' if cfg.telegram_bot_token else '✗ not configured'}")
    print(f"  Health port    : {cfg.health_port}")
    print(f"  Account bal    : ${cfg.account_balance:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    cfg = load_config(strict=False)
    print_config_summary(cfg)
