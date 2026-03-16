"""
Alpaca broker execution layer.

AlpacaTrader places real orders through the Alpaca API (paper or live)
and syncs the resulting positions back to the local SQLite database so
that all downstream reporting stays consistent.

Environment variables
---------------------
ALPACA_API_KEY      Alpaca API key ID
ALPACA_SECRET_KEY   Alpaca API secret key
ALPACA_MODE         "paper" (default) or "live"
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from storage.database import Database

log = logging.getLogger(__name__)

_PAPER_URL = "https://paper-api.alpaca.markets"
_LIVE_URL = "https://api.alpaca.markets"

# How long to wait for an order to fill before giving up (seconds).
_FILL_TIMEOUT = 30
_POLL_INTERVAL = 1


class AlpacaTrader:
    """
    Live / paper broker execution via the Alpaca API.

    Exposes the same public interface as PaperTrader so the two are
    interchangeable behind broker_factory.

    Args:
        db:  Optional Database instance (used to sync positions locally).
        api: Optional pre-built ``tradeapi.REST`` instance (for testing).
    """

    def __init__(
        self,
        db: Database | None = None,
        api: "tradeapi.REST | None" = None,
    ) -> None:
        self._db = db or Database()

        if api is not None:
            self._api = api
        else:
            import alpaca_trade_api as tradeapi

            key = os.environ.get("ALPACA_API_KEY", "")
            secret = os.environ.get("ALPACA_SECRET_KEY", "")
            mode = os.environ.get("ALPACA_MODE", "paper").lower()
            base_url = _LIVE_URL if mode == "live" else _PAPER_URL

            if not key or not secret:
                raise RuntimeError(
                    "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"
                )

            self._api = tradeapi.REST(
                key_id=key,
                secret_key=secret,
                base_url=base_url,
                api_version="v2",
            )

    # ------------------------------------------------------------------
    # Public API  (mirrors PaperTrader)
    # ------------------------------------------------------------------

    def track_trade(
        self,
        ticker: str,
        action: str,
        shares: int,
        price: float,
        stop_loss: "float | None" = None,
        take_profit: "float | None" = None,
        **kwargs: Any,
    ) -> dict:
        """
        Submit an order to Alpaca, wait for a fill, and sync locally.

        When both *stop_loss* and *take_profit* are provided the order is
        submitted as a bracket (OTO) order so the protective legs are
        created atomically on the exchange.

        Returns the same dict shape as ``PaperTrader.track_trade``.
        """
        ticker = ticker.upper()
        action = action.upper()
        if action not in ("BUY", "SELL"):
            raise ValueError(f"action must be BUY or SELL, got '{action}'")
        if shares < 1:
            raise ValueError(f"shares must be >= 1, got {shares}")

        side = "buy" if action == "BUY" else "sell"

        # Build order kwargs
        order_params: dict[str, Any] = dict(
            symbol=ticker,
            qty=round(shares, 4),
            side=side,
            type="market",
            time_in_force="day",
        )

        if stop_loss is not None and take_profit is not None:
            order_params["order_class"] = "bracket"
            order_params["stop_loss"] = {"stop_price": str(round(stop_loss, 2))}
            order_params["take_profit"] = {"limit_price": str(round(take_profit, 2))}

        try:
            order = self._api.submit_order(**order_params)
        except Exception as exc:
            log.error(
                "TRADE FAILED: %s %s %d shares — %s", action, ticker, shares, exc,
            )
            self._notify_trade_failed(ticker, action, shares, price, str(exc))
            raise

        # Poll until filled or timeout
        fill_price = self._wait_for_fill(order.id)
        if fill_price is None:
            fill_price = price  # fallback to quoted price

        # Sync position to local DB
        pnl = self._sync_position(ticker, action, shares, fill_price)

        trade_id = self._db.log_trade_history(
            ticker=ticker,
            action=action,
            shares=shares,
            price=fill_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pnl=pnl,
        )

        log.info(
            "TRADE EXECUTED: %s %s %d shares @ $%.2f (trade_id=%s, SL=%.2f, TP=%.2f)",
            action, ticker, shares, fill_price, trade_id,
            stop_loss or 0.0, take_profit or 0.0,
        )

        self._notify_trade_executed(
            ticker, action, shares, fill_price, stop_loss, take_profit,
        )

        return {
            "trade_id": trade_id,
            "ticker": ticker,
            "action": action,
            "shares": shares,
            "price": fill_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "pnl": pnl,
            "total_value": round(shares * fill_price, 2),
        }

    def get_portfolio(self) -> list[dict]:
        """Return current positions from Alpaca, synced to local DB."""
        positions = self._api.list_positions()
        result = []
        for pos in positions:
            row = {
                "ticker": pos.symbol,
                "shares": int(pos.qty),
                "avg_price": float(pos.avg_entry_price),
                "current_value": float(pos.market_value),
                "updated_at": None,
            }
            # Keep local DB in sync
            self._db.set_portfolio_position(
                ticker=row["ticker"],
                shares=row["shares"],
                avg_price=row["avg_price"],
                current_value=row["current_value"],
            )
            result.append(row)
        return result

    def get_trade_history(
        self,
        ticker: "str | None" = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return trade history from local DB (canonical audit trail)."""
        return self._db.get_trade_history(ticker=ticker, limit=limit)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _wait_for_fill(self, order_id: str) -> "float | None":
        """Poll Alpaca until the order fills or we time out."""
        elapsed = 0.0
        while elapsed < _FILL_TIMEOUT:
            order = self._api.get_order(order_id)
            if order.status == "filled":
                return float(order.filled_avg_price)
            if order.status in ("canceled", "expired", "rejected"):
                return None
            time.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL
        return None

    def _sync_position(
        self,
        ticker: str,
        action: str,
        shares: int,
        fill_price: float,
    ) -> float:
        """
        Update the local portfolio DB after an Alpaca fill.

        Returns realised PnL (0.0 for buys).
        """
        pnl = 0.0
        existing = self._db.get_portfolio_position(ticker)

        if action == "BUY":
            if existing:
                total_shares = existing["shares"] + shares
                new_avg = (
                    existing["shares"] * existing["avg_price"]
                    + shares * fill_price
                ) / total_shares
            else:
                total_shares = shares
                new_avg = fill_price

            self._db.set_portfolio_position(
                ticker=ticker,
                shares=total_shares,
                avg_price=new_avg,
                current_value=round(total_shares * fill_price, 2),
            )
        else:  # SELL
            if existing:
                pnl = round((fill_price - existing["avg_price"]) * shares, 2)
                remaining = existing["shares"] - shares
                if remaining <= 0:
                    self._db.delete_portfolio_position(ticker)
                else:
                    self._db.set_portfolio_position(
                        ticker=ticker,
                        shares=remaining,
                        avg_price=existing["avg_price"],
                        current_value=round(remaining * fill_price, 2),
                    )

        return pnl

    # ------------------------------------------------------------------
    # Telegram notifications (best-effort, never crashes the pipeline)
    # ------------------------------------------------------------------

    def _get_telegram(self):
        """Lazily build a TelegramNotifier, returning None if unconfigured."""
        if hasattr(self, "_telegram"):
            return self._telegram
        try:
            from notifications.telegram_bot import TelegramNotifier

            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if token and chat_id:
                self._telegram = TelegramNotifier(
                    bot_token=token, chat_id=chat_id,
                )
            else:
                self._telegram = None
        except Exception:
            self._telegram = None
        return self._telegram

    def _notify_trade_executed(
        self,
        ticker: str,
        action: str,
        shares: int,
        price: float,
        stop_loss: "float | None",
        take_profit: "float | None",
    ) -> None:
        """Send a Telegram alert for a successful trade. Never raises."""
        try:
            tg = self._get_telegram()
            if tg is None:
                return
            tg.send_trade_executed(
                ticker=ticker,
                action=action,
                shares=shares,
                price=price,
                stop_loss=stop_loss or 0.0,
                take_profit=take_profit or 0.0,
            )
        except Exception as exc:
            log.debug("Telegram trade notification failed (non-fatal): %s", exc)

    def _notify_trade_failed(
        self,
        ticker: str,
        action: str,
        shares: int,
        price: float,
        reason: str,
    ) -> None:
        """Send a Telegram alert for a failed trade. Never raises."""
        try:
            tg = self._get_telegram()
            if tg is None:
                return
            msg = (
                f"\u274c *Trade FAILED* \u2014 `{ticker}`\n"
                f"Action: {action} {shares} shares @ ${price:,.2f}\n"
                f"Reason: _{reason[:300]}_"
            )
            tg._send(msg)
        except Exception as exc:
            log.debug("Telegram failure notification failed (non-fatal): %s", exc)
