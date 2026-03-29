"""
MCP server for the News Trading System.

Exposes read-only tools that proxy the existing FastAPI endpoints,
so Claude can query signals, portfolio, status and performance data
directly in conversation.

Usage:
    python -m mcp_server.server          # stdio transport (default)
    NTS_API_URL=http://localhost:8001 python -m mcp_server.server

Environment:
    NTS_API_URL  Base URL of the NTS FastAPI service.
                 Default: https://news-trading-system-production.up.railway.app
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_API_URL = os.environ.get(
    "NTS_API_URL",
    "https://news-trading-system-production.up.railway.app",
).rstrip("/")

_TIMEOUT = 15  # seconds

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

server = Server("nts")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _api_get(path: str, params: dict[str, Any] | None = None) -> dict | list:
    """GET request to the NTS API. Returns parsed JSON."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_API_URL}{path}", params=params)
        resp.raise_for_status()
        return resp.json()


def _text(content: Any) -> list[TextContent]:
    """Wrap a value as MCP TextContent."""
    if isinstance(content, (dict, list)):
        text = json.dumps(content, indent=2, default=str)
    else:
        text = str(content)
    return [TextContent(type="text", text=text)]


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="get_signals",
        description=(
            "Fetch recent signal events from the News Trading System. "
            "Returns signals with ticker, strategy, confidence, sentiment, "
            "price, and outcome data."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max signals to return (1-500).",
                    "default": 20,
                },
                "days": {
                    "type": "integer",
                    "description": "Look-back window in days (1-365).",
                    "default": 7,
                },
                "ticker": {
                    "type": "string",
                    "description": "Filter by ticker symbol (e.g. AAPL).",
                },
                "strategy": {
                    "type": "string",
                    "description": "Filter by strategy name.",
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_portfolio",
        description=(
            "Get current portfolio state: total value, daily PnL, cash, "
            "and individual positions with entry/current prices."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_status",
        description=(
            "Get system status: uptime, trading mode, last/next session, "
            "and the active watchlist."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_performance",
        description=(
            "Get trading performance metrics: total trades, win rate, "
            "signals today, strong-buy count, and sessions completed."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
]


# ---------------------------------------------------------------------------
# MCP handlers
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        if name == "get_signals":
            params: dict[str, Any] = {
                "limit": arguments.get("limit", 20),
                "days": arguments.get("days", 7),
            }
            if arguments.get("ticker"):
                params["ticker"] = arguments["ticker"]
            # The FastAPI endpoint doesn't have a strategy param,
            # so we filter client-side if requested.
            strategy_filter = arguments.get("strategy")

            data = await _api_get("/api/signals", params=params)

            if strategy_filter and isinstance(data, list):
                strategy_filter = strategy_filter.upper()
                data = [
                    s for s in data
                    if (s.get("strategy") or "").upper() == strategy_filter
                ]

            summary = f"Returned {len(data)} signal(s).\n\n"
            return _text(summary + json.dumps(data, indent=2, default=str))

        elif name == "get_portfolio":
            data = await _api_get("/api/portfolio")
            return _text(data)

        elif name == "get_status":
            data = await _api_get("/api/status")
            return _text(data)

        elif name == "get_performance":
            data = await _api_get("/api/performance")
            return _text(data)

        else:
            return _text({"error": f"Unknown tool: {name}"})

    except httpx.HTTPStatusError as exc:
        return _text({
            "error": f"API returned {exc.response.status_code}",
            "detail": exc.response.text[:500],
        })
    except httpx.ConnectError:
        return _text({
            "error": "Cannot reach NTS API",
            "url": _API_URL,
            "hint": "Set NTS_API_URL or ensure the API is running.",
        })
    except Exception as exc:
        return _text({"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
