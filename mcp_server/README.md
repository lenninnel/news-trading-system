# NTS MCP Server

MCP (Model Context Protocol) server that makes the News Trading System queryable directly in Claude conversations.

## Tools

| Tool | Description |
|------|-------------|
| `get_signals` | Recent trading signals with optional day/strategy filters |
| `get_portfolio` | Current positions, P&L, and cash balance |
| `get_status` | System health, uptime, last/next session, watchlist |
| `get_performance` | Trade count, win rate, signals today |
| `get_signal_detail` | Deep dive on a ticker: scores, bull/bear debate, outcomes |

## Install

```bash
cd news-trading-system
pip install "mcp[cli]" httpx
```

## Configure for Claude Desktop

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nts-trading": {
      "command": "python",
      "args": ["-m", "mcp_server.nts_mcp"],
      "cwd": "/path/to/news-trading-system",
      "env": {
        "NTS_API_URL": "https://news-trading-system-production.up.railway.app"
      }
    }
  }
}
```

For local development, set `NTS_API_URL` to `http://localhost:8001`.

## Configure for Claude Code

Add to `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "nts-trading": {
      "command": "python",
      "args": ["-m", "mcp_server.nts_mcp"],
      "env": {
        "NTS_API_URL": "https://news-trading-system-production.up.railway.app"
      }
    }
  }
}
```

Or via CLI:

```bash
claude mcp add nts-trading python -- -m mcp_server.nts_mcp
```

## Run standalone (stdio)

```bash
python -m mcp_server.nts_mcp
```

Override the API URL:

```bash
NTS_API_URL=http://localhost:8001 python -m mcp_server.nts_mcp
```

## Example queries in Claude

- "What signals fired this week?"
- "Show me my portfolio"
- "Is the system running? When's the next session?"
- "How are we performing — win rate, trade count?"
- "Give me the full bull/bear analysis on AAPL"

## Tests

```bash
pytest tests/test_mcp_server.py -v
```
