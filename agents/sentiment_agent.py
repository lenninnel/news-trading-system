"""
Sentiment analysis agent powered by Claude.

SentimentAgent receives a single news headline and a ticker symbol, asks
Claude to classify the headline as bullish, bearish, or neutral, and returns
a structured result dict.

The agent is intentionally stateless between calls so it can be reused
across many headlines without side-effects.
"""

from __future__ import annotations

import json

import anthropic

from agents.base_agent import BaseAgent
from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, SCORE_MAP

_PROMPT_TEMPLATE = (
    'You are a financial sentiment analyst. For the stock ticker "{ticker}", '
    "classify the following headline as exactly one of: bullish, bearish, or neutral.\n\n"
    'Headline: "{headline}"\n\n'
    "Respond with ONLY valid JSON (no markdown) in this format:\n"
    '{{"sentiment": "bullish|bearish|neutral", "reason": "<one sentence>"}}'
)


class SentimentAgent(BaseAgent):
    """
    Classifies news headlines using Claude and returns a sentiment score.

    Attributes:
        _client: Shared Anthropic client instance.

    Example::

        agent = SentimentAgent()
        result = agent.run("Apple hits all-time high", "AAPL")
        # {"sentiment": "bullish", "score": 1, "reason": "...", "headline": "..."}
    """

    def __init__(self, api_key: str = ANTHROPIC_API_KEY) -> None:
        """
        Initialise the agent with an Anthropic client.

        Args:
            api_key: Anthropic API key. Defaults to settings.ANTHROPIC_API_KEY.
        """
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "SentimentAgent"

    def run(self, headline: str, ticker: str) -> dict:
        """
        Classify a single headline for the given ticker.

        Args:
            headline: The news headline text to analyse.
            ticker:   Stock ticker symbol (e.g. "AAPL").

        Returns:
            dict with keys:
                sentiment (str):  "bullish", "bearish", or "neutral"
                score     (int):  +1, 0, or -1
                reason    (str):  One-sentence justification from Claude
                headline  (str):  The original headline (passed through)

        Raises:
            json.JSONDecodeError: If Claude's response is not valid JSON.
            KeyError:             If the response is missing expected fields.
            anthropic.APIError:   On API communication failures.
        """
        prompt = _PROMPT_TEMPLATE.format(ticker=ticker, headline=headline)

        message = self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )

        text = message.content[0].text.strip()
        result: dict = json.loads(text)
        result["score"] = SCORE_MAP.get(result["sentiment"], 0)
        result["headline"] = headline
        return result
