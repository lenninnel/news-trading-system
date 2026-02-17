"""
News Trading Agent
------------------
Fetches financial headlines for a ticker symbol, scores sentiment using
Claude, and prints a BUY / SELL / HOLD recommendation.

Required environment variables:
    NEWSAPI_KEY   – API key from https://newsapi.org
    ANTHROPIC_API_KEY – API key from https://console.anthropic.com
"""

import json
import os
import sys

import anthropic
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEWSAPI_URL = "https://newsapi.org/v2/everything"
MAX_HEADLINES = 10  # Cap the number of headlines we analyse


# ---------------------------------------------------------------------------
# 1. Fetch headlines from NewsAPI
# ---------------------------------------------------------------------------

def fetch_headlines(ticker: str, api_key: str) -> list[str]:
    """Return up to MAX_HEADLINES recent headlines mentioning *ticker*."""
    params = {
        "q": ticker,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": MAX_HEADLINES,
        "apiKey": api_key,
    }
    resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
    resp.raise_for_status()

    articles = resp.json().get("articles", [])
    # Extract just the headline text
    headlines = [a["title"] for a in articles if a.get("title")]
    return headlines


# ---------------------------------------------------------------------------
# 2. Score each headline with Claude
# ---------------------------------------------------------------------------

def score_headline(headline: str, ticker: str, client: anthropic.Anthropic) -> dict:
    """Ask Claude to classify a single headline as bullish / bearish / neutral.

    Returns a dict like:
        {"sentiment": "bullish", "score": 1, "reason": "..."}
    """
    prompt = (
        f'You are a financial sentiment analyst. For the stock ticker "{ticker}", '
        f"classify the following headline as exactly one of: bullish, bearish, or neutral.\n\n"
        f'Headline: "{headline}"\n\n'
        "Respond with ONLY valid JSON (no markdown) in this format:\n"
        '{"sentiment": "bullish|bearish|neutral", "reason": "<one sentence>"}'
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text.strip()
    result = json.loads(text)

    # Map sentiment string to a numeric score
    score_map = {"bullish": 1, "neutral": 0, "bearish": -1}
    result["score"] = score_map.get(result["sentiment"], 0)
    return result


# ---------------------------------------------------------------------------
# 3. Combine scores into an overall sentiment rating
# ---------------------------------------------------------------------------

def aggregate_scores(scored: list[dict]) -> float:
    """Return the average numeric score across all headlines (-1 to +1)."""
    if not scored:
        return 0.0
    return sum(s["score"] for s in scored) / len(scored)


# ---------------------------------------------------------------------------
# 4. Generate recommendation
# ---------------------------------------------------------------------------

def recommendation(avg_score: float) -> str:
    """Map the average score to a simple trading signal."""
    if avg_score >= 0.3:
        return "BUY"
    elif avg_score <= -0.3:
        return "SELL"
    return "HOLD"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Read ticker from CLI args ---
    if len(sys.argv) < 2:
        print("Usage: python main.py <TICKER>")
        print("Example: python main.py AAPL")
        sys.exit(1)

    ticker = sys.argv[1].upper()

    # --- Validate API keys ---
    newsapi_key = os.environ.get("NEWSAPI_KEY")
    if not newsapi_key:
        sys.exit("Error: Set the NEWSAPI_KEY environment variable.")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        sys.exit("Error: Set the ANTHROPIC_API_KEY environment variable.")

    client = anthropic.Anthropic(api_key=anthropic_key)

    # --- Step 1: Fetch headlines ---
    print(f"\nFetching headlines for {ticker}...")
    headlines = fetch_headlines(ticker, newsapi_key)

    if not headlines:
        print("No headlines found. Try a different ticker or check your API key.")
        sys.exit(0)

    print(f"Found {len(headlines)} headline(s).\n")

    # --- Step 2: Score each headline ---
    scored: list[dict] = []
    for i, headline in enumerate(headlines, 1):
        print(f"[{i}/{len(headlines)}] Analysing: {headline}")
        try:
            result = score_headline(headline, ticker, client)
            result["headline"] = headline
            scored.append(result)
            icon = {"bullish": "+", "bearish": "-", "neutral": "~"}.get(result["sentiment"], "?")
            print(f"         [{icon}] {result['sentiment'].upper()} — {result['reason']}")
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"         [!] Skipped (parse error: {exc})")

    if not scored:
        print("\nCould not analyse any headlines.")
        sys.exit(0)

    # --- Step 3: Aggregate ---
    avg = aggregate_scores(scored)

    # --- Step 4: Recommend ---
    signal = recommendation(avg)

    # Count sentiments for the summary
    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    for s in scored:
        counts[s["sentiment"]] = counts.get(s["sentiment"], 0) + 1

    print("\n" + "=" * 60)
    print(f"  Ticker:       {ticker}")
    print(f"  Headlines:    {len(scored)} analysed")
    print(f"  Bullish:      {counts['bullish']}")
    print(f"  Bearish:      {counts['bearish']}")
    print(f"  Neutral:      {counts['neutral']}")
    print(f"  Avg Score:    {avg:+.2f}  (range -1.00 to +1.00)")
    print(f"  Signal:       {signal}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
