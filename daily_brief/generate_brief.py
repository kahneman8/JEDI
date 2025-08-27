"""phase1_daily_brief/generate_brief.py"""
import json, re, openai
from .config import MODEL, TEMPERATURE, MAX_TOKENS


def compose_and_generate(
    date: str,
    market_summaries: dict,
    economic_events: list,
    news_by_sector: dict,
    watchlist_alerts: list,
    emerging_themes: list,
    sentiment_indicators: dict,
) -> tuple:
    """
    Use GPT-5 Pro to assemble the brief. Returns (json_dict, markdown_string).
    """
    # Prepare text for sectors and watchlists
    sectors_text = ""
    for sector, items in news_by_sector.items():
        sectors_text += f"\n{sector}:\n"
        for it in items:
            title = it.get("headline", "")
            sentiment = it.get("sentiment", "")
            url = it.get("url", "")
            sectors_text += f"- {title} ({sentiment}) ({url})\n"

    watchlist_text = "".join(f"- {alert}\n" for alert in watchlist_alerts)
    themes_text = "".join(
        f"- **{t['theme']}**: {t['description']}\n" for t in emerging_themes
    )

    # System instructions and schema description
    system_prompt = (
        "You are an equity research AI assistant tasked with generating a structured morning market brief. "
        "Produce two outputs: first, a JSON object strictly following the provided schema; second, a well-formatted Markdown report. "
        "Every statement must end with a citation using the provided URLs. Do not invent facts."
    )

    schema_desc = """
JSON schema:
{
  "date": "<YYYY-MM-DD>",
  "market_summaries": {"global": "<text>", "asia": "<text>", "indonesia": "<text>"},
  "economic_events": [{"event": "<text>", "impact": "<text>"}],
  "news_by_sector": {
    "<Sector>": [
      {"headline": "<text>", "sentiment": "Positive|Negative|Neutral", "source": "<publisher>", "url": "<url>"}
    ]
  },
  "watchlist_alerts": ["<alert>"],
  "emerging_themes": [{"theme": "<name>", "description": "<sentence>"}],
  "sentiment_indicators": {
    "<Sector>": {"Positive": <int>, "Negative": <int>, "Neutral": <int>}
  }
}
The Markdown report should mirror the same data with headings and bullets and citation footnotes.
"""

    # Compose user content with provided data
    user_content = (
        f"Date: {date}\n\n"
        "News by sector:\n" + sectors_text + "\n"
        "Watchlist alerts:\n" + watchlist_text + "\n"
        "Emerging themes:\n" + themes_text + "\n"
        "(Economic events and market summaries can be inferred from this context if not provided.)"
    )

    messages = [
        {"role": "system", "content": system_prompt + schema_desc},
        {"role": "user", "content": user_content},
    ]

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    output = response.choices[0].message["content"]

    # Extract JSON from triple backticks or braces
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", output, re.S)
    if json_match:
        json_text = json_match.group(1)
        markdown_text = output.replace(json_match.group(0), "").strip()
    else:
        # Fallback: find first curly brace block as JSON
        start = output.find("{")
        end = output.rfind("}")
        json_text = output[start : end + 1]
        markdown_text = output[end + 1 :].strip()

    # Parse JSON safely
    try:
        brief_json = json.loads(json_text)
    except json.JSONDecodeError:
        # If JSON fails, parse loosely (replace newlines)
        brief_json = json.loads(json_text.replace("\n", " "))

    return brief_json, markdown_text
