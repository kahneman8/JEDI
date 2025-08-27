"""daily_brief/generate_brief.py"""
import os, json
from openai import OpenAI
from .config import MODEL, MAX_TOKENS, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

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
    Build the brief in two steps using the v1 SDK:
      1) JSON via Responses API + json_schema (strict)
      2) Markdown via Responses API (formatting only)
    Returns (json_dict, markdown_string).
    """
    # 1) Load the same schema used for validation so the model emits strict JSON
    schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
    schema = json.load(open(schema_path))

    # Provide the collected data as context; the model fills any missing summaries/events
    context = {
        "date": date,
        "market_summaries": market_summaries or {},
        "economic_events": economic_events or [],
        "news_by_sector": news_by_sector,
        "watchlist_alerts": watchlist_alerts,
        "emerging_themes": emerging_themes,
        "sentiment_indicators": sentiment_indicators,
    }

    prompt_json = (
        "You are an equity research assistant. Using ONLY the provided context JSON, "
        "produce a Morning Brief that strictly matches the schema (no extra fields). "
        "If market_summaries/economic_events are empty, infer concise content from the context. "
        "Preserve all URLs so every bullet can be cited.\n\n"
        f"Context JSON:\n{json.dumps(context, ensure_ascii=False)}"
    )

    r_json = client.responses.create(
        model=MODEL,
        input=prompt_json,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "MorningBrief",
                "schema": schema,
                "strict": True,
            },
        },
        max_output_tokens=MAX_TOKENS,
    )

    json_text = r_json.output_text
    brief_json = json.loads(json_text)

    # 2) Render Markdown from the validated JSON (formatting only)
    prompt_md = (
        "Format the following Morning Brief JSON into a clean Markdown report:\n"
        " - Title with date\n"
        " - Headings for Market Summaries, Economic Events, News by Sector, Watchlist Alerts, Emerging Themes\n"
        " - Bulleted items\n"
        " - Include source URLs at the end of each bullet as citations\n"
        " - Do NOT output JSON; only Markdown\n\n"
        f"{json.dumps(brief_json, ensure_ascii=False)}"
    )

    r_md = client.responses.create(
        model=MODEL,
        input=prompt_md,
        max_output_tokens=MAX_TOKENS,
    )
    brief_md = r_md.output_text

    return brief_json, brief_md
