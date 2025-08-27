import os, json
from openai import OpenAI
from .config import MODEL_COMPOSE, MAX_OUTPUT_TOKENS, OPENAI_API_KEY

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
    schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
    schema = json.load(open(schema_path))

    context = {
        "date": date,
        "market_summaries": market_summaries or {},
        "economic_events": economic_events or [],
        "news_by_sector": news_by_sector,
        "watchlist_alerts": watchlist_alerts,
        "emerging_themes": emerging_themes,
        "sentiment_indicators": sentiment_indicators,
    }

    # 1) JSON via strict schema
    prompt_json = (
        "You are an equity research assistant. Using only the provided context JSON, "
        "produce a Morning Brief that strictly matches the schema (no extra fields). "
        "If market_summaries/economic_events are empty, infer concise content from the context. "
        "Preserve all URLs so every bullet can be cited.\n\n"
        f"Context JSON:\n{json.dumps(context, ensure_ascii=False)}"
    )

    r_json = client.responses.create(
        model=MODEL_COMPOSE,
        input=prompt_json,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "MorningBrief", "schema": schema, "strict": True},
        },
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    brief_json = json.loads(r_json.output_text)

    # 2) Markdown
    prompt_md = (
        "Format the following Morning Brief JSON into a clean Markdown report:\n"
        " - Title with date\n"
        " - Headings for Market Summaries, Economic Events, News by Sector, Watchlist Alerts, Emerging Themes\n"
        " - Bulleted items; include source URLs at the end of each bullet\n"
        " - Output only Markdown\n\n"
        f"{json.dumps(brief_json, ensure_ascii=False)}"
    )

    r_md = client.responses.create(
        model=MODEL_COMPOSE,
        input=prompt_md,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    brief_md = r_md.output_text
    return brief_json, brief_md
