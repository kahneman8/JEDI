"""daily_brief/generate_brief.py"""
import os, json, re
from openai import OpenAI, NotFoundError
from .config import MODEL_COMPOSE, MAX_OUTPUT_TOKENS, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def _extract_first_json(text: str) -> str:
    """Return the first balanced JSON object from text."""
    if not text:
        raise ValueError("Empty model output")
    # Try fenced block first
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return m.group(1)
    # Scan for balanced braces
    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    raise ValueError("Unbalanced JSON braces in model output")

def compose_and_generate(
    date: str,
    market_summaries: dict,
    economic_events: list,
    news_by_sector: dict,
    watchlist_alerts: list,
    emerging_themes: list,
    sentiment_indicators: dict,
) -> tuple:
    # ---- Load schema (used in prompt + orchestrator validation) ----
    schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
    with open(schema_path, "r") as f:
        schema = json.load(f)

    context = {
        "date": date,
        "market_summaries": market_summaries or {},
        "economic_events": economic_events or [],
        "news_by_sector": news_by_sector,
        "watchlist_alerts": watchlist_alerts,
        "emerging_themes": emerging_themes,
        "sentiment_indicators": sentiment_indicators,
    }

    # ---------- 1) JSON (force JSON via response_format) ----------
    system_json = (
        "You are an equity research assistant. "
        "Output ONLY a single JSON object that STRICTLY matches the provided JSON Schema. "
        "No prose, no markdown. Keep URLs intact."
    )
    user_json = (
        f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Context JSON:\n{json.dumps(context, ensure_ascii=False)}"
    )

    try:
        r_json = client.chat.completions.create(
            model=MODEL_COMPOSE,
            messages=[
                {"role": "system", "content": system_json},
                {"role": "user", "content": user_json},
            ],
            # key change:
            response_format={"type": "json_object"},
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
    except NotFoundError:
        # fallback model if needed
        r_json = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_json},
                {"role": "user", "content": user_json},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )

    out_json_text = (r_json.choices[0].message.content or "").strip()
    # First try direct parse (should be valid JSON); fallback to extractor
    try:
        brief_json = json.loads(out_json_text)
    except Exception:
        brief_json = json.loads(_extract_first_json(out_json_text))

    # ---------- 2) Markdown ----------
    system_md = (
        "Format the following Morning Brief JSON into a clean Markdown report:\n"
        " - Title with date\n"
        " - Headings: Market Summaries, Economic Events, News by Sector, Watchlist Alerts, Emerging Themes\n"
        " - Bulleted items; include source URLs at the end of each bullet as citations\n"
        " - Output Markdown only (no JSON)."
    )
    user_md = json.dumps(brief_json, ensure_ascii=False)

    try:
        r_md = client.chat.completions.create(
            model=MODEL_COMPOSE,
            messages=[
                {"role": "system", "content": system_md},
                {"role": "user", "content": user_md},
            ],
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
    except NotFoundError:
        r_md = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_md},
                {"role": "user", "content": user_md},
            ],
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )

    brief_md = (r_md.choices[0].message.content or "").strip()
    return brief_json, brief_md
