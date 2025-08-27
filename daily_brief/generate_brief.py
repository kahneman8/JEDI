"""daily_brief/generate_brief.py"""
import os, json, re
from openai import OpenAI
from .config import MODEL, MAX_OUTPUT_TOKENS, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def _extract_json(text: str) -> str:
    """Pull JSON either from ```json ... ``` or the first {...} block."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return m.group(1)
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1:
        return text[s : e + 1]
    raise ValueError("No JSON block found in model output.")


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
    Build the brief in two chat-completion calls:
      1) JSON (strict-by-prompt, then validated by orchestrator)
      2) Markdown (formatting only)
    Returns (json_dict, markdown_string).
    """
    # Load schema (used in prompt + orchestrator validation)
    schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
    try:
        with open(schema_path, "r") as f:
            schema = json.load(f)
    except FileNotFoundError:
        # Minimal fallback (or add the file)
        schema = {
            "type": "object",
            "required": [
                "date",
                "market_summaries",
                "economic_events",
                "news_by_sector",
                "watchlist_alerts",
                "emerging_themes",
                "sentiment_indicators",
            ],
            "properties": {
                "date": {"type": "string"},
                "market_summaries": {"type": "object"},
                "economic_events": {"type": "array"},
                "news_by_sector": {"type": "object"},
                "watchlist_alerts": {"type": "array"},
                "emerging_themes": {"type": "array"},
                "sentiment_indicators": {"type": "object"},
            },
        }

    context = {
        "date": date,
        "market_summaries": market_summaries or {},
        "economic_events": economic_events or [],
        "news_by_sector": news_by_sector,
        "watchlist_alerts": watchlist_alerts,
        "emerging_themes": emerging_themes,
        "sentiment_indicators": sentiment_indicators,
    }

    # ---------- 1) JSON ----------
    system_json = (
        "You are an equity research assistant.\n"
        "Task: output ONLY a single JSON object that STRICTLY matches the provided JSON Schema. "
        "No prose, no code fences, no markdown—JSON only. "
        "If market_summaries/economic_events are empty, infer concise content from the context. "
        "Keep URLs intact so bullets can cite them."
    )
    user_json = (
        f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Context JSON:\n{json.dumps(context, ensure_ascii=False)}"
    )
    r_json = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_json},
            {"role": "user", "content": user_json},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    out_json_text = r_json.choices[0].message.content or ""
    try:
        brief_json = json.loads(_extract_json(out_json_text))
    except Exception:
        # If model obeyed and returned raw JSON, try direct parse
        brief_json = json.loads(out_json_text)

    # ---------- 2) Markdown ----------
    system_md = (
        "Format the following Morning Brief JSON into a clean Markdown report:\n"
        " - Title with date\n"
        " - Headings: Market Summaries, Economic Events, News by Sector, Watchlist Alerts, Emerging Themes\n"
        " - Bulleted items; include source URLs at the end of each bullet as citations\n"
        " - Output Markdown only (no JSON)."
    )
    user_md = json.dumps(brief_json, ensure_ascii=False)
    r_md = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_md},
            {"role": "user", "content": user_md},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    brief_md = r_md.choices[0].message.content or ""

    return brief_json, brief_md
