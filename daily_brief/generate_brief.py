import os, json, re, time, random
from openai import OpenAI
from .config import OPENAI_API_KEY, MODEL, MAX_COMPLETION_TOKENS

client = OpenAI(api_key=OPENAI_API_KEY)

def _extract_first_json(text: str) -> str:
    if not text or not text.strip():
        raise ValueError("Empty model output")
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return m.group(1)
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1:
        return text[s:e+1]
    raise ValueError("No JSON found")

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

    # JSON via Chat JSON-mode; fallback to extractor if needed
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
            model=MODEL,
            messages=[{"role":"system","content": system_json},
                      {"role":"user","content": user_json}],
            response_format={"type":"json_object"},
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        out_json_text = (r_json.choices[0].message.content or "").strip()
        brief_json = json.loads(out_json_text)
    except Exception:
        # fallback: try to parse first JSON object without JSON mode
        r_json = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content": system_json},
                      {"role":"user","content": user_json}],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        raw = (r_json.choices[0].message.content or "").strip()
        brief_json = json.loads(_extract_first_json(raw))

    # Markdown formatting (model); fallback to code formatting if empty
    system_md = (
        "Format the following Morning Brief JSON into a clean Markdown report:\n"
        " - Title with date\n"
        " - Headings for Market Summaries, Economic Events, News by Sector, Watchlist Alerts, Emerging Themes\n"
        " - Bulleted items; include source URLs at the end of each bullet as citations\n"
        " - Output Markdown only (no JSON)."
    )
    user_md = json.dumps(brief_json, ensure_ascii=False)

    try:
        r_md = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content": system_md},
                      {"role":"user","content": user_md}],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        brief_md = (r_md.choices[0].message.content or "").strip()
        if not brief_md:
            raise ValueError("empty md")
    except Exception:
        # deterministic Markdown fallback
        lines = []
        lines.append(f"# Morning Market Brief — {brief_json.get('date','')}")
        lines.append("")
        ms = brief_json.get("market_summaries", {})
        lines.append("## Market Summaries")
        lines.append(f"- **Global:** {ms.get('global','')}")
        lines.append(f"- **Asia:** {ms.get('asia','')}")
        lines.append(f"- **Indonesia:** {ms.get('indonesia','')}")
        lines.append("")
        lines.append("## Economic Events")
        evs = brief_json.get("economic_events", [])
        if evs:
            for e in evs:
                t = e.get("event","")
                if e.get("impact"):
                    t += f" — {e['impact']}"
                lines.append(f"- {t}")
        else:
            lines.append("- None")
        lines.append("")
        lines.append("## News by Sector")
        for sector, items in (brief_json.get("news_by_sector") or {}).items():
            lines.append(f"### {sector}")
            if not items:
                lines.append("- None")
            else:
                for it in items:
                    lines.append(f"- {it.get('headline','')} ({it.get('sentiment','')}) — [{it.get('source','source')}]({it.get('url','')})")
            lines.append("")
        lines.append("")
        lines.append("## Watchlist Alerts")
        alerts = brief_json.get("watchlist_alerts", [])
        if alerts:
            for a in alerts:
                lines.append(f"- {a}")
        else:
            lines.append("- None")
        lines.append("")
        lines.append("## Emerging Themes")
        themes = brief_json.get("emerging_themes", [])
        if themes:
            for t in themes:
                lines.append(f"- **{t.get('theme','')}**: {t.get('description','')}")
        else:
            lines.append("- None")
        lines.append("")
        brief_md = "\n".join(lines)

    return brief_json, brief_md
