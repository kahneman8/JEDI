import os, json, re, time, random, openai
from openai import OpenAI, NotFoundError
from urllib.parse import urlparse
from .config import OPENAI_API_KEY, MODEL_COMPOSE_PREF, MAX_COMPLETION_TOKENS

client = OpenAI(api_key=OPENAI_API_KEY)

def _chat_json_with_backoff(models, system, user, max_tokens):
    delay = 0.35
    for _ in range(6):
        for m in models:
            try:
                r = client.chat.completions.create(
                    model=m,
                    messages=[{"role":"system","content":system},
                              {"role":"user","content":user}],
                    response_format={"type":"json_object"},
                    max_completion_tokens=max_tokens,
                )
                txt = (r.choices[0].message.content or "").strip()
                if txt:
                    return json.loads(txt)
            except openai.RateLimitError:
                time.sleep(delay + random.uniform(0,0.3))
            except NotFoundError:
                continue
            except Exception:
                time.sleep(0.2)
        delay = min(delay * 2, 6.0)
    raise RuntimeError("Compose JSON failed after retries/fallbacks")

def _hostname(u: str) -> str:
    try:
        host = urlparse(u).netloc
        host = host.replace("www.", "").split(":")[0]
        return host or "source"
    except Exception:
        return "source"

def _render_markdown(brief_json: dict) -> str:
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
    evs = brief_json.get("economic_events", []) or []
    if evs:
        for e in evs:
            t = e.get("event","")
            imp = e.get("impact")
            lines.append(f"- {t}" + (f" — {imp}" if imp else ""))
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## News by Sector")
    nbs = brief_json.get("news_by_sector", {}) or {}
    for sector, items in nbs.items():
        lines.append(f"### {sector}")
        if not items:
            lines.append("- None")
        else:
            for it in items:
                h = it.get("headline","")
                s = it.get("sentiment","")
                u = it.get("url","")
                src = it.get("source") or _hostname(u)
                lines.append(f"- {h} ({s}) — [{src}]({u})")
        lines.append("")
    lines.append("")
    lines.append("## Watchlist Alerts")
    alerts = brief_json.get("watchlist_alerts", []) or []
    if alerts:
        for a in alerts:
            lines.append(f"- {a}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Emerging Themes")
    themes = brief_json.get("emerging_themes", []) or []
    if themes:
        for t in themes:
            lines.append(f"- **{t.get('theme','')}**: {t.get('description','')}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)

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
        "watchlist_alerts": watchlist_alerts or [],
        "emerging_themes": emerging_themes or [],
        "sentiment_indicators": sentiment_indicators or {},
    }

    # small pause to ease TPM rollover
    time.sleep(1.1)

    # ---- JSON via Chat JSON-mode with backoff + mini→full fallback ----
    sys_json = (
        "You are an equity research assistant. "
        "Output ONLY one JSON object that STRICTLY matches the provided JSON Schema. "
        "No prose, no markdown."
    )
    usr_json = (
        f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Context JSON:\n{json.dumps(context, ensure_ascii=False)}"
    )
    brief_json = _chat_json_with_backoff(MODEL_COMPOSE_PREF, sys_json, usr_json, MAX_COMPLETION_TOKENS)

    # ---- Markdown rendered locally (no LLM; zero risk of 429/empty) ----
    brief_md = _render_markdown(brief_json)

    return brief_json, brief_md
