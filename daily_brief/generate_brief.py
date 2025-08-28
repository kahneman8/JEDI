"""
Compose with LLM reasoning for summaries + deterministic JSON:
- LLM (MODEL_REASON) writes concise summaries for Global/Asia/Indonesia from fetched items.
- JSON is built only from fetched items (no fabricated URLs).
- Markdown rendered locally.
"""
import json, time, random
from typing import Dict, List
from urllib.parse import urlparse
from openai import OpenAI
from .config import OPENAI_API_KEY, MODEL_REASON, MAX_COMPLETION_TOKENS, SUMMARY_ITEMS_PER_REGION

client = OpenAI(api_key=OPENAI_API_KEY)

def _host(url: str) -> str:
    try:
        h = urlparse(url).netloc.replace("www.", "").split(":")[0]
        return h or "source"
    except Exception:
        return "source"

def _fallback_summary(items: List[Dict], label: str) -> str:
    pos = sum(1 for it in items if it.get("sentiment") == "Positive")
    neg = sum(1 for it in items if it.get("sentiment") == "Negative")
    neu = sum(1 for it in items if it.get("sentiment") == "Neutral")
    return f"{label} headlines: {len(items)} items (Positive {pos}, Negative {neg}, Neutral {neu})."

def _partition_by_region(all_items: List[Dict]):
    indo = [it for it in all_items if it.get("region") == "Indonesia"]
    asia = [it for it in all_items if it.get("region") == "Asia"]
    glob = [it for it in all_items if it.get("region") == "Global"]
    return glob, asia, indo

def _backoff(call, *args, **kwargs):
    delay = 0.35
    for attempt in range(5):
        try:
            return call(*args, **kwargs)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(delay + random.uniform(0,0.25))
            delay = min(delay*2, 5.0)

def _summarize_regions_with_llm(glob: List[Dict], asia: List[Dict], indo: List[Dict]) -> Dict[str,str]:
    def _fmt(items):
        lines = []
        for it in items[:SUMMARY_ITEMS_PER_REGION]:
            lines.append(f"- [{it.get('sector','Unknown')}/{it.get('sentiment','Neutral')}] {it.get('headline','')}")
        return "\n".join(lines) if lines else "(no items)"
    prompt = (
        "You are an equity research assistant. Write concise, factual summaries (1–2 sentences each) "
        "for Global, Asia, and Indonesia **based only on** the bullet lists below. Do **not** invent facts.\n\n"
        "Return ONLY this JSON object (no extra text):\n"
        "{\n"
        '  "global": "…",\n'
        '  "asia": "…",\n'
        '  "indonesia": "…"\n'
        "}\n\n"
        f"Global bullets:\n{_fmt(glob)}\n\n"
        f"Asia bullets:\n{_fmt(asia)}\n\n"
        f"Indonesia bullets:\n{_fmt(indo)}\n"
    )
    try:
        r = _backoff(
            client.chat.completions.create,
            model=MODEL_REASON,
            messages=[{"role":"user","content": prompt}],
            response_format={"type":"json_object"},
            max_completion_tokens=350,
        )
        txt = (r.choices[0].message.content or "").strip()
        data = json.loads(txt) if txt else {}
        g = data.get("global") or _fallback_summary(glob, "Global")
        a = data.get("asia") or _fallback_summary(asia, "Asia")
        i = data.get("indonesia") or _fallback_summary(indo, "Indonesia")
        return {"global": g, "asia": a, "indonesia": i}
    except Exception:
        return {
            "global": _fallback_summary(glob, "Global"),
            "asia": _fallback_summary(asia, "Asia"),
            "indonesia": _fallback_summary(indo, "Indonesia"),
        }

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
            imp = f" — {e.get('impact','')}" if e.get("impact") else ""
            lines.append(f"- {e.get('event','')}{imp}")
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
                reg = it.get("region", "Global")
                snt = it.get("sentiment", "Neutral")
                src = it.get("source", "source")
                url = it.get("url", "")
                headline = it.get("headline", "")
                lines.append(f"- [{reg}] {headline} ({snt}) — [{src}]({url})")
        lines.append("")
    lines.append("")
    lines.append("## Watchlist Alerts")
    alerts = brief_json.get("watchlist_alerts", []) or []
    if alerts:
        for a in alerts:
            base = a.get("alert", "")
            ref = a.get("reference_url")
            if ref:
                lines.append(f"- {base} — [source]({ref})")
            else:
                lines.append(f"- {base}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Emerging Themes")
    themes = brief_json.get("emerging_themes", []) or []
    if themes:
        for t in themes:
            reg = f" [{t.get('region')}]" if t.get("region") else ""
            lines.append(f"- **{t.get('theme','')}**{reg}: {t.get('description','')}")
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
    # Flatten all items for summaries
    all_items = []
    for items in news_by_sector.values():
        all_items.extend(items)
    # LLM summaries with fallback
    g, a, i = _partition_by_region(all_items)
    ms = _summarize_regions_with_llm(g, a, i)

    # Deterministic JSON (only fetched items)
    brief_json = {
        "date": date,
        "market_summaries": ms,
        "economic_events": economic_events or [],
        "news_by_sector": {},
        "watchlist_alerts": watchlist_alerts or [],
        "emerging_themes": emerging_themes or [],
        "sentiment_indicators": sentiment_indicators or {},
    }

    for sector, items in news_by_sector.items():
        out = []
        for it in items:
            url = it.get("url", "")
            out.append({
                "headline":  it.get("headline", ""),
                "source":    it.get("source") or _host(url),
                "url":       url,
                "region":    it.get("region", "Global"),
                "sentiment": it.get("sentiment", "Neutral"),
                "priority":  it.get("priority", 0),
                "theme":     it.get("theme", "")
            })
        brief_json["news_by_sector"][sector] = out

    brief_md = _render_markdown(brief_json)
    return brief_json, brief_md
