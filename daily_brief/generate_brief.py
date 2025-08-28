# daily_brief/generate_brief.py
"""
Deterministic compose (no LLM calls):
- Builds JSON strictly from fetched items (no fabricated URLs)
- Preserves region tags per item (Global | Asia | Indonesia)
- Passes through enriched themes/alerts
- Renders Markdown locally to avoid 429s
The orchestrator validates the JSON against schema.json.
"""

import json
from urllib.parse import urlparse


def _host(url: str) -> str:
    try:
        h = urlparse(url).netloc.replace("www.", "").split(":")[0]
        return h or "source"
    except Exception:
        return "source"


def _fallback_summary(items, label: str) -> str:
    pos = sum(1 for it in items if it.get("sentiment") == "Positive")
    neg = sum(1 for it in items if it.get("sentiment") == "Negative")
    neu = sum(1 for it in items if it.get("sentiment") == "Neutral")
    return f"{label} headlines: {len(items)} items (Positive {pos}, Negative {neg}, Neutral {neu})."


def _partition_by_region(all_items):
    indo = [it for it in all_items if it.get("region") == "Indonesia"]
    asia = [it for it in all_items if it.get("region") == "Asia"]
    glob = [it for it in all_items if it.get("region") == "Global"]
    return glob, asia, indo


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

    # Deterministic summaries (no model)
    g, a, i = _partition_by_region(all_items)
    ms = {
        "global":    (market_summaries or {}).get("global")    or _fallback_summary(g, "Global"),
        "asia":      (market_summaries or {}).get("asia")      or _fallback_summary(a, "Asia"),
        "indonesia": (market_summaries or {}).get("indonesia") or _fallback_summary(i, "Indonesia"),
    }

    # Build JSON strictly from fetched items (preserve region & real URLs)
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

    # Render Markdown locally
    brief_md = _render_markdown(brief_json)
    return brief_json, brief_md
