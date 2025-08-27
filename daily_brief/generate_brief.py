"""daily_brief/generate_brief.py

Builds the Morning Brief JSON deterministically, and renders Markdown in code.
Optionally uses an LLM to write short summaries; if the LLM call fails/empty,
we fall back to a simple heuristic. No JSON-from-LLM required.
"""
import os
import re
import json
from datetime import date
from urllib.parse import urlparse

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # allows running without SDK for local tests

from .config import (
    OPENAI_API_KEY,
    MODEL_COMPOSE,          # can be "gpt-4o" or whatever your org has
    MAX_OUTPUT_TOKENS,      # not strictly needed here, but kept for clarity
)

# --------- helpers ---------
def _hostname(u: str) -> str:
    try:
        netloc = urlparse(u).netloc
        host = netloc.replace("www.", "").split(":")[0]
        return host or "source"
    except Exception:
        return "source"

def _join_bullets(lines):
    return "".join(f"- {ln}\n" for ln in lines)

# crude regex inclusion helpers
_ID_WORDS = re.compile(r"\b(Indonesia|Indonesian|Jakarta|JCI|IDX|Bank Indonesia)\b", re.I)
_ASIA_WORDS = re.compile(r"\b(Asia|China|Japan|Korea|Taiwan|Hong|Nikkei|Hang|Kospi|SGX)\b", re.I)

def _is_indonesia_item(item):
    h = (item.get("headline") or "") + " " + (item.get("content") or "")
    u = item.get("url") or ""
    return (".id" in u) or bool(_ID_WORDS.search(h))

def _is_asia_item(item):
    # not Indonesia, but Asia keywords
    h = (item.get("headline") or "") + " " + (item.get("content") or "")
    return bool(_ASIA_WORDS.search(h)) and not _is_indonesia_item(item)

def _summ_text_from_items(items, default=""):
    # Join top headlines to a short prompt context
    heads = [f"- {it.get('headline','')}" for it in items[:8] if it.get("headline")]
    return "\n".join(heads) or default

def _llm_summarize(context_text: str, model_name: str) -> str:
    """Try a short LLM summary; return '' on failure so caller can fallback."""
    if not OpenAI or not OPENAI_API_KEY or not context_text.strip():
        return ""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system",
                 "content": ("You are an equity research assistant. "
                             "Write a concise 1–2 sentence summary in English. No lists, no markdown.")},
                {"role": "user", "content": context_text}
            ],
            max_completion_tokens=180,
        )
        txt = (resp.choices[0].message.content or "").strip()
        # sanitize for stray whitespace
        return " ".join(txt.split())
    except Exception:
        return ""

def _fallback_summary(items, label):
    # simple heuristic summary using counts
    pos = sum(1 for it in items if it.get("sentiment") == "Positive")
    neg = sum(1 for it in items if it.get("sentiment") == "Negative")
    neu = sum(1 for it in items if it.get("sentiment") == "Neutral")
    total = max(len(items), 1)
    return (f"{label} headlines today: {len(items)} items "
            f"(Positive {pos}, Negative {neg}, Neutral {neu}).")

# --------- main API ---------
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
    Returns (brief_json: dict, brief_md: str).
    JSON is assembled deterministically from inputs; we only *optionally*
    call a model to write short summaries. Markdown is rendered in code.
    """
    # 1) Prepare market summaries (LLM optional)
    # Split items into global/asia/indonesia buckets from news_by_sector
    all_items = []
    for sect_items in news_by_sector.values():
        all_items.extend(sect_items)

    indo_items = [it for it in all_items if _is_indonesia_item(it)]
    asia_items = [it for it in all_items if _is_asia_item(it)]
    # global = everything (brief trend)
    global_items = all_items

    # Use existing summaries if provided; otherwise try LLM then fallback
    global_sum = (market_summaries or {}).get("global", "")
    asia_sum = (market_summaries or {}).get("asia", "")
    indo_sum = (market_summaries or {}).get("indonesia", "")

    if not global_sum:
        ctx = _summ_text_from_items(global_items, "Global market headlines.")
        global_sum = _llm_summarize(ctx, MODEL_COMPOSE) or _fallback_summary(global_items, "Global")

    if not asia_sum:
        ctx = _summ_text_from_items(asia_items, "Asia market headlines.")
        asia_sum = _llm_summarize(ctx, MODEL_COMPOSE) or _fallback_summary(asia_items, "Asia")

    if not indo_sum:
        ctx = _summ_text_from_items(indo_items, "Indonesia market headlines.")
        indo_sum = _llm_summarize(ctx, MODEL_COMPOSE) or _fallback_summary(indo_items, "Indonesia")

    # 2) Assemble JSON deterministically (matches schema fields)
    brief_json = {
        "date": date,
        "market_summaries": {
            "global": global_sum,
            "asia": asia_sum,
            "indonesia": indo_sum,
        },
        "economic_events": economic_events or [],   # keep empty list if none
        "news_by_sector": {},                      # fill below
        "watchlist_alerts": watchlist_alerts or [],
        "emerging_themes": emerging_themes or [],
        "sentiment_indicators": sentiment_indicators or {},
    }

    # Normalise sector blocks
    for sector, items in sorted(news_by_sector.items()):
        blocks = []
        for it in items:
            headline = it.get("headline", "")
            sentiment = it.get("sentiment", "Neutral")
            url = it.get("url", "")
            source = it.get("source") or _hostname(url)
            blocks.append({
                "headline": headline,
                "sentiment": sentiment,
                "source": source,
                "url": url
            })
        brief_json["news_by_sector"][sector] = blocks

    # 3) Render Markdown in code (no LLM)
    md_lines = []
    md_lines.append(f"# Morning Market Brief — {date}")
    md_lines.append("")
    md_lines.append("## Market Summaries")
    md_lines.append(f"- **Global:** {brief_json['market_summaries']['global']}")
    md_lines.append(f"- **Asia:** {brief_json['market_summaries']['asia']}")
    md_lines.append(f"- **Indonesia:** {brief_json['market_summaries']['indonesia']}")
    md_lines.append("")

    md_lines.append("## Economic Events")
    if brief_json["economic_events"]:
        for ev in brief_json["economic_events"]:
            ev_text = f"{ev.get('event','')}"
            imp = ev.get("impact")
            if imp:
                ev_text += f" — {imp}"
            md_lines.append(f"- {ev_text}")
    else:
        md_lines.append("- None")
    md_lines.append("")

    md_lines.append("## News by Sector")
    for sector, items in brief_json["news_by_sector"].items():
        md_lines.append(f"### {sector}")
        if not items:
            md_lines.append("- None")
        else:
            for it in items:
                h = it["headline"]
                s = it["sentiment"]
                url = it["url"]
                src = it["source"]
                # include explicit URL for citation
                md_lines.append(f"- {h} ({s}) — [{src}]({url})")
        md_lines.append("")
    md_lines.append("")

    md_lines.append("## Watchlist Alerts")
    if brief_json["watchlist_alerts"]:
        for a in brief_json["watchlist_alerts"]:
            md_lines.append(f"- {a}")
    else:
        md_lines.append("- None")
    md_lines.append("")

    md_lines.append("## Emerging Themes")
    if brief_json["emerging_themes"]:
        for t in brief_json["emerging_themes"]:
            md_lines.append(f"- **{t.get('theme','')}**: {t.get('description','')}")
    else:
        md_lines.append("- None")
    md_lines.append("")

    brief_md = "\n".join(md_lines)
    return brief_json, brief_md
