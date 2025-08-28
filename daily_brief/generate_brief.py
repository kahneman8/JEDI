"""LLM-grounded theme extraction with explicit grounding:
- Pass only fetched items (indexed) as context.
- Ask for JSON object: {"themes":[{theme, description, support:[indices]}]}
- Post-process to attach region/priority/related_news and drop ungrounded themes.
- Falls back to a simple trending-term heuristic if LLM fails.
"""
import json, re, time, random
from collections import Counter
from typing import List, Dict
from openai import OpenAI
from .config import OPENAI_API_KEY, MODEL_REASON, THEMES_MAX

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Optional curated watchlist ---
try:
    with open("daily_brief/data/watchlist_curated.json") as f:
        WATCHLIST_CURATED = json.load(f)
except Exception:
    WATCHLIST_CURATED = []

def check_curated_watchlist(items: List[Dict]) -> List[str]:
    alerts = []
    for kw in WATCHLIST_CURATED:
        kwl = kw.lower()
        hits = [it for it in items if kwl in (it.get("headline","")+ " " + it.get("content","")).lower()]
        if hits:
            url = hits[0].get("url","")
            if len(hits) == 1:
                alerts.append(f"{kw}: {hits[0].get('headline','')} ({url})")
            else:
                alerts.append(f"{kw}: Mentioned in {len(hits)} stories ({url})")
    return alerts

def find_dynamic_trends(items: List[Dict], top_n: int = 3) -> List[str]:
    text = " ".join(it.get("headline","") for it in items)
    words = re.findall(r"\b[A-Z][a-z]{3,}\b", text)
    freq = Counter(words)
    curated = {w.lower() for w in WATCHLIST_CURATED}
    common = {"The","This","That","Market","Global","Today"}
    trending = [w for w,c in freq.most_common(12) if c>1 and w.lower() not in curated and w not in common]
    out = []
    for term in trending[:top_n]:
        url = next((it.get("url","") for it in items if term in it.get("headline","")), "")
        out.append(f"{term}: Trending in news (mentioned {freq[term]} times) ({url})")
    return out

# --- LLM helpers ---
def _backoff(call, *args, **kwargs):
    delay = 0.35
    for attempt in range(6):
        try:
            return call(*args, **kwargs)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(delay + random.uniform(0,0.25))
            delay = min(delay*2, 6.0)

def _majority_region(indices: List[int], idx2item: Dict[int, Dict]) -> str:
    counts = Counter(idx2item.get(i,{}).get("region","Global") for i in indices if i in idx2item)
    if not counts:
        return "Mixed"
    region, _ = counts.most_common(1)[0]
    return region or "Mixed"

def _related_from_support(indices: List[int], idx2item: Dict[int, Dict], max_related=5) -> List[str]:
    titles = []
    for i in indices:
        it = idx2item.get(i)
        if not it:
            continue
        h = it.get("headline","")
        if h:
            titles.append(h)
        if len(titles) >= max_related:
            break
    return titles

def find_emerging_themes(items: List[Dict], max_themes: int = None) -> List[Dict]:
    """Return enriched themes: [{theme, description, region, priority, related_news}]"""
    if not items:
        return []
    max_themes = max_themes or THEMES_MAX

    # Build compact, indexed context
    # Use 1-based indices for user-friendliness
    lines, idx2item = [], {}
    for i, it in enumerate(items, start=1):
        idx2item[i] = it
        lines.append(f"{i}. [{it.get('region','Global')}] {it.get('headline','')} ({it.get('sector','Unknown')}, {it.get('sentiment','Neutral')})")

    prompt = (
        "You are an equity research assistant. From the following indexed headlines, propose up to "
        f"{max_themes} emerging themes that appear across multiple items. "
        "Use only the provided headlines; do not invent facts or URLs.\n\n"
        "Return ONLY a JSON object with this schema:\n"
        "{\n"
        '  "themes": [\n'
        '    {\n'
        '      "theme": "short title",\n'
        '      "description": "one-sentence explanation",\n'
        '      "support": [<index>, <index>]   // 2-5 indices from the list that justify the theme\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Indexed headlines:\n" + "\n".join(lines)
    )

    # Try JSON-mode; fallback: basic heuristic if it fails
    try:
        resp = _backoff(
            client.chat.completions.create,
            model=MODEL_REASON,
            messages=[{"role":"user","content": prompt}],
            response_format={"type":"json_object"},
            max_completion_tokens=600,
        )
        txt = (resp.choices[0].message.content or "").strip()
        data = json.loads(txt) if txt else {"themes":[]}
        out = []
        for t in data.get("themes", []):
            support = [int(x) for x in t.get("support", []) if isinstance(x, int) and x in idx2item]
            if not support:
                continue
            region = _majority_region(support, idx2item)
            related = _related_from_support(support, idx2item)
            priority = 1.0 if len(support) >= 4 else 0.7 if len(support) >= 3 else 0.5
            out.append({
                "theme": t.get("theme","").strip()[:140],
                "description": t.get("description","").strip(),
                "region": region,
                "priority": priority,
                "related_news": related
            })
        if out:
            return out[:max_themes]
    except Exception:
        pass

    # Fallback: trending terms -> themes
    trends = find_dynamic_trends(items, top_n=max_themes)
    themes = []
    for tr in trends:
        term = tr.split(":")[0]
        # support by simple containment
        support_idx = [i for i,it in idx2item.items() if term in it.get("headline","")]
        region = _majority_region(support_idx, idx2item)
        related = _related_from_support(support_idx, idx2item)
        themes.append({
            "theme": term,
            "description": f"Multiple headlines reference {term}.",
            "region": region,
            "priority": 0.5,
            "related_news": related
        })
    return themes[:max_themes]
