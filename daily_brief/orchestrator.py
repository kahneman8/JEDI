import os, json, datetime, re
from jsonschema import validate, ValidationError

from .fetch_news import fetch_all_news
from .classify_sector import batch_assign_sector
from .analyze_sentiment import batch_assign_sentiment
from .detect_themes import check_curated_watchlist, find_dynamic_trends, find_emerging_themes
from .generate_brief import compose_and_generate
from .config import MAX_ARTICLES_TOTAL

_URL_IN_PARENS_RE = re.compile(r"\((https?://[^\s)]+)\)\s*$", re.I)

def _alerts_to_objects(alerts: list) -> list:
    out = []
    for a in alerts:
        url = None
        m = _URL_IN_PARENS_RE.search(a or "")
        if m:
            url = m.group(1)
            a = _URL_IN_PARENS_RE.sub("", a).strip()
        out.append({
            "alert": a,
            "related_topics": [],
            "reference_url": url
        })
    return out

def _group_by_sector(items):
    by = {}
    for it in items:
        sec = it.get("sector","Unknown")
        by.setdefault(sec, []).append(it)
    return by

def _sentiment_counts(by_sector):
    out = {}
    for sec, items in by_sector.items():
        c = {"Positive":0,"Negative":0,"Neutral":0}
        for it in items:
            lab = it.get("sentiment","Neutral")
            if lab in c: c[lab] += 1
        out[sec] = c
    return out

def _infer_theme_region(theme_title: str, items: list) -> str:
    if not theme_title: return "Mixed"
    tokens = [t for t in re.findall(r"[A-Za-z]{4,}", theme_title.lower())]
    if not tokens: return "Mixed"
    counts = {"Global":0,"Asia":0,"Indonesia":0}
    for it in items:
        text = (it.get("headline","") + " " + it.get("content","")).lower()
        if any(t in text for t in tokens):
            counts[it.get("region","Global")] = counts.get(it.get("region","Global"),0) + 1
    if not any(counts.values()): return "Mixed"
    return max(counts, key=counts.get)

def _attach_theme_metadata(themes: list, items: list, max_related=5):
    enriched = []
    for t in themes or []:
        title = t.get("theme","")
        desc  = t.get("description","")
        region = _infer_theme_region(title, items)
        # related headlines
        toks = [w for w in re.findall(r"[A-Za-z]{4,}", title.lower())]
        related = []
        for it in items:
            h = (it.get("headline","") or "")
            if any(w in h.lower() for w in toks):
                related.append(h)
            if len(related) >= max_related: break
        enriched.append({
            "theme": title,
            "description": desc,
            "region": region,
            "priority": 1 if len(related) >= 3 else 0.5,
            "related_news": related
        })
    return enriched

def run_morning_brief():
    date_str = datetime.date.today().isoformat()
    print(f"[{date_str}] Generating morning brief…")

    # 1) Fetch
    items = fetch_all_news()

    # 2) Sector (can be LLM or offline)
    batch_assign_sector(items)

    # 3) Sentiment (offline module recommended)
    batch_assign_sentiment(items)

    # 4) Group & counts
    by_sector = _group_by_sector(items)
    sentiment_indicators = _sentiment_counts(by_sector)

    # 5) Alerts & themes
    curated_alerts = check_curated_watchlist(items)
    dynamic_alerts = find_dynamic_trends(items)
    alerts_obj = _alerts_to_objects(curated_alerts + dynamic_alerts)
    themes = _attach_theme_metadata(find_emerging_themes(items), items)

    # 6) Compose (deterministic JSON + Markdown; only uses fetched URLs)
    brief_json, brief_md = compose_and_generate(
        date=date_str,
        market_summaries={},
        economic_events=[],
        news_by_sector=by_sector,
        watchlist_alerts=alerts_obj,
        emerging_themes=themes,
        sentiment_indicators=sentiment_indicators
    )

    # 7) Validate
    schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
    schema = json.load(open(schema_path))
    validate(instance=brief_json, schema=schema)
    print("JSON validation succeeded.")

    # 8) Save
    os.makedirs("outputs", exist_ok=True)
    jf = f"outputs/{date_str}_brief.json"
    mf = f"outputs/{date_str}_brief.md"
    with open(jf,"w") as f: json.dump(brief_json, f, indent=2)
    with open(mf,"w") as f: f.write(brief_md)
    print(f"Morning brief saved: {jf}, {mf}")

if __name__ == "__main__":
    run_morning_brief()
