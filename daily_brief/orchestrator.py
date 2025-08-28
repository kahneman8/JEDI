import os, json, datetime
from jsonschema import validate, ValidationError

from .fetch_news import fetch_all_news, dedupe_and_trim
from .classify_sector import batch_assign_sector
from .analyze_sentiment import batch_assign_sentiment
from .detect_themes import check_curated_watchlist, find_dynamic_trends, find_emerging_themes
from .generate_brief import compose_and_generate
from .config import MAX_ARTICLES_TOTAL
from .utils_cache import get as cache_get, set as cache_set

def run_morning_brief():
    date_str = datetime.date.today().isoformat()
    print(f"[{date_str}] Generating morning brief…")

    # 1) Fetch news, dedupe, cap
    news_items = fetch_all_news()
    news_items = dedupe_and_trim(news_items, limit=MAX_ARTICLES_TOTAL)

    # 2) Prefill from cache
    for it in news_items:
        url = it.get("url")
        if not url: continue
        c = cache_get(url)
        if c.get("sector"): it["sector"] = c["sector"]
        if c.get("sentiment"): it["sentiment"] = c["sentiment"]

    # 3) Classify sectors
    batch_assign_sector(news_items)

    # 4) Sentiment
    batch_assign_sentiment(news_items)

    # 5) Group by sector & indicators
    news_by_sector = {}
    for it in news_items:
        sector = it.get("sector", "Unknown")
        news_by_sector.setdefault(sector, []).append(it)

    sentiment_indicators = {}
    for sector, items in news_by_sector.items():
        counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for it in items:
            s = it.get("sentiment", "Neutral")
            if s in counts: counts[s] += 1
        sentiment_indicators[sector] = counts

    # 6) Watchlists & themes
    curated_alerts = check_curated_watchlist(news_items)
    dynamic_alerts = find_dynamic_trends(news_items)
    watchlist_alerts = curated_alerts + dynamic_alerts
    themes = find_emerging_themes(news_items)

    # 7) Compose
    brief_json, brief_md = compose_and_generate(
        date=date_str,
        market_summaries={},
        economic_events=[],
        news_by_sector=news_by_sector,
        watchlist_alerts=watchlist_alerts,
        emerging_themes=themes,
        sentiment_indicators=sentiment_indicators
    )

    # 8) Validate
    schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
    schema = json.load(open(schema_path))
    try:
        validate(instance=brief_json, schema=schema)
        print("JSON validation succeeded.")
    except ValidationError as e:
        print("JSON validation failed:", e)

    # 9) Save & cache
    os.makedirs("outputs", exist_ok=True)
    json_file = f"outputs/{date_str}_brief.json"
    md_file = f"outputs/{date_str}_brief.md"
    with open(json_file, "w") as jf: json.dump(brief_json, jf, indent=2)
    with open(md_file, "w") as mf: mf.write(brief_md)
    print(f"Morning brief saved: {json_file}, {md_file}")

    for it in news_items:
        url = it.get("url")
        if not url: continue
        cache_set(url, {"sector": it.get("sector"), "sentiment": it.get("sentiment")})

if __name__ == "__main__":
    run_morning_brief()
