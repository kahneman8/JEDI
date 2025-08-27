import json, datetime, os
from jsonschema import validate, ValidationError
from .fetch_news import fetch_all_news
from .classify_sector import batch_assign_sector
from .analyze_sentiment import batch_assign_sentiment
from .detect_themes import check_curated_watchlist, find_dynamic_trends, find_emerging_themes
from .generate_brief import compose_and_generate
from .config import GICS_SECTORS

def run_morning_brief():
 """
 Main entry point for the daily pipeline. Fetches news, classifies
 sectors, analyses sentiment, computes watchlists & themes, generates report,
 validates JSON and saves outputs to disk.
 """
 date_str = datetime.date.today().isoformat()
 print(f"[{date_str}] Generating morning brief…")
 # 1. Fetch news
 news_items = fetch_all_news()
 # 2. Classify sectors
 batch_assign_sector(news_items)
 # 3. Analyse sentiment
 batch_assign_sentiment(news_items)
 # 4. Group by sector
 news_by_sector = {}
 for it in news_items:
  sector = it.get("sector", "Unknown")
  news_by_sector.setdefault(sector, []).append(it)
 # 5. Compute sentiment indicators per sector
  sentiment_indicators = {}
 for sector, items in news_by_sector.items():
  counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
  for it in items:
   sentiment = it.get("sentiment", "Neutral")
   if sentiment in counts:
    counts[sentiment] += 1
  sentiment_indicators[sector] = counts
 # 6. Watchlist alerts
 curated_alerts = check_curated_watchlist(news_items)
 dynamic_alerts = find_dynamic_trends(news_items)
 watchlist_alerts = curated_alerts + dynamic_alerts
 # 7. Emerging themes
 themes = find_emerging_themes(news_items)
 # 8. Generate brief
 # Market summaries and economic events could be filled by the model; pass empty
 brief_json, brief_md = compose_and_generate(
  date=date_str,
  market_summaries={},
  economic_events=[],
  news_by_sector=news_by_sector,
  watchlist_alerts=watchlist_alerts,
  emerging_themes=themes,
  sentiment_indicators=sentiment_indicators
 )
 # 9. Validate JSON against schema
 schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
 schema = json.load(open(schema_path))
 try:
  validate(instance=brief_json, schema=schema)
  print("JSON validation succeeded.")
 except ValidationError as e:
  print("JSON validation failed:", e)
 # Optionally, handle or raise the error
 # 10. Save outputs
 os.makedirs("outputs", exist_ok=True)
 json_file = f"outputs/{date_str}_brief.json"
 md_file = f"outputs/{date_str}_brief.md"
 with open(json_file, "w") as jf:
  json.dump(brief_json, jf, indent=2)
 with open(md_file, "w") as mf:
  mf.write(brief_md)
 print(f"Morning brief saved: {json_file}, {md_file}")
if __name__ == "__main__":
 run_morning_brief()


