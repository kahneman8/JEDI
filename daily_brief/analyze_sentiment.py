"""phase1_daily_brief/analyze_sentiment.py"""
import json, openai
from .config import MODEL, TEMPERATURE
def _single_sentiment(text: str) -> str:
 """
 Fallback sentiment for a single text. Asks GPT-5 directly.
 """
 prompt = (
 f"Text: \"{text}\"\n"
 "What is the sentiment of this news for markets? "
 "Respond with one of: Positive, Negative, or Neutral."
 )
 resp = openai.ChatCompletion.create(
 model=MODEL,
 messages=[{"role": "user", "content": prompt}],
 temperature=TEMPERATURE
 )
 raw = resp.choices[0].message["content"].strip().capitalize()
 if "Positive" in raw:
 return "Positive"
 if "Negative" in raw:
 return "Negative"
 return "Neutral"
def batch_assign_sentiment(items: list) -> None:
 """
 Assign a sentiment label to each item. Sends multiple entries in one call.
 """
 if not items:
 return
 lines = [f"{i+1}. {it.get('content', it['headline'])}" for i, it in enumerate(items)]
 prompt = (
 "Classify each item’s sentiment as Positive, Negative, or Neutral. "
 "Return JSON list: [{{\"i\": <index>, \"sentiment\": \"Positive|Negative|Neutral\"}}, …].
\n\n"
 + "\n".join(lines)
 )
 try:
 resp = openai.ChatCompletion.create(
 model=MODEL,
 messages=[{"role": "user", "content": prompt}],
 temperature=TEMPERATURE
 )
 mapping = json.loads(resp.choices[0].message["content"])
 for entry in mapping:
 idx = entry.get("i") - 1
 sentiment = entry.get("sentiment", "Neutral").capitalize()
 if 0 <= idx < len(items):
 items[idx]["sentiment"] = sentiment
 except Exception:
 # Fallback: classify each item individually
 for it in items:
 it["sentiment"] = _single_sentiment(it.get("content", it.get("headline", "")))
