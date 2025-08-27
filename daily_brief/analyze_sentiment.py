"""daily_brief/analyze_sentiment.py"""
import json
from openai import OpenAI
from .config import MODEL, TEMPERATURE, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
ALLOWED = {"Positive", "Negative", "Neutral"}


def _single_sentiment(text: str) -> str:
    prompt = (
        f'Text: "{text}"\n'
        "What is the sentiment of this news for markets? "
        "Respond with one of: Positive, Negative, or Neutral."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = (resp.choices[0].message.content or "").strip().capitalize()
    if "Positive" in raw:
        return "Positive"
    if "Negative" in raw:
        return "Negative"
    return "Neutral"


def batch_assign_sentiment(items: list) -> None:
    if not items:
        return

    lines = [f"{i+1}. {it.get('content', it.get('headline',''))}" for i, it in enumerate(items)]
    prompt = (
        "Classify each item’s sentiment as Positive, Negative, or Neutral. "
        "Return JSON list: [{\"i\": <index>, \"sentiment\": \"Positive|Negative|Neutral\"}].\n\n"
        + "\n".join(lines)
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        payload = (resp.choices[0].message.content or "").strip()
        mapping = json.loads(payload)
        for entry in mapping:
            idx = int(entry.get("i", 0)) - 1
            sentiment = entry.get("sentiment", "Neutral").capitalize()
            if 0 <= idx < len(items):
                items[idx]["sentiment"] = sentiment if sentiment in ALLOWED else "Neutral"
    except Exception:
        for it in items:
            it["sentiment"] = _single_sentiment(it.get("content", it.get("headline", "")))
