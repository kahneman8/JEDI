"""phase1_daily_brief/detect_themes.py"""
import re, json, openai
from collections import Counter
from .config import (MODEL, TEMPERATURE)

# Load the curated watchlist from a JSON file in data/
try:
    with open("data/watchlist_curated.json") as f:
        WATCHLIST_CURATED = json.load(f)
except FileNotFoundError:
    WATCHLIST_CURATED = []


def check_curated_watchlist(items: list) -> list:
    """
    Scan each news item for curated keywords. If found, create an alert with a URL.
    """
    alerts = []
    for kw in WATCHLIST_CURATED:
        kw_lower = kw.lower()
        matches = [
            it for it in items
            if kw_lower in it.get("headline", "").lower()
            or kw_lower in it.get("content", "").lower()
        ]
        if matches:
            # If multiple matches, summarise the number of stories
            url = matches[0].get("url", "")
            if len(matches) == 1:
                alerts.append(f"{kw}: {matches[0]['headline']} ({url})")
            else:
                alerts.append(f"{kw}: Mentioned in {len(matches)} stories ({url})")
    return alerts


def find_dynamic_trends(items: list, top_n: int = 3) -> list:
    """
    Identify trending terms across headlines by counting capitalised words. Exclude
    words already in the curated list or common stopwords. Attach a representative URL.
    """
    text_corpus = " ".join(it.get("headline", "") for it in items)
    words = re.findall(r"\b[A-Z][a-z]{3,}\b", text_corpus)  # heuristic: capitalised words
    freq = Counter(words)
    curated_set = {w.lower() for w in WATCHLIST_CURATED}
    common_words = {"The", "This", "That", "Market", "Global", "Today"}
    trending = [
        w for w, c in freq.most_common(10)
        if c > 1 and w.lower() not in curated_set and w not in common_words
    ]
    alerts = []
    for term in trending[:top_n]:
        # find the first article mentioning this term
        url = next((it.get("url", "") for it in items if term in it.get("headline", "")), "")
        alerts.append(f"{term}: Trending in news (mentioned {freq[term]} times) ({url})")
    return alerts


def find_emerging_themes(items: list, max_themes: int = 3) -> list:
    """
    Use GPT-5 to propose 1–3 emerging themes from the list of headlines. Returns a
    list of dicts: {"theme": <title>, "description": <one sentence>}.
    """
    if not items:
        return []
    headlines_text = "\n".join(f"- {it.get('headline', '')}" for it in items[:20])
    prompt = (
        "Today's news headlines:\n"
        f"{headlines_text}\n\n"
        "Identify up to three key emerging themes or trends from these headlines.\n"
        "For each theme, provide a short title and a one-sentence description.\n"
        "Return JSON: [{\"theme\": <title>, \"description\": <sentence>}]."
    )
    try:
        resp = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        themes = json.loads(resp.choices[0].message["content"])
        return themes[:max_themes]
    except Exception:
        return []
