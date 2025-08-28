import os, re, json
from collections import Counter
from openai import OpenAI
from .config import OPENAI_API_KEY, MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

# Load curated watchlist
try:
    with open(os.path.join(os.path.dirname(__file__), "data", "watchlist_curated.json")) as f:
        WATCHLIST_CURATED = json.load(f)
except FileNotFoundError:
    WATCHLIST_CURATED = []

def check_curated_watchlist(items: list) -> list:
    alerts = []
    for kw in WATCHLIST_CURATED:
        kwl = kw.lower()
        matches = [
            it for it in items
            if kwl in (it.get("headline","").lower() + " " + it.get("content","").lower())
        ]
        if matches:
            url = matches[0].get("url", "")
            if len(matches) == 1:
                alerts.append(f"{kw}: {matches[0]['headline']} ({url})")
            else:
                alerts.append(f"{kw}: Mentioned in {len(matches)} stories ({url})")
    return alerts

def find_dynamic_trends(items: list, top_n: int = 3) -> list:
    text_corpus = " ".join(it.get("headline","") for it in items)
    words = re.findall(r"\b[A-Z][a-z]{3,}\b", text_corpus)
    freq = Counter(words)
    curated_set = {w.lower() for w in WATCHLIST_CURATED}
    common = {"The","This","That","Market","Global","Today"}
    trending = [w for w,c in freq.most_common(10) if c>1 and w.lower() not in curated_set and w not in common]
    alerts = []
    for term in trending[:top_n]:
        url = next((it.get("url","") for it in items if term in it.get("headline","")), "")
        alerts.append(f"{term}: Trending in news (mentioned {freq[term]} times) ({url})")
    return alerts

def find_emerging_themes(items: list, max_themes: int = 3) -> list:
    if not items:
        return []
    headlines = "\n".join(f"- {it.get('headline','')}" for it in items[:20])
    prompt = (
        "Today's news headlines:\n"
        f"{headlines}\n\n"
        "Identify up to three emerging themes supported by multiple headlines.\n"
        "Return JSON array: [{\"theme\": \"<title>\", \"description\": \"<one sentence>\"}]."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content": prompt}],
            max_completion_tokens=400,
        )
        txt = (resp.choices[0].message.content or "").strip()
        s, e = txt.find("["), txt.rfind("]")
        return json.loads(txt[s:e+1]) if s != -1 and e != -1 else []
    except Exception:
        return []
