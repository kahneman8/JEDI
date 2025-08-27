"""phase1_daily_brief/fetch_news.py"""
import json
import re
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from .config import (
    OPENAI_API_KEY,
    MODEL_UTILITY,  # use MODEL_UTILITY
    GLOBAL_QUERY,
    LOCAL_QUERY,
    SEARCH_MAX_RESULTS,
    FETCH_TIMEOUT_SEC,
    MAX_WORKERS,
    MAX_ARTICLES_TOTAL,
)

# Init OpenAI client (uses env var or explicit key)
client = OpenAI(api_key=OPENAI_API_KEY)


def _perform_search(query, max_results):
    results = []
    # Primary: Responses API + web_search tool
    try:
        resp = client.responses.create(
            model=MODEL_UTILITY,
            input=f"Find {max_results} latest headlines for: {query}",
            tools=[{"type": "web_search"}],
        )
        data = resp.model_dump() if hasattr(resp, "model_dump") else resp
        for out in (data.get("output") or []):
            for block in (out.get("content") or []):
                for ann in (block.get("annotations") or []):
                    if ann.get("type") == "url_citation":
                        url = ann.get("url") or ""
                        title = ann.get("title") or ""
                        if url:
                            results.append({"headline": title, "url": url})
        if results:
            return results[:max_results]
    except Exception:
        pass

    # Fallback: ask for JSON array of {headline,url}
    try:
        chat = client.chat.completions.create(
            model=MODEL_UTILITY,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Return a JSON array of objects with 'headline' and 'url' for the top "
                        f"{max_results} recent, reputable headlines about: {query}. "
                        "Only output JSON, no extra text."
                    ),
                }
            ],
        )
        txt = chat.choices[0].message.content
        s, e = txt.find("["), txt.rfind("]")
        if s != -1 and e != -1:
            results = json.loads(txt[s : e + 1])
    except Exception:
        results = []

    return results[:max_results]


def _fetch_article(item):
    """
    Fetch a single article URL. Extract paragraphs and trim content.
    Returns the same dict with added keys: content, source.
    """
    url = item.get("url")
    try:
        resp = requests.get(
            url,
            timeout=FETCH_TIMEOUT_SEC,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code != 200:
            item["content"] = ""
            return item
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        full_text = " ".join(paragraphs)
        item["content"] = full_text[:1000]  # Trim to first 1000 characters for efficiency
        item["source"] = _extract_source(url)
    except Exception:
        item["content"] = ""
    return item


def _extract_source(url: str) -> str:
    """
    Extract domain name as a human-readable source (e.g. reuters.com -> Reuters).
    """
    domain = re.sub(r"^https?://([^/]+)/.*", r"\1", url)
    domain = domain.replace("www.", "").split(".")[0]
    return domain.capitalize()


def search_and_retrieve_news(query: str) -> list:
    """
    Perform search and return a list of news articles with content.
    """
    raw_results = _perform_search(query, SEARCH_MAX_RESULTS)
    # Fetch article contents concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(_fetch_article, raw_results))
    return results


def dedupe_and_trim(items: list, limit: int) -> list:
    """
    Remove duplicate entries (by URL or headline) and trim to `limit` items.
    """
    seen = set()
    unique = []
    for item in items:
        key = (item.get("url", ""), (item.get("headline", "")).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def fetch_all_news() -> list:
    """
    Retrieve and combine global and local news, deduplicate and trim.
    """
    global_articles = search_and_retrieve_news(GLOBAL_QUERY)
    local_articles = search_and_retrieve_news(LOCAL_QUERY)
    all_items = global_articles + local_articles
    return dedupe_and_trim(all_items, limit=MAX_ARTICLES_TOTAL)
