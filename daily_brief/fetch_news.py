"""phase1_daily_brief/fetch_news.py"""
import re, requests, hashlib
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import openai

from .config import (
    MODEL, OPENAI_API_KEY, GLOBAL_QUERY, LOCAL_QUERY,
    SEARCH_MAX_RESULTS, FETCH_TIMEOUT_SEC, MAX_WORKERS, MAX_ARTICLES_TOTAL
)

openai.api_key = OPENAI_API_KEY


def _perform_search(query, max_results):
    """
    Use GPT-5 Pro with the web_search tool to find news URLs.
    Returns a list of dicts with keys: headline, url.
    """
    response = openai.Response.create(
        model=MODEL,
        input=f"Find {max_results} latest headlines for: {query}",
        tools=[{"type": "web_search"}]
    )
    results = []
    # Parse annotations containing url_citations
    for out in getattr(response, "output", []):
        content = getattr(out, "content", None)
        if not content:
            continue
        for item in content:
            for ann in item.get("annotations", []):
                if ann.get("type") == "url_citation":
                    results.append({
                        "headline": ann.get("title", ""),
                        "url": ann.get("url", "")
                    })
    # Limit the number of results
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
            headers={"User-Agent": "Mozilla/5.0"}
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
