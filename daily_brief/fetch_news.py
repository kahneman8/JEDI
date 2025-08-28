import json, re, requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from .config import (
    OPENAI_API_KEY, MODEL, GLOBAL_QUERY, LOCAL_QUERY, SEARCH_MAX_RESULTS,
    FETCH_TIMEOUT_SEC, MAX_WORKERS, MAX_ARTICLES_TOTAL, ASIA_HINTS, INDONESIA_HINTS
)

client = OpenAI(api_key=OPENAI_API_KEY)

def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host.split(":")[0].replace("www.", "")
    except Exception:
        return ""

def _extract_source(url: str) -> str:
    d = _domain(url)
    if not d: return "source"
    parts = d.split(".")
    core = parts[-2] if len(parts) >= 2 else parts[0]
    return core.capitalize()

def _perform_search(query, max_results):
    # Prefer web_search tool; fallback to JSON list
    results = []
    try:
        resp = client.responses.create(
            model=MODEL,
            input=f"Find {max_results} latest reputable headlines for: {query}",
            tools=[{"type": "web_search"}],
        )
        data = resp.model_dump() if hasattr(resp, "model_dump") else resp
        for out in (data.get("output") or []):
            for block in (out.get("content") or []):
                for ann in (block.get("annotations") or []):
                    if ann.get("type") == "url_citation":
                        url = ann.get("url") or ""
                        title = ann.get("title") or ""
                        if url and not _is_blacklisted(url):
                            results.append({"headline": title, "url": url})
        if results:
            return results[:max_results]
    except Exception:
        pass

    # Fallback: ask for JSON array; we'll verify by fetching content
    try:
        chat = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Return a JSON array of objects with 'headline' and 'url' for the top "
                    f"{max_results} reputable headlines about: {query}. Only JSON."
                ),
            }],
            max_completion_tokens=600,
        )
        txt = (chat.choices[0].message.content or "").strip()
        s, e = txt.find("["), txt.rfind("]")
        if s != -1 and e != -1:
            candidate = json.loads(txt[s:e+1])
            for it in candidate:
                url = it.get("url") or ""
                if url and not _is_blacklisted(url):
                    results.append({"headline": it.get("headline",""), "url": url})
    except Exception:
        pass

    return results[:max_results]

def _fetch_article(item):
    url = item.get("url")
    try:
        r = requests.get(url, timeout=FETCH_TIMEOUT_SEC, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200:
            item["content"] = ""
            return item
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        txt = " ".join(paragraphs)
        item["content"] = txt[:1200]
        item["source"]  = _extract_source(url)
    except Exception:
        item["content"] = ""
    return item

def _detect_region(headline: str, content: str, url: str, force_indonesia: bool) -> str:
    if force_indonesia:
        return "Indonesia"
    blob = f"{headline} {content}".lower()
    d = _domain(url)
    if any(h in blob for h in INDONESIA_HINTS) or d.endswith(".id"):
        return "Indonesia"
    if any(h in blob for h in ASIA_HINTS):
        return "Asia"
    return "Global"

def _dedupe(items, limit):
    seen, out = set(), []
    for it in items:
        key = (it.get("url",""), (it.get("headline","")).strip().lower())
        if key in seen: continue
        seen.add(key)
        out.append(it)
        if len(out) >= limit: break
    return out

def _search_and_retrieve(query: str, force_indonesia: bool) -> list:
    raw = _perform_search(query, SEARCH_MAX_RESULTS)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fetched = list(ex.map(_fetch_article, raw))
    # Keep only real, non-blacklisted, with content
    cleaned = []
    for it in fetched:
        url = it.get("url","")
        if not it.get("content"): continue
        if _is_blacklisted(url): continue
        it["region"] = _detect_region(it.get("headline",""), it.get("content",""), url, force_indonesia)
        cleaned.append(it)
    return cleaned

def fetch_all_news() -> list:
    global_items = _search_and_retrieve(GLOBAL_QUERY, force_indonesia=False)
    local_items  = _search_and_retrieve(LOCAL_QUERY,  force_indonesia=True)
    return _dedupe(global_items + local_items, MAX_ARTICLES_TOTAL)

