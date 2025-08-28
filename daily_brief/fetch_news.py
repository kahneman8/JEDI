import json, re, time, requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from .config import (
    OPENAI_API_KEY, MODEL, GLOBAL_QUERY, LOCAL_QUERY, SEARCH_MAX_RESULTS,
    FETCH_TIMEOUT_SEC, MAX_WORKERS, MAX_ARTICLES_TOTAL, BLACKLIST_DOMAINS,
    ASIA_HINTS, INDONESIA_HINTS,
    MIN_CONTENT_CHARS_GLOBAL, MIN_CONTENT_CHARS_ID,
    SEED_URLS_GLOBAL, SEED_URLS_INDONESIA
)

client = OpenAI(api_key=OPENAI_API_KEY)

def _log(msg: str):
    print(f"[fetch_news] {msg}")

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

def _is_blacklisted(url: str) -> bool:
    d = _domain(url)
    return any(d.endswith(bad) for bad in BLACKLIST_DOMAINS)

def _perform_search(label: str, query: str, max_results: int):
    _log(f"{label}: web_search start")
    results = []
    try:
        resp = client.responses.create(
            model=MODEL,
            input=f"Give {max_results} recent reputable headlines for: {query}",
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
    except Exception as e:
        _log(f"{label}: web_search error -> {type(e).__name__}")

    if results:
        _log(f"{label}: web_search results={len(results)}")
        return results[:max_results]

    # Fallback JSON
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
        _log(f"{label}: fallback JSON results={len(results)}")
    except Exception as e:
        _log(f"{label}: fallback JSON error -> {type(e).__name__}")

    return results[:max_results]

def _best_text_from_html(soup: BeautifulSoup) -> str:
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(paras).strip()
    if text:
        return text[:2000]
    # meta fallbacks
    for selector in [
        ("meta", {"name": "description"}),
        ("meta", {"property": "og:description"}),
        ("meta", {"name": "twitter:description"}),
    ]:
        tag = soup.find(*selector)
        if tag and tag.get("content"):
            return tag["content"].strip()[:400]
    title = soup.find("title")
    if title and title.get_text(strip=True):
        return title.get_text(strip=True)[:200]
    return ""

def _fetch_article(item):
    url = item.get("url")
    try:
        r = requests.get(
            url,
            timeout=FETCH_TIMEOUT_SEC,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.8",
            },
        )
        if r.status_code != 200:
            item["content"] = ""
            return item
        soup = BeautifulSoup(r.text, "html.parser")
        text = _best_text_from_html(soup)
        item["content"] = (text or "").strip()
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

def _scrape_seed_pages(seed_urls):
    out = []
    for base in seed_urls or []:
        try:
            r = requests.get(
                base, timeout=FETCH_TIMEOUT_SEC,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
            )
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            anchors = soup.find_all("a", href=True)
            cand = []
            for a in anchors:
                txt = (a.get_text(" ", strip=True) or "").strip()
                href = a["href"]
                if not txt or len(txt) < 24:
                    continue
                if href.startswith("/"):
                    href = urljoin(base, href)
                if not href.startswith("http"):
                    continue
                cand.append({"headline": txt, "url": href})
            seen = set()
            for c in cand:
                if c["url"] in seen: continue
                seen.add(c["url"])
                out.append(c)
                if len(out) >= 25:
                    break
        except Exception:
            continue
    _log(f"seed scrape links={len(out)} from {len(seed_urls or [])} pages")
    return out

def _search_and_retrieve(label: str, query: str, force_indonesia: bool, seeds=None) -> list:
    raw = _perform_search(label, query, SEARCH_MAX_RESULTS)
    if not raw and seeds:
        _log(f"{label}: using seeds fallback")
        raw = _scrape_seed_pages(seeds)

    if not raw:
        _log(f"{label}: no URLs")
        return []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fetched = list(ex.map(_fetch_article, raw))

    kept, dropped = [], 0
    min_chars = MIN_CONTENT_CHARS_ID if force_indonesia else MIN_CONTENT_CHARS_GLOBAL
    for it in fetched:
        url = it.get("url","")
        if _is_blacklisted(url):
            dropped += 1
            continue
        text = (it.get("content") or "").strip()
        if len(text) < min_chars:
            dropped += 1
            continue
        it["region"] = _detect_region(it.get("headline",""), text, url, force_indonesia)
        kept.append(it)

    _log(f"{label}: fetched={len(fetched)} kept={len(kept)} dropped={dropped} (min_chars={min_chars})")
    return kept

def fetch_all_news() -> list:
    # GLOBAL / ASIA pass
    global_items = _search_and_retrieve("GLOBAL/ASIA", GLOBAL_QUERY, force_indonesia=False, seeds=SEED_URLS_GLOBAL)

    # short pause to ensure the Indonesia search appears as a separate API call window
    time.sleep(1.0)

    # INDONESIA pass (force region + lower threshold + seeds)
    local_items  = _search_and_retrieve("INDONESIA", LOCAL_QUERY, force_indonesia=True, seeds=SEED_URLS_INDONESIA)

    all_items = _dedupe(global_items + local_items, MAX_ARTICLES_TOTAL)
    _log(f"TOTAL after dedupe: {len(all_items)} (global/asia={len(global_items)}, indonesia={len(local_items)})")
    return all_items
