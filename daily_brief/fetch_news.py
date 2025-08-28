import os
import sys
import json
import time
import requests
from urllib.parse import urlparse, urljoin, urlunparse, parse_qsl, urlencode
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from datetime import datetime
import openai

# Initialize OpenAI client (API key is expected in environment)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[fetch_news] ERROR: OpenAI API key not set in environment", file=sys.stderr)
    sys.exit(1)
# Use the OpenAI Responses API client for tool use (web_search)
from openai import OpenAI
client = OpenAI(api_key=api_key)

# Logging helper
def log(message: str):
    """Print a log message with [fetch_news] prefix."""
    print(f"[fetch_news] {message}")

# Utility: normalize domain from URL
def get_domain(url: str) -> str:
    """Return the base domain (without subdomains or port) from a URL."""
    try:
        host = urlparse(url).netloc.lower()
        return host.split(":")[0].replace("www.", "")
    except Exception:
        return ""

# Utility: short source name from URL
def extract_source_name(url: str) -> str:
    """Extract a short source name from a URL (e.g., 'ft.com' -> 'Ft')."""
    domain = get_domain(url)
    if not domain:
        return "Source"
    parts = domain.split(".")
    core = parts[-2] if len(parts) >= 2 else parts[0]
    return core.capitalize()

# Utility: remove tracking parameters (like utm_ queries) from URL
def strip_tracking_params(url: str) -> str:
    """Remove common tracking query parameters (utm_*) from a URL."""
    try:
        parsed = urlparse(url)
        # Filter out any query params that start with "utm_"
        clean_qs = [(k, v) for (k, v) in parse_qsl(parsed.query, keep_blank_values=True) 
                    if not k.lower().startswith("utm_")]
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params,
                            urlencode(clean_qs, doseq=True), parsed.fragment))
    except Exception:
        return url

# Set default search parameters and thresholds
MAX_RESULTS_PER_QUERY = 8  # max headlines to request per search query
MAX_THREADS = 4            # for parallel content fetch
# Minimum content length (characters) to keep article (global vs local regions)
MIN_CONTENT_CHARS_GLOBAL = 100
MIN_CONTENT_CHARS_LOCAL = 50
# Domains to skip entirely (non-news or unwanted sources)
BLACKLIST_DOMAINS = {"twitter.com", "facebook.com", "instagram.com", "youtube.com", "linkedin.com"}

def perform_search(region_label: str, query: str, max_results: int):
    """
    Use OpenAI Responses API with web_search tool to get headlines for the query.
    Returns a list of {'headline': ..., 'url': ...} results (up to max_results).
    Logs progress and errors according to conventions.
    """
    log(f"{region_label}: web_search start")
    results = []
    try:
        # Call OpenAI Responses API with web_search tool
        resp = client.responses.create(
            model="gpt-4o",  # model supporting tools (could be parameterized)
            input=f"Give {max_results} recent reputable headlines for: {query}",
            tools=[{"type": "web_search"}]
        )
        # Log the response ID for traceability
        try:
            print(f"[web_search] resp_id={getattr(resp, 'id', None)} label={region_label}")
        except Exception:
            pass
        # Extract URL citations from the response
        data = resp.model_dump() if hasattr(resp, "model_dump") else resp  # ensure it's a dict
        for output in (data.get("output") or []):
            for content_block in (output.get("content") or []):
                for ann in (content_block.get("annotations") or []):
                    if ann.get("type") == "url_citation":
                        url = ann.get("url") or ""
                        title = ann.get("title") or ""
                        if url and title:
                            # Filter out blacklisted domains early
                            if get_domain(url) not in BLACKLIST_DOMAINS:
                                results.append({"headline": title, "url": url})
    except openai.error.RateLimitError as e:
        # API rate limit reached
        log(f"{region_label}: web_search error -> RateLimitError")
        return None  # indicate a critical error for this region
    except NameError as e:
        # Any NameError (e.g., if something is not defined as expected)
        log(f"{region_label}: web_search error -> NameError")
        # Continue as non-critical (could try other queries if any)
        return []
    except Exception as e:
        # Other exceptions from OpenAI API call
        log(f"{region_label}: web_search error -> {type(e).__name__}")
        return []
    # Log number of results (if any)
    if results:
        log(f"{region_label}: web_search results={len(results)}")
    return results[:max_results]  # limit to max_results if more were returned

def fetch_article_content(item: dict) -> dict:
    """
    Fetch the article content for a given item (with 'url' and 'headline').
    Cleans the URL, fetches the page, extracts text, and adds 'content' and 'source'.
    Returns the updated item dict.
    """
    url = strip_tracking_params(item.get("url", ""))
    item["url"] = url  # update URL after stripping tracking parameters
    try:
        r = requests.get(url, timeout=10, headers={
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0 Safari/537.36"),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8"
        })
        if r.status_code != 200:
            item["content"] = ""  # failed to retrieve content
            return item
        soup = BeautifulSoup(r.text, "html.parser")
        # Try paragraphs first
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paras).strip()
        if text:
            item["content"] = text[:2000]  # take first 2000 chars of combined paragraphs
        else:
            # Fallback to meta descriptions
            desc = ""
            meta_tags = [
                ("meta", {"name": "description"}),
                ("meta", {"property": "og:description"}),
                ("meta", {"name": "twitter:description"})
            ]
            for tag_name, attrs in meta_tags:
                tag = soup.find(tag_name, attrs)
                if tag and tag.get("content"):
                    desc = tag["content"].strip()
                    if desc:
                        break
            if desc:
                item["content"] = desc[:400]
            else:
                # Last resort: page title
                title_tag = soup.find("title")
                item["content"] = title_tag.get_text(strip=True)[:200] if title_tag else ""
        # Add source name from URL
        item["source"] = extract_source_name(url)
    except Exception:
        # On any exception (request timeout, parse error, etc.), mark content empty
        item["content"] = ""
    return item

def deduplicate_items(items: list, max_items: int = None) -> list:
    """
    Deduplicate the list of news item dicts by URL + headline.
    Preserve order of first occurrence. If max_items is given, limit the output list to that many items.
    """
    seen_keys = set()
    unique_items = []
    for it in items:
        key = (it.get("url", ""), (it.get("headline", "") or "").strip().lower())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_items.append(it)
        if max_items and len(unique_items) >= max_items:
            break
    return unique_items

def main():
    # Determine regions and queries from CLI or config
    regions_config = []
    args = sys.argv[1:]
    config_path = None
    # Check if a config file is specified
    if args and args[0].endswith(".json"):
        config_path = args[0]
        args = args[1:]  # remove config path from region args if present
    if config_path:
        # Load regions from JSON file
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            log(f"ERROR: Failed to load config file '{config_path}' -> {type(e).__name__}")
            sys.exit(1)
        # Expect data to have a "regions" list
        if "regions" in data and isinstance(data["regions"], list):
            for reg in data["regions"]:
                if not isinstance(reg, dict) or "name" not in reg:
                    continue
                name = reg["name"]
                if "queries" in reg:
                    queries = reg["queries"] if isinstance(reg["queries"], list) else [reg["queries"]]
                elif "query" in reg:
                    queries = [reg["query"]]
                else:
                    # If no query specified, use default template
                    queries = [f"major market news today {name}"]
                regions_config.append({"name": name, "queries": queries})
        else:
            # If config format is unexpected, exit with error
            log("ERROR: Config JSON must contain a 'regions' list")
            sys.exit(1)
    # Process any region arguments from CLI (they override defaults if provided)
    for arg in args:
        # CLI arg format: "RegionName=optional query"
        if "=" in arg:
            name, q = arg.split("=", 1)
            name = name.strip()
            query = q.strip().strip('"').strip("'")  # remove any extra quotes in query
            regions_config.append({"name": name, "queries": [query]})
        else:
            # Only region name given, use default query template for it
            name = arg.strip()
            regions_config.append({"name": name, 
                                    "queries": [f"major market news today {name}"]})
    # If nothing provided, use default regions (GLOBAL/ASIA and INDONESIA)
    if not regions_config:
        regions_config = [
            {"name": "GLOBAL/ASIA", "queries": ["major market news today global/Asia"]},
            {"name": "INDONESIA", "queries": ["major market news today Indonesia"]}
        ]
    # Now process each region
    all_results = []        # to collect all results for global deduplication
    region_original_counts = {}  # track count per region before dedupe
    for region in regions_config:
        region_name = region["name"]
        queries = region.get("queries", [])
        region_items = []  # collected results for this region
        # Perform each search query for this region
        for q in queries:
            search_results = perform_search(region_name, q, max_results=MAX_RESULTS_PER_QUERY)
            if search_results is None:
                # A critical error (e.g., RateLimit) occurred, skip further queries for this region
                region_items = []  # no results due to error
                break
            # Extend region_items with new results (will deduplicate later globally)
            if search_results:
                region_items.extend(search_results)
        # If search_results came back None (RateLimitError), we skip processing this region
        if search_results is None:
            continue  # move to next region without fetching content or logging no-URLs (error already logged)
        # If no results for this region (after all queries), log and continue
        if not region_items:
            log(f"{region_name}: no URLs")
            region_original_counts[region_name] = 0
            continue
        # Deduplicate region_items (in case multiple queries yielded same link)
        region_items = deduplicate_items(region_items)
        # Fetch content for all region items in parallel (limited by MAX_THREADS)
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            fetched_items = list(executor.map(fetch_article_content, region_items))
        # Filter out items with blacklisted domains or too-short content
        filtered_items = []
        for item in fetched_items:
            url = item.get("url", "")
            # Skip if domain is blacklisted (additional check in case some slipped through)
            if get_domain(url) in BLACKLIST_DOMAINS:
                continue
            content = (item.get("content") or "").strip()
            if not content:
                continue  # drop items with no content retrieved
            # Apply minimum content length thresholds
            threshold = MIN_CONTENT_CHARS_LOCAL if region_name.upper().startswith("INDONESIA") else MIN_CONTENT_CHARS_GLOBAL
            if len(content) < threshold:
                continue
            # Tag the item with its region category
            item["region"] = region_name
            filtered_items.append(item)
        # Update region results and count
        region_items = filtered_items
        region_original_counts[region_name] = len(region_items)
        # Add to global list for deduplication across regions
        all_results.extend(region_items)
    # Deduplicate across all regions, optionally limit total results if needed
    unique_results = deduplicate_items(all_results)
    # Log total after deduplication with breakdown per region
    total_count = len(unique_results)
    # Build breakdown string in format: region1_count, region2_count, ...
    breakdown_parts = []
    for region_name, count in region_original_counts.items():
        # slugify region name for log (lowercase, spaces to underscores)
        slug = region_name.lower().replace(" ", "_")
        breakdown_parts.append(f"{slug}={count}")
    breakdown_str = ", ".join(breakdown_parts)
    log(f"TOTAL after dedupe: {total_count} ({breakdown_str})")
    # Prepare output JSON structure
    output = {
        "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "regions": []
    }
    for region in regions_config:
        name = region["name"]
        queries = region.get("queries", [])
        # Find all items in unique_results that belong to this region
        region_results = [item for item in unique_results if item.get("region") == name]
        # Use "query" key if single query, otherwise "queries"
        region_entry = {
            "name": name,
            "results": region_results
        }
        if len(queries) == 1:
            region_entry["query"] = queries[0]
        else:
            region_entry["queries"] = queries
        output["regions"].append(region_entry)
    # Print the final JSON output
    print(json.dumps(output, ensure_ascii=False, indent=2))
    
if __name__ == "__main__":
    main()
