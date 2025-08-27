"""daily_brief/classify_sector.py"""
import json, re, time, openai
from openai import OpenAI
from .config import MODEL, GICS_SECTORS, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

MAX_PER_BATCH = 12              # small batches to stay under TPM
MAX_RETRIES   = 6               # backoff tries
MAX_SLEEP_S   = 8.0


def _backoff_chat(messages):
    delay = 0.5
    for _ in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(model=MODEL, messages=messages)
        except openai.RateLimitError:
            time.sleep(delay)
            delay = min(delay * 2, MAX_SLEEP_S)


def _safe_json_array(txt: str):
    # pull first JSON array if extra prose slipped in
    s, e = txt.find('['), txt.rfind(']')
    if s != -1 and e != -1:
        return json.loads(txt[s:e+1])
    # last resort: basic coercion (quotes around keys if model slipped)
    txt = re.sub(r'(\{|\s)(\w+)\s*:', r'\1"\2":', txt)
    return json.loads(txt)


def _single_classify(headline: str, content: str = "") -> str:
    prompt = (
        f"Headline: {headline}\nContent: {content[:200]}\n"
        f"Which single GICS sector? Choose one from: {', '.join(GICS_SECTORS.keys())}.\n"
        "Return only the sector name."
    )
    resp = _backoff_chat([{"role": "user", "content": prompt}])
    if not resp:
        return "Unknown"
    answer = (resp.choices[0].message.content or "").strip()
    for sector in GICS_SECTORS:
        if sector in answer:
            return sector
    return "Unknown"


def _chunks(idx_list, n):
    for i in range(0, len(idx_list), n):
        yield idx_list[i:i+n]


def batch_assign_sector(items: list) -> None:
    if not items:
        return

    # Build global index list; process in small batches to reduce token load
    indices = list(range(len(items)))
    for group in _chunks(indices, MAX_PER_BATCH):
        # Prepare batch prompt with absolute indices
        lines = [f"{i+1}. {items[i].get('headline','')}" for i in group]
        prompt = (
            "Assign one GICS sector to each headline below.\n"
            f"Valid sectors: {', '.join(GICS_SECTORS.keys())}.\n"
            "Return JSON list: [{\"i\": <absolute_index>, \"sector\": <sector>}].\n\n" +
            "\n".join(lines)
        )

        resp = _backoff_chat([{"role": "user", "content": prompt}])

        # If rate-limited repeatedly or other error: skip batch fallback to per-item with backoff
        if not resp:
            for i in group:
                items[i]["sector"] = _single_classify(
                    items[i].get("headline", ""), items[i].get("content", "")
                )
            continue

        payload = (resp.choices[0].message.content or "").strip()
        try:
            mapping = _safe_json_array(payload)
        except Exception:
            # If JSON is malformed, try per-item with backoff (still batch-friendly)
            for i in group:
                items[i]["sector"] = _single_classify(
                    items[i].get("headline", ""), items[i].get("content", "")
                )
            continue

        # Apply mapping safely
        for entry in mapping:
            idx = int(entry.get("i", 0)) - 1
            sector = entry.get("sector", "Unknown")
            if 0 <= idx < len(items):
                items[idx]["sector"] = sector if sector in GICS_SECTORS else "Unknown"

    # Final sweep: anything still unset -> classify once
    for it in items:
        if not it.get("sector"):
            it["sector"] = _single_classify(it.get("headline", ""), it.get("content", ""))
