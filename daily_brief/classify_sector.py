"""Batch GICS sector classification with a single model (GPT-5), JSON-mode."""
import json, time, random, openai
from openai import OpenAI
from .config import OPENAI_API_KEY, MODEL, MAX_PER_BATCH, GICS_SECTORS, HEADLINE_ONLY_FOR_UTILITY

client = OpenAI(api_key=OPENAI_API_KEY)

def _backoff(call, *args, **kwargs):
    delay = 0.35
    for attempt in range(6):
        try:
            return call(*args, **kwargs)
        except openai.RateLimitError:
            if attempt == 5: raise
            time.sleep(delay + random.uniform(0, 0.25))
            delay = min(delay * 2, 6.0)
        except Exception:
            if attempt == 5: raise
            time.sleep(delay + random.uniform(0, 0.25))
            delay = min(delay * 2, 6.0)

def _batches(indices, size):
    for i in range(0, len(indices), size):
        yield indices[i:i+size]

def batch_assign_sector(items: list) -> None:
    if not items: return
    targets = [i for i, it in enumerate(items) if not it.get("sector")]
    if not targets: return

    valid = ", ".join(GICS_SECTORS)

    for group in _batches(targets, MAX_PER_BATCH):
        lines = []
        for i in group:
            text = items[i].get("headline","")
            if not HEADLINE_ONLY_FOR_UTILITY:
                text += " " + (items[i].get("content","")[:160])
            lines.append(f"{i+1}. {text}")

        prompt = (
            "Assign exactly one GICS sector to each headline from this set:\n"
            f"{valid}\n\n"
            "Return ONLY a JSON object:\n"
            '{"mapping":[{"i": <absolute_index>, "sector": "<sector>"}]}\n'
            "Use <absolute_index> as the 1-based index shown before each headline.\n\n"
            + "\n".join(lines)
        )

        resp = _backoff(
            client.chat.completions.create,
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_completion_tokens=600,
        )
        payload = (resp.choices[0].message.content or "").strip()
        data = json.loads(payload) if payload else {"mapping": []}

        for m in data.get("mapping", []):
            idx = int(m.get("i", 0)) - 1
            sec = m.get("sector", "Unknown")
            if 0 <= idx < len(items):
                items[idx]["sector"] = sec if sec in GICS_SECTORS else "Unknown"

    for it in items:
        if not it.get("sector"):
            it["sector"] = "Unknown"
