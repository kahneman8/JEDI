"""Batch sentiment classification with a single model (GPT-5), JSON-mode."""
import json, time, random, openai
from openai import OpenAI
from .config import OPENAI_API_KEY, MODEL, MAX_PER_BATCH, HEADLINE_ONLY_FOR_UTILITY

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

def batch_assign_sentiment(items: list) -> None:
    if not items: return
    targets = [i for i, it in enumerate(items) if not it.get("sentiment")]
    if not targets: return

    for group in _batches(targets, MAX_PER_BATCH):
        lines = []
        for i in group:
            text = items[i].get("headline","")
            if not HEADLINE_ONLY_FOR_UTILITY:
                text += " " + (items[i].get("content","")[:160])
            lines.append(f"{i+1}. {text}")

        prompt = (
            "For each headline, assign sentiment strictly as one of: Positive, Negative, Neutral.\n"
            "Return ONLY a JSON object:\n"
            '{"mapping":[{"i": <absolute_index>, "sentiment": "Positive|Negative|Neutral"}]}\n'
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
            lab = (m.get("sentiment","Neutral") or "Neutral").capitalize()
            if 0 <= idx < len(items):
                items[idx]["sentiment"] = lab if lab in {"Positive","Negative","Neutral"} else "Neutral"

    for it in items:
        if not it.get("sentiment"):
            it["sentiment"] = "Neutral"
