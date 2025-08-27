import json, time, random, openai
from openai import OpenAI
from .config import OPENAI_API_KEY, MODEL_UTILITY, MAX_PER_BATCH

client = OpenAI(api_key=OPENAI_API_KEY)

SENT_SCHEMA = {
    "name": "sent_map",
    "schema": {
        "type": "object",
        "properties": {
            "mapping": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {"type": "integer"},
                        "sentiment": {
                            "type": "string",
                            "enum": ["Positive", "Negative", "Neutral"]
                        }
                    },
                    "required": ["i", "sentiment"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["mapping"],
        "additionalProperties": False
    },
    "strict": True
}

def _backoff(call, *args, **kwargs):
    delay = 0.25
    for _ in range(6):
        try:
            return call(*args, **kwargs)
        except openai.RateLimitError:
            time.sleep(delay + random.uniform(0, 0.2))
            delay = min(delay * 2, 6.0)
        except Exception as e:
            raise e
    raise RuntimeError("Rate limit: retries exhausted")

def _chunks(n):
    i = 0
    while True:
        yield range(i, min(i + MAX_PER_BATCH, n))
        i += MAX_PER_BATCH
        if i >= n:
            break

def batch_assign_sentiment(items: list) -> None:
    """
    Assigns sentiment in-place for items missing 'sentiment'. Batched Responses API with strict schema.
    """
    if not items:
        return

    targets = [i for i, it in enumerate(items) if not it.get("sentiment")]
    if not targets:
        return

    for grp in _chunks(len(targets)):
        idxs = [targets[i] for i in grp]
        if not idxs:
            break

        lines = [f"{i+1}. {items[i].get('headline','')}" for i in idxs]
        prompt = (
            "For each headline, label sentiment strictly as one of: Positive, Negative, Neutral.\n"
            "Return JSON {\"mapping\": [{\"i\": <absolute_index>, \"sentiment\": \"Positive|Negative|Neutral\"}]} "
            "where <absolute_index> is the original global index (1-based).\n\n" +
            "\n".join(lines)
        )

        resp = _backoff(
            client.responses.create,
            model=MODEL_UTILITY,
            input=prompt,
            response_format={"type": "json_schema", "json_schema": SENT_SCHEMA}
        )

        payload = resp.output_text
        data = json.loads(payload)
        for m in data["mapping"]:
            real_idx = int(m["i"]) - 1
            if 0 <= real_idx < len(items):
                items[real_idx]["sentiment"] = m["sentiment"]

    # Default any missing to Neutral (no per-item model calls)
    for it in items:
        if not it.get("sentiment"):
            it["sentiment"] = "Neutral"
