import json, time, random, openai
from openai import OpenAI
from .config import (
    OPENAI_API_KEY, MODEL_UTILITY, MAX_PER_BATCH,
    HEADLINE_ONLY_FOR_UTILITY
)

client = OpenAI(api_key=OPENAI_API_KEY)

# Allowed sectors must match the rest of your system
GICS_SECTORS = {
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Health Care", "Financials",
    "Information Technology", "Communication Services", "Utilities", "Real Estate"
}

# Strict JSON schema for Responses API
SECTOR_SCHEMA = {
    "name": "sector_map",
    "schema": {
        "type": "object",
        "properties": {
            "mapping": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {"type": "integer"},
                        "sector": {"type": "string"}
                    },
                    "required": ["i", "sector"],
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
            # Bubble up non-rate errors
            raise e
    raise RuntimeError("Rate limit: retries exhausted")

def _chunks(n):
    i = 0
    while True:
        yield range(i, min(i + MAX_PER_BATCH, n))
        i += MAX_PER_BATCH
        if i >= n:
            break

def batch_assign_sector(items: list) -> None:
    """
    Assigns sector in-place for items missing 'sector'. Batched Responses API with strict schema.
    """
    if not items:
        return

    # Identify indices to classify
    targets = [i for i, it in enumerate(items) if not it.get("sector")]
    if not targets:
        return

    # Build valid sector list string for prompt
    sector_list = ", ".join(sorted(GICS_SECTORS))

    for grp in _chunks(len(targets)):
        idxs = [targets[i] for i in grp]
        if not idxs:
            break

        # Headline-only to minimize tokens
        lines = [f"{i+1}. {items[i].get('headline','')}" for i in idxs]
        prompt = (
            "Assign one **GICS** sector to each headline.\n"
            f"Valid sectors: {sector_list}.\n"
            "Return JSON in {\"mapping\": [{\"i\": <absolute_index>, \"sector\": \"<name>\"}]} "
            "where <absolute_index> is the original global index (1-based).\n\n" +
            "\n".join(lines)
        )

        # Call Responses API with strict JSON
        resp = _backoff(
            client.responses.create,
            model=MODEL_UTILITY,
            input=prompt,
            response_format={"type": "json_schema", "json_schema": SECTOR_SCHEMA}
        )

        payload = resp.output_text
        data = json.loads(payload)
        for m in data["mapping"]:
            real_idx = int(m["i"]) - 1
            sec = m["sector"]
            if 0 <= real_idx < len(items):
                items[real_idx]["sector"] = sec if sec in GICS_SECTORS else "Unknown"

    # Any remaining unset -> Unknown (no per-item model calls)
    for it in items:
        if not it.get("sector"):
            it["sector"] = "Unknown"
