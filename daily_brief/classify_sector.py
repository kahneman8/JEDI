"""phase1_daily_brief/classify_sector.py"""
import json
import openai
from .config import MODEL, TEMPERATURE, GICS_SECTORS


def _single_classify(headline: str, content: str = "") -> str:
    """
    Fallback method to classify a single news item if batch classification fails.
    """
    prompt = (
        "Headline: {h}\nContent: {c}\n"
        "Which GICS sector does this news belong to? Choose one from: {choices}."
    ).format(h=headline, c=content[:200], choices=", ".join(GICS_SECTORS.keys()))
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    answer = resp.choices[0].message["content"].strip()
    for sector in GICS_SECTORS:
        if sector in answer:
            return sector
    return "Unknown"


def batch_assign_sector(items: list) -> None:
    """
    Assign a sector to each item in place. Uses one GPT-5 call for multiple items.
    The model returns a JSON list of mappings {"i": index, "sector": sector}.
    """
    if not items:
        return

    numbered = [f"{i+1}. {it.get('headline', '')}" for i, it in enumerate(items)]
    sector_list = ", ".join(GICS_SECTORS.keys())
    prompt = (
        "Assign one GICS sector to each headline below. "
        "Valid sectors: {sectors}. "
        "Return JSON list: [{{\"i\": <index>, \"sector\": <sector>}}].\n\n"
        "{lines}"
    ).format(sectors=sector_list, lines="\n".join(numbered))

    try:
        resp = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        mapping = json.loads(resp.choices[0].message["content"])
        for entry in mapping:
            idx = int(entry.get("i", 0)) - 1
            sector = entry.get("sector", "Unknown")
            if 0 <= idx < len(items):
                items[idx]["sector"] = sector
    except Exception:
        # Fallback: classify each item individually
        for it in items:
            it["sector"] = _single_classify(
                it.get("headline", ""), it.get("content", "")
            )
