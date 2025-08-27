"""daily_brief/classify_sector.py"""
import json
from openai import OpenAI
from .config import MODEL, TEMPERATURE, GICS_SECTORS, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def _single_classify(headline: str, content: str = "") -> str:
    prompt = (
        f"Headline: {headline}\nContent: {content[:200]}\n"
        f"Which GICS sector does this news belong to? Choose one from: {', '.join(GICS_SECTORS.keys())}."
        "\nReturn only the sector name."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = (resp.choices[0].message.content or "").strip()
    for sector in GICS_SECTORS:
        if sector in answer:
            return sector
    return "Unknown"


def batch_assign_sector(items: list) -> None:
    if not items:
        return

    numbered = [f"{i+1}. {it.get('headline','')}" for i, it in enumerate(items)]
    prompt = (
        "Assign one GICS sector to each headline below.\n"
        f"Valid sectors: {', '.join(GICS_SECTORS.keys())}.\n"
        "Return JSON list: [{\"i\": <index>, \"sector\": <sector>}].\n\n"
        + "\n".join(numbered)
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        payload = (resp.choices[0].message.content or "").strip()
        mapping = json.loads(payload)
        for entry in mapping:
            idx = int(entry.get("i", 0)) - 1
            sector = entry.get("sector", "Unknown")
            if 0 <= idx < len(items):
                items[idx]["sector"] = sector if sector in GICS_SECTORS else "Unknown"
    except Exception:
        for it in items:
            it["sector"] = _single_classify(it.get("headline", ""), it.get("content", ""))
