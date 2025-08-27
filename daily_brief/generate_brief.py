# daily_brief/generate_brief.py
import os
import re
import json
import time
import random
from openai import OpenAI, NotFoundError
from .config import MODEL_COMPOSE, MAX_OUTPUT_TOKENS, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
FALLBACK_COMPOSE = "gpt-4o"  # used if MODEL_COMPOSE is unavailable


def _backoff_chat(call, *args, **kwargs):
    """
    Retry helper with exponential backoff + jitter for transient errors/empties.
    """
    delay = 0.35
    for attempt in range(6):
        try:
            return call(*args, **kwargs)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(delay + random.uniform(0, 0.25))
            delay = min(delay * 2, 6.0)


def _extract_first_json(text: str) -> str:
    """
    Extract the first JSON object from text.
    Supports ```json ... ``` blocks or scans for balanced braces.
    """
    if not text or not text.strip():
        raise ValueError("Empty model output")
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return m.group(1)
    s = text.find("{")
    if s == -1:
        raise ValueError("No '{' found in model output")
    depth = 0
    for i in range(s, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[s : i + 1]
    raise ValueError("Unbalanced JSON braces in model output")


def compose_and_generate(
    date: str,
    market_summaries: dict,
    economic_events: list,
    news_by_sector: dict,
    watchlist_alerts: list,
    emerging_themes: list,
    sentiment_indicators: dict,
) -> tuple:
    """
    Compose the Morning Brief in two steps (Chat Completions v1):
      1) JSON: robust strategy (json_object -> instruction-only -> fenced block).
      2) Markdown: format the validated JSON.
    Returns (brief_json: dict, brief_md: str).
    """
    # ---- Load schema (used to instruct model; orchestrator validates again) ----
    schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
    with open(schema_path, "r") as f:
        schema = json.load(f)

    context = {
        "date": date,
        "market_summaries": market_summaries or {},
        "economic_events": economic_events or [],
        "news_by_sector": news_by_sector,
        "watchlist_alerts": watchlist_alerts,
        "emerging_themes": emerging_themes,
        "sentiment_indicators": sentiment_indicators,
    }

    # ---------- 1) JSON (robust sequence) ----------
    system_json = (
        "You are an equity research assistant. "
        "Your job is to produce ONLY a single JSON object that STRICTLY matches the provided JSON Schema. "
        "No prose, no markdown, no code fences. Keep URLs intact."
    )
    user_payload = (
        f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Context JSON:\n{json.dumps(context, ensure_ascii=False)}"
    )

    models_try = [MODEL_COMPOSE, FALLBACK_COMPOSE]
    brief_json = None

    # Strategy A: response_format={"type":"json_object"}
    for m in models_try:
        try:
            r_json = _backoff_chat(
                client.chat.completions.create,
                model=m,
                messages=[
                    {"role": "system", "content": system_json},
                    {"role": "user", "content": user_payload},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=MAX_OUTPUT_TOKENS,
            )
            out_text = (r_json.choices[0].message.content or "").strip()
            if out_text:
                brief_json = json.loads(out_text)
                break
        except NotFoundError:
            continue
        except Exception:
            # fall through to Strategy B
            pass

    # Strategy B: instruction-only (no response_format)
    if brief_json is None:
        system_json_b = (
            "You are an equity research assistant.\n"
            "Return ONLY a valid JSON object that matches the schema. No extra text."
        )
        for m in models_try:
            try:
                r_json = _backoff_chat(
                    client.chat.completions.create,
                    model=m,
                    messages=[
                        {"role": "system", "content": system_json_b},
                        {"role": "user", "content": user_payload},
                    ],
                    max_completion_tokens=MAX_OUTPUT_TOKENS,
                )
                out_text = (r_json.choices[0].message.content or "").strip()
                if out_text:
                    try:
                        brief_json = json.loads(out_text)
                    except Exception:
                        brief_json = json.loads(_extract_first_json(out_text))
                    break
            except NotFoundError:
                continue
            except Exception:
                pass

    # Strategy C: fenced block request & extract
    if brief_json is None:
        system_json_c = (
            "You are an equity research assistant.\n"
            "Output the JSON object inside a fenced code block as ```json ... ``` that matches the schema."
        )
        for m in models_try:
            try:
                r_json = _backoff_chat(
                    client.chat.completions.create,
                    model=m,
                    messages=[
                        {"role": "system", "content": system_json_c},
                        {"role": "user", "content": user_payload},
                    ],
                    max_completion_tokens=MAX_OUTPUT_TOKENS,
                )
                out_text = (r_json.choices[0].message.content or "").strip()
                brief_json = json.loads(_extract_first_json(out_text))
                break
            except NotFoundError:
                continue
            except Exception:
                pass

    if brief_json is None:
        raise RuntimeError("Compose JSON failed: model returned empty or non-JSON output in all strategies")

    # ---------- 2) Markdown ----------
    system_md = (
        "Format the following Morning Brief JSON into a clean Markdown report:\n"
        " - Title with date\n"
        " - Headings: Market Summaries, Economic Events, News by Sector, Watchlist Alerts, Emerging Themes\n"
        " - Bulleted items; include source URLs at the end of each bullet as citations\n"
        " - Output Markdown only (no JSON)."
    )
    user_md = json.dumps(brief_json, ensure_ascii=False)

    brief_md = None
    for m in models_try:
        try:
            r_md = _backoff_chat(
                client.chat.completions.create,
                model=m,
                messages=[
                    {"role": "system", "content": system_md},
                    {"role": "user", "content": user_md},
                ],
                max_completion_tokens=MAX_OUTPUT_TOKENS,
            )
            md_text = (r_md.choices[0].message.content or "").strip()
            if md_text:
                brief_md = md_text
                break
        except NotFoundError:
            continue
        except Exception:
            pass

    if brief_md is None:
        raise RuntimeError("Compose Markdown failed: model returned empty output")

    return brief_json, brief_md
