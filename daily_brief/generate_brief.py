import os, json, re, time, random, openai
from openai import OpenAI, NotFoundError
from .config import OPENAI_API_KEY, MODEL_COMPOSE_PREF, MAX_COMPLETION_TOKENS

client = OpenAI(api_key=OPENAI_API_KEY)

def _chat_json_with_backoff(models, system, user, max_tokens):
    delay = 0.35
    for _ in range(6):                         # retries
        for m in models:
            try:
                r = client.chat.completions.create(
                    model=m,
                    messages=[{"role":"system","content":system},
                              {"role":"user","content":user}],
                    response_format={"type":"json_object"},
                    max_completion_tokens=max_tokens,
                )
                txt = (r.choices[0].message.content or "").strip()
                if txt:
                    return json.loads(txt)
            except openai.RateLimitError:
                time.sleep(delay + random.uniform(0,0.3))
            except NotFoundError:
                continue
            except Exception:
                time.sleep(0.2)
        delay = min(delay * 2, 6.0)
    raise RuntimeError("Compose JSON failed after retries/fallbacks")

def _chat_text_with_backoff(models, system, user, max_tokens):
    delay = 0.35
    for _ in range(6):
        for m in models:
            try:
                r = client.chat.completions.create(
                    model=m,
                    messages=[{"role":"system","content":system},
                              {"role":"user","content":user}],
                    max_completion_tokens=max_tokens,
                )
                txt = (r.choices[0].message.content or "").strip()
                if txt:
                    return txt
            except openai.RateLimitError:
                time.sleep(delay + random.uniform(0,0.3))
            except NotFoundError:
                continue
            except Exception:
                time.sleep(0.2)
        delay = min(delay * 2, 6.0)
    raise RuntimeError("Compose Markdown failed after retries/fallbacks")

def compose_and_generate(
    date: str,
    market_summaries: dict,
    economic_events: list,
    news_by_sector: dict,
    watchlist_alerts: list,
    emerging_themes: list,
    sentiment_indicators: dict,
) -> tuple:
    schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
    schema = json.load(open(schema_path))

    context = {
        "date": date,
        "market_summaries": market_summaries or {},
        "economic_events": economic_events or [],
        "news_by_sector": news_by_sector,
        "watchlist_alerts": watchlist_alerts or [],
        "emerging_themes": emerging_themes or [],
        "sentiment_indicators": sentiment_indicators or {},
    }

    # Small pause to let TPM window roll over if classification/sentiment just ran
    time.sleep(1.1)

    # ---- JSON ----
    sys_json = (
        "You are an equity research assistant. "
        "Output ONLY one JSON object that STRICTLY matches the provided JSON Schema. "
        "No prose, no markdown."
    )
    usr_json = (
        f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Context JSON:\n{json.dumps(context, ensure_ascii=False)}"
    )
    brief_json = _chat_json_with_backoff(MODEL_COMPOSE_PREF, sys_json, usr_json, MAX_COMPLETION_TOKENS)

    # ---- Markdown ----
    sys_md = (
        "Format the following Morning Brief JSON into clean Markdown:\n"
        " - Title with date\n"
        " - Headings: Market Summaries, Economic Events, News by Sector, Watchlist Alerts, Emerging Themes\n"
        " - Bulleted items; include source URLs at end of each bullet\n"
        " - Output Markdown only."
    )
    usr_md = json.dumps(brief_json, ensure_ascii=False)
    brief_md = _chat_text_with_backoff(MODEL_COMPOSE_PREF, sys_md, usr_md, MAX_COMPLETION_TOKENS)

    return brief_json, brief_md
