"""phase1_daily_brief/config.py"""
import os

#API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Model selection: use GPT-5 Pro for all LLM tasks
# Models
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")     # final JSON+Markdown compose
MODEL_CLASSIFY = os.getenv("OPENAI_MODEL_CLASSIFY", "gpt-5-mini")
MODEL_REASON = os.getenv("OPENAI_MODEL_REASON", "gpt-5")
MODEL_COMPOSE_PREF = ["gpt-5-mini", "gpt-5"]
# Maximum tokens for large outputs (adjust to your quota)

MAX_ARTICLES_TOTAL = int(os.getenv("MAX_ARTICLES_TOTAL", "12"))
MAX_PER_BATCH = int(os.getenv("MAX_PER_BATCH", "6"))
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "1200"))
MAX_OUTPUT_TOKENS = 2000

# Search queries for global & local news
GLOBAL_QUERY = "Global & Asia market news in the last 72 hours"
LOCAL_QUERY = "Indonesia market news in the last 72 hours"
SEARCH_MAX_RESULTS = 6 # results per query (top results only)
# News fetch settings
FETCH_TIMEOUT_SEC = 12 # seconds per HTTP fetch
MAX_ARTICLES_TOTAL = 20 # cap to control tokens & cost
MAX_WORKERS = 6 # thread pool size for fetching URLs
THEMES_MAX = int(os.getenv("THEMES_MAX", "6"))
SUMMARY_ITEMS_PER_REGION = int(os.getenv("SUMMARY_ITEMS_PER_REGION", "8"))
# Behavior
HEADLINE_ONLY_FOR_UTILITY = True #Classify & sentiment on headlines only
CACHE_PATH = "outputs/model_cache.json"

# Region Detection
ASIA_HINTS = {
    "asia","asian","china","chinese","japan","japanese","korea","korean","taiwan","hong kong",
    "nikkei","kospi","sgx","tse","shanghai","shenzhen","seoul","tokyo","taipei","singapore","malaysia","thailand","philippines","vietnam","india","indian"
}
INDONESIA_HINTS = {
    "indonesia","indonesian","jakarta","jci","idx","bank indonesia","bi","rupiah","idrupiah","idn"
}

# GICS sectors mapping (top-level sectors only)
GICS_SECTORS = {
 "Energy": "Oil, gas, coal and fuels",
 "Materials": "Raw materials (metals, chemicals, etc.)",
 "Industrials": "Manufacturing, aerospace, logistics",
 "Consumer Discretionary": "Non-essential goods & services",
 "Consumer Staples": "Essential goods (food, beverage, household)",
 "Health Care": "Pharma, biotech, medical devices",
 "Financials": "Banking, insurance, investment, REITs",
 "Information Technology": "Software, hardware, semiconductors",
 "Communication Services": "Telecom, media, entertainment",
 "Utilities": "Electric, water, gas utilities",
 "Real Estate": "Real estate"
}
