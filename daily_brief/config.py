"""phase1_daily_brief/config.py"""
import os

#API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Model selection: use GPT-5 Pro for all LLM tasks
# Models
MODEL = "gpt-5"     # final JSON+Markdown compose
MODEL_CLASSIFY = "gpt-5-mini"
MODEL_COMPOSE_PREF = ['gpt-5-mini", "gpt-5"]
# Maximum tokens for large outputs (adjust to your quota)
MAX_OUTPUT_TOKENS = 2000
MAX_COMPLETION_TOKENS = 1200
MAX_PER_BATCH = 6
# Search queries for global & local news
GLOBAL_QUERY = "global market news Asia overnight"
LOCAL_QUERY = "Indonesia market news today"
SEARCH_MAX_RESULTS = 6 # results per query (top results only)
# News fetch settings
FETCH_TIMEOUT_SEC = 12 # seconds per HTTP fetch
MAX_ARTICLES_TOTAL = 10 # cap to control tokens & cost
MAX_WORKERS = 6 # thread pool size for fetching URLs

# Behavior
HEADLINE_ONLY_FOR_UTILITY = True #Classify & sentiment on headlines only
CACHE_PATH = "outputs/model_cache.json"
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
