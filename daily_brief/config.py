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
# Limits & batching
MAX_ARTICLES_TOTAL = int(os.getenv("MAX_ARTICLES_TOTAL", "12"))
MAX_PER_BATCH = int(os.getenv("MAX_PER_BATCH", "6"))
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "900"))
SUMMARY_ITEMS_PER_REGION = int(os.getenv("SUMMARY_ITEMS_PER_REGION", "8"))
THEMES_MAX = int(os.getenv("THEMES_MAX", "3"))
SEARCH_MAX_RESULTS = 10
FETCH_TIMEOUT_SEC = 15
MAX_WORKERS = 6

MIN_CONTENT_CHARS_GLOBAL = int(os.getenv("MIN_CONTENT_CHARS_GLOBAL", "140"))
MIN_CONTENT_CHARS_ID     = int(os.getenv("MIN_CONTENT_CHARS_ID", "60"))

# Fetch/search
GLOBAL_QUERY = ("You are a buy-side AI analyst focused on global and asia equities market. Compile high-importance market relevant news in the Global + Asia region within the past 24 hours")
# Strong Indonesia hint (English + Bahasa + outlet hints given to the web_search tool)
LOCAL_QUERY  = ("You are a buy-side AI analyst focused on the Indonesian market. Compile high-importance market relevant news in the Indonesia region within the past 24 hours")
# Behavior
HEADLINE_ONLY_FOR_UTILITY = True
CACHE_PATH = "outputs/model_cache.json"

# Region detection
BLACKLIST_DOMAINS = {"example.com"}
ASIA_HINTS = {
    "asia","asian","china","chinese","japan","japanese","korea","korean","taiwan","hong kong",
    "nikkei","kospi","sgx","tse","shanghai","shenzhen","seoul","tokyo","taipei","singapore",
    "malaysia","thailand","philippines","vietnam","india","indian"
}
INDONESIA_HINTS = {
    "indonesia","indonesian","jakarta","jci","idx","bei","bank indonesia","bi","rupiah","idrupiah","idn","ihsg"
}

# GICS sectors
GICS_SECTORS = [
    "Energy","Materials","Industrials","Consumer Discretionary","Consumer Staples",
    "Health Care","Financials","Information Technology","Communication Services","Utilities","Real Estate"
]
