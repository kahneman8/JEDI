import os, json, hashlib
from .config import CACHE_PATH

def _load():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

def _save(data: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f, indent=2)

def _key(url: str) -> str:
    return hashlib.sha256((url or "").encode("utf-8")).hexdigest()

def get(url: str) -> dict:
    return _load().get(_key(url), {})

def set(url: str, value: dict):
    d = _load()
    d[_key(url)] = value
    _save(d)
