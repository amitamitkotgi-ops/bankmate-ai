"""
BANKMATE AI — config.py
Central settings. Values are read from the environment / .env file at call
time, so tests (and you) can change them without re-importing modules.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Load .env (if present). Real environment variables always win.
load_dotenv(BASE_DIR / ".env", override=False)

# Values people leave in .env by mistake — treat them as "no key".
_PLACEHOLDER_KEYS = {
    "",
    "your_groq_api_key_here",
    "paste_your_key_here",
    "paste_your_groq_api_key_here",
    "gsk_your_key_here",
    "changeme",
}

# Groq retires models regularly. The older llama-3.3-70b-versatile and
# llama-3.1-8b-instant were shut down on 16 Aug 2026 (free/developer tier).
# These are the current recommended replacements, tried in this order.
# You can override with GROQ_MODEL / GROQ_FALLBACK_MODELS in .env.
DEFAULT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]


def get_groq_api_key() -> str:
    """Return the Groq API key, or '' when missing / still a placeholder."""
    key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
    if key.lower() in _PLACEHOLDER_KEYS or key.lower().startswith("paste"):
        return ""
    return key


def get_model_candidates() -> list[str]:
    """Ordered, de-duplicated list of Groq model IDs to try."""
    primary = os.getenv("GROQ_MODEL", "").strip()
    extra = [m.strip() for m in os.getenv("GROQ_FALLBACK_MODELS", "").split(",") if m.strip()]
    ordered = ([primary] if primary else []) + extra + DEFAULT_MODELS
    seen: set[str] = set()
    result: list[str] = []
    for m in ordered:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def get_groq_base_url() -> str | None:
    """Optional custom Groq base URL (used by the offline integration test)."""
    value = os.getenv("GROQ_API_BASE", "").strip()
    return value or None


def get_llm_timeout() -> float:
    try:
        return max(3.0, float(os.getenv("GROQ_TIMEOUT_SECONDS", "20")))
    except ValueError:
        return 20.0


def get_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
    return [o.strip() for o in raw.split(",") if o.strip()]


def get_api_base() -> str:
    """Where the Streamlit app finds the FastAPI backend."""
    return os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")


# Chat limits (protect the Groq quota and the server)
CHAT_MAX_CHARS = 500
CHAT_RATE_LIMIT_PER_MINUTE = int(os.getenv("CHAT_RATE_LIMIT_PER_MINUTE", "20"))
SESSION_TTL_HOURS = 2
