"""
BANKMATE AI — guardrails.py
Checks applied to LLM-written answers BEFORE the customer sees them.

  sanitize_llm_text()   removes HTML, images, links and URLs (these can be used
                        to leak data or phish, e.g. a markdown image whose URL
                        contains the customer's balance).
  ungrounded_numbers()  finds ₹ amounts and percentages that do not appear in
                        the facts we gave the model (= possible hallucination).

If a check fails, the agent replaces the LLM text with the deterministic
template answer.
"""

from __future__ import annotations

import re

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_RUPEE = re.compile(r"₹\s?(\d[\d,]*(?:\.\d+)?)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

MAX_REPLY_CHARS = 1800


def sanitize_llm_text(text: str) -> str:
    out = _THINK.sub("", text or "")
    out = _MD_IMAGE.sub("", out)
    out = _MD_LINK.sub(r"\1", out)
    out = _URL.sub("", out)
    out = _HTML_TAG.sub("", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out[:MAX_REPLY_CHARS]


def _collect_numbers(obj, bucket: set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        bucket.add(round(float(obj), 2))
    elif isinstance(obj, str):
        for token in _NUMBER.findall(obj):
            try:
                bucket.add(round(float(token.replace(",", "")), 2))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_numbers(value, bucket)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_numbers(value, bucket)


def allowed_numbers(facts: dict) -> set[float]:
    bucket: set[float] = set()
    _collect_numbers(facts, bucket)
    return bucket


def ungrounded_numbers(text: str, facts: dict) -> list[str]:
    """Return ₹ amounts / percentages in `text` that are not present in `facts`."""
    allowed = allowed_numbers(facts)
    bad: list[str] = []
    for pattern, prefix in ((_RUPEE, "₹"), (_PERCENT, "%")):
        for match in pattern.finditer(text or ""):
            raw = match.group(1).rstrip(".,")
            try:
                value = round(float(raw.replace(",", "")), 2)
            except ValueError:
                continue
            if not any(abs(value - a) < 0.011 for a in allowed):
                bad.append(f"{prefix}{raw}" if prefix == "₹" else f"{raw}%")
    return bad
