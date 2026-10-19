"""
BANKMATE AI — privacy.py
Everything related to protecting customer data:

  1. Masking     — account number / email / phone shown to the customer.
  2. Redaction   — remove PII from free text BEFORE it reaches the LLM,
                   the conversation memory, or any log.
  3. Risk checks — detect messages that ask for credentials, other
                   customers' data, or try to hijack the assistant.

All patterns are intentionally conservative: when in doubt, redact.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 1. Masking (for data we DO show to the logged-in customer)
# ---------------------------------------------------------------------------


def mask_account(account_id: str) -> str:
    """ACC000001 -> *****0001 (last 4 only)."""
    if not account_id or len(account_id) <= 4:
        return "****"
    return "*" * (len(account_id) - 4) + account_id[-4:]


def mask_phone(phone: str) -> str:
    """9876543210 -> ******3210"""
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) < 4:
        return "****"
    return "*" * (len(digits) - 4) + digits[-4:]


def mask_email(email: str) -> str:
    """aarav.sharma@demobank.in -> a***a@demobank.in"""
    if not email or "@" not in email:
        return "****"
    local, _, domain = email.partition("@")
    if len(local) <= 2:
        masked = local[:1] + "***"
    else:
        masked = local[0] + "***" + local[-1]
    return f"{masked}@{domain}"


# ---------------------------------------------------------------------------
# 2. PII redaction (for free text that may reach the LLM or logs)
# ---------------------------------------------------------------------------

# Credentials: "my otp is 482913", "pin 1234", "password: abc123"
_SECRET = re.compile(
    r"\b(otp|pin|mpin|cvv|cvc|password|passcode)\b(?!\s*code)[^\d\n]{0,20}\d{3,8}\b",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_UPI = re.compile(r"\b[\w.\-]{2,}@[a-z]{2,15}\b", re.IGNORECASE)
_PAN = re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b")
_IFSC = re.compile(r"\b[A-Za-z]{4}0[A-Za-z0-9]{6}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
# 9-19 digits, optionally grouped by spaces/dashes: card, Aadhaar, account no.
_LONG_DIGITS = re.compile(r"(?<![\d₹])(?:\d[ -]?){8,18}\d(?!\d)")
# Our own customer / account IDs are fine to keep (pseudonymous).
_OWN_ID_SHAPE = re.compile(r"^(?:CUST|ACC|TXN)\d+$", re.IGNORECASE)


def _digits_label(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    n = len(digits)
    if n == 12:
        return "[AADHAAR_NO]"
    if 13 <= n <= 19:
        return "[CARD_NO]"
    return "[ACCOUNT_NO]"


def redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Replace sensitive values in `text` with placeholders.

    Returns (clean_text, sorted list of PII types found).
    Example: "my card 4111 1111 1111 1111" -> "my card [CARD_NO]", ["CARD_NO"]
    """
    if not text:
        return "", []
    found: set[str] = set()

    def sub(pattern: re.Pattern, label: str | None, s: str) -> str:
        def repl(m: re.Match) -> str:
            lab = label or _digits_label(m)
            found.add(lab.strip("[]"))
            return lab

        return pattern.sub(repl, s)

    out = text
    out = sub(_SECRET, "[SECRET]", out)
    out = sub(_EMAIL, "[EMAIL]", out)
    out = sub(_UPI, "[UPI_ID]", out)
    out = sub(_PAN, "[PAN]", out)
    out = sub(_IFSC, "[IFSC]", out)
    out = sub(_PHONE, "[PHONE]", out)
    out = sub(_LONG_DIGITS, None, out)
    return out, sorted(found)


def contains_unmasked_pii(text: str) -> list[str]:
    """Used on LLM OUTPUT: returns PII types still present in `text`."""
    _, found = redact_pii(text)
    return found


# ---------------------------------------------------------------------------
# 3. Risk checks on the user's message
# ---------------------------------------------------------------------------

BLOCK_CREDENTIALS = "credentials"
BLOCK_OTHER_CUSTOMER = "other_customer"
BLOCK_INJECTION = "injection"
BLOCK_EMPTY = "empty"
BLOCK_TOO_LONG = "too_long"

_CRED_WORDS = re.compile(
    r"\b(otp|one[- ]time password|mpin|atm pin|debit card pin|cvv|cvc|"
    r"password|passcode|net ?banking (?:id|login)|card number|"
    r"pin(?!\s*code))\b",
    re.IGNORECASE,
)

_CUST_ID = re.compile(r"\bcust[\s_-]?(\d{3,})\b", re.IGNORECASE)

_OTHER_PERSON = re.compile(
    r"\b(someone else'?s?|somebody else'?s?|another (?:customer|person|user)'?s?|"
    r"other (?:customers?|users?|people)'?s?|all customers|every customer|everyone'?s|"
    r"(?:my )?(?:friend|neighbou?r|colleague|boss|brother|sister|wife|husband|"
    r"father|mother|dad|mom|uncle|aunt|cousin|partner|girlfriend|boyfriend|ex)'?s?)"
    r"\b.{0,40}?\b(account|balance|transactions?|details|profile|statement|data|"
    r"salary|phone|email)\b",
    re.IGNORECASE,
)

_INJECTION = [
    re.compile(r"\bignore\b.{0,25}\b(previous|prior|above|earlier|all|your|the)\b.{0,25}"
               r"\b(instructions?|rules?|prompts?|guidelines?)\b", re.IGNORECASE),
    re.compile(r"\bdisregard\b.{0,30}\b(instructions?|rules?|prompt)\b", re.IGNORECASE),
    re.compile(r"\b(reveal|show|print|repeat|display|leak|tell me)\b.{0,25}"
               r"\b(system|hidden|initial|original|developer)\b.{0,10}\b(prompt|instructions?|message)\b",
               re.IGNORECASE),
    re.compile(r"\byou are now\b", re.IGNORECASE),
    re.compile(r"\b(developer|admin|god|sudo|jailbreak|dan) mode\b", re.IGNORECASE),
    re.compile(r"\bpretend (?:to be|you are|you're)\b", re.IGNORECASE),
    re.compile(r"\bact as (?:an? )?(?!bank)(?:admin|root|developer|hacker|dan)\b", re.IGNORECASE),
    re.compile(r"<\s*/?\s*(system|assistant|user_message|facts|history)\s*>", re.IGNORECASE),
]


def _normalise_customer_id(digits: str) -> str:
    return f"CUST{int(digits):04d}"


def check_message_risks(text: str, own_customer_id: str | None = None) -> str | None:
    """
    Return a block reason (one of BLOCK_*) if the message must not be
    processed normally, otherwise None.
    """
    if not text or not text.strip():
        return BLOCK_EMPTY

    for pattern in _INJECTION:
        if pattern.search(text):
            return BLOCK_INJECTION

    if _CRED_WORDS.search(text) or _SECRET.search(text):
        return BLOCK_CREDENTIALS

    for m in _CUST_ID.finditer(text):
        if own_customer_id is None or _normalise_customer_id(m.group(1)) != own_customer_id.upper():
            return BLOCK_OTHER_CUSTOMER

    if _OTHER_PERSON.search(text):
        return BLOCK_OTHER_CUSTOMER

    return None


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_input(text: str) -> str:
    """Strip control characters and collapse whitespace."""
    text = _CONTROL_CHARS.sub(" ", text or "")
    return re.sub(r"\s+", " ", text).strip()
