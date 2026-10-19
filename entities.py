"""
BANKMATE AI — entities.py
Deterministic extraction of structured values from a customer message:

    "I earn 60k a month, can I get a 5 lakh personal loan for 3 years?"
    -> {monthly_income: 60000, loan_amount: 500000, tenure_months: 36,
        product_id: "personal_loan"}

Why deterministic (not the LLM)?  Numbers decide money answers. A language
model can mis-read "5 lakh". Regular expressions cannot hallucinate, so the
LLM is only allowed to suggest NON-numeric things (intent, category...).
"""

from __future__ import annotations

import re
from datetime import date

import products as product_catalogue

# ---------------------------------------------------------------------------
# Allowed values (also used to validate anything the LLM suggests)
# ---------------------------------------------------------------------------

DEBIT_CATEGORIES = [
    "Groceries", "Dining", "Utilities", "Shopping",
    "Transport", "Health", "Entertainment", "Education",
]
CREDIT_CATEGORIES = ["Salary", "Other Income"]
ALL_CATEGORIES = DEBIT_CATEGORIES + CREDIT_CATEGORIES
TXN_TYPES = ["Credit", "Debit"]

MAX_CHAT_LIMIT = 20

_WORD_NUMBERS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "twelve": "12",
    "fifteen": "15", "twenty": "20",
}
_WORD_NUM_RE = re.compile(r"\b(" + "|".join(_WORD_NUMBERS) + r")\b")

_UNIT_VALUE = {
    "k": 1_000, "thousand": 1_000,
    "l": 100_000, "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "cr": 10_000_000, "crore": 10_000_000, "crores": 10_000_000,
}

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_PERIOD_N = re.compile(
    r"\b(?:in the |over the |for the )?(?:last|past|previous)\s+(\d{1,3})\s*(days?|weeks?|months?|years?|yrs?)\b"
)
_PERIOD_WORDS = [
    (re.compile(r"\b(?:last|past|previous)\s+(?:week)\b"), 7, "last 7 days"),
    (re.compile(r"\b(?:last|past|previous)\s+fortnight\b"), 14, "last 14 days"),
    (re.compile(r"\b(?:last|past|previous)\s+month\b"), 30, "last 30 days"),
    (re.compile(r"\b(?:last|past|previous)\s+quarter\b"), 90, "last 3 months"),
    (re.compile(r"\b(?:last|past|previous)\s+year\b"), 365, "last 12 months"),
    (re.compile(r"\bthis week\b"), None, "this week"),
    (re.compile(r"\bthis month\b|\bis mahine\b|\bis mahinay?\b"), None, "this month"),
    (re.compile(r"\bthis year\b"), None, "this year"),
    (re.compile(r"\btoday\b"), 1, "today"),
]
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365, "yr": 365}

_LIMIT_PATTERNS = [
    re.compile(
        r"\b(?:last|latest|recent|top|past|first|show|list|give me|get)\s+(\d{1,2})\s*(?:more\s+)?"
        r"(?:transactions?|txns?|payments?|purchases?|debits?|credits?|entries|items|records)\b"
    ),
    re.compile(
        r"\b(\d{1,2})\s+(?:recent|latest|last)\s+"
        r"(?:transactions?|txns?|payments?|purchases?|debits?|credits?)\b"
    ),
    re.compile(
        r"\b(?:last|latest|recent|top)\s+(\d{1,2})\b"
        r"(?!\s*(?:days?|weeks?|months?|years?|yrs?|lakhs?|lacs?|crores?|k\b|,))"
    ),
]

_TENURE = re.compile(r"\b(\d{1,3}(?:\.\d)?)\s*[- ]?(years?|yrs?|yr|months?|mos?)\b")

_MONEY = re.compile(
    r"(?<![\w.\-/])(?P<cur>₹|rs\.?|inr)?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>crores?|cr|lakhs?|lacs?|thousand|k|l)?(?![\w\-/])"
)

_EXISTING_EMI_BEFORE = re.compile(
    r"\b(existing|current|ongoing|other|already|running|paying)\b[^.\d]{0,22}\b(emis?|loans?|installments?)\b[^.\d]{0,12}$"
)
_EXISTING_EMI_AFTER = re.compile(r"^\W{0,3}(?:per month |monthly )?(?:of )?(?:existing|ongoing|current)\s+(?:emis?|loans?)")
_INCOME_BEFORE = re.compile(
    r"\b(income|salary|earn|earning|earns|earned|ctc|take[- ]?home|paid)\b[^.\d]{0,18}$"
)
_INCOME_AFTER = re.compile(r"^\W{0,3}(?:per month|monthly|a month|/month|pm|p\.m|per mth)\b")
_LOANISH_BEFORE = re.compile(r"\b(loan|borrow|need|want|amount|emi|finance|financing)\b[^.\d]{0,15}$")

_PRODUCT_PATTERNS = [
    ("personal_loan", re.compile(r"\bpersonal loans?\b")),
    ("education_loan", re.compile(r"\b(?:education(?:al)?|student|study) loans?\b")),
    ("home_loan", re.compile(r"\b(?:home|housing|house) loans?\b|\bmortgage\b")),
    ("auto_loan", re.compile(r"\b(?:auto|car|vehicle|bike|two[- ]wheeler) loans?\b")),
    ("fixed_deposit", re.compile(r"\bfixed deposits?\b|\bfds?\b")),
    ("recurring_deposit", re.compile(r"\brecurring deposits?\b|\brds?\b")),
    ("salary_account", re.compile(r"\bsalary account\b")),
    ("current_account", re.compile(r"\bcurrent account\b")),
    ("savings_account", re.compile(r"\bsavings? (?:bank )?account\b|\bsavings acc\b")),
]

_CATEGORY_PATTERNS = [
    ("Groceries", re.compile(r"\bgrocer(?:y|ies)\b|\bsupermarket\b|\bvegetables?\b|\bkirana\b")),
    ("Dining", re.compile(r"\bdining\b|\brestaurants?\b|\bfood\b|\bcafe\b|\bcoffee\b|\bbakery\b|\bswiggy\b|\bzomato\b|\beating out\b")),
    ("Utilities", re.compile(r"\butilit(?:y|ies)\b|\belectricity\b|\bwater bill\b|\bgas\b|\brecharge\b|\bbills?\b")),
    ("Shopping", re.compile(r"\bshopping\b|\bclothes\b|\bclothing\b|\bamazon\b|\bflipkart\b|\bhardware\b")),
    ("Transport", re.compile(r"\btransport\b|\bfuel\b|\bpetrol\b|\bdiesel\b|\bcab\b|\buber\b|\bola\b|\btravel\b|\bcommute\b|\bauto rickshaw\b")),
    ("Health", re.compile(r"\bhealth\b|\bmedical\b|\bpharmacy\b|\bmedicines?\b|\bhospital\b|\bdoctor\b")),
    ("Entertainment", re.compile(r"\bentertainment\b|\bmovies?\b|\bott\b|\bsubscriptions?\b|\bnetflix\b|\bgym\b")),
    ("Education", re.compile(r"\beducation\b|\bschool\b|\btuition\b|\bbooks?\b|\bfees?\b")),
    ("Salary", re.compile(r"\bsalary\b")),
    ("Other Income", re.compile(r"\brefunds?\b|\bcashback\b|\bdividends?\b|\binterest credit\b|\bbonus\b|\bfreelance\b")),
]

_CREDIT_WORDS = re.compile(r"\b(credits?|credited|received|deposits?|incoming|money in|income)\b")
_DEBIT_WORDS = re.compile(r"\b(debits?|debited|spent|spend|expenses?|withdrawals?|outgoing|money out|paid)\b")

# Which entity keys matter for which intent (everything else is dropped).
INTENT_ENTITY_KEYS = {
    "BALANCE": set(),
    "TRANSACTIONS": {"limit", "period_days", "period_label", "category", "txn_type"},
    "SPENDING_SUMMARY": {"period_days", "period_label", "category"},
    "PRODUCT_INFO": {"product_id", "product_kind", "focus"},
    "LOAN_ELIGIBILITY": {"product_id", "loan_amount", "tenure_months", "monthly_income", "existing_emi"},
    "PROFILE": set(),
    "PRIVACY": set(),
    "HELP": set(),
    "GREETING": set(),
    "OUT_OF_SCOPE": set(),
    "UNKNOWN": set(),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _overlaps(span: tuple[int, int], used: list[tuple[int, int]]) -> bool:
    return any(span[0] < u[1] and u[0] < span[1] for u in used)


def _to_number(num: str) -> float:
    return float(num.replace(",", ""))


def _clean_for_entities(text: str) -> str:
    t = (text or "").lower()
    t = _WORD_NUM_RE.sub(lambda m: _WORD_NUMBERS[m.group(1)], t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_entities(text: str, today: date | None = None) -> dict:
    today = today or date.today()
    t = _clean_for_entities(text)
    ents: dict = {}
    used: list[tuple[int, int]] = []

    # --- period (explicit "last N days", "last month", "this month"...) ---
    m = _PERIOD_N.search(t)
    if m:
        n = int(m.group(1))
        unit = m.group(2).rstrip("s")
        days = n * _UNIT_DAYS.get(unit, 1)
        if 1 <= days <= 365:
            ents["period_days"] = days
            ents["period_label"] = f"last {n} {m.group(2) if n != 1 else unit}"
            used.append(m.span())
    if "period_days" not in ents:
        for pattern, days, label in _PERIOD_WORDS:
            pm = pattern.search(t)
            if pm:
                if days is None:
                    if label == "this week":
                        days = today.weekday() + 1
                    elif label == "this month":
                        days = today.day
                    else:  # this year
                        days = today.timetuple().tm_yday
                ents["period_days"] = days
                ents["period_label"] = label
                used.append(pm.span())
                break

    # --- limit ("last 5 transactions") ---
    for pattern in _LIMIT_PATTERNS:
        lm = pattern.search(t)
        if lm and not _overlaps(lm.span(1), used):
            value = int(lm.group(1))
            if 1 <= value <= MAX_CHAT_LIMIT:
                ents["limit"] = value
                used.append(lm.span())
                break

    # --- tenure ("3 years", "36 months") ---
    for tm in _TENURE.finditer(t):
        if _overlaps(tm.span(), used):
            continue
        value = float(tm.group(1))
        months = round(value * 12) if tm.group(2).startswith(("y",)) else round(value)
        if 1 <= months <= 480:
            ents["tenure_months"] = months
            used.append(tm.span())
            break

    # --- money (loan amount / income / existing EMI) ---
    for mm in _MONEY.finditer(t):
        if _overlaps(mm.span(), used):
            continue
        cur, num, unit = mm.group("cur"), mm.group("num"), mm.group("unit")
        try:
            value = _to_number(num)
        except ValueError:
            continue
        has_marker = bool(cur or unit or "," in num)
        digits_only = re.sub(r"\D", "", num.split(".")[0])
        if not has_marker:
            if len(digits_only) < 4:
                continue  # too small to be money
            if 1990 <= value <= 2100:
                continue  # looks like a year
        if unit:
            value *= _UNIT_VALUE[unit]
        if not (1 <= value <= 1_000_000_000):
            continue

        before = t[max(0, mm.start() - 32): mm.start()]
        after = t[mm.end(): mm.end() + 26]
        if _EXISTING_EMI_BEFORE.search(before) or _EXISTING_EMI_AFTER.search(after):
            role = "existing_emi"
        elif _INCOME_BEFORE.search(before) or (
            _INCOME_AFTER.search(after) and not _LOANISH_BEFORE.search(before)
        ):
            role = "monthly_income"
        else:
            role = "loan_amount"
        ents.setdefault(role, round(value, 2) if value % 1 else int(value))
        used.append(mm.span())

    # --- product ---
    for pid, pattern in _PRODUCT_PATTERNS:
        if pattern.search(t):
            ents["product_id"] = pid
            break
    if "product_id" not in ents:
        if re.search(r"\bloans?\b", t):
            ents["product_kind"] = "loan"
        elif re.search(r"\bdeposits?\b", t):
            ents["product_kind"] = "deposit"
        elif re.search(r"\baccounts?\b", t) and re.search(r"\b(types?|open|offer|kinds?|options?)\b", t):
            ents["product_kind"] = "account"
    if re.search(r"\b(interest|rates?|roi|byaj)\b", t):
        ents["focus"] = "rates"

    # --- category (not when a product word explains the match, e.g. "education loan") ---
    if "product_id" not in ents:
        for cat, pattern in _CATEGORY_PATTERNS:
            if pattern.search(t):
                ents["category"] = cat
                break

    # --- transaction type ---
    has_credit = bool(_CREDIT_WORDS.search(t))
    has_debit = bool(_DEBIT_WORDS.search(t))
    if has_credit and not has_debit:
        ents["txn_type"] = "Credit"
    elif has_debit and not has_credit:
        ents["txn_type"] = "Debit"

    return ents


def keep_relevant(intent: str, entities: dict) -> dict:
    """Drop entities that are not meaningful for this intent."""
    allowed = INTENT_ENTITY_KEYS.get(intent, set())
    return {k: v for k, v in entities.items() if k in allowed}


# ---------------------------------------------------------------------------
# Validation of entities coming from an LLM (or from carried-over context)
# ---------------------------------------------------------------------------


def sanitize_llm_entities(raw: object) -> dict:
    """
    Accept only NON-numeric-money entities from the LLM and validate each one.
    Money, tenure, income and EMI are never taken from the LLM.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict = {}

    pid = raw.get("product_id")
    if isinstance(pid, str) and pid in product_catalogue.product_ids():
        out["product_id"] = pid

    kind = raw.get("product_kind")
    if isinstance(kind, str) and kind in ("loan", "deposit", "account"):
        out["product_kind"] = kind

    focus = raw.get("focus")
    if focus in ("rates", "features"):
        out["focus"] = focus

    cat = raw.get("category")
    if isinstance(cat, str):
        for allowed in ALL_CATEGORIES:
            if cat.strip().lower() == allowed.lower():
                out["category"] = allowed

    ttype = raw.get("txn_type")
    if isinstance(ttype, str):
        for allowed in TXN_TYPES:
            if ttype.strip().lower() == allowed.lower():
                out["txn_type"] = allowed

    for key, lo, hi in (("limit", 1, MAX_CHAT_LIMIT), ("period_days", 1, 365)):
        value = raw.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and lo <= int(value) <= hi:
            out[key] = int(value)
    if "period_days" in out and "period_label" not in out:
        out["period_label"] = f"last {out['period_days']} days"
    return out
