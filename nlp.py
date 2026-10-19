"""
BANKMATE AI — nlp.py
Rule-based intent classifier.

This is the OFFLINE FALLBACK. The main path uses the Groq LLM (llm.py) to
understand the question. This classifier is used when:
  * no GROQ_API_KEY is set,
  * the customer has not consented to AI (LLM) processing,
  * Groq is down / rate-limited / times out.

It needs no network and no API key.
"""

from __future__ import annotations

import re
from datetime import date

from entities import extract_entities, keep_relevant

# ---------------------------------------------------------------------------
# Intent constants
# ---------------------------------------------------------------------------

BALANCE = "BALANCE"
TRANSACTIONS = "TRANSACTIONS"
SPENDING_SUMMARY = "SPENDING_SUMMARY"
PRODUCT_INFO = "PRODUCT_INFO"
LOAN_ELIGIBILITY = "LOAN_ELIGIBILITY"
PROFILE = "PROFILE"
PRIVACY = "PRIVACY"
HELP = "HELP"
GREETING = "GREETING"
OUT_OF_SCOPE = "OUT_OF_SCOPE"
UNKNOWN = "UNKNOWN"

# Kept for backward compatibility with the old version of this project.
LOANS = PRODUCT_INFO
CUSTOMER_INFORMATION = PROFILE

ALL_INTENTS = [
    BALANCE, TRANSACTIONS, SPENDING_SUMMARY, PRODUCT_INFO, LOAN_ELIGIBILITY,
    PROFILE, PRIVACY, HELP, GREETING, OUT_OF_SCOPE, UNKNOWN,
]

# Intents the LLM is allowed to return.
LLM_INTENTS = [i for i in ALL_INTENTS if i != UNKNOWN]

# Intents that use the customer's own account data.
PERSONAL_DATA_INTENTS = {BALANCE, TRANSACTIONS, SPENDING_SUMMARY, PROFILE, LOAN_ELIGIBILITY}


def _p(*patterns: str) -> list[re.Pattern]:
    return [re.compile(p) for p in patterns]


# Order matters: more specific intents first.
_RULES: list[tuple[str, list[re.Pattern]]] = [
    (PRIVACY, _p(
        r"\bconsents?\b", r"\bprivacy\b", r"\bopt (in|out)\b",
        r"\bdata (sharing|protection|usage|policy|stored|collected)\b",
        r"\bhow is my data\b", r"\bwhat data (do|are) you\b",
        r"\bwhat do you (store|know|keep) about me\b", r"\bai processing\b",
    )),
    (LOAN_ELIGIBILITY, _p(
        r"\beligib", r"\bqualif(y|ied|ication)\b",
        r"\bcan i (get|take|apply for|avail|borrow)\b.*\b(loan|credit|finance)\b",
        r"\bhow much (loan|can i borrow|can i get)\b",
        r"\bmax(imum)? loan\b", r"\bemi\b", r"\bwill i (get|be approved)\b",
        r"\bloan amount i can\b", r"\bafford\b.*\bloan\b",
    )),
    (SPENDING_SUMMARY, _p(
        r"\bspending\b", r"\bhow much (did|have|do) i (spend|spent)\b",
        r"\bhow much i (spent|spend)\b", r"\bmy (total )?expenses?\b",
        r"\btotal (debits?|expenses?|spend)\b", r"\bsummar(y|ise|ize)\b",
        r"\boutflow\b", r"\bnet (cash )?flow\b", r"\bbreakdown\b",
        r"\bcategory wise\b", r"\bkharch", r"\bexpenditure\b",
        r"\b(spent|spend) on\b",
    )),
    (BALANCE, _p(
        r"\bbalance\b", r"\bhow much (money|funds|cash|do i have)\b",
        r"\bhow much (is )?(in|there)\b", r"\bavailable (funds|amount|money)\b",
        r"\bfunds available\b", r"\bkitna (paisa|paise|money)\b",
        r"\bcheck (my )?funds\b",
    )),
    (TRANSACTIONS, _p(
        r"\btransactions?\b", r"\btxns?\b",
        r"\brecent (payments?|purchases?|activity)\b",
        r"\bpayment history\b", r"\bpurchase history\b",
        r"\blast (few |\d+ )?(payments?|purchases?|debits?|credits?)\b",
        r"\bshow (my )?(payments?|purchases?|debits?|credits?)\b",
        r"\blist (my )?(payments?|debits?|credits?)\b",
        r"\bwhere did i spend\b", r"\blatest (payments?|activity|debits?|credits?)\b",
        r"\bstatement\b", r"\bhistory\b", r"\bwhat were my (last|recent)\b", r"\bdikhao\b",
        # plain "debits / credits / payments" (but not loan payments, which are product questions)
        r"^(?!.*\bloans?\b).*\b(debits|credits|payments|purchases|withdrawals)\b",
    )),
    (PRODUCT_INFO, _p(
        r"\bloans?\b", r"\binterest\b", r"\brate of interest\b", r"\broi\b", r"\bbyaj\b",
        r"\bfds?\b", r"\brds?\b", r"\bfixed deposits?\b", r"\brecurring deposits?\b",
        r"\b(savings|current|salary) accounts?\b", r"\bprocessing fees?\b",
        r"\btenure\b", r"\bproducts?\b", r"\boffer(s|ing|ings)?\b",
        r"\baccount types?\b", r"\bwhich accounts\b", r"\bwhat accounts\b",
        r"\baccounts? (can|do) (i|you)\b", r"\bcompare\b", r"\bmortgage\b",
        r"\bdocuments? required\b", r"\bdeposits?\b",
    )),
    (PROFILE, _p(
        r"\bprofile\b", r"\bcustomer (info|information|details?|data|id)\b",
        r"\bmy (info|information|details?|data|contact|name|email|phone|address)\b",
        r"\bwho am i\b", r"\bcontact (details?|info)\b",
        r"\bpersonal (details?|info|information)\b",
        r"\baccount (info|information|details?)\b",
    )),
    (HELP, _p(
        r"\bhelp\b", r"\bwhat can you\b", r"\bwhat (do|can|will) you do\b",
        r"\bwhat questions\b", r"\bhow (to|do i) use\b", r"\bcommands?\b",
        r"\bfeatures?\b", r"\bcapabilit", r"\bget started\b", r"\bguide\b",
    )),
]

_GREETING = re.compile(
    r"^(hi+|hello+|hey+|hola|namaste|namaskar|good (morning|afternoon|evening|day))\b"
)

_OUT_OF_SCOPE = _p(
    r"\bweather\b", r"\bjokes?\b", r"\b(cricket|football|ipl|match score)\b",
    r"\b(stocks?|shares?|bitcoin|crypto|ethereum|mutual funds?|invest)\b",
    r"\b(poem|story|song|recipe|movie recommendation)\b", r"\bnews\b",
)


def normalize(text: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace."""
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_intent(user_input: str) -> str:
    """Return one intent label for `user_input` (rule-based)."""
    if not user_input or not user_input.strip():
        return UNKNOWN
    text = normalize(user_input)

    for intent, patterns in _RULES:
        if any(p.search(text) for p in patterns):
            return intent

    # A greeting counts only when nothing else was asked ("hey show my profile" is a profile request).
    if len(text.split()) <= 4 and _GREETING.match(text):
        return GREETING

    if any(p.search(text) for p in _OUT_OF_SCOPE):
        return OUT_OF_SCOPE
    return UNKNOWN


def rule_based_parse(user_input: str, today: date | None = None) -> dict:
    """Return {'intent': ..., 'entities': {...}} using only rules."""
    intent = classify_intent(user_input)
    entities = keep_relevant(intent, extract_entities(user_input, today=today))
    return {"intent": intent, "entities": entities}
