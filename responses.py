"""
BANKMATE AI — responses.py
Deterministic answer templates.

These are used (a) when no LLM is available / allowed, and (b) as the safe
fallback when an LLM answer fails the guardrail. Every number comes from
services.py - nothing is calculated here.
"""

from __future__ import annotations

from config import CHAT_MAX_CHARS
from privacy import (
    BLOCK_CREDENTIALS, BLOCK_EMPTY, BLOCK_INJECTION, BLOCK_OTHER_CUSTOMER, BLOCK_TOO_LONG,
)

DATA_FOOTER = "*Synthetic demo data — not real banking information.*"
PRODUCT_FOOTER = "*Illustrative rates and limits only — not an offer, approval or financial advice.*"

HELP_TEXT = (
    "I can help you with:\n\n"
    "- **Balance** — *\"What is my balance?\"*\n"
    "- **Transactions** — *\"Show my last 5 transactions\"*, *\"Show my grocery payments this month\"*\n"
    "- **Spending summary** — *\"How much did I spend in the last 30 days?\"*\n"
    "- **Interest rates & products** — *\"What is the home loan interest rate?\"*, *\"FD rates\"*\n"
    "- **Loan eligibility & EMI** — *\"Can I get a 5 lakh personal loan for 3 years?\"*\n"
    "- **Profile** — *\"Show my profile\"*\n"
    "- **Privacy & consent** — *\"What is my consent status?\"*\n\n"
    "I will never ask for your OTP, PIN, CVV or password."
)

GREETING_TEXT = (
    "Hello! 👋 I'm BANKMATE AI, your demo banking assistant. "
    "Ask me about your balance, transactions, spending, loan eligibility or interest rates."
)

UNKNOWN_TEXT = (
    "I'm not sure I understood that. I can answer questions about your balance, transactions, "
    "spending, products and interest rates, and loan eligibility. Try one of these:\n\n"
    "- *What is my balance?*\n- *Show my last 5 transactions*\n- *What is the personal loan interest rate?*"
)

OUT_OF_SCOPE_TEXT = (
    "I can only help with banking questions for this demo bank (balance, transactions, spending, "
    "products, interest rates and loan eligibility). I can't give investment or general advice."
)

BLOCK_TEXT = {
    BLOCK_CREDENTIALS: (
        "🔒 For your safety I can't help with OTPs, PINs, CVV or passwords — and I will never ask for them. "
        "Please don't type them in this chat. If you need to reset or block something, "
        "use your bank's official app or helpline."
    ),
    BLOCK_OTHER_CUSTOMER: (
        "🔒 I can only show information about **your own** account. "
        "I can't share anyone else's balance, transactions or details."
    ),
    BLOCK_INJECTION: (
        "I can't change my rules or share my internal instructions. "
        "I'm happy to help with your balance, transactions, spending, loans or interest rates."
    ),
    BLOCK_EMPTY: "Please type a question, for example: *What is my balance?*",
    BLOCK_TOO_LONG: f"Your message is too long (maximum {CHAT_MAX_CHARS} characters). Please shorten it.",
}


def _filter_text(filters: dict) -> str:
    parts = []
    if filters.get("category"):
        parts.append(f"category **{filters['category']}**")
    if filters.get("type"):
        parts.append(f"type **{filters['type']}**")
    if filters.get("period"):
        parts.append(f"**{filters['period']}**")
    return (" (" + ", ".join(parts) + ")") if parts else ""


def render_balance(f: dict) -> str:
    if not f["accounts"]:
        return "I couldn't find any accounts for you."
    lines = ["Here is your balance:\n"]
    for a in f["accounts"]:
        lines.append(f"- **{a['account_type']}** account ({a['account_masked']}): **{a['balance_display']}**")
    if f["account_count"] > 1:
        lines.append(f"\nTotal across accounts: **{f['total_display']}**")
    return "\n".join(lines)


def render_transactions(f: dict) -> str:
    filt = _filter_text(f["filters"])
    if f["count"] == 0:
        return f"I couldn't find any transactions{filt}."
    lines = [f"Here are your **{f['count']}** most recent transactions{filt}:\n"]
    for t in f["transactions"]:
        sign, icon = ("+", "🟢") if t["type"] == "Credit" else ("-", "🔴")
        lines.append(f"- {icon} **{t['date']}** — {t['description']}: **{sign}{t['amount_display']}**")
    return "\n".join(lines)


def render_spending(f: dict) -> str:
    cat = f" on **{f['category']}**" if f.get("category") else ""
    if f["transaction_count"] == 0:
        return f"I found no transactions{cat} for the **{f['period_label']}**."
    lines = [
        f"**Spending summary — {f['period_label']}**{(' (' + f['category'] + ')') if f.get('category') else ''}\n",
        f"- Total spent: **{f['total_debit_display']}** across {f['debit_count']} payments",
        f"- Total received: **{f['total_credit_display']}** across {f['credit_count']} credits",
        f"- Net flow: **{f['net_flow_display']}**",
    ]
    if f["top_categories"] and not f.get("category"):
        lines.append("\nTop spending categories:")
        for c in f["top_categories"][:4]:
            lines.append(f"- {c['category']}: **{c['total_display']}** ({c['share_display']})")
    return "\n".join(lines)


def _product_detail(p: dict) -> list[str]:
    lines = [f"**{p['name']}** — {p['description']}", f"- Interest rate: **{p['rate_display']}**"]
    if p.get("tenure_display"):
        lines.append(f"- Tenure: {p['tenure_display']}")
    if p.get("amount_display"):
        lines.append(f"- Amount: {p['amount_display']}")
    if p.get("minimum_balance_display"):
        lines.append(f"- Minimum balance: {p['minimum_balance_display']}")
    if p.get("processing_fee"):
        lines.append(f"- Processing fee: {p['processing_fee']}")
    if p.get("min_monthly_income"):
        lines.append(f"- Minimum income: ₹{p['min_monthly_income']:,} per month")
    if p.get("features"):
        lines.append("- Features: " + "; ".join(p["features"]))
    if p.get("documents_required"):
        lines.append("- Documents: " + ", ".join(p["documents_required"]))
    return lines


def render_products(f: dict, entities: dict | None = None) -> str:
    entities = entities or {}
    if f["count"] == 0:
        return "I couldn't find that product. I can tell you about our accounts, deposits and loans."
    if f["count"] == 1:
        return "\n".join(_product_detail(f["products"][0]))
    rates_only = entities.get("focus") == "rates"
    lines = ["Here is what we offer:\n" if not rates_only else "Here are the interest rates:\n"]
    for p in f["products"]:
        extra = f"; tenure {p['tenure_display']}" if (p.get("tenure_display") and not rates_only) else ""
        lines.append(f"- **{p['name']}** — {p['rate_display']}{extra}")
    lines.append("\nAsk about any product by name for full details.")
    return "\n".join(lines)


_DECISION_TEXT = {
    "likely_eligible": "✅ **Likely eligible** (illustrative)",
    "may_not_be_eligible": "⚠️ **May not be eligible** at this amount/tenure (illustrative)",
    "not_eligible": "❌ **Not eligible** based on the checks below (illustrative)",
    "estimate_only": "ℹ️ **Indicative estimate**",
}


def render_eligibility(f: dict) -> str:
    if f.get("status") == "needs_info":
        opts = ", ".join(f.get("options", []))
        return f"{f['message']}" + (f"\n\nOptions: {opts}." if opts else "")

    lines = [f"**{f['product_name']} — illustrative eligibility check**\n"]
    lines.append(f"- Monthly income used: **{f['income_used_display']}** ({f['income_source']})")
    lines.append(f"- Existing EMIs: **{f['existing_emi_display']}** per month")
    if f.get("requested_amount"):
        lines.append(f"- Loan amount: **{f['requested_amount_display']}**")
    tenure_note = " (assumed — tell me your preferred tenure)" if f["tenure_assumed"] else ""
    lines.append(f"- Tenure: **{f['tenure_display']}**{tenure_note}")
    if f.get("emi_at_min_rate_display"):
        lines.append(
            f"- Estimated EMI: **{f['emi_at_min_rate_display']}** at {f['rate_min_display']} "
            f"to **{f['emi_at_max_rate_display']}** at {f['rate_max_display']}"
        )
    lines.append(f"- EMI room ({f['max_foir_pct']}% of income after existing EMIs): **{f['max_affordable_emi_display']}**")
    lines.append(f"\n{_DECISION_TEXT[f['decision']]}")
    if f["decision"] != "estimate_only" and f["failed_checks"]:
        for c in f["checks"]:
            if not c["passed"]:
                lines.append(f"- ❗ {c['name']}: {c['detail']}")
    lines.append(
        f"\nIndicative maximum loan for {f['tenure_display']} (at {f['rate_max_display']}): "
        f"**{f['max_eligible_amount_display']}**"
    )
    lines.append("\nThis is only an estimate, not a loan approval.")
    return "\n".join(lines)


def _on_off(flag: bool) -> str:
    return "ON ✅" if flag else "OFF ⛔"


def render_profile(f: dict) -> str:
    lines = [
        "**Your profile**\n",
        f"- Name: **{f['name']}**",
        f"- Customer ID: {f['customer_id']}",
        f"- E-mail: {f['email_masked']}",
        f"- Phone: {f['phone_masked']}",
        f"- Declared monthly income: {f['declared_monthly_income_display']}",
    ]
    for a in f["accounts"]:
        lines.append(f"- {a['account_type']} account {a['account_masked']} ({a['status']})")
    c = f["consent"]
    lines.append(
        f"- Consent: AI processing {_on_off(c['ai_processing'])}, query logging {_on_off(c['query_logging'])}, "
        f"marketing {_on_off(c['marketing'])}"
    )
    return "\n".join(lines)


def render_privacy(f: dict) -> str:
    c = f["consent"]
    lines = [
        "**Your privacy & consent settings**\n",
        f"- **AI processing** (Groq language model): {_on_off(c['ai_processing'])} — "
        + ("your question text (with personal values removed) and the facts needed to answer may be "
           "sent to the language model."
           if c["ai_processing"] else
           "nothing is sent to a language model; I answer with built-in rules."),
        f"- **Query logging**: {_on_off(c['query_logging'])} — "
        + ("a privacy-masked copy of your question is stored to improve the service."
           if c["query_logging"] else "your question text is not stored."),
        f"- **Marketing suggestions**: {_on_off(c['marketing'])}",
        "\nNever sent to the language model: your name, e-mail, phone, full account number, "
        "and any OTP/PIN/password. You can change these settings any time in the **Privacy & Consent** tab.",
    ]
    return "\n".join(lines)


def render(intent: str, facts: dict, entities: dict | None = None) -> str:
    """Deterministic reply for an intent whose tool has produced `facts`."""
    from nlp import (
        BALANCE, GREETING, HELP, LOAN_ELIGIBILITY, OUT_OF_SCOPE, PRIVACY, PRODUCT_INFO,
        PROFILE, SPENDING_SUMMARY, TRANSACTIONS,
    )

    if intent == BALANCE:
        return render_balance(facts)
    if intent == TRANSACTIONS:
        return render_transactions(facts)
    if intent == SPENDING_SUMMARY:
        return render_spending(facts)
    if intent == PRODUCT_INFO:
        return render_products(facts, entities)
    if intent == LOAN_ELIGIBILITY:
        return render_eligibility(facts)
    if intent == PROFILE:
        return render_profile(facts)
    if intent == PRIVACY:
        return render_privacy(facts)
    if intent == HELP:
        return HELP_TEXT
    if intent == GREETING:
        return GREETING_TEXT
    if intent == OUT_OF_SCOPE:
        return OUT_OF_SCOPE_TEXT
    return UNKNOWN_TEXT
