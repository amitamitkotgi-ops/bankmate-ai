"""
BANKMATE AI — services.py
Business logic shared by the REST API and the LangGraph agent tools.

Rules enforced here (so no caller can forget them):
  * customer_id always comes from the authenticated session, never from the LLM
  * account numbers, e-mail and phone are masked before leaving this layer
  * every number the assistant may quote is calculated here, not by the LLM
"""

from __future__ import annotations

import math

import database as db
import products as catalogue
from privacy import mask_account, mask_email, mask_phone
from utils import fmt_inr, fmt_months, fmt_pct

DATA_NOTICE = (
    "All data shown is entirely synthetic and for demonstration purposes only. "
    "It does not represent real customer accounts or transactions."
)


class ServiceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _require_customer(customer_id: str) -> dict:
    customer = db.get_customer(customer_id)
    if not customer:
        raise ServiceError("customer_not_found", "Customer record not found.")
    return customer


# ---------------------------------------------------------------------------
# Accounts, transactions, analytics
# ---------------------------------------------------------------------------


def get_balance(customer_id: str) -> dict:
    _require_customer(customer_id)
    accounts = db.get_accounts(customer_id)
    items = [
        {
            "account_masked": mask_account(a["account_id"]),
            "account_type": a["account_type"],
            "balance": a["balance"],
            "balance_display": fmt_inr(a["balance"]),
            "currency": a["currency"],
            "status": a["status"],
        }
        for a in accounts
    ]
    total = round(sum(a["balance"] for a in accounts), 2)
    return {
        "accounts": items,
        "account_count": len(items),
        "total_balance": total,
        "total_display": fmt_inr(total),
    }


def get_transactions(customer_id: str, limit: int = 5, period_days: int | None = None,
                     category: str | None = None, txn_type: str | None = None,
                     period_label: str | None = None) -> dict:
    _require_customer(customer_id)
    rows = db.get_transactions(customer_id, limit=limit, days=period_days,
                               category=category, txn_type=txn_type)
    items = [
        {
            "transaction_id": r["transaction_id"],
            "account_masked": mask_account(r["account_id"]),
            "date": r["transaction_date"],
            "description": r["description"],
            "category": r["category"],
            "amount": r["amount"],
            "amount_display": fmt_inr(r["amount"]),
            "type": r["transaction_type"],
            "status": r["status"],
        }
        for r in rows
    ]
    return {
        "count": len(items),
        "transactions": items,
        "filters": {
            "limit": limit,
            "period": period_label if period_days else None,
            "category": category,
            "type": txn_type,
        },
    }


def get_spending_summary(customer_id: str, period_days: int | None = 30,
                         category: str | None = None, period_label: str | None = None) -> dict:
    _require_customer(customer_id)
    s = db.get_analytics_summary(customer_id, days=period_days, category=category)
    total_debit = s["total_debit"]
    top = []
    for row in s["by_category"][:5]:
        share = round(row["total"] / total_debit * 100, 1) if total_debit else 0.0
        top.append({
            "category": row["category"],
            "total": row["total"],
            "total_display": fmt_inr(row["total"]),
            "count": row["count"],
            "share_pct": share,
            "share_display": fmt_pct(share),
        })
    if not period_days:
        label = "all available history"
    else:
        label = period_label or f"last {period_days} days"
    return {
        "period_label": label,
        "period_days": period_days,
        "category": category,
        "transaction_count": s["transaction_count"],
        "debit_count": s["debit_count"],
        "credit_count": s["credit_count"],
        "total_credit": s["total_credit"],
        "total_credit_display": fmt_inr(s["total_credit"]),
        "total_debit": total_debit,
        "total_debit_display": fmt_inr(total_debit),
        "net_flow": s["net_flow"],
        "net_flow_display": fmt_inr(s["net_flow"]),
        "top_categories": top,
        "currency": "INR",
    }


# ---------------------------------------------------------------------------
# Product information (public information)
# ---------------------------------------------------------------------------


def get_products(product_id: str | None = None, kind: str | None = None) -> dict:
    if product_id:
        p = catalogue.get_product(product_id)
        items = [catalogue.with_display(p)] if p else []
    else:
        items = [catalogue.with_display(p) for p in catalogue.list_products(kind)]
    return {
        "count": len(items),
        "products": items,
        "requested_product_id": product_id,
        "requested_kind": kind,
        "disclaimer": catalogue.disclaimer(),
    }


# ---------------------------------------------------------------------------
# Loan eligibility (illustrative maths, never an approval)
# ---------------------------------------------------------------------------


def calc_emi(principal: float, annual_rate_pct: float, months: int) -> float:
    """Standard reducing-balance EMI formula."""
    if months <= 0:
        raise ValueError("months must be positive")
    r = annual_rate_pct / 12 / 100
    if r == 0:
        return principal / months
    growth = (1 + r) ** months
    return principal * r * growth / (growth - 1)


def max_principal_for_emi(emi: float, annual_rate_pct: float, months: int) -> float:
    """Largest loan whose EMI is <= `emi`."""
    if emi <= 0 or months <= 0:
        return 0.0
    r = annual_rate_pct / 12 / 100
    if r == 0:
        return emi * months
    growth = (1 + r) ** months
    return emi * (growth - 1) / (r * growth)


def _default_tenure(product: dict) -> int:
    return min(product["max_tenure_months"], max(product["min_tenure_months"], 60))


def check_loan_eligibility(customer_id: str, product_id: str | None = None,
                           loan_amount: float | None = None, tenure_months: int | None = None,
                           monthly_income: float | None = None,
                           existing_emi: float | None = None) -> dict:
    customer = _require_customer(customer_id)

    if not product_id:
        return {"status": "needs_info", "missing": ["loan_type"],
                "message": "Which loan type? Personal, Education, Home or Auto.",
                "options": [p["name"] for p in catalogue.list_products("loan")]}
    product = catalogue.get_product(product_id)
    if not product or product["category"] != "loan":
        return {"status": "needs_info", "missing": ["loan_type"],
                "message": "That is not a loan product. Choose Personal, Education, Home or Auto.",
                "options": [p["name"] for p in catalogue.list_products("loan")]}

    income_source = "stated in this chat"
    if monthly_income is None:
        monthly_income = customer["declared_monthly_income"]
        income_source = "declared in your profile"
    if existing_emi is None:
        existing_emi = customer["existing_emi"]

    tenure_assumed = tenure_months is None
    months = tenure_months if tenure_months else _default_tenure(product)

    rate_min, rate_max = product["rate_min"], product["rate_max"]
    max_foir = product.get("max_foir", 0.5)
    affordable_emi = max(0.0, round(max_foir * monthly_income - existing_emi, 2))

    # Conservative estimate: use the HIGHEST illustrative rate.
    max_amount_by_income = math.floor(max_principal_for_emi(affordable_emi, rate_max, months) / 1000) * 1000
    max_eligible = int(max(0, min(max_amount_by_income, product["max_amount"])))

    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str, hard: bool = True) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail, "hard": hard})

    age = customer.get("age")
    if age is not None:
        add("Age", product["min_age"] <= age <= product["max_age"],
            f"Allowed age {product['min_age']}–{product['max_age']} years")
    add("Monthly income", monthly_income >= product.get("min_monthly_income", 0),
        f"Minimum income {fmt_inr(product.get('min_monthly_income', 0), 0)} per month")
    add("Tenure", product["min_tenure_months"] <= months <= product["max_tenure_months"],
        f"Allowed tenure {fmt_months(product['min_tenure_months'])} – {fmt_months(product['max_tenure_months'])}")

    emi_min = emi_max = None
    if loan_amount:
        add("Loan amount", product["min_amount"] <= loan_amount <= product["max_amount"],
            f"Allowed amount {fmt_inr(product['min_amount'], 0)} – {fmt_inr(product['max_amount'], 0)}")
        emi_min = round(calc_emi(loan_amount, rate_min, months), 2)
        emi_max = round(calc_emi(loan_amount, rate_max, months), 2)
        add("Affordability", emi_max <= affordable_emi,
            f"EMI should stay within {int(max_foir * 100)}% of income after existing EMIs "
            f"(room for {fmt_inr(affordable_emi)} per month)", hard=False)

    hard_failed = [c for c in checks if c["hard"] and not c["passed"]]
    soft_failed = [c for c in checks if not c["hard"] and not c["passed"]]
    if loan_amount is None:
        decision = "estimate_only"
    elif hard_failed:
        decision = "not_eligible"
    elif soft_failed:
        decision = "may_not_be_eligible"
    else:
        decision = "likely_eligible"

    return {
        "status": "ok",
        "product_id": product["product_id"],
        "product_name": product["name"],
        "decision": decision,
        "income_used": monthly_income,
        "income_used_display": fmt_inr(monthly_income, 0),
        "income_source": income_source,
        "existing_emi": existing_emi,
        "existing_emi_display": fmt_inr(existing_emi, 0),
        "requested_amount": loan_amount,
        "requested_amount_display": fmt_inr(loan_amount, 0) if loan_amount else None,
        "tenure_months": months,
        "tenure_display": fmt_months(months),
        "tenure_assumed": tenure_assumed,
        "rate_min": rate_min,
        "rate_max": rate_max,
        "rate_min_display": fmt_pct(rate_min),
        "rate_max_display": fmt_pct(rate_max),
        "emi_at_min_rate": emi_min,
        "emi_at_min_rate_display": fmt_inr(emi_min) if emi_min is not None else None,
        "emi_at_max_rate": emi_max,
        "emi_at_max_rate_display": fmt_inr(emi_max) if emi_max is not None else None,
        "max_affordable_emi": affordable_emi,
        "max_affordable_emi_display": fmt_inr(affordable_emi),
        "max_foir_pct": int(max_foir * 100),
        "max_eligible_amount": max_eligible,
        "max_eligible_amount_display": fmt_inr(max_eligible, 0),
        "checks": checks,
        "failed_checks": [c["name"] for c in checks if not c["passed"]],
        "disclaimer": catalogue.disclaimer(),
    }


# ---------------------------------------------------------------------------
# Profile and consent
# ---------------------------------------------------------------------------


def get_profile(customer_id: str) -> dict:
    c = _require_customer(customer_id)
    accounts = db.get_accounts(customer_id)
    return {
        "customer_id": c["customer_id"],
        "name": c["name"],
        "email_masked": mask_email(c["email"]),
        "phone_masked": mask_phone(c["phone"]),
        "age": c["age"],
        "declared_monthly_income": c["declared_monthly_income"],
        "declared_monthly_income_display": fmt_inr(c["declared_monthly_income"], 0),
        "existing_emi_display": fmt_inr(c["existing_emi"], 0),
        "accounts": [
            {"account_masked": mask_account(a["account_id"]),
             "account_type": a["account_type"], "status": a["status"]}
            for a in accounts
        ],
        "consent": c["consent"],
    }


def get_consent(customer_id: str) -> dict:
    consent = db.get_consent(customer_id)
    if consent is None:
        raise ServiceError("customer_not_found", "Customer record not found.")
    return consent


def set_consent(customer_id: str, username: str | None, ai_processing: bool | None = None,
                query_logging: bool | None = None, marketing: bool | None = None) -> dict:
    before = get_consent(customer_id)
    after = db.update_consent(customer_id, ai_processing, query_logging, marketing)
    if after is None:
        raise ServiceError("customer_not_found", "Customer record not found.")
    changes = [k for k in ("ai_processing", "query_logging", "marketing") if before[k] != after[k]]
    if changes:
        detail = ", ".join(f"{k}={'on' if after[k] else 'off'}" for k in changes)
        db.write_audit("CONSENT_CHANGED", username, customer_id, detail)
    return after


def get_privacy_info(customer_id: str) -> dict:
    consent = get_consent(customer_id)
    return {
        "consent": consent,
        "what_is_processed": [
            "Your question text, after card numbers, phone numbers, e-mail, PAN, Aadhaar "
            "and similar values are removed.",
            "Only the amounts and labels needed to answer (never your full e-mail, phone or full account number).",
        ],
        "never_sent_to_llm": ["Profile details (name, e-mail, phone)", "Passwords, OTPs, PINs, CVV"],
        "llm_provider": "Groq (only when AI processing consent is ON)",
    }


# ---------------------------------------------------------------------------
# Chat event logging (consent-aware)
# ---------------------------------------------------------------------------


def record_chat_event(customer_id: str, username: str | None, masked_query: str, intent: str | None,
                      tool: str | None, mode: str, llm_used: bool, blocked: str | None,
                      pii_types: list[str]) -> None:
    """
    Audit log: ALWAYS records metadata only (intent, mode) — never message text.
    Query log: stores the PII-redacted text ONLY if the customer consented.
    """
    db.write_audit(
        "CHAT_BLOCKED" if blocked else "CHAT_TURN", username, customer_id,
        f"intent={intent} tool={tool} mode={mode} llm={int(llm_used)} blocked={blocked or '-'} "
        f"pii={','.join(pii_types) or '-'}",
    )
    consent = db.get_consent(customer_id)
    if consent and consent["query_logging"]:
        db.log_query(customer_id, masked_query, intent, tool, mode, llm_used, blocked, pii_types)
