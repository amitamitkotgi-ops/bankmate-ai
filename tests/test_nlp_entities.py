"""Entity extraction and rule-based intent classification."""

from datetime import date

import pytest

from entities import extract_entities, keep_relevant, sanitize_llm_entities
from nlp import (
    BALANCE, GREETING, HELP, LOAN_ELIGIBILITY, OUT_OF_SCOPE, PRIVACY, PRODUCT_INFO, PROFILE,
    SPENDING_SUMMARY, TRANSACTIONS, UNKNOWN, classify_intent, rule_based_parse,
)

TODAY = date(2026, 10, 12)


@pytest.mark.parametrize("text,intent", [
    ("What is my account balance?", BALANCE),
    ("How much money do I have?", BALANCE),
    ("check my balance", BALANCE),
    ("kitna paisa hai mere account mein", BALANCE),
    ("Show my recent transactions.", TRANSACTIONS),
    ("Where did I spend my money?", TRANSACTIONS),
    ("show my statement", TRANSACTIONS),
    ("what were my last 5 debits", TRANSACTIONS),
    ("show my grocery transactions", TRANSACTIONS),
    ("show my grocery debits from the last 90 days", TRANSACTIONS),
    ("my payments this month", TRANSACTIONS),
    ("when is my loan payment due", PRODUCT_INFO),
    ("how much did I spend this month", SPENDING_SUMMARY),
    ("give me my spending summary", SPENDING_SUMMARY),
    ("what is my net cash flow", SPENDING_SUMMARY),
    ("kitna kharcha hua", SPENDING_SUMMARY),
    ("what is the education loan interest rate", PRODUCT_INFO),
    ("what loans do you offer", PRODUCT_INFO),
    ("fd interest rates", PRODUCT_INFO),
    ("which accounts can I open", PRODUCT_INFO),
    ("processing fee for home loan", PRODUCT_INFO),
    ("am I eligible for a personal loan", LOAN_ELIGIBILITY),
    ("can I get a loan of 5 lakh", LOAN_ELIGIBILITY),
    ("what is the EMI for 5 lakh over 3 years", LOAN_ELIGIBILITY),
    ("how much loan can I get", LOAN_ELIGIBILITY),
    ("show my profile", PROFILE),
    ("who am I", PROFILE),
    ("what is my consent status", PRIVACY),
    ("how is my data used", PRIVACY),
    ("can I opt out of ai processing", PRIVACY),
    ("help", HELP),
    ("what can you do", HELP),
    ("hi", GREETING),
    ("Hey show my profile", PROFILE),
    ("hello what is my balance", BALANCE),
    ("good morning", GREETING),
    ("tell me a joke", OUT_OF_SCOPE),
    ("which stock should I buy", OUT_OF_SCOPE),
    ("blah blah xyz", UNKNOWN),
    ("", UNKNOWN),
])
def test_rule_intents(text, intent):
    assert classify_intent(text) == intent


def test_greeting_only_when_short():
    # long message that starts with "hi" must still reach the real intent
    assert classify_intent("hi can you show my balance") == BALANCE


class TestEntities:
    def test_full_eligibility_sentence(self):
        e = extract_entities("I earn 60k a month, can I get a 5 lakh personal loan for 3 years?", TODAY)
        assert e == {"monthly_income": 60000, "loan_amount": 500000, "tenure_months": 36,
                     "product_id": "personal_loan"}

    def test_indian_grouping(self):
        assert extract_entities("emi for 5,00,000 over 36 months", TODAY)["loan_amount"] == 500000

    def test_crore_and_decimals(self):
        e = extract_entities("home loan of 1.5 crore for 20 years", TODAY)
        assert e["loan_amount"] == 15_000_000 and e["tenure_months"] == 240

    def test_rupee_symbol(self):
        assert extract_entities("loan of ₹7.5 lakh", TODAY)["loan_amount"] == 750000

    def test_income_and_existing_emi(self):
        e = extract_entities("my salary is 85000 and existing emi 10000, home loan 50 lakh 20 years", TODAY)
        assert e["monthly_income"] == 85000 and e["existing_emi"] == 10000 and e["loan_amount"] == 5_000_000

    def test_word_numbers(self):
        assert extract_entities("show my last five transactions", TODAY)["limit"] == 5
        assert extract_entities("loan for five years", TODAY)["tenure_months"] == 60

    def test_limit_vs_period(self):
        e = extract_entities("last 3 months of transactions", TODAY)
        assert e["period_days"] == 90 and "limit" not in e
        e = extract_entities("last 3 transactions", TODAY)
        assert e["limit"] == 3 and "period_days" not in e

    @pytest.mark.parametrize("text,days", [
        ("last week", 7), ("past month", 30), ("last 7 days", 7), ("last year", 365),
        ("this month", 12), ("today", 1), ("last quarter", 90),
    ])
    def test_periods(self, text, days):
        assert extract_entities(f"spending {text}", TODAY)["period_days"] == days

    def test_dates_and_years_are_not_money(self):
        assert "loan_amount" not in extract_entities("transactions on 2026-10-01", TODAY)
        assert "loan_amount" not in extract_entities("loan rates in 2026", TODAY)

    def test_product_and_category(self):
        assert extract_entities("education loan rate", TODAY)["product_id"] == "education_loan"
        assert "category" not in extract_entities("education loan rate", TODAY)
        assert extract_entities("grocery spending", TODAY)["category"] == "Groceries"
        assert extract_entities("fd rates", TODAY)["product_id"] == "fixed_deposit"

    def test_txn_type(self):
        assert extract_entities("show my credits", TODAY)["txn_type"] == "Credit"
        assert extract_entities("show my debits", TODAY)["txn_type"] == "Debit"
        assert "txn_type" not in extract_entities("show credits and debits", TODAY)

    def test_keep_relevant_drops_noise(self):
        # "salary" must not become a Salary category for an eligibility question
        parsed = rule_based_parse("my salary is 50000, can I get a personal loan", TODAY)
        assert parsed["intent"] == LOAN_ELIGIBILITY
        assert "category" not in parsed["entities"]
        assert keep_relevant(BALANCE, {"limit": 5}) == {}


class TestLlmEntityValidation:
    def test_valid_values_kept(self):
        out = sanitize_llm_entities({"product_id": "home_loan", "category": "groceries",
                                     "txn_type": "debit", "period_days": 30, "limit": 5, "focus": "rates"})
        assert out["product_id"] == "home_loan" and out["category"] == "Groceries"
        assert out["txn_type"] == "Debit" and out["period_days"] == 30 and out["limit"] == 5

    def test_invalid_values_dropped(self):
        out = sanitize_llm_entities({"product_id": "../../etc/passwd", "category": "'; DROP TABLE x;--",
                                     "txn_type": "Transfer", "period_days": 99999, "limit": 500,
                                     "focus": "evil", "product_kind": "yacht"})
        assert out == {}

    def test_money_is_never_taken_from_llm(self):
        out = sanitize_llm_entities({"loan_amount": 99999999, "monthly_income": 1, "tenure_months": 5,
                                     "customer_id": "CUST0002"})
        assert out == {}

    def test_non_dict(self):
        assert sanitize_llm_entities("hello") == {}
        assert sanitize_llm_entities(None) == {}

    def test_bool_not_a_number(self):
        assert sanitize_llm_entities({"limit": True}) == {}
