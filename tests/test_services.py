"""Business logic: EMI maths, eligibility, data consistency, masking, consent."""

import sqlite3
from datetime import timedelta

import pytest

import database as db
import products
import services
from utils import fmt_inr, fmt_months


class TestFormatting:
    @pytest.mark.parametrize("value,expected", [
        (0, "₹0.00"), (999.5, "₹999.50"), (1000, "₹1,000.00"), (123456.789, "₹1,23,456.79"),
        (1234567.5, "₹12,34,567.50"), (-2500, "-₹2,500.00"), (10_000_000, "₹1,00,00,000.00"),
    ])
    def test_inr_indian_grouping(self, value, expected):
        assert fmt_inr(value) == expected

    def test_inr_no_decimals(self):
        assert fmt_inr(500000, 0) == "₹5,00,000"

    def test_months(self):
        assert fmt_months(36) == "3 years" and fmt_months(18) == "1 year 6 months" and fmt_months(6) == "6 months"


class TestEmiMaths:
    def test_known_value(self):
        # 5,00,000 @ 10.5% for 36 months -> about 16,251
        assert services.calc_emi(500000, 10.5, 36) == pytest.approx(16251.22, abs=0.01)

    def test_zero_rate(self):
        assert services.calc_emi(120000, 0, 12) == 10000

    def test_roundtrip(self):
        emi = services.calc_emi(750000, 9.0, 60)
        assert services.max_principal_for_emi(emi, 9.0, 60) == pytest.approx(750000, rel=1e-6)

    def test_invalid_months(self):
        with pytest.raises(ValueError):
            services.calc_emi(1000, 10, 0)

    def test_no_room(self):
        assert services.max_principal_for_emi(0, 10, 12) == 0.0


class TestEligibility:
    def test_needs_loan_type(self):
        r = services.check_loan_eligibility("CUST0001")
        assert r["status"] == "needs_info" and r["missing"] == ["loan_type"]

    def test_deposit_is_not_a_loan(self):
        r = services.check_loan_eligibility("CUST0001", product_id="fixed_deposit")
        assert r["status"] == "needs_info"

    def test_likely_eligible(self):
        r = services.check_loan_eligibility("CUST0001", "personal_loan", 200000, 36,
                                            monthly_income=100000, existing_emi=0)
        assert r["decision"] == "likely_eligible" and r["failed_checks"] == []
        assert r["income_source"] == "stated in this chat"

    def test_affordability_failure(self):
        r = services.check_loan_eligibility("CUST0001", "personal_loan", 2000000, 24,
                                            monthly_income=30000, existing_emi=0)
        assert r["decision"] == "may_not_be_eligible" and "Affordability" in r["failed_checks"]

    def test_income_below_minimum_is_hard_fail(self):
        r = services.check_loan_eligibility("CUST0001", "personal_loan", 100000, 24,
                                            monthly_income=10000, existing_emi=0)
        assert r["decision"] == "not_eligible" and "Monthly income" in r["failed_checks"]

    def test_amount_outside_product_range(self):
        r = services.check_loan_eligibility("CUST0001", "personal_loan", 10_000_000, 36,
                                            monthly_income=1_000_000, existing_emi=0)
        assert r["decision"] == "not_eligible" and "Loan amount" in r["failed_checks"]

    def test_tenure_outside_range(self):
        r = services.check_loan_eligibility("CUST0001", "personal_loan", 100000, 120,
                                            monthly_income=100000, existing_emi=0)
        assert r["decision"] == "not_eligible" and "Tenure" in r["failed_checks"]

    def test_estimate_when_no_amount(self):
        r = services.check_loan_eligibility("CUST0001", "home_loan", None, None,
                                            monthly_income=100000, existing_emi=10000)
        assert r["decision"] == "estimate_only" and r["tenure_assumed"] is True
        assert r["max_eligible_amount"] > 0
        # 50% FOIR: 50,000 - 10,000 = 40,000 EMI room
        assert r["max_affordable_emi"] == 40000

    def test_max_amount_respects_emi_room(self):
        r = services.check_loan_eligibility("CUST0001", "personal_loan", None, 36,
                                            monthly_income=60000, existing_emi=0)
        emi_for_max = services.calc_emi(r["max_eligible_amount"], 18.0, 36)
        assert emi_for_max <= r["max_affordable_emi"] + 0.01

    def test_uses_profile_income_when_not_stated(self):
        r = services.check_loan_eligibility("CUST0001", "personal_loan", 100000, 24)
        customer = db.get_customer("CUST0001")
        assert r["income_used"] == customer["declared_monthly_income"]
        assert r["income_source"] == "declared in your profile"

    def test_never_says_approved(self):
        r = services.check_loan_eligibility("CUST0001", "personal_loan", 100000, 24, monthly_income=100000)
        assert "approv" not in str(r["decision"]).lower()
        assert "not a loan approval" in r["disclaimer"]


class TestProducts:
    def test_catalogue_valid(self):
        data = services.get_products()
        assert data["count"] == 9 and "illustrative" in data["disclaimer"].lower()

    def test_filter_kind(self):
        assert {p["category"] for p in services.get_products(kind="loan")["products"]} == {"loan"}

    def test_single(self):
        p = services.get_products(product_id="home_loan")["products"][0]
        assert p["rate_display"].startswith("8.35%") and "tenure_display" in p

    def test_unknown(self):
        assert services.get_products(product_id="nope")["count"] == 0

    def test_all_have_rates(self):
        for p in products.all_products():
            assert p["rate_min"] <= p["rate_max"]


class TestSyntheticData:
    def test_counts(self):
        with db.get_connection() as c:
            assert c.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 50
            assert c.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] > 2000
            assert c.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] >= 50

    def test_ledger_is_consistent(self):
        """balance == opening balance + credits - debits, for every account."""
        with db.get_connection() as c:
            rows = c.execute("""
                SELECT a.account_id, a.opening_balance, a.balance,
                  COALESCE(SUM(CASE WHEN t.transaction_type='Credit' THEN t.amount ELSE -t.amount END),0) AS net
                FROM accounts a LEFT JOIN transactions t ON t.account_id = a.account_id
                GROUP BY a.account_id""").fetchall()
        for r in rows:
            assert r["balance"] == pytest.approx(r["opening_balance"] + r["net"], abs=0.02)
            assert r["balance"] > 0

    def test_dates_are_recent_and_not_future(self):
        with db.get_connection() as c:
            newest, oldest = c.execute("SELECT MAX(transaction_date), MIN(transaction_date) FROM transactions").fetchone()
        assert newest <= db.today().isoformat()
        assert oldest >= (db.today() - timedelta(days=366)).isoformat()

    def test_every_transaction_has_known_category(self):
        from entities import ALL_CATEGORIES
        with db.get_connection() as c:
            cats = {r[0] for r in c.execute("SELECT DISTINCT category FROM transactions")}
        assert cats <= set(ALL_CATEGORIES)

    def test_recent_window_has_data(self):
        for cid in ("CUST0001", "CUST0002", "CUST0003"):
            assert services.get_transactions(cid, limit=50, period_days=30)["count"] > 0

    def test_initialize_is_idempotent(self):
        before = db.count_customers()
        db.initialize_database()
        assert db.count_customers() == before


class TestSpending:
    def test_net_flow_is_credit_minus_debit(self):
        s = services.get_spending_summary("CUST0001", 365)
        assert s["net_flow"] == pytest.approx(s["total_credit"] - s["total_debit"], abs=0.01)

    def test_category_breakdown_sums_to_total_debit(self):
        s = db.get_analytics_summary("CUST0001", 365)
        assert sum(c["total"] for c in s["by_category"]) == pytest.approx(s["total_debit"], abs=0.05)

    def test_category_filter(self):
        s = services.get_spending_summary("CUST0001", 365, category="Groceries")
        assert s["category"] == "Groceries" and s["total_credit"] == 0

    def test_shares_add_up_roughly(self):
        s = services.get_spending_summary("CUST0001", 365)
        assert all(0 <= c["share_pct"] <= 100 for c in s["top_categories"])


class TestMaskingInServices:
    def test_balance_has_no_raw_account_id(self):
        b = services.get_balance("CUST0001")
        assert "ACC0" not in str(b)
        assert b["accounts"][0]["account_masked"].endswith("0001")

    def test_transactions_masked(self):
        t = services.get_transactions("CUST0001", limit=3)
        assert "ACC0" not in str(t)

    def test_profile_masks_contact_details(self):
        p = services.get_profile("CUST0001")
        assert "@demobank.in" in p["email_masked"] and "aarav.sharma" not in str(p)
        customer = db.get_customer("CUST0001")
        assert customer["phone"] not in str(p)

    def test_unknown_customer(self):
        with pytest.raises(services.ServiceError):
            services.get_balance("CUST9999")


class TestConsent:
    def test_defaults_for_demo_users(self):
        assert db.get_consent("CUST0005")["ai_processing"] is False
        assert db.get_consent("CUST0001") == {**db.get_consent("CUST0001"), "ai_processing": True,
                                              "query_logging": True, "marketing": True}

    def test_update_and_audit(self, restore_consent):
        services.set_consent("CUST0002", "demo_priya", marketing=True)
        assert db.get_consent("CUST0002")["marketing"] is True
        with db.get_connection() as c:
            row = c.execute("SELECT detail FROM audit_logs WHERE event_type='CONSENT_CHANGED' "
                            "ORDER BY log_id DESC LIMIT 1").fetchone()
        assert "marketing=on" in row["detail"]

    def test_partial_update_keeps_others(self, restore_consent):
        before = db.get_consent("CUST0001")
        services.set_consent("CUST0001", "demo_aarav", query_logging=False)
        after = db.get_consent("CUST0001")
        assert after["query_logging"] is False
        assert after["ai_processing"] == before["ai_processing"] and after["marketing"] == before["marketing"]

    def test_unknown_customer(self):
        with pytest.raises(services.ServiceError):
            services.set_consent("CUST9999", "x", ai_processing=True)


class TestQueryLogging:
    def test_logged_only_with_consent(self, restore_consent):
        db.update_consent("CUST0002", query_logging=False)
        with db.get_connection() as c:
            n0 = c.execute("SELECT COUNT(*) FROM query_logs WHERE customer_id='CUST0002'").fetchone()[0]
        services.record_chat_event("CUST0002", "demo_priya", "show balance", "BALANCE", "get_balance",
                                   "rules", False, None, [])
        with db.get_connection() as c:
            assert c.execute("SELECT COUNT(*) FROM query_logs WHERE customer_id='CUST0002'").fetchone()[0] == n0
        db.update_consent("CUST0002", query_logging=True)
        services.record_chat_event("CUST0002", "demo_priya", "show balance", "BALANCE", "get_balance",
                                   "rules", False, None, [])
        with db.get_connection() as c:
            assert c.execute("SELECT COUNT(*) FROM query_logs WHERE customer_id='CUST0002'").fetchone()[0] == n0 + 1

    def test_audit_never_contains_message_text(self, restore_consent):
        services.record_chat_event("CUST0001", "demo_aarav", "secret question text", "BALANCE", "get_balance",
                                   "rules", False, None, [])
        with db.get_connection() as c:
            details = " ".join(r[0] for r in c.execute("SELECT detail FROM audit_logs"))
        assert "secret question text" not in details


class TestSchemaMigration:
    def test_old_database_is_backed_up_and_rebuilt(self, tmp_path, monkeypatch):
        old = tmp_path / "old.db"
        conn = sqlite3.connect(old)
        conn.execute("CREATE TABLE customers (customer_id TEXT)")  # old/incompatible schema, user_version=0
        conn.commit()
        conn.close()
        monkeypatch.setattr(db, "DB_PATH", str(old))
        db.initialize_database()
        assert db.count_customers() == 50
        assert any(p.name.startswith("old.db.old-") for p in tmp_path.iterdir())
