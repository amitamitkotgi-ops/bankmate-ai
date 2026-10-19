"""FastAPI endpoints: authentication, authorization (IDOR), masking, consent, chat."""

import pytest

import config
import database as db
from conftest import DEMO


class TestHealthAndHeaders:
    def test_health(self, client):
        d = client.get("/health").json()
        assert d["status"] == "ok" and "Connected" in d["database"] and "data_notice" in d
        assert set(d["llm"]) >= {"configured", "available", "model", "reason"}

    def test_health_never_exposes_a_key(self, client):
        assert "gsk_" not in client.get("/health").text

    def test_security_headers(self, client):
        r = client.get("/health")
        assert r.headers["cache-control"] == "no-store" and r.headers["x-content-type-options"] == "nosniff"

    def test_chat_graph_mermaid(self, client):
        m = client.get("/chat/graph").json()["mermaid"]
        assert "preprocess" in m and "guardrail" in m and "classify" in m


class TestLogin:
    def test_success_all_demo_users(self, client):
        for username, password, cid in DEMO.values():
            r = client.post("/auth/login", json={"username": username, "password": password})
            assert r.status_code == 200 and r.json()["customer_id"] == cid
            assert len(r.json()["session_token"]) >= 30

    @pytest.mark.parametrize("body", [
        {"username": "demo_aarav", "password": "wrong"},
        {"username": "nobody", "password": "Demo@1234"},
        {"username": "demo_aarav", "password": ""},
    ])
    def test_failures_are_401_or_400(self, client, body):
        assert client.post("/auth/login", json=body).status_code in (400, 401)

    def test_same_message_for_unknown_and_wrong_password(self, client):
        a = client.post("/auth/login", json={"username": "demo_aarav", "password": "x"})
        b = client.post("/auth/login", json={"username": "ghost_user", "password": "x"})
        assert a.status_code == b.status_code == 401 and a.json() == b.json()

    def test_password_hash_never_returned(self, client):
        r = client.post("/auth/login", json={"username": "demo_aarav", "password": "Demo@1234"})
        assert "pbkdf2" not in r.text and "password_hash" not in r.text

    def test_hash_is_salted_and_not_plain(self):
        h1, h2 = db.hash_password("Demo@1234"), db.hash_password("Demo@1234")
        assert h1 != h2 and "Demo@1234" not in h1 and h1.startswith("pbkdf2_sha256$")
        assert db.verify_password("Demo@1234", h1) and not db.verify_password("nope", h1)

    def test_lockout_same_for_real_and_fake_users(self, client):
        for name in ("lockout_ghost", "demo_sneha"):
            codes = [client.post("/auth/login", json={"username": name, "password": "bad"}).status_code
                     for _ in range(6)]
            assert codes[:5] == [401] * 5 and codes[5] == 429, (name, codes)
        # even the right password is refused while locked
        assert client.post("/auth/login", json={"username": "demo_sneha", "password": "Demo@3456"}).status_code == 429
        db.reset_failed_attempts("demo_sneha")
        db.reset_failed_attempts("lockout_ghost")
        assert client.post("/auth/login", json={"username": "demo_sneha", "password": "Demo@3456"}).status_code == 200

    def test_oversized_username_rejected(self, client):
        assert client.post("/auth/login", json={"username": "a" * 500, "password": "x"}).status_code == 422

    def test_audit_log_records_login_events(self, client):
        client.post("/auth/login", json={"username": "demo_aarav", "password": "Demo@1234"})
        with db.get_connection() as c:
            types = {r[0] for r in c.execute("SELECT event_type FROM audit_logs")}
        assert "LOGIN_SUCCESS" in types and "LOGIN_FAILED" in types


class TestAuthorization:
    PATHS = ["/customers/{c}", "/accounts/{c}", "/transactions/{c}", "/analytics/summary/{c}", "/consent/{c}"]

    @pytest.mark.parametrize("path", PATHS)
    def test_requires_token(self, client, path):
        assert client.get(path.format(c="CUST0001")).status_code == 401

    @pytest.mark.parametrize("path", PATHS)
    def test_bad_token(self, client, path):
        assert client.get(path.format(c="CUST0001"), headers={"X-Session-Token": "bogus"}).status_code == 401

    @pytest.mark.parametrize("path", PATHS)
    def test_cannot_read_other_customer(self, client, login, path):
        h, _ = login("aarav")
        assert client.get(path.format(c="CUST0002"), headers=h).status_code == 403

    def test_cannot_write_other_customers_consent(self, client, login):
        h, _ = login("aarav")
        r = client.put("/consent/CUST0002", json={"marketing": True}, headers=h)
        assert r.status_code == 403

    def test_nonexistent_customer_is_403_not_404(self, client, login):
        h, _ = login("aarav")
        assert client.get("/accounts/CUST9999", headers=h).status_code == 403  # no existence leak

    def test_customer_list_endpoint_is_gone(self, client, login):
        """The old public /customers/list leaked every customer's name."""
        h, _ = login("aarav")
        assert client.get("/customers/list").status_code == 401   # now parsed as a customer_id, needs a token
        r = client.get("/customers/list", headers=h)
        assert r.status_code == 403 and "Priya" not in r.text

    def test_logout_invalidates_token(self, client, login):
        h, cid = login("aarav")
        assert client.get(f"/accounts/{cid}", headers=h).status_code == 200
        client.post("/auth/logout", headers=h)
        assert client.get(f"/accounts/{cid}", headers=h).status_code == 401
        assert client.post("/chat", json={"message": "hi"}, headers=h).status_code == 401

    def test_sessions_are_isolated(self, client, login):
        h1, c1 = login("aarav")
        h2, c2 = login("priya")
        b1 = client.get(f"/accounts/{c1}", headers=h1).json()["total_balance"]
        b2 = client.get(f"/accounts/{c2}", headers=h2).json()["total_balance"]
        assert b1 != b2
        assert client.get(f"/accounts/{c1}", headers=h2).status_code == 403

    def test_expired_session_rejected(self, client, login):
        import api
        h, cid = login("aarav")
        api._sessions[h["X-Session-Token"]]["expires_at"] = "2000-01-01T00:00:00+00:00"
        assert client.get(f"/accounts/{cid}", headers=h).status_code == 401


class TestMaskingOverTheWire:
    def test_account_masked(self, client, login):
        h, cid = login("aarav")
        text = client.get(f"/accounts/{cid}", headers=h).text
        assert "ACC0" not in text and "*" in text

    def test_transactions_masked_and_have_category(self, client, login):
        h, cid = login("aarav")
        data = client.get(f"/transactions/{cid}?limit=5", headers=h).json()
        assert data["count"] == 5 and "ACC0" not in str(data) and data["transactions"][0]["category"]

    def test_profile_masks_phone_and_email_on_the_server(self, client, login):
        h, cid = login("aarav")
        text = client.get(f"/customers/{cid}", headers=h).text
        customer = db.get_customer(cid)
        assert customer["phone"] not in text and customer["email"] not in text


class TestDataEndpoints:
    def test_transaction_filters(self, client, login):
        h, cid = login("aarav")
        d = client.get(f"/transactions/{cid}?limit=50&days=90&type=Debit&category=Groceries", headers=h).json()
        assert all(t["type"] == "Debit" and t["category"] == "Groceries" for t in d["transactions"])

    @pytest.mark.parametrize("qs", ["limit=0", "limit=51", "limit=abc", "days=0", "days=999", "type=Hack"])
    def test_bad_query_params_422(self, client, login, qs):
        h, cid = login("aarav")
        assert client.get(f"/transactions/{cid}?{qs}", headers=h).status_code == 422

    def test_sql_injection_is_inert(self, client, login):
        h, cid = login("aarav")
        r = client.get(f"/transactions/{cid}", params={"category": "Groceries' OR '1'='1"}, headers=h)
        assert r.status_code == 200 and r.json()["count"] == 0
        assert client.get("/accounts/CUST0001' OR '1'='1", headers=h).status_code == 403

    def test_analytics_math(self, client, login):
        h, cid = login("aarav")
        d = client.get(f"/analytics/summary/{cid}", headers=h).json()
        assert d["net_flow"] == pytest.approx(d["total_credit"] - d["total_debit"], abs=0.01)


class TestProductsPublic:
    def test_list_without_login(self, client):
        d = client.get("/products").json()
        assert d["count"] == 9 and "disclaimer" in d

    def test_filter(self, client):
        assert client.get("/products?category=loan").json()["count"] == 4
        assert client.get("/products?category=hack").status_code == 422

    def test_single(self, client):
        assert client.get("/products/personal_loan").json()["product"]["name"] == "Personal Loan"
        assert client.get("/products/nope").status_code == 404

    def test_eligibility_requires_login(self, client):
        assert client.post("/loans/eligibility", json={"product_id": "personal_loan"}).status_code == 401

    def test_eligibility(self, client, login):
        h, _ = login("aarav")
        r = client.post("/loans/eligibility", headers=h, json={
            "product_id": "personal_loan", "loan_amount": 200000, "tenure_months": 36,
            "monthly_income": 100000, "existing_emi": 0})
        assert r.status_code == 200 and r.json()["decision"] == "likely_eligible"

    @pytest.mark.parametrize("body", [
        {"product_id": "personal_loan", "loan_amount": -5},
        {"product_id": "personal_loan", "tenure_months": 0},
        {"product_id": "personal_loan", "monthly_income": -1},
        {"product_id": "personal_loan", "customer_id": "CUST0002", "loan_amount": "abc"},
    ])
    def test_eligibility_validation(self, client, login, body):
        h, _ = login("aarav")
        assert client.post("/loans/eligibility", headers=h, json=body).status_code == 422

    def test_eligibility_unknown_product(self, client, login):
        h, _ = login("aarav")
        assert client.post("/loans/eligibility", headers=h, json={"product_id": "x"}).status_code == 404

    def test_eligibility_ignores_customer_id_in_body(self, client, login):
        """customer_id is not an input at all — it always comes from the session."""
        h, _ = login("aarav")
        r = client.post("/loans/eligibility", headers=h, json={
            "product_id": "personal_loan", "loan_amount": 100000, "tenure_months": 24, "customer_id": "CUST0002"})
        assert r.status_code == 200
        assert r.json()["income_used"] == db.get_customer("CUST0001")["declared_monthly_income"]


class TestConsentEndpoints:
    def test_read_and_update(self, client, login, restore_consent):
        h, cid = login("priya")
        assert client.get(f"/consent/{cid}", headers=h).json()["consent"]["marketing"] is False
        r = client.put(f"/consent/{cid}", json={"marketing": True}, headers=h)
        assert r.status_code == 200 and r.json()["consent"]["marketing"] is True
        assert client.get(f"/consent/{cid}", headers=h).json()["consent"]["marketing"] is True

    def test_rejects_non_boolean(self, client, login):
        h, cid = login("priya")
        assert client.put(f"/consent/{cid}", json={"marketing": "maybe"}, headers=h).status_code == 422


class TestChatEndpoint:
    def test_requires_token(self, client):
        assert client.post("/chat", json={"message": "hi"}).status_code == 401

    def test_basic_reply_shape(self, client, login):
        h, _ = login("aarav")
        d = client.post("/chat", json={"message": "what is my balance"}, headers=h).json()
        assert d["intent"] == "BALANCE" and d["tool"] == "get_balance" and "₹" in d["reply"]
        assert set(d) >= {"reply", "intent", "mode", "llm_used", "entities", "guardrail_flags"}

    def test_customer_id_in_body_is_ignored(self, client, login):
        h1, c1 = login("aarav")
        _, c2 = login("priya")
        mine = client.get(f"/accounts/{c1}", headers=h1).json()["total_display"]
        d = client.post("/chat", json={"message": "what is my balance", "customer_id": c2}, headers=h1).json()
        assert mine in d["reply"]

    def test_message_too_long_gets_friendly_reply(self, client, login):
        h, _ = login("aarav")
        d = client.post("/chat", json={"message": "a" * 1000}, headers=h).json()
        assert d["blocked"] == "too_long"

    def test_message_over_hard_limit_422(self, client, login):
        h, _ = login("aarav")
        assert client.post("/chat", json={"message": "a" * 5000}, headers=h).status_code == 422

    def test_empty_message(self, client, login):
        h, _ = login("aarav")
        assert client.post("/chat", json={"message": "   "}, headers=h).json()["blocked"] == "empty"

    def test_rate_limit(self, client, login, monkeypatch):
        monkeypatch.setattr(config, "CHAT_RATE_LIMIT_PER_MINUTE", 3)
        h, _ = login("rohan")
        codes = [client.post("/chat", json={"message": "hi"}, headers=h).status_code for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]

    def test_reset_chat(self, client, login):
        h, _ = login("aarav")
        assert client.delete("/chat", headers=h).status_code == 200
