"""
BANKMATE AI — app.py
Streamlit frontend for the AI-Powered Banking Customer Query Assistant.
Talks only to the FastAPI backend (the Groq key never reaches the browser).
All data displayed is synthetic and for demonstration purposes only.
"""

from __future__ import annotations

import html

import pandas as pd
import requests
import streamlit as st

import config
import responses
from utils import fmt_inr

API_BASE = config.get_api_base()
APP_NAME = "BANKMATE AI"
TAGLINE = "Ask. Understand. Bank Smarter."

st.set_page_config(page_title="BANKMATE AI", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
  html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
  .stApp { background-color: #0B1120; color: #E2E8F0; }
  .bm-header { padding: 1rem 0 0.5rem 0; border-bottom: 1px solid #1E3A5F; margin-bottom: 1rem; }
  .bm-brand { font-size: 1.8rem; font-weight: 700; color: #38BDF8; letter-spacing: 1px; }
  .bm-tagline { font-size: 0.86rem; color: #94A3B8; margin-top: 0.1rem; }
  .bm-demo-badge { display: inline-block; background: #1B2940; border: 1px solid #F59E0B; color: #F59E0B;
    border-radius: 4px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; margin-left: 0.7rem; }
  .bm-card { background: #151F32; border: 1px solid #1E3A5F; border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
  .bm-card-title { font-size: 0.72rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.8px; }
  .bm-card-value { font-size: 1.55rem; font-weight: 700; color: #38BDF8; }
  .bm-card-sub { font-size: 0.78rem; color: #94A3B8; margin-top: 0.2rem; }
  .bm-badge { display: inline-block; border-radius: 4px; padding: 2px 8px; font-size: 0.72rem; font-weight: 600; margin-right: 6px; }
  .bm-badge-llm { background: #14532D; color: #22C55E; }
  .bm-badge-rules { background: #1B2940; color: #F59E0B; border: 1px solid #F59E0B; }
  .bm-badge-guard { background: #450A0A; color: #F87171; }
  .bm-section-header { font-size: 0.78rem; font-weight: 700; color: #64748B; text-transform: uppercase;
    letter-spacing: 1px; margin: 1rem 0 0.5rem 0; border-bottom: 1px solid #1E3A5F; padding-bottom: 0.3rem; }
  [data-testid="stSidebar"] { background-color: #0F172A; border-right: 1px solid #1E3A5F; }
  .bm-disclaimer { background: #1B2940; border-left: 3px solid #F59E0B; padding: 0.6rem 0.9rem; font-size: 0.72rem;
    color: #94A3B8; border-radius: 0 4px 4px 0; margin-top: 0.5rem; }
  .bm-ok { font-size: 0.75rem; color: #22C55E; font-weight: 600; }
  .bm-warn { font-size: 0.75rem; color: #F59E0B; font-weight: 600; }
  .bm-err { font-size: 0.75rem; color: #EF4444; font-weight: 600; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

esc = html.escape  # every dynamic value that goes into unsafe HTML must pass through this

FALLBACK_TEXT = {
    "consent_off": "AI processing consent is OFF",
    "no_api_key": "no Groq API key set",
    "invalid_api_key": "Groq rejected the API key",
    "rate_limited": "Groq rate limit reached",
    "timeout": "Groq timed out",
    "connection": "could not reach Groq",
    "model_unavailable": "no Groq model available",
    "temporarily_paused": "Groq paused after errors",
    "bad_response": "Groq reply could not be used",
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _request(method: str, path: str, token: str | None = None, timeout: int = 10, **kwargs):
    headers = {"X-Session-Token": token} if token else {}
    try:
        resp = requests.request(method, f"{API_BASE}{path}", headers=headers, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        return None, "Backend not reachable. Start it with: python -m uvicorn api:app --reload", 0
    except requests.exceptions.Timeout:
        return None, "The request timed out. Please try again.", 0
    except Exception as exc:  # noqa: BLE001
        return None, str(exc), 0
    if resp.status_code == 200:
        return resp.json(), None, 200
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:  # noqa: BLE001
        detail = resp.text
    return None, str(detail), resp.status_code


def api_get(path, token=None, params=None):
    return _request("GET", path, token, params=params)[:2]


def api_post(path, body, token=None, timeout=10):
    return _request("POST", path, token, timeout=timeout, json=body)


def api_put(path, body, token=None):
    return _request("PUT", path, token, json=body)[:2]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_session() -> None:
    defaults = {"authenticated": False, "session_token": None, "customer_id": None,
                "username": None, "messages": []}
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


_init_session()


def do_logout(message: str | None = None) -> None:
    token = st.session_state.get("session_token")
    if token:
        api_post("/auth/logout", {}, token=token)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _init_session()
    if message:
        st.session_state["flash"] = message
    st.rerun()


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


def show_login_page() -> None:
    st.markdown(
        f'<div class="bm-header"><span class="bm-brand">🏦 {APP_NAME}</span>'
        f'<span class="bm-demo-badge">DEMO PROTOTYPE</span>'
        f'<div class="bm-tagline">{TAGLINE}</div></div>', unsafe_allow_html=True)

    health, err = api_get("/health")
    if err:
        st.error(f"⚠️ Backend unavailable: {err}")
        st.stop()
    if flash := st.session_state.pop("flash", None):
        st.warning(flash)

    col, _, _ = st.columns([1.3, 0.2, 1])
    with col:
        st.subheader("Sign In")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="e.g. demo_aarav")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
        if submitted:
            if not username.strip() or not password:
                st.error("Please enter both username and password.")
            else:
                data, err, code = _request("POST", "/auth/login", json={"username": username.strip(), "password": password})
                if data:
                    st.session_state.update(authenticated=True, session_token=data["session_token"],
                                            customer_id=data["customer_id"], username=data["username"], messages=[])
                    st.rerun()
                elif code == 429:
                    st.error(f"🔒 {err}")
                elif code == 0:
                    st.error(err)
                else:
                    st.error("Invalid username or password. Please try again.")

        st.markdown(
            '<div class="bm-disclaimer"><strong>Demo logins (fictional)</strong><br>'
            '<code>demo_aarav</code> / <code>Demo@1234</code> — all consents ON<br>'
            '<code>demo_priya</code> / <code>Demo@5678</code> — marketing OFF<br>'
            '<code>demo_rohan</code> / <code>Demo@9012</code> — query logging OFF<br>'
            '<code>demo_sneha</code> / <code>Demo@3456</code> — logging &amp; marketing OFF<br>'
            '<code>demo_kiran</code> / <code>Demo@7890</code> — AI processing OFF (no data to the LLM)<br><br>'
            '⚠️ Never use real banking credentials here.</div>', unsafe_allow_html=True)


if not st.session_state.authenticated:
    show_login_page()
    st.stop()

# ===========================================================================
# AUTHENTICATED AREA
# ===========================================================================

TOKEN = st.session_state.session_token
CID = st.session_state.customer_id

profile, profile_err = api_get(f"/customers/{CID}", TOKEN)
if profile_err and "expired" in profile_err.lower():
    do_logout("Your session expired. Please sign in again.")
health, _ = api_get("/health")
llm_status = (health or {}).get("llm", {})

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div class="bm-brand">🏦 {APP_NAME}</div><div class="bm-tagline">{TAGLINE}</div>',
                unsafe_allow_html=True)
    st.markdown("---")
    if health:
        st.markdown('<span class="bm-ok">● API connected</span>', unsafe_allow_html=True)
        if llm_status.get("available"):
            st.markdown(f'<span class="bm-ok">● Groq LLM ready</span><br>'
                        f'<span style="font-size:0.7rem;color:#64748B;">{esc(str(llm_status.get("model")))}</span>',
                        unsafe_allow_html=True)
        else:
            reason = FALLBACK_TEXT.get(llm_status.get("reason") or "", "not available")
            st.markdown(f'<span class="bm-warn">● Rules mode ({esc(reason)})</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="bm-err">● API offline</span>', unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:0.82rem;color:#94A3B8;margin-top:0.6rem;">👤 '
                f'<strong style="color:#38BDF8;">{esc(st.session_state.username)}</strong><br>'
                f'<span style="font-size:0.72rem;color:#64748B;">Customer ID: {esc(CID)}</span></div>',
                unsafe_allow_html=True)
    if st.button("🚪 Sign Out", use_container_width=True):
        do_logout()

    st.markdown('<div class="bm-section-header">Try asking</div>', unsafe_allow_html=True)
    for q in ["What is my account balance?", "Show my last 5 transactions",
              "How much did I spend in the last 30 days?", "What is the home loan interest rate?",
              "Can I get a 5 lakh personal loan for 3 years?", "What is my consent status?"]:
        if st.button(q, key=f"sugg_{q}", use_container_width=True):
            st.session_state["_inject_question"] = q

    st.markdown('<div class="bm-disclaimer">⚠️ All data is <strong>entirely synthetic</strong>. '
                'No real banking data or credentials are used.</div>', unsafe_allow_html=True)
    st.markdown('<br><span style="font-size:0.7rem;color:#475569;">BANKMATE AI v3.0 · LangGraph + Groq</span>',
                unsafe_allow_html=True)

st.markdown(f'<div class="bm-header"><span class="bm-brand">🏦 {APP_NAME}</span>'
            f'<span class="bm-demo-badge">DEMO</span>'
            f'<div class="bm-tagline">{TAGLINE} — AI-Powered Banking Customer Query Assistant</div></div>',
            unsafe_allow_html=True)

tab_chat, tab_dash, tab_products, tab_privacy = st.tabs(
    ["💬 Assistant", "📊 Dashboard", "🏦 Products & Eligibility", "🔒 Privacy & Consent"])


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def _badge(meta: dict) -> str:
    if meta.get("blocked"):
        return '<span class="bm-badge bm-badge-guard">🛡️ Safety guard</span>'
    if meta.get("llm_used"):
        return f'<span class="bm-badge bm-badge-llm">🤖 Groq · {esc(str(meta.get("model") or ""))}</span>'
    reason = FALLBACK_TEXT.get(meta.get("fallback_reason") or "", "")
    return f'<span class="bm-badge bm-badge-rules">⚙️ Rules{(" · " + esc(reason)) if reason else ""}</span>'


with tab_chat:
    if not st.session_state.messages:
        name = (profile or {}).get("name", "there")
        with st.chat_message("assistant"):
            st.markdown(f"Hello, **{name}**! I'm your demo banking assistant. Ask about your balance, "
                        "transactions, spending, interest rates or loan eligibility.")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])  # markdown only - HTML is NOT rendered
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                st.markdown(_badge(meta), unsafe_allow_html=True)
                with st.expander("How I answered"):
                    st.json({k: meta.get(k) for k in
                             ("intent", "tool", "classifier", "entities", "pii_redacted",
                              "guardrail_flags", "blocked", "followup") if meta.get(k) not in (None, [], {}, False)}
                            or {"intent": meta.get("intent")})

    col_in, col_reset = st.columns([6, 1])
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True, help="Clear this conversation"):
            _request("DELETE", "/chat", TOKEN)
            st.session_state.messages = []
            st.rerun()
    injected = st.session_state.pop("_inject_question", None)
    typed = st.chat_input("Ask a banking question…")
    active = injected or typed

    if active:
        st.session_state.messages.append({"role": "user", "content": active})
        with st.spinner("Thinking…"):
            data, err, code = api_post("/chat", {"message": active}, TOKEN, timeout=90)
        if code == 401:
            do_logout("Your session expired. Please sign in again.")
        if data:
            st.session_state.messages.append({"role": "assistant", "content": data["reply"], "meta": data})
        else:
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {err}"})
        st.rerun()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

with tab_dash:
    accounts, acct_err = api_get(f"/accounts/{CID}", TOKEN)
    summary, _ = api_get(f"/analytics/summary/{CID}", TOKEN, {"days": 365})

    c1, c2, c3, c4 = st.columns(4)
    if accounts:
        with c1:
            st.markdown(f'<div class="bm-card"><div class="bm-card-title">Total balance</div>'
                        f'<div class="bm-card-value">{esc(accounts["total_display"])}</div>'
                        f'<div class="bm-card-sub">{accounts["account_count"]} account(s) · INR</div></div>',
                        unsafe_allow_html=True)
        with c2:
            lines = "<br>".join(f'{esc(a["account_type"])} {esc(a["account_masked"])} · {esc(a["balance_display"])}'
                                for a in accounts["accounts"])
            st.markdown(f'<div class="bm-card"><div class="bm-card-title">Accounts</div>'
                        f'<div class="bm-card-sub" style="font-size:0.85rem;">{lines}</div></div>',
                        unsafe_allow_html=True)
    else:
        st.error(acct_err or "Balance unavailable")
    if summary:
        with c3:
            st.markdown(f'<div class="bm-card"><div class="bm-card-title">Credits (last 12 months)</div>'
                        f'<div class="bm-card-value" style="color:#22C55E">{esc(summary["total_credit_display"])}</div>'
                        f'<div class="bm-card-sub">{summary["credit_count"]} credits</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="bm-card"><div class="bm-card-title">Debits (last 12 months)</div>'
                        f'<div class="bm-card-value" style="color:#EF4444">{esc(summary["total_debit_display"])}</div>'
                        f'<div class="bm-card-sub">Net: {esc(summary["net_flow_display"])}</div></div>', unsafe_allow_html=True)

    if profile:
        st.markdown(f'<div style="font-size:0.8rem;color:#64748B;">👤 <strong style="color:#94A3B8">{esc(profile["name"])}</strong>'
                    f' · {esc(profile["email_masked"])} · {esc(profile["phone_masked"])} '
                    f'<span style="color:#F59E0B;font-size:0.7rem;">[DEMO DATA · masked]</span></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="bm-section-header">Transactions</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    cat = f1.selectbox("Category", ["All", "Groceries", "Dining", "Utilities", "Shopping", "Transport",
                                    "Health", "Entertainment", "Education", "Salary", "Other Income"])
    ttype = f2.selectbox("Type", ["All", "Credit", "Debit"])
    days = f3.selectbox("Period", [("Last 30 days", 30), ("Last 90 days", 90), ("Last 12 months", 365)],
                        format_func=lambda x: x[0])[1]
    limit = f4.slider("Rows", 5, 50, 10)
    params = {"limit": limit, "days": days}
    if cat != "All":
        params["category"] = cat
    if ttype != "All":
        params["type"] = ttype
    txns, txn_err = api_get(f"/transactions/{CID}", TOKEN, params)
    if txns and txns["transactions"]:
        df = pd.DataFrame(txns["transactions"])[["date", "description", "category", "amount", "type", "status"]]
        df["amount"] = df["amount"].map(fmt_inr)
        df.columns = ["Date", "Description", "Category", "Amount", "Type", "Status"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    elif txn_err:
        st.error(txn_err)
    else:
        st.info("No transactions match these filters.")

    if summary and summary["top_categories"]:
        st.markdown('<div class="bm-section-header">Where your money went (last 12 months)</div>', unsafe_allow_html=True)
        chart_df = pd.DataFrame(summary["top_categories"]).set_index("category")[["total"]]
        st.bar_chart(chart_df)


# ---------------------------------------------------------------------------
# Products and eligibility
# ---------------------------------------------------------------------------

with tab_products:
    catalogue_data, cat_err = api_get("/products")
    if cat_err:
        st.error(cat_err)
    else:
        for label, kind in (("Accounts", "account"), ("Deposits", "deposit"), ("Loans", "loan")):
            items = [p for p in catalogue_data["products"] if p["category"] == kind]
            st.markdown(f'<div class="bm-section-header">{label}</div>', unsafe_allow_html=True)
            cols = st.columns(max(len(items), 1))
            for col, p in zip(cols, items):
                extra = f'<br>Tenure: {esc(p["tenure_display"])}' if p.get("tenure_display") else ""
                col.markdown(
                    f'<div class="bm-card"><div class="bm-card-title">{esc(p["name"])}</div>'
                    f'<div style="color:#38BDF8;font-weight:600;font-size:0.9rem;">{esc(p["rate_display"])}</div>'
                    f'<div class="bm-card-sub">{esc(p["description"])}{extra}</div></div>', unsafe_allow_html=True)
        st.caption(f"⚠️ {catalogue_data['disclaimer']}")

        st.markdown('<div class="bm-section-header">Loan eligibility & EMI calculator</div>', unsafe_allow_html=True)
        loans = [p for p in catalogue_data["products"] if p["category"] == "loan"]
        with st.form("elig_form"):
            e1, e2 = st.columns(2)
            product = e1.selectbox("Loan type", loans, format_func=lambda p: p["name"])
            amount = e2.number_input("Loan amount (₹)", min_value=10000, max_value=100_000_000, value=500000, step=10000)
            e3, e4, e5 = st.columns(3)
            years = e3.number_input("Tenure (years)", min_value=1, max_value=30, value=3)
            income = e4.number_input("Monthly income (₹) — 0 = use my profile", min_value=0, value=0, step=5000)
            emi = e5.number_input("Existing EMIs (₹/month) — blank = profile", min_value=0, value=0, step=500)
            go = st.form_submit_button("Check eligibility", use_container_width=True)
        if go:
            body = {"product_id": product["product_id"], "loan_amount": amount, "tenure_months": int(years * 12)}
            if income:
                body["monthly_income"] = income
            if emi:
                body["existing_emi"] = emi
            result, err, _ = api_post("/loans/eligibility", body, TOKEN)
            if result:
                st.markdown(responses.render_eligibility(result))
                st.caption(f"⚠️ {result['disclaimer']}")
            else:
                st.error(err)


# ---------------------------------------------------------------------------
# Privacy and consent
# ---------------------------------------------------------------------------

with tab_privacy:
    st.markdown("### Your data, your choice")
    st.write("These switches control how your data is used. Changes apply immediately to your next message.")
    consent_data, c_err = api_get(f"/consent/{CID}", TOKEN)
    if c_err:
        st.error(c_err)
    else:
        c = consent_data["consent"]
        with st.form("consent_form"):
            ai = st.checkbox("AI processing — let the assistant use a language model (Groq)", value=c["ai_processing"],
                             help="ON: your question (card/phone/e-mail/PAN/OTP removed) plus the minimum facts needed "
                                  "are sent to Groq. OFF: nothing leaves this server; built-in rules answer.")
            logging_on = st.checkbox("Query logging — store a privacy-masked copy of my questions",
                                     value=c["query_logging"])
            marketing = st.checkbox("Marketing suggestions — allow product tips in answers", value=c["marketing"])
            if st.form_submit_button("Save my choices", use_container_width=True):
                saved, err = api_put(f"/consent/{CID}", {"ai_processing": ai, "query_logging": logging_on,
                                                          "marketing": marketing}, TOKEN)
                if saved:
                    st.success("Saved. Your new choices are active.")
                else:
                    st.error(err)
        if c.get("updated_at"):
            st.caption(f"Last updated: {c['updated_at']}")
        st.markdown(
            "**Always protected, whatever you choose**\n"
            "- Account number, e-mail and phone are masked.\n"
            "- Card numbers, Aadhaar, PAN, phone, e-mail, OTP/PIN/password are removed from your message "
            "before it is processed or logged.\n"
            "- Your name, e-mail and phone are never sent to the language model.\n"
            "- You can only ever see your own data.")
