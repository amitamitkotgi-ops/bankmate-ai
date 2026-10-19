"""Headless UI test: the real Streamlit script runs against the real API (via TestClient)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import config
import database as db


@pytest.fixture()
def ui(client, monkeypatch):
    import requests

    def fake_request(method, url, headers=None, timeout=None, **kw):
        path = url.replace(config.get_api_base(), "")
        return client.request(method, path, headers=headers, params=kw.get("params"), json=kw.get("json"))

    monkeypatch.setattr(requests, "request", fake_request)
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30)
    at.run()
    return at


def sign_in(at, user="demo_aarav", pw="Demo@1234"):
    at.text_input[0].set_value(user)
    at.text_input[1].set_value(pw)
    at.button[0].click()
    at.run()
    return at


def test_login_page_renders(ui):
    assert not ui.exception
    assert any("Sign In" in s.value for s in ui.subheader)


def test_wrong_password_shows_error(ui):
    sign_in(ui, pw="wrong")
    assert any("Invalid username" in e.value for e in ui.error)
    assert not ui.session_state["authenticated"]


def test_login_and_dashboard(ui):
    sign_in(ui)
    assert not ui.exception and ui.session_state["authenticated"]
    assert ui.session_state["customer_id"] == "CUST0001"
    assert len(ui.tabs) == 4


def test_chat_end_to_end(ui):
    sign_in(ui)
    ui.session_state["_inject_question"] = "What is my account balance?"
    ui.run()
    assert not ui.exception
    msgs = ui.session_state["messages"]
    assert msgs[-1]["role"] == "assistant" and "₹" in msgs[-1]["content"]
    assert msgs[-1]["meta"]["intent"] == "BALANCE"


def test_html_in_user_message_is_not_executed(ui):
    sign_in(ui)
    ui.session_state["_inject_question"] = "<script>alert(1)</script> what is my balance"
    ui.run()
    assert not ui.exception
    # user text is shown through st.markdown (no unsafe_allow_html) so it is inert
    assert all("unsafe" not in str(m.proto) for m in ui.chat_message)


def test_consent_saved_from_ui(ui, restore_consent):
    sign_in(ui, "demo_priya", "Demo@5678")
    boxes = {c.label: c for c in ui.checkbox}
    marketing = next(c for l, c in boxes.items() if l.startswith("Marketing"))
    marketing.check()
    next(b for b in ui.button if b.label == "Save my choices").click()
    ui.run()
    assert db.get_consent("CUST0002")["marketing"] is True


def test_sign_out(ui):
    sign_in(ui)
    next(b for b in ui.sidebar.button if "Sign Out" in b.label).click()
    ui.run()
    assert not ui.session_state["authenticated"]
