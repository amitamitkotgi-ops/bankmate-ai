"""Data-consideration checks: generated JSON/log files and the labelled evaluation set."""

import json
import re

import pytest

import generate_data as gd
import products
from nlp import classify_intent
from privacy import check_message_risks, redact_pii


def test_products_json_is_valid_structured_data():
    data = json.loads(gd.PRODUCTS_PATH.read_text(encoding="utf-8"))
    assert data["synthetic"] is True and data["currency"] == "INR" and "illustrative" in data["disclaimer"].lower()
    ids = [p["product_id"] for p in data["products"]]
    assert len(ids) == len(set(ids)) == 9
    assert {p["category"] for p in data["products"]} == {"account", "deposit", "loan"}
    for p in data["products"]:
        assert 0 <= p["rate_min"] <= p["rate_max"] <= 30
        if p["category"] == "loan":
            for k in ("min_amount", "max_amount", "min_tenure_months", "max_tenure_months",
                      "min_age", "max_age", "min_monthly_income", "max_foir"):
                assert k in p, (p["product_id"], k)
            assert p["min_amount"] < p["max_amount"] and p["min_tenure_months"] < p["max_tenure_months"]


def test_products_loader_rejects_bad_catalogue(tmp_path, monkeypatch):
    bad = tmp_path / "p.json"
    bad.write_text(json.dumps({"products": [{"product_id": "x"}]}))
    monkeypatch.setattr(products, "PRODUCTS_PATH", bad)
    products.reload_catalogue()
    try:
        with pytest.raises(products.ProductCatalogueError):
            products.all_products()
    finally:
        monkeypatch.undo()
        products.reload_catalogue()


def test_generation_is_deterministic():
    assert gd.build_query_set() == gd.build_query_set()
    assert gd.build_products() == gd.build_products()


def test_query_log_format_and_masking():
    lines = [l for l in gd.QUERY_LOG_PATH.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
    assert len(lines) == 600
    pattern = re.compile(r'^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \| Q\d{6} \| CUST\d{4} \| S\d{3} \| ".+"$')
    assert all(pattern.match(l) for l in lines)
    for l in lines:                       # the log itself must be free of PII
        assert redact_pii(l.split(" | ", 4)[4])[1] == []


def test_eval_set_covers_every_intent_and_style():
    items = [json.loads(l) for l in gd.EVAL_SET_PATH.read_text(encoding="utf-8").splitlines()]
    assert {i["intent"] for i in items if i["intent"]} >= {
        "BALANCE", "TRANSACTIONS", "SPENDING_SUMMARY", "PRODUCT_INFO", "LOAN_ELIGIBILITY",
        "PROFILE", "PRIVACY", "HELP", "GREETING", "OUT_OF_SCOPE"}
    assert {i["style"] for i in items} == {"clean", "noisy", "hinglish", "security"}


def test_rules_accuracy_floor():
    items = [json.loads(l) for l in gd.EVAL_SET_PATH.read_text(encoding="utf-8").splitlines()]
    scored = [i for i in items if not i["block"]]
    ok = 0
    for i in scored:
        got = classify_intent(i["text"])
        ok += got == i["intent"] or (i["intent"] == "OUT_OF_SCOPE" and got in ("OUT_OF_SCOPE", "UNKNOWN"))
    assert ok / len(scored) >= 0.93
    clean = [i for i in scored if i["style"] == "clean"]
    assert all(classify_intent(i["text"]) == i["intent"] or i["intent"] == "OUT_OF_SCOPE" for i in clean)


def test_every_security_case_is_blocked():
    for i in (x for x in map(json.loads, gd.EVAL_SET_PATH.read_text(encoding="utf-8").splitlines()) if x["block"]):
        assert check_message_risks(i["text"], "CUST0001") == i["block"], i
