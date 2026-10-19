"""
BANKMATE AI — products.py
Loads and validates the banking product catalogue (data/products.json).
"""

from __future__ import annotations

import json
from functools import lru_cache

from generate_data import PRODUCT_DISCLAIMER, PRODUCTS_PATH, ensure_products_file
from utils import fmt_inr, fmt_months

REQUIRED_KEYS = {"product_id", "name", "category", "rate_min", "rate_max", "rate_basis", "description"}
CATEGORIES = {"account", "deposit", "loan"}


class ProductCatalogueError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load() -> dict:
    ensure_products_file()
    try:
        data = json.loads(PRODUCTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductCatalogueError(f"Cannot read {PRODUCTS_PATH.name}: {exc}") from exc

    products = data.get("products")
    if not isinstance(products, list) or not products:
        raise ProductCatalogueError("products.json has no products")
    seen = set()
    for p in products:
        missing = REQUIRED_KEYS - set(p)
        if missing:
            raise ProductCatalogueError(f"Product {p.get('product_id')} missing keys: {sorted(missing)}")
        if p["category"] not in CATEGORIES:
            raise ProductCatalogueError(f"Product {p['product_id']} has bad category {p['category']!r}")
        if p["product_id"] in seen:
            raise ProductCatalogueError(f"Duplicate product_id {p['product_id']}")
        seen.add(p["product_id"])
    return data


def reload_catalogue() -> None:
    _load.cache_clear()


def disclaimer() -> str:
    return _load().get("disclaimer", PRODUCT_DISCLAIMER)


def all_products() -> list[dict]:
    return [dict(p) for p in _load()["products"]]


def product_ids() -> list[str]:
    return [p["product_id"] for p in _load()["products"]]


def get_product(product_id: str) -> dict | None:
    for p in _load()["products"]:
        if p["product_id"] == product_id:
            return dict(p)
    return None


def list_products(category: str | None = None) -> list[dict]:
    items = all_products()
    if category:
        items = [p for p in items if p["category"] == category]
    return items


def with_display(product: dict) -> dict:
    """Add human-readable *_display fields (used in chat answers and the UI)."""
    p = dict(product)
    if p["rate_max"] == 0 and p["rate_min"] == 0:
        p["rate_display"] = "No interest paid"
    elif p["rate_min"] == p["rate_max"]:
        p["rate_display"] = f"{p['rate_min']}% {p['rate_basis']}"
    else:
        p["rate_display"] = f"{p['rate_min']}% – {p['rate_max']}% {p['rate_basis']}"
    if "min_tenure_months" in p:
        p["tenure_display"] = f"{fmt_months(p['min_tenure_months'])} – {fmt_months(p['max_tenure_months'])}"
    if "min_amount" in p:
        p["amount_display"] = f"{fmt_inr(p['min_amount'], 0)} – {fmt_inr(p['max_amount'], 0)}"
    if "minimum_balance" in p:
        p["minimum_balance_display"] = fmt_inr(p["minimum_balance"], 0)
    return p
