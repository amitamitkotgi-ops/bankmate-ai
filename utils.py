"""
BANKMATE AI — utils.py
Small formatting helpers shared by the backend and the Streamlit app.
"""

from __future__ import annotations


def fmt_inr(amount: float | int | None, decimals: int = 2) -> str:
    """Format a number as Indian rupees with lakh/crore grouping: ₹12,34,567.50"""
    if amount is None:
        return "—"
    negative = amount < 0
    value = abs(float(amount))
    text = f"{value:.{decimals}f}"
    whole, _, frac = text.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    out = "₹" + whole + (f".{frac}" if decimals else "")
    return "-" + out if negative else out


def fmt_pct(value: float) -> str:
    """10.5 -> '10.5%', 7.0 -> '7.0%'"""
    return f"{value:.1f}%"


def fmt_months(months: int) -> str:
    """36 -> '3 years', 18 -> '1 year 6 months', 6 -> '6 months'"""
    years, rem = divmod(int(months), 12)
    parts = []
    if years:
        parts.append(f"{years} year" + ("s" if years != 1 else ""))
    if rem:
        parts.append(f"{rem} month" + ("s" if rem != 1 else ""))
    return " ".join(parts) if parts else "0 months"
