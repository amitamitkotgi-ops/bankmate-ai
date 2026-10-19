"""
BANKMATE AI — database.py
SQLite schema, synthetic data generation, authentication helpers and queries.
All data is entirely fictional and for demonstration purposes only.

Data compliance features
  * customer profiles carry CONSENT FLAGS (AI processing, query logging, marketing)
  * only masked / redacted text is ever stored in query logs, and only with consent
  * audit_logs keep security events WITHOUT message content
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import random
import secrets
import sqlite3
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "bankmate.db")
SCHEMA_VERSION = 3


def today() -> date:
    """Single place to get 'today' (easy to patch in tests)."""
    return date.today()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Password utilities (PBKDF2-HMAC-SHA256, standard library only)
# ---------------------------------------------------------------------------

# OWASP (2023+) recommends 600,000 iterations for PBKDF2-HMAC-SHA256.
# The count is stored inside each hash, so it can be raised later without
# breaking old hashes. Tests lower this number to run fast.
_ITERATIONS = 600_000
_HASH_PREFIX = "pbkdf2_sha256"


def hash_password(plain: str) -> str:
    """Return 'pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>'."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", plain.encode(), salt.encode(), _ITERATIONS)
    return f"{_HASH_PREFIX}${_ITERATIONS}${salt}${dk.hex()}"


def verify_password(plain: str, stored: str) -> bool:
    """Constant-time comparison of a plain password against a stored hash."""
    try:
        algo, iterations, salt, stored_hash = stored.split("$", 3)
        if algo != _HASH_PREFIX:
            return False
        dk = hashlib.pbkdf2_hmac("sha256", plain.encode(), salt.encode(), int(iterations))
        return hmac.compare_digest(dk.hex(), stored_hash)
    except Exception:
        return False


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return hash_password("dummy-password-for-timing")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id             TEXT PRIMARY KEY,
    name                    TEXT NOT NULL,
    email                   TEXT NOT NULL,
    phone                   TEXT NOT NULL,
    age                     INTEGER NOT NULL,
    declared_monthly_income REAL NOT NULL,
    existing_emi            REAL NOT NULL DEFAULT 0,
    consent_ai_processing   INTEGER NOT NULL DEFAULT 0,
    consent_query_logging   INTEGER NOT NULL DEFAULT 0,
    consent_marketing       INTEGER NOT NULL DEFAULT 0,
    consent_updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL UNIQUE REFERENCES customers(customer_id),
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS login_throttle (
    username        TEXT PRIMARY KEY,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until    TEXT
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id      TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id),
    account_type    TEXT NOT NULL,
    opening_balance REAL NOT NULL,
    balance         REAL NOT NULL,
    currency        TEXT NOT NULL DEFAULT 'INR',
    status          TEXT NOT NULL DEFAULT 'Active'
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id   TEXT PRIMARY KEY,
    customer_id      TEXT NOT NULL REFERENCES customers(customer_id),
    account_id       TEXT NOT NULL REFERENCES accounts(account_id),
    transaction_date TEXT NOT NULL,
    description      TEXT NOT NULL,
    category         TEXT NOT NULL,
    amount           REAL NOT NULL,
    transaction_type TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'Completed'
);

CREATE TABLE IF NOT EXISTS audit_logs (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time  TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    username    TEXT,
    customer_id TEXT,
    detail      TEXT
);

CREATE TABLE IF NOT EXISTS query_logs (
    log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time   TEXT NOT NULL,
    customer_id  TEXT NOT NULL,
    intent       TEXT,
    tool         TEXT,
    mode         TEXT,
    llm_used     INTEGER NOT NULL DEFAULT 0,
    blocked      TEXT,
    pii_types    TEXT,
    masked_query TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transactions_customer
    ON transactions(customer_id, transaction_date DESC);
CREATE INDEX IF NOT EXISTS idx_accounts_customer ON accounts(customer_id);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_query_logs_customer ON query_logs(customer_id);
"""


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Sneha", "Kiran", "Divya", "Arjun", "Meera", "Vikram", "Ananya",
    "Rahul", "Pooja", "Suresh", "Kavya", "Nikhil", "Swathi", "Aditya", "Deepika", "Sanjay", "Lakshmi",
    "Harish", "Nandini", "Ravi", "Anjali", "Mohan", "Sunita", "Ajay", "Rekha", "Vinod", "Geeta",
    "Sachin", "Preeti", "Manish", "Usha", "Dinesh", "Savita", "Rajesh", "Hema", "Naveen", "Shilpa",
    "Sunil", "Radha", "Ganesh", "Padma", "Manoj", "Smita", "Vivek", "Seema", "Ashok", "Rani",
]
LAST_NAMES = [
    "Sharma", "Patel", "Kumar", "Singh", "Gupta", "Reddy", "Nair", "Iyer", "Joshi", "Mehta",
    "Verma", "Pillai", "Rao", "Mishra", "Bhatt", "Chauhan", "Desai", "Tiwari", "Pandey", "Shah",
    "Malhotra", "Kapoor", "Chopra", "Saxena", "Srivastava", "Jain", "Agarwal", "Bansal", "Mittal", "Goyal",
    "Naik", "Patil", "Gaikwad", "More", "Shinde", "Deshpande", "Kulkarni", "Jadhav", "Pawar", "Bhosale",
    "Menon", "Krishnan", "Nambiar", "Varma", "Sudha", "Rajan", "Bhat", "Shetty", "Kamath", "Hegde",
]

# (description, category, min amount, max amount, weight)
DEBIT_CATALOG = [
    ("Grocery Store Payment", "Groceries", 200, 4500, 4),
    ("Restaurant Payment", "Dining", 250, 3500, 2),
    ("Coffee Shop", "Dining", 80, 600, 2),
    ("Bakery Purchase", "Dining", 60, 700, 1),
    ("Electricity Bill", "Utilities", 400, 4500, 1),
    ("Mobile Recharge", "Utilities", 199, 999, 1),
    ("Water Bill", "Utilities", 150, 900, 1),
    ("Gas Cylinder", "Utilities", 850, 1200, 1),
    ("Online Shopping", "Shopping", 300, 12000, 2),
    ("Clothing Store", "Shopping", 500, 9000, 1),
    ("Hardware Store", "Shopping", 150, 5000, 1),
    ("Fuel Station", "Transport", 300, 4000, 2),
    ("Auto Rickshaw Pay", "Transport", 40, 400, 2),
    ("Medical Pharmacy", "Health", 100, 3500, 1),
    ("OTT Subscription", "Entertainment", 149, 999, 1),
    ("Movie Ticket", "Entertainment", 200, 1200, 1),
    ("Gym Membership", "Entertainment", 500, 3000, 1),
    ("School Fee", "Education", 1500, 25000, 1),
    ("Book Store", "Education", 150, 2500, 1),
]
OTHER_CREDIT_CATALOG = [
    ("Freelance Payment", "Other Income", 2000, 25000),
    ("UPI Transfer Received", "Other Income", 500, 10000),
    ("Refund Credit", "Other Income", 100, 5000),
    ("Interest Credit", "Other Income", 50, 2500),
    ("Cashback Credit", "Other Income", 10, 500),
    ("Bonus Credit", "Other Income", 5000, 40000),
]

# Demo logins (FICTIONAL). Consent flags differ on purpose so you can
# demonstrate how consent changes the assistant's behaviour:
#   aarav: all ON            priya: no marketing      rohan: no query logging
#   sneha: privacy-minded    kiran: AI processing OFF (no data sent to the LLM)
DEMO_USERS = [
    {"user_id": "USR0001", "customer_id": "CUST0001", "username": "demo_aarav", "password": "Demo@1234",
     "consent": (1, 1, 1)},
    {"user_id": "USR0002", "customer_id": "CUST0002", "username": "demo_priya", "password": "Demo@5678",
     "consent": (1, 1, 0)},
    {"user_id": "USR0003", "customer_id": "CUST0003", "username": "demo_rohan", "password": "Demo@9012",
     "consent": (1, 0, 1)},
    {"user_id": "USR0004", "customer_id": "CUST0004", "username": "demo_sneha", "password": "Demo@3456",
     "consent": (1, 0, 0)},
    {"user_id": "USR0005", "customer_id": "CUST0005", "username": "demo_kiran", "password": "Demo@7890",
     "consent": (0, 1, 1)},
]


def _month_start(ref: date, months_back: int) -> date:
    month_index = ref.year * 12 + (ref.month - 1) - months_back
    return date(month_index // 12, month_index % 12 + 1, 1)


def _month_end(start: date) -> date:
    nxt = _month_start(start, -1)
    return nxt - timedelta(days=1)


def _generate_synthetic_data(conn: sqlite3.Connection) -> None:
    """Insert synthetic customers, accounts and transactions (only if empty)."""
    if conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] > 0:
        return

    rng = random.Random(42)  # deterministic
    ref_today = today()
    now = _now_iso()
    demo_consent = {u["customer_id"]: u["consent"] for u in DEMO_USERS}

    customers, accounts, transactions = [], [], []
    weights = [w for *_, w in DEBIT_CATALOG]

    for i in range(1, 51):
        cid = f"CUST{i:04d}"
        first, last = FIRST_NAMES[i - 1], LAST_NAMES[i - 1]
        email = f"{first.lower()}.{last.lower()}@demobank.in"
        phone = f"9{rng.randint(100000000, 999999999)}"
        age = rng.randint(21, 58)
        income = rng.choice(range(25000, 150001, 5000))
        existing_emi = rng.choice([0, 0, 0, 3000, 5000, 8000, 12000])
        ai_c, log_c, mkt_c = demo_consent.get(
            cid, (int(rng.random() < 0.85), int(rng.random() < 0.7), int(rng.random() < 0.4))
        )
        customers.append((cid, f"{first} {last}", email, phone, age, float(income),
                          float(existing_emi), ai_c, log_c, mkt_c, now))

        primary_id = f"ACC{i:06d}"
        account_ids = [primary_id]
        if rng.random() < 0.35:
            account_ids.append(f"ACC{500000 + i:06d}")
        net = {a: 0.0 for a in account_ids}

        txn_rows = []  # (date, desc, category, amount, type, account)
        for m in range(12):
            start = _month_start(ref_today, m)
            end = min(_month_end(start), ref_today)
            span = (end - start).days
            txn_rows.append((start, "Salary Credit", "Salary", float(income), "Credit", primary_id))
            for _ in range(rng.randint(4, 8)):
                d = start + timedelta(days=rng.randint(0, max(span, 0)))
                desc, cat, lo, hi, _w = rng.choices(DEBIT_CATALOG, weights=weights, k=1)[0]
                amount = round(rng.uniform(lo, hi), 2)
                acc = rng.choice(account_ids) if len(account_ids) > 1 and rng.random() < 0.3 else primary_id
                txn_rows.append((d, desc, cat, amount, "Debit", acc))
            if rng.random() < 0.6:
                d = start + timedelta(days=rng.randint(0, max(span, 0)))
                desc, cat, lo, hi = rng.choice(OTHER_CREDIT_CATALOG)
                txn_rows.append((d, desc, cat, round(rng.uniform(lo, hi), 2), "Credit", primary_id))

        txn_rows.sort(key=lambda r: (r[0], r[1]))
        for seq, (d, desc, cat, amount, ttype, acc) in enumerate(txn_rows, 1):
            transactions.append((f"TXN{i:04d}{seq:04d}", cid, acc, d.isoformat(), desc, cat,
                                 amount, ttype, "Completed"))
            net[acc] += amount if ttype == "Credit" else -amount

        for idx, acc_id in enumerate(account_ids):
            acc_type = "Savings" if idx == 0 else "Current"
            opening = float(rng.randint(10000, 150000))
            opening = max(opening, round(-net[acc_id] + rng.randint(2000, 20000), 2))
            balance = round(opening + net[acc_id], 2)
            accounts.append((acc_id, cid, acc_type, round(opening, 2), balance, "INR", "Active"))

    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?)", customers)
    conn.executemany("INSERT INTO accounts VALUES (?,?,?,?,?,?,?)", accounts)
    conn.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?)", transactions)
    conn.commit()


def _seed_demo_users(conn: sqlite3.Connection) -> None:
    for u in DEMO_USERS:
        if not conn.execute("SELECT 1 FROM users WHERE user_id = ?", (u["user_id"],)).fetchone():
            conn.execute(
                "INSERT INTO users (user_id, customer_id, username, password_hash) VALUES (?,?,?,?)",
                (u["user_id"], u["customer_id"], u["username"], hash_password(u["password"])),
            )
    conn.commit()


def _needs_rebuild() -> bool:
    """True when an existing database file was made by an older version."""
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        finally:
            conn.close()
        return version != SCHEMA_VERSION
    except sqlite3.DatabaseError:
        return True


def initialize_database() -> None:
    """Create tables and seed synthetic data. Safe to call repeatedly."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    if _needs_rebuild():
        backup = f"{DB_PATH}.old-{datetime.now():%Y%m%d%H%M%S}"
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(DB_PATH + suffix):
                os.replace(DB_PATH + suffix, backup + suffix)
        logger.warning("Old database schema found; backed up to %s and rebuilding.", backup)

    with get_connection() as conn:
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        _generate_synthetic_data(conn)
        _seed_demo_users(conn)


# ---------------------------------------------------------------------------
# Audit and query logs
# ---------------------------------------------------------------------------


def write_audit(event_type: str, username: str | None = None,
                customer_id: str | None = None, detail: str = "") -> None:
    """Security-relevant event. NEVER put message text or secrets in `detail`."""
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO audit_logs (event_time, event_type, username, customer_id, detail) "
                "VALUES (?,?,?,?,?)",
                (_now_iso(), event_type, username, customer_id, detail[:300]),
            )
            conn.commit()
    except Exception as exc:  # logging must never break the request
        logger.error("Audit log write failed: %s", exc)


def log_query(customer_id: str, masked_query: str, intent: str | None, tool: str | None,
              mode: str, llm_used: bool, blocked: str | None, pii_types: list[str]) -> None:
    """Store a PII-redacted query. Call only when the customer consented to logging."""
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO query_logs (event_time, customer_id, intent, tool, mode, llm_used, "
                "blocked, pii_types, masked_query) VALUES (?,?,?,?,?,?,?,?,?)",
                (_now_iso(), customer_id, intent, tool, mode, int(llm_used), blocked,
                 ",".join(pii_types), masked_query[:500]),
            )
            conn.commit()
    except Exception as exc:
        logger.error("Query log write failed: %s", exc)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_MAX_ATTEMPTS = 5
_LOCKOUT_MINUTES = 15


def get_user_by_username(username: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT user_id, customer_id, username, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def is_locked_out(username: str) -> bool:
    """Lockout is tracked per username string (real or not) so responses never reveal who exists."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT locked_until FROM login_throttle WHERE username = ?", (username,)
        ).fetchone()
    if not row or not row["locked_until"]:
        return False
    try:
        until = datetime.fromisoformat(row["locked_until"])
    except ValueError:
        return False
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) < until


def record_failed_attempt(username: str) -> None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT failed_attempts FROM login_throttle WHERE username = ?", (username,)
        ).fetchone()
        count = (row["failed_attempts"] if row else 0) + 1
        locked_until = None
        if count >= _MAX_ATTEMPTS:
            locked_until = (datetime.now(timezone.utc) + timedelta(minutes=_LOCKOUT_MINUTES)).isoformat()
        conn.execute(
            "INSERT INTO login_throttle (username, failed_attempts, locked_until) VALUES (?,?,?) "
            "ON CONFLICT(username) DO UPDATE SET failed_attempts=excluded.failed_attempts, "
            "locked_until=excluded.locked_until",
            (username, count, locked_until),
        )
        conn.commit()
    write_audit("LOGIN_FAILED", username, None, f"attempt {count}")


def reset_failed_attempts(username: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM login_throttle WHERE username = ?", (username,))
        conn.commit()


def authenticate_user(username: str, password: str) -> dict | None:
    """
    Return {'user_id','customer_id','username'} on success, else None.
    Runs a password hash even for unknown users so timing does not reveal
    whether the username exists.
    """
    if is_locked_out(username):
        write_audit("LOGIN_LOCKED", username, None, "locked out")
        return None

    user = get_user_by_username(username)
    if not user:
        verify_password(password, _dummy_hash())
        record_failed_attempt(username)
        return None

    if not verify_password(password, user["password_hash"]):
        record_failed_attempt(username)
        return None

    reset_failed_attempts(username)
    write_audit("LOGIN_SUCCESS", username, user["customer_id"], "authenticated")
    return {"user_id": user["user_id"], "customer_id": user["customer_id"], "username": user["username"]}


# ---------------------------------------------------------------------------
# Customer data queries (always parameterised)
# ---------------------------------------------------------------------------


def _consent_from_row(row: sqlite3.Row) -> dict:
    return {
        "ai_processing": bool(row["consent_ai_processing"]),
        "query_logging": bool(row["consent_query_logging"]),
        "marketing": bool(row["consent_marketing"]),
        "updated_at": row["consent_updated_at"],
    }


def get_customer(customer_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["consent"] = _consent_from_row(row)
    for key in ("consent_ai_processing", "consent_query_logging", "consent_marketing", "consent_updated_at"):
        d.pop(key, None)
    return d


def get_consent(customer_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT consent_ai_processing, consent_query_logging, consent_marketing, consent_updated_at "
            "FROM customers WHERE customer_id = ?", (customer_id,),
        ).fetchone()
    return _consent_from_row(row) if row else None


def update_consent(customer_id: str, ai_processing: bool | None = None,
                   query_logging: bool | None = None, marketing: bool | None = None) -> dict | None:
    sets, params = [], []
    for column, value in (("consent_ai_processing", ai_processing),
                          ("consent_query_logging", query_logging),
                          ("consent_marketing", marketing)):
        if value is not None:
            sets.append(f"{column} = ?")
            params.append(int(bool(value)))
    if not sets:
        return get_consent(customer_id)
    sets.append("consent_updated_at = ?")
    params.append(_now_iso())
    params.append(customer_id)
    with get_connection() as conn:
        cur = conn.execute(f"UPDATE customers SET {', '.join(sets)} WHERE customer_id = ?", params)
        conn.commit()
        if cur.rowcount == 0:
            return None
    return get_consent(customer_id)


def get_accounts(customer_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT account_id, customer_id, account_type, balance, currency, status "
            "FROM accounts WHERE customer_id = ? ORDER BY account_id", (customer_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_transactions(customer_id: str, limit: int = 10, days: int | None = None,
                     category: str | None = None, txn_type: str | None = None) -> list[dict]:
    limit = max(1, min(int(limit), 100))
    sql = ("SELECT transaction_id, account_id, transaction_date, description, category, amount, "
           "transaction_type, status FROM transactions WHERE customer_id = ?")
    params: list = [customer_id]
    if days:
        sql += " AND transaction_date >= ?"
        params.append((today() - timedelta(days=int(days))).isoformat())
    if category:
        sql += " AND category = ?"
        params.append(category)
    if txn_type:
        sql += " AND transaction_type = ?"
        params.append(txn_type)
    sql += " ORDER BY transaction_date DESC, transaction_id DESC LIMIT ?"
    params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_analytics_summary(customer_id: str, days: int | None = None,
                          category: str | None = None) -> dict:
    """Totals plus a per-category debit breakdown for the chosen window."""
    where = "WHERE customer_id = ?"
    params: list = [customer_id]
    if days:
        where += " AND transaction_date >= ?"
        params.append((today() - timedelta(days=int(days))).isoformat())
    if category:
        where += " AND category = ?"
        params.append(category)

    with get_connection() as conn:
        row = conn.execute(
            f"""SELECT COUNT(*) AS transaction_count,
                       COALESCE(SUM(CASE WHEN transaction_type='Credit' THEN amount END), 0) AS total_credit,
                       COALESCE(SUM(CASE WHEN transaction_type='Debit'  THEN amount END), 0) AS total_debit,
                       COALESCE(SUM(CASE WHEN transaction_type='Credit' THEN 1 END), 0) AS credit_count,
                       COALESCE(SUM(CASE WHEN transaction_type='Debit'  THEN 1 END), 0) AS debit_count
                FROM transactions {where}""", params,
        ).fetchone()
        cats = conn.execute(
            f"""SELECT category, ROUND(SUM(amount), 2) AS total, COUNT(*) AS count
                FROM transactions {where} AND transaction_type = 'Debit'
                GROUP BY category ORDER BY total DESC""", params,
        ).fetchall()

    d = dict(row)
    d["total_credit"] = round(d["total_credit"], 2)
    d["total_debit"] = round(d["total_debit"], 2)
    d["net_flow"] = round(d["total_credit"] - d["total_debit"], 2)
    d["by_category"] = [dict(c) for c in cats]
    return d


def count_customers() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
