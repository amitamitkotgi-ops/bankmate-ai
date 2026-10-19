"""
BANKMATE AI — api.py
FastAPI backend: mock banking services + the LangGraph chat endpoint.
All data is fictional and for demonstration purposes only.

Security model
  * POST /auth/login returns a random session token (in-memory, 2 h expiry).
  * Every customer endpoint needs the token AND the customer_id in the URL
    must match the token's customer (no IDOR). /chat and /loans/eligibility
    take NO customer_id at all - it always comes from the session.
  * Account number, e-mail and phone are masked on the server.
  * The Groq API key lives only in this process (.env) - never in the browser.

PROTOTYPE NOTE: sessions are kept in memory. For production use a shared
store (Redis/DB), HTTPS and HttpOnly cookies. See README.md.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
import database
import products as catalogue
import services
from agent import BankingAgent
from services import DATA_NOTICE, ServiceError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sessions (in-memory prototype)
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}
_MAX_SESSIONS = 2000
_chat_hits: dict[str, deque] = {}


def _purge_expired_sessions() -> None:
    now = datetime.now(timezone.utc)
    for token in [t for t, s in _sessions.items() if datetime.fromisoformat(s["expires_at"]) < now]:
        _sessions.pop(token, None)
        _chat_hits.pop(token, None)
    while len(_sessions) > _MAX_SESSIONS:  # drop the oldest
        oldest = next(iter(_sessions))
        _sessions.pop(oldest, None)
        _chat_hits.pop(oldest, None)


def _create_session(customer_id: str, username: str) -> str:
    _purge_expired_sessions()
    token = secrets.token_urlsafe(32)
    _sessions[token] = {
        "customer_id": customer_id,
        "username": username,
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=config.SESSION_TTL_HOURS)).isoformat(),
    }
    return token


def _get_session(token: str) -> dict | None:
    session = _sessions.get(token)
    if not session:
        return None
    try:
        if datetime.now(timezone.utc) > datetime.fromisoformat(session["expires_at"]):
            _sessions.pop(token, None)
            return None
    except (ValueError, KeyError):
        return None
    return session


def _session_ref(token: str) -> str:
    """Opaque id used as the chat-memory key (the raw token is never stored elsewhere)."""
    return hashlib.sha256(token.encode()).hexdigest()[:24]


def _end_session(token: str) -> None:
    _sessions.pop(token, None)
    _chat_hits.pop(token, None)
    get_agent().reset(_session_ref(token))


def _require_session(x_session_token: str | None = Header(default=None)) -> dict:
    if not x_session_token:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")
    session = _get_session(x_session_token)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalid. Please log in again.")
    return {**session, "token": x_session_token}


def _authorize_customer(session: dict, customer_id: str) -> None:
    if session["customer_id"] != customer_id:
        logger.warning("Authorization failure: customer %s tried to access %s",
                       session.get("customer_id"), customer_id)
        database.write_audit("ACCESS_DENIED", session.get("username"), session.get("customer_id"),
                             "tried to access another customer's resource")
        raise HTTPException(status_code=403,
                            detail="Access denied. You may only access your own account information.")


def _check_chat_rate_limit(token: str) -> None:
    now = time.monotonic()
    hits = _chat_hits.setdefault(token, deque())
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= config.CHAT_RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many messages. Please wait a moment and try again.")
    hits.append(now)


# ---------------------------------------------------------------------------
# Agent (created at start-up; tests can replace it)
# ---------------------------------------------------------------------------

_agent: BankingAgent | None = None


def get_agent() -> BankingAgent:
    global _agent
    if _agent is None:
        _agent = BankingAgent()
    return _agent


def set_agent(agent: BankingAgent | None) -> None:
    global _agent
    _agent = agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.initialize_database()
    agent = get_agent()
    status = agent.llm.status()
    if status["available"]:
        logger.info("Groq LLM enabled (model: %s)", status["model"])
    else:
        logger.warning("Groq LLM NOT active (%s). The assistant will use the rule-based fallback. "
                       "Put your key in .env as GROQ_API_KEY=...", status["reason"])
    yield


app = FastAPI(
    title="BANKMATE AI Banking API",
    description=("Mock banking services + LangGraph/Groq chat agent for the TCS Technology Day "
                 "'AI-Powered Banking Customer Query Assistant'. All data is synthetic."),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get_cors_origins(),
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "X-Session-Token"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=200)


class LoginResponse(BaseModel):
    session_token: str
    customer_id: str
    username: str
    message: str


class ChatRequest(BaseModel):
    message: str = Field(max_length=2000)


class ChatResponse(BaseModel):
    reply: str
    intent: str | None = None
    tool: str | None = None
    mode: str
    llm_used: bool
    model: str | None = None
    classifier: str | None = None
    entities: dict = {}
    blocked: str | None = None
    pii_redacted: list[str] = []
    guardrail_flags: list[str] = []
    fallback_reason: str | None = None
    followup: bool = False


class ConsentUpdate(BaseModel):
    ai_processing: bool | None = None
    query_logging: bool | None = None
    marketing: bool | None = None


class EligibilityRequest(BaseModel):
    product_id: str = Field(max_length=50)
    loan_amount: float | None = Field(default=None, gt=0, le=1_000_000_000)
    tenure_months: int | None = Field(default=None, ge=1, le=480)
    monthly_income: float | None = Field(default=None, ge=0, le=100_000_000)
    existing_emi: float | None = Field(default=None, ge=0, le=100_000_000)


def _service_call(fn, *args, **kwargs) -> Any:
    try:
        return fn(*args, **kwargs)
    except ServiceError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from exc


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
def health() -> Any:
    try:
        db_status = f"Connected — {database.count_customers()} demo customers available"
    except Exception:
        db_status = "Unavailable"
    return {
        "status": "ok",
        "application": "BANKMATE AI Banking API",
        "database": db_status,
        "llm": get_agent().llm.status(),
        "data_notice": DATA_NOTICE,
    }


@app.get("/chat/graph", tags=["Chat"])
def chat_graph() -> Any:
    """Mermaid diagram of the LangGraph workflow (for docs / demos)."""
    return {"mermaid": get_agent().mermaid()}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


@app.post("/auth/login", response_model=LoginResponse, tags=["Authentication"])
def login(body: LoginRequest) -> Any:
    username = body.username.strip()
    password = body.password
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required.")

    # Same answer whether or not the username exists.
    if database.is_locked_out(username):
        raise HTTPException(status_code=429,
                            detail="Too many failed attempts. Please try again in 15 minutes.")
    user = database.authenticate_user(username, password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = _create_session(user["customer_id"], user["username"])
    return LoginResponse(session_token=token, customer_id=user["customer_id"],
                         username=user["username"], message="Login successful. Welcome to BANKMATE AI.")


@app.post("/auth/logout", tags=["Authentication"])
def logout(x_session_token: str | None = Header(default=None)) -> Any:
    if x_session_token and x_session_token in _sessions:
        _end_session(x_session_token)
    return {"message": "Logged out successfully."}


# ---------------------------------------------------------------------------
# Customer data (token + own customer_id required)
# ---------------------------------------------------------------------------


@app.get("/customers/{customer_id}", tags=["Customers"])
def get_customer(customer_id: str, session: dict = Depends(_require_session)) -> Any:
    _authorize_customer(session, customer_id)
    profile = _service_call(services.get_profile, customer_id)
    return {**profile, "data_notice": DATA_NOTICE}


@app.get("/accounts/{customer_id}", tags=["Accounts"])
def get_accounts(customer_id: str, session: dict = Depends(_require_session)) -> Any:
    _authorize_customer(session, customer_id)
    data = _service_call(services.get_balance, customer_id)
    return {"customer_id": customer_id, **data, "data_notice": DATA_NOTICE}


@app.get("/transactions/{customer_id}", tags=["Transactions"])
def get_transactions(
    customer_id: str,
    limit: int = Query(default=10, ge=1, le=50),
    days: int | None = Query(default=None, ge=1, le=365),
    category: str | None = Query(default=None, max_length=30),
    type: str | None = Query(default=None, pattern="^(Credit|Debit)$"),
    session: dict = Depends(_require_session),
) -> Any:
    _authorize_customer(session, customer_id)
    data = _service_call(services.get_transactions, customer_id, limit=limit, period_days=days,
                         category=category, txn_type=type, period_label=f"last {days} days" if days else None)
    return {"customer_id": customer_id, **data, "data_notice": DATA_NOTICE}


@app.get("/analytics/summary/{customer_id}", tags=["Analytics"])
def analytics_summary(
    customer_id: str,
    days: int | None = Query(default=365, ge=1, le=365),
    session: dict = Depends(_require_session),
) -> Any:
    _authorize_customer(session, customer_id)
    data = _service_call(services.get_spending_summary, customer_id, period_days=days)
    return {"customer_id": customer_id, **data, "data_notice": DATA_NOTICE}


@app.get("/consent/{customer_id}", tags=["Privacy"])
def read_consent(customer_id: str, session: dict = Depends(_require_session)) -> Any:
    _authorize_customer(session, customer_id)
    return {"customer_id": customer_id, "consent": _service_call(services.get_consent, customer_id)}


@app.put("/consent/{customer_id}", tags=["Privacy"])
def update_consent(customer_id: str, body: ConsentUpdate, session: dict = Depends(_require_session)) -> Any:
    _authorize_customer(session, customer_id)
    consent = _service_call(services.set_consent, customer_id, session.get("username"),
                            body.ai_processing, body.query_logging, body.marketing)
    return {"customer_id": customer_id, "consent": consent}


# ---------------------------------------------------------------------------
# Products (public information) and eligibility
# ---------------------------------------------------------------------------


@app.get("/products", tags=["Products"])
def list_products(category: str | None = Query(default=None, pattern="^(account|deposit|loan)$")) -> Any:
    return services.get_products(kind=category)


@app.get("/products/{product_id}", tags=["Products"])
def get_product(product_id: str) -> Any:
    data = services.get_products(product_id=product_id)
    if not data["count"]:
        raise HTTPException(status_code=404, detail="Product not found.")
    return {"product": data["products"][0], "disclaimer": data["disclaimer"]}


@app.post("/loans/eligibility", tags=["Products"])
def loan_eligibility(body: EligibilityRequest, session: dict = Depends(_require_session)) -> Any:
    if not catalogue.get_product(body.product_id):
        raise HTTPException(status_code=404, detail="Product not found.")
    return _service_call(services.check_loan_eligibility, session["customer_id"], body.product_id,
                         body.loan_amount, body.tenure_months, body.monthly_income, body.existing_emi)


# ---------------------------------------------------------------------------
# Chat (LangGraph agent)
# ---------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(body: ChatRequest, session: dict = Depends(_require_session)) -> Any:
    _check_chat_rate_limit(session["token"])
    result = get_agent().chat(
        session_id=_session_ref(session["token"]),
        customer_id=session["customer_id"],   # from the session, never from the request
        username=session["username"],
        message=body.message,
    )
    return result


@app.delete("/chat", tags=["Chat"])
def reset_chat(session: dict = Depends(_require_session)) -> Any:
    get_agent().reset(_session_ref(session["token"]))
    return {"message": "Conversation cleared."}
