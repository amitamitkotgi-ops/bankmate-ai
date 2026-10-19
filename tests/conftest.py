"""Shared pytest setup. Uses a temporary database; never touches data/bankmate.db."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

# Never let a real key in .env leak into tests.
os.environ["GROQ_API_KEY"] = ""

import database as db  # noqa: E402

db._ITERATIONS = 1000  # fast password hashing for tests only

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
db.DB_PATH = _tmp.name

import api  # noqa: E402
from agent import BankingAgent  # noqa: E402
from llm import LLMUnavailable  # noqa: E402

DEMO = {
    "aarav": ("demo_aarav", "Demo@1234", "CUST0001"),
    "priya": ("demo_priya", "Demo@5678", "CUST0002"),
    "rohan": ("demo_rohan", "Demo@9012", "CUST0003"),
    "sneha": ("demo_sneha", "Demo@3456", "CUST0004"),
    "kiran": ("demo_kiran", "Demo@7890", "CUST0005"),
}


class FakeLLM:
    """Test double with the same interface as llm.GroqLLM. Records everything it is shown."""

    def __init__(self, intent=None, entities=None, compose_text=None, fail_classify=None, fail_compose=None):
        self.intent = intent
        self.entities = entities or {}
        self.compose_text = compose_text
        self.fail_classify = fail_classify
        self.fail_compose = fail_compose
        self.seen_classify: list[str] = []
        self.seen_compose: list[dict] = []

    def available(self):
        return True

    def status(self):
        return {"configured": True, "available": True, "model": "fake-model", "reason": None}

    def classify(self, message, history):
        self.seen_classify.append(message)
        if self.fail_classify:
            raise LLMUnavailable(self.fail_classify)
        from nlp import classify_intent
        return {"intent": self.intent or classify_intent(message), "entities": self.entities, "model": "fake-model"}

    def compose(self, question, intent, facts, history):
        self.seen_compose.append({"question": question, "intent": intent, "facts": facts})
        if self.fail_compose:
            raise LLMUnavailable(self.fail_compose)
        if callable(self.compose_text):
            return {"text": self.compose_text(facts), "model": "fake-model"}
        return {"text": self.compose_text or "Here is your answer.", "model": "fake-model"}


@pytest.fixture(scope="session", autouse=True)
def _database():
    db.initialize_database()
    yield
    try:
        os.unlink(_tmp.name)
    except OSError:
        pass


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from llm import GroqLLM

    api.set_agent(BankingAgent(llm=GroqLLM(api_key="")))  # rules mode: deterministic
    with TestClient(api.app) as c:
        yield c
    api.set_agent(None)


@pytest.fixture()
def login(client):
    def _login(who: str = "aarav"):
        username, password, cid = DEMO[who]
        r = client.post("/auth/login", json={"username": username, "password": password})
        assert r.status_code == 200, r.text
        return {"X-Session-Token": r.json()["session_token"]}, cid

    return _login


@pytest.fixture()
def restore_consent():
    """Snapshot and restore consent flags of all demo customers after a test."""
    before = {cid: db.get_consent(cid) for *_, cid in DEMO.values()}
    yield
    for cid, c in before.items():
        db.update_consent(cid, c["ai_processing"], c["query_logging"], c["marketing"])
