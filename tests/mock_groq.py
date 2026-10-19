"""
A tiny fake of the Groq HTTP API, so the REAL langchain-groq client can be
tested end to end without a network or an API key.

Behaviour is controlled through the module-level STATE dict.
"""

from __future__ import annotations

import json
import re
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import nlp

GOOD_KEY = "gsk_test_good_key"

STATE: dict = {}


def reset_state() -> None:
    STATE.clear()
    STATE.update({
        "requests": [],            # every chat request received
        "retired": set(),          # model ids that answer "decommissioned"
        "mode": "ok",              # ok | rate_limit | server_error | slow | empty | garbage | bad_intent | fenced | think
        "reject_json_mode": False,
        "classify_override": None,
        "compose_text": None,
        "live_models": ["whisper-large-v3", "openai/gpt-oss-20b", "meta-llama/llama-guard-4-12b"],
        "sleep": 0.0,
    })


reset_state()
app = FastAPI()


def _completion(model: str, content: str) -> dict:
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


def _error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message,
                                                              "type": "invalid_request_error", "code": code}})


def _classify(user_text: str) -> str:
    m = re.search(r"<user_message>\n(.*?)\n</user_message>", user_text, re.DOTALL)
    parsed = nlp.rule_based_parse(m.group(1) if m else user_text)
    intent = parsed["intent"] if parsed["intent"] != "UNKNOWN" else "OUT_OF_SCOPE"
    e = parsed["entities"]
    entities = {k: e.get(k) for k in ("product_id", "product_kind", "focus", "category", "txn_type",
                                      "period_days", "limit")}
    return json.dumps({"intent": intent, "entities": entities})


def _compose(user_text: str) -> str:
    m = re.search(r"<facts>\n(.*?)\n</facts>", user_text, re.DOTALL)
    facts = json.loads(m.group(1)) if m else {}
    if "total_display" in facts:
        return f"Your total balance is {facts['total_display']}."
    if facts.get("transactions"):
        t = facts["transactions"][0]
        return f"Your latest transaction was {t['description']} for {t['amount_display']}."
    if facts.get("products"):
        p = facts["products"][0]
        return f"{p['name']}: {p['rate_display']}."
    return "Here is the information you asked for."


@app.post("/openai/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    auth = request.headers.get("authorization", "")
    model = body.get("model", "")
    messages = body.get("messages", [])
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    STATE["requests"].append({"model": model, "auth": auth, "system": system, "user": user,
                              "response_format": body.get("response_format"),
                              "reasoning_effort": body.get("reasoning_effort"),
                              "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens")})

    if auth != f"Bearer {GOOD_KEY}":
        return _error(401, "Invalid API Key", "invalid_api_key")
    if model in STATE["retired"]:
        return _error(400, f"The model `{model}` has been decommissioned and is no longer supported. "
                           "Please refer to https://console.groq.com/docs/deprecations for a recommendation "
                           "on which model to use instead.", "model_decommissioned")
    if STATE["reject_json_mode"] and body.get("response_format"):
        return _error(400, "response_format is not supported by this model", "invalid_request_error")

    mode = STATE["mode"]
    if mode == "rate_limit":
        return JSONResponse(status_code=429, headers={"retry-after": "0"},
                            content={"error": {"message": "Rate limit reached", "type": "tokens",
                                               "code": "rate_limit_exceeded"}})
    if mode == "server_error":
        return JSONResponse(status_code=500, content={"error": {"message": "boom", "type": "server_error"}})
    if mode == "slow":
        time.sleep(STATE["sleep"])
    if mode == "empty":
        return _completion(model, "")

    is_classifier = "intent classifier" in system
    if is_classifier:
        if mode == "garbage":
            return _completion(model, "I think the customer wants their balance, probably.")
        if mode == "bad_intent":
            return _completion(model, json.dumps({"intent": "DROP_TABLE_USERS", "entities": {}}))
        payload = STATE["classify_override"] or _classify(user)
        if mode == "fenced":
            payload = f"```json\n{payload}\n```"
        if mode == "think":
            payload = f"<think>The user wants something.</think>{payload}"
        return _completion(model, payload)

    text = STATE["compose_text"] or _compose(user)
    return _completion(model, text() if callable(text) else text)


@app.get("/openai/v1/models")
async def list_models(request: Request):
    if request.headers.get("authorization") != f"Bearer {GOOD_KEY}":
        return _error(401, "Invalid API Key", "invalid_api_key")
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "test"}
                                       for m in STATE["live_models"]]}


class MockGroqServer:
    def __init__(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> "MockGroqServer":
        self._thread.start()
        for _ in range(100):
            if self._server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("mock Groq server did not start")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
