"""
BANKMATE AI — agent.py
The conversational brain, built with LangGraph.

    START
      │
  preprocess ── blocked? ──► refuse ─────────────────────────┐
      │ (clean + redact PII + security checks)               │
  classify  (Groq LLM  ─or─  rules fallback)                 │
      │                                                      │
      ├─► tool_balance ─────┐                                │
      ├─► tool_transactions ┤                                │
      ├─► tool_spending ────┤                                │
      ├─► tool_products ────┼─► compose (LLM or template)    │
      ├─► tool_eligibility ─┤        │                       │
      ├─► tool_profile ─────┤        ▼                       │
      ├─► tool_privacy ─────┘     guardrail ──────────────┐  │
      └─► static (help/greeting/fallback) ───────────────►│  │
                                                          ▼  ▼
                                                       finalize ─► END

Security design
  * customer_id is set by the API from the login session. The LLM never sees
    it and cannot choose it.
  * The LLM never calls tools. It only returns an intent (validated against a
    fixed list); our code decides which tool runs.
  * Only PII-redacted text and masked facts ever go to the LLM, and only when
    the customer's AI-processing consent flag is ON.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import database as db
import products as catalogue
import responses
import services
from config import CHAT_MAX_CHARS
from entities import extract_entities, keep_relevant, sanitize_llm_entities
from guardrails import sanitize_llm_text, ungrounded_numbers
from llm import GroqLLM, LLMUnavailable
from nlp import (
    BALANCE, GREETING, HELP, LOAN_ELIGIBILITY, OUT_OF_SCOPE, PRIVACY, PRODUCT_INFO, PROFILE,
    SPENDING_SUMMARY, TRANSACTIONS, UNKNOWN, rule_based_parse,
)
from privacy import (
    BLOCK_EMPTY, BLOCK_TOO_LONG, check_message_risks, clean_input, contains_unmasked_pii, redact_pii,
)

logger = logging.getLogger(__name__)

COMPOSE_INTENTS = {BALANCE, TRANSACTIONS, SPENDING_SUMMARY, PRODUCT_INFO, LOAN_ELIGIBILITY}
DATA_INTENTS = {BALANCE, TRANSACTIONS, SPENDING_SUMMARY, PRODUCT_INFO, LOAN_ELIGIBILITY, PROFILE, PRIVACY}
PRODUCT_FOOTER_INTENTS = {PRODUCT_INFO, LOAN_ELIGIBILITY}
CONTEXT_INTENTS = {BALANCE, TRANSACTIONS, SPENDING_SUMMARY, PRODUCT_INFO, LOAN_ELIGIBILITY}

MAX_HISTORY_ITEMS = 12
MAX_SESSIONS_IN_MEMORY = 500

INTENT_TO_NODE = {
    BALANCE: "tool_balance",
    TRANSACTIONS: "tool_transactions",
    SPENDING_SUMMARY: "tool_spending",
    PRODUCT_INFO: "tool_products",
    LOAN_ELIGIBILITY: "tool_eligibility",
    PROFILE: "tool_profile",
    PRIVACY: "tool_privacy",
}


class AgentState(TypedDict, total=False):
    # ---- inputs (set by the API for every turn; customer_id comes from the session) ----
    customer_id: str
    username: str
    raw_message: str
    consent: dict
    history: list
    context: dict
    # ---- per-turn working values ----
    clean_message: str
    pii_found: list
    block_reason: str | None
    intent: str
    entities: dict
    classifier: str
    followup: bool
    tool: str | None
    facts: dict
    template_reply: str
    reply: str
    llm_composed: bool
    model: str | None
    fallback_reason: str | None
    guardrail_flags: list
    mode: str


class BankingAgent:
    def __init__(self, llm: GroqLLM | None = None):
        self.llm = llm if llm is not None else GroqLLM()
        self._memory: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ memory

    def _load_memory(self, session_id: str) -> dict:
        with self._lock:
            mem = self._memory.get(session_id)
            if mem is None:
                mem = {"history": [], "context": {}}
            self._memory[session_id] = mem
            self._memory.move_to_end(session_id)
            while len(self._memory) > MAX_SESSIONS_IN_MEMORY:
                self._memory.popitem(last=False)
            return {"history": list(mem["history"]), "context": dict(mem["context"])}

    def _save_memory(self, session_id: str, history: list, context: dict) -> None:
        with self._lock:
            self._memory[session_id] = {"history": history[-MAX_HISTORY_ITEMS:], "context": context}

    def reset(self, session_id: str) -> None:
        """Forget a conversation (called on logout and on 'Reset chat')."""
        with self._lock:
            self._memory.pop(session_id, None)

    # --------------------------------------------------------------------- run

    def chat(self, session_id: str, customer_id: str, username: str, message: str) -> dict:
        consent = db.get_consent(customer_id) or {"ai_processing": False, "query_logging": False,
                                                  "marketing": False}
        mem = self._load_memory(session_id)
        state = self.graph.invoke({
            "customer_id": customer_id,
            "username": username,
            "raw_message": message or "",
            "consent": consent,
            "history": mem["history"],
            "context": mem["context"],
        })
        self._save_memory(session_id, state.get("history", []), state.get("context", {}))
        return {
            "reply": state["reply"],
            "intent": state.get("intent"),
            "tool": state.get("tool"),
            "mode": state.get("mode", "rules"),
            "llm_used": state.get("classifier") == "llm" or bool(state.get("llm_composed")),
            "model": state.get("model"),
            "classifier": state.get("classifier"),
            "entities": state.get("entities", {}),
            "blocked": state.get("block_reason"),
            "pii_redacted": state.get("pii_found", []),
            "guardrail_flags": state.get("guardrail_flags", []),
            "fallback_reason": state.get("fallback_reason"),
            "followup": bool(state.get("followup")),
        }

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    # ------------------------------------------------------------------- nodes

    def _n_preprocess(self, state: AgentState) -> dict:
        cleaned = clean_input(state.get("raw_message", ""))
        block = None
        if not cleaned:
            block = BLOCK_EMPTY
        elif len(cleaned) > CHAT_MAX_CHARS:
            block = BLOCK_TOO_LONG
        else:
            block = check_message_risks(cleaned, state.get("customer_id"))
        clean_message, pii = redact_pii(cleaned[:CHAT_MAX_CHARS])
        return {
            "clean_message": clean_message,
            "pii_found": pii,
            "block_reason": block,
            "intent": UNKNOWN, "entities": {}, "classifier": "rules", "followup": False,
            "tool": None, "facts": {}, "template_reply": "", "reply": "", "llm_composed": False,
            "model": None, "fallback_reason": None, "guardrail_flags": [], "mode": "rules",
        }

    @staticmethod
    def _route_after_preprocess(state: AgentState) -> str:
        return "refuse" if state.get("block_reason") else "classify"

    def _n_refuse(self, state: AgentState) -> dict:
        reason = state["block_reason"]
        return {"reply": responses.BLOCK_TEXT.get(reason, responses.UNKNOWN_TEXT),
                "intent": "BLOCKED", "classifier": "guard", "mode": "guard"}

    def _n_classify(self, state: AgentState) -> dict:
        message = state["clean_message"]
        consent = state.get("consent", {})
        history = state.get("history", [])
        context = state.get("context", {})

        regex_entities = extract_entities(message)
        classifier, model, fallback = "rules", None, None
        llm_entities: dict = {}

        if not consent.get("ai_processing"):
            fallback = "consent_off"
            intent = rule_based_parse(message)["intent"]
        elif not self.llm.available():
            fallback = self.llm.status().get("reason") or "llm_unavailable"
            intent = rule_based_parse(message)["intent"]
        else:
            try:
                result = self.llm.classify(message, history)
                intent, model = result["intent"], result.get("model")
                llm_entities = sanitize_llm_entities(result.get("entities"))  # defence in depth
                classifier = "llm"
            except LLMUnavailable as exc:
                fallback = exc.reason
                intent = rule_based_parse(message)["intent"]

        # Deterministic values win; the LLM may only fill gaps with validated non-numeric values.
        merged = dict(regex_entities)
        for key, value in llm_entities.items():
            merged.setdefault(key, value)

        # Follow-ups: "and for a home loan?" / "3 years" after an eligibility question.
        followup = False
        ctx_intent = context.get("intent")
        ctx_entities = context.get("entities", {})
        if context.get("awaiting") and intent in (UNKNOWN, OUT_OF_SCOPE, PRODUCT_INFO) \
                and keep_relevant(LOAN_ELIGIBILITY, merged):
            intent, followup = LOAN_ELIGIBILITY, True  # answering our "which loan type?" question
        elif ctx_intent and intent in (UNKNOWN, OUT_OF_SCOPE) and keep_relevant(ctx_intent, merged):
            intent, followup = ctx_intent, True
        if ctx_intent == LOAN_ELIGIBILITY and intent == LOAN_ELIGIBILITY:
            followup = followup or bool(ctx_entities)
            entities = {**ctx_entities, **keep_relevant(intent, merged)}
        elif followup:
            entities = {**ctx_entities, **keep_relevant(intent, merged)}
        else:
            entities = keep_relevant(intent, merged)

        return {"intent": intent, "entities": entities, "classifier": classifier,
                "model": model, "fallback_reason": fallback, "followup": followup}

    @staticmethod
    def _route_intent(state: AgentState) -> str:
        return INTENT_TO_NODE.get(state["intent"], "static")

    # -- tools (each wraps a services.* call; customer_id is from the session) ----

    def _run_tool(self, state: AgentState, tool: str, fn) -> dict:
        try:
            facts = fn()
        except services.ServiceError as exc:
            facts = {"error": exc.message}
        except Exception:  # never leak internals to the customer
            logger.exception("Tool %s failed", tool)
            facts = {"error": "Sorry, I couldn't fetch that right now. Please try again."}
        return {"tool": tool, "facts": facts}

    def _n_balance(self, state: AgentState) -> dict:
        return self._run_tool(state, "get_balance", lambda: services.get_balance(state["customer_id"]))

    def _n_transactions(self, state: AgentState) -> dict:
        e = state["entities"]
        return self._run_tool(state, "get_transactions", lambda: services.get_transactions(
            state["customer_id"], limit=e.get("limit", 5), period_days=e.get("period_days"),
            category=e.get("category"), txn_type=e.get("txn_type"), period_label=e.get("period_label")))

    def _n_spending(self, state: AgentState) -> dict:
        e = state["entities"]
        return self._run_tool(state, "get_spending_summary", lambda: services.get_spending_summary(
            state["customer_id"], period_days=e.get("period_days", 30), category=e.get("category"),
            period_label=e.get("period_label") if e.get("period_days") else "last 30 days"))

    def _n_products(self, state: AgentState) -> dict:
        e = state["entities"]
        return self._run_tool(state, "get_products", lambda: services.get_products(
            product_id=e.get("product_id"), kind=e.get("product_kind")))

    def _n_eligibility(self, state: AgentState) -> dict:
        e = state["entities"]
        return self._run_tool(state, "check_loan_eligibility", lambda: services.check_loan_eligibility(
            state["customer_id"], product_id=e.get("product_id"), loan_amount=e.get("loan_amount"),
            tenure_months=e.get("tenure_months"), monthly_income=e.get("monthly_income"),
            existing_emi=e.get("existing_emi")))

    def _n_profile(self, state: AgentState) -> dict:
        return self._run_tool(state, "get_profile", lambda: services.get_profile(state["customer_id"]))

    def _n_privacy(self, state: AgentState) -> dict:
        return self._run_tool(state, "get_privacy_info", lambda: services.get_privacy_info(state["customer_id"]))

    def _n_static(self, state: AgentState) -> dict:
        intent = state["intent"]
        return {"reply": responses.render(intent, {}, state.get("entities")), "tool": None}

    # -- compose -----------------------------------------------------------------

    def _n_compose(self, state: AgentState) -> dict:
        intent, facts = state["intent"], state["facts"]
        if "error" in facts:
            return {"reply": facts["error"], "template_reply": facts["error"]}

        template = responses.render(intent, facts, state.get("entities"))
        update: dict = {"template_reply": template, "reply": template}

        consent = state.get("consent", {})
        if consent.get("ai_processing") and intent in COMPOSE_INTENTS:
            if not self.llm.available():
                update["fallback_reason"] = state.get("fallback_reason") or self.llm.status().get("reason")
                return update
            llm_facts = {k: v for k, v in facts.items() if k != "disclaimer"}
            try:
                result = self.llm.compose(state["clean_message"], intent, llm_facts, state.get("history", []))
                update.update({"reply": result["text"], "llm_composed": True,
                               "model": result.get("model") or state.get("model")})
            except LLMUnavailable as exc:
                update["fallback_reason"] = exc.reason
        return update

    # -- guardrail ---------------------------------------------------------------

    def _n_guardrail(self, state: AgentState) -> dict:
        intent, facts = state["intent"], state.get("facts", {})
        reply, flags = state["reply"], list(state.get("guardrail_flags", []))
        llm_composed = bool(state.get("llm_composed"))

        if llm_composed:
            cleaned = sanitize_llm_text(reply)
            if cleaned != reply.strip():
                flags.append("sanitized_output")
            bad_numbers = ungrounded_numbers(cleaned, facts)
            if bad_numbers:
                flags.append("ungrounded_numbers")
            if contains_unmasked_pii(cleaned):
                flags.append("pii_in_output")
            if bad_numbers or "pii_in_output" in flags or len(cleaned) < 5:
                reply, llm_composed = state.get("template_reply") or reply, False
                logger.warning("Guardrail replaced an LLM answer (flags=%s)", flags)
            else:
                reply = cleaned

        has_error = "error" in facts
        extras: list[str] = []
        consent = state.get("consent", {})
        if (intent == BALANCE and not has_error and consent.get("marketing")
                and facts.get("total_balance", 0) >= 100_000):
            fd = catalogue.get_product("fixed_deposit")
            if fd:
                fd_display = catalogue.with_display(fd)["rate_display"]
                extras.append(f"💡 Tip: you have a healthy balance. A Fixed Deposit currently offers "
                              f"{fd_display}. Ask me for details.")
        if intent in PRODUCT_FOOTER_INTENTS and not has_error:
            extras.append(responses.PRODUCT_FOOTER)
        if intent in DATA_INTENTS and not has_error:
            extras.append(responses.DATA_FOOTER)
        if extras:
            reply = reply.rstrip() + "\n\n" + "\n\n".join(extras)

        classifier = state.get("classifier")
        return {"reply": reply, "llm_composed": llm_composed, "guardrail_flags": flags,
                "mode": "llm" if (classifier == "llm" or llm_composed) else "rules"}

    # -- finalize ----------------------------------------------------------------

    def _n_finalize(self, state: AgentState) -> dict:
        intent = state["intent"]
        history = list(state.get("history", []))
        history.append({"role": "user", "text": state["clean_message"]})
        history.append({"role": "assistant", "text": state["reply"][:400]})

        context = dict(state.get("context", {}))
        if not state.get("block_reason") and intent in CONTEXT_INTENTS and "error" not in state.get("facts", {}):
            context = {"intent": intent, "entities": state.get("entities", {}),
                       "awaiting": intent == LOAN_ELIGIBILITY
                       and state.get("facts", {}).get("status") == "needs_info"}

        llm_used = state.get("classifier") == "llm" or bool(state.get("llm_composed"))
        try:
            services.record_chat_event(
                customer_id=state["customer_id"], username=state.get("username"),
                masked_query=state["clean_message"], intent=intent, tool=state.get("tool"),
                mode=state.get("mode", "rules"), llm_used=llm_used, blocked=state.get("block_reason"),
                pii_types=state.get("pii_found", []),
            )
        except Exception:
            logger.exception("Could not record chat event")
        return {"history": history[-MAX_HISTORY_ITEMS:], "context": context}

    # ------------------------------------------------------------------- graph

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("preprocess", self._n_preprocess)
        g.add_node("refuse", self._n_refuse)
        g.add_node("classify", self._n_classify)
        g.add_node("tool_balance", self._n_balance)
        g.add_node("tool_transactions", self._n_transactions)
        g.add_node("tool_spending", self._n_spending)
        g.add_node("tool_products", self._n_products)
        g.add_node("tool_eligibility", self._n_eligibility)
        g.add_node("tool_profile", self._n_profile)
        g.add_node("tool_privacy", self._n_privacy)
        g.add_node("static", self._n_static)
        g.add_node("compose", self._n_compose)
        g.add_node("guardrail", self._n_guardrail)
        g.add_node("finalize", self._n_finalize)

        g.add_edge(START, "preprocess")
        g.add_conditional_edges("preprocess", self._route_after_preprocess,
                                {"refuse": "refuse", "classify": "classify"})
        g.add_edge("refuse", "finalize")
        g.add_conditional_edges(
            "classify", self._route_intent,
            {**{node: node for node in INTENT_TO_NODE.values()}, "static": "static"},
        )
        for node in INTENT_TO_NODE.values():
            g.add_edge(node, "compose")
        g.add_edge("static", "guardrail")
        g.add_edge("compose", "guardrail")
        g.add_edge("guardrail", "finalize")
        g.add_edge("finalize", END)
        return g.compile()


__all__ = ["BankingAgent", "AgentState"]
