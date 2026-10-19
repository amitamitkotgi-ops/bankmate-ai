# Audit Report — BANKMATE AI (v2 → v3)

Scope: the uploaded `AI_hackathon.zip` (v2, 102 passing tests) was read end to end, checked against the
TCS problem statement, then rebuilt. Every fix below has at least one automated test (see tests/).

## A. Gaps against the problem statement

| # | Finding in v2 | Fix in v3 |
|---|---|---|
| 1 | No LLM, no LangGraph, no Groq — keyword rules only | LangGraph workflow + Groq classifier/composer, rules kept as offline fallback |
| 2 | No **loan eligibility** (named in the problem) | `check_loan_eligibility` with EMI formula, FOIR 50% rule, age/income/tenure/amount checks |
| 3 | Interest rates only for loans, no deposit/account rates | `data/products.json`: 3 accounts, FD, RD, 4 loans |
| 4 | No product JSON file | `data/products.json` (validated on load) |
| 5 | No customer **consent flags** | 3 flags in DB, enforced in code, editable in UI and `PUT /consent/{id}` |
| 6 | No simulated query logs / generation script | `generate_data.py` → 600 masked log lines + labelled eval set |
| 7 | No conversation context | Short memory per session, follow-ups, multi-turn eligibility |
| 8 | Transactions had no categories | 10 categories → category filters and spending breakdown |
| 9 | Data dated 2024, so "last 30 days" returned nothing | Dates generated relative to today, balances = opening + ledger (tested) |

## B. Security and privacy findings in v2

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | High | Public `/customers/list` returned all 50 customer names without login | Endpoint removed; test proves it is gone |
| 2 | High | Phone and e-mail were returned in full by the API (masked only in the UI) | Masked on the server |
| 3 | High | Chat bubbles inserted raw user text into HTML (`unsafe_allow_html`) | Chat uses `st.chat_message` (markdown only); all other dynamic HTML is escaped |
| 4 | Medium | Lockout message (429) only appeared for real usernames → username enumeration | Throttle table keyed by any username; same answer for real and fake |
| 5 | Medium | No password hash for unknown users → timing difference | Dummy hash verified for unknown users |
| 6 | Low | PBKDF2 at 260,000 rounds | 600,000 (OWASP 2023+); count stored in each hash |
| 7 | Low | Label "YTD" for a 12-month total | Label corrected to "last 12 months" |
| 8 | Low | Shipped a stale `.db`, `-wal`, `-shm` and `__pycache__` in the zip | Removed; DB is generated on first start. Old DBs are backed up and rebuilt automatically |

## C. New risks created by adding an LLM — and the controls

| Risk | Control | Test |
|---|---|---|
| PII sent to a third party | Redaction before LLM/memory/logs; name, e-mail, phone, full account never in prompts | `test_pii_never_reaches_groq`, `test_facts_sent_to_llm_contain_no_identity_data` |
| Customer without consent | AI consent OFF → LLM never called | `test_ai_consent_off_means_llm_never_called` |
| Prompt injection | Injection patterns blocked before LLM; prompts mark text as data; LLM output limited to a fixed intent list | `test_blocked_before_llm` |
| LLM picks another customer / tool | `customer_id` only from session; tools chosen by our code | `test_classifier_cannot_pick_another_customer` |
| LLM invents numbers | Money/tenure/income only from regex; every ₹ and % in the answer checked against facts | `test_invented_amount_is_replaced`, `test_hallucinated_amount_is_caught_with_real_client` |
| Data leak via markdown image/link | Links, images, HTML stripped from LLM text | `test_links_images_html_are_stripped` |
| Groq outage / bad key / rate limit | Automatic fallback to rules; circuit breaker; key problems never crash the app | `TestFailures` (10 tests) |
| Retired Groq models | Model list + auto-discovery | `TestModelRetirement` |
| Credential phishing | OTP/PIN/CVV/password messages refused, never forwarded | `test_credentials` |
| Over-logging | Audit log never stores message text; query log only with consent and only redacted text | `test_audit_never_contains_message_text` |

## D. Bugs found by my own testing while building v3 (all fixed)

1. "Hey show my profile" was treated as a plain greeting → greeting now only wins when nothing else was asked.
2. "show my grocery debits from the last 90 days" was not understood by the rules → plain debits/credits/payments patterns added.
3. A leftover `or True` condition in eligibility code → removed.
4. The first draft assumed old Groq model names would work → replaced after checking Groq's deprecation page.

## E. Not covered / not verified

- **Real Groq calls were not made** (no key or internet in the build environment). Run `python scripts/check_groq.py`.
- No load test; sessions and memory are in-process (single server only).
- No HTTPS/cookie hardening, no real bank integrations, no real credit scoring.
- Synthetic evaluation set is template-based, so its accuracy numbers are optimistic.
