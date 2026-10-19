"""
Check that YOUR Groq key and model work. Run this once after pasting the key into .env:

    python scripts/check_groq.py

It never prints the key. It tests: key accepted -> which models exist ->
one classification -> one answer, using the same code the app uses.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from llm import GroqLLM, LLMUnavailable  # noqa: E402


def main() -> int:
    key = config.get_groq_api_key()
    if not key:
        print("X  No Groq key found. Open the file  .env  and set:  GROQ_API_KEY=your_key_here")
        return 1
    print(f"OK Key found in .env (length {len(key)}, not shown)")

    llm = GroqLLM()
    try:
        live = llm.discover_live_models()
        print(f"OK Groq accepted the key. {len(live)} chat models are available now.")
        print("   Configured order:", ", ".join(llm.models))
        missing = [m for m in llm.models if m not in live]
        if missing:
            print("!  Not in Groq's live list (will be skipped):", ", ".join(missing))
    except Exception as exc:  # noqa: BLE001
        print(f"X  Could not list models: {type(exc).__name__}. Check the key and your internet.")
        return 1

    try:
        out = llm.classify("show my last 3 transactions", [])
        print(f"OK Classifier works  -> intent={out['intent']} entities={out['entities']} model={out['model']}")
        facts = {"total_display": "₹1,23,456.00", "account_count": 1}
        ans = llm.compose("what is my balance", "BALANCE", facts, [])
        print(f"OK Composer works    -> {ans['text'][:160]!r}")
    except LLMUnavailable as exc:
        print(f"X  LLM call failed: {exc.reason}")
        return 1
    print("\nAll good. Start the app; the sidebar will show 'Groq LLM ready'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
