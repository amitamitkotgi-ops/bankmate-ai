"""
Measure intent accuracy on data/query_eval_set.jsonl.

  python scripts/evaluate_intents.py            # rule-based fallback only
  python scripts/evaluate_intents.py --llm      # also the Groq LLM (needs GROQ_API_KEY)

NOTE: the eval set is generated from templates that resemble the rules, so the
rules score is optimistic. The 'noisy' (typos) and 'hinglish' rows show where
the LLM helps most.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_data import EVAL_SET_PATH  # noqa: E402
from nlp import classify_intent  # noqa: E402
from privacy import check_message_risks  # noqa: E402


def load():
    return [json.loads(l) for l in EVAL_SET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def same(expected, got):
    if expected == "OUT_OF_SCOPE":
        return got in ("OUT_OF_SCOPE", "UNKNOWN")
    return expected == got


def evaluate(predict):
    stats = defaultdict(lambda: [0, 0])
    wrong = []
    for item in load():
        if item["block"]:
            continue
        ok = same(item["intent"], predict(item["text"]))
        for key in ("ALL", item["style"], item["intent"]):
            stats[key][0] += ok
            stats[key][1] += 1
        if not ok:
            wrong.append(item)
    return stats, wrong


def security_report():
    items = [i for i in load() if i["block"]]
    hit = sum(check_message_risks(i["text"], "CUST0001") == i["block"] for i in items)
    return hit, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--show-errors", type=int, default=0)
    args = ap.parse_args()

    predictors = {"rules": classify_intent}
    if args.llm:
        from llm import GroqLLM, LLMUnavailable
        llm = GroqLLM()
        if not llm.available():
            sys.exit("No usable GROQ_API_KEY - cannot run --llm")

        def llm_predict(text):
            try:
                return llm.classify(text, [])["intent"]
            except LLMUnavailable:
                return "ERROR"
        predictors["groq"] = llm_predict

    for name, fn in predictors.items():
        stats, wrong = evaluate(fn)
        print(f"\n=== {name} ===")
        for key in ("ALL", "clean", "noisy", "hinglish"):
            ok, n = stats[key]
            print(f"{key:10s} {ok:4d}/{n:<4d} {100*ok/max(n,1):5.1f}%")
        for e in wrong[: args.show_errors]:
            print("  miss:", e["text"], "->", e["intent"])
    hit, n = security_report()
    print(f"\nSecurity guard: {hit}/{n} attack/credential/other-customer queries blocked before the LLM")


if __name__ == "__main__":
    main()
