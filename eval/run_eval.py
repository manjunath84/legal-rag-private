#!/usr/bin/env python
"""Run the eval/demo set through CRAG and report pass/fail.

    uv run python eval/run_eval.py

CRAG is the headline pattern: it must answer the answerable questions (expected
substring present) AND abstain on the unanswerable ones. This doubles as the demo
script and is the bridge to Week 5 (Evals & Observability).
"""

from pathlib import Path

import yaml

from raglab.crag import ABSTAIN, answer_crag

QA_PATH = Path(__file__).parent / "qa_set.yaml"


def _abstained(text: str) -> bool:
    t = text.lower()
    return "i don't know" in t or "not in the provided" in t or text.strip() == ABSTAIN


def main() -> int:
    cases = yaml.safe_load(QA_PATH.read_text())
    passed = 0
    print(f"{'OK':<4}{'KIND':<14}QUESTION")
    print("-" * 80)
    for case in cases:
        res = answer_crag(case["question"])
        if case["answerable"]:
            ok = any(e.lower() in res.answer.lower() for e in case["expect"])
            kind = "answerable"
        else:
            ok = _abstained(res.answer)
            kind = "unanswerable"
        passed += ok
        mark = "✅" if ok else "❌"
        print(f"{mark:<4}{kind:<14}{case['question']}")
        if not ok:
            print(f"      ↳ got: {res.answer[:120]}")
    print("-" * 80)
    total = len(cases)
    print(f"{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
