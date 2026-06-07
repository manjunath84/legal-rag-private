#!/usr/bin/env python
"""Deterministic retrieval comparison: vector-only vs BM25 vs hybrid (RRF).

    uv run python scripts/compare_retrieval.py [top_k]   # default top_k=4

Measures **retrieval recall@k** — does the gold chunk (identified by a unique
substring in qa_set.yaml) appear in the top-k retrieved chunks? This is the right
way to compare retrieval methods: it isolates retrieval and involves NO LLM, so
there's no grading/generation nondeterminism to muddy the result.

(End-to-end answer quality, incl. abstention, is measured separately by
eval/run_eval.py — which does involve the model and so can vary run to run.)
"""

import sys
from pathlib import Path

import yaml

from raglab.config import Settings
from raglab.hybrid import bm25_search, hybrid_search
from raglab.retrieve import vector_search

QA_PATH = Path(__file__).resolve().parent.parent / "eval" / "qa_set.yaml"


def _hit(retriever, q: str, gold: str, cfg: Settings, k: int) -> bool:
    return any(gold.lower() in c.text.lower() for c, _ in retriever(q, cfg, k))


def main() -> int:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cases = [c for c in yaml.safe_load(QA_PATH.read_text()) if c.get("gold")]
    cfg_v = Settings(retrieval_mode="vector")
    cfg_h = Settings(retrieval_mode="hybrid")

    print(f"Retrieval recall@{k}  (gold chunk present in top-{k}?)\n")
    print(f"{'VEC':<5}{'BM25':<6}{'HYB':<5}QUESTION")
    print("-" * 84)
    tally = {"vec": 0, "bm25": 0, "hyb": 0}
    for c in cases:
        q, gold = c["question"], c["gold"]
        v = _hit(vector_search, q, gold, cfg_v, k)
        b = _hit(bm25_search, q, gold, cfg_h, k)
        h = _hit(hybrid_search, q, gold, cfg_h, k)
        tally["vec"] += v
        tally["bm25"] += b
        tally["hyb"] += h
        m = lambda x: "✅" if x else "❌"  # noqa: E731
        print(f"{m(v):<5}{m(b):<6}{m(h):<5}{q}")
    print("-" * 84)
    n = len(cases)
    print(f"vector: {tally['vec']}/{n}    bm25: {tally['bm25']}/{n}    hybrid: {tally['hyb']}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
