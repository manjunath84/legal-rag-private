#!/usr/bin/env python
"""Deterministic retrieval comparison: vector / BM25 / hybrid / rerank.

    uv run python scripts/compare_retrieval.py [top_k]   # default top_k=4

Measures retrieval recall@k — does the gold chunk (identified by a unique
substring in qa_set.yaml) appear in the top-k retrieved chunks?

  VEC   — dense embeddings only
  BM25  — keyword search only
  HYB   — BM25 + dense fused with RRF (hybrid)
  RERANK — hybrid candidates re-scored by cross-encoder

The RERANK column reflects the full Stage 2b pipeline and is the number to
report in portfolio writeups. The three preceding columns are the A/B baseline.

Tip: run at k=1 to see ranking quality (not just recall):
    uv run python scripts/compare_retrieval.py 1
"""

import sys
from pathlib import Path

import yaml

from raglab.config import Settings
from raglab.hybrid import bm25_search, hybrid_search
from raglab.retrieve import retrieve, vector_search

QA_PATH = Path(__file__).resolve().parent.parent / "eval" / "qa_set.yaml"


def _normalise(s: str) -> str:
    """Collapse all whitespace to a single space for substring matching."""
    return " ".join(s.lower().split())


def _hit(retriever, q: str, gold: str, cfg: Settings, k: int) -> bool:
    gold_n = _normalise(gold)
    return any(gold_n in _normalise(c.text) for c, _ in retriever(q, cfg, k))


def _hit_rerank(q: str, gold: str, cfg: Settings, k: int) -> bool:
    gold_n = _normalise(gold)
    return any(gold_n in _normalise(c.text) for c, _ in retrieve(q, cfg, k))


def main() -> int:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cases = [c for c in yaml.safe_load(QA_PATH.read_text()) if c.get("gold")]

    cfg_v = Settings(retrieval_mode="vector", rerank_enabled=False)
    cfg_h = Settings(retrieval_mode="hybrid", rerank_enabled=False)
    cfg_r = Settings(retrieval_mode="hybrid", rerank_enabled=True)

    print(f"Retrieval recall@{k}  (gold chunk present in top-{k}?)\n")
    print(f"{'VEC':<5}{'BM25':<6}{'HYB':<6}{'RERANK':<8}QUESTION")
    print("-" * 92)
    tally = {"vec": 0, "bm25": 0, "hyb": 0, "rnk": 0}
    for c in cases:
        q, gold = c["question"], c["gold"]
        v = _hit(vector_search, q, gold, cfg_v, k)
        b = _hit(bm25_search, q, gold, cfg_h, k)
        h = _hit(hybrid_search, q, gold, cfg_h, k)
        r = _hit_rerank(q, gold, cfg_r, k)
        tally["vec"] += v
        tally["bm25"] += b
        tally["hyb"] += h
        tally["rnk"] += r
        m = lambda x: "✅" if x else "❌"  # noqa: E731
        print(f"{m(v):<5}{m(b):<6}{m(h):<6}{m(r):<8}{q}")
    print("-" * 92)
    n = len(cases)
    print(f"vector: {tally['vec']}/{n}   bm25: {tally['bm25']}/{n}   "
          f"hybrid: {tally['hyb']}/{n}   rerank: {tally['rnk']}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
