# src/raglab/hybrid.py
"""Hybrid retrieval: BM25 keyword search + dense vectors, fused with RRF.

Why: dense embeddings miss exact legal terms — Stage 1 measured the NDA "term"
clause ranking #5 because "term" doesn't embed strongly. BM25 nails such keyword
matches (section names, party names, numbers); dense nails paraphrases. Fusing
both with Reciprocal Rank Fusion gives the best of each without tuning score
scales.

RRF score for a chunk = sum over retrievers of 1 / (rrf_k + rank). A chunk that
ranks well in BOTH lists wins; one that ranks well in only one still scores.
The BM25 index is built from Chroma's stored chunks (Chroma stays the single
source of truth) and cached per process, keyed by collection size.
"""

import re

from rank_bm25 import BM25Okapi

from raglab import store
from raglab.chunk import Chunk
from raglab.config import Settings, settings
from raglab.retrieve import vector_search

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# Cache: collection name -> (chunk_count, chunks, BM25 index). Rebuilt when the
# collection size changes (e.g. after re-ingest).
_bm25_cache: dict[str, tuple[int, list[Chunk], BM25Okapi]] = {}


def _get_bm25(cfg: Settings) -> tuple[list[Chunk], BM25Okapi]:
    n = store.count(cfg)
    cached = _bm25_cache.get(cfg.collection)
    if cached and cached[0] == n:
        return cached[1], cached[2]
    chunks = store.all_chunks(cfg)
    bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
    _bm25_cache[cfg.collection] = (n, chunks, bm25)
    return chunks, bm25


def bm25_search(query_text: str, cfg: Settings = settings, k: int | None = None
                ) -> list[tuple[Chunk, float]]:
    """Keyword retrieval. Returns up to k (Chunk, bm25_score), best first."""
    chunks, bm25 = _get_bm25(cfg)
    if not chunks:
        return []
    scores = bm25.get_scores(_tokenize(query_text))
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda cs: cs[1], reverse=True)
    return ranked[: (k or cfg.candidate_k)]


def reciprocal_rank_fusion(rankings: list[list[Chunk]], top: int, rrf_k: int
                           ) -> list[tuple[Chunk, float]]:
    """Fuse several ranked chunk lists into one. Pure function (no I/O), so it's
    unit-testable. RRF score(chunk) = sum over lists of 1 / (rrf_k + rank)."""
    by_id: dict[str, Chunk] = {}
    rrf: dict[str, float] = {}
    for ranked in rankings:
        for rank, chunk in enumerate(ranked, start=1):
            by_id[chunk.id] = chunk
            rrf[chunk.id] = rrf.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
    return [(by_id[cid], score) for cid, score in ordered[:top]]


def hybrid_search(query_text: str, cfg: Settings = settings, k: int | None = None
                  ) -> list[tuple[Chunk, float]]:
    """Fuse dense + BM25 rankings with Reciprocal Rank Fusion.

    Returns up to k (Chunk, rrf_score) pairs, best first. The returned score is
    the RRF score (higher = better), not a distance.
    """
    dense = [c for c, _ in vector_search(query_text, cfg, cfg.candidate_k)]
    sparse = [c for c, _ in bm25_search(query_text, cfg, cfg.candidate_k)]
    return reciprocal_rank_fusion([dense, sparse], k or cfg.top_k, cfg.rrf_k)
