# src/raglab/rerank.py
"""Cross-encoder reranker: re-score retrieved candidates as (query, chunk) pairs.

Why: BM25 + vector retrieval scores each text independently, so two chunks with
near-identical keywords (NDA governing-law vs MSA governing-law) score almost the
same. A cross-encoder reads the full (question, chunk) pair and can distinguish
"this NDA question is answered by the NDA governing-law chunk, not the MSA one."

Usage: call rerank() after retrieve() to re-sort the candidate pool before
returning the final top-k to the generator.

The model is loaded once and cached (lazy, first call only). CrossEncoder ships
inside sentence-transformers — no additional dependency needed.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from raglab.chunk import Chunk

_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_encoder: CrossEncoder | None = None


def _get_encoder() -> CrossEncoder:
    global _encoder
    if _encoder is None:
        _encoder = CrossEncoder(_MODEL)
    return _encoder


def rerank(
    query: str,
    hits: list[tuple[Chunk, float]],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    """Re-score candidates as (query, chunk) pairs; return top_k best.

    Input scores (RRF or cosine) are replaced by cross-encoder logits.
    Logits may be negative; higher is still better.
    """
    if not hits:
        return hits
    encoder = _get_encoder()
    pairs = [(query, chunk.text) for chunk, _ in hits]
    scores: list[float] = encoder.predict(pairs).tolist()
    ranked = sorted(zip(hits, scores, strict=True), key=lambda x: x[1], reverse=True)
    return [(chunk, float(score)) for (chunk, _orig), score in ranked[:top_k]]
