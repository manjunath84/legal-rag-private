# src/raglab/retrieve.py
"""Retrieve the nearest chunks and format them for prompting.

`retrieve()` dispatches on `settings.retrieval_mode`:
  - "vector"  — dense embeddings only (Stage 1 behaviour)
  - "hybrid"  — BM25 + dense fused with Reciprocal Rank Fusion (Stage 2, default)

Callers (rag_simple, memory, crag) use `retrieve()` and get whichever mode is
configured, so the whole app benefits from hybrid without per-caller changes.
"""

from raglab import store
from raglab.chunk import Chunk
from raglab.config import Settings, settings
from raglab.embed import embed_query


def cite(chunk: Chunk) -> str:
    """Human-readable citation tag, e.g. 'opinion.pdf, p.3' or 'contract.txt'."""
    return f"{chunk.source}, p.{chunk.page}" if chunk.page is not None else chunk.source


def vector_search(query_text: str, cfg: Settings = settings, k: int | None = None
                  ) -> list[tuple[Chunk, float]]:
    """Dense retrieval: embed the query and pull the nearest chunks from Chroma."""
    vec = embed_query(query_text, cfg.embed_model)
    return store.query(vec, k or cfg.top_k, cfg)


def retrieve(query_text: str, cfg: Settings = settings, k: int | None = None
             ) -> list[tuple[Chunk, float]]:
    """Mode-aware retrieval with optional cross-encoder reranking.

    When rerank_enabled, fetches candidate_k candidates first, then the
    cross-encoder re-sorts them and returns the top effective_k.
    """
    effective_k = k or cfg.top_k
    fetch_k = cfg.candidate_k if cfg.rerank_enabled else effective_k

    if cfg.retrieval_mode == "hybrid":
        from raglab.hybrid import hybrid_search
        hits = hybrid_search(query_text, cfg, fetch_k)
    else:
        hits = vector_search(query_text, cfg, fetch_k)

    if cfg.rerank_enabled:
        from raglab.rerank import rerank
        hits = rerank(query_text, hits, effective_k)

    return hits


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    """Render retrieved chunks as a context block with citation tags."""
    blocks = []
    for chunk, _score in hits:
        blocks.append(f"[{cite(chunk)}]\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)
