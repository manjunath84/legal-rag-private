# src/raglab/retrieve.py
"""Embed a query and pull the nearest chunks; format them for prompting."""

from raglab import store
from raglab.chunk import Chunk
from raglab.config import Settings, settings
from raglab.embed import embed_query


def cite(chunk: Chunk) -> str:
    """Human-readable citation tag, e.g. 'opinion.pdf, p.3' or 'contract.txt'."""
    return f"{chunk.source}, p.{chunk.page}" if chunk.page is not None else chunk.source


def retrieve(query_text: str, cfg: Settings = settings, k: int | None = None
             ) -> list[tuple[Chunk, float]]:
    vec = embed_query(query_text, cfg.embed_model)
    return store.query(vec, k or cfg.top_k, cfg)


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    """Render retrieved chunks as a numbered context block with citation tags."""
    blocks = []
    for chunk, _score in hits:
        blocks.append(f"[{cite(chunk)}]\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)
