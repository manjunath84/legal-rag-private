# src/raglab/store.py
"""Chroma persistent vector store wrapper.

We supply our own embeddings (from embed.py) rather than letting Chroma pick a
default model — that keeps the embedding choice explicit and local. The store
holds chunk text + metadata (source, page) so retrieval can reconstruct Chunks
and cite them.
"""

import chromadb

from raglab.chunk import Chunk
from raglab.config import Settings, settings


def get_collection(cfg: Settings = settings):
    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    # cosine space matches our normalized embeddings.
    return client.get_or_create_collection(
        name=cfg.collection, metadata={"hnsw:space": "cosine"}
    )


def add_chunks(
    chunks: list[Chunk], embeddings: list[list[float]], cfg: Settings = settings
) -> None:
    if not chunks:
        return
    col = get_collection(cfg)
    col.upsert(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "page": c.page if c.page is not None else -1}
                   for c in chunks],
    )


def count(cfg: Settings = settings) -> int:
    return get_collection(cfg).count()


def reset(cfg: Settings = settings) -> None:
    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    try:
        client.delete_collection(cfg.collection)
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass


def query(embedding: list[float], k: int, cfg: Settings = settings) -> list[tuple[Chunk, float]]:
    """Return up to k (Chunk, distance) pairs, nearest first (lower distance = closer)."""
    col = get_collection(cfg)
    res = col.query(query_embeddings=[embedding], n_results=k)
    out: list[tuple[Chunk, float]] = []
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    for cid, text, meta, dist in zip(ids, docs, metas, dists, strict=True):
        page = meta.get("page", -1)
        chunk = Chunk(
            id=cid, text=text, source=meta.get("source", "?"),
            page=None if page == -1 else int(page),
        )
        out.append((chunk, float(dist)))
    return out
