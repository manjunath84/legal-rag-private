# src/raglab/chunk.py
"""Character-window chunking with overlap.

Deliberately simple and transparent (Stage 1). Each chunk keeps its source +
page so retrieval can cite precisely. Splitting prefers paragraph/sentence
boundaries near the window edge to avoid cutting mid-sentence.
"""

import hashlib
from dataclasses import dataclass

from raglab.ingest import Document


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None


def _split_one(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Back up to the nearest boundary (paragraph > newline > space) within
            # the last 20% of the window, so we cut cleanly rather than mid-word.
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
            if cut > size * 0.8:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_documents(docs: list[Document], size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        for piece in _split_one(doc.text, size, overlap):
            cid = hashlib.sha1(
                f"{doc.source}|{doc.page}|{piece[:64]}|{len(out)}".encode()
            ).hexdigest()[:16]
            out.append(Chunk(id=cid, text=piece, source=doc.source, page=doc.page))
    return out
