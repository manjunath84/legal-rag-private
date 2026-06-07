# src/raglab/chunk.py
"""Chunking strategies for ingested documents.

Two strategies (controlled by config.chunk_strategy):

  "section"  (default) — detect section boundaries first (numbered headings,
             markdown headers), keep each section whole if it fits within
             `size`; fall back to char-window inside oversized sections.
             Fixes the clause-fragmentation bug where char-window chopping
             split the NDA governing-law clause across two chunks.

  "char"     — original character-window + overlap, with paragraph/sentence
             back-up at the window edge. Kept for A/B comparison.

Each chunk carries its source + page for citation.
"""

import hashlib
import re
from dataclasses import dataclass

from raglab.ingest import Document

# Matches section boundaries in the legal corpus:
#   "6. GOVERNING LAW."  (numbered + ALLCAPS — NDA style)
#   "## 7. Governing Law" / "## Background"  (markdown — MSA / opinion style)
#   "# MASTER SERVICES AGREEMENT"  (top-level markdown title)
_HEADER_RE = re.compile(r"(?m)^(?:\d+\.\s+[A-Z]|#{1,3}\s+\S)")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None


def _chunk_id(source: str, page: int | None, text: str, pos: int) -> str:
    return hashlib.sha1(
        f"{source}|{page}|{text[:64]}|{pos}".encode()
    ).hexdigest()[:16]


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


def _split_sections(text: str) -> list[str]:
    """Split text at section headers; returns one string per section.

    Text before the first header becomes a preamble section. Each subsequent
    section includes its header line plus the content that follows.
    """
    boundaries = [m.start() for m in _HEADER_RE.finditer(text)]
    if not boundaries:
        return [text.strip()] if text.strip() else []
    sections: list[str] = []
    if boundaries[0] > 0:
        preamble = text[: boundaries[0]].strip()
        if preamble:
            sections.append(preamble)
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)
    return sections


def _section_aware_chunks(doc: Document, size: int, overlap: int) -> list[str]:
    """Return text pieces using section-aware strategy."""
    pieces: list[str] = []
    for section in _split_sections(doc.text):
        if len(section) <= size:
            pieces.append(section)
        else:
            pieces.extend(_split_one(section, size, overlap))
    return pieces


def chunk_documents(
    docs: list[Document], size: int, overlap: int, strategy: str = "section"
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        pieces = (
            _section_aware_chunks(doc, size, overlap)
            if strategy == "section"
            else _split_one(doc.text, size, overlap)
        )
        for piece in pieces:
            cid = _chunk_id(doc.source, doc.page, piece, len(out))
            out.append(Chunk(id=cid, text=piece, source=doc.source, page=doc.page))
    return out
