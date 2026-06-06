# src/raglab/ingest.py
"""Load source documents into a uniform shape.

Stage 1: simple text extraction only (.pdf via pypdf, .txt, .md). No table/layout
parsing — that's Stage 2 (Docling/unstructured). PDFs are read per page so each
unit carries a page number for citations.
"""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}


@dataclass
class Document:
    text: str
    source: str  # filename
    page: int | None  # 1-based page for PDFs; None for txt/md


def load_file(path: Path) -> list[Document]:
    """Return one Document per page (PDF) or one Document (txt/md)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(text=text, source=path.name, page=None)] if text.strip() else []
    raise ValueError(f"Unsupported file type: {path.suffix} (supported: {sorted(SUPPORTED)})")


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    docs: list[Document] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(text=text, source=path.name, page=i))
    return docs


def load_dir(directory: Path) -> list[Document]:
    """Load every supported file under `directory` (recursively)."""
    docs: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            docs.extend(load_file(path))
    return docs
