from raglab.chunk import _split_sections, chunk_documents
from raglab.ingest import Document


def test_short_doc_is_single_chunk():
    docs = [Document(text="A short clause.", source="x.txt", page=None)]
    chunks = chunk_documents(docs, size=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == "A short clause."
    assert chunks[0].source == "x.txt"


def test_long_doc_splits_with_overlap():
    text = ". ".join(f"Sentence number {i} about contracts" for i in range(200))
    docs = [Document(text=text, source="big.md", page=3)]
    chunks = chunk_documents(docs, size=300, overlap=50)
    assert len(chunks) > 1
    # metadata is carried onto every chunk
    assert all(c.source == "big.md" and c.page == 3 for c in chunks)
    # ids are unique
    assert len({c.id for c in chunks}) == len(chunks)


def test_empty_doc_yields_no_chunks():
    docs = [Document(text="   \n  ", source="empty.txt", page=None)]
    assert chunk_documents(docs, size=100, overlap=10) == []


# --- Section-aware chunking ---

_NDA_SNIPPET = (
    "5. NO LICENSE.\nNothing in this Agreement grants any license.\n\n"
    "6. GOVERNING LAW.\nThis Agreement is governed by the laws of the State of\n"
    "Delaware, without regard to conflict-of-laws principles.\n\n"
    "7. ENTIRE AGREEMENT.\nThis constitutes the entire understanding."
)


def test_section_splits_at_numbered_headings():
    sections = _split_sections(_NDA_SNIPPET)
    headings = [s.split("\n")[0] for s in sections]
    assert "5. NO LICENSE." in headings
    assert "6. GOVERNING LAW." in headings
    assert "7. ENTIRE AGREEMENT." in headings


def test_governing_law_section_stays_whole():
    """The NDA governing-law clause must land in a single chunk (Stage 2b claim)."""
    docs = [Document(text=_NDA_SNIPPET, source="nda.txt", page=None)]
    chunks = chunk_documents(docs, size=1000, overlap=100, strategy="section")
    gov_chunks = [c for c in chunks if "GOVERNING LAW" in c.text]
    assert len(gov_chunks) == 1, "governing-law section split across multiple chunks"
    # The text has a hard line wrap; both halves must be in the same chunk.
    assert "State of" in gov_chunks[0].text
    assert "Delaware" in gov_chunks[0].text


def test_sentence_case_numbered_line_not_a_boundary():
    """'1. The party shall...' must NOT be treated as a section header."""
    text = "BACKGROUND.\nIntroductory text.\n\n1. The party shall comply with all rules."
    sections = _split_sections(text)
    # The numbered sentence-case line should NOT split the document.
    assert len(sections) == 1, "sentence-case numbered line incorrectly treated as heading"


def test_char_strategy_unchanged():
    """strategy='char' must behave identically to pre-Stage-2b behaviour."""
    text = ". ".join(f"Sentence {i}" for i in range(100))
    docs = [Document(text=text, source="a.txt", page=None)]
    chunks = chunk_documents(docs, size=200, overlap=30, strategy="char")
    assert len(chunks) > 1
    assert all(c.source == "a.txt" for c in chunks)
