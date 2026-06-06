from raglab.chunk import Chunk
from raglab.retrieve import cite, format_context


def _chunk(text, source, page):
    return Chunk(id="x", text=text, source=source, page=page)


def test_cite_with_and_without_page():
    assert cite(_chunk("t", "opinion.pdf", 3)) == "opinion.pdf, p.3"
    assert cite(_chunk("t", "nda.txt", None)) == "nda.txt"


def test_format_context_includes_citation_tags():
    hits = [
        (_chunk("Governing law is Delaware.", "nda.txt", None), 0.1),
        (_chunk("Payment within 45 days.", "msa.md", 2), 0.2),
    ]
    out = format_context(hits)
    assert "[nda.txt]" in out
    assert "[msa.md, p.2]" in out
    assert "Delaware" in out and "45 days" in out
