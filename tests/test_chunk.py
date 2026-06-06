from raglab.chunk import chunk_documents
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
