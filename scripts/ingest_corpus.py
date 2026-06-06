#!/usr/bin/env python
"""Build / refresh the Chroma index from a folder of legal documents.

    uv run python scripts/ingest_corpus.py [data/legal] [--reset]

All processing is local: load -> chunk -> embed (sentence-transformers) -> Chroma.
Nothing leaves the machine.
"""

import argparse
from pathlib import Path

from raglab import store
from raglab.chunk import chunk_documents
from raglab.config import settings
from raglab.embed import embed_texts
from raglab.ingest import load_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(settings.data_dir),
                    help="directory of .pdf/.txt/.md docs")
    ap.add_argument("--reset", action="store_true", help="drop the collection first")
    args = ap.parse_args()

    directory = Path(args.path)
    if not directory.exists():
        print(f"No such directory: {directory}")
        return 1

    if args.reset:
        store.reset()
        print(f"Reset collection '{settings.collection}'.")

    docs = load_dir(directory)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        print(f"No text extracted from {directory}.")
        return 1

    print(f"Loaded {len(docs)} doc-units → {len(chunks)} chunks. Embedding "
          f"with {settings.embed_model} ...")
    embeddings = embed_texts([c.text for c in chunks])
    store.add_chunks(chunks, embeddings)
    print(f"Indexed. Collection '{settings.collection}' now holds {store.count()} chunks "
          f"at {settings.chroma_dir}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
