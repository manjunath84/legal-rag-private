# src/raglab/hyde.py
"""HyDE (Hypothetical Document Embeddings) — bonus pattern.

A short question embeds differently from the dense passage that answers it. HyDE
first asks the model to DRAFT a plausible answer, embeds THAT, and retrieves with
it — the draft "looks like" the documents, improving recall on vague queries.

Exposes both retrieval paths so the UI can A/B them side by side.
"""

from openai import OpenAI

from raglab import prompts, store
from raglab.chunk import Chunk
from raglab.config import Settings, settings
from raglab.embed import embed_query
from raglab.llm import chat


def hypothetical_answer(question: str, cfg: Settings = settings,
                        client: OpenAI | None = None) -> str:
    return chat(
        [
            {"role": "system", "content": prompts.HYDE_SYSTEM},
            {"role": "user", "content": prompts.hyde_user(question)},
        ],
        cfg,
        client,
        temperature=0.3,
        max_tokens=160,
    )


def retrieve_hyde(
    question: str,
    cfg: Settings = settings,
    client: OpenAI | None = None,
    k: int | None = None,
) -> tuple[str, list[tuple[Chunk, float]]]:
    """Return (the hypothetical passage, hits retrieved using its embedding)."""
    draft = hypothetical_answer(question, cfg, client)
    vec = embed_query(draft, cfg.embed_model)
    hits = store.query(vec, k or cfg.top_k, cfg)
    return draft, hits
