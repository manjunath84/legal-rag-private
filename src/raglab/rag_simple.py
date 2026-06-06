# src/raglab/rag_simple.py
"""Simple RAG: retrieve -> stuff context -> answer with citations.

The foundation pattern ("open-book exam"). Returns both the answer and the hits
so the UI can show the sources the answer was grounded in.
"""

from dataclasses import dataclass, field

from openai import OpenAI

from raglab import prompts
from raglab.chunk import Chunk
from raglab.config import Settings, settings
from raglab.llm import chat
from raglab.retrieve import format_context, retrieve


@dataclass
class RagResult:
    answer: str
    hits: list[tuple[Chunk, float]] = field(default_factory=list)
    note: str = ""  # optional pipeline note (e.g. CRAG abstention reason)


def answer(
    question: str,
    cfg: Settings = settings,
    client: OpenAI | None = None,
    k: int | None = None,
) -> RagResult:
    hits = retrieve(question, cfg, k)
    if not hits:
        return RagResult(
            answer="I don't know — that is not in the provided documents.",
            hits=[],
            note="no chunks retrieved",
        )
    context = format_context(hits)
    reply = chat(
        [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {"role": "user", "content": prompts.answer_user(question, context)},
        ],
        cfg,
        client,
    )
    return RagResult(answer=reply, hits=hits)
