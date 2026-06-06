# src/raglab/memory.py
"""RAG with Memory: condense a follow-up into a standalone question before retrieval.

Without this, "what about its termination clause?" retrieves on the pronoun and
misses. We rewrite it to "What is the termination clause of <X>?" using the chat
history, then run the normal Simple-RAG answer.
"""

from openai import OpenAI

from raglab import prompts
from raglab.config import Settings, settings
from raglab.llm import chat
from raglab.rag_simple import RagResult, answer

# A turn is (role, text) where role is "user" or "assistant".
Turn = tuple[str, str]


def _format_history(history: list[Turn], max_turns: int = 6) -> str:
    recent = history[-max_turns:]
    return "\n".join(f"{role}: {text}" for role, text in recent)


def condense(question: str, history: list[Turn], cfg: Settings = settings,
             client: OpenAI | None = None) -> str:
    """Rewrite a follow-up to be self-contained. First turn passes through unchanged."""
    if not history:
        return question
    rewritten = chat(
        [
            {"role": "system", "content": prompts.CONDENSE_SYSTEM},
            {"role": "user", "content": prompts.condense_user(_format_history(history), question)},
        ],
        cfg,
        client,
        temperature=0.0,
        max_tokens=120,
    )
    return rewritten.strip() or question


def answer_with_memory(
    question: str,
    history: list[Turn],
    cfg: Settings = settings,
    client: OpenAI | None = None,
    k: int | None = None,
) -> RagResult:
    standalone = condense(question, history, cfg, client)
    result = answer(standalone, cfg, client, k)
    if standalone != question:
        result.note = f"condensed → {standalone}"
    return result
