# src/raglab/prompts.py
"""Prompt templates. Kept in one place so the legal-aware voice is consistent.

The grader/condense prompts are written for a local 12B model that does NOT
support structured output (verified via scripts/spike_endpoint.py): they ask for
a single bare word, which we parse by hand.
"""

# --- Answering ---
ANSWER_SYSTEM = (
    "You are a careful legal-document assistant. Answer ONLY from the provided "
    "context excerpts. Cite the source and page for every claim using the form "
    "[source, p.N]. If the context does not contain the answer, say exactly: "
    "\"I don't know — that is not in the provided documents.\" Never use outside "
    "knowledge. Be concise and precise; do not speculate."
)


def answer_user(question: str, context: str) -> str:
    return (
        f"Context excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with [source, p.N] citations."
    )


# --- CRAG relevance grader (hand-parsed, no structured output) ---
GRADE_SYSTEM = (
    "You grade whether a document excerpt is relevant to answering a question. "
    "Reply with a single word: yes or no. No punctuation, no explanation."
)


def grade_user(question: str, excerpt: str) -> str:
    return (
        f"Question: {question}\n\nExcerpt:\n{excerpt}\n\n"
        "Is this excerpt relevant? Answer yes or no."
    )


# --- Memory: condense a follow-up into a standalone question ---
CONDENSE_SYSTEM = (
    "Given a conversation and a follow-up message, rewrite the follow-up as a "
    "standalone question that makes sense without the history. Resolve pronouns "
    "to the document type being discussed (e.g. 'it' -> 'the NDA'), but keep the "
    "question MINIMAL: do not add party names, dates, or other details unless "
    "they are needed to disambiguate. Shorter is better for retrieval. "
    "Output ONLY the rewritten question, nothing else."
)


def condense_user(history: str, follow_up: str) -> str:
    return f"Conversation so far:\n{history}\n\nFollow-up: {follow_up}\n\nStandalone question:"


# --- HyDE: draft a hypothetical answer to embed for retrieval ---
HYDE_SYSTEM = (
    "Write a short, plausible passage (2-4 sentences) that would answer the "
    "question as if it appeared in a legal document. It is fine to be generic; "
    "this draft is only used to improve document retrieval, not shown to the user."
)


def hyde_user(question: str) -> str:
    return f"Question: {question}\n\nHypothetical answer passage:"
