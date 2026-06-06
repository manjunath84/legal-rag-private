# src/raglab/crag.py
"""Corrective RAG (CRAG) as a LangGraph state machine.

Flow:  retrieve -> grade each chunk (yes/no) -> branch
         - any relevant   -> generate answer from the relevant chunks
         - none relevant   -> abstain ("I don't know")

The grader is a HAND-PARSED yes/no prompt, NOT llm.with_structured_output(...).
The local mlx_vlm.server supports neither response_format=json_object nor
function-calling (verified by scripts/spike_endpoint.py), so we ask for a bare
word and parse it. LangGraph earns its place as the branch/state machine — the
same substrate Week 3 (Agentic) will reuse — not for structured output.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from raglab import prompts
from raglab.chunk import Chunk
from raglab.config import Settings, settings
from raglab.llm import chat
from raglab.rag_simple import RagResult
from raglab.retrieve import format_context, retrieve

ABSTAIN = "I don't know — that is not in the provided documents."


def parse_grade(reply: str) -> bool:
    """True iff the grader's bare-word reply starts with 'yes' (case/space tolerant)."""
    return reply.strip().lower().lstrip("\"'`*").startswith("yes")


class CRAGState(TypedDict):
    question: str
    hits: list[tuple[Chunk, float]]
    relevant: list[tuple[Chunk, float]]
    answer: str
    note: str


def _build_graph(cfg: Settings, client: OpenAI | None):
    def retrieve_node(state: CRAGState) -> CRAGState:
        state["hits"] = retrieve(state["question"], cfg)
        return state

    def grade_node(state: CRAGState) -> CRAGState:
        relevant: list[tuple[Chunk, float]] = []
        for chunk, score in state["hits"]:
            reply = chat(
                [
                    {"role": "system", "content": prompts.GRADE_SYSTEM},
                    {"role": "user", "content": prompts.grade_user(state["question"], chunk.text)},
                ],
                cfg,
                client,
                temperature=0.0,
                max_tokens=4,
            )
            if parse_grade(reply):
                relevant.append((chunk, score))
        state["relevant"] = relevant
        state["note"] = f"graded {len(state['hits'])} chunks → {len(relevant)} relevant"
        return state

    def generate_node(state: CRAGState) -> CRAGState:
        context = format_context(state["relevant"])
        state["answer"] = chat(
            [
                {"role": "system", "content": prompts.ANSWER_SYSTEM},
                {"role": "user", "content": prompts.answer_user(state["question"], context)},
            ],
            cfg,
            client,
        )
        return state

    def abstain_node(state: CRAGState) -> CRAGState:
        state["answer"] = ABSTAIN
        state["note"] += " → abstained (no relevant context)"
        return state

    def route(state: CRAGState) -> str:
        return "generate" if state["relevant"] else "abstain"

    g = StateGraph(CRAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("generate", generate_node)
    g.add_node("abstain", abstain_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route, {"generate": "generate", "abstain": "abstain"})
    g.add_edge("generate", END)
    g.add_edge("abstain", END)
    return g.compile()


def answer_crag(
    question: str,
    cfg: Settings = settings,
    client: OpenAI | None = None,
) -> RagResult:
    graph = _build_graph(cfg, client)
    final: CRAGState = graph.invoke(
        {"question": question, "hits": [], "relevant": [], "answer": "", "note": ""}
    )
    return RagResult(answer=final["answer"], hits=final["relevant"], note=final["note"])
