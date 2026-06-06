# app.py — the ONLY Streamlit file. All logic lives in src/raglab/ (unit-tested).
"""Private legal RAG chatbot — runs fully on-device against a local LLM.

Modes (sidebar):
  Chat        — plain local model, no retrieval
  Simple RAG  — retrieve → answer with citations
  +Memory     — condense follow-ups, then Simple RAG
  CRAG        — grade retrieved chunks; abstain if none are relevant
  HyDE (A/B)  — compare raw-question vs hypothetical-answer retrieval
"""

import streamlit as st

from raglab import store
from raglab.config import settings
from raglab.crag import answer_crag
from raglab.hyde import retrieve_hyde
from raglab.llm import chat
from raglab.memory import answer_with_memory
from raglab.rag_simple import answer
from raglab.retrieve import cite, retrieve

st.set_page_config(page_title="Private Legal RAG", page_icon="⚖️", layout="centered")

MODES = ["Chat", "Simple RAG", "+Memory", "CRAG", "HyDE (A/B)"]

with st.sidebar:
    st.title("⚖️ Private Legal RAG")
    st.caption("Fully on-device. No query or document data leaves this machine.")
    mode = st.radio("Mode", MODES, index=3)
    settings.top_k = st.slider("Top-k retrieved", 1, 8, settings.top_k)
    st.divider()
    st.metric("Indexed chunks", store.count())
    st.caption(f"Model: `{settings.llm_model}`")
    st.caption(f"Endpoint: `{settings.llm_base_url}`")
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

st.session_state.setdefault("history", [])  # list[(role, text)]


def _render_sources(hits) -> None:
    if not hits:
        return
    with st.expander(f"📎 Sources ({len(hits)})"):
        for chunk, score in hits:
            st.markdown(f"**{cite(chunk)}**  ·  distance `{score:.3f}`")
            st.caption(chunk.text[:400] + ("…" if len(chunk.text) > 400 else ""))


# Replay history
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("Ask about the legal documents…")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if mode == "Chat":
            msgs = [{"role": r, "content": t} for r, t in st.session_state.history]
            msgs.append({"role": "user", "content": prompt})
            reply = chat(msgs)
            st.markdown(reply)

        elif mode == "Simple RAG":
            with st.spinner("Retrieving + answering…"):
                res = answer(prompt)
            st.markdown(res.answer)
            _render_sources(res.hits)
            reply = res.answer

        elif mode == "+Memory":
            with st.spinner("Condensing + retrieving…"):
                res = answer_with_memory(prompt, st.session_state.history)
            if res.note:
                st.caption(f"🧠 {res.note}")
            st.markdown(res.answer)
            _render_sources(res.hits)
            reply = res.answer

        elif mode == "CRAG":
            with st.spinner("Retrieving → grading → answering…"):
                res = answer_crag(prompt)
            st.caption(f"🔍 {res.note}")
            st.markdown(res.answer)
            _render_sources(res.hits)
            reply = res.answer

        else:  # HyDE (A/B)
            with st.spinner("Drafting hypothetical → retrieving both ways…"):
                raw_hits = retrieve(prompt)
                draft, hyde_hits = retrieve_hyde(prompt)
                res = answer(prompt)  # answer uses raw retrieval for the reply
            st.markdown(res.answer)
            with st.expander("🧪 HyDE hypothetical passage (used only for retrieval)"):
                st.write(draft)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Raw-question retrieval**")
                for c, s in raw_hits:
                    st.caption(f"{cite(c)} · `{s:.3f}`")
            with col2:
                st.markdown("**HyDE retrieval**")
                for c, s in hyde_hits:
                    st.caption(f"{cite(c)} · `{s:.3f}`")
            reply = res.answer

    st.session_state.history.append(("user", prompt))
    st.session_state.history.append(("assistant", reply))
