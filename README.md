# Private Legal RAG — on-device retrieval over legal documents

A **fully on-device** Retrieval-Augmented Generation (RAG) chatbot for **legal documents**.
It runs against a **local** open LLM (Gemma-4-12B via `mlx_vlm.server`) with **local embeddings**
(sentence-transformers) and a **local** vector store (Chroma).

> **The thesis: privacy is the product.** Legal (attorney–client privilege), medical (HIPAA), and
> financial (compliance) work all *forbid* sending documents to third-party APIs. So *"runs fully
> on-prem — your documents never leave your infrastructure"* isn't a limitation here; it's the
> selling point. The same provider-agnostic core can later point at a hosted model for a public demo
> while the private deployment stays air-gapped.

**Honest privacy claim:** no query or document data leaves the machine *at inference*. The model
weights and the embedding model were one-time online downloads; after that the system runs offline.

GenAcademy "Mastering Agentic AI" — **Week 2 (Grounding AI with RAG & Context Engineering), Stage 1.**

---

## What it does

Five modes, selectable in the sidebar, building up the RAG patterns in order of value:

| Mode | Pattern | What it adds |
|---|---|---|
| **Chat** | — | Plain local model, no retrieval (baseline) |
| **Simple RAG** | Simple RAG | Retrieve top-k chunks → answer **with `[source, p.N]` citations** |
| **+Memory** | RAG with Memory | Condenses follow-ups ("what about *its* term?") into standalone questions before retrieving |
| **CRAG** | Corrective RAG | **Grades** each retrieved chunk for relevance; if none are relevant, **abstains** ("I don't know") instead of hallucinating |
| **HyDE (A/B)** | Hypothetical Document Embeddings | Drafts a hypothetical answer, embeds *that* to retrieve; shows raw-vs-HyDE retrieval side by side |

**CRAG is the differentiator** — the "won't cite what it can't find" behavior is the canonical legal-AI
requirement (lawyers have been sanctioned for hallucinated citations).

## Run it

Prereq — start the local model server (one-time `uv pip install -U mlx-vlm`):

```bash
uv run python -m mlx_vlm.server --model mlx-community/gemma-4-12B-it-4bit --port 8085
```

Then:

```bash
uv sync
uv run python scripts/spike_endpoint.py        # Step Zero: probe the endpoint's capabilities
uv run python scripts/ingest_corpus.py data/legal --reset   # build the Chroma index
uv run streamlit run app.py                     # chat UI
```

Quality gates & eval:

```bash
uv run ruff check .
uv run pytest -q                  # 10 unit tests (pure core, no model needed)
uv run python eval/run_eval.py    # 10 legal Q&A incl. 2 unanswerable → CRAG must abstain
```

## Architecture

Pure core / thin view (same discipline as the Week-1 project): **all logic in `src/raglab/`** (unit-tested,
no Streamlit), Streamlit only in `app.py`. This is what lets the core be *lifted* into later stages.

```
ingest → chunk → embed (sentence-transformers) → Chroma           # data layer (raw, transparent)
              query → retrieve → [grade → branch] → answer          # CRAG via LangGraph state machine
```

- `llm.py` — `openai` SDK pointed at the local endpoint (the Week-1 `base_url` trick, pointed local).
- `crag.py` — **LangGraph** graph: `retrieve → grade → (generate | abstain)`. Chosen because CRAG *is*
  a state machine, and it's the same substrate Week 3 (Agentic) will reuse.

### The one decision that differs from every CRAG tutorial

The canonical LangGraph CRAG tutorial grades chunks with `llm.with_structured_output(...)`, which relies
on OpenAI **function-calling**. `scripts/spike_endpoint.py` verifies the local server supports **neither**
`response_format=json_object` **nor** function-calling:

```
1. Basic completion ............... OK
2. response_format=json_object .... NOT SUPPORTED (400)
3. tools / function-calling ....... IGNORED (no tool_calls)
```

So the grader is a **plain yes/no prompt, parsed by hand** (`parse_grade`). LangGraph still earns its
place — for the branch/state machine, not for structured output.

## A measured retrieval limitation (and the Stage-2 fix)

The eval harness caught a real failure: *"How long is the term of the NDA?"* The answer chunk ranked **#5**
under pure semantic search (distance 0.375 vs. the #4 cutoff at 0.372 — essentially tied), because the
keyword **"term"** doesn't embed strongly. Raising `top_k` to 5 surfaces it for this corpus, but the honest
fix is **hybrid search (BM25 + vector)** — keyword search nails exact legal terms. That's a **Stage 2**
item, motivated by measured evidence, not guesswork.

## Roadmap (this repo is Stage 1 of a portfolio piece)

| Stage | Deliverable |
|---|---|
| **1 — here** | Private local RAG engine (Simple → Memory → CRAG, HyDE bonus) over sample legal docs, eval harness |
| 2 | **Enterprise document parsing** (Docling / unstructured — tables, layout, OCR) + **hybrid search/rerank** |
| 3 | Generalize the core to **financial** (SEC filings) and **medical**; hosted-model adapter |
| 4 | Public **Hugging Face Space** demo (synthetic docs) + polished portfolio writeup |

## Data

`data/legal/` ships **synthetic** sample documents (a mutual NDA, a master services agreement, and a
fictional appellate opinion) — safe to commit and to demo publicly. Drop real/sensitive documents into a
gitignored folder and they stay local.

Stack: Python 3.12 · `uv` · Streamlit · Chroma · sentence-transformers · `openai` SDK · LangGraph ·
ruff + pytest.
