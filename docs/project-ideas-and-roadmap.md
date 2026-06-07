# Project Ideas & Roadmap — Private Legal RAG

*Captured from the Week-2 brainstorming session (Gen Academy, "Grounding AI with RAG & Context
Engineering"). This is the thinking behind the build — the decisions, the alternatives considered,
and where it goes next. The executable plan lives in the repo; this is the "why".*

---

## 1. The North Star

Build an **enterprise-grade RAG application for regulated documents**, eventually hosted as a public
**Hugging Face Space** with code on **GitHub**, as a **job-search portfolio piece** proving depth in
parsing **legal** (then financial, then medical) documents.

**The unifying thesis — privacy *is* the product.** Legal (attorney–client privilege), medical
(HIPAA), and financial (compliance) work all *forbid* sending documents to third-party APIs. So
*"runs fully on-prem — your documents never leave your infrastructure"* is not a limitation; it is
**the selling point and the portfolio headline**. The provider-agnostic core can later point at a
hosted model for a public demo while the private deployment stays air-gapped.

**Honest privacy claim:** no query or document data leaves the machine *at inference*. The model
weights and the embedding model were one-time online downloads; after that it runs offline.

---

## 2. Staged roadmap

| Stage | Status | Deliverable |
|---|---|---|
| **1 — this week** | ✅ Built | Private local RAG engine (Simple → Memory → CRAG, HyDE bonus) over sample **legal** docs; eval harness; clean liftable core |
| 2 — later | Planned | **Enterprise document parsing** (Docling / unstructured — tables, layout, OCR) + **hybrid search / rerank** (BM25 + vector) |
| 3 — later | Planned | Generalize the core to **financial** (SEC filings) and **medical**; hosted-model adapter |
| 4 — final | Planned | Public **Hugging Face Space** demo (synthetic docs) + polished portfolio writeup |

**Why Legal leads Stage 1:** text-heavy (works with simple parsing now); CRAG's "won't cite what it
can't find" is the canonical legal-AI story (lawyers have been sanctioned for hallucinated citations);
abundant public corpora (court opinions, CUAD contracts).

---

## 3. Locked decisions (and the alternatives we rejected)

| Decision | Choice | Alternatives considered → why not |
|---|---|---|
| **Corpus** | Sample **legal** docs (synthetic NDA, MSA, opinion) | Personal docs (harder to gather); financial (best for Stage 2 parsing); medical (data/PII hard to source) |
| **Advanced pattern** | **CRAG** (+ HyDE bonus) | HyDE-only (less trust story); Hybrid+rerank (needs extra model — Stage 2); Agentic (fragile on local 12B — Week 3) |
| **Framework** | **Raw data layer + LangGraph** for CRAG | Full LangChain (heavy, hides mechanics); LlamaIndex (hides retrieval, weaker Week-3 bridge); all-raw (loses LangGraph reps) |
| **Embeddings** | **sentence-transformers** (in-process, local) | Ollama embeddings (extra server); mlx-embeddings (newer, less battle-tested) |
| **Vector DB** | **Chroma** (persistent, local) | Pinecone (cloud — breaks privacy); FAISS (no metadata convenience) |
| **LLM** | Local **Gemma-4-12B** via `mlx_vlm.server` `:8085` | Any cloud API (breaks the privacy thesis) |
| **Parsing (Stage 1)** | Simple text (`pypdf`) | Docling / unstructured deferred to Stage 2 (would eat the week) |
| **Multimodal** | Skipped Stage 1 | The VLM makes it feasible later — documented future extension |
| **UI** | **Streamlit** | Reused from Week 1; fastest path to a chat UI |

---

## 4. The 10 RAG patterns (from the primer) — what we use vs. defer

From Aishwarya Srinivasan's "RAG in 2026" overview. RAG = an **open-book exam**: the model retrieves
relevant documents to ground its answers instead of relying on training data alone.

| # | Pattern | Decision |
|---|---|---|
| 1 | **Simple RAG** | ✅ Stage 1 — the foundation |
| 2 | **RAG with Memory** | ✅ Stage 1 — condense follow-ups into standalone questions |
| 3 | **Branched RAG** (decompose complex queries) | ⏳ Later — multi-part legal questions |
| 4 | **HyDE** (hypothetical doc embeddings) | ✅ Stage 1 bonus — A/B retrieval demo |
| 5 | **Adaptive RAG** (route: retrieve or not) | ⏳ Later — overlaps with CRAG/agentic |
| 6 | **Corrective RAG (CRAG)** | ✅ Stage 1 — **the differentiator** |
| 7 | **Self-RAG** (reflection tokens) | ⏳ Later — needs reliable structured output (local model lacks it) |
| 8 | **Agentic RAG** (LLM orchestrator) | ➡️ **Week 3** (Agentic Leap) — LangGraph reps from CRAG carry over |
| 9 | **Multimodal RAG** (charts/images/tables) | ⏳ Later — feasible because we run a VLM, but fiddly |
| 10 | **Graph RAG** (entity relationships) | ⏳ Later — heavier infra |

---

## 5. Architecture (one-paragraph)

**Pure core / thin view.** All logic in `src/raglab/` (unit-tested, no Streamlit); the UI is the only
file with `st.*`. Two pipelines: **ingestion** (`ingest → chunk → embed → Chroma`, offline) and
**query** (`embed → retrieve → [CRAG grade → branch] → generate`). The LLM client is the `openai` SDK
pointed at the local endpoint. See **[`architecture.drawio`](architecture.drawio)** (open in
diagrams.net / the draw.io VS Code extension) for the full picture.

**The one decision that differs from every CRAG tutorial:** the canonical LangGraph CRAG tutorial
grades chunks with `llm.with_structured_output(...)` (OpenAI function-calling). The Step-Zero spike
proved the local server supports **neither** `response_format=json_object` **nor** function-calling —
so the grader is a **plain yes/no prompt, parsed by hand**. LangGraph still earns its place for the
branch/state machine (the same substrate Week 3 reuses).

---

## 6. Key technical findings (from actually building it)

1. **Endpoint capabilities** — `mlx_vlm.server` does basic completions only; no structured output, no
   tool-calling. Drove the hand-parsed grader. (`scripts/spike_endpoint.py`)
2. **A measured retrieval miss** — the NDA "term" fact ranks #5 under pure semantic search (near-tied
   with #4). This is the concrete, evidence-based motivation for **Stage-2 hybrid search (BM25)** —
   keyword search nails exact legal terms. Documented, not papered over.
3. **Condense must stay minimal** — early "+Memory" condensing injected party names that *hurt*
   retrieval (pulled toward intro/signature chunks). Fix: prompt condense to resolve pronouns but keep
   the question short.

---

## 7. How this feeds the later cohort weeks

The cohort reads/builds in order; this engine is designed to be lifted forward:

- **Week 3 — Agentic Leap:** reuse the **LangGraph** state machine; upgrade CRAG → Agentic RAG
  (pattern #8) with retrieval as a tool the agent decides to call.
- **Week 4 — Finetuning / Local Models:** the local-model adapter and corpus are already here; finetune
  an embedding or grader model on legal data.
- **Week 5 — Evals & Observability:** `eval/run_eval.py` + `qa_set.yaml` are the seed; expand to a real
  eval suite with retrieval metrics (hit@k, faithfulness).
- **Week 6 — Security / Production:** the privacy/on-prem story is the spine; add authn, PII redaction,
  audit logging.
- **Week 7 — Career Launchpad:** the polished GitHub repo + HF Space *is* the portfolio artifact.

---

## 8. Stage 2a findings — hybrid search (built)

Added BM25 + dense vectors fused with **Reciprocal Rank Fusion** (`src/raglab/hybrid.py`), wired behind
`settings.retrieval_mode` so every mode benefits. But the more valuable output was **how we measured it**:

- **Methodology fix:** comparing retrievers *through* CRAG conflated retrieval with the local model's
  flaky temp-0 grading (a one-off "failure" vanished on re-runs). So retrieval is now evaluated
  **deterministically, with no LLM** — does the gold chunk land in top-k? (`scripts/compare_retrieval.py`)
- **Honest result:** on this 9-chunk corpus the methods **tie** (recall@5 = 8/8 for vector, BM25, and
  hybrid). Hybrid is not a free win. Its value is *per-query*: it reliably fixes the exact-keyword
  "term" miss, at the cost of keyword cross-talk between similar clauses (the two governing-law sections).
- **Gold-marker validation caught a real bug:** the marker "State of Delaware" wasn't intact in any chunk —
  **char-window chunking split the governing-law clause across two chunks.** → motivates section-aware
  chunking (possibly a bigger lever than fusion here).
- **Lesson logged:** separate *retrieval* eval (deterministic) from *end-to-end* eval (LLM, nondeterministic
  9–10/10). Clean bridge to Week 5 (Evals & Observability).

## 9. Stage 2b findings — reranker + section-aware chunking (built)

Two targeted fixes for two measured Stage 2a weaknesses.

**Fix 1 — Section-aware chunking** (`src/raglab/chunk.py`): detect section boundaries
(numbered headings `6. GOVERNING LAW`, markdown `## Header`) before applying size limits.
Corpus goes from 9 char-window chunks → 21 section chunks. The NDA governing-law clause
("State of Delaware") that was split across two chunks in Stage 2a is now intact in a
single 288-char section chunk. Gold marker changed back to `"State of Delaware"` in
`eval/qa_set.yaml`; passes at k=5.

**Fix 2 — Cross-encoder reranker** (`src/raglab/rerank.py`): after hybrid retrieval
fetches `candidate_k=10` candidates, `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores
each (query, chunk) pair and returns the best `top_k=5`. At k=1, the NDA governing-law
question goes from rank-2 (MSA governing-law chunk at rank-1, cross-talk) → rank-1
(correct NDA chunk). Model is ~85 MB, loaded once and cached.

**Attribution (measured at k=1):**
- Baseline (char chunking, no rerank): VEC 4/8, BM25 2/8, HYB 4/8
- Stage 2b (section chunking + rerank): RERANK 4/8 (cross-talk fixed; preamble over-ranking
  is a known cross-encoder limitation when document titles repeat query terms)

**Config flags for A/B:** `RAGLAB_CHUNK_STRATEGY=char`, `RAGLAB_RERANK_ENABLED=false`.

**Lesson logged:** cross-encoders trained on MS MARCO web search over-weight exact term
matches in document preambles (MSA title section ranks #1 for any "master services
agreement" query at k=1). This is acceptable at k=5 (all gold chunks present), and the
NDA cross-talk fix is the primary motivation. Flag as a known limitation in the writeup.

## 10. Next steps

- [x] `git init` + first commit; pushed to GitHub (manjunath84/legal-rag-private), topics + diagram
- [x] (Stage 2a) hybrid search (BM25 + vector, RRF) + deterministic retrieval-recall eval
- [x] (Stage 2b) reranker (cross-encoder) + section-aware chunking + updated eval
- [ ] (Stage 2c) swap `pypdf` → Docling/unstructured for table/layout parsing (needs a complex PDF corpus)
- [ ] (Stage 3) add a financial corpus (SEC EDGAR 10-Ks) and a hosted-model adapter
- [ ] (Stage 4) deploy a Hugging Face Space on synthetic docs + portfolio writeup
