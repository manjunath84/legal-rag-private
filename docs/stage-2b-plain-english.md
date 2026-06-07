# Stage 2b — What We're Building and Why (Plain English)

*This document explains the two problems Stage 2a uncovered and the two fixes Stage 2b will build.*
*No jargon — think of the RAG system as a very fast research assistant.*

---

## Quick recap: how the chatbot works

When you ask a question, the chatbot does this in order:

1. **Fetch** — search through all the document chunks and grab the most relevant ones
2. **Grade** — check whether those chunks actually contain an answer (CRAG)
3. **Answer** — write a response using only what was found, with citations

Stage 2a added **hybrid search** to step 1: instead of searching by meaning alone, we now also search by keywords, then combine the two lists. That helped.

But the Stage 2a tests revealed **two new problems**, both in step 1 (the fetch step). Stage 2b fixes them.

---

## Problem 1 — Two documents, same words, wrong one wins ("keyword cross-talk")

### What happens today

We have two contracts in the corpus:
- **Mutual NDA** (Northwind Analytics ↔ Cedar Grove Software)
- **Master Services Agreement** (Atlas Cloud Services ↔ Riverside Health Group)

Both contracts have a "Governing Law" section. Here's a snippet from each:

> **NDA:** *"This Agreement is governed by and construed in accordance with the laws of the **State of Delaware**..."*

> **MSA:** *"This Agreement shall be governed by and construed in accordance with the laws of the **State of New York**..."*

When you search for **"What is the governing law of the NDA?"**, the keyword search (BM25) sees the words "governing", "law", "state" and finds **both** governing law sections. They score almost identically on keywords — so the wrong document's section can slip in at rank #1.

### The analogy

Imagine you ask a paralegal: *"Find the termination clause in the Northwind NDA."*
The paralegal pulls out every document that contains the word "termination" and hands them all to you in a random pile. That's what keyword cross-talk looks like.

### The fix: a Reranker

A reranker is a second, smarter scoring pass. It reads the **full question** and the **full chunk** together (not separately), and scores how well this specific chunk answers this specific question.

```
Before fix (Stage 2a):
  Retrieve 10 candidates → take the top 5 by combined keyword+vector score

After fix (Stage 2b):
  Retrieve 10 candidates → reranker reads each pair (question + chunk)
                         → re-sorts by relevance → take the top 5
```

The reranker is a small local model (~85 MB, downloaded once — nothing leaves the laptop). It's much heavier than simple keyword matching, which is why we don't use it for the full document corpus — only for the 10 already-shortlisted candidates.

**New file:** `src/raglab/rerank.py`
**New config flag:** `rerank_enabled = true` (can be turned off for A/B testing)

---

## Problem 2 — The document chopper cuts sentences in half ("clause fragmentation")

### What happens today

Before we can search, every document gets cut into small pieces ("chunks") of about 800 characters. This is a simple mechanical cut: every 800 characters, make a new chunk.

Here's the actual governing law clause in the NDA:

```
"This Agreement is governed by and construed in accordance with the laws of the State of
Delaware, without regard to its conflict-of-laws principles..."
```

The character-window chopper happened to cut right here:

```
Chunk 5 ends:  "...the laws of the State of"
Chunk 6 starts: "Delaware, without regard..."
```

Now **no single chunk contains the phrase "State of Delaware"**. When you search for Delaware's governing law, the retrieval misses entirely — neither chunk has the complete phrase.

This is exactly the gold-marker bug we found in Stage 2a: "State of Delaware" wasn't retrievable from any chunk, and we had to change our test marker to "New Castle County" (a phrase that happened not to get split).

### The analogy

Imagine photocopying a contract page-by-page, but your copier sometimes cuts a sentence mid-word at the page boundary. When you later search for that sentence, you can't find it.

### The fix: Section-aware chunking

Instead of cutting every 800 characters blindly, detect the natural section boundaries in the document first:

```
Blind chopper (today):
  "...laws of the State of" | "Delaware, without regard..."  ← split mid-clause

Section-aware chopper (Stage 2b):
  Section "6. GOVERNING LAW" is kept whole:
  "This Agreement is governed by...New Castle County, Delaware."
```

Section boundaries are things like:
- Numbered headings: `6. GOVERNING LAW`, `3. TERM`
- Markdown headers: `## Section 2`
- Blank line + ALL CAPS line

If a section is still too long (e.g., a 3,000-word definition section), we fall back to the old chopper inside that section — but only after trying to respect the boundary.

**Changed file:** `src/raglab/chunk.py` — adds `section_aware_chunk()` alongside the existing character-window function
**New config flag:** `chunk_strategy = "section"` | `"char"` (default: `"section"`)

---

## How we'll know it worked

Same deterministic eval harness built in Stage 2a — no LLM involved, just "does the gold chunk land in the top-k?":

| Test | What passes | What it proves |
|---|---|---|
| `compare_retrieval.py` with section chunking | "State of Delaware" is now intact in a single chunk (no split) | Section-aware chunking fixed the fragmentation |
| `compare_retrieval.py` with reranker on | NDA governing-law at rank 1, MSA governing-law pushed down | Reranker fixed the cross-talk |
| `eval/run_eval.py` end-to-end | Still 10/10 (or 9-10/10) | Nothing regressed |

---

## What changes for you as a user

Nothing visible. Same Streamlit UI, same five modes. Under the hood:
- Questions retrieve 10 candidates instead of 5, then the reranker selects the best 5
- Documents are chunked at section boundaries when ingested

If you re-ingest (`uv run python scripts/ingest_corpus.py data/legal --reset`), the index rebuilds with section-aware chunks. Old char-window index stays untouched unless you reset.

---

## Stage 2b summary table

| What | Why | New code |
|---|---|---|
| **Reranker** — re-scores the 10 candidates after hybrid search | Fixes keyword cross-talk between similar clauses across documents | `src/raglab/rerank.py` |
| **Section-aware chunking** — cuts at section boundaries, not every 800 chars | Fixes clause fragmentation (governing law clause split across chunks) | `src/raglab/chunk.py` (new strategy) |
| **Config flags** | A/B testing: `rerank_enabled`, `chunk_strategy` | `src/raglab/config.py` |
| **Eval** | Same deterministic recall@k harness — no new test infrastructure needed | `scripts/compare_retrieval.py` (unchanged) |
