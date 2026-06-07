# SciFact Claim Verifier

A small RAG service that takes a scientific claim and decides whether it's **supported**, **refuted**, or **not_enough_info**, citing the specific abstracts it relied on. Built as a 2-week consolidation project against the [BeIR/SciFact](https://huggingface.co/datasets/BeIR/scifact) benchmark.

This is calibration practice, not a production system. The goal is to measure retrieval quality against a public benchmark with ground truth.

---

## What it does

Given a claim like *"Active Ly49Q prevents neutrophil polarization,"* the service:

1. Embeds the claim with OpenAI `text-embedding-3-small`
2. Retrieves the top-K most similar abstracts from a Qdrant vector database (5183 indexed SciFact abstracts)
3. Passes the claim + retrieved abstracts to `gpt-4o-mini` with a structured-output schema
4. Returns a typed verdict, the supporting chunk IDs, and 1–3 sentences of reasoning
5. Logs the request to SQLite for later inspection

A separate eval script measures retrieval quality against the SciFact `qrels` (ground-truth relevance judgments).

---

## Architecture

```
                ┌──────────────────┐
                │  Streamlit UI    │
                └────────┬─────────┘
                         │ HTTP
                         ▼
                ┌──────────────────┐
                │   FastAPI        │  /verify
                └────────┬─────────┘
                         │
        ┌────────────────┼───────────────────┐
        ▼                ▼                   ▼
   embed query     Qdrant top-K        LLM verdict
   (OpenAI)        (cosine sim)        (gpt-4o-mini,
                                        structured output)
        │                                    │
        └─────────────┬──────────────────────┘
                      ▼
             SQLite request log
```

A separate `eval.py` script bypasses the API and calls `retrieve_documents` directly across all 300 SciFact test queries, comparing retrieved corpus IDs to the qrels.

---

## Stack

- **Web service**: FastAPI + uvicorn
- **Vector DB**: Qdrant (Docker, local)
- **Embeddings**: OpenAI `text-embedding-3-small` (1536-dim, cosine)
- **Generation**: OpenAI `gpt-4o-mini` with Pydantic structured output
- **Logging**: SQLite via SQLModel
- **UI**: Streamlit
- **Package manager**: `uv`

---

## How to run

**1. Install dependencies**

```bash
uv sync
```

**2. Start Qdrant**

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**3. Set your OpenAI key**

Create `.env` at the project root:

```
OPENAI_API_KEY=sk-...
```

**4. Ingest the corpus (one-time, ~3–5 minutes, ~2¢)**

```bash
cd src
uv run ingest.py
```

**5. Start the API**

```bash
uv run uvicorn api:app --reload --port 8001
```

Visit `http://127.0.0.1:8001/docs` for the interactive Swagger UI.

**6. (Optional) Start the Streamlit UI**

In a second terminal:

```bash
uv run streamlit run app.py
```

**7. Run eval**

```bash
cd src
uv run eval.py
```

---

## Eval methodology

- **Dataset**: BeIR/SciFact test split (300 queries with ground-truth relevance, 5183 corpus documents)
- **Retrieval**: top-10 dense retrieval over OpenAI embeddings
- **Metrics**:
  - **Recall@5** / **Recall@10**: of the relevant docs for a query, what fraction are in the top K?
  - **MRR** (Mean Reciprocal Rank): on average, how high up the list does the first relevant doc appear?
- **No LLM judge**: this measures *retrieval* quality only. Generation quality (whether the verdict is correct) is intentionally out of scope.
- Skipped 809 queries from the full `queries` split that have no test-set qrels (these are train/dev queries with no ground truth in the test set).

---

## Baseline results

Run: `baseline-2026-06-07`

| Metric | Value |
|---|---|
| Queries evaluated | 300 |
| Recall@5 | **0.797** |
| Recall@10 | **0.854** |
| MRR | **0.680** |
| Wall time | 113s |

**Interpretation**

- Recall@5 of 0.797 is **above** the typical published baseline (~0.70–0.75) for `text-embedding-3-small` on SciFact. Likely contributors: concatenating `title + text` for ingestion (most baselines embed `text` alone), and the relative strength of the OpenAI embedding model versus older SBERT baselines.
- MRR of 0.680 means the first relevant doc lands at roughly position 1.5 on average — the system is good at *ranking*, not just *finding*.
- The gap between Recall@5 and Recall@10 is small (+5.7%). Most retrieval misses are "the relevant doc isn't in the top 10 at all," not "it's at rank 7 instead of rank 4." Re-ranking won't help those queries; only better embeddings or hybrid search would.

---

## Ablation

_(To be filled in after Step 8.)_

Planned: pick one of
- **A.** Compare embedding models (local `bge-small-en-v1.5` vs. OpenAI)
- **B.** Add a cross-encoder reranker (retrieve top-20, rerank to top-5)
- **C.** Chunking experiment (single-chunk-per-abstract vs. sentence-level chunks)

---

## What I'd improve with more time

- **Add an LLM-as-judge** for generation quality, not just retrieval. The current eval only measures whether the right *document* was retrieved, not whether the LLM's *verdict* was correct.
- **Hybrid retrieval** (BM25 + dense) — likely the biggest single lift for the queries currently missed entirely.
- **Local model comparison** — run end-to-end with `bge-small-en-v1.5` embeddings and a local LM Studio LLM to demonstrate cost discipline.
- **Per-query failure analysis** — eval currently produces aggregate numbers; per-query inspection of low-RR queries would reveal *why* retrieval fails (vocabulary mismatch? multi-concept queries? polarity?).
- **Production hardening** — proper async, retries, request validation beyond Pydantic basics, observability.

(Deliberately out of scope for this consolidation project — saved for Phase 2.)

---

## Project layout

```
.
├── app.py                  # Streamlit UI
├── src/
│   ├── api.py              # FastAPI app (POST /verify)
│   ├── ingest.py           # Embed corpus, upsert to Qdrant
│   ├── retrieve.py         # Query → top-K RetrievedChunk
│   ├── generate.py         # Claim + chunks → ClaimVerdict
│   ├── eval.py             # Retrieval eval over SciFact qrels
│   ├── db.py               # SQLite logging (requests + eval runs)
│   └── models.py           # Pydantic schemas
├── data/
│   └── verifications.db    # SQLite log
├── notebooks/              # Hand-test scratch
└── tests/                  # Sanity tests
```
