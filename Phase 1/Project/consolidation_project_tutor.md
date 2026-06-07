# Consolidation Project — RAG on Benchmark Data

A 2-week side project that runs alongside the ML Sprint. Goal: cement Phase 1 Week 1-4 skills with real code, against a public benchmark, without competing with the sprint's mental load.

**What this is NOT:**
- Not your signature project (that's the Approval Agent in Phase 2)
- Not interview-leading material (it's calibration practice)
- Not a research project (no novel techniques)
- Not a deployment exercise (local-only is fine)

**What this IS:**
- A working RAG system you built end-to-end against a standard benchmark
- Calibration on what "good retrieval" looks like with concrete pass-rate numbers
- A second small portfolio item with measurable results
- Practice infrastructure you can extend later

**Time budget**: ~10-15 hours over 2 weeks. ~1 hour per weekday evening + a weekend session.

**Scope discipline rule**: if you find yourself adding agents, complex eval harnesses, or a fancy UI, stop. Save that for Phase 2.

---

## The Project: Scientific Claim RAG

You're building a RAG service that takes a scientific claim like *"Apples reduce the risk of heart disease"* and answers whether the claim is supported by retrieved scientific abstracts, with citations.

This maps perfectly to the **BeIR/SciFact** dataset:
- ~5K scientific abstracts (your chunks)
- ~300 queries (scientific claims)
- Ground-truth labels (which abstracts are relevant to each claim, and whether they support or refute it)

Small enough to run end-to-end in minutes. Real enough to count as a benchmark. Clean enough that you won't fight the data.

---

## Why SciFact specifically

A few reasons:

1. **Size**: 5K documents is small enough to embed in 5-10 minutes, even with API calls. Fast iteration.
2. **Pre-chunked**: each "document" is already a short scientific abstract. You don't have to build a chunking pipeline — though you can experiment with it for ablations.
3. **Clean ground truth**: relevance judgments are binary and verified. Your eval logic will be simple.
4. **Published baselines**: BM25 gets recall@5 around 0.65-0.70. Dense retrieval (SBERT-class) gets 0.70-0.75. You'll know if your system is in the right ballpark.
5. **Realistic domain**: scientific claim verification is an active production use case. Not toy.

If you want something even smaller for first iteration, **BeIR/NFCorpus** is similar but ~3K docs. Same structure.

---

## Architecture

Single FastAPI service. SQLite for logs. Qdrant for vectors. Streamlit for demo. That's it.

```
[Streamlit UI / curl]
       ↓
[FastAPI service]
       ↓
[Pydantic request model]
       ↓
[Embed query with OpenAI text-embedding-3-small]
       ↓
[Qdrant top-K vector search]
       ↓
[Build prompt with retrieved abstracts]
       ↓
[LLM call: claim verification + citations]
       ↓
[Pydantic response model]
       ↓
[Log to SQLite: query, retrieved IDs, response, latency, cost]
       ↓
[Return to caller]
```

Separately:

```
[Eval script]
       ↓
[Load BeIR/SciFact queries + ground truth]
       ↓
[Run service on each query]
       ↓
[Compare retrieved IDs to ground truth → retrieval metrics]
       ↓
[Optional: LLM judge for answer correctness]
       ↓
[Aggregate results: recall@5, MRR, accuracy]
       ↓
[Save run results to SQLite]
       ↓
[Print summary report]
```

No agent loop. No tool calling. No multi-step reasoning. Single-step RAG. This is the point.

---

## Week 1: Ingestion and Service

### Day 1-2 (evenings, ~1 hour each): Setup

- Initialize project with `uv init`
- Set up dependencies: `fastapi`, `pydantic`, `httpx`, `qdrant-client`, `openai`, `datasets` (HuggingFace), `sqlmodel`, `python-dotenv`
- Create the directory structure:
  ```
  rag-scifact/
  ├── src/
  │   ├── api.py         # FastAPI app
  │   ├── ingest.py      # Loads data, embeds, indexes
  │   ├── retrieve.py    # Query → top-K chunks
  │   ├── generate.py    # Chunks + query → LLM answer
  │   ├── eval.py        # Runs eval against BeIR ground truth
  │   ├── models.py      # Pydantic schemas
  │   └── db.py          # SQLite logging
  ├── data/              # Cached dataset files
  ├── tests/             # A few sanity tests
  ├── .env               # API keys, never committed
  └── README.md
  ```
- Set up Qdrant locally:
  ```
  docker run -p 6333:6333 -p 6334:6334 \
      -v $(pwd)/qdrant_storage:/qdrant/storage \
      qdrant/qdrant
  ```
- Verify everything starts. No real logic yet.

### Day 3-4 (evenings): Ingestion pipeline

- Use the `datasets` library to load BeIR/SciFact:
  ```python
  from datasets import load_dataset
  corpus = load_dataset("BeIR/scifact", "corpus")
  ```
- Inspect the structure. Each document has `_id`, `title`, `text`. About 5K docs.
- Write `src/ingest.py` to:
  - Iterate the corpus
  - For each document, create a single "chunk" combining title + text (or split if you want to experiment with chunking — but the abstracts are short, so single-chunk-per-doc is reasonable)
  - Embed in batches of 100 using OpenAI's `text-embedding-3-small`
  - Upsert to Qdrant with `_id` as the point ID and `{title, text}` as payload
- Run it. Verify Qdrant has ~5K points.

This is where the "Phase 1 cementing" actually happens. You'll write async batched embedding code, handle rate limits, and deal with the Qdrant client. All from Week 3 of Phase 1.

### Day 5 (evening): Retrieval

- Write `src/retrieve.py`: takes a query string, returns top-K Pydantic objects with chunk text, title, and similarity score
- Manually test with 5-10 queries from the SciFact query set. Eyeball whether the retrieved abstracts are relevant.
- This is also where you decide: top-5? top-10? Start with top-5; you can change later.

### Weekend (1 session, 2-3 hours): API + Generate + Streamlit

- Write `src/generate.py`: given a claim and retrieved chunks, builds a prompt asking the LLM to assess the claim and cite chunks. Use structured output (Pydantic schema) so the response includes `verdict`, `supporting_chunks`, `reasoning`.
- Write `src/api.py`: FastAPI app with one POST endpoint `/verify`. Takes a claim, runs retrieve → generate → return.
- Add SQLite logging in `src/db.py`. Log every request: claim, retrieved chunk IDs, LLM response, latency, token counts.
- Build a 30-line Streamlit page that hits the API. Just a text input, a "Verify" button, and a result display showing the verdict, supporting chunks, and reasoning.
- End of weekend: you can paste a claim, click verify, see a structured answer with citations. End-to-end works.

---

## Week 2: Eval, Iteration, Documentation

### Day 1-2 (evenings): The eval script

This is the most interesting part of the project.

Write `src/eval.py` to:

1. Load the BeIR/SciFact query set (`load_dataset("BeIR/scifact", "queries")`) — ~300 claims
2. Load the relevance judgments (`load_dataset("BeIR/scifact-qrels")`) — which document IDs are relevant for each query
3. For each query:
   - Embed it
   - Retrieve top-K from Qdrant
   - Check: are the relevant doc IDs from the qrels in your top-K results?
4. Compute metrics:
   - **Recall@5**: of relevant docs, what fraction are in top-5? Average across queries.
   - **Recall@10**: same with top-10
   - **MRR (Mean Reciprocal Rank)**: how high up does the first relevant doc appear?
5. Print a summary report
6. Save the run results to SQLite with a run ID, git commit hash, and timestamp

The key trick for retrieval eval: you're comparing your retrieved doc IDs against a *known correct set*. You don't need an LLM judge. Pure set arithmetic.

### Day 3 (evening): First baseline run

- Run the eval. Get your first numbers.
- Expect something like: recall@5 around 0.60-0.75, depending on embedding model and chunking.
- Write the numbers down in a `results.md` file. This is your baseline.
- Compare to published numbers for BeIR/SciFact. Most leaderboards list these. You should be in the same ballpark for OpenAI `text-embedding-3-small`.

### Day 4-5 (evenings): One ablation

Pick ONE thing to vary and measure the effect. Suggestions:

**Option A: Compare embedding models.** Re-embed the corpus with `bge-small-en-v1.5` (via `sentence-transformers`, local). Run eval. Compare to baseline. Document the difference. Bonus: track inference time and cost.

**Option B: Add a re-ranker.** Retrieve top-20, re-rank with a cross-encoder like `BAAI/bge-reranker-base`, keep top-5. Run eval. Compare to no-rerank baseline.

**Option C: Chunking experiment.** Instead of one chunk per abstract, try splitting longer abstracts into sentence-level chunks. Re-ingest. Eval. Compare.

Pick the one that interests you most. Don't do all three — pick one, document it well. The point is to practice the *experiment → measure → document* loop, not to maximize improvements.

### Weekend (1 session, 1-2 hours): Write it up

Write a clean README:
- What the project is
- How it works (architecture diagram, even hand-drawn)
- How to run it
- Eval methodology
- Baseline results
- The one ablation you ran, with results
- What you'd improve with more time

Push to GitHub. Pin it on your profile. Optionally: write a one-paragraph LinkedIn post about it.

---

## What this project teaches you (and what it doesn't)

**Teaches:**
- Real benchmark eval workflow (you'll never confuse retrieval eval with generation eval again)
- Calibration of what "good" looks like (around 0.7 recall@5 for a domain-specific dense retriever)
- Async LLM and embedding code patterns
- Vector DB + payload pattern
- The discipline of measuring before tuning
- A small but real second portfolio piece

**Does NOT teach (saved for Phase 2):**
- Agent design
- Tool calling
- LLM-as-judge methodology
- Custom eval case design
- Failure mode tracking workflow
- Multi-model comparison framework
- Production hardening

That's by design. This project's job is to consolidate Weeks 1-4. Don't let it grow.

---

## How to talk about it in interviews

You'll be able to say:

*"As a calibration exercise, I built a scientific claim verification RAG against the BeIR/SciFact benchmark — about 5K abstracts and 300 queries with ground-truth relevance. With OpenAI's text-embedding-3-small and top-5 retrieval, I got recall@5 of 0.X, which is consistent with the published baseline for that embedding model. I ran one ablation comparing [your choice] and saw the metric move from 0.X to 0.Y. The biggest thing I learned was [specific insight]."*

That's a specific, defensible, calibrated answer. Not your signature project — but a real demonstration that you can work with public benchmarks, which most candidates can't.

---

## A few practical notes

**On dataset loading**: the BeIR datasets on Hugging Face sometimes have minor naming quirks. If `load_dataset("BeIR/scifact", "corpus")` doesn't work, check the dataset card on the HF website for the exact subset names. They sometimes use `default` or have separate repos for queries/corpus/qrels.

**On qrels format**: relevance judgments are typically a separate "qrels" set with `query-id`, `corpus-id`, `score` columns. Score of 1 = relevant, 0 = not relevant (or absent). You'll build a dict mapping query_id → set of relevant corpus_ids.

**On cost**: embedding 5K docs with text-embedding-3-small costs maybe 2 cents. Running 300 eval queries with GPT-4o-mini for generation costs maybe 20 cents. The full project including ablations: under $5. Don't worry about it.

**On local-only mode**: if you want to keep this fully free, use `bge-small-en-v1.5` (local) for embeddings and your LM Studio GPT OSS 20B for generation. You get to demonstrate cost discipline. Output a comparison table at the end: hosted vs local performance and cost.

**On testing**: don't write extensive tests. Maybe 5 sanity tests: ingestion works, retrieval returns expected shape, generate handles empty results, eval script runs without crash. Test discipline lives in Phase 2.

---

## Stop conditions

Stop the project when:

- Your README is written and pushed to GitHub
- You have a baseline number + one ablation result
- You can describe it in 60 seconds

Do NOT continue working on it just because you have ideas. The opportunity cost is the ML Sprint. Move on.

---

## When to think about it again

After Phase 2 ships, if you want a second polished portfolio piece, you can come back to this and extend it:
- Add an LLM judge for answer correctness (now that you've built one in Phase 2)
- Add the comparison between hosted and local models (formalized)
- Wrap it in a nicer UI
- Try it against a harder benchmark (HotpotQA for multi-hop reasoning)

But that's "after Phase 2 if you have time." For now: ship, document, move on.

Good luck. This is exactly the kind of small, focused, well-bounded project that keeps your hands warm while your brain absorbs the ML Sprint. Both halves of your prep will be stronger for it.
