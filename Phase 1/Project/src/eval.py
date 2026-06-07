"""
SciFact retrieval evaluation.

Loads queries (test split) + qrels (test split), runs retrieve_documents
on each query, computes recall@5, recall@10, MRR, prints summary,
saves run to SQLite.
"""

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, cast

from datasets import load_dataset

from db import init_db, log_eval_run
from retrieve import retrieve_documents

EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 10  # retrieve 10, derive recall@5 from the same result
RUN_NOTES = "baseline"

# ── 1. Load data ─────────────────────────────────────────────────────
print("Loading queries and qrels...")
queries = load_dataset("BeIR/scifact", "queries", split="queries")
qrels_raw = load_dataset("BeIR/scifact-qrels", split="test")

# ── 2. Build qrels lookup: query_id (int) → set of relevant corpus IDs (int)
qrels: dict[int, set[int]] = defaultdict(set)
for row in qrels_raw:
    r = cast(dict[str, Any], row)
    qrels[int(r["query-id"])].add(int(r["corpus-id"]))

print(f"  {len(queries)} queries, {len(qrels)} have ground-truth relevance judgments")

# ── 3. Eval loop ─────────────────────────────────────────────────────
recall_5_scores: list[float] = []
recall_10_scores: list[float] = []
rr_scores: list[float] = []
skipped = 0

wall_start = time.perf_counter()

for i, query in enumerate(queries):
    q = cast(dict[str, Any], query)
    query_id = int(q["_id"])
    claim = str(q["text"])

    # Skip queries with no ground truth
    if query_id not in qrels:
        skipped += 1
        continue

    truth = qrels[query_id]

    # Retrieve top-K
    results = retrieve_documents(claim, k=TOP_K)
    retrieved_ids = [chunk.id for chunk in results]

    # Recall@5
    top_5 = set(retrieved_ids[:5])
    recall_5_scores.append(len(truth & top_5) / len(truth))

    # Recall@10
    top_10 = set(retrieved_ids[:10])
    recall_10_scores.append(len(truth & top_10) / len(truth))

    # Reciprocal Rank (1-indexed)
    rr = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in truth:
            rr = 1.0 / rank
            break
    rr_scores.append(rr)

    # Progress
    if (i + 1) % 50 == 0:
        print(f"  processed {i + 1}/{len(queries)} queries...")

wall_seconds = time.perf_counter() - wall_start

# ── 4. Aggregate ─────────────────────────────────────────────────────
num_evaluated = len(recall_5_scores)
avg_recall_5 = sum(recall_5_scores) / num_evaluated
avg_recall_10 = sum(recall_10_scores) / num_evaluated
avg_mrr = sum(rr_scores) / num_evaluated

run_id = f"{RUN_NOTES}-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

# ── 5. Print summary ─────────────────────────────────────────────────
print(f"\nSciFact eval — {RUN_NOTES}")
print(f"  Queries:       {num_evaluated} (skipped {skipped} with no qrels)")
print(f"  Recall@5:      {avg_recall_5:.3f}")
print(f"  Recall@10:     {avg_recall_10:.3f}")
print(f"  MRR:           {avg_mrr:.3f}")
print(f"  Wall time:     {wall_seconds:.1f}s")
print(f"  Saved as run:  {run_id}")

# ── 6. Save to SQLite ────────────────────────────────────────────────
init_db()
log_eval_run(
    run_id=run_id,
    embedding_model=EMBEDDING_MODEL,
    top_k=TOP_K,
    num_queries=num_evaluated,
    recall_at_5=avg_recall_5,
    recall_at_10=avg_recall_10,
    mrr=avg_mrr,
    wall_time_seconds=wall_seconds,
    notes=RUN_NOTES,
)
