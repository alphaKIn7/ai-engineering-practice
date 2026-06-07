"""
Run retrieval evaluation: recall@5 on the GDPR RAG system.

Loads eval_dataset.json, queries Qdrant for each question,
checks if the gold_chunk_id is in the top-5 results.
"""

import json
import os
from pathlib import Path

from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_qdrant = QdrantClient(url="http://localhost:6333")

COLLECTION = "gdpr"
K = 5

# ── Load eval dataset ────────────────────────────────────────────────
eval_data = json.loads(Path("eval_dataset.json").read_text())
print(f"Loaded {len(eval_data)} eval questions\n")

# ── Run evaluation ───────────────────────────────────────────────────
hits = 0

for i, item in enumerate(eval_data):
    question = item["question"]
    gold_id = item["gold_chunk_id"]

    # Embed the question
    emb = client.embeddings.create(model="text-embedding-3-small", input=question)

    # Retrieve top-K from Qdrant
    result = client_qdrant.query_points(
        collection_name=COLLECTION,
        query=emb.data[0].embedding,
        limit=K,
    )

    retrieved_ids = [point.id for point in result.points]
    found = gold_id in retrieved_ids

    if found:
        hits += 1
        status = "✅"
    else:
        status = "❌"

    print(f"{status} [{i+1}/{len(eval_data)}] gold={gold_id}  retrieved={retrieved_ids}  q=\"{question[:60]}...\"")

# ── Results ───────────────────────────────────────────────────────────
recall = hits / len(eval_data)
print(f"\n{'='*50}")
print(f"Recall@{K}: {hits}/{len(eval_data)} = {recall:.1%}")
print(f"{'='*50}")
