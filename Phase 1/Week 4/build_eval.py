"""
Build an eval dataset for GDPR RAG retrieval.

Steps:
  1. Load & chunk the GDPR text identically to your notebook
  2. Sample 30 chunks at random
  3. Use GPT-4o-mini to generate one realistic question per chunk
  4. Save (question, gold_chunk_id) pairs to eval_dataset.json
"""

import json
import random
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── 1. Load & chunk (same settings as your notebook) ──────────────────
text = Path("gdpr.txt").read_text()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(text)
print(f"Total chunks: {len(chunks)}")

# ── 2. Sample 30 chunks ──────────────────────────────────────────────
random.seed(42)  # reproducible
sample_indices = random.sample(range(len(chunks)), 30)

# ── 3. Generate one question per sampled chunk ────────────────────────
eval_dataset = []

for i, chunk_id in enumerate(sample_indices):
    chunk_text = chunks[chunk_id]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal QA expert. Given a passage from the GDPR, "
                    "generate exactly ONE specific, realistic question that a "
                    "compliance officer might ask, whose answer is clearly "
                    "contained in the passage. Return ONLY the question, "
                    "nothing else."
                ),
            },
            {"role": "user", "content": chunk_text},
        ],
        temperature=0.7,
    )

    question = response.choices[0].message.content.strip()
    eval_dataset.append({
        "question": question,
        "gold_chunk_id": chunk_id,
        "gold_chunk_text": chunk_text[:200],  # preview for debugging
    })

    print(f"[{i+1}/30] Chunk {chunk_id} → {question[:80]}...")

# ── 4. Save ───────────────────────────────────────────────────────────
output_path = Path("eval_dataset.json")
output_path.write_text(json.dumps(eval_dataset, indent=2))
print(f"\n✅ Saved {len(eval_dataset)} eval pairs to {output_path}")
