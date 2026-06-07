"""Hand-test: run retrieve on 5 real SciFact claims, print top score per query."""

from datasets import load_dataset
from retrieve import retrieve_documents

queries = load_dataset("BeIR/scifact", "queries", split="queries")

# Pick 5 claims spread across the dataset
test_indices = [0, 10, 50, 100, 200]

for idx in test_indices:
    claim = queries[idx]["text"]
    results = retrieve_documents(claim)

    print(f"\n{'='*70}")
    print(f"Query [{idx}]: {claim[:100]}...")
    for i, result in enumerate(results):
        print(f"  Rank : {i+1}")
        print(f"  Score : {result.score:.4f}")
        print(f"  Title : {result.title[:80]}")
        print(f"  Text : {result.text[:80]}")
        print(f"  ID    : {result.id}")
