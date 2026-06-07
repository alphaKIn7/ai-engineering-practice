import os
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# 1. Clients & env
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(url="http://localhost:6333")

COLLECTION = "scifact"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIM = 1536
BATCH_SIZE = 100

# 2. Load corpus
ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
print(f"Loaded {len(ds)} documents from BeIR/SciFact corpus")

# 3. Drop + recreate collection
if qdrant_client.collection_exists(COLLECTION):
    qdrant_client.delete_collection(COLLECTION)
qdrant_client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
)
print(f"Created collection '{COLLECTION}' ({VECTOR_DIM}-dim, cosine)")

# 4. Batch loop: embed + upsert
total = len(ds)

for start in range(0, total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total)
    batch = ds[start:end]

    # Build the strings to embed: "title: … \n\n text: …"
    texts = []
    for title, text in zip(batch["title"], batch["text"]):
        if title:
            texts.append(f"{title}\n\n{text}")
        else:
            texts.append(text)

    # Embed the batch
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)

    # Build PointStructs — use the corpus _id (cast to int) as the point ID
    points = []
    for i, (doc_id, title, text) in enumerate(
        zip(batch["_id"], batch["title"], batch["text"])
    ):
        points.append(
            PointStruct(
                id=int(doc_id),
                vector=response.data[i].embedding,
                payload={"title": title, "text": text},
            )
        )

    # Upsert
    qdrant_client.upsert(collection_name=COLLECTION, points=points)
    print(f"  upserted {start+1}–{end} / {total}")

# 5. Final count
info = qdrant_client.get_collection(COLLECTION)
print(f"\nDone. Collection '{COLLECTION}' has {info.points_count} points.")