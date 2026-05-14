# Vector Databases

Vector databases are purpose-built for storing and searching embeddings at scale.

## Why Not Just Use a Loop?

For 100 documents, brute-force cosine similarity works fine. For 10 million, you need an index. Vector DBs use algorithms like HNSW to find approximate nearest neighbors in milliseconds.

## Qdrant

Qdrant is open-source and runs locally via Docker. It supports filtering, payloads, and multiple distance metrics.

## Pinecone

Pinecone is a managed service — no infrastructure to worry about. Good for prototyping but costs money at scale.

## When to Use What

- **Learning / prototyping**: Qdrant locally or even a simple NumPy array.
- **Production**: Qdrant Cloud, Pinecone, or pgvector if you already use Postgres.
