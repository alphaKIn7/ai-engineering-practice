# Embeddings Basics

Embeddings are dense vector representations of text. They capture semantic meaning in a way that allows mathematical comparison.

## Why Embeddings Matter

Traditional keyword search fails when users use different words for the same concept. Embeddings solve this by mapping semantically similar text to nearby points in vector space.

## Key Properties

- **Dimensionality**: Common models produce 384 to 1536 dimensions.
- **Cosine similarity**: The standard way to compare two embeddings. Values range from -1 to 1, where 1 means identical meaning.
- **Dense vs sparse**: Unlike bag-of-words (sparse), embeddings pack meaning into every dimension.

## Popular Models

- `text-embedding-3-small` (OpenAI) — 1536 dims, cheap, good quality.
- `bge-small-en-v1.5` — 384 dims, runs locally, surprisingly strong.
