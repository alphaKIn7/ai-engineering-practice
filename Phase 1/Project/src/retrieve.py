import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from models import RetrievedChunk


# 1. Clients & env
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(url="http://localhost:6333")

COLLECTION = "scifact"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5

def retrieve_documents(query: str, k: int = TOP_K) -> list[RetrievedChunk]:
    # 2. Embed the query
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )

    # 3. Query Qdrant
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION,
        query=response.data[0].embedding,
        limit=k,
    )

    # 4. Transform ScoredPoints → RetrievedChunks
    return [
        RetrievedChunk(
            id=int(point.id),
            title=point.payload["title"],
            text=point.payload["text"],
            score=point.score,
        )
        for point in search_result.points
        if point.payload is not None
    ]


if __name__ == "__main__":
    for chunk in retrieve_documents("What is the capital of France?"):
        print(f"\nTitle: {chunk.title}\n")
        print(f"Text: {chunk.text}\n")
        print(f"Score: {chunk.score}\n")
        print(f"ID: {chunk.id}\n")
    