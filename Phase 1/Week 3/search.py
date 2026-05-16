import typer
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_qdrant = QdrantClient(url = "http://localhost:6333")
app = typer.Typer()

@app.command()
def index(directory: str):
    content = []
    file_names = []
    for file in Path(directory).glob("**/*.md"):
        content.append(file.read_text())
        file_names.append(file)
    
    
    embedding = client.embeddings.create(model="text-embedding-3-small", input=content)
    client_qdrant.delete_collection("my-docs") 
    client_qdrant.create_collection(
        collection_name="my-docs",
        vectors_config=VectorParams(size=len(embedding.data[0].embedding), distance=Distance.COSINE)
    )

    points = []
    for i, file in enumerate(file_names):
        points.append(PointStruct(id=i, vector=embedding.data[i].embedding, payload={"text": content[i][:200], "filename": file.name}))

    client_qdrant.upsert(collection_name="my-docs", points=points)
    print(f"Indexed {len(file_names)} files.")

    
@app.command()
def query(search_query: str):
    print(f"Querying for {search_query}")
    search_embedding = client.embeddings.create(model="text-embedding-3-small", input=search_query)
    search_result = client_qdrant.query_points(
        collection_name="my-docs",
        query=search_embedding.data[0].embedding,
        limit=5
    )    
    for point in search_result.points:
        print(point.score, point.payload["filename"])    
        
if __name__ == "__main__":
    app()