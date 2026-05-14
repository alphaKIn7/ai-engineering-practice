# AI Engineer Phase 1 — Tutor Notes

A 6-week guided walkthrough of the foundations you need before building the signature project. Treat this like a textbook with a friend explaining things — read the "why this matters" section first, then work through the code, then do the exercise.

**How to use this:**
- Don't skip the intuition sections. Most people skip them and that's why they can't explain things in interviews.
- Type out every code example yourself. Don't copy-paste. Muscle memory matters.
- Do the mini-project at the end of each week before moving on.
- Keep a `notes.md` per week with things that surprised you.

---

## Week 1: Modern Python Reset

### Why this matters

Power Automate taught you workflow thinking, but AI engineering interviews assume you can write clean, typed, async Python without thinking about it. The bar isn't "can you code" — it's "can you write code an interviewer immediately trusts."

Three things separate a Python beginner from someone who looks senior:
1. Type hints everywhere (so code is self-documenting)
2. Pydantic for data validation (so bad data fails loudly, fast)
3. Async fluency (so you can call APIs in parallel without crying)

### Type hints — the 80/20

Python is dynamically typed, but in 2026 nobody writes production Python without type hints. They're not enforced at runtime, but tools like `mypy` and `pyright` catch bugs before they happen, and IDEs use them for autocomplete.

```python
# Bad — what does this return? What's `items`?
def total(items):
    return sum(i["price"] * i["qty"] for i in items)

# Good — anyone can read this
from typing import TypedDict

class LineItem(TypedDict):
    price: float
    qty: int

def total(items: list[LineItem]) -> float:
    return sum(i["price"] * i["qty"] for i in items)
```

Key syntax to memorize:
- `list[int]`, `dict[str, int]`, `tuple[int, str]` — built-in generics (Python 3.9+)
- `int | None` — union types (Python 3.10+), replaces the old `Optional[int]`
- `Callable[[int, str], bool]` — for function arguments
- `T = TypeVar("T")` for generic functions

### Pydantic v2 — your new best friend

Pydantic is *the* library for data validation in Python AI code. Every LLM framework uses it. You'll use it constantly for:
- Validating LLM outputs (the LLM said it returned JSON — did it actually?)
- Defining API request/response schemas
- Configuration loading

```python
from pydantic import BaseModel, Field
from datetime import datetime

class Receipt(BaseModel):
    vendor: str
    amount: float = Field(gt=0, description="Amount in USD, must be positive")
    date: datetime
    category: str | None = None
    
# This works — Pydantic coerces "2026-05-13" to datetime automatically
r = Receipt(vendor="Starbucks", amount=4.50, date="2026-05-13")
print(r.amount)  # 4.5

# This fails loudly — exactly what you want
try:
    bad = Receipt(vendor="X", amount=-5, date="invalid")
except Exception as e:
    print(e)  # Shows exactly which fields failed and why
```

The killer feature for AI work: `model_validate_json()` parses an LLM's JSON output and validates it in one call. If the LLM hallucinated a field, you find out immediately.

```python
llm_output = '{"vendor": "Uber", "amount": 23.50, "date": "2026-05-13"}'
receipt = Receipt.model_validate_json(llm_output)  # Either works or raises
```

### Async basics

Synchronous code waits for each operation to finish. Async lets you do other work while waiting. For LLM calls (which take seconds), this is huge — calling 10 LLMs in parallel takes the same time as calling 1.

```python
import asyncio
import httpx

# Synchronous — slow
def fetch_sync(urls: list[str]) -> list[str]:
    results = []
    with httpx.Client() as client:
        for url in urls:
            results.append(client.get(url).text)
    return results

# Async — fast
async def fetch_async(urls: list[str]) -> list[str]:
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [r.text for r in responses]

# Run an async function
results = asyncio.run(fetch_async(["https://example.com"] * 10))
```

Three rules:
1. `async def` defines an async function (called a "coroutine")
2. `await` is how you wait for one (only legal inside `async def`)
3. `asyncio.gather(*tasks)` runs many in parallel

If this feels weird coming from Power Automate, here's the mental model: in Power Automate, parallel branches run independently and you wait at the join. `asyncio.gather` is the join.

### Tooling: `uv` instead of pip

`uv` is the modern Python package manager — 10-100x faster than pip, handles virtual envs automatically.

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Start a new project
uv init my-project
cd my-project

# Add packages
uv add pydantic httpx anthropic

# Run your code
uv run python main.py
```

That's it. No more `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.

### Week 1 mini-project: Markdown Notes Indexer

Build a CLI tool that:
1. Takes a directory of `.md` files
2. Extracts headings, word count, last-modified time
3. Stores results in a Pydantic model
4. Outputs a JSON summary

Requirements:
- Use type hints everywhere
- Use pydantic for the output schema
- Use `pathlib` not `os.path`
- Use `asyncio.gather` to process files in parallel (even though file I/O is fast — practice the pattern)

```python
# Starter
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime
import asyncio
import aiofiles  # uv add aiofiles

class NoteSummary(BaseModel):
    path: str
    headings: list[str]
    word_count: int
    last_modified: datetime

async def summarize_note(path: Path) -> NoteSummary:
    # Your code here
    ...

async def main(directory: Path) -> list[NoteSummary]:
    files = list(directory.glob("**/*.md"))
    return await asyncio.gather(*[summarize_note(f) for f in files])
```

### Common pitfalls

- **Mutable default arguments**: `def f(x: list = [])` is a trap. Use `def f(x: list | None = None)` and check inside.
- **Forgetting `await`**: `result = some_async_function()` gives you a coroutine object, not the result. Always `await` it.
- **Mixing sync and async**: You can't `await` inside a regular function. You can't call `asyncio.run` from inside another `asyncio.run`.

---

## Week 2: APIs & LLM Basics

### Why this matters

Every AI system is a series of API calls dressed up in fancy abstractions. If you understand the raw API call, you understand everything. Frameworks like LangChain hide this from you — which is why people who only know LangChain struggle in interviews. You'll skip the framework for now.

### FastAPI essentials

FastAPI is the Python web framework. It uses Pydantic, has async built-in, and auto-generates docs. You'll use it to expose your AI services as APIs.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    user_id: str

class ChatResponse(BaseModel):
    reply: str
    tokens_used: int

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # Your LLM call here
    return ChatResponse(reply=f"You said: {req.message}", tokens_used=10)

# Run with: uv run fastapi dev main.py
# Auto-generated docs at http://localhost:8000/docs
```

The magic: FastAPI uses your Pydantic models to validate incoming JSON, generate OpenAPI specs, and produce Swagger UI for free.

### Calling LLM APIs

You'll call three providers in this section. Get API keys for at least Anthropic and OpenAI. Store them in a `.env` file and load with `python-dotenv`.

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

**Anthropic (Claude):**

```python
from anthropic import Anthropic

client = Anthropic()  # auto-reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain RAG in one sentence."}
    ]
)
print(response.content[0].text)
print(response.usage)  # input_tokens, output_tokens
```

**OpenAI:**

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain RAG in one sentence."}
    ]
)
print(response.choices[0].message.content)
print(response.usage)
```

**Your local LM Studio (this is your edge):**

LM Studio exposes an OpenAI-compatible API on `localhost:1234`. So:

```python
from openai import OpenAI

local_client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"  # required arg but ignored
)

response = local_client.chat.completions.create(
    model="openai/gpt-oss-20b",  # whatever you've loaded
    messages=[
        {"role": "user", "content": "Explain RAG in one sentence."}
    ]
)
print(response.choices[0].message.content)
```

In interviews, mention this casually: *"I prototype against my local 20B model first to iterate fast and stay private, then swap to a hosted model for production."* That's a senior signal.

### Prompting fundamentals

Three patterns to know cold:

**1. System prompt + user message** — for setting persona/rules:

```python
messages = [
    {"role": "system", "content": "You are a strict JSON-only response generator. Never use markdown or prose."},
    {"role": "user", "content": "List 3 colors with hex codes."}
]
```

**2. Few-shot** — when zero-shot isn't reliable enough:

```python
messages = [
    {"role": "user", "content": "Classify: 'I love this!'"},
    {"role": "assistant", "content": "positive"},
    {"role": "user", "content": "Classify: 'This is garbage.'"},
    {"role": "assistant", "content": "negative"},
    {"role": "user", "content": "Classify: 'It's fine I guess.'"},
]
```

**3. Structured outputs** — force JSON conforming to a schema:

```python
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str  # "positive" | "negative" | "neutral"
    confidence: float
    reasoning: str

# OpenAI structured outputs
response = client.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "How do you feel about Mondays?"}],
    response_format=Sentiment,
)
result: Sentiment = response.choices[0].message.parsed
```

This is *much* better than asking the model to "return JSON" and hoping. The provider enforces it.

### Week 2 mini-project: Unified LLM Wrapper

Build a library that exposes one function:

```python
async def complete(
    prompt: str,
    provider: Literal["anthropic", "openai", "local"],
    model: str | None = None,
    system: str | None = None,
    response_schema: type[BaseModel] | None = None,
) -> CompletionResult:
    ...

class CompletionResult(BaseModel):
    text: str
    parsed: BaseModel | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    provider: str
    model: str
```

Requirements:
- Logs every call to a SQLite database (using `sqlmodel` — it's pydantic + SQL)
- Computes cost from a hard-coded price table per model
- Times the call with `time.perf_counter()`
- Handles structured outputs for at least OpenAI

This becomes your foundation for every project after this. Don't skip the logging — it's how you debug everything later.

### Common pitfalls

- **Don't commit API keys**. `.gitignore` your `.env`. Use `python-dotenv` to load it.
- **Tokens ≠ words**. ~1 token ≈ 0.75 words for English. Use `tiktoken` (OpenAI) to count precisely.
- **Rate limits exist**. Wrap calls in retry logic (`tenacity` library: `@retry(stop=stop_after_attempt(3), wait=wait_exponential())`)
- **Streaming is its own thing**. Don't worry about it Week 2 — come back later.

---

## Week 3: Embeddings & Vector Search

### Why this matters

Every RAG system, every semantic search, every "find similar documents" feature uses embeddings. If you can't explain what an embedding *is* without hand-waving, you'll lose interview points.

### What is an embedding actually?

An embedding is a fixed-length list of numbers that represents the *meaning* of a piece of text. Two texts with similar meaning will have similar numbers.

```
"dog"           → [0.2, -0.5, 0.8, ..., 0.1]   (1536 numbers)
"puppy"         → [0.21, -0.49, 0.81, ..., 0.09]  ← very close to "dog"
"refrigerator"  → [-0.7, 0.3, -0.2, ..., 0.6]   ← far from "dog"
```

The model that creates these (e.g., `text-embedding-3-small`) has been trained so that semantically related text ends up near each other in this high-dimensional space.

"Similar" is measured with **cosine similarity** — the cosine of the angle between two vectors. Range: -1 (opposite) to 1 (identical). For practical purposes, anything above ~0.7 is "very similar" for OpenAI embeddings; the exact threshold depends on the model.

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Two embeddings from your model
print(cosine_similarity(emb_dog, emb_puppy))      # ~0.85
print(cosine_similarity(emb_dog, emb_fridge))     # ~0.15
```

### Getting embeddings

```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["dog", "puppy", "refrigerator"]
)
embeddings = [e.embedding for e in response.data]  # list of 1536-dim vectors
```

You can also run embedding models locally — `bge-small-en-v1.5` is a strong, tiny model. Use `sentence-transformers`:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = model.encode(["dog", "puppy", "refrigerator"])
```

Running locally on your M4: instant, free, private. Worth mentioning in interviews.

### Vector databases — why you need one

If you have 10 documents, you can compute cosine similarity against all of them in a loop. If you have 10 million, that's hopeless. Vector databases (Qdrant, Pinecone, pgvector) use indexes (HNSW is the standard) that find the top-K nearest neighbors in milliseconds even at billion-scale.

### Qdrant locally

Qdrant is open-source, runs in Docker, and is production-grade. Set it up:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

Then in Python:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="http://localhost:6333")

# Create a collection
client.create_collection(
    collection_name="notes",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Insert points
client.upsert(
    collection_name="notes",
    points=[
        PointStruct(
            id=1,
            vector=embedding_for_dog,  # list[float] of length 1536
            payload={"text": "dog", "category": "animal"}
        ),
        PointStruct(
            id=2,
            vector=embedding_for_puppy,
            payload={"text": "puppy", "category": "animal"}
        ),
    ]
)

# Search
results = client.search(
    collection_name="notes",
    query_vector=embedding_for_query,
    limit=5,
    query_filter=None,  # can filter by payload, e.g., category="animal"
)
for r in results:
    print(r.score, r.payload["text"])
```

The `payload` is metadata you store alongside the vector — usually the original text and any filterable fields.

### Week 3 mini-project: Semantic Search CLI

Build a CLI that:
1. Indexes a folder of text files (one embedding per file, or per paragraph if you want bonus points)
2. Takes a query, embeds it, returns top-5 matches with scores
3. Compares two embedding models: OpenAI's `text-embedding-3-small` vs local `bge-small-en-v1.5`
4. Writes a small report: which model felt better for *your* corpus and why

Use Qdrant for storage. Use Click or Typer for the CLI.

```python
# Usage
$ uv run python search.py index ./my_notes/
Indexed 47 documents.

$ uv run python search.py query "how do I handle errors"
[0.81] retry_logic.txt
[0.76] error_handling.md
[0.65] debugging_tips.md
```

### Common pitfalls

- **Don't compare embeddings across models**. An OpenAI embedding has no meaningful relationship to a BGE embedding.
- **Dimension matters for cost/speed, not always quality**. `text-embedding-3-small` (1536-d) is fine for almost everything. `3-large` (3072-d) is rarely worth the cost.
- **Normalize for cosine, or don't, but be consistent**. Many models output already-normalized vectors. Qdrant handles this for you with `Distance.COSINE`.
- **Embed once, search many**. Caching embeddings is critical — re-embedding the same text wastes money and time.

---

## Week 4: RAG Properly

### Why this matters

RAG (Retrieval-Augmented Generation) is the most common pattern in production AI. Almost every interview will ask about it. The "what" is easy; the "why this chunking" and "what failed" is where you stand out.

### What RAG actually is

A normal LLM call: `prompt → LLM → answer`. Limited to what the LLM was trained on.

RAG: `prompt → retrieve relevant docs → stuff them into the prompt → LLM → answer`. Now the LLM can answer about your private data.

```
User question: "What's our refund policy for international orders?"
                          ↓
              Embed question, search vector DB
                          ↓
         Top-3 relevant chunks from policy docs
                          ↓
   Prompt: "Using only this context: [chunks]
            Answer: What's our refund policy..."
                          ↓
                       LLM answer
```

That's it. The rest is engineering details — and the details are where it lives or dies.

### Chunking strategies

You can't put a whole 500-page PDF in a prompt. You split it into chunks, embed each chunk, retrieve the relevant ones.

**Fixed-size chunking**: Split every N characters/tokens.
- Pros: Simple, predictable.
- Cons: Breaks sentences and ideas mid-thought.
- When to use: Highly repetitive content (logs, transcripts).

**Recursive chunking**: Split on `\n\n`, then `\n`, then `.`, then space — preserving structure where possible.
- Pros: Respects natural boundaries.
- Cons: Variable chunk sizes.
- When to use: Default choice for most documents.

**Semantic chunking**: Split where the topic changes (using embedding similarity between sentences).
- Pros: Each chunk is one coherent idea.
- Cons: Expensive to compute, complex.
- When to use: Very heterogeneous content; usually overkill.

**Structure-aware chunking**: For markdown/HTML/code, split by sections/functions.
- Pros: Chunks have semantic meaning.
- Cons: Format-specific.
- When to use: Technical docs, code, structured content.

Use `langchain-text-splitters` even if you don't use the rest of LangChain — the splitters are best-in-class:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # target size in characters
    chunk_overlap=50,      # overlap so context isn't lost at boundaries
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(long_document)
```

The `chunk_overlap` matters more than people realize: if a sentence describes something at the end of chunk A and elaborates at the start of chunk B, you lose the connection unless they overlap.

### Retrieval — beyond just top-K

Naive RAG: embed query, find top-5 similar chunks, stuff into prompt. This often fails because:

1. **The most semantically similar chunk isn't always the most useful**
2. **Synonyms get high cosine scores but might be off-topic**
3. **Exact terms (product names, IDs, dates) need keyword matching, not embeddings**

Solutions:

**Hybrid search**: Combine dense (embedding) and sparse (BM25/keyword) search. Average or weight the scores.

```python
# Qdrant supports hybrid search natively via sparse vectors
# Or use Reciprocal Rank Fusion (RRF) to combine results from two searches

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return dict(sorted(scores.items(), key=lambda x: -x[1]))
```

**Re-ranking**: Retrieve top-20 with cheap embedding search, then re-rank with a more expensive cross-encoder that scores `(query, chunk)` directly.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")
pairs = [[query, chunk.text] for chunk in initial_results]
scores = reranker.predict(pairs)
reranked = sorted(zip(initial_results, scores), key=lambda x: -x[1])[:5]
```

Cross-encoders are slower but more accurate because they let the model attend to the query and chunk together rather than computing them separately.

### The prompt template

```python
RAG_PROMPT = """You are an assistant answering questions using only the provided context.

If the context doesn't contain the answer, say "I don't know based on the provided context." Do not make up information.

<context>
{context}
</context>

<question>
{question}
</question>

Provide a clear, concise answer with citations to the relevant context chunks (e.g., [chunk 1])."""

def build_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(
        f"[chunk {i+1}]\n{c.text}" for i, c in enumerate(chunks)
    )
    return RAG_PROMPT.format(context=context, question=question)
```

Two things to defend in interviews:
1. **Anti-hallucination instruction** ("say I don't know")
2. **Citations** so you can verify what the answer came from

### Week 4 mini-project: RAG over Policy Docs

Pick a public document set (good options: a country's tax code, GDPR text, an open-source project's docs). Build:

1. An ingestion script: reads PDFs/markdown, chunks them, embeds, stores in Qdrant
2. A query function: takes a question, retrieves top chunks, builds prompt, calls LLM
3. **A small eval**: write 20 question-answer pairs by hand. For each, check:
   - Did the right chunk get retrieved? (retrieval@5)
   - Did the LLM answer correctly? (faithfulness — does the answer match the chunk?)

Try two chunking strategies. Compare. Write a one-page README explaining your choices.

### Common pitfalls

- **Chunks too big** → relevant info gets diluted, prompts get expensive
- **Chunks too small** → context fragments, answers lack coherence
- **No metadata filtering** → retrieving from irrelevant documents (always filter by source, date, type if you can)
- **Not handling "no results"** → if nothing's above similarity threshold, say so, don't force an answer
- **Forgetting to deduplicate** → near-identical chunks all retrieved together

---

## Week 5: Agents & Tool Calling

### Why this matters

This is where your Power Platform background gives you an edge. An "agent" is just a workflow where the LLM decides the next step instead of a human-defined branch. You already think this way.

### What is an agent, really

A simple LLM call: one prompt in, one response out.

An agent: a loop where the LLM can call **tools** (functions you provide), observe results, and decide what to do next.

```
User: "What's the weather in Delhi and Milan?"
  ↓
LLM: "I'll call get_weather('Delhi')"
  ↓ (you execute the tool)
Tool result: "32°C, sunny"
  ↓
LLM: "I'll call get_weather('Milan')"
  ↓
Tool result: "18°C, rainy"
  ↓
LLM: "Delhi is 32°C and sunny; Milan is 18°C and rainy."
```

In Power Automate terms: the LLM is the trigger + decision logic, tools are the connectors, and the loop is your Do Until pattern.

### Tool calling mechanics

Modern APIs let you declare tools as JSON schemas. The LLM either responds with text *or* with a tool call. You execute the tool, send back the result, loop.

```python
from anthropic import Anthropic
import json

client = Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    }
]

def get_weather(city: str) -> str:
    # In real code, call a weather API. Here, fake it.
    return f"{city}: 25°C, partly cloudy"

def run_agent(user_message: str, max_iterations: int = 10) -> str:
    messages = [{"role": "user", "content": user_message}]
    
    for iteration in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )
        
        # Append assistant's response to history
        messages.append({"role": "assistant", "content": response.content})
        
        # If LLM didn't call a tool, we're done
        if response.stop_reason != "tool_use":
            return "".join(
                block.text for block in response.content if block.type == "text"
            )
        
        # Execute each tool call and append results
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "get_weather":
                    result = get_weather(**block.input)
                else:
                    result = f"Unknown tool: {block.name}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        
        messages.append({"role": "user", "content": tool_results})
    
    return "Max iterations reached"
```

That's a working agent. Read it carefully — every "agent framework" is doing some version of this loop, often with bugs.

### Patterns to know

**ReAct (Reason + Act)**: The LLM verbalizes its reasoning before each action. Often built into the system prompt:

```
"Think step by step. Before each tool call, explain why you're calling it. After each result, explain what you learned."
```

**Plan-and-Execute**: LLM first writes a plan (steps 1, 2, 3...), then executes each. Better for complex multi-step tasks.

**Reflection**: After completing a task, the LLM critiques its own output and tries again. Expensive but improves quality.

For interviews, you should be able to whiteboard the basic agent loop above. Read Anthropic's "Building effective agents" post — it's the canonical reference and it's deliberately framework-agnostic.

### Tight tool boundaries

The Reddit post you screenshotted called this out: *"building a small agent with tight tool boundaries + logging + evals tends to stand out."*

Tight boundaries means:
- Each tool does **one thing**, not five
- Tool inputs are strictly validated (Pydantic)
- Tools return structured data, not free-form strings when possible
- Tools have clear, specific descriptions (the LLM picks based on these)

Bad: `do_database_stuff(query: str)`
Good: `search_customers_by_email(email: str)`, `get_order_details(order_id: str)`

### Week 5 mini-project: Research Agent

Build an agent that, given a research question, can:
1. Search the web (use SerpAPI, Tavily, or DuckDuckGo)
2. Fetch a URL's content
3. Summarize text

Loop until it has a complete answer, then return a summary with sources.

Requirements:
- 3 tools, well-typed
- Logs every tool call to your SQLite logger from Week 2
- A max iteration limit (don't accidentally burn $100)
- Returns sources cited

Stretch: add a `reflect` step that asks the LLM "is your answer complete?" before returning.

### Common pitfalls

- **Infinite loops**: Always have a max iteration count.
- **Tool result too big**: Truncate or summarize before feeding back to the LLM. A 50K-token tool result will blow your context.
- **Silent tool failures**: If a tool throws, return an error message to the LLM as the tool result. Let *it* decide what to do.
- **Too many tools**: Past ~10 tools, LLMs get confused. Group related ones into a single tool with a `action` parameter, or use multiple agents.
- **Not logging tool inputs/outputs**: You'll be unable to debug anything.

---

## Week 6: Evals & Observability

### Why this matters

The Reddit guide said it clearly: *"Evals are everything... being able to talk about hallucination rates and automated eval frameworks is what actually gets you the offer."*

Most candidates can build a RAG demo. Almost none can show: "Here's the eval I built, here are the failure modes I caught, here's how I improved accuracy from 67% to 84%." That's the signal.

### Why evals are hard

Traditional software: you write a unit test, output matches expected, ✓ or ✗.

LLMs: output is "The capital of France is Paris." Expected was "Paris is the capital of France." Both correct. String match fails. Now what?

Three eval approaches:

1. **Exact match / regex** — for structured outputs (JSON schemas, classifications)
2. **LLM-as-judge** — use a stronger LLM to score answers
3. **Human review** — slow but gold-standard; use for the hard cases

You'll combine all three.

### Building a golden dataset

Before any code: write down 30–50 example inputs with expected outputs (or expected qualities). Cover:

- **Happy path**: typical queries, correct answers
- **Edge cases**: empty inputs, very long inputs, ambiguous queries
- **Adversarial**: prompt injection attempts, off-topic queries, queries with no answer in your data
- **Regressions**: things that previously broke; lock them down with tests

Format:

```python
from pydantic import BaseModel

class EvalCase(BaseModel):
    id: str
    input: str
    expected_output: str | None  # for exact match cases
    expected_qualities: list[str]  # for LLM-judge cases, e.g., ["mentions refund window", "cites policy doc"]
    category: str  # "happy_path" | "edge_case" | "adversarial" | "regression"
    notes: str = ""
```

Store as JSONL or YAML, version control it. This is your test suite.

### LLM-as-judge

For open-ended outputs, use another LLM to grade. Critical: the judge prompt matters as much as the system prompt.

```python
JUDGE_PROMPT = """You are evaluating an AI assistant's answer.

Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}

Score the candidate answer on these dimensions (0-5 each):

1. **Accuracy**: Does it match the reference factually?
2. **Completeness**: Does it cover all key points?
3. **Faithfulness**: Does it avoid claims not in the reference?

Respond with JSON: {{"accuracy": int, "completeness": int, "faithfulness": int, "reasoning": str}}"""
```

Two principles:
- **Always use a stronger model as judge than the model being judged** (e.g., judge GPT-4o-mini outputs with Claude Sonnet)
- **Calibrate by spot-checking**: review 10 of the judge's scores manually. If you disagree often, fix the prompt.

### Putting it together: an eval harness

```python
import asyncio
from pydantic import BaseModel
from datetime import datetime

class EvalResult(BaseModel):
    case_id: str
    candidate_output: str
    scores: dict[str, float]
    passed: bool
    latency_ms: int
    cost_usd: float
    timestamp: datetime

async def run_eval(
    cases: list[EvalCase],
    system_under_test: callable,  # your RAG/agent function
    judge: callable,              # LLM-judge function
) -> list[EvalResult]:
    
    async def evaluate_one(case: EvalCase) -> EvalResult:
        start = time.perf_counter()
        output = await system_under_test(case.input)
        latency = int((time.perf_counter() - start) * 1000)
        
        scores = await judge(case, output)
        passed = all(s >= 3 for s in scores.values())
        
        return EvalResult(
            case_id=case.id,
            candidate_output=output.text,
            scores=scores,
            passed=passed,
            latency_ms=latency,
            cost_usd=output.cost_usd,
            timestamp=datetime.utcnow(),
        )
    
    return await asyncio.gather(*[evaluate_one(c) for c in cases])
```

Run this on every change. Track pass rates over time. **This is what you'll demo in interviews.**

### Observability for LLM apps

Beyond evals, you need to see what's happening in production. Log every call with:

- Inputs (truncated if huge)
- Outputs (full)
- Model, prompt version, temperature
- Latency, tokens, cost
- User/session ID
- Any tool calls made
- Errors

Two paths:
- **DIY**: SQLite + a simple Streamlit dashboard. Fine for portfolio projects.
- **Hosted**: Langfuse, Helicone, Phoenix (Arize). Real teams use these. Langfuse has a generous free tier and works great for portfolio projects.

Mention specific tools by name in interviews. It signals you've seen production.

### Tracking failure modes

For each failure your eval catches, log:
- Input that broke it
- What it produced
- What it should have produced
- Root cause hypothesis
- Fix applied
- Verified fixed (eval passes now)

This becomes your "failure modes" interview story. Two or three of these and you're golden.

### Week 6 mini-project: Eval Harness for Week 5 Agent

Apply everything to your research agent:

1. Write 30 eval cases covering all categories
2. Build the eval harness (don't reach for a library — write it)
3. Run it. Find at least 5 failures.
4. For each failure, write a one-line failure mode + your fix
5. Re-run. Show the pass rate going up.
6. Produce a one-page report: pass rate by category, average latency, average cost, failure modes corrected

Bonus: set up Langfuse and pipe your logs into it. Take screenshots — they're great for portfolio README.

### Common pitfalls

- **Eval dataset too small**: 5 cases tells you nothing. Aim for 30+ minimum.
- **Eval dataset only happy path**: catches nothing useful.
- **Judge model = candidate model**: same biases, useless.
- **Not versioning prompts**: you'll change a prompt, see worse evals, and have no idea what you changed.
- **Optimizing to the eval set**: classic ML mistake. Keep a separate "held out" set you don't iterate against.

---

## Closing notes

By end of Week 6, you should have:

- ~5 small GitHub repos demonstrating each capability
- A logging library you reuse across them
- A working eval harness you can apply to any LLM system
- Notes on every concept that surprised you
- A clear sense of what you don't know yet (write this down — it's the bridge to Phase 2)

**Before moving to Phase 2**, ask yourself:

- Can I explain cosine similarity to a 10-year-old?
- Can I whiteboard the agent loop from memory?
- Can I name three chunking strategies and when to use each?
- Can I name three reasons RAG fails in production?
- Have I actually built an eval harness, not just read about one?

If yes to all → start Phase 2 (the Approval Agent).
If no → spend an extra week on the weakest area. There's no prize for rushing.

---

## Resources I'd actually recommend (skip the rest)

- **Anthropic's "Building effective agents"** — best agent reference, period
- **Hamel Husain's blog**, especially "Your AI product needs evals" — eval philosophy
- **Lilian Weng's blog** (lilianweng.github.io) — deeper dives into agents, prompt engineering
- **OpenAI Cookbook on GitHub** — practical code examples
- **Eugene Yan's "Patterns for LLM Applications"** — the architecture patterns post
- **Chip Huyen's "Designing ML Systems"** book — if you want broader ML systems context

Avoid: any course charging $500+ promising AI engineer jobs, most YouTube "build a $X startup with LangChain" tutorials, anything that hides the underlying API calls from you.
