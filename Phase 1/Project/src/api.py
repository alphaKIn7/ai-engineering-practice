import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from db import init_db, log_verification
from generate import verify_claim
from models import VerifyRequest, VerifyResponse
from retrieve import retrieve_documents


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    init_db()
    yield


app = FastAPI(title="SciFact Claim Verifier", lifespan=lifespan)


@app.post("/verify")
def verify(req: VerifyRequest) -> VerifyResponse:
    start = time.perf_counter()

    # 1. Retrieve
    chunks = retrieve_documents(req.claim, k=req.k)

    # 2. Generate
    result = verify_claim(req.claim, chunks)

    # 3. Timing
    latency_ms = int((time.perf_counter() - start) * 1000)

    # 4. Log
    log_verification(
        claim=req.claim,
        retrieved_chunk_ids=[c.id for c in chunks],
        verdict=result.verdict.verdict,
        supporting_chunks=result.verdict.supporting_chunks,
        reasoning=result.verdict.reasoning,
        latency_ms=latency_ms,
        prompt_tokens=result.input_tokens,
        completion_tokens=result.output_tokens,
    )

    # 5. Return
    return VerifyResponse(
        verdict=result.verdict,
        chunks=chunks,
        latency_ms=latency_ms,
    )
