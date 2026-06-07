from typing import Literal

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    id: int
    title: str
    text: str
    score: float


class ClaimVerdict(BaseModel):
    verdict: Literal["supported", "refuted", "not_enough_info"]
    supporting_chunks: list[int]
    reasoning: str


class GenerationResult(BaseModel):
    verdict: ClaimVerdict
    input_tokens: int
    output_tokens: int


class VerifyRequest(BaseModel):
    claim: str
    k: int = 5


class VerifyResponse(BaseModel):
    verdict: ClaimVerdict
    chunks: list[RetrievedChunk]
    latency_ms: int
