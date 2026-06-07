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
