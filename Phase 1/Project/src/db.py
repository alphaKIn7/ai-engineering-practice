import json
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Field, Session, SQLModel, create_engine

DB_PATH = Path(__file__).parent.parent / "data" / "verifications.db"

engine = create_engine(f"sqlite:///{DB_PATH}")


class VerificationLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: str
    claim: str
    retrieved_chunk_ids: str  # JSON-serialized list[int]
    verdict: str
    supporting_chunks: str   # JSON-serialized list[int]
    reasoning: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)


def log_verification(
    claim: str,
    retrieved_chunk_ids: list[int],
    verdict: str,
    supporting_chunks: list[int],
    reasoning: str,
    latency_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    row = VerificationLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        claim=claim,
        retrieved_chunk_ids=json.dumps(retrieved_chunk_ids),
        verdict=verdict,
        supporting_chunks=json.dumps(supporting_chunks),
        reasoning=reasoning,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    with Session(engine) as session:
        session.add(row)
        session.commit()


if __name__ == "__main__":
    init_db()
    log_verification(
        claim="Active Ly49Q prevents neutrophil polarization",
        retrieved_chunk_ids=[5531479, 4346436, 1263446],
        verdict="refuted",
        supporting_chunks=[5531479],
        reasoning="Test row — hand-written fake data.",
        latency_ms=1234,
        prompt_tokens=500,
        completion_tokens=80,
    )
    print(f"Logged test row to {DB_PATH}")
