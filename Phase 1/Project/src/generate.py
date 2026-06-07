import os

from dotenv import load_dotenv
from openai import OpenAI
from models import ClaimVerdict, GenerationResult, RetrievedChunk

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """\
You are a scientific claim evaluator. You assess whether a claim is \
supported or refuted by the provided research abstracts.

Rules:
- Base your verdict ONLY on the provided abstracts. Do not use outside knowledge.
- If the abstracts do not address the claim, verdict must be "not_enough_info".
- Cite the chunk IDs that justify your verdict in supporting_chunks.
- If verdict is "not_enough_info", supporting_chunks must be empty.
- Keep reasoning to 1-3 sentences."""

MODEL = "gpt-4o-mini"


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for chunk in chunks:
        parts.append(
            f"[Chunk {chunk.id}]\n"
            f"Title: {chunk.title}\n"
            f"Text: {chunk.text}\n"
        )
    return "\n".join(parts)


def verify_claim(claim: str, chunks: list[RetrievedChunk]) -> GenerationResult:
    user_message = f"Claim: {claim}\n\nEvidence:\n{_format_chunks(chunks)}"

    response = openai_client.responses.parse(
        model=MODEL,
        temperature=0,
        instructions=SYSTEM_PROMPT,
        input=user_message,
        text_format=ClaimVerdict,
    )

    verdict = response.output_parsed
    if verdict is None:
        raise ValueError("LLM returned no structured output (refusal or parse failure)")

    usage = response.usage
    if usage is None:
        raise ValueError("OpenAI response missing usage data")

    return GenerationResult(
        verdict=verdict,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


if __name__ == "__main__":
    from retrieve import retrieve_documents

    claim = "Active Ly49Q prevents neutrophil polarization"
    chunks = retrieve_documents(claim)
    result = verify_claim(claim, chunks)

    print(f"Claim    : {claim}")
    print(f"Verdict  : {result.verdict.verdict}")
    print(f"Chunks   : {result.verdict.supporting_chunks}")
    print(f"Reasoning: {result.verdict.reasoning}")
    print(f"Tokens   : {result.input_tokens} in / {result.output_tokens} out")
