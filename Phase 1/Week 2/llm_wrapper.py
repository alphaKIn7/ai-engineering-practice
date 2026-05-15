from datetime import datetime
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
from anthropic import Anthropic
import time
import os
from dotenv import load_dotenv
from sqlmodel import SQLModel, Field, create_engine, Session


load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
local_client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")
DATABASE_URL = "sqlite:///llm_logs.db"
engine = create_engine(DATABASE_URL)

price_table = {
    "claude-sonnet-4-6": {"input" : 0.003, "output" : 0.015},
    "gpt-4o-mini": {"input" : 0.00015, "output" : 0.0006},
    "google/gemma-3-12b": {"input" : 0, "output" : 0}
}

class CompletionResult(BaseModel):
    text: str 
    parsed: BaseModel | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    provider: str
    model: str

class CallLog(SQLModel, table = True):
    id: int | None = Field(default = None, primary_key = True)
    prompt: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    text: str
    timestamp: datetime = Field(default_factory=datetime.now)

def create_db():
    SQLModel.metadata.create_all(engine)


async def complete(
    prompt: str,
    provider: Literal["openai", "anthropic", "local"],
    model: str | None = None,
    system: str | None = None,
    response_schema: type[BaseModel] | None = None
) -> CompletionResult:

    defaults = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-6",
        "local": "google/gemma-3-12b"
    }
    m = model or defaults[provider]
    
    start = time.perf_counter()
    response = None
    text = ""
    parsed = None
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0
    latency_ms = 0
    params = dict()
    match provider:
        case "openai":
            params = {
                "model": m,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            }
            if system is not None:
                params["messages"].insert(0, {"role": "system", "content": system})
            
            if response_schema is not None:
                params["response_format"] = response_schema
                response = openai_client.chat.completions.parse(**params)
                parsed = response.choices[0].message.parsed
            else:
                response = openai_client.chat.completions.create(**params)
                parsed = None

            text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
     
        case "anthropic":
            params = {
                "model": m,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            }
            if system is not None:
                params["system"] = system
            response = anthropic_client.messages.create(**params)
            text = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            parsed = None
            
        case "local":
            params = {
                "model": m,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            }
            if system is not None:
                params["messages"].insert(0, {"role": "system", "content": system})
            
            if response_schema is not None:
                params["response_format"] = response_schema
                response = local_client.chat.completions.parse(**params)
                parsed = response.choices[0].message.parsed
            else:
                response = local_client.chat.completions.create(**params)
                parsed = None

            text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
        
    latency_ms = int((time.perf_counter() - start) * 1000)

    cost_usd = (input_tokens/1000 * price_table[m]["input"]) + (output_tokens/1000 * price_table[m]["output"])

    with Session(engine) as session:
        log = CallLog(
            prompt=prompt,
            provider=provider,
            model=m,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            text=text
        )
        session.add(log)
        session.commit()
    
    return CompletionResult(
        text = text,
        parsed = parsed,
        input_tokens= input_tokens,
        output_tokens= output_tokens,
        cost_usd= cost_usd,
        latency_ms = latency_ms,
        provider = provider,
        model = m
    )

create_db()