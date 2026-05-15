from fastapi import FastAPI
from pydantic import BaseModel
from anthropic import Anthropic
from dotenv import load_dotenv
import os


load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    tokens_used: int

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat(chatRequest: ChatRequest) -> ChatResponse:
    response = client.messages.create(
        model= "claude-sonnet-4-6",
        max_tokens= 1000,
        messages= [
            {"role": "user", "content": chatRequest.message}
        ]
    )
    return ChatResponse(reply= response.content[0].text, tokens_used=response.usage.input_tokens + response.usage.output_tokens)