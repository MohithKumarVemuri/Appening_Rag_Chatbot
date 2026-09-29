from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.graph import build_rag_graph

app = FastAPI(title="Agentic AI RAG Chatbot")
graph = build_rag_graph()


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: List[str]
    confidence_score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    try:
        result = graph.invoke({"question": req.query})
    except Exception as e:
        print("CHAT ERROR:", repr(e))
        raise HTTPException(status_code=502, detail=str(e)[:500])
    return ChatResponse(
        answer=result["answer"],
        retrieved_chunks=result["context"],
        confidence_score=result["score"],
    )

# Run: uvicorn app:app --reload