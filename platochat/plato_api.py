"""
PLATO chatbot — HTTP API.

A thin FastAPI wrapper around the RAG pipeline in ``plato_core.py``. It exposes
the contract the web front-end (``local_site/js/chatbot.js``) expects:

    POST /api/chat
      request : { "message": str,
                  "history": [ {"role": "human"|"ai", "content": str}, ... ] }
      response: { "reply": "<markdown>", "sources": "<markdown>"|null }

    GET  /health  ->  { "status": "ok" }

Run it:
    cd platochat
    uvicorn plato_api:app --reload --port 8000

Requires Weaviate running locally (same as the Streamlit app) and the env vars
in ``.env`` (OPENWEBUI_API_KEY, LANGFUSE_*). Model loading happens once when
``plato_core`` is imported, so the first start is slow.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import plato_core

app = FastAPI(title="PLATO Chatbot API", version="0.1.0")

# The page is served from a different origin during local dev
# (python -m http.server on :8090), so allow cross-origin calls. Tighten this
# allow-list before any non-local deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str = Field(description='"human" or "ai"')
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    sources: str | None = None


@app.on_event("shutdown")
def _close_weaviate() -> None:
    # Release the Weaviate connection cleanly on server shutdown.
    try:
        plato_core.weaviate_client.close()
    except Exception:
        pass


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # Defined with `def` (not `async def`) so FastAPI runs this blocking,
    # CPU/IO-heavy pipeline in a worker thread instead of the event loop.
    reply, sources = plato_core.answer_question(
        req.message,
        [m.model_dump() for m in req.history],
    )
    if reply is None:
        reply = "Sorry, something went wrong. Please try again."
    return ChatResponse(reply=reply, sources=sources)
