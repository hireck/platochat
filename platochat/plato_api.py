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
in ``.env`` (OPENWEBUI_API_KEY; for tracing LANGFUSE_* or OTEL_EXPORTER_OTLP_*,
see tracing.py). Model loading happens once when ``plato_core`` is imported,
so the first start is slow.

Which web pages may call the API (CORS):
    A browser only lets a page read this API's responses if the page's origin
    (scheme + host + port, e.g. ``https://platopub.phys.au.dk``) is on the
    allow-list. Set it with ``PLATO_CORS_ORIGINS``, comma-separated, in the
    environment or in ``.env``:

        PLATO_CORS_ORIGINS=https://platopub.phys.au.dk

    Unset, it allows only the local dev site (``http://localhost:8090`` and
    ``http://127.0.0.1:8090``, where ``make dev`` serves ``local_site/``). Set
    to an empty string it allows no cross-origin page at all, which is right
    when the page and the API are served from the same origin. Origins are
    written without a path or trailing slash. This restrains browsers only --
    it does not stop scripts or curl from calling the API.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import plato_core

app = FastAPI(title="PLATO Chatbot API", version="0.1.0")

# The page is served from a different origin than the API -- during local dev
# local_site/serve.py on :8090 -- so the browser needs the API's permission
# before it hands the page a response. See the module docstring.
DEFAULT_CORS_ORIGINS = "http://localhost:8090,http://127.0.0.1:8090"


def _cors_origins() -> list[str]:
    # Read after `import plato_core`, which loads .env.
    raw = os.environ.get("PLATO_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    # A browser sends the origin without a trailing slash, and the match is
    # exact, so "https://host/" in the setting would silently never match.
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
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
