"""
FinanceHQ FastAPI service — entry point.

Start:
    uvicorn main:app --reload --port 8001

Endpoints:
    POST /sessions                         — create a named session
    POST /sessions/{id}/documents          — upload a PDF into a session
    GET  /sessions/{id}/status             — poll session + per-doc status
    POST /query                            — RAG query, returns full JSON response
    POST /query/stream                     — same query, streams answer via SSE
    GET  /health                           — liveness check
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.logging import get_logger
from app.api.query import router as query_router
from app.api.sessions import router as sessions_router

logger = get_logger(__name__)

app = FastAPI(
    title="FinanceHQ",
    description="RAG service for loan document Q&A",
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router)
app.include_router(query_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
