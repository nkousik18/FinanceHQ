"""
FinanceHQ FastAPI service — entry point.

Start:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /query          — full RAG pipeline (retrieve → classify → prompt → LLM)
    GET  /health         — liveness check
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.logging import get_logger
from app.api.query import router as query_router

logger = get_logger(__name__)

app = FastAPI(
    title="FinanceHQ",
    description="RAG service for loan document Q&A",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
