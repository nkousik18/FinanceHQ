"""
POST /query — full RAG pipeline.

Flow:
    1. Validate request (session_id + question)
    2. Retrieve top-k chunks from FAISS (session-scoped)
    3. Classify intent (MiniLM zero-shot)
    4. Route to the right prompt template
    5. Call LLM via Bytez
    6. Return structured response
"""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.retrieval.retriever import retrieve, RetrieverError
from app.retrieval.intent_classifier import get_intent_classifier
from app.prompts.router import route
from app.llm.bytez_client import get_llm_client, BytezInferenceError
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    session_id: str = Field(..., description="Document session ID")
    question: str   = Field(..., min_length=1, description="User question")
    top_k: int      = Field(5, ge=1, le=20, description="Chunks to retrieve")


class QueryResponse(BaseModel):
    answer: str
    intent: str
    variant: str          # prompt variant — for MLflow tracking later
    chunks_used: int
    context_words: int
    latency_ms: float


# ------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    t0 = time.monotonic()

    logger.info(
        "query_received",
        session_id=req.session_id,
        question_preview=req.question[:80],
        top_k=req.top_k,
    )

    # 1. Retrieve
    try:
        chunks = retrieve(req.session_id, req.question, top_k=req.top_k)
    except RetrieverError as exc:
        logger.warning("retrieval_failed", error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc))

    # 2. Classify intent
    classifier = get_intent_classifier()
    intents = classifier.classify(req.question)

    # 3. Route prompt
    routed = route(req.question, intents, chunks)

    # 4. LLM call
    try:
        llm = get_llm_client()
        llm_resp = llm.complete(routed.prompt)
    except BytezInferenceError as exc:
        logger.error("llm_failed", error=str(exc))
        raise HTTPException(status_code=502, detail=f"LLM inference failed: {exc}")

    total_ms = round((time.monotonic() - t0) * 1000, 1)

    logger.info(
        "query_complete",
        session_id=req.session_id,
        intent=routed.intent.value,
        variant=routed.variant,
        llm_latency_ms=llm_resp.latency_ms,
        total_latency_ms=total_ms,
    )

    return QueryResponse(
        answer=llm_resp.text,
        intent=routed.intent.value,
        variant=routed.variant,
        chunks_used=routed.chunks_used,
        context_words=routed.context_words,
        latency_ms=total_ms,
    )
