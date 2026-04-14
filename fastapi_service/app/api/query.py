"""
POST /query — full RAG pipeline with MLflow tracking.

Flow:
    1. Validate request (session_id + question)
    2. Retrieve top-k chunks from FAISS (session-scoped)
    3. Classify intent (MiniLM zero-shot)
    4. Route to the right prompt template
    5. Call LLM via Bytez
    6. Fire background task → log run to MLflow
    7. Return structured response
"""
from __future__ import annotations

import time

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.retrieval.retriever import retrieve, RetrieverError
from app.retrieval.intent_classifier import get_intent_classifier
from app.prompts.router import route
from app.llm.bytez_client import get_llm_client, BytezInferenceError
from app.tracking.mlflow_tracker import get_tracker
from app.core.config import get_settings
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
    variant: str          # prompt variant — logged to MLflow for A/B tracking
    chunks_used: int
    context_words: int
    latency_ms: float


# ------------------------------------------------------------------
# Background MLflow logging
# ------------------------------------------------------------------

def _log_to_mlflow(
    session_id: str,
    question: str,
    answer: str,
    prompt: str,
    intent: str,
    variant: str,
    top_k: int,
    chunks_used: int,
    context_words: int,
    llm_latency_ms: float,
    total_latency_ms: float,
    retrieved_chunks: list[dict],
) -> None:
    """
    Runs in a BackgroundTask thread — never blocks the HTTP response.
    Silently swallows errors so a tracking failure never breaks the API.
    """
    try:
        settings = get_settings()
        tracker = get_tracker()
        tracker.log_query_run(
            session_id=session_id,
            question=question,
            answer=answer,
            prompt=prompt,
            intent=intent,
            variant=variant,
            model_id=settings.bytez_model,
            top_k=top_k,
            chunks_used=chunks_used,
            context_words=context_words,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
            retrieved_chunks=retrieved_chunks,
        )
    except Exception as exc:
        # Tracking must never crash the endpoint
        logger.warning("mlflow_logging_failed", error=str(exc))


# ------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, background_tasks: BackgroundTasks) -> QueryResponse:
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

    # 5. Log to MLflow in background (non-blocking)
    retrieved_chunks_dicts = [
        {"rank": rc.rank, "score": rc.score, "text": rc.chunk.text[:300]}
        for rc in chunks
    ]
    background_tasks.add_task(
        _log_to_mlflow,
        session_id=req.session_id,
        question=req.question,
        answer=llm_resp.text,
        prompt=routed.prompt,
        intent=routed.intent.value,
        variant=routed.variant,
        top_k=req.top_k,
        chunks_used=routed.chunks_used,
        context_words=routed.context_words,
        llm_latency_ms=llm_resp.latency_ms,
        total_latency_ms=total_ms,
        retrieved_chunks=retrieved_chunks_dicts,
    )

    return QueryResponse(
        answer=llm_resp.text,
        intent=routed.intent.value,
        variant=routed.variant,
        chunks_used=routed.chunks_used,
        context_words=routed.context_words,
        latency_ms=total_ms,
    )
