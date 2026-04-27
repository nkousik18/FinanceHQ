"""
POST /query        — full RAG pipeline, returns complete response.
POST /query/stream — same pipeline, streams answer via real Groq SSE.

Flow (both endpoints):
    1. Validate request (session_id + question)
    2. Retrieve top-k chunks from FAISS (session-scoped)
    3. Classify intent (MiniLM zero-shot)
    4. Route to the right prompt template
    5. Call Groq LLM (sync for /query, native streaming for /query/stream)
    6. Fire background task → log run to MLflow
    7. /query: return structured JSON
       /query/stream: stream real tokens as SSE, then send done event with metadata
"""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.retrieval.retriever import retrieve, RetrieverError
from app.retrieval.intent_classifier import get_intent_classifier
from app.prompts.router import route
from app.llm.bytez_client import get_llm_client, stream_tokens, LLMInferenceError
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
    variant: str
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
    try:
        settings = get_settings()
        tracker  = get_tracker()
        tracker.log_query_run(
            session_id=session_id,
            question=question,
            answer=answer,
            prompt=prompt,
            intent=intent,
            variant=variant,
            model_id=settings.groq_model,
            top_k=top_k,
            chunks_used=chunks_used,
            context_words=context_words,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
            retrieved_chunks=retrieved_chunks,
        )
    except Exception as exc:
        logger.warning("mlflow_logging_failed", error=str(exc))


# ------------------------------------------------------------------
# POST /query
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

    try:
        chunks = retrieve(req.session_id, req.question, top_k=req.top_k)
    except RetrieverError as exc:
        logger.warning("retrieval_failed", error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc))

    classifier = get_intent_classifier()
    intents    = classifier.classify(req.question)
    routed     = route(req.question, intents, chunks)

    try:
        llm      = get_llm_client()
        llm_resp = llm.complete(routed.prompt)
    except LLMInferenceError as exc:
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


# ------------------------------------------------------------------
# POST /query/stream — real Groq token streaming via SSE
# ------------------------------------------------------------------

@router.post("/query/stream")
async def query_stream(
    req: QueryRequest, background_tasks: BackgroundTasks
) -> StreamingResponse:
    """
    Same RAG pipeline as POST /query.
    Streams real tokens from Groq's streaming API as SSE events.

    SSE format:
        data: {"token": "Hello"}         ← one per token while streaming
        data: {"done": true, ...meta}    ← final event with intent/latency/etc.
    """
    t0 = time.monotonic()

    logger.info(
        "query_stream_received",
        session_id=req.session_id,
        question_preview=req.question[:80],
        top_k=req.top_k,
    )

    # Steps 1–3 run before the stream opens so errors return normal HTTP codes
    try:
        chunks = retrieve(req.session_id, req.question, top_k=req.top_k)
    except RetrieverError as exc:
        logger.warning("stream_retrieval_failed", error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc))

    classifier = get_intent_classifier()
    intents    = classifier.classify(req.question)
    routed     = route(req.question, intents, chunks)

    retrieved_chunks_dicts = [
        {"rank": rc.rank, "score": rc.score, "text": rc.chunk.text[:300]}
        for rc in chunks
    ]

    async def _event_stream():
        full_answer  = []
        llm_start    = time.monotonic()

        try:
            async for token in stream_tokens(routed.prompt):
                full_answer.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except LLMInferenceError as exc:
            logger.error("stream_llm_failed", error=str(exc))
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return

        llm_latency  = round((time.monotonic() - llm_start) * 1000, 1)
        total_ms     = round((time.monotonic() - t0) * 1000, 1)
        answer_text  = "".join(full_answer)

        logger.info(
            "query_stream_complete",
            session_id=req.session_id,
            intent=routed.intent.value,
            variant=routed.variant,
            total_latency_ms=total_ms,
        )

        # Log to MLflow after stream finishes (non-blocking via executor)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            None,
            _log_to_mlflow,
            req.session_id, req.question, answer_text, routed.prompt,
            routed.intent.value, routed.variant, req.top_k,
            routed.chunks_used, routed.context_words,
            llm_latency, total_ms, retrieved_chunks_dicts,
        )

        done_event = {
            "done": True,
            "intent": routed.intent.value,
            "variant": routed.variant,
            "chunks_used": routed.chunks_used,
            "context_words": routed.context_words,
            "latency_ms": total_ms,
        }
        yield f"data: {json.dumps(done_event)}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")
