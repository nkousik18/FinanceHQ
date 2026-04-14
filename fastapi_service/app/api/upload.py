"""
POST /upload         — ingest a PDF: upload → extract → chunk → index.
GET  /sessions/{id}/status — poll the session status written by the pipeline.

The pipeline runs in a FastAPI BackgroundTask so the HTTP response returns
immediately with session_id + PENDING status, and the client polls /status.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.pipeline.pipeline import run_pipeline, SessionStatus, StatusRecord, _write_status
from app.storage.s3_client import get_s3_client, S3Error
from app.storage.keys import S3Keys
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

MAX_PDF_SIZE_MB = 20
MAX_PDF_BYTES   = MAX_PDF_SIZE_MB * 1024 * 1024


# ------------------------------------------------------------------
# Response schemas
# ------------------------------------------------------------------

class UploadResponse(BaseModel):
    session_id: str
    status: str
    message: str


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    filename: str | None
    total_pages: int
    total_chunks: int
    created_at: str
    updated_at: str
    error: str | None


# ------------------------------------------------------------------
# Background wrapper
# ------------------------------------------------------------------

def _run_pipeline_bg(pdf_bytes: bytes, filename: str, session_id: str) -> None:
    """
    Thin wrapper so the background task signature matches BackgroundTasks.add_task.
    Errors are caught and written to the status record inside run_pipeline itself.
    """
    run_pipeline(pdf_bytes, filename=filename, session_id=session_id)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Loan PDF file"),
) -> UploadResponse:
    """
    Accept a PDF, kick off the ingestion pipeline in the background,
    and return a session_id immediately.

    The client should poll GET /sessions/{session_id}/status until
    status == READY (or FAILED).
    """
    # Basic content-type guard
    content_type = file.content_type or ""
    if "pdf" not in content_type.lower() and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_PDF_SIZE_MB} MB.",
        )

    # Generate a session_id here so we can return it immediately
    session_id = str(uuid.uuid4())

    logger.info(
        "upload_received",
        session_id=session_id,
        filename=file.filename,
        size_bytes=len(pdf_bytes),
    )

    # Write PENDING status immediately so the polling endpoint works right away
    now = datetime.now(timezone.utc).isoformat()
    pending = StatusRecord(
        session_id=session_id,
        status=SessionStatus.PENDING,
        created_at=now,
        updated_at=now,
        filename=file.filename,
    )
    _write_status(pending)

    # Kick off pipeline in background
    background_tasks.add_task(
        _run_pipeline_bg,
        pdf_bytes=pdf_bytes,
        filename=file.filename or "document.pdf",
        session_id=session_id,
    )

    return UploadResponse(
        session_id=session_id,
        status=SessionStatus.PENDING.value,
        message="Pipeline started. Poll /sessions/{session_id}/status for progress.",
    )


@router.get("/sessions/{session_id}/status", response_model=SessionStatusResponse)
def get_session_status(session_id: str) -> SessionStatusResponse:
    """
    Return the current processing status for a session.

    Status values: PENDING → EXTRACTING → CLEANING → CHUNKING → INDEXING → READY | FAILED
    """
    s3 = get_s3_client()
    key = S3Keys.session_status(session_id)

    try:
        raw = s3.download_text(key)
    except S3Error:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Corrupt status record in S3.")

    return SessionStatusResponse(
        session_id=data.get("session_id", session_id),
        status=data.get("status", "UNKNOWN"),
        filename=data.get("filename"),
        total_pages=data.get("total_pages", 0),
        total_chunks=data.get("total_chunks", 0),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        error=data.get("error"),
    )
