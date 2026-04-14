"""
Session management endpoints.

    POST /sessions                          — create a named session
    POST /sessions/{session_id}/documents  — upload a PDF into an existing session
    GET  /sessions/{session_id}/status     — poll session + per-doc status

Flow:
    1. Client creates a session → gets back session_id
    2. Client uploads N PDFs to that session (each runs the pipeline in background)
    3. Client polls /status until session status == READY
    4. Client calls POST /query with the session_id
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.pipeline.pipeline import (
    create_session,
    run_pipeline,
    _write_status,
    _read_status,
    SessionStatus,
    DocStatus,
    StatusRecord,
    DocRecord,
)
from app.storage.s3_client import get_s3_client, S3Error
from app.storage.keys import S3Keys
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])

MAX_PDF_MB    = 20
MAX_PDF_BYTES = MAX_PDF_MB * 1024 * 1024


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    name: str = Field("", description="Optional human-readable label for the session")


class CreateSessionResponse(BaseModel):
    session_id: str
    name: str
    status: str


class UploadDocumentResponse(BaseModel):
    session_id: str
    doc_id: str
    filename: str
    status: str
    message: str


class DocStatusResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    pages: int
    chunks: int
    uploaded_at: str
    updated_at: str
    error: str | None


class SessionStatusResponse(BaseModel):
    session_id: str
    name: str
    status: str
    total_pages: int
    total_chunks: int
    created_at: str
    updated_at: str
    documents: list[DocStatusResponse]
    error: str | None


# ------------------------------------------------------------------
# Background wrapper
# ------------------------------------------------------------------

def _run_pipeline_bg(pdf_bytes: bytes, session_id: str, filename: str, doc_id: str) -> None:
    """Thin wrapper for BackgroundTasks — errors are handled inside run_pipeline."""
    run_pipeline(pdf_bytes, session_id=session_id, filename=filename, doc_id=doc_id)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post("", response_model=CreateSessionResponse, status_code=201)
def create_new_session(req: CreateSessionRequest) -> CreateSessionResponse:
    """
    Create a new empty session.
    Returns a session_id to use for all subsequent document uploads and queries.
    """
    record = create_session(name=req.name)
    logger.info("session_created_via_api", session_id=record.session_id, name=req.name)
    return CreateSessionResponse(
        session_id=record.session_id,
        name=record.name,
        status=record.status.value,
    )


@router.post("/{session_id}/documents", response_model=UploadDocumentResponse, status_code=202)
async def upload_document(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Loan PDF file"),
) -> UploadDocumentResponse:
    """
    Upload a PDF into an existing session.

    The pipeline runs in the background. Poll GET /sessions/{session_id}/status
    until the document status == READY before querying.

    Multiple documents can be uploaded to the same session — they are all
    merged into a single FAISS index that /query searches across.
    """
    # Validate session exists
    try:
        _read_status(session_id)
    except S3Error:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    # Content-type check
    content_type = file.content_type or ""
    if "pdf" not in content_type.lower() and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum is {MAX_PDF_MB} MB.",
        )

    doc_id = str(uuid.uuid4())
    filename = file.filename or "document.pdf"

    logger.info(
        "document_upload_received",
        session_id=session_id,
        doc_id=doc_id,
        filename=filename,
        size_bytes=len(pdf_bytes),
    )

    # Register the doc as PENDING in the status before the background task fires
    now = datetime.now(timezone.utc).isoformat()
    record = _read_status(session_id)
    doc = DocRecord(
        doc_id=doc_id,
        filename=filename,
        status=DocStatus.PENDING,
        uploaded_at=now,
        updated_at=now,
    )
    record.documents.append(doc)
    from app.pipeline.pipeline import _compute_session_status
    record.status = _compute_session_status(record.documents)
    record.updated_at = now
    _write_status(record)

    background_tasks.add_task(
        _run_pipeline_bg,
        pdf_bytes=pdf_bytes,
        session_id=session_id,
        filename=filename,
        doc_id=doc_id,
    )

    return UploadDocumentResponse(
        session_id=session_id,
        doc_id=doc_id,
        filename=filename,
        status=DocStatus.PENDING.value,
        message="Pipeline started. Poll /sessions/{session_id}/status for progress.",
    )


@router.get("/{session_id}/status", response_model=SessionStatusResponse)
def get_session_status(session_id: str) -> SessionStatusResponse:
    """
    Return the current status of a session and all its documents.

    Session status:
        EMPTY      — no documents uploaded yet
        PROCESSING — at least one document is being processed
        READY      — at least one document is indexed (safe to query)
        FAILED     — all documents failed

    Document status:
        PENDING → EXTRACTING → CLEANING → CHUNKING → INDEXING → READY | FAILED
    """
    try:
        record = _read_status(session_id)
    except S3Error:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Corrupt session record: {exc}")

    return SessionStatusResponse(
        session_id=record.session_id,
        name=record.name,
        status=record.status.value,
        total_pages=record.total_pages,
        total_chunks=record.total_chunks,
        created_at=record.created_at,
        updated_at=record.updated_at,
        error=record.error,
        documents=[
            DocStatusResponse(
                doc_id=d.doc_id,
                filename=d.filename,
                status=d.status.value,
                pages=d.pages,
                chunks=d.chunks,
                uploaded_at=d.uploaded_at,
                updated_at=d.updated_at,
                error=d.error,
            )
            for d in record.documents
        ],
    )
