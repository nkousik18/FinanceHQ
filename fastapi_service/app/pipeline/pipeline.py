"""
Pipeline orchestrator — multi-doc sessions.

Session lifecycle:
    POST /sessions               → create_session()   → EMPTY
    POST /sessions/{id}/documents → run_pipeline()     → PROCESSING → READY / FAILED

Per-document lifecycle (within run_pipeline):
    PENDING → EXTRACTING → CLEANING → CHUNKING → INDEXING → READY / FAILED

Session-level status rules:
    EMPTY      — created, no documents uploaded yet
    PROCESSING — at least one document is in-flight
    READY      — at least one document indexed (queryable), none in-flight
    FAILED     — every document failed

S3 artifacts written per session (merged across all docs — what /query reads):
    chunks/{session_id}/chunks.json       ← merged chunk texts + metadata
    chunks/{session_id}/embeddings.npy    ← merged float32 embeddings
    chunks/{session_id}/faiss.index       ← merged FAISS index

S3 artifacts written per document (audit trail):
    uploads/{session_id}/{doc_id}/original.pdf
    extracted/{session_id}/{doc_id}/raw_text.txt
    extracted/{session_id}/{doc_id}/textract_response.json
    chunks/{session_id}/{doc_id}/chunks.json

NOTE: concurrent uploads to the same session are NOT safe — two simultaneous
merge_into_session_index calls will race on the S3 read-modify-write.
Use sequential uploads or add a distributed lock for production.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from app.pipeline.extractor import extract_document, ExtractionError
from app.pipeline.cleaner import clean_extraction
from app.pipeline.validator import validate_extraction
from app.pipeline.chunker import chunk_document
from app.pipeline.indexer import merge_into_session_index, IndexingError
from app.storage.s3_client import get_s3_client, S3Error
from app.storage.keys import S3Keys
from app.core.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Status enums
# ------------------------------------------------------------------

class SessionStatus(str, Enum):
    EMPTY      = "EMPTY"        # session created, no docs yet
    PROCESSING = "PROCESSING"   # at least one doc in-flight
    READY      = "READY"        # at least one doc indexed, none in-flight
    FAILED     = "FAILED"       # all docs failed


class DocStatus(str, Enum):
    PENDING    = "PENDING"
    EXTRACTING = "EXTRACTING"
    CLEANING   = "CLEANING"
    CHUNKING   = "CHUNKING"
    INDEXING   = "INDEXING"
    READY      = "READY"
    FAILED     = "FAILED"


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------

@dataclass
class DocRecord:
    doc_id: str
    filename: str
    status: DocStatus
    uploaded_at: str
    updated_at: str
    pages: int = 0
    chunks: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DocRecord":
        d = dict(d)
        d["status"] = DocStatus(d["status"])
        return cls(**d)


@dataclass
class StatusRecord:
    session_id: str
    status: SessionStatus
    created_at: str
    updated_at: str
    name: str = ""
    total_pages: int = 0
    total_chunks: int = 0
    documents: list[DocRecord] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["documents"] = [doc.to_dict() for doc in self.documents]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StatusRecord":
        d = dict(d)
        d["status"] = SessionStatus(d["status"])
        d["documents"] = [DocRecord.from_dict(doc) for doc in d.get("documents", [])]
        return cls(**d)


# ------------------------------------------------------------------
# Session-level status computation
# ------------------------------------------------------------------

def _compute_session_status(docs: list[DocRecord]) -> SessionStatus:
    """Derive session status from the current state of all its documents."""
    if not docs:
        return SessionStatus.EMPTY

    statuses = {d.status for d in docs}
    in_flight = {DocStatus.PENDING, DocStatus.EXTRACTING, DocStatus.CLEANING,
                 DocStatus.CHUNKING, DocStatus.INDEXING}

    if statuses & in_flight:
        return SessionStatus.PROCESSING
    if DocStatus.READY in statuses:
        return SessionStatus.READY
    return SessionStatus.FAILED   # every doc failed


# ------------------------------------------------------------------
# S3 helpers
# ------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(record: StatusRecord) -> None:
    """Persist status.json to S3 — best-effort, never raises."""
    try:
        s3 = get_s3_client()
        s3.upload_json(
            S3Keys.session_status(record.session_id),
            json.dumps(record.to_dict(), ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("status_write_failed", session_id=record.session_id, error=str(exc))


def _read_status(session_id: str) -> StatusRecord:
    """Load session status from S3. Raises S3Error if session doesn't exist."""
    s3 = get_s3_client()
    raw = s3.download_text(S3Keys.session_status(session_id))
    return StatusRecord.from_dict(json.loads(raw))


def _update_doc(record: StatusRecord, doc: DocRecord) -> None:
    """Replace the matching doc in record.documents and recompute session status."""
    doc.updated_at = _now_iso()
    record.documents = [d if d.doc_id != doc.doc_id else doc for d in record.documents]
    record.status = _compute_session_status(record.documents)
    record.total_pages = sum(d.pages for d in record.documents)
    record.total_chunks = sum(d.chunks for d in record.documents)
    record.updated_at = _now_iso()
    _write_status(record)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def create_session(
    name: str = "",
    session_id: Optional[str] = None,
) -> StatusRecord:
    """
    Create an empty session and persist its status to S3.
    Returns the StatusRecord (status=EMPTY).
    """
    sid = session_id or str(uuid.uuid4())
    now = _now_iso()
    record = StatusRecord(
        session_id=sid,
        status=SessionStatus.EMPTY,
        created_at=now,
        updated_at=now,
        name=name,
    )
    _write_status(record)
    logger.info("session_created", session_id=sid, name=name)
    return record


def run_pipeline(
    pdf_bytes: bytes,
    session_id: str,
    filename: str = "document.pdf",
    doc_id: Optional[str] = None,
) -> StatusRecord:
    """
    Process one PDF and merge it into the session's FAISS index.

    Parameters
    ----------
    pdf_bytes:  Raw PDF content.
    session_id: Existing session (created via create_session).
    filename:   Original filename.
    doc_id:     Reuse an existing doc ID (e.g. retry). Auto-generated if None.

    Returns
    -------
    The updated SessionStatusRecord.
    """
    did = doc_id or str(uuid.uuid4())
    now = _now_iso()

    # ------------------------------------------------------------------
    # Load session — fail fast if it doesn't exist
    # ------------------------------------------------------------------
    try:
        record = _read_status(session_id)
    except S3Error:
        # Session doesn't exist — create it implicitly so the pipeline still runs
        logger.warning("session_not_found_creating_implicit", session_id=session_id)
        record = create_session(session_id=session_id)

    # Register the new document as PENDING
    doc = DocRecord(
        doc_id=did,
        filename=filename,
        status=DocStatus.PENDING,
        uploaded_at=now,
        updated_at=now,
    )
    record.documents.append(doc)
    record.status = _compute_session_status(record.documents)
    record.updated_at = now
    _write_status(record)

    s3 = get_s3_client()

    def _set_doc_status(status: DocStatus) -> None:
        doc.status = status
        _update_doc(record, doc)
        logger.info("doc_stage", session_id=session_id, doc_id=did, status=status.value)

    try:
        # ----------------------------------------------------------
        # Stage 1 — Upload raw PDF to per-doc S3 key
        # ----------------------------------------------------------
        pdf_key = S3Keys.doc_pdf(session_id, did)
        s3.upload_bytes(pdf_key, pdf_bytes, content_type="application/pdf")
        logger.info("doc_pdf_uploaded", session_id=session_id, doc_id=did)

        # ----------------------------------------------------------
        # Stage 2 — Extract (Textract) — use per-doc output keys
        # ----------------------------------------------------------
        _set_doc_status(DocStatus.EXTRACTING)
        extraction = extract_document(
            session_id=session_id,
            pdf_key=pdf_key,
            raw_text_key=S3Keys.doc_raw_text(session_id, did),
            textract_response_key=S3Keys.doc_textract_response(session_id, did),
        )
        doc.pages = extraction.total_pages

        # ----------------------------------------------------------
        # Stage 3 — Clean
        # ----------------------------------------------------------
        _set_doc_status(DocStatus.CLEANING)
        cleaned = clean_extraction(extraction)

        # ----------------------------------------------------------
        # Stage 4 — Validate (hard-block on ERROR severity only)
        # ----------------------------------------------------------
        report = validate_extraction(cleaned)
        if not report.passed:
            error_issues = [i for i in report.issues if i.severity.value == "error"]
            if error_issues:
                raise ValueError(f"Validation failed: {'; '.join(i.message for i in error_issues)}")
            logger.warning("validation_warnings", session_id=session_id, doc_id=did,
                           issues=len(report.issues))

        # ----------------------------------------------------------
        # Stage 5 — Chunk — stamp doc_id into chunk IDs
        # ----------------------------------------------------------
        _set_doc_status(DocStatus.CHUNKING)
        chunks = chunk_document(cleaned)
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{session_id}-{did}-{i}"

        # Save per-doc chunks for audit
        import json as _json
        s3.upload_json(
            S3Keys.doc_chunks(session_id, did),
            _json.dumps([c.to_dict() for c in chunks], ensure_ascii=False),
        )

        # ----------------------------------------------------------
        # Stage 6 — Merge into session-level FAISS index
        # ----------------------------------------------------------
        _set_doc_status(DocStatus.INDEXING)
        total = merge_into_session_index(session_id, chunks)
        doc.chunks = len(chunks)

        # ----------------------------------------------------------
        # Done
        # ----------------------------------------------------------
        _set_doc_status(DocStatus.READY)
        logger.info(
            "doc_pipeline_complete",
            session_id=session_id,
            doc_id=did,
            pages=doc.pages,
            chunks=doc.chunks,
            session_total_chunks=total,
        )

    except Exception as exc:
        doc.status = DocStatus.FAILED
        doc.error = str(exc)
        _update_doc(record, doc)
        logger.error(
            "doc_pipeline_failed",
            session_id=session_id,
            doc_id=did,
            error=str(exc),
            exc_info=True,
        )

    return record
