"""
AWS Textract extractor for loan PDFs.

Strategy:
  - 1-page PDFs  → synchronous DetectDocumentText (fast, cheap)
  - Multi-page   → asynchronous StartDocumentTextDetection (required by AWS for multi-page)

Textract features used:
  - LAYOUT analysis  → preserves reading order across multi-column layouts
  - Handwriting detection via WORD confidence scores
  - Tables via AnalyzeDocument (FeatureTypes=["TABLES", "FORMS"])

Output: ExtractionResult dataclass — raw text, page texts, confidence stats, handwriting flag.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import get_settings
from app.core.logging import get_logger
from app.storage.s3_client import get_s3_client, S3Error
from app.storage.keys import S3Keys

logger = get_logger(__name__)

# Textract job polling
_POLL_INTERVAL_SEC = 5
_MAX_POLL_ATTEMPTS = 120  # 10 minutes max


class ExtractionError(Exception):
    """Raised when Textract extraction fails unrecoverably."""


@dataclass
class PageResult:
    page_number: int
    text: str
    word_count: int
    avg_confidence: float        # 0–100
    low_confidence_words: int    # words below 80% confidence
    has_handwriting: bool        # any HANDWRITING block detected


@dataclass
class ExtractionResult:
    session_id: str
    full_text: str
    pages: list[PageResult] = field(default_factory=list)
    total_pages: int = 0
    overall_avg_confidence: float = 0.0
    has_handwriting: bool = False
    textract_job_id: Optional[str] = None   # None for sync jobs
    raw_response_s3_key: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _build_textract_client():
    settings = get_settings()
    return boto3.client(
        "textract",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )


def _parse_blocks(blocks: list[dict]) -> tuple[dict[int, PageResult], list[dict]]:
    """
    Parse Textract blocks into per-page results.
    Returns (page_map, all_blocks).
    """
    # Group LINE blocks by page, collect WORD confidence
    page_lines: dict[int, list[str]] = {}
    page_word_stats: dict[int, list[float]] = {}
    page_handwriting: dict[int, bool] = {}

    for block in blocks:
        block_type = block.get("BlockType")
        page = block.get("Page", 1)

        if block_type == "LINE":
            page_lines.setdefault(page, []).append(block.get("Text", ""))

        elif block_type == "WORD":
            confidence = block.get("Confidence", 0.0)
            page_word_stats.setdefault(page, []).append(confidence)

            # Textract sets TextType=HANDWRITING for handwritten words
            if block.get("TextType") == "HANDWRITING":
                page_handwriting[page] = True

    page_map: dict[int, PageResult] = {}
    all_pages = set(page_lines.keys()) | set(page_word_stats.keys())

    for page in sorted(all_pages):
        lines = page_lines.get(page, [])
        word_confidences = page_word_stats.get(page, [])
        avg_conf = sum(word_confidences) / len(word_confidences) if word_confidences else 0.0
        low_conf = sum(1 for c in word_confidences if c < 80)

        page_map[page] = PageResult(
            page_number=page,
            text="\n".join(lines),
            word_count=len(word_confidences),
            avg_confidence=round(avg_conf, 2),
            low_confidence_words=low_conf,
            has_handwriting=page_handwriting.get(page, False),
        )

    return page_map, blocks


def _blocks_to_extraction_result(
    session_id: str,
    blocks: list[dict],
    job_id: Optional[str] = None,
) -> ExtractionResult:
    page_map, _ = _parse_blocks(blocks)

    pages = list(page_map.values())
    full_text = "\n\n--- Page Break ---\n\n".join(p.text for p in pages)

    total_words = sum(p.word_count for p in pages)
    overall_conf = (
        sum(p.avg_confidence * p.word_count for p in pages) / total_words
        if total_words > 0 else 0.0
    )
    has_hw = any(p.has_handwriting for p in pages)

    return ExtractionResult(
        session_id=session_id,
        full_text=full_text,
        pages=pages,
        total_pages=len(pages),
        overall_avg_confidence=round(overall_conf, 2),
        has_handwriting=has_hw,
        textract_job_id=job_id,
    )


# ------------------------------------------------------------------
# Sync extraction (single-page or small docs ≤ 1 page)
# ------------------------------------------------------------------

def _extract_sync(
    textract_client,
    s3_bucket: str,
    s3_key: str,
    session_id: str,
) -> ExtractionResult:
    """
    DetectDocumentText — synchronous, only works for single-page PDFs or images.
    """
    logger.info("textract_sync_start", session_id=session_id, key=s3_key)
    try:
        response = textract_client.detect_document_text(
            Document={"S3Object": {"Bucket": s3_bucket, "Name": s3_key}}
        )
    except (BotoCoreError, ClientError) as exc:
        raise ExtractionError(f"Textract sync call failed: {exc}") from exc

    blocks = response.get("Blocks", [])
    logger.info("textract_sync_done", session_id=session_id, block_count=len(blocks))
    return _blocks_to_extraction_result(session_id, blocks, job_id=None), response


# ------------------------------------------------------------------
# Async extraction (multi-page PDFs — required by Textract)
# ------------------------------------------------------------------

def _start_async_job(
    textract_client,
    s3_bucket: str,
    s3_key: str,
    session_id: str,
) -> str:
    """Start a Textract async job. Returns job_id."""
    logger.info("textract_async_start", session_id=session_id, key=s3_key)
    try:
        response = textract_client.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": s3_bucket, "Name": s3_key}},
            JobTag=session_id[:64],  # Textract tag limit
        )
    except (BotoCoreError, ClientError) as exc:
        raise ExtractionError(f"Failed to start Textract async job: {exc}") from exc

    job_id = response["JobId"]
    logger.info("textract_async_job_started", session_id=session_id, job_id=job_id)
    return job_id


def _poll_async_job(
    textract_client,
    job_id: str,
    session_id: str,
) -> list[dict]:
    """
    Poll until job completes. Paginate and collect all blocks.
    Raises ExtractionError on failure or timeout.
    """
    for attempt in range(1, _MAX_POLL_ATTEMPTS + 1):
        try:
            response = textract_client.get_document_text_detection(JobId=job_id)
        except (BotoCoreError, ClientError) as exc:
            raise ExtractionError(f"Textract poll failed (job={job_id}): {exc}") from exc

        status = response.get("JobStatus")
        logger.info(
            "textract_async_poll",
            session_id=session_id,
            job_id=job_id,
            attempt=attempt,
            status=status,
        )

        if status == "SUCCEEDED":
            # Paginate remaining pages
            all_blocks: list[dict] = list(response.get("Blocks", []))
            next_token = response.get("NextToken")
            while next_token:
                page_resp = textract_client.get_document_text_detection(
                    JobId=job_id, NextToken=next_token
                )
                all_blocks.extend(page_resp.get("Blocks", []))
                next_token = page_resp.get("NextToken")
            logger.info(
                "textract_async_complete",
                session_id=session_id,
                job_id=job_id,
                total_blocks=len(all_blocks),
            )
            return all_blocks, response

        elif status == "FAILED":
            status_msg = response.get("StatusMessage", "unknown")
            raise ExtractionError(
                f"Textract job failed (job={job_id}): {status_msg}"
            )

        # PARTIAL or IN_PROGRESS — keep polling
        time.sleep(_POLL_INTERVAL_SEC)

    raise ExtractionError(
        f"Textract job timed out after {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_SEC}s (job={job_id})"
    )


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def extract_document(session_id: str) -> ExtractionResult:
    """
    Main extraction entry point.

    Expects the PDF already uploaded to S3 at S3Keys.upload_pdf(session_id).
    Saves raw Textract JSON response to S3 for audit.
    Returns ExtractionResult.

    Routing:
        - Page count determined via a quick Textract sync probe on first page.
        - If total pages == 1  → sync
        - If total pages >= 2  → async
    """
    settings = get_settings()
    s3 = get_s3_client()
    textract = _build_textract_client()

    pdf_key = S3Keys.upload_pdf(session_id)

    # Verify PDF exists in S3 before calling Textract
    if not s3.exists(pdf_key):
        raise ExtractionError(
            f"PDF not found in S3 at key={pdf_key} for session={session_id}"
        )

    # Determine page count via PyMuPDF (cheap — no Textract call needed)
    page_count = _get_page_count(s3, pdf_key)
    logger.info("pdf_page_count", session_id=session_id, pages=page_count)

    raw_response: dict = {}

    if page_count <= settings.textract_async_threshold_pages - 1:
        # Single page — use sync
        result, raw_response = _extract_sync(textract, settings.s3_bucket, pdf_key, session_id)
    else:
        # Multi-page — use async
        job_id = _start_async_job(textract, settings.s3_bucket, pdf_key, session_id)
        all_blocks, raw_response = _poll_async_job(textract, job_id, session_id)
        result = _blocks_to_extraction_result(session_id, all_blocks, job_id=job_id)

    # Persist raw Textract response to S3 for auditability
    raw_key = S3Keys.textract_response(session_id)
    s3.upload_json(raw_key, json.dumps(raw_response, default=str))
    result.raw_response_s3_key = raw_key

    # Persist extracted text to S3
    s3.upload_text(S3Keys.raw_text(session_id), result.full_text)

    logger.info(
        "extraction_complete",
        session_id=session_id,
        total_pages=result.total_pages,
        has_handwriting=result.has_handwriting,
        overall_confidence=result.overall_avg_confidence,
    )

    return result


def _get_page_count(s3_client, pdf_key: str) -> int:
    """
    Download PDF bytes and count pages using PyMuPDF.
    PyMuPDF is only used here for metadata — Textract does all text work.
    """
    import fitz  # PyMuPDF

    pdf_bytes = s3_client.download_bytes(pdf_key)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = doc.page_count
    doc.close()
    return count
