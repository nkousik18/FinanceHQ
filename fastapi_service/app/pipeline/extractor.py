"""
AWS Textract extractor for loan PDFs.

Uses AnalyzeDocument / StartDocumentAnalysis with TABLES + FORMS.
This gives us:
  - LINE/WORD blocks  → raw text per page
  - KEY_VALUE_SET     → form field key-value pairs (loan amount, name, etc.)
  - TABLE/CELL        → structured table rows (financial details, schedules)

Routing:
  - 1-page  → synchronous AnalyzeDocument
  - Multi-page → asynchronous StartDocumentAnalysis (required by AWS)

Output: ExtractionResult — text, tables, form fields, confidence stats.
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
from app.storage.s3_client import get_s3_client
from app.storage.keys import S3Keys

logger = get_logger(__name__)

_POLL_INTERVAL_SEC = 5
_MAX_POLL_ATTEMPTS = 120  # 10 min max
_FEATURE_TYPES = ["TABLES", "FORMS"]


class ExtractionError(Exception):
    """Raised when Textract extraction fails unrecoverably."""


# ------------------------------------------------------------------
# Result dataclasses
# ------------------------------------------------------------------

@dataclass
class TableCell:
    row: int
    col: int
    text: str
    confidence: float


@dataclass
class Table:
    page: int
    rows: int
    cols: int
    cells: list[TableCell] = field(default_factory=list)

    def to_text(self) -> str:
        """Render table as pipe-delimited text for LLM context."""
        grid: dict[tuple[int, int], str] = {
            (c.row, c.col): c.text for c in self.cells
        }
        lines = []
        for r in range(1, self.rows + 1):
            row_vals = [grid.get((r, c), "") for c in range(1, self.cols + 1)]
            lines.append(" | ".join(row_vals))
        return "\n".join(lines)


@dataclass
class FormField:
    key: str
    value: str
    key_confidence: float
    value_confidence: float
    page: int


@dataclass
class PageResult:
    page_number: int
    text: str
    word_count: int
    avg_confidence: float
    low_confidence_words: int
    has_handwriting: bool


@dataclass
class ExtractionResult:
    session_id: str
    full_text: str
    pages: list[PageResult] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    form_fields: list[FormField] = field(default_factory=list)
    total_pages: int = 0
    overall_avg_confidence: float = 0.0
    has_handwriting: bool = False
    textract_job_id: Optional[str] = None
    raw_response_s3_key: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------
# Textract client
# ------------------------------------------------------------------

def _build_textract_client():
    s = get_settings()
    return boto3.client(
        "textract",
        aws_access_key_id=s.aws_access_key_id,
        aws_secret_access_key=s.aws_secret_access_key,
        region_name=s.aws_region,
    )


# ------------------------------------------------------------------
# Block parsers
# ------------------------------------------------------------------

def _parse_text_blocks(blocks: list[dict]) -> dict[int, PageResult]:
    """Parse LINE + WORD blocks into per-page PageResult."""
    page_lines: dict[int, list[str]] = {}
    page_word_conf: dict[int, list[float]] = {}
    page_handwriting: dict[int, bool] = {}

    for b in blocks:
        btype = b.get("BlockType")
        page = b.get("Page", 1)

        if btype == "LINE":
            page_lines.setdefault(page, []).append(b.get("Text", ""))

        elif btype == "WORD":
            conf = b.get("Confidence", 0.0)
            page_word_conf.setdefault(page, []).append(conf)
            if b.get("TextType") == "HANDWRITING":
                page_handwriting[page] = True

    page_map: dict[int, PageResult] = {}
    all_pages = set(page_lines) | set(page_word_conf)

    for page in sorted(all_pages):
        lines = page_lines.get(page, [])
        confs = page_word_conf.get(page, [])
        avg = sum(confs) / len(confs) if confs else 0.0
        low = sum(1 for c in confs if c < 80)

        page_map[page] = PageResult(
            page_number=page,
            text="\n".join(lines),
            word_count=len(confs),
            avg_confidence=round(avg, 2),
            low_confidence_words=low,
            has_handwriting=page_handwriting.get(page, False),
        )

    return page_map


def _parse_tables(blocks: list[dict]) -> list[Table]:
    """Parse TABLE + CELL blocks into Table objects."""
    block_map = {b["Id"]: b for b in blocks}
    tables: list[Table] = []

    for b in blocks:
        if b.get("BlockType") != "TABLE":
            continue

        page = b.get("Page", 1)
        cell_ids = [
            rel_id
            for rel in b.get("Relationships", [])
            if rel["Type"] == "CHILD"
            for rel_id in rel["Ids"]
        ]

        cells: list[TableCell] = []
        max_row = 0
        max_col = 0

        for cell_id in cell_ids:
            cell = block_map.get(cell_id)
            if not cell or cell.get("BlockType") != "CELL":
                continue

            row = cell.get("RowIndex", 0)
            col = cell.get("ColumnIndex", 0)
            max_row = max(max_row, row)
            max_col = max(max_col, col)
            conf = cell.get("Confidence", 0.0)

            # Get text from WORD children of this cell
            word_ids = [
                wid
                for rel in cell.get("Relationships", [])
                if rel["Type"] == "CHILD"
                for wid in rel["Ids"]
            ]
            words = [
                block_map[wid].get("Text", "")
                for wid in word_ids
                if wid in block_map and block_map[wid].get("BlockType") == "WORD"
            ]
            cell_text = " ".join(words)

            cells.append(TableCell(row=row, col=col, text=cell_text, confidence=conf))

        if cells:
            tables.append(Table(page=page, rows=max_row, cols=max_col, cells=cells))

    return tables


def _parse_form_fields(blocks: list[dict]) -> list[FormField]:
    """Parse KEY_VALUE_SET blocks into FormField objects."""
    block_map = {b["Id"]: b for b in blocks}
    fields: list[FormField] = []

    for b in blocks:
        if b.get("BlockType") != "KEY_VALUE_SET":
            continue
        if "KEY" not in b.get("EntityTypes", []):
            continue

        page = b.get("Page", 1)
        key_conf = b.get("Confidence", 0.0)

        # Get key text
        key_word_ids = [
            wid
            for rel in b.get("Relationships", [])
            if rel["Type"] == "CHILD"
            for wid in rel["Ids"]
        ]
        key_text = " ".join(
            block_map[wid].get("Text", "")
            for wid in key_word_ids
            if wid in block_map and block_map[wid].get("BlockType") == "WORD"
        ).strip()

        # Get linked value block
        value_block_ids = [
            wid
            for rel in b.get("Relationships", [])
            if rel["Type"] == "VALUE"
            for wid in rel["Ids"]
        ]

        value_text = ""
        value_conf = 0.0

        for val_id in value_block_ids:
            val_block = block_map.get(val_id)
            if not val_block:
                continue
            value_conf = val_block.get("Confidence", 0.0)
            val_word_ids = [
                wid
                for rel in val_block.get("Relationships", [])
                if rel["Type"] == "CHILD"
                for wid in rel["Ids"]
            ]
            value_text = " ".join(
                block_map[wid].get("Text", "")
                for wid in val_word_ids
                if wid in block_map and block_map[wid].get("BlockType") == "WORD"
            ).strip()

        if key_text:
            fields.append(FormField(
                key=key_text,
                value=value_text,
                key_confidence=round(key_conf, 2),
                value_confidence=round(value_conf, 2),
                page=page,
            ))

    return fields


def _assemble_result(
    session_id: str,
    blocks: list[dict],
    job_id: Optional[str] = None,
) -> ExtractionResult:
    """Build ExtractionResult from all blocks."""
    page_map = _parse_text_blocks(blocks)
    tables = _parse_tables(blocks)
    form_fields = _parse_form_fields(blocks)

    pages = list(page_map.values())

    # Full text = page text + table text injected inline
    table_text_by_page: dict[int, list[str]] = {}
    for t in tables:
        table_text_by_page.setdefault(t.page, []).append(t.to_text())

    page_sections = []
    for p in pages:
        section = p.text
        if p.page_number in table_text_by_page:
            section += "\n\n[TABLE]\n" + "\n\n[TABLE]\n".join(
                table_text_by_page[p.page_number]
            )
        page_sections.append(section)

    full_text = "\n\n--- Page Break ---\n\n".join(page_sections)

    total_words = sum(p.word_count for p in pages)
    overall_conf = (
        sum(p.avg_confidence * p.word_count for p in pages) / total_words
        if total_words > 0 else 0.0
    )

    return ExtractionResult(
        session_id=session_id,
        full_text=full_text,
        pages=pages,
        tables=tables,
        form_fields=form_fields,
        total_pages=len(pages),
        overall_avg_confidence=round(overall_conf, 2),
        has_handwriting=any(p.has_handwriting for p in pages),
        textract_job_id=job_id,
    )


# ------------------------------------------------------------------
# Sync (single-page)
# ------------------------------------------------------------------

def _extract_sync(textract, s3_bucket: str, s3_key: str, session_id: str) -> tuple[ExtractionResult, dict]:
    logger.info("textract_sync_start", session_id=session_id, key=s3_key)
    try:
        response = textract.analyze_document(
            Document={"S3Object": {"Bucket": s3_bucket, "Name": s3_key}},
            FeatureTypes=_FEATURE_TYPES,
        )
    except (BotoCoreError, ClientError) as exc:
        raise ExtractionError(f"Textract sync call failed: {exc}") from exc

    blocks = response.get("Blocks", [])
    logger.info("textract_sync_done", session_id=session_id, block_count=len(blocks))
    return _assemble_result(session_id, blocks, job_id=None), response


# ------------------------------------------------------------------
# Async (multi-page)
# ------------------------------------------------------------------

def _start_async_job(textract, s3_bucket: str, s3_key: str, session_id: str) -> str:
    logger.info("textract_async_start", session_id=session_id, key=s3_key)
    try:
        response = textract.start_document_analysis(
            DocumentLocation={"S3Object": {"Bucket": s3_bucket, "Name": s3_key}},
            FeatureTypes=_FEATURE_TYPES,
            JobTag=session_id[:64],
        )
    except (BotoCoreError, ClientError) as exc:
        raise ExtractionError(f"Failed to start Textract async job: {exc}") from exc

    job_id = response["JobId"]
    logger.info("textract_async_job_started", session_id=session_id, job_id=job_id)
    return job_id


def _poll_async_job(textract, job_id: str, session_id: str) -> tuple[list[dict], dict]:
    for attempt in range(1, _MAX_POLL_ATTEMPTS + 1):
        try:
            response = textract.get_document_analysis(JobId=job_id)
        except (BotoCoreError, ClientError) as exc:
            raise ExtractionError(f"Textract poll failed (job={job_id}): {exc}") from exc

        status = response.get("JobStatus")
        logger.info("textract_async_poll", session_id=session_id, job_id=job_id,
                    attempt=attempt, status=status)

        if status == "SUCCEEDED":
            all_blocks: list[dict] = list(response.get("Blocks", []))
            next_token = response.get("NextToken")
            while next_token:
                page_resp = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
                all_blocks.extend(page_resp.get("Blocks", []))
                next_token = page_resp.get("NextToken")

            logger.info("textract_async_complete", session_id=session_id,
                        job_id=job_id, total_blocks=len(all_blocks))
            return all_blocks, response

        elif status == "FAILED":
            raise ExtractionError(
                f"Textract job failed (job={job_id}): {response.get('StatusMessage')}"
            )

        time.sleep(_POLL_INTERVAL_SEC)

    raise ExtractionError(
        f"Textract timed out after {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_SEC}s (job={job_id})"
    )


# ------------------------------------------------------------------
# Page count helper
# ------------------------------------------------------------------

def _get_page_count(s3_client, pdf_key: str) -> int:
    import fitz
    pdf_bytes = s3_client.download_bytes(pdf_key)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = doc.page_count
    doc.close()
    return count


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def extract_document(session_id: str) -> ExtractionResult:
    """
    Main extraction entry point.

    Expects PDF already at S3Keys.upload_pdf(session_id).
    Saves raw Textract JSON + extracted text to S3.
    Returns ExtractionResult with text, tables, and form fields.
    """
    settings = get_settings()
    s3 = get_s3_client()
    textract = _build_textract_client()

    pdf_key = S3Keys.upload_pdf(session_id)

    if not s3.exists(pdf_key):
        raise ExtractionError(
            f"PDF not found in S3 at key={pdf_key} for session={session_id}"
        )

    page_count = _get_page_count(s3, pdf_key)
    logger.info("pdf_page_count", session_id=session_id, pages=page_count)

    if page_count < settings.textract_async_threshold_pages:
        result, raw_response = _extract_sync(textract, settings.s3_bucket, pdf_key, session_id)
    else:
        job_id = _start_async_job(textract, settings.s3_bucket, pdf_key, session_id)
        all_blocks, raw_response = _poll_async_job(textract, job_id, session_id)
        result = _assemble_result(session_id, all_blocks, job_id=job_id)

    # Persist raw response + extracted text to S3
    raw_key = S3Keys.textract_response(session_id)
    s3.upload_json(raw_key, json.dumps(raw_response, default=str))
    result.raw_response_s3_key = raw_key
    s3.upload_text(S3Keys.raw_text(session_id), result.full_text)

    logger.info(
        "extraction_complete",
        session_id=session_id,
        total_pages=result.total_pages,
        tables=len(result.tables),
        form_fields=len(result.form_fields),
        has_handwriting=result.has_handwriting,
        overall_confidence=result.overall_avg_confidence,
    )

    return result
