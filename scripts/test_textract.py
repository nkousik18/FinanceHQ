"""
Explore Textract output on a real loan PDF.
Uses AnalyzeDocument (TABLES + FORMS) for full extraction.

Usage:
    python scripts/test_textract.py path/to/loan.pdf

Saves full raw JSON to scripts/output/{filename}_raw.json
"""
import sys
import os
import json
import time
from pathlib import Path

# Allow imports from fastapi_service
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fastapi_service"))

import boto3
import fitz
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from app.core.logging import setup_logging, get_logger

load_dotenv()
setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"), environment="development")
logger = get_logger("test_textract")

BUCKET = os.environ["S3_BUCKET"]
REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]
FEATURE_TYPES = ["TABLES", "FORMS"]
POLL_INTERVAL = 5
MAX_POLLS = 120
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def make_clients():
    session = boto3.Session(
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=REGION,
    )
    return session.client("s3"), session.client("textract")


def get_page_count(local_path: Path) -> int:
    doc = fitz.open(str(local_path))
    count = doc.page_count
    doc.close()
    return count


def upload_pdf(s3, local_path: Path) -> str:
    key = f"textract-tests/{local_path.name}"
    logger.info("uploading_pdf", file=local_path.name, bucket=BUCKET, key=key)
    s3.upload_file(str(local_path), BUCKET, key)
    logger.info("upload_complete", key=key)
    return key


def run_sync(textract, s3_key: str) -> dict:
    logger.info("textract_sync_start", key=s3_key, features=FEATURE_TYPES)
    response = textract.analyze_document(
        Document={"S3Object": {"Bucket": BUCKET, "Name": s3_key}},
        FeatureTypes=FEATURE_TYPES,
    )
    logger.info("textract_sync_complete", block_count=len(response.get("Blocks", [])))
    return response


def run_async(textract, s3_key: str) -> dict:
    logger.info("textract_async_start", key=s3_key, features=FEATURE_TYPES)
    start = textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": BUCKET, "Name": s3_key}},
        FeatureTypes=FEATURE_TYPES,
    )
    job_id = start["JobId"]
    logger.info("textract_job_started", job_id=job_id)

    for attempt in range(1, MAX_POLLS + 1):
        resp = textract.get_document_analysis(JobId=job_id)
        status = resp["JobStatus"]
        logger.info("textract_poll", job_id=job_id, attempt=attempt, status=status)

        if status == "SUCCEEDED":
            all_blocks = list(resp.get("Blocks", []))
            next_token = resp.get("NextToken")
            page_num = 1
            while next_token:
                page_num += 1
                page_resp = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
                all_blocks.extend(page_resp.get("Blocks", []))
                next_token = page_resp.get("NextToken")
                logger.debug("textract_pagination", page=page_num, total_blocks_so_far=len(all_blocks))

            resp["Blocks"] = all_blocks
            logger.info("textract_async_complete", job_id=job_id, total_blocks=len(all_blocks))
            return resp

        elif status == "FAILED":
            logger.error("textract_job_failed", job_id=job_id, reason=resp.get("StatusMessage"))
            sys.exit(1)

        time.sleep(POLL_INTERVAL)

    logger.error("textract_job_timeout", job_id=job_id, max_polls=MAX_POLLS)
    sys.exit(1)


def analyse_and_print(response: dict, filename: str):
    blocks = response.get("Blocks", [])
    block_map = {b["Id"]: b for b in blocks}

    # Block type summary
    type_counts: dict[str, int] = {}
    for b in blocks:
        t = b.get("BlockType", "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1

    logger.info("block_type_summary", **type_counts)

    # Per-page stats
    page_lines: dict[int, list[str]] = {}
    page_word_conf: dict[int, list[float]] = {}
    page_handwriting: dict[int, list[str]] = {}

    for b in blocks:
        btype = b.get("BlockType")
        page = b.get("Page", 1)
        if btype == "LINE":
            page_lines.setdefault(page, []).append(b.get("Text", ""))
        elif btype == "WORD":
            conf = b.get("Confidence", 0.0)
            page_word_conf.setdefault(page, []).append(conf)
            if b.get("TextType") == "HANDWRITING":
                page_handwriting.setdefault(page, []).append(b.get("Text", ""))

    all_pages = sorted(set(page_lines) | set(page_word_conf))
    logger.info("pages_detected", count=len(all_pages))

    for page in all_pages:
        confs = page_word_conf.get(page, [])
        hw = page_handwriting.get(page, [])
        avg = sum(confs) / len(confs) if confs else 0.0
        low = sum(1 for c in confs if c < 80)

        logger.info(
            "page_stats",
            page=page,
            word_count=len(confs),
            avg_confidence=round(avg, 1),
            low_confidence_words=low,
            handwriting_word_count=len(hw),
            handwriting_sample=hw[:5] if hw else [],
        )

        lines = page_lines.get(page, [])
        logger.debug("page_text_sample", page=page, first_5_lines=lines[:5])

    # Form fields
    kv_pairs = []
    for b in blocks:
        if b.get("BlockType") != "KEY_VALUE_SET" or "KEY" not in b.get("EntityTypes", []):
            continue
        key_wids = [wid for rel in b.get("Relationships", []) if rel["Type"] == "CHILD" for wid in rel["Ids"]]
        key_text = " ".join(
            block_map[w].get("Text", "") for w in key_wids
            if w in block_map and block_map[w].get("BlockType") == "WORD"
        ).strip()
        val_ids = [wid for rel in b.get("Relationships", []) if rel["Type"] == "VALUE" for wid in rel["Ids"]]
        val_text = ""
        for vid in val_ids:
            vb = block_map.get(vid)
            if not vb:
                continue
            vwids = [wid for rel in vb.get("Relationships", []) if rel["Type"] == "CHILD" for wid in rel["Ids"]]
            val_text = " ".join(
                block_map[w].get("Text", "") for w in vwids
                if w in block_map and block_map[w].get("BlockType") == "WORD"
            ).strip()
        if key_text:
            kv_pairs.append((b.get("Page", 0), key_text, val_text))

    logger.info("form_fields_extracted", count=len(kv_pairs))
    for page, k, v in sorted(kv_pairs):
        logger.debug("form_field", page=page, key=k, value=v)

    # Tables
    table_count = 0
    for b in blocks:
        if b.get("BlockType") != "TABLE":
            continue
        table_count += 1
        page = b.get("Page", "?")
        cell_ids = [wid for rel in b.get("Relationships", []) if rel["Type"] == "CHILD" for wid in rel["Ids"]]
        max_row = max_col = 0
        for cid in cell_ids:
            cell = block_map.get(cid)
            if not cell or cell.get("BlockType") != "CELL":
                continue
            max_row = max(max_row, cell.get("RowIndex", 0))
            max_col = max(max_col, cell.get("ColumnIndex", 0))
        logger.info("table_detected", table_num=table_count, page=page, rows=max_row, cols=max_col)

    logger.info("analysis_complete", total_tables=table_count, total_form_fields=len(kv_pairs))

    # Save raw JSON
    out_path = OUTPUT_DIR / f"{Path(filename).stem}_raw.json"
    with open(out_path, "w") as f:
        json.dump(response, f, indent=2, default=str)
    logger.info("raw_json_saved", path=str(out_path))


def main():
    if len(sys.argv) < 2:
        logger.error("missing_argument", usage="python scripts/test_textract.py path/to/document.pdf")
        sys.exit(1)

    local_path = Path(sys.argv[1]).resolve()
    if not local_path.exists():
        logger.error("file_not_found", path=str(local_path))
        sys.exit(1)

    s3, textract = make_clients()
    page_count = get_page_count(local_path)
    logger.info("pdf_loaded", file=local_path.name, pages=page_count)

    s3_key = upload_pdf(s3, local_path)
    response = run_sync(textract, s3_key) if page_count == 1 else run_async(textract, s3_key)
    analyse_and_print(response, local_path.name)


if __name__ == "__main__":
    main()
