"""
Test chunking + embedding + indexing pipeline locally.
Reads from saved Textract JSON — no Textract API call needed.
DOES upload FAISS index + chunks to S3.

Usage:
    python scripts/test_chunk_index.py scripts/output/loan1_raw.json
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fastapi_service"))

from dotenv import load_dotenv
from app.core.logging import setup_logging, get_logger
from app.pipeline.extractor import _assemble_result
from app.pipeline.cleaner import clean_extraction
from app.pipeline.validator import validate_extraction
from app.pipeline.chunker import chunk_document
from app.pipeline.indexer import build_and_save_index

load_dotenv()
setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"), environment="development")
logger = get_logger("test_chunk_index")


def main():
    if len(sys.argv) < 2:
        logger.error("usage", msg="python scripts/test_chunk_index.py scripts/output/loan1_raw.json")
        sys.exit(1)

    json_path = Path(sys.argv[1]).resolve()
    session_id = json_path.stem.replace("_raw", "")

    logger.info("pipeline_start", session_id=session_id)

    # Step 1: Reconstruct extraction
    with open(json_path) as f:
        data = json.load(f)
    extraction = _assemble_result(session_id=session_id, blocks=data["Blocks"], job_id=None)
    logger.info("extraction_loaded", pages=extraction.total_pages, fields=len(extraction.form_fields))

    # Step 2: Clean
    cleaned = clean_extraction(extraction)

    # Step 3: Validate
    report = validate_extraction(cleaned)
    if not report.passed:
        logger.error("validation_failed", summary=report.summary())
        sys.exit(1)
    logger.info("validation_passed", summary=report.summary())

    # Step 4: Chunk
    chunks = chunk_document(cleaned)

    # Step 5: Index + save to S3
    total = build_and_save_index(session_id, chunks)

    print(f"\n{'='*60}")
    print(f"CHUNK + INDEX SUMMARY")
    print(f"{'='*60}")
    print(f"  Session:      {session_id}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Indexed:      {total} vectors")
    print(f"  Avg words:    {round(sum(c.word_count for c in chunks)/len(chunks), 1)}")
    print(f"\n  Sample chunks:")
    for c in chunks[:3]:
        preview = c.text[:120].replace("\n", " ")
        print(f"  [{c.chunk_id}] p{c.page} | {c.word_count}w | {preview}...")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
